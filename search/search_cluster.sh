#!/bin/bash
# SLURM entry point for every W&B sweep in this repo — the cluster twin of
# ./search.sh.
#
# The design rests on ONE fact about W&B sweeps: the controller lives on W&B's
# servers, and a worker is just `wandb agent --count 1 <sweep_id>` pulling the
# next trial from that shared queue. Workers never coordinate with each other,
# so a "cluster search" is simply a bag of those workers spread across SLURM
# jobs. Nothing about the search space, the metric, or the infeasibility handling
# changes — this script reuses search/build_sweep.py and search/sweep_runner.py
# verbatim (see search/configs/README.md for the space itself). It only adds the
# scheduling layer:
#
#   1. discover partitions / free GPUs / queue pressure and pick where to run;
#   2. create the sweep ONCE on the login node (compute nodes then join it live);
#   3. smoke-test one trial to measure real GPU memory → agents-per-GPU;
#   4. generate a self-contained, self-resubmitting sbatch worker script;
#   5. submit N independent copies of it.
#
#   total workers = num_jobs × (gpus_per_job × agents_per_gpu)
#
# Time-limit behaviour
# --------------------
# Each job asks SLURM for `--signal=B:USR1@180`, so ~3 min before the wall-time
# kill it catches USR1 and resubmits ITSELF (one job → one replacement), then
# exits cleanly. The sweep state is server-side, so the only thing ever lost is
# the handful of trials in flight at that moment — which the controller simply
# reissues. The pool therefore renews indefinitely until you stop it:
#
#   ./search_cluster.sh --stop <log-dir-or-jobname>
#
# writes a STOP sentinel (so no in-flight trap resubmits) and `scancel --name`s
# every job sharing the name, including freshly resubmitted ones.
#
# Usage:
#   ./search_cluster.sh                                   # fully interactive
#   ./search_cluster.sh --algorithm c2rl-sac-cvstem --env car \
#       --partition gpu --num-jobs 4 --gpus-per-job 1 --time 12:00:00 -y
#   ./search_cluster.sh --stop c2rl-sac-cvstem_classic-car-v0_20260722_120000
#
# Flags (all optional; anything omitted is prompted for):
#   --algorithm NAME     a stem in search/configs/ (e.g. ppo, c2rl-sac-cvstem)
#   --env NAME           car|cartpole|segway|turtlebot|quadrotor, an Isaac Lab
#                        task id, or 'all' for every classic env
#   --method M           bayes (default) | grid | random
#   --partition NAME     SLURM partition (default: recommended by discovery)
#   --num-jobs N         independent sbatch jobs to submit (default: prompted)
#   --gpus-per-job N     GPUs each job requests via --gres=gpu:N (default 1)
#   --agents-per-gpu N   override the smoke-test packing result
#   --no-probe           skip the smoke test; requires --agents-per-gpu
#   --time HH:MM:SS      per-job wall-time (default 24:00:00; self-resubmit makes
#                        short limits fine, they just renew more often)
#   --account NAME       --account for sbatch (omitted if unset)
#   --qos NAME           --qos for sbatch (omitted if unset)
#   --cpus-per-gpu N     --cpus-per-task = N × gpus_per_job (default 8)
#   --mem SPEC           --mem per job (e.g. 32G; omitted if unset → node default)
#   --activate CMD       shell line to activate the env inside the job
#                        (e.g. 'conda activate crl'); default: none
#   --timeout T          per-run watchdog inside a worker (default 24h)
#   --runs-per-agent N   stop each worker after N runs (default 0 = unbounded)
#   --stop TAG           STOP + scancel a running search (log-dir name or jobname)
#   -y, --yes            skip confirmation prompts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Pretty output (identical scheme to search.sh) ─────────────────────────── #
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_BOLD="$(tput bold)"; C_DIM="$(tput dim)"; C_RESET="$(tput sgr0)"
    C_CYAN="$(tput setaf 6)"; C_GREEN="$(tput setaf 2)"; C_YELLOW="$(tput setaf 3)"
    C_RED="$(tput setaf 1)"; C_BLUE="$(tput setaf 4)"; C_MAGENTA="$(tput setaf 5)"
else
    C_BOLD=""; C_DIM=""; C_RESET=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_MAGENTA=""
fi
_rule() { printf '%s\n' "${C_DIM}────────────────────────────────────────────────────────────${C_RESET}" >&2; }
_header() { echo "" >&2; _rule; printf '%s\n' "${C_BOLD}${C_CYAN}  $1${C_RESET}" >&2; _rule; }
_info()    { printf '%s\n' "  ${C_BLUE}ℹ${C_RESET}  $1" >&2; }
_success() { printf '%s\n' "  ${C_GREEN}✓${C_RESET}  $1" >&2; }
_warn()    { printf '%s\n' "  ${C_YELLOW}⚠${C_RESET}  $1" >&2; }
_error()   { printf '%s\n' "  ${C_RED}✗${C_RESET}  $1" >&2; }
_kv()      { printf '  %s%-18s%s %s\n' "${C_DIM}" "$1" "${C_RESET}" "${C_BOLD}$2${C_RESET}" >&2; }

prompt_choice() {
    local prompt_text="$1"; shift
    local opts=("$@") opt
    printf '\n%s\n' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}$prompt_text${C_RESET}" >&2
    PS3="  ${C_DIM}› ${C_RESET}"
    select opt in "${opts[@]}"; do
        if [[ -n "$opt" ]]; then echo "$opt"; return; fi
        _warn "Invalid choice, try again."
    done
}

# ── Defaults / flag parsing ───────────────────────────────────────────────── #
ALGORITHM=""; ENV_ARG=""; METHOD="bayes"
PARTITION=""; NUM_JOBS=""; GPUS_PER_JOB=1; AGENTS_PER_GPU=""
WALLTIME="24:00:00"; ACCOUNT=""; QOS=""; CPUS_PER_GPU=8; MEM=""
ACTIVATE=""; PER_RUN_TIMEOUT=24h; RUNS_PER_AGENT=0
PROBE=1; YES=0; STOP_TAG=""; SAFETY="0.85"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --num-jobs) NUM_JOBS="$2"; shift 2 ;;
        --gpus-per-job) GPUS_PER_JOB="$2"; shift 2 ;;
        --agents-per-gpu) AGENTS_PER_GPU="$2"; shift 2 ;;
        --no-probe) PROBE=0; shift ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --cpus-per-gpu) CPUS_PER_GPU="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --activate) ACTIVATE="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        --runs-per-agent) RUNS_PER_AGENT="$2"; shift 2 ;;
        --stop) STOP_TAG="$2"; shift 2 ;;
        -y|--yes) YES=1; shift ;;
        -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { _error "Required command '$1' not found — are you on a SLURM login node?"; exit 1; }; }

# ── --stop: halt a running search ─────────────────────────────────────────── #
# Order matters: drop the STOP sentinel FIRST so any worker whose USR1 trap fires
# mid-scancel sees it and declines to resubmit, THEN scancel every job sharing
# the name (running + queued + already-resubmitted).
if [[ -n "$STOP_TAG" ]]; then
    need scancel
    LOG_DIR="$SCRIPT_DIR/logs/$STOP_TAG"
    JOBNAME="$STOP_TAG"
    if [[ -f "$LOG_DIR/jobname" ]]; then JOBNAME="$(cat "$LOG_DIR/jobname")"; fi
    if [[ -d "$LOG_DIR" ]]; then
        touch "$LOG_DIR/STOP"
        _success "Wrote STOP sentinel: $LOG_DIR/STOP"
    else
        _warn "No log dir $LOG_DIR — will still scancel by name '$JOBNAME'."
    fi
    _info "scancel --name=$JOBNAME"
    scancel --name="$JOBNAME" || true
    _success "Requested cancellation of all jobs named '$JOBNAME'."
    exit 0
fi

need sbatch; need sinfo; need squeue; need scontrol; need srun

# ── Algorithm (globbed from configs/, same as search.sh) ──────────────────── #
mapfile -t ALGOS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
[[ "${#ALGOS[@]}" -eq 0 ]] && { _error "No algorithm configs in $CONFIG_DIR."; exit 1; }
_header "contractionRL cluster sweep launcher"
[[ -z "$ALGORITHM" ]] && ALGORITHM=$(prompt_choice "Select algorithm to sweep:" "${ALGOS[@]}")
[[ ! -f "$CONFIG_DIR/$ALGORITHM.yaml" ]] && { _error "No config '$ALGORITHM.yaml'. Available: ${ALGOS[*]}"; exit 1; }
_success "Algorithm: ${C_BOLD}$ALGORITHM${C_RESET}"

# ── Env ───────────────────────────────────────────────────────────────────── #
CLASSIC_ENVS=("classic-car-v0" "classic-cartpole-v0" "classic-segway-v0" "classic-turtlebot-v0" "classic-quadrotor-v0")
ISAAC_ENVS=("Humanoid-PathTracking-v0" "Humanoid-VelTracking-v0" "Manipulator-PathTracking-v0" "Manipulator-VelTracking-v0" "Quadruped-PathTracking-v0" "Quadruped-VelTracking-v0")
if [[ -z "$ENV_ARG" ]]; then
    ENV_ARG=$(prompt_choice "Select env to sweep (or 'all' for every classic env):" \
        car cartpole segway turtlebot quadrotor "${ISAAC_ENVS[@]}" all)
fi
if [[ "$ENV_ARG" == "all" ]]; then
    ENVS=("${CLASSIC_ENVS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0)             ENVS=("classic-car-v0") ;;
        cartpole|classic-cartpole-v0)   ENVS=("classic-cartpole-v0") ;;
        segway|classic-segway-v0)       ENVS=("classic-segway-v0") ;;
        turtlebot|classic-turtlebot-v0) ENVS=("classic-turtlebot-v0") ;;
        quadrotor|classic-quadrotor-v0) ENVS=("classic-quadrotor-v0") ;;
        *)
            _found=0; for e in "${ISAAC_ENVS[@]}"; do [[ "$e" == "$ENV_ARG" ]] && _found=1; done
            [[ "$_found" -ne 1 ]] && { _error "Unknown env '$ENV_ARG'."; exit 1; }
            ENVS=("$ENV_ARG") ;;
    esac
fi
_success "Env(s): ${C_BOLD}${ENVS[*]}${C_RESET}"

# ── Stage 1: partition & GPU discovery ────────────────────────────────────── #
# Best-effort: SLURM output formats vary between sites, so every parse degrades
# gracefully and the raw `sinfo` is always printed for a human sanity check.
#
# free GPUs: walk `scontrol show nodes -o` (one node per line), and for nodes in
# an allocatable state (IDLE/MIXED) subtract GresUsed=gpu:… from Gres=gpu:…,
# crediting the difference to each of the node's Partitions.
declare -A PART_FREE PART_TOTAL PART_PENDING

_gres_gpu_count() {  # extract the trailing integer of a gpu:[type:]N token
    grep -oE 'gpu:[^,]*' <<<"$1" | grep -oE '[0-9]+' | tail -n1
}

_header "Partition / GPU discovery"
{
    while IFS= read -r line; do
        state=$(grep -oE 'State=[^ ]+' <<<"$line" | cut -d= -f2 | tr 'A-Z' 'a-z')
        [[ "$state" =~ idle|mix ]] || continue
        gres=$(grep -oE 'Gres=[^ ]+' <<<"$line" | cut -d= -f2-)
        gused=$(grep -oE 'GresUsed=[^ ]+' <<<"$line" | cut -d= -f2-)
        parts=$(grep -oE 'Partitions=[^ ]+' <<<"$line" | cut -d= -f2-)
        [[ "$gres" == *gpu:* ]] || continue
        total=$(_gres_gpu_count "$gres"); used=$(_gres_gpu_count "$gused")
        [[ -z "$total" ]] && continue; [[ -z "$used" ]] && used=0
        free=$(( total - used )); (( free < 0 )) && free=0
        IFS=',' read -ra plist <<<"$parts"
        for p in "${plist[@]}"; do
            PART_TOTAL[$p]=$(( ${PART_TOTAL[$p]:-0} + total ))
            PART_FREE[$p]=$(( ${PART_FREE[$p]:-0} + free ))
        done
    done < <(scontrol show nodes -o 2>/dev/null)
} || _warn "Could not parse 'scontrol show nodes' — showing raw sinfo only."

# pending GPU jobs per partition (queue pressure proxy)
while IFS='|' read -r p _; do
    [[ -z "$p" ]] && continue
    PART_PENDING[$p]=$(( ${PART_PENDING[$p]:-0} + 1 ))
done < <(squeue -h -t PD -o "%P" 2>/dev/null | sed 's/*//')

printf '\n  %-18s %8s %8s %10s\n' "PARTITION" "FREE-GPU" "TOT-GPU" "PENDING" >&2
_rule
RECOMMENDED=""; BEST_FREE=-1
for p in "${!PART_TOTAL[@]}"; do
    f=${PART_FREE[$p]:-0}; t=${PART_TOTAL[$p]:-0}; q=${PART_PENDING[$p]:-0}
    printf '  %-18s %8s %8s %10s\n' "$p" "$f" "$t" "$q" >&2
    # recommend the partition with the most free GPUs, tie-broken by least queue.
    if (( f > BEST_FREE )) || { (( f == BEST_FREE )) && [[ -n "$RECOMMENDED" ]] && (( q < ${PART_PENDING[$RECOMMENDED]:-0} )); }; then
        BEST_FREE=$f; RECOMMENDED=$p
    fi
done
_rule
_info "Raw sinfo (partition / avail / timelimit / nodes / state / gres):"
sinfo -o "  %P %a %l %D %t %G" 2>/dev/null | sed 's/^/    /' >&2 || true
[[ -n "$RECOMMENDED" ]] && _success "Most free GPUs right now: ${C_BOLD}$RECOMMENDED${C_RESET} (${BEST_FREE} free)"
_info "Est. start for pending work: ${C_DIM}squeue --start${C_RESET} (run it to see SLURM's own estimates)."

if [[ -z "$PARTITION" ]]; then
    if [[ -n "$RECOMMENDED" ]]; then
        printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Partition${C_RESET} ${C_DIM}[$RECOMMENDED]${C_RESET}: " >&2
        read -r PARTITION; PARTITION="${PARTITION:-$RECOMMENDED}"
    else
        printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Partition${C_RESET}: " >&2
        read -r PARTITION
    fi
fi
[[ -z "$PARTITION" ]] && { _error "No partition chosen."; exit 1; }
_success "Partition: ${C_BOLD}$PARTITION${C_RESET}"

# ── num-jobs / gpus-per-job ───────────────────────────────────────────────── #
if [[ -z "$NUM_JOBS" ]]; then
    _default_jobs=$(( BEST_FREE > 0 ? BEST_FREE / GPUS_PER_JOB : 1 )); (( _default_jobs < 1 )) && _default_jobs=1
    printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Independent sbatch jobs to submit${C_RESET} ${C_DIM}[$_default_jobs]${C_RESET}: " >&2
    read -r NUM_JOBS; NUM_JOBS="${NUM_JOBS:-$_default_jobs}"
fi
_success "Jobs: ${C_BOLD}$NUM_JOBS${C_RESET}  (×${GPUS_PER_JOB} GPU each)"

# ── Stage 2: create the sweep(s) on the login node ────────────────────────── #
# ONE sweep per env. Compute nodes join it live over the internet, so this is the
# only step that needs login-node connectivity. Parsing mirrors search.sh.
cd "$REPO_DIR"

# Every sbatch worker must authenticate to W&B as THIS account, or it silently
# runs detached from the sweep — the classic "jobs run but the sweep stays empty"
# failure. Verify credentials here, once: `wandb login` persists them to a
# shared-home ~/.netrc that the compute nodes read, so a login-node check is a
# check for every job. (If WANDB_API_KEY is exported instead, sbatch's default
# --export=ALL carries it; only a site with a NON-shared home needs that.)
if ! wandb login --verify >/dev/null 2>&1; then
    _error "wandb is not authenticated here — run 'wandb login' first."
    _info  "Every sbatch worker joins the sweep with these same credentials."
    exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    _info "Auth via shared-home ~/.netrc. If compute nodes don't share \$HOME, export WANDB_API_KEY."
fi
_success "wandb authenticated — all jobs will join sweeps under this account."

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
declare -A ENV_SWEEP ENV_LOGDIR ENV_JOBNAME

for ENV in "${ENVS[@]}"; do
    LOG_DIR="$SCRIPT_DIR/logs/${ALGORITHM}_${ENV}_${RUN_TS}"
    mkdir -p "$LOG_DIR"
    SWEEP_YAML="$LOG_DIR/sweep.yaml"
    if ! python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV" \
            --method "$METHOD" --out "$SWEEP_YAML" > /dev/null; then
        _error "Failed to build sweep yaml for $ENV — skipping."; continue
    fi
    SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1) || true
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}' || true)
    if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
        _error "Failed to create sweep for $ENV."
        echo "$SWEEP_INIT_OUTPUT" | sed 's/^/      /' >&2; continue
    fi
    # jobname must be short-ish and unique per (env, launch) so --stop hits only
    # this search. Kept in a file so --stop can recover it from the log-dir name.
    JOBNAME="crl-${ALGORITHM}-${ENV}-${RUN_TS}"; JOBNAME="${JOBNAME:0:60}"
    echo "$JOBNAME" > "$LOG_DIR/jobname"
    echo "$SWEEP_ID" > "$LOG_DIR/sweep_id"
    ENV_SWEEP[$ENV]="$SWEEP_ID"; ENV_LOGDIR[$ENV]="$LOG_DIR"; ENV_JOBNAME[$ENV]="$JOBNAME"
    _success "$ENV → sweep ${C_BOLD}$SWEEP_ID${C_RESET}"
done
[[ "${#ENV_SWEEP[@]}" -eq 0 ]] && { _error "No sweeps created — nothing to submit."; exit 1; }

# ── Stage 3: smoke test → agents-per-GPU ──────────────────────────────────── #
# Measure the REAL per-run footprint instead of guessing it. Run ONE trial under
# srun on the chosen partition with a single dedicated GPU, so whole-GPU
# `memory.used` IS this run's usage, sample its peak for a bit, then:
#
#   agents_per_gpu = floor( total_MB × SAFETY / peak_MB )
#
# SAFETY leaves headroom for PyTorch's caching allocator + fragmentation, and for
# SAC's replay buffer still growing past the sampling window. Probed once on the
# first env as representative; override with --agents-per-gpu, or skip with
# --no-probe (which then REQUIRES the override).
if [[ -z "$AGENTS_PER_GPU" ]]; then
    if [[ "$PROBE" -ne 1 ]]; then
        _error "--no-probe requires --agents-per-gpu N."; exit 1
    fi
    PROBE_ENV="${ENVS[0]}"
    _header "Smoke test  ${C_DIM}(${ALGORITHM} on ${PROBE_ENV}, 1 GPU, ~2 min)${C_RESET}"
    NUM_ENVS=$(python - "$ALGORITHM" <<'PY'
import sys, yaml, pathlib
algo = sys.argv[1]
cfg = yaml.safe_load((pathlib.Path("search/configs")/f"{algo}.yaml").read_text())
print(int(cfg["num_envs"]))
PY
)
    CLASSIC_FLAG=""; [[ "$PROBE_ENV" == classic-* ]] && CLASSIC_FLAG="--classic"
    PROBE_LOG="${ENV_LOGDIR[${ENVS[0]}]}/probe.log"
    # The probe body runs on the compute node: launch one training run in the
    # background, poll nvidia-smi for ~120s recording the max used/total, kill the
    # tree, and print a single MEM= line the launcher greps for.
    PROBE_OUT=$(srun --partition="$PARTITION" --gres=gpu:1 \
        --cpus-per-task="$CPUS_PER_GPU" --time=00:10:00 --job-name="crl-probe" \
        ${ACCOUNT:+--account="$ACCOUNT"} ${QOS:+--qos="$QOS"} \
        bash -c '
            set -uo pipefail
            cd "'"$REPO_DIR"'"
            '"${ACTIVATE:+$ACTIVATE;}"'
            python scripts/skrl/train.py '"$CLASSIC_FLAG"' \
                --task "'"$PROBE_ENV"'" --algorithm "'"$ALGORITHM"'" \
                --num_envs "'"$NUM_ENVS"'" --skip_final_eval \
                > "'"$PROBE_LOG"'" 2>&1 &
            child=$!
            peak=0; total=0
            for _ in $(seq 1 40); do
                kill -0 "$child" 2>/dev/null || break
                read u t < <(nvidia-smi --query-gpu=memory.used,memory.total \
                              --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ",")
                [[ -n "${u:-}" ]] && (( u > peak )) && peak=$u
                [[ -n "${t:-}" ]] && total=$t
                sleep 3
            done
            kill -TERM -"$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
            sleep 2; kill -KILL "$child" 2>/dev/null || true
            echo "MEM=${peak}/${total}"
        ' 2>>"$PROBE_LOG") || true
    MEM_LINE=$(grep -oE 'MEM=[0-9]+/[0-9]+' <<<"$PROBE_OUT" | tail -n1)
    if [[ -z "$MEM_LINE" ]]; then
        _error "Smoke test produced no MEM= reading. See $PROBE_LOG"
        _info  "Re-run with --agents-per-gpu N to bypass the probe."
        exit 1
    fi
    PEAK_MB=${MEM_LINE#MEM=}; PEAK_MB=${PEAK_MB%/*}
    TOTAL_MB=${MEM_LINE#*/}
    if [[ -z "$PEAK_MB" || "$PEAK_MB" -le 0 ]]; then
        _error "Smoke test measured 0 MB — the run may have crashed. See $PROBE_LOG"; exit 1
    fi
    AGENTS_PER_GPU=$(python - "$PEAK_MB" "$TOTAL_MB" "$SAFETY" <<'PY'
import sys, math
peak, total, safety = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(max(1, math.floor(total * safety / peak)))
PY
)
    _success "Peak ${C_BOLD}${PEAK_MB} MB${C_RESET} / ${TOTAL_MB} MB total → ${C_BOLD}${AGENTS_PER_GPU}${C_RESET} agent(s)/GPU (safety ${SAFETY})"
fi
[[ "$AGENTS_PER_GPU" -ge 1 ]] 2>/dev/null || { _error "agents-per-gpu must be ≥1 (got '$AGENTS_PER_GPU')."; exit 1; }

AGENTS_PER_JOB=$(( GPUS_PER_JOB * AGENTS_PER_GPU ))
TOTAL_WORKERS=$(( NUM_JOBS * AGENTS_PER_JOB ))

# ── Summary + confirm ─────────────────────────────────────────────────────── #
_header "Cluster sweep summary"
_kv "Algorithm"      "$ALGORITHM"
_kv "Env(s)"         "${ENVS[*]}"
_kv "Partition"      "$PARTITION"
_kv "Jobs / env"     "$NUM_JOBS"
_kv "GPUs / job"     "$GPUS_PER_JOB"
_kv "Agents / GPU"   "$AGENTS_PER_GPU"
_kv "Agents / job"   "$AGENTS_PER_JOB"
_kv "Workers / env"  "$TOTAL_WORKERS"
_kv "Wall-time"      "$WALLTIME  (self-resubmits on USR1@180)"
_rule
if [[ "$YES" -ne 1 ]]; then
    printf '%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Submit these jobs now?${C_RESET} ${C_DIM}[y/N]${C_RESET} " >&2
    read -r CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted — nothing submitted (sweeps already exist on W&B)."; exit 0; }
fi

# ── Stage 4+5: generate a self-resubmitting worker script per env, submit N ── #
for ENV in "${!ENV_SWEEP[@]}"; do
    SWEEP_ID="${ENV_SWEEP[$ENV]}"; LOG_DIR="${ENV_LOGDIR[$ENV]}"; JOBNAME="${ENV_JOBNAME[$ENV]}"
    JOB_SCRIPT="$LOG_DIR/job.sbatch"
    CPUS_PER_TASK=$(( CPUS_PER_GPU * GPUS_PER_JOB ))

    # The sweep id is fully qualified (entity/project/sweepid). Pull entity+project
    # out so the job can export them: `wandb agent entity/project/sweepid` already
    # forces sweep membership, but pinning WANDB_ENTITY/WANDB_PROJECT too keeps
    # sweep_runner.py's bad-trial resume in the SAME project instead of a default
    # wandb auto-derives from the cwd. Empty (and thus skipped) if a non-qualified
    # id ever comes back.
    WB_ENTITY=""; WB_PROJECT=""
    if [[ "$SWEEP_ID" == */*/* ]]; then
        WB_ENTITY="${SWEEP_ID%%/*}"; _rest="${SWEEP_ID#*/}"; WB_PROJECT="${_rest%%/*}"
    fi

    # The generated script is fully self-contained: everything it needs is baked
    # in, so a self-resubmit (`sbatch "$0"`) needs no orchestrator and never
    # re-probes or re-creates the sweep. It only runs workers and renews itself.
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=$JOBNAME"
        echo "#SBATCH --partition=$PARTITION"
        echo "#SBATCH --gres=gpu:$GPUS_PER_JOB"
        echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK"
        echo "#SBATCH --time=$WALLTIME"
        # B: → deliver USR1 to the batch shell (not the job steps); @180 → 3 min
        # before the wall-time SIGKILL, giving the resubmit time to land.
        echo "#SBATCH --signal=B:USR1@180"
        echo "#SBATCH --output=$LOG_DIR/slurm_%j.out"
        [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=$ACCOUNT"
        [[ -n "$QOS" ]]     && echo "#SBATCH --qos=$QOS"
        [[ -n "$MEM" ]]     && echo "#SBATCH --mem=$MEM"
        cat <<EOF
set -uo pipefail
cd "$REPO_DIR"
${ACTIVATE:+$ACTIVATE}

STOP_FILE="$LOG_DIR/STOP"
SWEEP_ID="$SWEEP_ID"
AGENTS_PER_GPU=$AGENTS_PER_GPU
PER_RUN_TIMEOUT="$PER_RUN_TIMEOUT"
RUNS_PER_AGENT=$RUNS_PER_AGENT
# Every worker in every job resolves to this exact sweep (same entity/project).
${WB_ENTITY:+export WANDB_ENTITY="$WB_ENTITY"}
${WB_PROJECT:+export WANDB_PROJECT="$WB_PROJECT"}

# One job → one replacement: on the pre-kill USR1, resubmit THIS script (unless a
# STOP sentinel says the search is being torn down), then exit cleanly. The sweep
# state is server-side, so only the trials in flight right now are lost — the
# controller just reissues them to the replacement.
resubmit() {
    if [[ -f "\$STOP_FILE" ]]; then
        echo "[\$(date '+%F %T')] USR1 but STOP present — not resubmitting."
        exit 0
    fi
    echo "[\$(date '+%F %T')] USR1 — resubmitting \$0 before wall-time kill."
    sbatch "\$0" || echo "[\$(date '+%F %T')] WARNING: resubmit sbatch failed."
    exit 0
}
trap resubmit USR1

# Pin each worker to a GPU SLURM actually gave THIS job. SLURM sets
# CUDA_VISIBLE_DEVICES to the allocated device ids; nvidia-smi ignores that and
# lists the whole node, so counting with nvidia-smi would let workers spill onto
# GPUs the job doesn't own (clobbering other users). Read the allocation instead,
# and set CUDA_VISIBLE_DEVICES per worker to one id from it.
if [[ -n "\${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra JOB_GPUS <<< "\$CUDA_VISIBLE_DEVICES"
elif [[ -n "\${SLURM_GPUS_ON_NODE:-}" ]]; then
    mapfile -t JOB_GPUS < <(seq 0 \$(( SLURM_GPUS_ON_NODE - 1 )))
else
    mapfile -t JOB_GPUS < <(seq 0 \$(( \$(nvidia-smi -L 2>/dev/null | wc -l) - 1 )))
fi
[[ "\${#JOB_GPUS[@]}" -lt 1 ]] && { echo "No GPUs allocated — aborting."; exit 1; }
echo "[\$(date '+%F %T')] job \$SLURM_JOB_ID: GPUs [\${JOB_GPUS[*]}] × \$AGENTS_PER_GPU agent(s) on sweep \$SWEEP_ID"

for gpu in "\${JOB_GPUS[@]}"; do
    for (( a=0; a<AGENTS_PER_GPU; a++ )); do
        (
            run_count=0
            while true; do
                [[ -f "\$STOP_FILE" ]] && break
                echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: (re)starting wandb agent"
                CUDA_VISIBLE_DEVICES=\$gpu timeout "\$PER_RUN_TIMEOUT" wandb agent --count 1 "\$SWEEP_ID"
                run_count=\$(( run_count + 1 ))
                if [[ "\$RUNS_PER_AGENT" -gt 0 && "\$run_count" -ge "\$RUNS_PER_AGENT" ]]; then
                    echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: reached \$RUNS_PER_AGENT runs — stopping."
                    break
                fi
                [[ -f "\$STOP_FILE" ]] && break
                sleep 5
            done
        ) &
    done
done

# `wait` is interrupted by the trapped USR1, which then resubmits + exits. Absent
# that (e.g. RUNS_PER_AGENT reached on every worker), it returns normally.
wait
EOF
    } > "$JOB_SCRIPT"
    chmod +x "$JOB_SCRIPT"

    _header "$ENV  ${C_DIM}(submitting $NUM_JOBS job(s))${C_RESET}"
    _info "Worker script: ${C_DIM}$JOB_SCRIPT${C_RESET}"
    for (( k=1; k<=NUM_JOBS; k++ )); do
        SUBMIT_OUT=$(sbatch "$JOB_SCRIPT" 2>&1) || { _error "sbatch failed: $SUBMIT_OUT"; continue; }
        _success "  $SUBMIT_OUT"
    done
    _info "Stop this search: ${C_BOLD}$0 --stop $(basename "$LOG_DIR")${C_RESET}"
done

_header "All jobs submitted"
_info "Monitor:  ${C_BOLD}squeue --me${C_RESET}   |   W&B dashboard (project contractionRL-Search)"
_info "Each job self-resubmits ~3 min before its wall-time; --stop halts the whole pool."
echo "" >&2

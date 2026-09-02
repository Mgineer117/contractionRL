#!/bin/bash
# SLURM entry point for every W&B sweep in this repo — the cluster twin of
# ./search.sh.
#
# The design rests on one fact about W&B sweeps: the controller lives on W&B's
# servers, and a worker is just `wandb agent --count 1 <sweep_id>` pulling the
# next trial from that shared queue. Workers never coordinate with each other,
# so a "cluster search" is simply a bag of those workers spread across SLURM
# jobs. Nothing about the search space, the metric, or the infeasibility handling
# changes — this script reuses search/build_sweep.py and search/sweep_runner.py
# verbatim (see search/configs/README.md for the space itself). It only adds the
# scheduling layer:
#
#   1. discover partitions / free GPUs / queue pressure and pick where to run;
#   2. create the sweep once on the login node (compute nodes then join it live);
#   3. smoke-test one trial to measure real GPU memory → agents-per-GPU;
#   4. generate a self-contained, self-resubmitting sbatch worker script;
#   5. submit N independent copies of it.
#
#   total workers = num_jobs × (gpus_per_job × agents_per_gpu)
#
# Time-limit behaviour
# --------------------
# Each job asks SLURM for `--signal=B:USR1@180`, so ~3 min before the wall-time
# kill it catches USR1 and resubmits itself (one job → one replacement), then
# exits cleanly. The sweep state is server-side, so the only thing ever lost is
# the handful of trials in flight at that moment — which the controller simply
# reissues. The pool therefore renews indefinitely until you stop it:
#
#   ./commands/search_cluster.sh --stop <log-dir-or-jobname>
#
# writes a STOP sentinel (so no in-flight trap resubmits) and `scancel --name`s
# every job sharing the name, including freshly resubmitted ones.
#
# Usage:
#   ./commands/search_cluster.sh                                   # fully interactive
#   ./commands/search_cluster.sh --algorithm c2rl-sac-cvstem --env car \
#       --partition gpu --num-jobs 4 --gpus-per-job 1 --time 12:00:00 -y
#   ./commands/search_cluster.sh --stop c2rl-sac-cvstem_classic-car-v0_20260722_120000
#
# Flags (all optional; anything omitted is prompted for):
#   --algorithm NAME     a stem in search/configs/ (e.g. ppo, c2rl-sac-cvstem)
#   --env NAME           car|cartpole|segway|quadrotor, an Isaac Lab
#                        task id, or 'all' for every classic env
#   --method M           bayes (default) | grid | random
#   --project NAME       W&B project for the sweep (default: build_sweep.py's
#                        contractionRL-Search). Use the main project for a fixed
#                        factorial design -- a gamma x seed grid is an experiment
#                        whose runs belong beside the other results, not a
#                        hyperparameter search to be kept apart from them.
#   --partition NAME     SLURM partition (default: recommended by discovery)
#   --num-jobs N         independent sbatch jobs to submit (default: prompted)
#   --gpus-per-job N     GPUs each job requests via --gres=gpu:N (default 1)
#   --gpu-type NAME      pin the GPU model: --gres=gpu:NAME:N (e.g. a100). Use it
#                        on heterogeneous partitions to keep jobs off GPUs your
#                        torch build has no kernels for (e.g. V100 / CC 7.0 under
#                        a CUDA-13 build). Applies to the smoke test too.
#   --constraint feat    --constraint=feat for sbatch/srun, if the site defines
#                        node features instead of typed gres. Unlike --gres this
#                        Does support or: --constraint 'H100|L40S'
#   --exclude-gpu-type T resolve every node carrying gres type T to a nodelist and
#                        --exclude it. This is how you say "any GPU except V100",
#                        which --gres cannot express (it takes one type only).
#                        Repeatable. NOTE: by default this is done automatically —
#                        every model too old for the installed torch is detected
#                        and excluded, so all the supported models stay usable.
#   --min-cc X.Y         compute-capability floor (default: read from the local
#                        torch's get_arch_list, else 7.5)
#   --no-auto-exclude    don't auto-exclude old GPU models
#   --exclude NODELIST   --exclude=NODELIST passed through verbatim
#   --agents-per-gpu N   override the smoke-test packing result
#   --max-agents-per-gpu N  hard ceiling on the packing result (default 16)
#   --min-probe-mb N     reject a smoke-test peak below this as a failed run
#                        (default 256; an idle GPU reads a few MB)
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
# See commands/search.sh: the script lives in commands/, its config space and its
# logs live in search/, and both are named off the repo root.
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_DIR/search/configs"
SEARCH_DIR="$REPO_DIR/search"

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
# GPU selection. On a heterogeneous partition (scavenger is), a bare
# --gres=gpu:N can land a job on any model — including a V100 (compute
# capability 7.0), which a CUDA-13 torch build has no kernels for: every CUDA op
# then dies with cudaErrorNoKernelImageForDevice at the first tensor op, before
# wandb logs anything, so the trial shows up as an empty "crashed" run. Pinning
# the type (or a node feature) keeps jobs on GPUs the install actually supports.
GPU_TYPE=""; CONSTRAINT=""; EXCLUDE=""; EXCLUDE_GPU_TYPES=""
# AUTO_EXCLUDE: keep every GPU model the installed torch has kernels for and drop
# only the too-old ones, instead of forfeiting the partition to a single pinned
# model. MIN_CC (compute capability ×10) is auto-detected from torch when empty.
AUTO_EXCLUDE=1; MIN_CC=""
# Packing guard rails. MIN_PROBE_MB: a run using less than this obviously never
# reached the GPU (an idle GPU reads a few MB) — reject rather than divide total
# VRAM by ~0 and get a nonsense count. MAX_AGENTS_PER_GPU: hard ceiling so even a
# legitimately tiny run can't ask for hundreds of processes on one GPU.
MIN_PROBE_MB=256; MAX_AGENTS_PER_GPU=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --project) WANDB_PROJECT_OVERRIDE="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --num-jobs) NUM_JOBS="$2"; shift 2 ;;
        --gpus-per-job) GPUS_PER_JOB="$2"; shift 2 ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --constraint) CONSTRAINT="$2"; shift 2 ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        # repeatable; auto-exclusion appends to the same list
        --exclude-gpu-type) EXCLUDE_GPU_TYPES+="${EXCLUDE_GPU_TYPES:+ }$2"; shift 2 ;;
        --min-cc) MIN_CC="$(awk -v v="$2" 'BEGIN{printf "%d", v*10 + 0.5}')"; shift 2 ;;
        --no-auto-exclude) AUTO_EXCLUDE=0; shift ;;
        --agents-per-gpu) AGENTS_PER_GPU="$2"; shift 2 ;;
        --max-agents-per-gpu) MAX_AGENTS_PER_GPU="$2"; shift 2 ;;
        --min-probe-mb) MIN_PROBE_MB="$2"; shift 2 ;;
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
# Order matters: drop the STOP sentinel first so any worker whose USR1 trap fires
# mid-scancel sees it and declines to resubmit, then scancel every job sharing
# the name (running + queued + already-resubmitted).
if [[ -n "$STOP_TAG" ]]; then
    need scancel
    LOG_DIR="$SEARCH_DIR/logs/$STOP_TAG"
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
CLASSIC_ENVS=("classic-car-v0" "classic-car-v1" "classic-cartpole-v0" "classic-segway-v0" "classic-quadrotor-v0")
ISAAC_ENVS=("Humanoid-PathTracking-v0" "Humanoid-VelTracking-v0" "Manipulator-PathTracking-v0" "Manipulator-VelTracking-v0" "Quadruped-PathTracking-v0" "Quadruped-VelTracking-v0")
if [[ -z "$ENV_ARG" ]]; then
    ENV_ARG=$(prompt_choice "Select env to sweep (or 'all' for every classic env):" \
        car cartpole segway quadrotor "${ISAAC_ENVS[@]}" all)
fi
if [[ "$ENV_ARG" == "all" ]]; then
    ENVS=("${CLASSIC_ENVS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0)             ENVS=("classic-car-v0") ;;
        car_v1|classic-car-v1)   ENVS=("classic-car-v1") ;;
        cartpole|classic-cartpole-v0)   ENVS=("classic-cartpole-v0") ;;
        segway|classic-segway-v0)       ENVS=("classic-segway-v0") ;;
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
# an allocatable state (idle/mixed) subtract GresUsed=gpu:… from Gres=gpu:…,
# crediting the difference to each of the node's Partitions.
# PART_TYPES  — models on allocatable (idle/mix) nodes, for the "what's free now"
#               table. PART_TYPES_ALL — models on every node regardless of state,
#               which is what the old-GPU auto-exclusion must key off: a V100 that
#               is merely alloc/down right now would otherwise go unclassified and
#               happily take a job the moment it frees up.
declare -A PART_FREE PART_TOTAL PART_PENDING PART_TYPES PART_TYPES_ALL

_gres_gpu_count() {  # extract the trailing integer of a gpu:[type:]N token
    grep -oE 'gpu:[^,]*' <<<"$1" | grep -oE '[0-9]+' | tail -n1
}
# Model name out of a typed gres token: "gpu:a100:4" -> "a100" ("" when untyped,
# e.g. a bare "gpu:4"). Surfaced per partition so a heterogeneous queue is
# Visible at selection time rather than discovered by a job dying on the wrong model.
_gres_gpu_type() {
    grep -oE 'gpu:[A-Za-z][A-Za-z0-9_.-]*:' <<<"$1" | head -n1 | cut -d: -f2
}

_header "Partition / GPU discovery"
{
    while IFS= read -r line; do
        state=$(grep -oE 'State=[^ ]+' <<<"$line" | cut -d= -f2 | tr 'A-Z' 'a-z')
        gres=$(grep -oE 'Gres=[^ ]+' <<<"$line" | cut -d= -f2-)
        gused=$(grep -oE 'GresUsed=[^ ]+' <<<"$line" | cut -d= -f2-)
        parts=$(grep -oE 'Partitions=[^ ]+' <<<"$line" | cut -d= -f2-)
        [[ "$gres" == *gpu:* ]] || continue
        # Record the model for every state first — auto-exclusion needs the full
        # inventory, not just what happens to be free at this instant.
        _t_all=$(_gres_gpu_type "$gres")
        if [[ -n "$_t_all" ]]; then
            IFS=',' read -ra _pall <<<"$parts"
            for p in "${_pall[@]}"; do
                [[ " ${PART_TYPES_ALL[$p]:-} " != *" $_t_all "* ]] && \
                    PART_TYPES_ALL[$p]="${PART_TYPES_ALL[$p]:-}${PART_TYPES_ALL[$p]:+ }$_t_all"
            done
        fi
        [[ "$state" =~ idle|mix ]] || continue
        total=$(_gres_gpu_count "$gres"); used=$(_gres_gpu_count "$gused")
        [[ -z "$total" ]] && continue; [[ -z "$used" ]] && used=0
        free=$(( total - used )); (( free < 0 )) && free=0
        gtype=$(_gres_gpu_type "$gres")
        IFS=',' read -ra plist <<<"$parts"
        for p in "${plist[@]}"; do
            PART_TOTAL[$p]=$(( ${PART_TOTAL[$p]:-0} + total ))
            PART_FREE[$p]=$(( ${PART_FREE[$p]:-0} + free ))
            # accumulate distinct models so a mixed partition shows "a100,v100"
            if [[ -n "$gtype" && " ${PART_TYPES[$p]:-} " != *" $gtype "* ]]; then
                PART_TYPES[$p]="${PART_TYPES[$p]:-}${PART_TYPES[$p]:+ }$gtype"
            fi
        done
    done < <(scontrol show nodes -o 2>/dev/null)
} || _warn "Could not parse 'scontrol show nodes' — showing raw sinfo only."

# pending GPU jobs per partition (queue pressure proxy)
while IFS='|' read -r p _; do
    [[ -z "$p" ]] && continue
    PART_PENDING[$p]=$(( ${PART_PENDING[$p]:-0} + 1 ))
done < <(squeue -h -t PD -o "%P" 2>/dev/null | sed 's/*//')

printf '\n  %-18s %8s %8s %10s  %s\n' "PARTITION" "FREE-GPU" "TOT-GPU" "PENDING" "GPU TYPES" >&2
_rule
RECOMMENDED=""; BEST_FREE=-1
for p in "${!PART_TOTAL[@]}"; do
    f=${PART_FREE[$p]:-0}; t=${PART_TOTAL[$p]:-0}; q=${PART_PENDING[$p]:-0}
    printf '  %-18s %8s %8s %10s  %s\n' "$p" "$f" "$t" "$q" "${PART_TYPES[$p]:-untyped}" >&2
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

# ── GPU model ─────────────────────────────────────────────────────────────── #
# A bare --gres=gpu:N on a mixed partition is a silent trial-killer: land on a
# model the torch build has no kernels for (V100 = CC 7.0 under a CUDA-13 build)
# and every CUDA op raises cudaErrorNoKernelImageForDevice at the first tensor
# op — before wandb logs anything, so the run is recorded as an empty "crashed".
# Rather than making the user pin one model (which forfeits every other GPU in a
# mixed partition), keep every model the local torch has kernels for and exclude
# only the too-old ones. That is strictly better on a queue like scavenger: the
# job can land on whichever supported GPU frees up first.
_avail_types="${PART_TYPES[$PARTITION]:-}"

# Compute capability by gres model name, ×10 so bash can compare them as ints.
# Extend this table when the cluster gains a model it doesn't know about.
declare -A GPU_CC=(
    [K80]=37   [P100]=60  [P40]=61   [GTX1080]=61 [TitanXp]=61
    [V100]=70  [TitanV]=70
    [T4]=75    [QuadroRTX6000]=75 [QuadroRTX8000]=75 [RTX6000]=75 [RTX2080Ti]=75
    [A100]=80  [A30]=80
    [A40]=86   [A6000]=86 [A5000]=86 [A4000]=86 [RTX3090]=86
    [L4]=89    [L40]=89   [L40S]=89  [RTX4090]=89
    [H100]=90  [H200]=90  [GH200]=90
    [B200]=100 [B100]=100
)

# The floor: the lowest arch the installed torch was built for. Asking torch
# itself (get_arch_list -> sm_75/sm_90/...) beats hardcoding, because the answer
# changes with the wheel's CUDA build — a cu130 wheel has no sm_70, a cu126 one
# does. Best-effort: if torch isn't importable on the login node, fall back to
# --min-cc (default 7.5, i.e. exclude Volta and older).
if [[ -z "$MIN_CC" ]]; then
    MIN_CC=$(python - <<'PY' 2>/dev/null
try:
    import torch
    ccs = []
    for a in torch.cuda.get_arch_list():
        d = a.split("_")[-1]
        if d.isdigit():
            ccs.append(int(d[:-1]) * 10 + int(d[-1]))
    print(min(ccs) if ccs else "")
except Exception:
    print("")
PY
)
fi
if [[ -z "$MIN_CC" ]]; then
    MIN_CC=75
    _info "torch not importable here — assuming min compute capability 7.5 (override with --min-cc)."
else
    _info "Installed torch supports compute capability ≥ ${C_BOLD}$((MIN_CC/10)).$((MIN_CC%10))${C_RESET}"
fi

# Classify the partition's models, and auto-exclude every one below the floor.
# Keyed off PART_TYPES_ALL (every node state), not the free-now list.
_all_types="${PART_TYPES_ALL[$PARTITION]:-$_avail_types}"
if [[ "$AUTO_EXCLUDE" -eq 1 && -n "$_all_types" ]]; then
    _keep=""; _drop=""; _unknown=""
    for _t in $_all_types; do
        _cc="${GPU_CC[$_t]:-}"
        if [[ -z "$_cc" ]]; then _unknown+="${_unknown:+ }$_t"; _keep+="${_keep:+ }$_t"
        elif (( _cc < MIN_CC )); then _drop+="${_drop:+ }$_t"
        else _keep+="${_keep:+ }$_t"; fi
    done
    [[ -n "$_unknown" ]] && _warn "Unknown GPU model(s): $_unknown — keeping them (add to GPU_CC, or use --exclude-gpu-type)."
    if [[ -n "$_drop" ]]; then
        _warn "Too old for this torch build: ${C_BOLD}${_drop}${C_RESET} — excluding those nodes."
        _success "Will use: ${C_BOLD}${_keep:-<none>}${C_RESET}"
        for _t in $_drop; do
            EXCLUDE_GPU_TYPES+="${EXCLUDE_GPU_TYPES:+ }$_t"
        done
        [[ -z "$_keep" ]] && { _error "No supported GPU model left in '$PARTITION'."; exit 1; }
    else
        _success "All GPU models in '$PARTITION' are supported: ${C_BOLD}${_all_types}${C_RESET}"
    fi
fi
# A gres type is a single token. Catching a multi-token answer here matters:
# "gpu:A B C:1" is not an or — sbatch would split it on whitespace and silently
# request only the first, or reject the directive outright.
if [[ -n "$GPU_TYPE" && ! "$GPU_TYPE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    _error "--gpu-type must be ONE gres type name (got: '$GPU_TYPE')."
    _info "SLURM's --gres cannot OR types. To allow several models instead:"
    _info "  • --exclude-gpu-type V100        keep every model EXCEPT that one"
    _info "  • --constraint 'H100|L40S'       if the site defines node FEATURES (sinfo -N -o '%N %f')"
    _info "  • or submit one batch per type"
    exit 1
fi
if [[ -n "$GPU_TYPE" && -n "$_avail_types" && " $_avail_types " != *" $GPU_TYPE "* ]]; then
    _warn "'$GPU_TYPE' is not among partition '$PARTITION' types ($_avail_types) — gres names are case-sensitive."
fi

# Resolve every excluded model to the nodes carrying it. One pass over the node
# list for all types. Scans all nodes (not just idle/mix like the discovery walk)
# so a currently-busy bad node is still excluded when it later frees up.
if [[ -n "$EXCLUDE_GPU_TYPES" ]]; then
    _ex_nodes=""
    while IFS= read -r line; do
        _nn=$(grep -oE 'NodeName=[^ ]+' <<<"$line" | cut -d= -f2)
        _gr=$(grep -oE 'Gres=[^ ]+' <<<"$line" | cut -d= -f2-)
        [[ -z "$_nn" ]] && continue
        for _t in $EXCLUDE_GPU_TYPES; do
            if [[ "$_gr" == *"gpu:$_t:"* ]]; then
                _ex_nodes+="${_ex_nodes:+,}$_nn"; break
            fi
        done
    done < <(scontrol show nodes -o 2>/dev/null)
    if [[ -z "$_ex_nodes" ]]; then
        _warn "No nodes found for gres type(s) '$EXCLUDE_GPU_TYPES' (case-sensitive) — nothing excluded."
    else
        EXCLUDE="${EXCLUDE:+$EXCLUDE,}$_ex_nodes"
        _success "Excluding $EXCLUDE_GPU_TYPES nodes: ${C_BOLD}$_ex_nodes${C_RESET}"
    fi
fi

# What both the smoke test and the generated jobs ask SLURM for.
gres_spec() { printf 'gpu:%s%s' "${GPU_TYPE:+$GPU_TYPE:}" "$1"; }
_success "GPU request: ${C_BOLD}--gres=$(gres_spec "$GPUS_PER_JOB")${C_RESET}${CONSTRAINT:+  --constraint=$CONSTRAINT}${EXCLUDE:+  --exclude=$EXCLUDE}"

# ── num-jobs / gpus-per-job ───────────────────────────────────────────────── #
if [[ -z "$NUM_JOBS" ]]; then
    _default_jobs=$(( BEST_FREE > 0 ? BEST_FREE / GPUS_PER_JOB : 1 )); (( _default_jobs < 1 )) && _default_jobs=1
    printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Independent sbatch jobs to submit${C_RESET} ${C_DIM}[$_default_jobs]${C_RESET}: " >&2
    read -r NUM_JOBS; NUM_JOBS="${NUM_JOBS:-$_default_jobs}"
fi
_success "Jobs: ${C_BOLD}$NUM_JOBS${C_RESET}  (×${GPUS_PER_JOB} GPU each)"

# ── Stage 2: create the sweep(s) on the login node ────────────────────────── #
# One sweep per env. Compute nodes join it live over the internet, so this is the
# only step that needs login-node connectivity. Parsing mirrors search.sh.
cd "$REPO_DIR"

# Every sbatch worker must authenticate to W&B as this account, or it silently
# runs detached from the sweep — the classic "jobs run but the sweep stays empty"
# failure. Verify credentials here, once: `wandb login` persists them to a
# shared-home ~/.netrc that the compute nodes read, so a login-node check is a
# check for every job. (If WANDB_API_KEY is exported instead, sbatch's default
# --export=all carries it; only a site with a non-shared home needs that.)
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
    LOG_DIR="$SEARCH_DIR/logs/${ALGORITHM}_${ENV}_${RUN_TS}"
    mkdir -p "$LOG_DIR"
    SWEEP_YAML="$LOG_DIR/sweep.yaml"
    if ! python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV" \
            --method "$METHOD" --out "$SWEEP_YAML" \
            ${WANDB_PROJECT_OVERRIDE:+--project "$WANDB_PROJECT_OVERRIDE"} > /dev/null; then
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
# Measure the real per-run footprint instead of guessing it. Run one trial under
# srun on the chosen partition with a single GPU, and sample the peak GPU memory
# of that run's process subtree via `--query-compute-apps` — not a per-GPU
# `memory.used` row, because nvidia-smi orders those by device index, not by
# owner, so on a non-isolated multi-GPU node "row 0" can be an idle neighbour
# (this is what produced the 1 MB → 20889 agents/GPU nonsense). Then:
#
#   agents_per_gpu = min( MAX_AGENTS_PER_GPU, floor( total_MB × safety / peak_MB ) )
#
# Safety leaves headroom for PyTorch's caching allocator + fragmentation, and for
# SAC's replay buffer still growing past the sampling window; a peak below
# MIN_PROBE_MB means the run never reached the GPU (crash / wrong env) and is
# rejected rather than trusted. Probed once on the first env as representative;
# override with --agents-per-gpu, or skip with --no-probe (which requires it).
if [[ -z "$AGENTS_PER_GPU" ]]; then
    if [[ "$PROBE" -ne 1 ]]; then
        _error "--no-probe requires --agents-per-gpu N."; exit 1
    fi
    PROBE_ENV="${ENVS[0]}"
    _header "Smoke test  ${C_DIM}(${ALGORITHM} on ${PROBE_ENV}, 1 GPU, ~2 min)${C_RESET}"
    # The config filename stem (e.g. c2rl-ppo-cvstem) selects the search space;
    # the train.py --algorithm value is the yaml's `algorithm:` field (e.g.
    # c2rl-ppo). build_sweep.py passes the latter, so the probe must too — the
    # stem is not a registered cfg entry point (only cvstem-lqr happens to match).
    read -r PROBE_ALGO NUM_ENVS < <(python - "$ALGORITHM" <<'PY'
import sys, yaml, pathlib
algo = sys.argv[1]
cfg = yaml.safe_load((pathlib.Path("search/configs")/f"{algo}.yaml").read_text())
print(cfg["algorithm"], int(cfg["num_envs"]))
PY
)
    CLASSIC_FLAG=""; [[ "$PROBE_ENV" == classic-* ]] && CLASSIC_FLAG="--classic"
    PROBE_LOG="${ENV_LOGDIR[${ENVS[0]}]}/probe.log"
    # The probe body runs on the compute node: launch one training run in the
    # background, poll for ~120s recording the max GPU memory used by the run's
    # Own process subtree (pid-matched, so it can't read an idle neighbour GPU),
    # kill the tree, and print a single MEM=peak/total line the launcher greps for.
    # Same gres/constraint as the real jobs: probing on a different GPU model
    # would size agents-per-GPU against memory the jobs never actually get.
    PROBE_OUT=$(srun --partition="$PARTITION" --gres="$(gres_spec 1)" \
        --cpus-per-task="$CPUS_PER_GPU" --time=00:10:00 --job-name="crl-probe" \
        ${CONSTRAINT:+--constraint="$CONSTRAINT"} ${EXCLUDE:+--exclude="$EXCLUDE"} \
        ${ACCOUNT:+--account="$ACCOUNT"} ${QOS:+--qos="$QOS"} \
        bash -c '
            set -uo pipefail
            cd "'"$REPO_DIR"'"
            '"${ACTIVATE:+$ACTIVATE;}"'
            python scripts/skrl/train.py '"$CLASSIC_FLAG"' \
                --task "'"$PROBE_ENV"'" --algorithm "'"$PROBE_ALGO"'" \
                --num_envs "'"$NUM_ENVS"'" --skip_final_eval \
                > "'"$PROBE_LOG"'" 2>&1 &
            child=$!
            # All pids in the training process tree — GPU memory is attributed to
            # whichever pid opened the CUDA context, which may be a worker child.
            subtree() { local p=$1 c; echo "$p"; for c in $(pgrep -P "$p" 2>/dev/null); do subtree "$c"; done; }
            peak=0
            total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc "0-9")
            for _ in $(seq 1 40); do
                kill -0 "$child" 2>/dev/null || break
                mine=" $(subtree "$child" | tr "\n" " ") "
                used=0
                while IFS="," read -r pid mem; do
                    pid="${pid//[[:space:]]/}"; mem="${mem//[[:space:]]/}"
                    [[ "$mem" =~ ^[0-9]+$ ]] || continue
                    [[ "$mine" == *" $pid "* ]] && used=$(( used + mem ))
                done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
                (( used > peak )) && peak=$used
                sleep 3
            done
            kill -TERM -"$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
            sleep 2; kill -KILL "$child" 2>/dev/null || true
            echo "MEM=${peak}/${total}"
        ' 2>>"$PROBE_LOG") || true
    _probe_failed() {  # shared failure exit: show why, then how to proceed
        _error "$1"
        _info  "Likely causes: the env wasn't activated (pass --activate 'conda activate <env>'),"
        _info  "the run crashed, or this site doesn't expose per-process GPU accounting."
        _info  "Last lines of the probe log ($PROBE_LOG):"
        tail -n 15 "$PROBE_LOG" 2>/dev/null | sed 's/^/      /' >&2 || true
        _info  "Or bypass the probe entirely with --agents-per-gpu N."
        exit 1
    }
    MEM_LINE=$(grep -oE 'MEM=[0-9]+/[0-9]+' <<<"$PROBE_OUT" | tail -n1)
    [[ -z "$MEM_LINE" ]] && _probe_failed "Smoke test produced no MEM= reading."
    PEAK_MB=${MEM_LINE#MEM=}; PEAK_MB=${PEAK_MB%/*}
    TOTAL_MB=${MEM_LINE#*/}
    # A peak below MIN_PROBE_MB means the run never really allocated on the GPU (an
    # idle GPU reads a few MB). Reject it — dividing total VRAM by ~0 is exactly
    # what yielded the absurd 20889 agents/GPU before this guard existed.
    if [[ -z "$PEAK_MB" || "$PEAK_MB" -lt "$MIN_PROBE_MB" ]]; then
        _probe_failed "Smoke test peak was ${PEAK_MB:-?} MB (< ${MIN_PROBE_MB} MB) — the run never reached the GPU."
    fi
    if [[ -z "$TOTAL_MB" || "$TOTAL_MB" -le 0 ]]; then
        _probe_failed "Smoke test could not read total GPU memory (got '${TOTAL_MB:-}')."
    fi
    UNCAPPED=$(python - "$PEAK_MB" "$TOTAL_MB" "$SAFETY" <<'PY'
import sys, math
peak, total, safety = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(max(1, math.floor(total * safety / peak)))
PY
)
    AGENTS_PER_GPU="$UNCAPPED"
    if (( AGENTS_PER_GPU > MAX_AGENTS_PER_GPU )); then
        _warn "Packing suggests ${UNCAPPED} agents/GPU from a ${PEAK_MB} MB run — capping at ${MAX_AGENTS_PER_GPU} (raise with --max-agents-per-gpu)."
        AGENTS_PER_GPU="$MAX_AGENTS_PER_GPU"
    fi
    _success "Peak ${C_BOLD}${PEAK_MB} MB${C_RESET} / ${TOTAL_MB} MB total → ${C_BOLD}${AGENTS_PER_GPU}${C_RESET} agent(s)/GPU (safety ${SAFETY})"
fi
[[ "$AGENTS_PER_GPU" =~ ^[0-9]+$ && "$AGENTS_PER_GPU" -ge 1 ]] || { _error "agents-per-gpu must be a positive integer (got '$AGENTS_PER_GPU')."; exit 1; }

AGENTS_PER_JOB=$(( GPUS_PER_JOB * AGENTS_PER_GPU ))
TOTAL_WORKERS=$(( NUM_JOBS * AGENTS_PER_JOB ))

# ── Summary + confirm ─────────────────────────────────────────────────────── #
_header "Cluster sweep summary"
_kv "Algorithm"      "$ALGORITHM"
_kv "Env(s)"         "${ENVS[*]}"
_kv "Partition"      "$PARTITION"
_kv "Jobs / env"     "$NUM_JOBS"
_kv "GPUs / job"     "$GPUS_PER_JOB  (--gres=$(gres_spec "$GPUS_PER_JOB"))"
[[ -n "$CONSTRAINT" ]] && _kv "Constraint" "$CONSTRAINT"
[[ -n "$EXCLUDE" ]]    && _kv "Excluded nodes" "$EXCLUDE"
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
    # sweep_runner.py's bad-trial resume in the same project instead of a default
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
        echo "#SBATCH --gres=$(gres_spec "$GPUS_PER_JOB")"
        [[ -n "$CONSTRAINT" ]] && echo "#SBATCH --constraint=$CONSTRAINT"
        [[ -n "$EXCLUDE" ]]    && echo "#SBATCH --exclude=$EXCLUDE"
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
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PER_RUN_TIMEOUT="$PER_RUN_TIMEOUT"
RUNS_PER_AGENT=$RUNS_PER_AGENT
# Free VRAM a worker demands before it will claim a trial. ~2.1 GiB steady
# plus the CMG regression's 840 MiB peak, rounded up. Override with
# MIN_FREE_MB in the environment for a heavier task.
MIN_FREE_MB=\${MIN_FREE_MB:-3500}
# Every worker in every job resolves to this exact sweep (same entity/project).
${WB_ENTITY:+export WANDB_ENTITY="$WB_ENTITY"}
${WB_PROJECT:+export WANDB_PROJECT="$WB_PROJECT"}

# One job → one replacement: on the pre-kill USR1, resubmit this script (unless a
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

# Pin each worker to a GPU SLURM actually gave this job. SLURM sets
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
            probe_fail=0
            while true; do
                [[ -f "\$STOP_FILE" ]] && break
                # Preflight the GPU before asking for a trial. On scavenger the
                # card is shared with jobs SLURM does not account for, and one
                # that is already full makes the trial die inside the CMG
                # regression's eigh (840 MiB) — after wandb has handed it a grid
                # cell, which is then spent. A worker in that state respawns
                # every ~20 s and can eat a whole 80-cell grid in minutes, so
                # waiting for room is strictly cheaper than starting blind.
                # Asked through torch, not nvidia-smi: on a MIG node \$gpu is a
                # UUID ("MIG-08963485-..."), which nvidia-smi -i rejects — the
                # query then returns empty and a "-n" test skips the guard
                # silently, which is how it read as working while doing nothing.
                # mem_get_info() reports the device CUDA will actually open,
                # index or MIG UUID alike.
                # stderr is KEPT, because how the probe fails is the whole
                # signal. mem_get_info() has to open a CUDA context of its own,
                # so on a full card the probe cannot run -- an unreadable read is
                # the STRONGEST evidence the GPU is busy, not an excuse to skip
                # the check. This block used to print "PROCEEDING BLIND" after 5
                # such reads and take a cell anyway. Measured cost of that on
                # job 10206316 (car_weak, 2026-08-29): one trial had grown to
                # 20430 MiB of a 23028 MiB A10, leaving 654 MiB, so every probe
                # failed, every worker proceeded blind, and 2196 sweep cells were
                # claimed and killed in gym.make -- one every 12 seconds for 8
                # hours. Fails closed now.
                probe_out=\$(CUDA_VISIBLE_DEVICES=\$gpu python -c \\
                    'import torch;print(torch.cuda.mem_get_info()[0]//1048576)' 2>&1 | tail -1)
                free_mb=\$(printf '%s' "\$probe_out" | grep -oE '^[0-9]+\$' || true)
                if [[ -z "\$free_mb" ]] && printf '%s' "\$probe_out" \\
                        | grep -qiE 'out of memory|CUDA error|no CUDA-capable'; then
                    free_mb=0        # full is not unreadable — fall through to the wait
                fi
                if [[ ! "\$free_mb" =~ ^[0-9]+\$ ]]; then
                    # A probe that fails for some OTHER reason (no torch, dead
                    # driver) is a broken worker, not a busy card. Retire it
                    # loudly: the job keeps its GPU but stops eating cells, which
                    # is strictly better than guessing.
                    probe_fail=\$(( probe_fail + 1 ))
                    if [[ "\$probe_fail" -lt 5 ]]; then
                        echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: VRAM probe unreadable (\$probe_fail/5) — waiting; last: \$probe_out"
                        sleep 60
                        continue
                    fi
                    echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: VRAM probe broken 5x — RETIRING this worker, NOT taking cells. last: \$probe_out"
                    break
                elif [[ "\$free_mb" -lt "\$MIN_FREE_MB" ]]; then
                    probe_fail=0
                    echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: \${free_mb} MiB free < \$MIN_FREE_MB — waiting, NOT taking a cell"
                    sleep 120
                    continue
                else
                    probe_fail=0
                fi
                echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: (re)starting wandb agent"
                # Private wandb dir per (job, gpu, agent). Every worker used to
                # share \$REPO_DIR/wandb, and the reaper below deletes run-* older
                # than a minute -- so with several workers running concurrently,
                # one worker's reaper deleted another worker's live run directory.
                # The victim then died at wandb.finish() with
                # "OSError: [Errno 116] Stale file handle", after training had
                # completed and its metric was already logged. Measured: 14 of 70
                # trials. Scoping the directory makes the reaper touch only runs
                # this agent created.
                # On scratch, not in the repo. /u is quota'd at 103 GB and 500k
                # inodes and is shared with conda envs and other projects; a
                # single active run's run-*.wandb transaction log can reach
                # several GB, and 15 concurrent workers put /u at 94 GB of 103 GB
                # within an hour. ~/scratch is a separate filesystem with 10 TB
                # and no inode cap, so the sweep's local cache cannot exhaust the
                # home quota and take the cluster session down with it.
                export WANDB_DIR="\$HOME/scratch/wandb_sweeps/\${SLURM_JOB_ID:-nojob}_g\${gpu}_a\${a}"
                mkdir -p "\$WANDB_DIR"
                # `timeout` signals ONLY its direct child. wandb agent runs
                # train.py as a GRANDCHILD, which therefore survives the watchdog
                # as an orphan and keeps its VRAM for the rest of the job.
                # Measured on job 10206316: two trainers still resident at
                # 31h43m under this 24h cap, one holding 20430 MiB, which is what
                # starved every later trial on that card.
                #
                # So run the trial in its own session -- the bash -c writes the
                # new session leader's pid (== the new process group id) before
                # exec'ing, which is race-free, unlike scraping it with pgrep --
                # and kill the whole group once the call returns, however it
                # returned.
                _pgf=\$(mktemp)
                setsid --wait bash -c 'echo \$\$ > "\$1"; shift; exec "\$@"' _ "\$_pgf" \\
                    env CUDA_VISIBLE_DEVICES=\$gpu timeout "\$PER_RUN_TIMEOUT" \\
                    wandb agent --count 1 "\$SWEEP_ID" || true
                _pgid=\$(cat "\$_pgf" 2>/dev/null || true); rm -f "\$_pgf"
                if [[ "\$_pgid" =~ ^[0-9]+\$ ]]; then
                    kill -TERM -"\$_pgid" 2>/dev/null || true
                    sleep 10
                    kill -KILL -"\$_pgid" 2>/dev/null || true
                fi
                # Reap the local run cache. A c2rl-ppo trial leaves ~1 GB in
                # wandb/run-*, so an 80-trial grid alone is ~80 GB and a 103 GB
                # home quota fills before the sweep finishes — at which point
                # every worker dies writing checkpoints, and the failure looks
                # like a training bug rather than a full disk. Only run-* is
                # removed (already streamed to the server); offline-run-* is
                # left alone because for those the local copy is the data.
                # Scoped to this agent's private dir -- see above.
                find "\$WANDB_DIR" -maxdepth 1 -name 'run-*' -type d \\
                    -mmin +1 -exec rm -rf {} + 2>/dev/null || true
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
_info "Monitor:  ${C_BOLD}squeue --me${C_RESET}   |   W&B dashboard (project ${WANDB_PROJECT_OVERRIDE:-contractionRL-Search})"
_info "Each job self-resubmits ~3 min before its wall-time; --stop halts the whole pool."
echo "" >&2

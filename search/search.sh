#!/bin/bash
# Single entry point for every W&B sweep in this repo.
#
# Prompts for algorithm / env / agents / GPU, previews the exact metric and
# hyperparameter ranges that will be searched, then launches the sweep detached
# (nohup) so it survives closing the terminal.
#
# The searched space is NOT defined here — it lives in search/configs/, one
# yaml per algorithm, applying to every env. This script only discovers those
# configs, and search/build_sweep.py turns the chosen one into a W&B sweep yaml.
# To change what is searched, or to add an algorithm, edit search/configs/
# (see its README.md); nothing here needs touching.
#
# Usage:
#   ./search.sh                                        # fully interactive
#   ./search.sh --algorithm cvstem-lqr --env car --gpu 0 --num-agents 3 -y
#   ./search.sh --algorithm c2rl-ppo-cvstem --env all --gpu 0 --runs-per-agent 40 -y
#
# Flags (all optional; anything omitted is prompted for):
#   --algorithm NAME     a stem in search/configs/ (e.g. ppo, c3m, cvstem-lqr)
#   --env NAME           car|cartpole|segway|turtlebot|quadrotor, an Isaac Lab
#                        task id, or 'all' for every classic env
#   --num-agents N       parallel wandb agents per env (default 3)
#   --gpu N              pin every agent to this GPU (default: round-robin)
#   --method M           bayes (default) | grid | random
#   --timeout T          per-run watchdog (default 24h)
#   --runs-per-agent N   stop each agent after N runs (default 0 = unbounded)
#   -y, --yes            skip the confirmation prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"

NUM_AGENTS=""
ALGORITHM=""
ENV_ARG=""
GPU_ARG=""
METHOD="bayes"
PER_RUN_TIMEOUT=24h
RUNS_PER_AGENT=0
YES=0
DETACHED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --gpu) GPU_ARG="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        --runs-per-agent) RUNS_PER_AGENT="$2"; shift 2 ;;
        -y|--yes) YES=1; shift ;;
        # Internal: set on the nohup'd re-exec (see "Detach" below) so the
        # second pass launches instead of detaching again.
        --detached) DETACHED=1; YES=1; shift ;;
        -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

prompt_choice() {
    local prompt_text="$1"; shift
    local opts=("$@") opt
    echo "$prompt_text" >&2
    select opt in "${opts[@]}"; do
        if [[ -n "$opt" ]]; then echo "$opt"; return; fi
        echo "Invalid choice, try again." >&2
    done
}

# ── Algorithm ───────────────────────────────────────────────────────────── #
# Discovered by globbing configs/ — adding a yaml there is all it takes to make
# a new algorithm selectable here.
mapfile -t ALGOS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
if [[ "${#ALGOS[@]}" -eq 0 ]]; then
    echo "No algorithm configs found in $CONFIG_DIR." >&2
    exit 1
fi
if [[ -z "$ALGORITHM" ]]; then
    ALGORITHM=$(prompt_choice "Select algorithm to sweep:" "${ALGOS[@]}")
fi
if [[ ! -f "$CONFIG_DIR/$ALGORITHM.yaml" ]]; then
    echo "No config '$ALGORITHM.yaml' in $CONFIG_DIR. Available: ${ALGOS[*]}" >&2
    exit 1
fi

# ── Env ─────────────────────────────────────────────────────────────────── #
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
            # Isaac Lab task id — accept only ids that actually exist.
            _found=0
            for e in "${ISAAC_ENVS[@]}"; do [[ "$e" == "$ENV_ARG" ]] && _found=1; done
            if [[ "$_found" -ne 1 ]]; then
                echo "Unknown env '$ENV_ARG'. Classic: car/cartpole/segway/turtlebot/quadrotor (or 'all'). Isaac Lab: ${ISAAC_ENVS[*]}" >&2
                exit 1
            fi
            ENVS=("$ENV_ARG")
            ;;
    esac
fi

if [[ -z "$NUM_AGENTS" ]]; then
    read -r -p "Number of parallel wandb agents to spawn per env [3]: " NUM_AGENTS
    NUM_AGENTS="${NUM_AGENTS:-3}"
fi

# ── GPU ─────────────────────────────────────────────────────────────────── #
# CUDA_VISIBLE_DEVICES=<index> for a nonexistent index silently hides the only
# real GPU from PyTorch, so the index is validated against what nvidia-smi
# actually reports rather than trusted.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "nvidia-smi reports 0 GPUs — aborting." >&2
    exit 1
fi
if [[ -z "$GPU_ARG" ]]; then
    mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
    GPU_CHOICE=$(prompt_choice "Select GPU to pin this sweep to (or 'all' to round-robin across every GPU):" "${GPU_NAMES[@]}" "all")
    [[ "$GPU_CHOICE" != "all" ]] && GPU_ARG="${GPU_CHOICE%%:*}"
fi
if [[ -n "$GPU_ARG" ]]; then
    if ! [[ "$GPU_ARG" =~ ^[0-9]+$ ]] || [[ "$GPU_ARG" -ge "$NUM_GPUS" ]]; then
        echo "--gpu $GPU_ARG is out of range — nvidia-smi reports $NUM_GPUS GPU(s) (0-$((NUM_GPUS - 1)))." >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR/.."

# ── Preview ─────────────────────────────────────────────────────────────── #
if [[ "$DETACHED" -ne 1 ]]; then
    echo ""
    echo "=========================================="
    echo "About to sweep: $ALGORITHM   (method: $METHOD)"
    echo "Env(s): ${ENVS[*]}"
    echo "Agents per env: $NUM_AGENTS   GPU: ${GPU_ARG:-all detected (round-robin)}"
    if [[ "$RUNS_PER_AGENT" -gt 0 ]]; then
        echo "Runs per agent: $RUNS_PER_AGENT (total per env: $((RUNS_PER_AGENT * NUM_AGENTS)))"
    else
        echo "Runs per agent: unbounded (bayes never exhausts — kill to stop)"
    fi
    echo "=========================================="
    # Preview the REAL generated sweep, not a re-description of it — what is
    # shown here is exactly what wandb will be handed.
    python search/build_sweep.py --algorithm "$ALGORITHM" --env "${ENVS[0]}" \
        --method "$METHOD"
    echo "=========================================="
    echo ""

    if [[ "$YES" -ne 1 ]]; then
        read -r -p "Launch this sweep now, detached via nohup? [y/N] " CONFIRM
        if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
            echo "Aborted — nothing launched."
            exit 0
        fi
    fi
fi

# ── Detach ──────────────────────────────────────────────────────────────── #
# Everything above is interactive, so detaching happens only now, once every
# choice is resolved: re-exec this script under nohup with those choices as
# explicit flags plus --detached, then exit. The child ignores SIGHUP and its
# agent subshells inherit that, so closing the terminal leaves the sweep running.
if [[ "$DETACHED" -ne 1 ]]; then
    mkdir -p "$SCRIPT_DIR/logs"
    NOHUP_LOG="$SCRIPT_DIR/logs/nohup_${ALGORITHM}_${ENV_ARG}_$(date '+%Y%m%d_%H%M%S').log"
    nohup "$SCRIPT_DIR/search.sh" \
        --algorithm "$ALGORITHM" --env "$ENV_ARG" --num-agents "$NUM_AGENTS" \
        --method "$METHOD" --timeout "$PER_RUN_TIMEOUT" \
        --runs-per-agent "$RUNS_PER_AGENT" \
        ${GPU_ARG:+--gpu "$GPU_ARG"} \
        --detached > "$NOHUP_LOG" 2>&1 &
    disown
    echo "Launched (PID $!) — detached via nohup, safe to close this terminal."
    echo "Tail progress with: tail -f $NOHUP_LOG"
    exit 0
fi

# ── Launch ──────────────────────────────────────────────────────────────── #
RUN_TS="$(date '+%Y%m%d_%H%M%S')"

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU="${GPU_ARG:-$((i % NUM_GPUS))}"

    LOG_DIR="$SCRIPT_DIR/logs/${ALGORITHM}_${ENV}_${RUN_TS}"
    mkdir -p "$LOG_DIR"

    echo "=========================================="
    echo "Initializing WandB Sweep for $ENV ($ALGORITHM) on GPU $GPU..."
    echo "=========================================="

    # Project and sweep name both come from build_sweep.py: ONE fixed project
    # (contractionRL-Search) for everything, with the sweep named
    # "<env>-<algorithm>". Relaunching the same env+algorithm therefore reuses
    # the same name — wandb sweep names are not unique and each launch still
    # gets its own sweep id, so the runs stay grouped by what they're sweeping
    # rather than by when it was launched.
    #
    # Generated yamls are per-launch artifacts, so they go to the log dir rather
    # than into search/ (which is versioned, and holds the space itself).
    SWEEP_YAML="$LOG_DIR/sweep.yaml"
    # `|| { ...; continue; }` on each fallible step below, because this script
    # runs under `set -euo pipefail`: without it a failure here aborts the WHOLE
    # script, so with --env all a single bad env would silently take every
    # remaining env down with it instead of being skipped.
    if ! python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV" \
            --method "$METHOD" --out "$SWEEP_YAML" > /dev/null; then
        echo "Failed to build a sweep yaml for $ENV — skipping."
        continue
    fi

    # `|| true` twice, for two different set -e traps:
    #   - the command substitution itself fails if `wandb sweep` exits nonzero;
    #   - `pipefail` fails the grep pipeline when grep matches NOTHING, which is
    #     exactly the "sweep creation failed" case the check below handles.
    # Without these the -z/Error branch is unreachable dead code.
    SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1) || true
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}' || true)
    if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
        echo "Failed to create sweep for $ENV."
        echo "Output: $SWEEP_INIT_OUTPUT"
        continue
    fi
    echo "Sweep ID created: $SWEEP_ID"
    echo "Starting $NUM_AGENTS self-restarting agents in parallel..."

    for j in $(seq 1 "$NUM_AGENTS"); do
        LOGFILE="$LOG_DIR/agent_${j}.log"
        echo "Starting Agent $j (auto-restart, ${PER_RUN_TIMEOUT} watchdog)... Logging to $LOGFILE"
        (
            run_count=0
            while true; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] (re)starting wandb agent for $SWEEP_ID" >> "$LOGFILE"
                CUDA_VISIBLE_DEVICES=$GPU timeout "$PER_RUN_TIMEOUT" wandb agent --count 1 "$SWEEP_ID" >> "$LOGFILE" 2>&1
                status=$?
                run_count=$((run_count + 1))
                # bayes never exhausts its space, so --runs-per-agent is the
                # only thing that ever stops an agent short of being killed.
                if [[ "$RUNS_PER_AGENT" -gt 0 && "$run_count" -ge "$RUNS_PER_AGENT" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb agent exited (status=$status) — reached ${RUNS_PER_AGENT}/${RUNS_PER_AGENT} runs, stopping" >> "$LOGFILE"
                    break
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb agent exited (status=$status) — restarting in 5s" >> "$LOGFILE"
                sleep 5
            done
        ) &
    done

    echo "All $NUM_AGENTS agents for $ENV launched in the background."
    echo ""
done

echo "Monitor progress on your W&B dashboard!"
echo "Each agent restarts itself — kill this script's process group to stop."
wait

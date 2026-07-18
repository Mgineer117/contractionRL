#!/bin/bash
# Single entry point for all classic-env W&B sweeps (replaces the old
# run_c2rl_ppo_sweeps.sh and run_c3m_sweeps.sh — see their git history for
# the watchdog-restart rationale this script inherits unchanged).
#
# Usage:
#   ./run_sweep.sh --algorithm <ppo|sac|c3m|lqr|sdlqr|cvstem-lqr|c2rl> \
#                   --env <car|cartpole|segway|turtlebot|quadrotor> \
#                   [--rl ppo|sac] [--cm ccm|cvstem] \
#                   [--num-agents N] [--timeout T] [--runs-per-agent N] [--gpu N]
#
# --rl and --cm only apply to --algorithm c2rl (C2RL is trained via one of two
# RL sub-agents, PPO or SAC, on top of one of two contraction-metric-generator
# synthesis pipelines, CCM or CV-STEM — see c2rl.py's module docstring). Every
# other algorithm has a single fixed parameter set and --rl/--cm are rejected.
# Any of --algorithm/--env/--rl/--cm may be omitted and will be prompted for
# interactively.
#
# --num-agents (default 3), --timeout (default 24h, per-run watchdog), and
# --runs-per-agent (default 0 = unbounded) mirror the old scripts' flags.
#
# --gpu N pins EVERY env/agent in this run to CUDA_VISIBLE_DEVICES=N. Omit to
# round-robin across every GPU nvidia-smi reports (the old behavior) — one
# env per GPU, cycling if there are more envs than GPUs.
set -euo pipefail

NUM_AGENTS=3
PER_RUN_TIMEOUT=24h
RUNS_PER_AGENT=0
ALGORITHM=""
ENV_ARG=""
RL=""
CM=""
GPU_ARG=""

usage() {
    grep -E '^# ?' "$0" | sed -E 's/^# ?//' | head -n 20
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --rl) RL="$2"; shift 2 ;;
        --cm|--cm_formulation|--cmg_method) CM="$2"; shift 2 ;;
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        --runs-per-agent) RUNS_PER_AGENT="$2"; shift 2 ;;
        --gpu) GPU_ARG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib_sweep_params.sh
source "$SCRIPT_DIR/lib_sweep_params.sh"  # prompt_choice, metric_block, parameters_block

# ── Algorithm ───────────────────────────────────────────────────────────── #
if [[ -z "$ALGORITHM" ]]; then
    ALGORITHM=$(prompt_choice "Select algorithm to sweep:" ppo sac c3m lqr sdlqr cvstem-lqr c2rl)
fi
ALGORITHM=$(echo "$ALGORITHM" | tr '[:upper:]' '[:lower:]')

case "$ALGORITHM" in
    ppo|sac|c3m|lqr|sdlqr|cvstem-lqr|c2rl) ;;
    *) echo "Unknown --algorithm '$ALGORITHM' — expected ppo/sac/c3m/lqr/sdlqr/cvstem-lqr/c2rl." >&2; exit 1 ;;
esac

if [[ "$ALGORITHM" != "c2rl" && ( -n "$RL" || -n "$CM" ) ]]; then
    echo "--rl/--cm only apply to --algorithm c2rl (algorithm '$ALGORITHM' has a single fixed parameter set) — ignoring." >&2
    RL=""
    CM=""
fi

# ── C2RL sub-choices ────────────────────────────────────────────────────── #
if [[ "$ALGORITHM" == "c2rl" ]]; then
    if [[ -z "$RL" ]]; then
        RL=$(prompt_choice "C2RL: select the RL sub-agent driving the deployed policy:" ppo sac)
    fi
    RL=$(echo "$RL" | tr '[:upper:]' '[:lower:]')
    case "$RL" in
        ppo|sac) ;;
        *) echo "Unknown --rl '$RL' — expected ppo or sac." >&2; exit 1 ;;
    esac

    if [[ -z "$CM" ]]; then
        CM=$(prompt_choice "C2RL: select the contraction-metric-generator synthesis method:" ccm cvstem)
    fi
    CM=$(echo "$CM" | tr '[:upper:]' '[:lower:]')
    case "$CM" in
        ccm|cvstem) ;;
        *) echo "Unknown --cm '$CM' — expected ccm or cvstem." >&2; exit 1 ;;
    esac

    TRAIN_ALGORITHM="c2rl-${RL}"
    ALGO_TAG="c2rl_${RL}_${CM}"
else
    TRAIN_ALGORITHM="$ALGORITHM"
    ALGO_TAG="${ALGORITHM//-/_}"
fi

# ── Env ─────────────────────────────────────────────────────────────────── #
ALL_ENVS=("classic-cartpole-v0" "classic-turtlebot-v0" "classic-segway-v0" "classic-car-v0" "classic-quadrotor-v0")

if [[ -z "$ENV_ARG" ]]; then
    ENV_ARG=$(prompt_choice "Select env to sweep (or 'all' for every classic env):" car segway turtlebot cartpole quadrotor all)
fi

if [[ "$ENV_ARG" == "all" ]]; then
    ENVS=("${ALL_ENVS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0) ENV_NORM="classic-car-v0" ;;
        segway|classic-segway-v0) ENV_NORM="classic-segway-v0" ;;
        turtlebot|classic-turtlebot-v0) ENV_NORM="classic-turtlebot-v0" ;;
        cartpole|classic-cartpole-v0) ENV_NORM="classic-cartpole-v0" ;;
        quadrotor|classic-quadrotor-v0) ENV_NORM="classic-quadrotor-v0" ;;
        *) echo "Unknown env '$ENV_ARG' — expected car/segway/turtlebot/cartpole/quadrotor (or classic-*-v0), or 'all'." >&2; exit 1 ;;
    esac
    ENVS=("$ENV_NORM")
fi

# Real GPU count on THIS machine — see run_c2rl_ppo_sweeps.sh's original
# comment: CUDA_VISIBLE_DEVICES=<index> for a nonexistent index silently
# hides the only real GPU from PyTorch, so envs round-robin over however
# many GPUs nvidia-smi actually reports (unless --gpu pins a single one).
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "nvidia-smi reports 0 GPUs — aborting." >&2
    exit 1
fi
if [[ -n "$GPU_ARG" ]]; then
    if ! [[ "$GPU_ARG" =~ ^[0-9]+$ ]] || [[ "$GPU_ARG" -ge "$NUM_GPUS" ]]; then
        echo "--gpu $GPU_ARG is out of range — nvidia-smi reports $NUM_GPUS GPU(s) (0-$((NUM_GPUS - 1)))." >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR/.."

# ── Per-algorithm metric + parameters block ────────────────────────────── #
# PPO-based algorithms are on-policy and benefit from massively parallel
# envs; everything else (SAC-like: samples from a large buffer) needs far
# fewer — mirrors train.py's _default_num_envs_classic.
case "$TRAIN_ALGORITHM" in
    ppo|c2rl-ppo) NUM_ENVS=1024 ;;
    *) NUM_ENVS=64 ;;
esac

# wandb ignores a run's own --wandb_project when it's launched by a sweep —
# the sweep's own `project:` (set below) is what every launched run lands
# in. Leaving this unset does NOT fall back to a shared "contractionRL"
# project — wandb instead auto-derives a project name from the local path
# (observed: "contractionRL-scripts_skrl"), silently mixing unrelated runs
# together. So every invocation of this script gets its OWN project, tagged
# with algorithm and a launch timestamp, so two sweeps of the same algorithm
# started minutes apart (e.g. a retry, or testing) never share a project and
# never get mistaken for each other.
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
PROJECT_LINE="project: contractionRL-${ALGO_TAG}-${RUN_TS}"

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU="${GPU_ARG:-$((i % NUM_GPUS))}"

    LOG_DIR="logs/search/${ALGO_TAG}_${ENV}_${RUN_TS}"
    mkdir -p "$LOG_DIR"

    echo "=========================================="
    echo "Initializing WandB Sweep for $ENV ($ALGO_TAG) on GPU $GPU..."
    echo "=========================================="

    SWEEP_YAML="search/sweep_${ALGO_TAG}_${ENV}_${RUN_TS}.yaml"
    {
        echo "program: scripts/skrl/train.py"
        echo "$PROJECT_LINE"
        echo "method: bayes"
        metric_block
        echo ""
        parameters_block
        echo ""
        echo "command:"
        echo "  - \${env}"
        echo "  - python"
        echo "  - \${program}"
        echo "  - \"--classic\""
        echo "  - \"--task\""
        echo "  - \"$ENV\""
        echo "  - \"--algorithm\""
        echo "  - \"$TRAIN_ALGORITHM\""
        echo "  - \"--num_envs\""
        echo "  - \"$NUM_ENVS\""
        echo "  - \${args}"
    } > "$SWEEP_YAML"

    SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1)
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}')

    if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
        echo "Failed to create sweep for $ENV."
        echo "Output: $SWEEP_INIT_OUTPUT"
        continue
    fi

    echo "Sweep ID created: $SWEEP_ID"
    echo "Starting $NUM_AGENTS self-restarting agents in parallel..."

    for j in $(seq 1 "$NUM_AGENTS"); do
        LOGFILE="$LOG_DIR/agent_${j}.log"
        if [[ "$RUNS_PER_AGENT" -gt 0 ]]; then
            echo "Starting Agent $j (auto-restart, ${PER_RUN_TIMEOUT} watchdog, ${RUNS_PER_AGENT} runs)... Logging to $LOGFILE"
        else
            echo "Starting Agent $j (auto-restart, ${PER_RUN_TIMEOUT} watchdog)... Logging to $LOGFILE"
        fi
        (
            run_count=0
            while true; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] (re)starting wandb agent for $SWEEP_ID" >> "$LOGFILE"
                CUDA_VISIBLE_DEVICES=$GPU timeout "$PER_RUN_TIMEOUT" wandb agent --count 1 "$SWEEP_ID" >> "$LOGFILE" 2>&1
                status=$?
                run_count=$((run_count + 1))
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
echo "Each agent now restarts itself forever — Ctrl+C (or kill this script's process group) to stop."
wait

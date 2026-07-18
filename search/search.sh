#!/bin/bash
# Interactive front end for run_sweep.sh: prompts for algorithm (and, for
# c2rl, the RL sub-agent + CM synthesis method), env, and how many parallel
# wandb agents to spawn; previews the exact metric + hyperparameter ranges
# that will be searched; then launches run_sweep.sh under `nohup` + `disown`
# so the sweep keeps running after this terminal closes.
#
# Usage: ./search.sh   (no flags — everything is prompted for)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib_sweep_params.sh
source "$SCRIPT_DIR/lib_sweep_params.sh"  # prompt_choice, metric_block, parameters_block

prompt_int() {
    local prompt_text="$1" default="$2" reply
    read -r -p "$prompt_text [$default]: " reply
    echo "${reply:-$default}"
}

# ── Gather choices ─────────────────────────────────────────────────────── #
ALGORITHM=$(prompt_choice "Select algorithm to sweep:" ppo sac c3m lqr sdlqr cvstem-lqr c2rl)

RL=""
CM=""
if [[ "$ALGORITHM" == "c2rl" ]]; then
    RL=$(prompt_choice "C2RL: select the RL sub-agent driving the deployed policy:" ppo sac)
    CM=$(prompt_choice "C2RL: select the contraction-metric-generator synthesis method:" ccm cvstem)
fi

ENV_ARG=$(prompt_choice "Select env to sweep (or 'all' for every classic env):" car segway turtlebot cartpole quadrotor all)

NUM_AGENTS=$(prompt_int "Number of parallel wandb agents to spawn per env" 3)

# ── GPU picker ──────────────────────────────────────────────────────────── #
# Classic-env training (train.py's --classic route) has no --device flag —
# GPU selection is CUDA_VISIBLE_DEVICES re-indexing only, so CPU/MPS aren't
# reachable here; if nvidia-smi finds nothing, skip straight to run_sweep.sh
# and let its own hard "0 GPUs" check fire.
GPU_ARG=""
mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
if [[ "${#GPU_NAMES[@]}" -gt 0 ]]; then
    GPU_CHOICE=$(prompt_choice "Select GPU to pin this sweep to (or 'all' to round-robin across every detected GPU):" "${GPU_NAMES[@]}" "all")
    if [[ "$GPU_CHOICE" != "all" ]]; then
        GPU_ARG="${GPU_CHOICE%%:*}"
    fi
fi

# ── Preview ─────────────────────────────────────────────────────────────── #
if [[ "$ALGORITHM" == "c2rl" ]]; then
    ALGO_TAG="c2rl_${RL}_${CM}"
    SUMMARY_LABEL="c2rl (rl=$RL, cm=$CM)"
else
    ALGO_TAG="${ALGORITHM//-/_}"
    SUMMARY_LABEL="$ALGORITHM"
fi

echo ""
echo "=========================================="
echo "About to sweep: $SUMMARY_LABEL on env(s): $ENV_ARG"
echo "Agents per env: $NUM_AGENTS"
echo "GPU: ${GPU_ARG:-all detected (round-robin)}"
echo "=========================================="
metric_block
echo ""
parameters_block
echo "=========================================="
echo ""

read -r -p "Launch this sweep now, detached via nohup? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted — nothing launched."
    exit 0
fi

mkdir -p "$SCRIPT_DIR/../logs/search"
TS="$(date '+%Y%m%d_%H%M%S')"
NOHUP_LOG="$SCRIPT_DIR/../logs/search/nohup_${ALGO_TAG}_${TS}.log"

RUN_SWEEP_ARGS=(--algorithm "$ALGORITHM" --env "$ENV_ARG" --num-agents "$NUM_AGENTS")
if [[ "$ALGORITHM" == "c2rl" ]]; then
    RUN_SWEEP_ARGS+=(--rl "$RL" --cm "$CM")
fi
if [[ -n "$GPU_ARG" ]]; then
    RUN_SWEEP_ARGS+=(--gpu "$GPU_ARG")
fi

nohup "$SCRIPT_DIR/run_sweep.sh" "${RUN_SWEEP_ARGS[@]}" > "$NOHUP_LOG" 2>&1 &
disown

echo "Launched (PID $!) — detached via nohup, safe to close this terminal."
echo "Tail progress with: tail -f $NOHUP_LOG"

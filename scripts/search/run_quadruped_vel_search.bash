#!/bin/bash
# SAC / PPO hyperparameter search — Quadruped-VelTracking-v0
#
# Usage:
#   bash scripts/search/run_quadruped_vel_search.bash              # SAC, 20 trials
#   bash scripts/search/run_quadruped_vel_search.bash --algorithm PPO
#   bash scripts/search/run_quadruped_vel_search.bash --sweep_id <ID>   # join existing
#
# Each trial is a full Isaac Sim process; run one at a time per GPU.
# For multi-GPU add CUDA_VISIBLE_DEVICES=N before each `wandb agent` call.

set -euo pipefail

TASK="Quadruped-VelTracking-v0"
PROJECT="contractionRL-search"
ALGORITHM="SAC"
COUNT=20
NUM_ENVS=128
TIMESTEPS=50000
LOG_DIR="logs/search"
SWEEP_ID=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm)  ALGORITHM="$2"; shift 2 ;;
        --count)      COUNT="$2";     shift 2 ;;
        --num_envs)   NUM_ENVS="$2";  shift 2 ;;
        --timesteps)  TIMESTEPS="$2"; shift 2 ;;
        --sweep_id)   SWEEP_ID="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  contractionRL $ALGORITHM Search — $TASK"
echo "  Trials: $COUNT  |  Envs: $NUM_ENVS  |  Steps/trial: $TIMESTEPS"
echo "============================================================"

# ── 1. Create sweep (or reuse existing) ──────────────────────────────────────
if [ -z "$SWEEP_ID" ]; then
    echo "Creating WandB sweep..."
    SWEEP_OUT=$(python scripts/search/search_algo.py \
        --task       "$TASK" \
        --algorithm  "$ALGORITHM" \
        --project    "$PROJECT" \
        --num_envs   "$NUM_ENVS" \
        --timesteps  "$TIMESTEPS" \
        --count      0 2>&1)
    echo "$SWEEP_OUT"
    SWEEP_ID=$(echo "$SWEEP_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

    if [ -z "$SWEEP_ID" ]; then
        echo "ERROR: Could not parse sweep ID from output above."
        exit 1
    fi
fi

echo "Sweep ID: $SWEEP_ID"

# ── 2. Determine wandb entity ─────────────────────────────────────────────────
ENTITY=$(python -c "import wandb; print(wandb.api.default_entity or 'UIUC-LIRA')" 2>/dev/null || echo "UIUC-LIRA")

# ── 3. Run agent ─────────────────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/quadruped_vel_${ALGORITHM}_${SWEEP_ID}.txt"
echo "Running $COUNT trials → $LOG_FILE"
echo ""

wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}" \
    --count "$COUNT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Search complete."
echo "  Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}"
echo "  Log:       $LOG_FILE"
echo "============================================================"

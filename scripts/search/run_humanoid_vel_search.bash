#!/bin/bash
# SAC / PPO hyperparameter search — Humanoid-VelTracking-v0
#
# Usage:
#   bash scripts/search/run_humanoid_vel_search.bash
#   bash scripts/search/run_humanoid_vel_search.bash --algorithm PPO
#   bash scripts/search/run_humanoid_vel_search.bash --sweep_id <ID>

set -euo pipefail

TASK="Humanoid-VelTracking-v0"
PROJECT="contractionRL-search"
ALGORITHM="SAC"
COUNT=20
NUM_ENVS=64        # H1 is heavier; fewer envs than quadruped
TIMESTEPS=50000
LOG_DIR="logs/search"
SWEEP_ID=""

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
        echo "ERROR: Could not parse sweep ID."
        exit 1
    fi
fi

ENTITY=$(python -c "import wandb; print(wandb.api.default_entity or 'UIUC-LIRA')" 2>/dev/null || echo "UIUC-LIRA")
LOG_FILE="$LOG_DIR/humanoid_vel_${ALGORITHM}_${SWEEP_ID}.txt"
echo "Sweep ID: $SWEEP_ID"
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

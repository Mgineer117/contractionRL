#!/bin/bash
# Usage: ./c3m.sh [num-agents]
NUM_AGENTS=${1:-1}

LOG_DIR="logs/search/quadruped_path_tracking_c3m"
mkdir -p $LOG_DIR

echo "Initializing WandB Sweep for Quadruped-PathTracking-v0 (c3m)..."
SWEEP_ID=$(python search/quadruped_path_tracking/c3m.py | tail -n 1)

if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
    echo "Failed to create sweep."
    exit 1
fi

echo "Sweep ID created: $SWEEP_ID"
echo "Starting $NUM_AGENTS agents in parallel..."

for i in $(seq 1 $NUM_AGENTS); do
    LOGFILE="$LOG_DIR/agent_${i}.log"
    echo "Starting Agent $i... Logging to $LOGFILE"
    wandb agent --count 100 $SWEEP_ID > $LOGFILE 2>&1 &
done

echo "All $NUM_AGENTS agents launched in the background."
echo "Monitor progress on your W&B dashboard!"
wait

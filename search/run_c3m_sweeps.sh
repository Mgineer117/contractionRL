#!/bin/bash
# Usage: ./run_c3m_sweeps.sh [num-agents-per-env]
NUM_AGENTS=${1:-3}

ENVS=("classic-cartpole-v0" "classic-turtlebot-v0" "classic-segway-v0" "classic-car-v0")
GPUS=(0 1 2 3)

cd "$(dirname "$0")/.."

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU="${GPUS[$i]}"
    
    LOG_DIR="logs/search/c3m_${ENV}"
    mkdir -p $LOG_DIR
    
    echo "=========================================="
    echo "Initializing WandB Sweep for $ENV on GPU $GPU..."
    echo "=========================================="
    
    cat <<YML > search/sweep_${ENV}.yaml
program: scripts/skrl/train.py
method: bayes
metric:
  name: "Reward / Total reward (mean)"
  goal: maximize

parameters:
  agent.lbd:
    distribution: uniform
    min: 0.01
    max: 3.0
  agent.eps:
    distribution: log_uniform_values
    min: 1e-3
    max: 1.0
  agent.W_lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  agent.actor_lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  agent.cmg_updates_per_policy_update:
    values: [1, 5, 10, 30]

command:
  - \${env}
  - python
  - \${program}
  - "--classic"
  - "--task"
  - "$ENV"
  - "--algorithm"
  - "c3m"
  - "--num_envs"
  - "4"
  - "--analytical"
  - "dynamics"
  - \${args}
YML

    # Parse sweep ID from wandb stdout
    SWEEP_INIT_OUTPUT=$(wandb sweep search/sweep_${ENV}.yaml 2>&1)
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}')
    
    if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
        echo "Failed to create sweep for $ENV."
        echo "Output: $SWEEP_INIT_OUTPUT"
        continue
    fi
    
    echo "Sweep ID created: $SWEEP_ID"
    echo "Starting $NUM_AGENTS agents in parallel..."
    
    for j in $(seq 1 $NUM_AGENTS); do
        LOGFILE="$LOG_DIR/agent_${j}.log"
        echo "Starting Agent $j... Logging to $LOGFILE"
        CUDA_VISIBLE_DEVICES=$GPU wandb agent --count 100 $SWEEP_ID > $LOGFILE 2>&1 &
    done

    echo "All $NUM_AGENTS agents for $ENV launched in the background."
    echo ""
done

echo "Monitor progress on your W&B dashboard!"
wait

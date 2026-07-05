#!/bin/bash
# Usage: ./run_c3m_sweeps.sh [num-agents-per-env] [per-run-timeout]
#
# `wandb agent --count 100` occasionally stops advancing partway through a
# sweep — e.g. after one long-running C3M job (individual runs here can take
# several hours) completes and prints its final eval, no further run gets
# launched even though `--count` hasn't been reached and the sweep itself is
# still active on the W&B server. This was NOT reproduced as a hang inside
# this repo's own training code (a from-scratch classic-C3M run — with
# WANDB_MODE=offline, which still exercises the same local wandb-service
# process as a real run — completes and exits cleanly on its own); it looks
# like `wandb agent`'s own long-lived polling loop losing track of the sweep.
# Rather than chase that upstream, each slot below is a small watchdog:
# `wandb agent --count 1` does exactly one run and exits, and the surrounding
# `while true` relaunches it forever, so the sweep keeps moving even if any
# single `wandb agent`/training subprocess invocation stalls or dies. PER_RUN_TIMEOUT
# force-kills a stuck iteration so the loop can't wedge on a true hang either.
NUM_AGENTS=${1:-3}
PER_RUN_TIMEOUT=${2:-24h}  # generous vs. the ~9h a single C3M run has taken

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
  # Stability / convergence_score = contraction_rate / overshoot — rewards
  # fast contraction with little overshoot, higher is better. Requires
  # eval_interval > 0 in the C3M agent cfg (default 100) so C3MSkrlTrainer.eval()
  # actually logs it periodically during the sweep.
  name: "Stability / convergence_score"
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
  agent.actor_architecture:  # policy (CLActor w1/w2) hidden layers
    values: [[64, 64], [128, 128], [256, 256], [128, 128, 128]]
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
    echo "Starting $NUM_AGENTS self-restarting agents in parallel..."

    for j in $(seq 1 $NUM_AGENTS); do
        LOGFILE="$LOG_DIR/agent_${j}.log"
        echo "Starting Agent $j (auto-restart, ${PER_RUN_TIMEOUT} watchdog)... Logging to $LOGFILE"
        (
            while true; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] (re)starting wandb agent for $SWEEP_ID" >> "$LOGFILE"
                CUDA_VISIBLE_DEVICES=$GPU timeout "$PER_RUN_TIMEOUT" wandb agent --count 1 "$SWEEP_ID" >> "$LOGFILE" 2>&1
                status=$?
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

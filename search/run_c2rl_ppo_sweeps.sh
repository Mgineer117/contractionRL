#!/bin/bash
# Usage: ./run_c2rl_ppo_sweeps.sh [num-agents-per-env] [per-run-timeout] [runs-per-agent] [env]
#
# runs-per-agent (default: unbounded) caps how many `--count 1` trials each
# watchdog slot executes before it stops restarting — e.g. 3 agents x 30 runs
# each = 90 trials per env. Omit (or pass 0) to keep the original forever-
# restart behavior (Ctrl+C to stop).
#
# env (default: unset) restricts the sweep to a single classic env — accepts
# either the short name (car/segway/turtlebot/cartpole) or the full task id
# (classic-car-v0/...). Omit to sweep all four classic envs in parallel, one
# per GPU, as before.
#
# Mirrors run_c3m_sweeps.sh (see that file's comment for why each slot is a
# self-restarting `wandb agent --count 1` watchdog loop instead of a single
# `--count 100` agent). Sweeps C2RL-PPO's Mahalanobis-reward / CMG-synthesis
# hyperparameters:
#   - agent.discount_factor                              — deployed policy's γ
#     (single policy, no con/opt duality; low γ has empirically contracted
#     better here — see car's skrl_c2rl_ppo_cfg.yaml default of 0.01)
#   - cm.cm_eps                                           — strict-definiteness
#     margin on the contraction LMI (cmg_method="ccm" and "cvstem" both use it)
#   - cm.cvstem_r_scaler                                  — R = cvstem_r_scaler·I
#     in the CV-STEM Riccati term (cmg_method="cvstem" only — the repo default
#     as of 2026-07-17; see c2rl.py's cmg_method field)
NUM_AGENTS=${1:-3}
PER_RUN_TIMEOUT=${2:-24h}
RUNS_PER_AGENT=${3:-0}  # 0 = unbounded
ENV_ARG=${4:-}

# The algorithm-specific project this script's runs (and sweeps) log to —
# keeps c2rl_ppo sweeps out of the shared "contractionRL" project other
# algorithms (C3M, SD-LQR, ...) log to. Overrides each env cfg's
# agent.experiment.wandb_kwargs.project via train.py's --wandb_project.
WANDB_PROJECT="contractionRL-c2rl_ppo"

ALL_ENVS=("classic-cartpole-v0" "classic-turtlebot-v0" "classic-segway-v0" "classic-car-v0")
ALL_GPUS=(0 1 2 3)

if [[ -z "$ENV_ARG" ]]; then
    ENVS=("${ALL_ENVS[@]}")
    GPUS=("${ALL_GPUS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0) ENV_NORM="classic-car-v0" ;;
        segway|classic-segway-v0) ENV_NORM="classic-segway-v0" ;;
        turtlebot|classic-turtlebot-v0) ENV_NORM="classic-turtlebot-v0" ;;
        cartpole|classic-cartpole-v0) ENV_NORM="classic-cartpole-v0" ;;
        *) echo "Unknown env '$ENV_ARG' — expected car/segway/turtlebot/cartpole (or classic-*-v0)." >&2; exit 1 ;;
    esac
    ENVS=("$ENV_NORM")
    GPU_NORM=0
    for k in "${!ALL_ENVS[@]}"; do
        [[ "${ALL_ENVS[$k]}" == "$ENV_NORM" ]] && GPU_NORM="${ALL_GPUS[$k]}"
    done
    GPUS=("$GPU_NORM")
fi

cd "$(dirname "$0")/.."

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU="${GPUS[$i]}"

    LOG_DIR="logs/search/c2rl_ppo_${ENV}"
    mkdir -p $LOG_DIR

    echo "=========================================="
    echo "Initializing WandB Sweep for $ENV on GPU $GPU..."
    echo "=========================================="

    cat <<YML > search/sweep_c2rl_ppo_${ENV}.yaml
program: scripts/skrl/train.py
method: bayes
metric:
  # Same rationale as C3M's sweep (run_c3m_sweeps.sh): C2RL is trained on the
  # Mahalanobis reward, which is a proxy for the contraction certificate, not
  # the certificate itself — optimize the certified quantity directly. AUC
  # (area under the normalized error curve e(t)/e(0)) is minimized, unlike
  # C3M's sweep which maximizes contraction_score — see contraction_metrics.py.
  # Unlike C3M, C2RL has no eval_interval knob: StatManagerEnvWrapper injects
  # a fresh Stability/* dict into info["log"] every env.step() (contraction_
  # metrics.py's StatManagerEnvWrapper.step), and C2RLSkrlTrainer forwards it
  # to agent.track_data every step (_forward_env_log) — so this metric is
  # always populated once the eval-buffer envs complete their first episode.
  name: "Stability/auc_mean"
  goal: minimize

parameters:
  agent.discount_factor:
    distribution: uniform
    min: 0.01
    max: 0.999
  cm.cm_eps:
    distribution: log_uniform_values
    min: 1e-3
    max: 1.0
  cm.cvstem_r_scaler:
    distribution: log_uniform_values
    min: 0.1
    max: 10.0

command:
  - \${env}
  - python
  - \${program}
  - "--classic"
  - "--task"
  - "$ENV"
  - "--algorithm"
  - "c2rl-ppo"
  - "--wandb_project"
  - "$WANDB_PROJECT"
  # PPO is on-policy — matches train.py's own _DEFAULT_NUM_ENVS_PPO_CLASSIC.
  - "--num_envs"
  - "1024"
  # Analytical dynamics is the DEFAULT for classic contraction envs (train.py:
  # use_empirical_dynamics defaults False). Pass --use_empirical_dynamics only
  # to learn a NeuralDynamics instead.
  - \${args}
YML

    # Parse sweep ID from wandb stdout
    SWEEP_INIT_OUTPUT=$(wandb sweep search/sweep_c2rl_ppo_${ENV}.yaml 2>&1)
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

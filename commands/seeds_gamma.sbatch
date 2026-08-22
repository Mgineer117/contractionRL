#!/bin/bash
#SBATCH --job-name=crl-seeds
#SBATCH --output=cluster_runs/seeds_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#
# TEN SEEDS at ONE discount factor for ONE env. Not a sweep: commands/run_seeds.sh
# trains a FIXED config across seeds and aggregates the Stability CSVs, which is
# what a per-gamma seed distribution needs. (search_cluster.sh would instead put
# W&B agents on a search space; its own header says it searches, this does not.)
#
# One job per (env, gamma) cell so a single cell can be resubmitted without
# touching the other five, and so the three partitions can be used at once.
#
# --extra carries the only axis that is not already in the env yaml. Everything
# else -- timesteps, num_envs, the pinned PPO hyperparameters -- stays in the
# config, so the CLI names only what actually varies.
#
# python -u equivalent: run_seeds.sh writes its own per-run logs, and this
# wrapper's stdout is small, but keep the flush habit (see docs/cluster.md 1b).
set -uo pipefail
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$HOME/contractionRL"
echo "[$(date)] host=$(hostname) env=$ENVNAME gamma=$GAMMA seeds=10"

# Cluster checkout still has run_seeds.sh at the repo ROOT (no commands/ dir there
# yet -- that move is local-only), so call it by the path that exists on the host.
# --detached is the internal flag run_seeds.sh sets on its own nohup re-exec. It
# is exactly what an sbatch needs: WITHOUT it the script forks itself under nohup
# and returns immediately, so the batch job "completes" in ~15 s and SLURM tears
# down the cgroup, killing the training it just launched. With it, the seed loop
# runs in the FOREGROUND and the job lives as long as the work does.
# --discount_factor, NOT --agent.discount_factor=. The dotted form only works
# where scripts/skrl/train_utils.py has apply_cli_dotted_overrides; the cluster's
# checkout does not, and there the arg lands in argparse's leftovers and is
# DROPPED WITHOUT A WORD. That is not hypothetical: the 2026-08-21 batch ran all
# six cells at the yaml's pinned discount_factor: 0.01 and produced byte-identical
# per-seed AUCs for gamma 0.01 and gamma 0.999. The explicit flag exists in both
# checkouts, so it is the portable one.
./run_seeds.sh --detached \
    --algorithms c2rl-ppo \
    --env "$ENVNAME" \
    --seeds 10 \
    --gpu 0 \
    --parallel 5 \
    --tag "gamma${GAMMA}_${ENVNAME}" \
    --extra "--discount_factor $GAMMA" \
    -y
rc=$?
echo "[$(date)] done rc=$rc env=$ENVNAME gamma=$GAMMA"
exit 0

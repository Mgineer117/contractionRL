#!/bin/bash
#SBATCH --job-name=crl-buildcm
#SBATCH --output=cluster_runs/buildcm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#
# Solve one env's offline CM dataset (the {x -> W*(x)} the C2RL cvstem path
# loads). Driven by ENVNAME.
#
# NO GPU: this is a MOSEK interior-point solve on the CPU. Asking for one would
# queue behind GPU contention for nothing.
#
# --mem=200G, not the 32G the training jobs use: a previous build was OOM-killed
# on a 30 GB box (docs/cluster.md §1), and the quadrotor solve currently resident
# is holding ~86 GB. The joint SDP materializes an LMI block per sample.
#
# NEVER scavenger, whatever the queue looks like. It is preemptible, and a
# cartpole build there was preempted and requeued after ~2.5 h with SLURM
# truncating --output on restart, losing the progress twice over. A multi-hour
# solve has no checkpoint, so preemption is total loss. IllinoisComputes-GPU with
# --account=huytran1-ic is 3-day and not preempted (docs/cluster.md §1); the wall
# here is still the standing 2 days.
#
# python -u: the solve prints one line per phase and nothing for hours after. A
# wall-limit SIGTERM does not flush a block-buffered pipe, so without -u a job
# that dies at the wall leaves a log claiming it never started.
set -uo pipefail
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$HOME/contractionRL"
echo "[$(date)] host=$(hostname) build cm dataset env=$ENVNAME"
python -u scripts/build_cm_dataset.py --task "$ENVNAME" --algorithm c2rl-ppo
rc=$?
echo "[$(date)] done rc=$rc env=$ENVNAME"
# Propagate the real status: the RL jobs chain on afterok, so a failed solve must
# NOT release 30 training jobs that would each die in synthesize_cmg.
exit $rc

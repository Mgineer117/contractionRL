#!/bin/bash
# Plain-run gamma x seed grid for c2rl-ppo. Not a wandb sweep.
#
# A sweep hands a worker a grid cell before the trial proves it can run, so a
# node that OOMs in the first seconds spends the cell and the grid finishes with
# holes -- 15 of 80 went that way in five minutes. Here the cell is the SLURM
# array index: nothing is consumed by failing, and re-running a lost cell is
# `sbatch --array=<idx>`.
#
#   sbatch --array=0-159%14 --partition=scavenger search/run_grid.sbatch
#
# Index layout (160 = 2 arms x 4 gammas x 20 seeds):
#   arm   = idx % 2          0 = normalization on, 1 = off
#   gamma = gammas[idx/2/20]
#   seed  = (idx/2) % 20
# Arm alternates on consecutive indices on purpose: any contiguous slice of the
# array is ~half on and half off, so splitting the range across partitions
# cannot align the treatment with the hardware.
#
#SBATCH --job-name=crl-grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out

set -uo pipefail
REPO=/u/minjae5/contractionRL
cd "$REPO" || exit 1
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GAMMAS=(0.01 0.9 0.99 0.999)
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
ARM=$(( IDX % 2 ))
CELL=$(( IDX / 2 ))
GAMMA=${GAMMAS[$(( CELL / 20 ))]}
SEED=$(( CELL % 20 ))

if [[ "$ARM" -eq 0 ]]; then
    NORM=(--use_value_norm true  --use_reward_norm true)
    TAG=normON
else
    NORM=(--use_value_norm false --use_reward_norm false)
    TAG=normOFF
fi
NAME="c2rl_${TAG}_g${GAMMA}_s${SEED}"
echo "[$(date '+%F %T')] idx=$IDX -> $NAME on $(hostname), CUDA=$CUDA_VISIBLE_DEVICES"

# Wait for real room on this device before starting. Asked through torch, not
# nvidia-smi: on a MIG node CUDA_VISIBLE_DEVICES is a UUID, which `nvidia-smi -i`
# rejects -- it returns empty and a naive check then passes. mem_get_info()
# resolves index and MIG UUID alike. ~2.1 GiB steady + the CMG regression's
# 840 MiB eigh workspace, rounded up.
MIN_FREE_MB=${MIN_FREE_MB:-3500}
wait_for_vram() {
    local waited=0
    while [[ "$waited" -lt 3600 ]]; do
        local free
        free=$(python -c 'import torch;print(torch.cuda.mem_get_info()[0]//1048576)' 2>/dev/null | tail -1)
        [[ "$free" =~ ^[0-9]+$ ]] && [[ "$free" -ge "$MIN_FREE_MB" ]] && return 0
        echo "[$(date '+%F %T')] $NAME: ${free:-unreadable} MiB free < $MIN_FREE_MB — waiting"
        sleep 120; waited=$(( waited + 120 ))
    done
    echo "[$(date '+%F %T')] $NAME: no room after 1h — giving up, cell is NOT lost (resubmit --array=$IDX)"
    return 1
}

for attempt in 1 2 3; do
    wait_for_vram || exit 75
    echo "[$(date '+%F %T')] $NAME: attempt $attempt"
    python scripts/skrl/train.py --classic --task classic-cartpole-v0 --algorithm c2rl-ppo \
        --num_envs 1024 --ref_offset 1 --num_timesteps 200000 \
        --discount_factor "$GAMMA" --seed "$SEED" "${NORM[@]}" \
        --wandb_project contractionRL --wandb_run_name "$NAME" --wandb_batch grid_plain
    rc=$?
    # Reap the local run cache regardless of outcome: ~1 GB per trial against a
    # 103 GB home quota, and a full disk kills every other job on the node too.
    # Both locations: train.py defaults WANDB_DIR to ~/scratch/wandb (so runs
    # land in ~/scratch/wandb/wandb/run-*), but a caller that pre-sets WANDB_DIR
    # -- as `wandb agent` does -- puts them in $REPO/wandb instead. ~/scratch is
    # on the same /u quota, so cleaning only one of the two still fills the disk.
    for d in "$REPO/wandb" "$HOME/scratch/wandb/wandb"; do
        [[ -d "$d" ]] && find "$d" -maxdepth 1 -name 'run-*' -type d -mmin +1 \
            -exec rm -rf {} + 2>/dev/null
    done
    [[ "$rc" -eq 0 ]] && { echo "[$(date '+%F %T')] $NAME: done"; exit 0; }
    echo "[$(date '+%F %T')] $NAME: exit $rc — retrying"
    sleep 60
done
echo "[$(date '+%F %T')] $NAME: failed 3 attempts"
exit 1

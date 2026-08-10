#!/bin/bash
# env x gamma x seed grid for c2rl-ppo, for the contraction-taxonomy experiment.
# Plain SLURM array runs, NOT a wandb sweep -- see run_grid.sbatch's header for
# why (a sweep spends a grid cell on a trial that OOMs in its first seconds; here
# the cell IS the array index, so nothing is consumed by failing).
#
# Usage -- the env list is positional, so ONE file serves both arms:
#
#   # Arm B, the within-plant control: 2 envs x 5 gammas x 30 seeds = 300
#   sbatch --array=0-299%14 --partition=scavenger search/run_class_gamma_grid.sbatch \
#       "classic-car-v0 classic-car_weak-v0"
#
#   # Arm A, across plants: 6 envs x 5 gammas x 30 seeds = 900
#   sbatch --array=0-899%14 --partition=scavenger search/run_class_gamma_grid.sbatch \
#       "classic-car-v0 classic-quadrotor-v0 classic-segway-v0 classic-cartpole-v0 classic-tora-v0 classic-auv-v0"
#
# Index layout (gamma FASTEST, seed slowest):
#   gamma = GAMMAS[idx % G]
#   env   = ENVS[(idx / G) % E]
#   seed  = idx / (G * E)
# gamma varies fastest ON PURPOSE: gamma is the treatment, so any contiguous
# slice of the array handed to a partition is gamma-balanced and the treatment
# can never align with the hardware. Splitting a range across scavenger and
# ic-express is then safe by construction.
#
# A lost cell is re-run with `sbatch --array=<idx> ... "<same env list>"`. The
# env list must be passed identically or the index decodes to a different cell.
#
#SBATCH --job-name=crl-clsgrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# 18 h, not 6. A cell is ~2.5 h, but ref_length AUTO scales with 1/(1-gamma), so a
# gamma=0.999 cell carries a 2504-wide observation against gamma=0.5's 19-wide and
# takes far longer; at 6 h those cells time out (measured: one died at 6:00:26).
# The limit must be right at submit time -- scontrol update TimeLimit= is REFUSED
# to a normal user, who may only lower a limit, never raise it. Override with
# `sbatch --time=07:45:00` on ic-express, whose partition cap is 8 h; do not put
# gamma >= 0.99 cells there.
#SBATCH --time=18:00:00
#SBATCH --output=%x-%A_%a.out

set -uo pipefail
REPO=/u/minjae5/contractionRL
cd "$REPO" || exit 1
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

read -r -a ENVS <<< "${1:?pass the env list as one quoted argument}"
GAMMAS=(0.5 0.9 0.95 0.99 0.999)
NSEEDS=${NSEEDS:-30}
STEPS=${STEPS:-200000}
NUM_ENVS=${NUM_ENVS:-1024}

G=${#GAMMAS[@]}
E=${#ENVS[@]}
TOTAL=$(( G * E * NSEEDS ))
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
if [[ "$IDX" -ge "$TOTAL" ]]; then
    echo "idx $IDX >= grid size $TOTAL (${E} envs x ${G} gammas x ${NSEEDS} seeds) -- bad --array range"
    exit 2
fi

GAMMA=${GAMMAS[$(( IDX % G ))]}
ENV=${ENVS[$(( (IDX / G) % E ))]}
SEED=$(( IDX / (G * E) ))

SHORT=${ENV#classic-}; SHORT=${SHORT%-v0}
NAME="c2rl_${SHORT}_g${GAMMA}_s${SEED}"

# env / algorithm / gamma / seed_idx reach wandb as CONFIG keys, written by
# train.py itself (see the wandb_kwargs block there) -- not as tags, and not via
# WANDB_* env vars. Config is what the runs table can group and filter on, and
# doing it in train.py means every run gets the fields, not just ones launched
# through this script.
echo "[$(date '+%F %T')] idx=$IDX/$TOTAL -> $NAME on $(hostname), CUDA=$CUDA_VISIBLE_DEVICES"

# Wait for real room on THIS device before starting. Asked through torch, not
# nvidia-smi: on a MIG node CUDA_VISIBLE_DEVICES is a UUID, which `nvidia-smi -i`
# rejects -- it returns empty and a naive check then passes. mem_get_info()
# resolves index and MIG UUID alike.
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
    echo "[$(date '+%F %T')] $NAME: no room after 1h — cell NOT lost (resubmit --array=$IDX)"
    return 1
}

for attempt in 1 2 3; do
    wait_for_vram || exit 75
    echo "[$(date '+%F %T')] $NAME: attempt $attempt"
    python scripts/skrl/train.py --classic --task "$ENV" --algorithm c2rl-ppo \
        --num_envs "$NUM_ENVS" --ref_offset 1 --num_timesteps "$STEPS" \
        --discount_factor "$GAMMA" --seed "$SEED" \
        --wandb_project contractionRL --wandb_run_name "$NAME" --wandb_batch class_gamma
    rc=$?
    # Reap the local run cache regardless of outcome: ~1 GB per trial against a
    # 103 GB home quota, and a full disk kills every OTHER job on the node too.
    # BOTH locations: train.py defaults WANDB_DIR to ~/scratch/wandb (so runs land
    # in ~/scratch/wandb/wandb/run-*), but a caller that pre-sets WANDB_DIR -- as
    # `wandb agent` does -- puts them in $REPO/wandb. ~/scratch is on the SAME /u
    # quota, so cleaning only one of the two still fills the disk.
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

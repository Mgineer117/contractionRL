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
NSEEDS=${NSEEDS:-10}
# THE WINDOW IS PINNED, AND THIS IS A TREATMENT DECISION, not a convenience
# override of the yaml. ref_length AUTO sizes the observation to 1/(1-gamma),
# so it makes observation width a FUNCTION of the treatment: 19-wide at
# gamma=0.5 against 2504-wide at gamma=0.999. Any "effect of gamma" measured
# that way is really an effect of gamma AND of a 132x change in input
# dimension, and the two cannot be separated after the fact.
#
# Pinning it to the full episode gives every policy a Markov state -- the whole
# reference trajectory is in the observation regardless of gamma -- so the
# architecture is identical across cells and gamma is the only thing that moves.
# The cost is that low-gamma policies carry a wider input than their horizon
# needs, which is a capacity/compute cost, not a bias, and it is CONSTANT across
# gamma so it cannot confound.
#
# 500 == max_episode_len for all four envs here (car, car_weak, segway,
# cartpole: dt=0.03, time_bound=15.0). Verify before adding an env with a
# different episode length -- pinning past the cap silently does nothing.
PIN_REF_LEN=${PIN_REF_LEN:-500}
# No STEPS / NUM_ENVS here on purpose: those live in each env's yaml. See the
# train.py invocation below.

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
# 2800, not 3500: measured steady use is ~2.1 GiB plus the CMG regression's
# ~840 MiB eigh workspace, so 3500 was rejecting GPUs that would in fact have fit.
MIN_FREE_MB=${MIN_FREE_MB:-2800}
wait_for_vram() {
    local waited=0
    while [[ "$waited" -lt 3600 ]]; do
        local free
        free=$(python -c 'import torch;print(torch.cuda.mem_get_info()[0]//1048576)' 2>/dev/null | tail -1)
        [[ "$free" =~ ^[0-9]+$ ]] && [[ "$free" -ge "$MIN_FREE_MB" ]] && return 0
        echo "[$(date '+%F %T')] $NAME: ${free:-unreadable} MiB free < $MIN_FREE_MB — waiting"
        sleep 120; waited=$(( waited + 120 ))
    done
    # TRY ANYWAY rather than giving up. Measured on the sibling grid: 4 of 96 cells
    # burned an hour and exited 75 on a GPU reporting 2318 MiB free against a 3500
    # threshold -- marginal, not hopeless. Exiting turned a maybe into a certain
    # loss AND consumed the array slot; SLURM marks exit 75 FAILED and does not
    # requeue. Falling through means a genuine shortage surfaces as a CUDA OOM,
    # which the retry loop below already handles.
    echo "[$(date '+%F %T')] $NAME: still short after 1h — trying anyway; a real"
    echo "    shortage will OOM and be retried (resubmit --array=$IDX if it fails out)"
    return 0
}

for attempt in 1 2 3; do
    wait_for_vram
    echo "[$(date '+%F %T')] $NAME: attempt $attempt"
    # ONLY the swept axes (gamma, seed) and run identity are passed. Everything
    # else -- timesteps, num_envs, ref_offset, learning rate, network -- comes from
    # the env's own yaml, which is the single source of truth. Overriding a config
    # value from a launch script silently detaches the run from the committed
    # config: the yaml then documents something the run did not do, and per-env
    # tuning (segway/cartpole need 500k, car 300k) gets flattened to one number.
    python scripts/skrl/train.py --classic --task "$ENV" --algorithm c2rl-ppo \
        --discount_factor "$GAMMA" --seed "$SEED" --ref_length "$PIN_REF_LEN" \
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

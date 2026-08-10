#!/bin/bash
# gamma x WINDOW-ARM x seed, to separate two explanations of the high-gamma
# overshoot spike that the first grid could not tell apart.
#
# THE OBSERVATION (segway, 2 seeds, reproducible):
#   gamma  auc_last/auc_min   peak(overshoot)
#   0.5          1.17               1.09
#   0.9          1.11 / 1.02        1.10 / 1.14
#   0.95         1.06               1.10
#   0.99        11.5  / 9.4         3.88 / 3.25     <-- spikes, BOTH seeds
#   0.999        1.16 / (partial)   1.27
# Non-monotonic: 0.99 is far worse than 0.999. Any pure "high gamma is unstable"
# story predicts the opposite ordering, so something else is moving.
#
# THE CONFOUND. ref_length AUTO sizes the observation window to 1/(1-gamma),
# CAPPED at the episode length (500):
#   gamma 0.9 -> 11    0.95 -> 21    0.97 -> 34
#   gamma 0.99 -> 101  0.995 -> 201  0.999 -> 500 (== FULL episode)
# So gamma=0.999 is the ONLY cell whose window covers the whole trajectory, while
# gamma=0.99 sees 101 of 500 steps. gamma and window width are not separable in
# that design.
#
# THE TEST. Same gamma grid, two arms:
#   arm 0  AUTO    ref_length = 1/(1-gamma), capped   (status quo)
#   arm 1  PINNED  ref_length = 500 for every gamma   (window held CONSTANT)
# Arm 1 holds the observation fixed and varies only the discount, which is the
# comparison the claim actually needs.
#
#   H1  the spike is a WINDOW effect: it vanishes in arm 1.
#       => "high gamma needs a horizon-sized window" -- an implementation
#          requirement, and the AUTO cap is the bug.
#   H2  the spike is a DISCOUNT effect: it survives in arm 1.
#       => a genuine gamma knee in overshoot, which is the paper claim.
# The two are distinguished by arm 1's peak at gamma=0.99 alone.
#
# Pinning ref_length is a config override, unlike every other parameter here --
# it is the TREATMENT in arm 1, not a convenience.
#
#   sbatch --array=0-95%12 --partition=eng-research-gpu --account=huytran1-ae-eng \
#          --time=18:00:00 search/run_gamma_window_grid.sbatch classic-segway-v0
#
# Index layout (96 = 6 gammas x 2 arms x 8 seeds), gamma FASTEST so any
# contiguous slice is gamma-balanced and the treatment cannot align with hardware:
#   gamma = GAMMAS[idx % G]
#   arm   = (idx / G) % 2
#   seed  = idx / (G * 2)
#
#SBATCH --job-name=crl-gwgrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/u/minjae5/contractionRL/grid_logs/%x-%A_%a.out

set -uo pipefail
REPO=/u/minjae5/contractionRL
cd "$REPO" || exit 1
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}

ENV=${1:-classic-segway-v0}
GAMMAS=(0.9 0.95 0.97 0.99 0.995 0.999)
NSEEDS=${NSEEDS:-8}
PIN_LEN=${PIN_LEN:-500}   # == max_episode_len for segway/car; the AUTO cap

G=${#GAMMAS[@]}
TOTAL=$(( G * 2 * NSEEDS ))
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
if [[ "$IDX" -ge "$TOTAL" ]]; then
    echo "idx $IDX >= grid size $TOTAL (${G} gammas x 2 arms x ${NSEEDS} seeds)"
    exit 2
fi

GAMMA=${GAMMAS[$(( IDX % G ))]}
ARM=$(( (IDX / G) % 2 ))
SEED=$(( IDX / (G * 2) ))

SHORT=${ENV#classic-}; SHORT=${SHORT%-v0}
if [[ "$ARM" -eq 0 ]]; then
    WIN=()                       # AUTO
    TAG=auto
else
    WIN=(--ref_length "$PIN_LEN")
    TAG=pin${PIN_LEN}
fi
NAME="gw_${SHORT}_${TAG}_g${GAMMA}_s${SEED}"
echo "[$(date '+%F %T')] idx=$IDX/$TOTAL -> $NAME on $(hostname), CUDA=$CUDA_VISIBLE_DEVICES"

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
    # gamma, seed and (in arm 1) ref_length are the swept axes. timesteps,
    # num_envs and everything else come from the env yaml -- see
    # run_class_gamma_grid.sbatch for why nothing else is overridden here.
    python scripts/skrl/train.py --classic --task "$ENV" --algorithm c2rl-ppo \
        --discount_factor "$GAMMA" --seed "$SEED" "${WIN[@]}" \
        --wandb_project contractionRL --wandb_run_name "$NAME" \
        --wandb_batch gamma_window
    rc=$?
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

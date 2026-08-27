#!/bin/bash
#SBATCH --job-name=crl-seeds
#SBATCH --output=cluster_runs/seeds_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#
# One (env, gamma, seed-subset) cell. Driven by ENVNAME, GAMMA and SEEDS.
#
# --parallel 1, NOT 5. Five seeds sharing one GPU was measured at 0.41 it/s per
# seed against 14.0 it/s on a dedicated GPU -- a 34x penalty, not the ~5x that
# fair sharing would cost. The window-500 encoder at num_envs=1 is latency-bound
# (a GRU walking 500/stride-3 = 167 sequentially dependent points on a batch of
# one), so concurrent processes thrash rather than interleave. That single
# setting is what timed out five of nine cells on 2026-08-22.
#
# SEED_OFFSET / SEED_COUNT are exported as plain integers and the comma list is
# built HERE, because sbatch --export is itself comma-delimited: passing
# SEEDS="6,7" through it keeps only "6", and run_seeds.sh then reads a bare
# integer as a COUNT (its line 173) and silently runs seeds 0-5 instead of 6-7.
# That is exactly what happened on the first attempt -- every job ran overlapping
# low seeds, none ever reached seeds 5-9, and the banner read "seeds=6".
#
# run_seeds.sh takes a LIST when the value contains a separator (line 176), so
# seq -s, produces precisely the subset this job owns.
#
# Sizing: ONE SEED PER JOB. The binding rate is not the 5-14 it/s measured at
# gamma=0.01 but the 1.38-1.72 it/s measured at gamma>=0.99, where PPO's
# kl_threshold early-stop never fires so all learning_epochs x mini_batches = 20
# gradient steps run every update. At 1.38 it/s, 100k steps is 20.1 h.
#   gamma 0.01        100k  ~5.6 h/seed
#   gamma 0.99/0.999  100k  ~20.1 h/seed
# Two seeds of a high-gamma cell is 40 h against a 48 h wall -- inside it only if
# the node matches the machine that rate was measured on. One seed per job is 20 h,
# a 2.4x margin that holds even on a much slower A10, and it makes each job
# independently complete: a lost job costs one seed, not a whole cell.
# See docs/cluster.md.
#
# python -u equivalent: run_seeds.sh writes its own per-run logs, but keep the
# flush habit (docs/cluster.md 1b) -- a wall-limit SIGTERM does not flush.
set -uo pipefail
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$HOME/contractionRL"
# Trailing comma, always. run_seeds.sh reads a value matching ^[0-9]+$ as a
# COUNT and anything else as a LIST (its lines 175-179), so a single-seed job
# passing "7" would silently train seeds 0-6. The comma forces the LIST branch
# for every SEED_COUNT, and `IFS=", " read -a` drops the trailing empty field,
# so "7," -> (7) and "7,8," -> (7 8).
SEEDS="$(seq -s, "$SEED_OFFSET" $((SEED_OFFSET + SEED_COUNT - 1))),"
echo "[$(date)] host=$(hostname) env=$ENVNAME gamma=$GAMMA seeds=$SEEDS"
# Guard: a mangled export must fail loudly, not silently retrain seed 0.
case "$SEEDS" in
    *,*) : ;;
    *) echo "FATAL: SEEDS='$SEEDS' has no comma; run_seeds.sh would read it as a COUNT"; exit 2 ;;
esac

# --ref_length is NOT pinned: train.py's AUTO already sets it to the env's own
# max_episode_len (see its "ref_length FULL" line), which is the full episode for
# every env and is what the Markov requirement needs -- the episodic return from t
# touches xref[t .. T-1], so the window has to reach the episode end. Pinning 500
# was correct while every env ran 500 steps; segway now runs 2000, and the pin
# would have handed it a window covering a quarter of its own episode.
#
# Crucially AUTO does not vary with gamma, so the sweep still changes only gamma
# and never the observation width -- which is what the pin was there to guarantee.
#
# --discount_factor, NOT --agent.discount_factor=. The dotted form is resolved by
# train_utils.apply_cli_dotted_overrides, which this checkout does not have, so it
# lands in argparse's leftovers and is DROPPED WITHOUT A WORD. That is not
# hypothetical: the 2026-08-21 batch ran all six cells at the yaml's pinned
# discount_factor: 0.01 and returned byte-identical per-seed AUCs for gamma 0.01
# and gamma 0.999. Both flags below exist in both checkouts.
./run_seeds.sh --detached \
    --algorithms c2rl-ppo \
    --env "$ENVNAME" \
    --seeds "$SEEDS" \
    --gpu 0 \
    --parallel 1 \
    --tag "gamma${GAMMA}_${ENVNAME}" \
    --extra "--discount_factor $GAMMA" \
    -y
rc=$?
echo "[$(date)] done rc=$rc env=$ENVNAME gamma=$GAMMA seeds=$SEEDS"
exit 0

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
# Sizing, against the PESSIMISTIC 5 it/s (the probe ran on an H200; A10 nodes are
# slower) so every job fits 48 h on the worst hardware it might land on:
#   car / cartpole  100k steps  5.6 h/seed  -> 5 seeds =  28 h
#   car_weak        300k steps 16.7 h/seed  -> 2 seeds =  33 h
# At the measured 14 it/s these are 10 h and 12 h. See docs/cluster.md.
#
# python -u equivalent: run_seeds.sh writes its own per-run logs, but keep the
# flush habit (docs/cluster.md 1b) -- a wall-limit SIGTERM does not flush.
set -uo pipefail
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$HOME/contractionRL"
SEEDS=$(seq -s, "$SEED_OFFSET" $((SEED_OFFSET + SEED_COUNT - 1)))
echo "[$(date)] host=$(hostname) env=$ENVNAME gamma=$GAMMA seeds=$SEEDS"
# Guard: a mangled export must fail loudly, not silently retrain seed 0.
case "$SEEDS" in
    *,*) : ;;
    *) echo "FATAL: SEEDS='$SEEDS' has no comma; run_seeds.sh would read it as a COUNT"; exit 2 ;;
esac

# --ref_length 500 pins the window to the FULL episode (max_episode_len is 500 for
# every classic env here: dt 0.03, time_bound 15), the same at every gamma, with
# the tail repeating the last reference point. Left unset, train.py's AUTO sizes it
# from 1/(1-gamma) -- 2 points at gamma=0.01, 101 at 0.99, 500 at 0.999 -- so the
# sweep would vary gamma AND the observation width together and could not tell a
# gamma effect from an architecture effect.
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
    --extra "--discount_factor $GAMMA --ref_length 500" \
    -y
rc=$?
echo "[$(date)] done rc=$rc env=$ENVNAME gamma=$GAMMA seeds=$SEEDS"
exit 0

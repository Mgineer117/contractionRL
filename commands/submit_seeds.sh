#!/bin/bash
# Submit the six (env, gamma) cells: 10 seeds each, one job per cell.
#
# One cell per job so a single gamma can be resubmitted without disturbing the
# other five, and so all three GPU partitions carry work at once. Partition
# assignment is round-robin over gamma, with each partition's own account.
#
# WALL TIME IS PER-ENV, measured rather than guessed. Phase B runs ~4.3 it/s with
# --parallel 5, and the two envs train for different lengths (their yamls say
# 100k and 300k), so:
#   car       100k -> 6.5 h/run  x 2 waves of 5 = ~13 h  -> 24 h is fine
#   car_weak  300k -> 19.4 h/run x 2 waves of 5 = ~39 h  -> 24 h TIMES OUT
# car_weak therefore gets 2 days, which every partition here allows (csl 7 d,
# IllinoisComputes-GPU 3 d, eng-research-gpu exactly 2 d). run_seeds.sh has no
# resume, so a wall kill loses the unfinished seeds outright.
set -uo pipefail
cd "$HOME/contractionRL" || exit 1

ONLY="${1:-}"          # optional substring filter, e.g. car_weak

i=0
for ENVNAME in classic-car-v0 classic-car_weak-v0; do
    for GAMMA in 0.01 0.99 0.999; do
        case $((i % 3)) in
            0) P=csl;                  A=csl ;;
            1) P=eng-research-gpu;     A=huytran1-ae-eng ;;
            2) P=IllinoisComputes-GPU; A=huytran1-ic ;;
        esac
        i=$((i + 1))
        case "$ENVNAME" in
            *car_weak*) T=2-00:00:00 ;;
            *)          T=1-00:00:00 ;;
        esac
        short=${ENVNAME#classic-}; short=${short%-v0}
        if [ -n "$ONLY" ] && [[ "$ENVNAME" != *"$ONLY"* ]]; then continue; fi
        sbatch --partition="$P" --account="$A" --time="$T" \
               --job-name="crl-seeds-${short}-g${GAMMA}" \
               --export=ALL,ENVNAME="$ENVNAME",GAMMA="$GAMMA" \
               seeds_gamma.sbatch
    done
done

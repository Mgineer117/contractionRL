#!/bin/bash
# Submit the six (env, gamma) cells: 10 seeds each, one job per cell.
#
# One cell per job so a single gamma can be resubmitted without disturbing the
# other five, and so all three GPU partitions carry work at once. Partition
# assignment is round-robin over gamma, with each partition's own account and its
# own wall cap (csl 7d, eng-research-gpu 2d, IllinoisComputes-GPU 3d).
set -uo pipefail
cd "$HOME/contractionRL" || exit 1

i=0
for ENVNAME in classic-car-v0 classic-car_weak-v0; do
    for GAMMA in 0.01 0.99 0.999; do
        case $((i % 3)) in
            0) P=csl;                  A=csl;              T=1-00:00:00 ;;
            1) P=eng-research-gpu;     A=huytran1-ae-eng;  T=1-00:00:00 ;;
            2) P=IllinoisComputes-GPU; A=huytran1-ic;      T=1-00:00:00 ;;
        esac
        short=${ENVNAME#classic-}; short=${short%-v0}
        sbatch --partition="$P" --account="$A" --time="$T" \
               --job-name="crl-seeds-${short}-g${GAMMA}" \
               --export=ALL,ENVNAME="$ENVNAME",GAMMA="$GAMMA" \
               seeds_gamma.sbatch
        i=$((i + 1))
    done
done

#!/bin/bash
# Launch the c2rl-ppo-cvstem sweep with the GPU chosen BY MEASUREMENT, per env.
#
# Footprint is a property of the env crossed with the densest stride the sweep
# may sample, not of the algorithm, and it varies 5x across the envs one search
# config serves. scripts/gpu_for_env.py computes it (retained points =
# episode/min_stride, ~11.8 MiB per point at num_envs 1024, 1.25x headroom) and
# returns the CHEAPEST card that fits -- an A10 whenever an A10 is enough, a
# larger card only where the window genuinely does not fit.
#
# Hand-placing is what this replaces, and it had already gone wrong twice:
# segway's 2000-step episode needs 23.6 GiB at stride 1, so it was first put on a
# 22.06 GiB A10 (a trial asked for 26.80 GiB and died), then on an L40S at two
# agents (28.8 x 2 = 57.6 GiB against 45).
set -u
cd "$(dirname "$0")/.." || exit 1
ENVS=${ENVS:-"classic-car-v0 classic-car_weak-v0 classic-segway-v0 classic-quadrotor-v0"}
JOBS=${JOBS:-3}

squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /^crl-/ {print $1}' | while read -r j; do scancel "$j"; done
sleep 8

for e in $ENVS; do
    read -r PART ACCT AGENTS <<< "$(python scripts/gpu_for_env.py --env "$e" --shell 2>/dev/null | tail -1)"
    if [ "$PART" = "NONE" ] || [ -z "${PART:-}" ]; then
        echo "!! $e does not fit any card at the densest stride — skipping" >&2
        continue
    fi
    echo "=== $e -> $PART ($AGENTS agents/GPU)"
    ./search/search_cluster.sh --algorithm c2rl-ppo-cvstem --env "$e" \
        --project contractionRL-Search --partition "$PART" --account "$ACCT" \
        --num-jobs "$JOBS" --gpus-per-job 1 --time 2-00:00:00 \
        --agents-per-gpu "$AGENTS" --no-probe -y 2>&1 \
      | grep -E "→ sweep|Submitted batch job|✗" | head -6
done
echo "@@@@ LAUNCH DONE"

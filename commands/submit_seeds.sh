#!/bin/bash
# Submit the gamma batch: 4 envs x 3 gammas x 10 seeds = 120 jobs, ONE SEED EACH.
#
# Env order is the requested one -- cartpole, segway, car-v0, car-v1.
#
# ONE SEED PER JOB (sizing lives in seeds_gamma.sbatch). A high-gamma seed is
# ~20 h of the 48 h wall, so each job keeps a 2.4x margin and a lost job costs one
# seed instead of a whole cell. 120 jobs is also 120 independently schedulable
# units, which fills three partitions far better than 12 fat cells could.
#
# PREFLIGHT + DEPENDENCY. Every env is first checked against its own CM dataset
# with scripts/preflight_cm_data.py. An env that fails does NOT get skipped and
# does NOT block the others: this submits a build_cm job for it and chains its 30
# training jobs behind that build with --dependency=afterok. So the whole batch is
# launched in one pass and SLURM does the waiting, instead of a human remembering
# to come back in 36 h.
#
# That check is the whole reason this script exists in this form. The 2026-08-24
# batch lost 22 runs because data/classic/car_v1/cm_data_*.npz was absent on the
# cluster after the car_weak rename, and would have lost 30 more to cartpole's
# dataset having been solved at N=100000 while its yaml asks for 10000 -- both
# invisible until each job had already registered a W&B run and died.
#
# Partitions round-robin over the whole (env, gamma, seed) sequence rather than
# over gamma, so no partition inherits a whole env or a whole gamma: an outage
# then degrades every cell a little instead of destroying three cells. All three
# allow at least 2 days (csl 7 d, IllinoisComputes-GPU 3 d, eng-research-gpu
# exactly 2 d), so the wall is a uniform 2-00:00:00 per the standing 48 h rule.
set -uo pipefail
cd "$HOME/contractionRL" || exit 1

ONLY="${1:-}"                    # optional substring filter on the env id
DRY="${DRY:-0}"                  # DRY=1 prints the sbatch lines without submitting
SEEDS_TOTAL="${SEEDS_TOTAL:-10}"
GAMMAS="${GAMMAS:-0.01 0.99 0.999}"
mkdir -p cluster_runs

_sbatch_id() {  # echo the job id, or empty on failure
    local out
    out=$("$@" 2>&1) || { echo "  sbatch FAILED: $out" >&2; return 1; }
    echo "$out" | grep -oE '[0-9]+$'
}

i=0
n=0
for ENVNAME in classic-cartpole-v0 classic-segway-v0 classic-car-v0 classic-car-v1; do
    if [ -n "$ONLY" ] && [[ "$ENVNAME" != *"$ONLY"* ]]; then continue; fi
    short=${ENVNAME#classic-}

    # Does this env's configured CM dataset exist and match its yaml?
    DEP=""
    if python scripts/preflight_cm_data.py "$ENVNAME" >/dev/null 2>&1; then
        echo "== $ENVNAME: CM dataset ready"
    else
        python scripts/preflight_cm_data.py "$ENVNAME" 2>&1 | grep -E '^\[' || true
        if [ "$DRY" = 1 ]; then
            echo "== $ENVNAME: would submit build_cm + chain 30 jobs on it"
            DEP="afterok:BUILDID"
        else
            BID=$(_sbatch_id sbatch \
                --partition=IllinoisComputes-GPU --account=huytran1-ic \
                --time=2-00:00:00 --job-name="crl-buildcm-${short}" \
                --export=ALL,ENVNAME="$ENVNAME" \
                commands/build_cm.sbatch) || { echo "== $ENVNAME: build submit failed, SKIPPING" >&2; continue; }
            echo "== $ENVNAME: CM dataset missing -> build job $BID, chaining 30 jobs on it"
            DEP="afterok:$BID"
        fi
    fi

    for GAMMA in $GAMMAS; do
        for SEED in $(seq 0 $((SEEDS_TOTAL - 1))); do
            case $((i % 3)) in
                0) P=csl;                  A=csl ;;
                1) P=eng-research-gpu;     A=huytran1-ae-eng ;;
                2) P=IllinoisComputes-GPU; A=huytran1-ic ;;
            esac
            i=$((i + 1)); n=$((n + 1))
            CMD=(sbatch
                 --partition="$P" --account="$A" --time=2-00:00:00
                 --job-name="crl-${short}-g${GAMMA}-s${SEED}"
                 --export=ALL,ENVNAME="$ENVNAME",GAMMA="$GAMMA",SEED_OFFSET="$SEED",SEED_COUNT=1)
            [ -n "$DEP" ] && CMD+=(--dependency="$DEP")
            CMD+=(commands/seeds_gamma.sbatch)
            if [ "$DRY" = 1 ]; then printf '%s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
        done
    done
done
echo "submitted=$n training jobs (dry=$DRY)"

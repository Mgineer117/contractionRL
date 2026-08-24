#!/bin/bash
# Free cluster storage before a relaunch: drop training logs and job stdout.
#
# The Campus Cluster limit that actually bites is the INODE quota, not bytes, and
# a 120-job batch is the wrong time to discover it -- a job that cannot create its
# run directory dies after wandb has already registered the run, which reads as a
# crash rather than as a full disk. `quota` output is printed before and after so
# the reclaim is a measurement, not a hope.
#
# DELETES: logs/ (per-run dirs, tensorboard, seed nohup logs), cluster_runs/
# (sbatch stdout), and stray wandb/ scratch.
# KEEPS: data/ (the CM datasets -- hours of MOSEK each, and a missing one is
# exactly what broke the last batch), results/ (aggregated CSVs), and the repo.
#
# Also removes the quarantine dirs from the aggregation double-counting incidents
# (~/crl_quarantine, logs/_stale_prefix_*): they exist only to be excluded from a
# glob, and nothing reads them.
set -uo pipefail
cd "$HOME/contractionRL" || exit 1

DRY="${DRY:-0}"
echo "=== quota before ==="; quota 2>/dev/null || echo "(quota unavailable)"

_du() { du -sh --  "$1" 2>/dev/null | cut -f1; }
_n()  { find "$1" -xdev 2>/dev/null | wc -l; }

for T in logs cluster_runs wandb "$HOME/crl_quarantine"; do
    [ -e "$T" ] || { echo "-- $T: absent"; continue; }
    echo "-- $T: $(_du "$T") in $(_n "$T") inodes"
    if [ "$DRY" = 1 ]; then echo "   DRY: would rm -rf $T"; else rm -rf -- "$T"; fi
done

# Recreate the two the launcher writes into, so a job never fails on mkdir.
mkdir -p logs cluster_runs
echo "=== quota after ==="; quota 2>/dev/null || echo "(quota unavailable)"
echo "kept: $(_du data) in data/, $(_du results 2>/dev/null) in results/"

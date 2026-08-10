"""Backfill env/gamma/seed onto the class_gamma runs.

The grid launched before WANDB_RUN_GROUP was set, so those runs carry no env
field at all -- the only identifier is the name string, and substring matching
on it is wrong ("car" matches cartpole and car_weak too). This adds the grid
coordinates as real config keys plus a group, so plots can split on them.

Additive and idempotent: nothing existing is overwritten.
"""
import re
import sys

import wandb

DRY = "--apply" not in sys.argv
PAT = re.compile(r"^c2rl_(?P<env>.+)_g(?P<gamma>[0-9.]+)_s(?P<seed>\d+)$")

api = wandb.Api()
runs = list(api.runs("contractionRL", filters={"config.run_batch": "class_gamma"},
                     per_page=500))
print(f"{len(runs)} runs in batch class_gamma  (dry-run={DRY})")

changed = skipped = 0
for r in runs:
    m = PAT.match(r.name)
    if not m:
        print(f"  SKIP unparseable name: {r.name}")
        skipped += 1
        continue
    env, gamma, seed = m["env"], float(m["gamma"]), int(m["seed"])
    # Sanity: the name's gamma must agree with what the agent actually ran.
    logged = r.config.get("discount_factor")
    if logged is not None and abs(float(logged) - gamma) > 1e-9:
        print(f"  MISMATCH {r.name}: name says {gamma}, config says {logged} -- skipped")
        skipped += 1
        continue
    if DRY:
        print(f"  {r.name:30s} -> env={env:10s} gamma={gamma:<6} seed={seed}")
    else:
        r.config.update({"env": env, "gamma": gamma, "seed_idx": seed})
        r.group = env
        r.update()
    changed += 1

print(f"{'would update' if DRY else 'updated'} {changed}, skipped {skipped}")
if DRY:
    print("re-run with --apply to write")

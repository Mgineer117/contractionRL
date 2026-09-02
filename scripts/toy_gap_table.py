"""Collect every C2RL run's |V^pi - V*| into one table, keyed by what varied.

``train.py`` writes ``optimality_gap.json`` next to each run's checkpoints, but
the json says nothing about which env, reference mode or discount produced it --
so a directory full of them is unreadable. Each run's ``params/agent.yaml`` (or
its stdout log) does say, and this joins the two.

    python scripts/toy_gap_table.py --root logs/classic/c2rl-ppo
    python scripts/toy_gap_table.py --logs "$CLAUDE_JOB_DIR/tmp/tr_*.log"
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

# The tag train_*.log carries: <env>_<mode>_g<gamma>_s<seed>
TAG = re.compile(r"tr_(?P<env>\w+?)_(?P<mode>stabilizing|contractive)_"
                 r"g(?P<gamma>[\d.]+)_s(?P<seed>\d+)\.log$")
LINE = re.compile(r"([\d.eE+-]+)% of \|V\*\|")


def from_logs(pattern: str):
    rows = []
    for f in sorted(glob.glob(pattern)):
        m = TAG.search(f)
        if not m:
            continue
        txt = pathlib.Path(f).read_text(errors="ignore")
        hit = [ln for ln in txt.splitlines() if "V^pi - V*" in ln]
        if not hit:
            rows.append({**m.groupdict(), "pct": None, "line": None})
            continue
        pct = LINE.search(hit[-1])
        rows.append({**m.groupdict(), "pct": float(pct.group(1)) if pct else None,
                     "line": hit[-1]})
    return rows


def from_runs(root: str):
    rows = []
    for j in sorted(glob.glob(f"{root}/**/optimality_gap.json", recursive=True)):
        d = json.loads(pathlib.Path(j).read_text())
        rows.append({"env": d.get("task", "?"), "mode": d.get("reference_mode", "?"),
                     "gamma": d.get("gamma", "?"), "seed": d.get("seed", "?"),
                     "pct": d.get("pct_of_vstar"), "line": j})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default=None, help="glob of train stdout logs")
    p.add_argument("--root", default="logs/classic/c2rl-ppo")
    a = p.parse_args()

    rows = from_logs(a.logs) if a.logs else from_runs(a.root)
    if not rows:
        print("[gap] nothing found.")
        return 1

    by = defaultdict(list)
    for r in rows:
        if r["pct"] is not None:
            by[(r["env"], r["mode"], r["gamma"])].append(r["pct"])
    missing = [r for r in rows if r["pct"] is None]

    print("\n  env   reference     gamma    seeds   |V^pi - V*| as % of |V*|")
    print("  " + "-" * 62)
    for k in sorted(by):
        v = np.array(by[k])
        print(f"  {k[0]:<5} {k[1]:<12} {k[2]:<8} {len(v):<7d} "
              f"{v.mean():.3f} +/- {v.std():.3f}   (min {v.min():.3f}, max {v.max():.3f})")
    if missing:
        names = sorted(f"{m['env']}/{m['mode']}/g{m['gamma']}/s{m['seed']}"
                       for m in missing)
        print(f"\n  {len(missing)} run(s) with no gap line yet: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

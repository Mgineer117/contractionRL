"""Write one certified (lbd, r, w_lb, w_ub) into every yaml for an env.

An env's algorithms spell the same quantities differently::

    skrl_c2rl_{ppo,sac}_cfg.yaml   cm.lbd  cm.w_lb    cm.w_ub    cm.cvstem_r_scaler
    skrl_c3m_cfg.yaml              cm.lbd  cm.w_lb    cm.w_ub    (no Riccati weight)
    skrl_cvstem_lqr_cfg.yaml       cm.lbd  cm.cm_w_lb cm.cm_w_ub agent.r_scaler

so updating "the config" after a lambda search updates one algorithm and leaves
the others certifying a rate nobody solved for. That is how segway came to ship
lbd=0.0514 under c2rl and 0.0152 under cvstem-lqr, and tora 0.3902/r=3.2 against
0.1171/r=0.1. ``tests/test_sdp_config_wiring.py`` fails on exactly that; this is
how you make it pass.

Edits are line-surgical (regex on the assignment, in place). These yamls carry
paragraphs of load-bearing commentary and a pyyaml round-trip would delete all of
it, so nothing here parses-and-dumps.

    python scripts/set_certified_lambda.py --env segway --lbd 0.0514 --r 6.4
    python scripts/set_certified_lambda.py --env segway --lbd 0.0514 --r 6.4 --apply

Dry-run by default: it prints the diff and changes nothing without ``--apply``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIC = os.path.join(ROOT, "source/contractionRL/contractionRL/tasks/direct/classic")

# algorithm -> {canonical name: yaml key}. A key absent here is simply not
# written for that algorithm (c3m has no Riccati weight at all).
SPELLINGS = {
    "c2rl_ppo":   {"lbd": "lbd", "w_lb": "w_lb", "w_ub": "w_ub", "r": "cvstem_r_scaler"},
    "c2rl_sac":   {"lbd": "lbd", "w_lb": "w_lb", "w_ub": "w_ub", "r": "cvstem_r_scaler"},
    "c3m":        {"lbd": "lbd", "w_lb": "w_lb", "w_ub": "w_ub"},
    "cvstem_lqr": {"lbd": "lbd", "w_lb": "cm_w_lb", "w_ub": "cm_w_ub", "r": "r_scaler"},
}


def _set(text: str, key: str, value: str) -> tuple[str, str | None]:
    """Replace ``  key: old`` with ``  key: value``, keeping indent and comment."""
    pat = re.compile(rf"^(\s*{re.escape(key)}:[ \t]*)([^\s#]+)([ \t]*(?:#.*)?)$", re.M)
    m = pat.search(text)
    if m is None:
        return text, None
    # Numeric compare, not textual: 1000.0 and 1000 are the same envelope, and
    # rewriting one as the other is a diff that says a value changed when none did.
    try:
        if float(m.group(2)) == float(value):
            return text, m.group(2)
    except ValueError:
        if m.group(2) == value:
            return text, m.group(2)
    return pat.sub(lambda mm: f"{mm.group(1)}{value}{mm.group(3)}", text, count=1), m.group(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="short name, e.g. segway")
    p.add_argument("--lbd", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--w-lb", "--w_lb", type=float, default=None)
    p.add_argument("--w-ub", "--w_ub", type=float, default=None)
    p.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = p.parse_args()

    vals = {"lbd": f"{args.lbd:.4f}", "r": f"{args.r:g}"}
    if args.w_lb is not None:
        vals["w_lb"] = f"{args.w_lb:g}"
    if args.w_ub is not None:
        vals["w_ub"] = f"{args.w_ub:g}"

    base = os.path.join(CLASSIC, args.env, "agents")
    if not os.path.isdir(base):
        print(f"no such env: {base}", file=sys.stderr)
        return 2
    touched = 0
    for algo, spell in SPELLINGS.items():
        path = os.path.join(base, f"skrl_{algo}_cfg.yaml")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            text = orig = fh.read()
        for canon, new in vals.items():
            key = spell.get(canon)
            if key is None:
                continue
            text, old = _set(text, key, new)
            if old is None:
                print(f"  {algo:11s} {key:16s} ABSENT — not added "
                      f"(a key this config never had is not one it silently wants)")
            elif float(old) != float(new):
                print(f"  {algo:11s} {key:16s} {old} -> {new}")
        if text != orig:
            touched += 1
            if args.apply:
                with open(path, "w") as fh:
                    fh.write(text)
    if not args.apply:
        print(f"\ndry run — {touched} file(s) would change. Re-run with --apply.")
    else:
        print(f"\nwrote {touched} file(s). Now:\n"
              f"  python -m pytest tests/test_sdp_config_wiring.py -q -k {args.env}\n"
              f"  python scripts/build_cm_dataset.py --task classic-{args.env}-v0 "
              f"--algorithm c2rl-ppo   # lbd is in the cache key\n"
              f"  python scripts/verify_cm_dataset.py --task classic-{args.env}-v0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

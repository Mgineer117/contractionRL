"""One table: what every env is, and which of the theory's assumptions it can test.

The claim the contraction theorems make is about an OPTIMAL policy on a plant
with a state-dependent rate. Whether a given env can be used to test that claim
depends on properties scattered across env.py files, CM datasets and SOS
certificates, so this collects them in one place rather than in prose that goes
stale.

Columns, and why each is here:

  x/u          dimension. Also decides whether exhaustive value iteration is
               tractable: 2 states is, 4 is not (65^4 = 18M cells x actions).
  poly         f and B polynomial in x. Prerequisite for the SOS certificate --
               nothing else about the plant changes.
  exact W      an SOS certificate exists in data/toy/<env>/sos_cm.npz, i.e. the
               contraction condition holds as an algebraic identity over the
               WHOLE box. Classic envs certify by sampling instead: their lambda
               provably holds at the CM dataset's N draws and is interpolated
               between them.
  B(x)         B actually depends on x. Where it does not, all state-dependence
               lives in A(x), which is the cleaner experiment.
  lam spread   max/min of the local closed-loop rate over the box. This is what
               "state-dependent contraction rate" MEANS operationally, and the
               check mark is >= 1.5x. A flat field (car at 1.00x) is a valid
               control, not a defect -- it is the class-II baseline.
  global V*    can V* be computed with no local minima, i.e. by exhaustive value
               iteration? Only the 2-state toy envs. This is the one column that
               decides whether J^pi - J* is measurable at all: every other method
               (multi-start NLP, MPC lookahead) returns a FEASIBLE trajectory, so
               it upper-bounds J* and can only ever prove a policy is bad.

    python scripts/env_taxonomy.py
    python scripts/env_taxonomy.py --markdown
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))
sys.path.insert(0, str(REPO / "scripts"))

import contractionRL.tasks.direct.classic as classic  # noqa: E402
import contractionRL.tasks.direct.toy as toy  # noqa: E402

TOY = toy.TOY_ENVS

# >= this max/min ratio in lam(x) counts as a state-dependent rate. 1.5x is well
# clear of the ~1.0x a genuinely uniform plant returns (car measures 1.0000) and
# well under the smallest real spread (cartpole 2.22x).
SPREAD_MARK = 1.5

CLASSIC = ("car", "car_v1", "cartpole", "segway", "quadrotor",
           "ball_and_beam", "two_link_arm", "aircraft", "tora")


def _make(name):
    eid = toy.env_id(name) if name in TOY else classic.env_id(name)
    return gym.make(eid, num_envs=2, device="cpu").unwrapped


def _b_depends_on_x(env, n=256, tol=1e-6):
    lo, hi = env.X_MIN.numpy(), env.X_MAX.numpy()
    pts = torch.as_tensor(np.random.default_rng(0).uniform(lo, hi, (n, len(lo))),
                          dtype=torch.float32)
    with torch.no_grad():
        _, B, _ = env.get_f_and_B(pts, need_null=False)
    return float((B - B[:1]).abs().max()) > tol


def _rate_spread(env, name, is_toy):
    """max/min of lam(x) over the shipped metric.

    One path for both families: SOS writes the same ``{x -> W*(x)}`` npz the
    CV-STEM SDP does, so the only difference is which directory it lands in and
    how the guarantee was obtained -- not how the rate is read back.
    """
    from contractionRL import cm_data
    from find_x_init import local_rates
    try:
        d = np.load(cm_data.find_npz(name), allow_pickle=True)
    except FileNotFoundError:
        return None, None
    n = min(2000, len(d["x"]))
    lam = local_rates(env, d["x"][:n], d["W"][:n], float(d["r_scaler"]))
    lam = lam[lam > 0]
    if lam.size == 0:
        return None, None
    return float(lam.max() / lam.min()), float(lam.min())



def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--markdown", action="store_true", help="emit a markdown table")
    p.add_argument("--envs", default=",".join(TOY + CLASSIC))
    a = p.parse_args()

    yes, no = ("yes", "-") if not a.markdown else ("✅", "—")
    rows = []
    for name in [e.strip() for e in a.envs.split(",") if e.strip()]:
        try:
            env = _make(name)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ! {name}: {e}", file=sys.stderr)
            continue
        is_toy = name in TOY
        spread, lam = _rate_spread(env, name, is_toy)
        src = ("SOS (exact)" if is_toy else "CM dataset (sampled)") \
            if spread is not None else "none"
        T = int(round(float(env.time_bound) / float(env.dt)))
        rows.append({
            "env": name,
            "family": "toy" if is_toy else "classic",
            "x": int(env.num_dim_x),
            "u": int(env.num_dim_control),
            "T": T,
            "poly": yes if is_toy else no,
            "exactW": yes if (is_toy and lam is not None) else no,
            "Bx": yes if _b_depends_on_x(env) else no,
            "spread": spread,
            "sd": (yes if (spread is not None and spread >= SPREAD_MARK) else
                   (no if spread is not None else "?")),
            "lam": lam,
            "globalV": yes if int(env.num_dim_x) <= 2 else no,
            "src": src,
        })

    hdr = ["env", "fam", "x", "u", "T", "poly", "exact W", "B(x)",
           "lam", "spread", "state-dep", "global V*", "certificate"]
    def fmt(r):
        return [r["env"], r["family"], str(r["x"]), str(r["u"]), str(r["T"]),
                r["poly"], r["exactW"], r["Bx"],
                f"{r['lam']:.4f}" if r["lam"] is not None else "?",
                f"{r['spread']:.2f}x" if r["spread"] is not None else "?",
                r["sd"], r["globalV"], r["src"]]

    table = [hdr] + [fmt(r) for r in rows]
    w = [max(len(t[c]) for t in table) for c in range(len(hdr))]
    if a.markdown:
        print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
        print("|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|")
        for t in table[1:]:
            print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(t)) + " |")
    else:
        for k, t in enumerate(table):
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(t)))
            if k == 0:
                print("  ".join("-" * w[i] for i in range(len(hdr))))
    print(f"\nstate-dep marks a lam(x) spread >= {SPREAD_MARK}x over the box. "
          f"'exact W' = SOS identity over the whole box; classic envs certify at "
          f"N sampled states instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How the CERTIFIED optimal control changes with the discount, region by region.

The toy plants have a state-dependent contraction rate: ``lambda(x)`` varies by
5.6x on mg. ``reference_mode="contractive"`` walks the reference across that
variation on purpose, so one episode visits both a slow region and a fast one.
This script asks what the optimum does differently in each.

The prediction worth testing is that a myopic optimum cannot use contraction at
all. At ``gamma = 0.01`` only the next step is worth anything, so the optimal
control has to cancel the error NOW wherever it is; at ``gamma = 0.99`` the
optimum can leave an error alone in a region where the plant's own dynamics will
contract it, and spend its effort where they will not. If that is right, the
high-gamma control effort should fall as ``lambda`` rises and the low-gamma one
should not care.

Both controls are read from the moment-SOS packs (u*, certified by the LB/UB
bracket), rolled through the real env, and binned by the local rate at the state
they were applied at.

    python scripts/toy_gamma_controls.py --task toy-mg-v0
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np
import sympy as sp
import torch
from scipy.linalg import eigh

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))


def _lam_fn(key: str):
    """lambda(x) from the deployed SOS metric -- the same estimator as
    scripts/toy_rate_field.py, so the bins here and the field there agree."""
    from contractionRL.solvers import sos_cm as S

    d = np.load(sorted(glob.glob(str(REPO / "data/toy" / key / "cm_data_*.npz")))[-1])
    co = {str(a): float(b) for a, b in zip(d["sos_coeff_names"], d["sos_coeff_values"])}
    W, ws = S._w_poly(int(d["sos_w_degree"]))
    Wn = W.subs({s: co[str(s)] for s in ws})
    Gf = sp.lambdify(S.XS, S._lmi(key, 0, Wn, float(d["r_scaler"])), "numpy")
    Wf = sp.lambdify(S.XS, Wn, "numpy")
    n = int(np.asarray(Wf(0.0, 0.0)).shape[0])

    def lam(x):
        out = np.empty(len(x))
        for i, xi in enumerate(x):
            Wv = np.asarray(Wf(*xi), float).reshape(n, n)
            Gv = np.asarray(Gf(*xi), float).reshape(n, n)
            out[i] = -0.5 * eigh(0.5 * (Gv + Gv.T), Wv, eigvals_only=True).max()
        return out
    return lam


def rollout(task: str, pack, ref_mode: str):
    """(lambda, |u - uref|, |e|) at every (task, step) the optimum visits."""
    import contractionRL.tasks.direct.toy  # noqa: F401
    import gymnasium as gym
    from contractionRL import cm_data

    key = task.removeprefix("toy-").removesuffix("-v0")
    us = np.asarray(pack["u_star"]).reshape(len(pack["u_star"]), -1, 1)
    env = gym.make(task, num_envs=us.shape[0], device="cpu",
                   reference_mode=ref_mode).unwrapped
    env.configure_ref_window(length=env.full_ref_length(int(pack["ref_offset"])),
                             offset=int(pack["ref_offset"]), gamma=float(pack["gamma"]))
    cm_data.attach_metric(env, key, device="cpu", tag="[gctl]")
    env.set_group_references(pack["xref"], pack["uref"], pack["x0"])
    env.reset()
    lam_of = _lam_fn(key)
    xs, du, err = [], [], []
    for t in range(us.shape[1]):
        x = env.x_t.clone().numpy()
        u = torch.as_tensor(us[:, t], dtype=torch.float32)
        # The env applies u directly; uref is what the CONTROLLER would add back,
        # so u - uref is the feedback part -- the effort the optimum chose to
        # spend beyond just riding the reference.
        ur = env.uref[torch.arange(env.num_envs), env.time_steps].clone()
        xs.append(x)
        du.append((u - ur).abs().sum(-1).numpy())
        err.append(np.linalg.norm(x - env.xref[torch.arange(env.num_envs),
                                               env.time_steps].numpy(), axis=-1))
        env.step(u)
    xs = np.concatenate(xs)
    return lam_of(xs), np.concatenate(du), np.concatenate(err)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="toy-mg-v0")
    p.add_argument("--reference-mode", "--reference_mode", dest="ref_mode",
                   default="contractive", choices=["stabilizing", "contractive"])
    p.add_argument("--gammas", nargs="+", type=float, default=[0.01, 0.99])
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--steps", type=int, default=None,
                   help="compare only the first N steps, so different horizons "
                        "are read over the same window (default: each in full)")
    a = p.parse_args()
    key = a.task.removeprefix("toy-").removesuffix("-v0")
    root = REPO / "data/toy" / key
    stem = "global" if a.ref_mode == "stabilizing" else f"global_{a.ref_mode}"

    got = {}
    for g in a.gammas:
        cand = [root / f"{stem}_g{g:g}.npz", root / f"{stem}.npz"]
        path = next((c for c in cand if c.exists()
                     and abs(float(np.load(c)["gamma"]) - g) < 1e-12), None)
        if path is None:
            print(f"[gctl] no {a.ref_mode} pack at gamma={g}; solve it:\n"
                  f"    python scripts/precompute_global.py --task {a.task} "
                  f"--reference-mode {a.ref_mode} --gamma {g}")
            continue
        got[g] = (path, np.load(path))

    if len(got) < 2:
        print("[gctl] need at least two discounts to compare.")
        return 1

    # One bin edge set for every gamma, from the pooled rates, or the rows are
    # not comparable across the table.
    runs = {g: rollout(a.task, pk, a.ref_mode) for g, (_, pk) in got.items()}
    if a.steps:
        n_t = len(next(iter(got.values()))[1]["x0"])
        runs = {g: tuple(v.reshape(-1, n_t)[:a.steps].ravel() for v in vals)
                for g, vals in runs.items()}
    edges = np.quantile(np.concatenate([v[0] for v in runs.values()]),
                        np.linspace(0, 1, a.bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    print(f"\n[gctl] {key} / {a.ref_mode} reference: certified optimal control by "
          f"local contraction rate\n"
          f"       bins are quantiles of lambda(x) pooled over every gamma, so a "
          f"row is the same region in each column.\n")
    hdr = "  lambda bin          n   " + "".join(f"|u-uref| g={g:<6g} <e> g={g:<8g}"
                                                 for g in runs)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in range(a.bins):
        cells, n0 = "", 0
        for g, (lam, du, err) in runs.items():
            m = (lam > edges[b]) & (lam <= edges[b + 1])
            n0 = max(n0, int(m.sum()))
            cells += (f"{du[m].mean():>15.4f}{err[m].mean():>16.4f}"
                      if m.any() else f"{'-':>15}{'-':>16}")
        print(f"  [{edges[b]:.3f}, {edges[b + 1]:.3f}] {n0:>7d}   {cells}")

    print()
    for g, (lam, du, err) in runs.items():
        lo = lam <= edges[1]
        hi = lam > edges[-2]
        if lo.any() and hi.any():
            print(f"  gamma={g:<5g} effort slowest-bin -> fastest-bin: "
                  f"{du[lo].mean():.4f} -> {du[hi].mean():.4f} "
                  f"({du[hi].mean() / max(du[lo].mean(), 1e-12):.3f}x), "
                  f"H={len(du) // len(next(iter(got.values()))[1]['x0'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

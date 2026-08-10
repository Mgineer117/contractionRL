"""Exact per-state ``λ*(x)`` on a 2-D grid over each ACTIVE state pair.

Answers "where in the box does the certified rate vary, and along which
coordinates" -- the spatial version of the scalar spread reported in
``docs/dynamics_taxonomy.md`` §2.1. A class-II plant comes out flat to solver
noise; a class-III plant shows structure, and the structure names the states that
bind the certificate.

``λ*(x) = sup{λ : P({x}) feasible}`` by bisection on the one-sample joint SDP
(``ncm_synthesis.cvstem_joint``), which is legitimate per-state because
Proposition 3 decouples the program: ``λ*(S) = min_k λ*(x_k)``.

Dimensions the dynamics does not depend on are dropped ("inert"): for the car,
``f`` and ``B`` are independent of ``(pos_x, pos_y)``, so a grid over them would
be constant by construction and waste SDPs. Inertness is MEASURED, not assumed --
each dimension is swept while the others are held, and one whose Jacobians never
move is dropped.

    python scripts/pairs_lambda_map.py --task classic-car_weak-v0
    python scripts/pairs_lambda_map.py --task classic-car-v0 --n-grid 15

Writes ``figures/pairs_<env>.png`` and ``figures/data/pairs_<env>.npz``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from contractionRL.agents.skrl.ncm_synthesis import cvstem_joint, drift_jacobians

ROOT = Path(__file__).resolve().parents[1]


def active_dims(env, x_mid, x_min, x_max, *, tol=1e-9, n_probe=5):
    """Which state dimensions the Jacobians actually depend on.

    Sweeps one dimension at a time across its box with the rest held at the
    midpoint, and calls a dimension INERT when neither A nor B moves. Cheap
    (n_probe forward evaluations per dim) and exact for the plants here.
    """
    def jac(x):
        A, B = drift_jacobians(env.get_f_and_B, x[None, :])
        return A[0], B[0]

    A0, B0 = jac(x_mid)
    act = []
    for d in range(len(x_mid)):
        moved = False
        for v in np.linspace(x_min[d], x_max[d], n_probe):
            x = x_mid.copy()
            x[d] = v
            A, B = jac(x)
            if np.abs(A - A0).max() > tol or np.abs(B - B0).max() > tol:
                moved = True
                break
        if moved:
            act.append(d)
    return act


def lam_star(A, B, *, eps, dt, r, w_lb, w_ub, lo=1e-3, hi=20.0, tol=5e-3):
    """sup{λ : the one-sample program is feasible}, by bisection.

    Feasibility is downward-closed in λ (Corollary 1), so bisection is valid.
    Returns 0.0 when even ``lo`` is infeasible.
    """
    def feas(lbd):
        return cvstem_joint(A, B, lbd=lbd, eps=eps, dt=dt, r_scaler=r,
                            w_lb=w_lb, w_ub=w_ub) is not None

    if not feas(lo):
        return 0.0
    if feas(hi):
        return hi
    while hi - lo > tol * max(lo, 1.0):
        mid = 0.5 * (lo + hi)
        if feas(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="classic-car_weak-v0 or car_weak")
    p.add_argument("--n-grid", type=int, default=15)
    # No --lbd: lambda is not an INPUT here. lambda*(x) is the sup over lambda,
    # found by bisection, so quoting a fixed lambda on this figure (as the older
    # hand-made pairs_*.png did) invites reading it as the rate being tested.
    p.add_argument("--r", type=float, default=1.6)
    p.add_argument("--w-lb", type=float, default=1e-3)
    p.add_argument("--w-ub", type=float, default=1e3)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--outdir", default=str(ROOT / "figures"))
    args = p.parse_args()

    import gymnasium as gym
    import matplotlib
    matplotlib.use("Agg")
    import contractionRL.tasks.direct.classic  # noqa: F401
    import matplotlib.pyplot as plt

    task = args.task if args.task.startswith("classic-") else f"classic-{args.task}-v0"
    short = task.removeprefix("classic-").removesuffix("-v0")
    env = gym.make(task, num_envs=1, device="cpu").unwrapped

    x_min = env.X_MIN.cpu().numpy().astype(float)
    x_max = env.X_MAX.cpu().numpy().astype(float)
    names = np.array(list(env.state_names) or [f"x{i}" for i in range(len(x_min))])
    x_mid = 0.5 * (x_min + x_max)

    act = active_dims(env, x_mid, x_min, x_max)
    inert = [i for i in range(len(x_min)) if i not in act]
    print(f"[pairs] {short}: active dims {[names[i] for i in act]}, "
          f"inert (dropped) {[names[i] for i in inert]}")

    pairs = [(act[i], act[j]) for i in range(len(act)) for j in range(i + 1, len(act))]
    if not pairs:
        raise SystemExit(f"{short}: fewer than 2 active dims — nothing to pair")

    n = args.n_grid
    Z = np.zeros((len(pairs), n, n))
    gx = np.zeros((len(pairs), n))
    gy = np.zeros((len(pairs), n))
    t0 = time.time()
    for k, (dx, dy) in enumerate(pairs):
        gx[k] = np.linspace(x_min[dx], x_max[dx], n)
        gy[k] = np.linspace(x_min[dy], x_max[dy], n)
        for i, yv in enumerate(gy[k]):
            for j, xv in enumerate(gx[k]):
                x = x_mid.copy()
                x[dx], x[dy] = xv, yv
                A, B = drift_jacobians(env.get_f_and_B, x[None, :])
                Z[k, i, j] = lam_star(A, B, eps=args.eps, dt=args.dt, r=args.r,
                                      w_lb=args.w_lb, w_ub=args.w_ub)
            print(f"  pair {names[dx]}x{names[dy]}  row {i + 1}/{n}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)

    spread = np.array([z.max() / z.min() if z.min() > 0 else np.inf for z in Z])
    print(f"[pairs] lambda* spread per pair: "
          f"{ {f'{names[a]}x{names[b]}': round(float(s), 6) for (a, b), s in zip(pairs, spread)} }")

    outdir = Path(args.outdir)
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "data" / f"pairs_{short}.npz",
             Z=Z, pair_dims=np.array(pairs), grid_x=gx, grid_y=gy,
             held_spread=spread, state_names=names,
             active_dims=np.array(act), inert_names=names[inert],
             x_min=x_min, x_max=x_max, r=args.r,
             w_lb=args.w_lb, w_ub=args.w_ub, eps=args.eps, dt=args.dt,
             n_grid=n, n_draws=0)

    ncol = len(pairs)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol + 1.4, 4.2), squeeze=False)
    for k, (dx, dy) in enumerate(pairs):
        ax = axes[0][k]
        im = ax.pcolormesh(gx[k], gy[k], Z[k], shading="nearest", cmap="RdBu_r")
        ax.set_xlabel(names[dx])
        ax.set_ylabel(names[dy])
        ax.set_title(f"exact $\\lambda^*$   spread {spread[k]:.4g}")
        fig.colorbar(im, ax=ax)
    # Two short lines, not one long one: a single-panel figure is ~5.6 in wide and
    # the one-line form was clipped at both ends.
    fig.suptitle(f"{short}\n"
                 f"r={args.r}, $w_{{lb}}$={args.w_lb}, $w_{{ub}}$={args.w_ub}, "
                 f"$\\epsilon$={args.eps}   "
                 f"inert: {', '.join(names[inert]) or 'none'}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = outdir / f"pairs_{short}.png"
    fig.savefig(out, dpi=140)
    print(f"[pairs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

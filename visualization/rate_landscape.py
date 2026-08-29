"""2D slice of the certified contraction rate, with the spawn box and the
CV-STEM trajectories drawn on top of it.

One picture answering three questions that are usually checked separately:

  background  lam(x), the LOCAL closed-loop contraction rate under the certified
              metric -- the largest lam satisfying
                  A_cl' M + M A_cl + 2 lam M <= 0,
                  A_cl = A - B K,  K = (1/r) B' M,  M = W(x)^-1
              solved as a generalized eigenproblem at each grid point. Dark =
              slow = hard. This is the quantity rule.md Step 3 tells you to spawn
              inside, so it should be possible to SEE whether the box does.

  spawn       where episodes actually begin, drawn from the env's own
              define_initial_state -- never from a named box, since which box
              reset() reads depends on whether X_INIT is set.

  paths       trajectories under the certified controller u = uref - K(x)e, with
              the achieved AUC in the title. Marked where a state CLAMPS: a
              clamped step means the plant is no longer f+Bu and the certificate
              describes nothing, so those points are exactly where the picture
              stops being trustworthy.

The two free dims are held at the spawn median, so the slice passes through the
region episodes really occupy rather than through the origin.

    python visualization/rate_landscape.py --env segway --dims 1,3
    python visualization/rate_landscape.py --env car --dims 0,1 --out figures/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

import contractionRL.tasks.direct.classic  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from contractionRL.agents.skrl.math_utils import jacobian  # noqa: E402


def _short(env_name: str) -> str:
    s = env_name.removeprefix("classic-").removesuffix("-v0")
    return "car_v1" if s in ("car-v1", "car_v1") else s


def load_metric(short: str):
    paths = sorted(glob.glob(os.path.join(_ROOT, f"data/classic/{short}/*.npz")))
    if not paths:
        raise SystemExit(f"{short}: no CM dataset — nothing certified to plot.")
    d = np.load(paths[0])
    return (np.asarray(d["x"], np.float64), np.asarray(d["W"], np.float64),
            os.path.basename(paths[0]))


def local_rate(env, pts, xs, W, r, want_dist=False):
    """lam(x) at each row of `pts`, using the nearest certified W.

    Returns the distance to that neighbour too, because it is not a diagnostic
    detail -- it is the difference between a certified number and a made-up one.
    W is only known AT the dataset states; everywhere else this substitutes the
    nearest one, and in 4-D a 10000-point cloud is sparse enough that a 2-D slice
    at fixed free dims can sit far from all of them. There the substituted metric
    is simply the wrong metric, and the "rate" it yields can come out NEGATIVE --
    -2.95 on segway, against a certified 0.0514 that is a lower bound. Those
    points are masked rather than plotted, since a plot that shows the plant
    expanding where the certificate says it contracts is worse than a gap.
    """
    t = torch.as_tensor(pts, dtype=torch.float32).requires_grad_()
    with torch.enable_grad():
        f, B, _ = env.get_f_and_B(t)
    A = jacobian(f, t, create_graph=False).detach().numpy().astype(np.float64)
    Bn = B.detach().numpy().astype(np.float64)
    tree = torch.as_tensor(xs, dtype=torch.float32)
    idx = torch.cdist(torch.as_tensor(pts, dtype=torch.float32), tree).argmin(-1).numpy()
    dist = torch.cdist(torch.as_tensor(pts, dtype=torch.float32),
                       tree).min(-1).values.numpy()
    out = np.empty(len(pts))
    for k in range(len(pts)):
        M = np.linalg.inv(W[idx[k]])
        Acl = A[k] - Bn[k] @ ((1.0 / r) * Bn[k].T @ M)
        S = Acl.T @ M + M @ Acl
        out[k] = -0.5 * sla.eigh(0.5 * (S + S.T), M, eigvals_only=True)[-1]
    return (out, dist) if want_dist else out


def rollout(env, xs, W, r, steps):
    """Certified controller. Returns paths, clamp mask, AUC, e(T)/e(0)."""
    env.reset()
    tree = torch.as_tensor(xs, dtype=torch.float32)
    P, C, E, NN = [], [], [], []
    for t in range(steps):
        xt = env.x_t
        P.append(xt.numpy().copy())
        _d = torch.cdist(xt, tree)
        idx = _d.argmin(-1).numpy()
        NN.append(float(_d.min(-1).values.mean()))
        xr = env.xref[:, min(t, env.xref.shape[1] - 1)]
        ur = env.uref[:, min(t, env.uref.shape[1] - 1)].numpy()
        e = env.wrap_angles(xt - xr).numpy().astype(np.float64)
        with torch.no_grad():
            _, B, _ = env.get_f_and_B(xt)
        Bn = B.numpy().astype(np.float64)
        u = np.array([ur[i] - ((1.0 / r) * Bn[i].T @ np.linalg.inv(W[idx[i]])) @ e[i]
                      for i in range(len(e))])
        E.append(np.linalg.norm(e, axis=-1))
        env.step(u.astype(np.float32))
        nxt = env.x_t.numpy()
        # a state sitting exactly on the box edge was clamped there
        C.append((np.isclose(nxt, env.X_MIN.numpy(), atol=1e-6)
                  | np.isclose(nxt, env.X_MAX.numpy(), atol=1e-6)).any(-1))
    E = np.array(E)
    e0 = np.maximum(E[0], 1e-12)
    auc = float((float(env.dt) * (E / e0).sum(axis=0)).mean())
    return np.array(P), np.array(C), auc, float((E[-1] / e0).mean()), float(np.mean(NN))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True)
    p.add_argument("--dims", default=None, help="i,j (default: the two most rate-relevant)")
    p.add_argument("--grid", type=int, default=70)
    p.add_argument("--paths", type=int, default=12)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--algorithm", default="c2rl-ppo")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    short = _short(args.env)
    task = f"classic-{short}-v0" if short != "car_v1" else "classic-car-v1"
    with open(os.path.join(_ROOT, "source/contractionRL/contractionRL/tasks/direct/"
                           f"classic/{short}/agents/"
                           f"skrl_{args.algorithm.replace('-', '_')}_cfg.yaml")) as fh:
        cm = yaml.safe_load(fh)["cm"]
    r = float(cm["cvstem_r_scaler"]) + 1e-5
    xs, W, npz = load_metric(short)

    env = gym.make(task, num_envs=args.paths, device="cpu").unwrapped
    names = list(getattr(env, "state_names", None) or
                 [f"x{i}" for i in range(int(env.num_dim_x))])
    steps = args.steps or int(env.episode_len)

    # Where episodes really start -- ask the env, never a box name.
    with torch.no_grad():
        _, _, x0 = env.define_initial_state(torch.arange(2048))
    x0 = x0.numpy()

    if args.dims:
        i, j = (int(v) for v in args.dims.split(","))
    else:
        lam_s = local_rate(env, xs[:1500], xs, W, r)
        rel = [abs(np.corrcoef(lam_s, np.abs(xs[:1500, k]))[0, 1])
               for k in range(len(names))]
        i, j = int(np.argsort(rel)[-1]), int(np.argsort(rel)[-2])

    lo, hi = env.X_MIN.numpy(), env.X_MAX.numpy()
    # lam is evaluated AT THE CERTIFIED STATES and then projected onto the
    # slice, NOT on a grid. The LMI
    #     A_cl(x)' M(x) + M(x) A_cl(x) + 2 lam M(x) <= 0
    # only means anything when the metric and the Jacobian are taken at the SAME
    # state. Pairing W(nearest dataset state) with A(grid point) violates that,
    # and it does not fail gracefully: on segway it returned lam = -2.95, i.e.
    # the plant expanding, where the certificate guarantees at least +0.0514.
    # (A nearest-neighbour DISTANCE mask does not rescue it either -- I tried,
    # and it masked 0.0% while the negative values remained. The defect is the
    # mismatched pairing, not the distance.)
    #
    # So: exact lam at each dataset state, then a 2-D binned median over the
    # slice dims. Bins with no certified state stay empty rather than being
    # interpolated into existence.
    from scipy.stats import binned_statistic_2d
    lam_pts = local_rate(env, xs, xs, W, r)
    stat, ei, ej, _ = binned_statistic_2d(
        xs[:, i], xs[:, j], lam_pts, statistic="median",
        bins=[np.linspace(lo[i], hi[i], args.grid // 2),
              np.linspace(lo[j], hi[j], args.grid // 2)])
    GI, GJ = np.meshgrid(0.5 * (ei[1:] + ei[:-1]), 0.5 * (ej[1:] + ej[:-1]))
    lam_plot = np.ma.masked_invalid(stat.T)
    ok = lam_pts
    far = np.isnan(stat)

    P, C, auc, ratio, nn = rollout(env, xs, W, r, steps)
    # The overlaid controller uses W from the NEAREST certified state, so the
    # figure reports how close that neighbour actually was. It is not the whole
    # story though, and quadrotor is the counterexample worth recording: its
    # neighbours ARE close (0.9x the cloud spacing) and the rollout still ends at
    # AUC ~40 with the error 7-13x its start. Two hypotheses I checked and
    # DISPROVED first -- nearest-neighbour distance (0.9x, fine) and forward-Euler
    # instability (max|1+dt*eig(A_cl)| = 0.962, stable). The per-dim trace gives
    # the real answer:
    #     vel/attitude   0.077 -> 0.143, 0.072 -> 0.119   held
    #     position       0.073 -> 1.90,  0.081 -> 1.48    drifts
    # The certified gain stabilises the fast states and lets position walk. That
    # is consistent with Step 5, where a full-state LQR converged on quadrotor
    # (0.126) and this controller did not: contraction is an INCREMENTAL property
    # -- neighbouring trajectories approach each other -- and it does not by
    # itself buy tracking authority on the slow, indirectly-actuated states.
    # So the AUC on this figure is a real measurement of the certified controller,
    # not of the plant's best achievable tracking.
    spacing = float(np.median(torch.cdist(torch.as_tensor(xs[:1500], dtype=torch.float32),
                                          torch.as_tensor(xs, dtype=torch.float32))
                              .kthvalue(2, dim=-1).values.numpy()))
    nn_ratio = nn / max(spacing, 1e-9)
    trust = nn_ratio <= 3.0

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    m = ax.contourf(GI, GJ, lam_plot, levels=24, cmap="viridis")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"certified local contraction rate  $\lambda(x)$   (dark = slow = hard)")
    ax.contour(GI, GJ, lam_plot, levels=[float(cm["lbd"])], colors="w",
               linewidths=2.0, linestyles="--")

    ax.scatter(x0[:, i], x0[:, j], s=7, c="white", alpha=.35, edgecolors="none",
               label="initial states (define_initial_state)", zorder=3)
    for k in range(P.shape[1]):
        ax.plot(P[:, k, i], P[:, k, j], lw=1.0, color="crimson", alpha=.75,
                zorder=4, label="CV-STEM trajectory" if k == 0 else None)
    ax.scatter(P[0, :, i], P[0, :, j], s=42, marker="o", facecolors="none",
               edgecolors="crimson", lw=1.6, zorder=5, label="start")
    ax.scatter(P[-1, :, i], P[-1, :, j], s=60, marker="*", c="gold",
               edgecolors="k", lw=.5, zorder=6, label="end")
    if C.any():
        ax.scatter(P[:-1][C[:-1]][:, i] if P[:-1][C[:-1]].ndim > 1 else [],
                   P[:-1][C[:-1]][:, j] if P[:-1][C[:-1]].ndim > 1 else [],
                   s=9, c="red", marker="x", zorder=7,
                   label=f"CLAMPED ({100*C.mean():.1f}% of steps)")

    ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i]-lo[i], hi[j]-lo[j],
                               fill=False, ec="k", lw=1.2, ls=":"))
    ax.set_xlabel(names[i])
    ax.set_ylabel(names[j])
    _auc_txt = f"AUC {auc:.2f}, e(T)/e(0) {ratio:.4f}"
    if not trust:
        _auc_txt += (f"  [nearest certified state {nn_ratio:.0f}x the cloud spacing "
                     f"-- gain is interpolated]")
    if ratio > 1.0:
        _auc_txt += "  -- certified controller does NOT track here"
    ax.set_title(f"{short}   certified $\\lambda$={cm['lbd']}, r={cm['cvstem_r_scaler']}   |   "
                 f"{_auc_txt}\n"
                 f"$\\lambda(x)$ where certified: min {ok.min():.3f}  med {np.median(ok):.3f}  "
                 f"max {ok.max():.3f}   (dashed = certified $\\lambda$; "
                 f"grey = {100*far.mean():.0f}% extrapolated, not plotted)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=.85)
    os.makedirs(os.path.join(_ROOT, args.out), exist_ok=True)
    path = os.path.join(_ROOT, args.out, f"rate_landscape_{short}_{names[i]}_{names[j]}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    print(f"  lam(x) where certified: min {ok.min():.4f}  median {np.median(ok):.4f} "
          f" max {ok.max():.4f}   ({100*far.mean():.1f}% of the slice masked as extrapolation)")
    print(f"  AUC {auc:.3f}   e(T)/e(0) {ratio:.4f}   clamped {100*C.mean():.2f}% of steps"
          f"   nn/spacing {nn_ratio:.1f}{'' if trust else '  <- gain is INTERPOLATED, AUC unreliable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

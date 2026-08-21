"""Min-projected λ*(x) plots: cached data in, styled figure out.

Two jobs, deliberately separable:

  --generate   compute figures/data/minproj_<env>.npz  (the expensive half: one
               λ* bisection per grid cell, each a joint CV-STEM SDP)
  (default)    read that npz and draw figures/minproj_<env>.png

Keeping them apart is the point. The λ* grid costs thousands of SDP solves and
never changes for a fixed (env, envelope), so restyling a figure must not re-solve
it — the cached npz is the interface, and the eight envs already cached are
replotted in seconds.

"Min-projected": the figure is a 2D slice over the two dims the dynamics actually
depend on, and every OTHER dim is projected out by taking the MINIMUM λ* over a
sample of its box. Minimum, not mean: λ* is a certificate over a set, so the
honest value at a slice point is the worst case behind it.

x0 is drawn as a DENSITY, not a scatter. Two thousand points over a 15x15 slice
is past the count where individual markers read as anything — they cover the
heatmap and hide it. A filled contour of the same samples shows where the reset
distribution actually concentrates, which is the question the panel answers.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))
sys.path.insert(0, str(REPO / "scripts"))

FIG_DIR = REPO / "figures"
DATA_DIR = FIG_DIR / "data"

# Every text element, one place. The defaults are matplotlib's ~10 pt, which is
# unreadable once a figure is scaled into a paper column.
FS_TITLE = 20
FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 15
FS_CBAR = 16

LINEWIDTH = 3.2          # trajectory stroke
ARROW_FRAC = 0.026       # arrow length as a fraction of the panel
ARROW_WIDTH = 0.0075     # arrow shaft width (axes fraction)


def pretty(env: str) -> str:
    """car_weak -> Car_weak? No: Car Weak. Underscores are word breaks here."""
    return " ".join(w.capitalize() for w in env.split("_"))


# ────────────────────────────── generate ─────────────────────────────────── #

def generate(env_name: str, *, grid: int, n_other: int, n_x0: int,
             n_traj: int, seed: int) -> pathlib.Path:
    import gymnasium as gym
    import torch

    import contractionRL.tasks.direct.classic  # noqa: F401
    from contractionRL.agents.skrl.ncm_synthesis import sample_state_box
    from lambda_subsets import active_dims_auto, jacobians, max_lambda  # noqa: E402

    env = gym.make(f"classic-{env_name}-v0", num_envs=n_traj, device="cpu").unwrapped
    x_min = env.X_MIN.detach().cpu().numpy().astype(np.float64)
    x_max = env.X_MAX.detach().cpu().numpy().astype(np.float64)

    cfg = _cm_cfg(env_name)
    kw = dict(eps=cfg["cm_eps"], dt=1.0, solver="MOSEK", r_scaler=cfg["r"],
              w_lb=cfg["w_lb"], w_ub=cfg["w_ub"])

    # The two dims to plot: the last two the dynamics actually depend on. For the
    # car that is (yaw, vel) — position drops out because f ignores x and y, so a
    # slice over position would be constant by construction.
    dims = active_dims_auto(env)[-2:]
    gx = np.linspace(x_min[dims[0]], x_max[dims[0]], grid)
    gy = np.linspace(x_min[dims[1]], x_max[dims[1]], grid)

    rng = np.random.default_rng(seed)
    # One draw of the projected-out dims, REUSED at every cell, so neighbouring
    # cells differ because of the slice coordinates and not because of the RNG.
    other = rng.uniform(x_min, x_max, size=(n_other, x_min.size))

    Z = np.empty((grid, grid), dtype=np.float64)
    print(f"[minproj] {env_name}: {grid}x{grid} cells, min over {n_other} "
          f"projected samples, dims={[env.state_names[d] for d in dims]}", flush=True)
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            pts = other.copy()
            pts[:, dims[0]] = xv
            pts[:, dims[1]] = yv
            A, B = jacobians(env, pts.astype(np.float32))
            # min over the projected dims: each sample alone is one certificate,
            # and the slice inherits the worst of them.
            lams = [max_lambda(A[k:k + 1], B[k:k + 1], kw)[0] for k in range(len(pts))]
            Z[j, i] = float(np.min(lams))
        print(f"[minproj]   column {i + 1}/{grid} done", flush=True)

    x0 = sample_state_box(env.X_MIN, env.X_MAX, n=n_x0, seed=seed).astype(np.float64)

    # Trajectories: the env's own reference rollout, which is what the panel is
    # about — where the closed loop actually goes, not where the box allows.
    traj = _rollout(env, n_traj)

    out = DATA_DIR / f"minproj_{env_name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Z=Z, dims=np.asarray(dims), grid_x=gx, grid_y=gy,
             exact=np.asarray(True), lbd=np.asarray(cfg["lbd"]),
             r=np.asarray(cfg["r"]), w_lb=np.asarray(cfg["w_lb"]),
             w_ub=np.asarray(cfg["w_ub"]), traj=traj, x0=x0,
             state_names=np.asarray(list(env.state_names)),
             x_min=x_min, x_max=x_max)
    print(f"[minproj] wrote {out}")
    return out


def _cm_cfg(env_name: str) -> dict:
    """The env's shipped (lbd, r, envelope) — the figure must describe what ships."""
    import yaml
    p = (REPO / "source/contractionRL/contractionRL/tasks/direct/classic"
         / env_name / "agents/skrl_c2rl_ppo_cfg.yaml")
    cm = yaml.safe_load(p.read_text(encoding="utf-8"))["cm"]
    return {"lbd": float(cm["lbd"]), "r": float(cm["cvstem_r_scaler"]),
            "w_lb": float(cm["w_lb"]), "w_ub": float(cm["w_ub"]),
            "cm_eps": float(cm["cm_eps"])}


def _rollout(env, n_traj: int) -> np.ndarray:
    """Open-loop-along-reference rollout: x(t) under the env's own uref."""
    import torch
    obs, _ = env.reset(seed=0)
    T = int(env.max_episode_len)
    xs = [env.x_t.detach().cpu().numpy().copy()]
    for t in range(T):
        u = env.uref[:, min(t, env.uref.shape[1] - 1)]
        obs, *_ = env.step(u.detach().cpu().numpy()
                           if isinstance(u, torch.Tensor) else u)
        xs.append(env.x_t.detach().cpu().numpy().copy())
    return np.transpose(np.asarray(xs, dtype=np.float64), (1, 0, 2))


# ─────────────────────────────── plot ────────────────────────────────────── #

def plot(env_name: str) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = DATA_DIR / f"minproj_{env_name}.npz"
    if not src.exists():
        raise SystemExit(f"no cached data at {src} — run with --generate first")
    z = np.load(src, allow_pickle=True)
    Z, gx, gy = z["Z"], z["grid_x"], z["grid_y"]
    dims = z["dims"]
    names = [str(s) for s in z["state_names"]]
    dx, dy = int(dims[0]), int(dims[1])

    fig, ax = plt.subplots(figsize=(9.5, 7.6))

    # λ*(x) background. pcolormesh over the cell EDGES, not centres, so a cell's
    # colour covers the region it was computed for.
    ex = _edges(gx)
    ey = _edges(gy)
    mesh = ax.pcolormesh(ex, ey, Z, cmap="viridis", shading="flat")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    # A constant field is stated on the colorbar, not the title: it is a fact
    # about lambda*, and the old figures wasted title width on it while
    # matplotlib's offset text ("1e-14+4.88") overlapped the title anyway.
    lab = r"$\lambda^*(x)$"
    if float(Z.max() - Z.min()) < 1e-6:
        lab += f"  (constant {float(Z.mean()):.4g})"
    cb.set_label(lab, fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_TICK)
    # A near-constant field makes matplotlib print an offset like "1e-14+4.88",
    # which collided with the title in the old figures and told the reader
    # nothing. Say "constant" once instead.
    cb.formatter.set_useOffset(False)
    cb.update_ticks()

    # x0 as a DENSITY, not 2000 markers.
    x0 = z["x0"]
    _density(ax, x0[:, dx], x0[:, dy], gx, gy)

    # Trajectories, thicker, with direction arrows.
    traj = z["traj"]
    for k in range(traj.shape[0]):
        tx, ty = traj[k, :, dx], traj[k, :, dy]
        ax.plot(tx, ty, "-", color="#d62728", lw=LINEWIDTH, alpha=0.9,
                zorder=4, label=r"$x(t)$" if k == 0 else None)
        _arrows(ax, tx, ty)

    ax.set_xlabel(names[dx], fontsize=FS_LABEL)
    ax.set_ylabel(names[dy], fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(ex[0], ex[-1])
    ax.set_ylim(ey[0], ey[-1])

    spread = float(Z.max() - Z.min())
    ax.set_title(
        f"{pretty(env_name)}:  $\\lambda$={float(z['lbd']):g}, r={float(z['r']):g}, "
        f"$w_{{lb}}$={float(z['w_lb']):g}, $w_{{ub}}$={float(z['w_ub']):g}",
        fontsize=FS_TITLE, pad=14)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9)

    out = FIG_DIR / f"minproj_{env_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[minproj] wrote {out}  ({names[dx]} x {names[dy]}, "
          f"lambda* spread {spread:.3g})")
    return out


def _edges(c: np.ndarray) -> np.ndarray:
    """Cell centres -> edges, so pcolormesh colours the region each cell covers."""
    step = (c[-1] - c[0]) / max(len(c) - 1, 1)
    return np.concatenate([c - step / 2.0, [c[-1] + step / 2.0]])


def _density(ax, u, v, gx, gy) -> None:
    """Filled-contour density of x0 with a light outline, plus one legend proxy."""
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.lines import Line2D
    lo_u, hi_u = _edges(gx)[[0, -1]]
    lo_v, hi_v = _edges(gy)[[0, -1]]
    inside = (u >= lo_u) & (u <= hi_u) & (v >= lo_v) & (v <= hi_v)
    u, v = u[inside], v[inside]
    if u.size < 8:
        ax.scatter(u, v, s=26, c="white", edgecolors="k", zorder=5,
                   label=r"$x_0$")
        return
    # 2D histogram, smoothed by the contour interpolation itself. A histogram
    # rather than a KDE: no bandwidth to justify, and the reset draw is uniform
    # over a box, so the honest picture is "flat inside, zero outside" and a KDE
    # would round the corners it is meant to show.
    # 12 bins, not 18: at n=2000 an 18x18 grid holds ~6 counts per bin, so the
    # contours trace Poisson noise and invent structure. Coarser bins plus one
    # smoothing pass, and the levels start at 0 so a uniform box draw -- which is
    # exactly what reset() produces -- actually looks uniform instead of mottled.
    H, ue, ve = np.histogram2d(u, v, bins=(12, 12),
                               range=[[lo_u, hi_u], [lo_v, hi_v]])
    uc = 0.5 * (ue[:-1] + ue[1:])
    vc = 0.5 * (ve[:-1] + ve[1:])
    dens = _smooth(H.T / max(H.sum(), 1.0))
    if dens.max() <= 0:
        return
    levels = np.linspace(0.0, dens.max(), 7)
    ax.contourf(uc, vc, dens, levels=levels, cmap="Greys", alpha=0.4, zorder=2)
    ax.contour(uc, vc, dens, levels=levels[1:], colors="white",
               linewidths=1.0, alpha=0.6, zorder=3)
    ax.plot([], [], marker="s", ls="none", ms=12, color="0.35",
            label=r"$x_0$ density")


def _smooth(a: np.ndarray) -> np.ndarray:
    """Separable [1,2,1]/4 pass — enough to stop Poisson noise reading as shape.

    Hand-rolled rather than scipy.ndimage: this module is imported by a plotting
    entry point that should not pull a new dependency for three lines of blur.
    """
    k = np.array([1.0, 2.0, 1.0])
    k = k / k.sum()

    def blur(m):
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, m)
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, out)

    # Edge correction. np.convolve(mode="same") zero-pads, so without dividing by
    # the blurred indicator the border loses mass and a UNIFORM draw renders with
    # a bright centre and dark rim -- structure the data does not have. reset()
    # samples uniformly over the box, so flat-inside is the correct picture.
    num = blur(a)
    den = blur(np.ones_like(a))
    return num / np.where(den > 0, den, 1.0)


def _arrows(ax, tx, ty, every: int = 45) -> None:
    """Direction arrows along one trajectory, at a fixed fraction of the panel.

    NOT scale_units="xy": consecutive trajectory samples are ~1e-2 apart in data
    units, so an arrow drawn at the step vector's own length collapses to a dot
    (which is what the first version did). The direction is normalized and the
    length comes from the axis range, so an arrow reads the same whatever units
    the two plotted states happen to have.
    """
    idx = np.arange(every, len(tx) - 1, every)
    if idx.size == 0:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    span_x, span_y = x1 - x0, y1 - y0
    du = np.diff(tx)[idx]
    dv = np.diff(ty)[idx]
    # Normalize in PANEL space, else a state with a wide range dominates the
    # direction and every arrow points the same way.
    nu, nv = du / span_x, dv / span_y
    mag = np.hypot(nu, nv)
    keep = mag > 1e-9
    if not keep.any():
        return
    L = ARROW_FRAC
    ax.quiver(tx[idx][keep], ty[idx][keep],
              (nu[keep] / mag[keep]) * L * span_x,
              (nv[keep] / mag[keep]) * L * span_y,
              angles="xy", scale_units="xy", scale=1.0,
              width=ARROW_WIDTH, headwidth=3.6, headlength=4.0,
              headaxislength=3.4, color="#7f0f14", zorder=6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default="car,car_weak,segway,cartpole,quadrotor",
                   help="comma-separated short env names")
    p.add_argument("--generate", action="store_true",
                   help="recompute the lambda* grid (expensive) before plotting")
    p.add_argument("--grid", type=int, default=15)
    p.add_argument("--n-other", "--n_other", type=int, default=8,
                   help="samples of the projected-out dims per cell (the min is "
                        "taken over these)")
    p.add_argument("--n-x0", "--n_x0", type=int, default=2000)
    p.add_argument("--n-traj", "--n_traj", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    envs = [e.strip() for e in a.envs.split(",") if e.strip()]
    for e in envs:
        if a.generate or not (DATA_DIR / f"minproj_{e}.npz").exists():
            generate(e, grid=a.grid, n_other=a.n_other, n_x0=a.n_x0,
                     n_traj=a.n_traj, seed=a.seed)
        plot(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

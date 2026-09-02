"""Min-projected λ*(x) plots: cached data in, styled figure out.

Two jobs, deliberately separable:

  --generate   compute figures/data/minproj_<env>.npz  (the expensive half: one
               λ* bisection per grid cell, each a joint CV-STEM SDP)
  (default)    read that npz and draw figures/minproj_<env>.svg

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
VIOL_FRAC = 0.05   # same budget as find_uniform_lambda
DATA_DIR = FIG_DIR / "data"

# Every text element, one place. The defaults are matplotlib's ~10 pt, which is
# unreadable once a figure is scaled into a paper column.
FS_TITLE = 20
FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 15
FS_CBAR = 16

LINEWIDTH = 3.2          # trajectory stroke
N_TRAJ_SHOWN = 1         # ONE x(t) and its xref(t); more reads as a tangle
ARROW_FRAC = 0.026       # arrow length as a fraction of the panel
ARROW_WIDTH = 0.0075     # arrow shaft width (axes fraction)


# Title text, where capitalizing the short name is not what should be shown.
# The car pair is titled v0/v1 on BOTH sides: "Car" against "Car v1" would read
# as a base case beside a variant, when they are two settings of one plant.
DISPLAY_NAME = {"car": "Car v0", "car_v1": "Car v1"}


def env_id(env: str) -> str:
    """Short name -> gym id, via the one map the classic package owns."""
    from contractionRL.tasks.direct.classic import env_id as _resolve
    return _resolve(env)


def pretty(env: str) -> str:
    """Short name -> title text. two_link_arm -> Two Link Arm; car_v1 -> Car v1."""
    if env in DISPLAY_NAME:
        return DISPLAY_NAME[env]
    return " ".join(w.capitalize() for w in env.split("_"))


# ────────────────────────────── generate ─────────────────────────────────── #

def _make_env(env_name: str, n_traj: int, ref_mode: str, migrate_gain: float):
    """gym.make + the reference mode.

    Set post-construction because ``reference_mode`` is read by
    ``_rollout_reference`` at reset() time, not baked in at __init__. Matters for
    car/car_v1: their ``sample_reference_controls`` only drives the STEERING
    channel, so in "stabilizing" mode the reference velocity is frozen at its draw
    for the whole episode and the reference can never leave the low-rate region.
    The migration term is the only thing that moves it."""
    import gymnasium as gym
    env = gym.make(env_id(env_name), num_envs=n_traj, device="cpu").unwrapped
    env.reference_mode = ref_mode
    env.migrate_gain = migrate_gain
    # The constructor already built a 64-reference pool (_pooled_system_reset
    # amortizes _rollout_reference over a batch), so without this the mode set
    # above is silently ignored: reset() serves cached arbitrary-mode references
    # and _migrate_uref is never called. Measured: 0 calls, reference velocity
    # frozen at its draw. Clearing forces the pool to be rebuilt under the mode.
    env._ref_pool = None
    if ref_mode == "contractive" and env.XREF_INIT_FAST_MIN is None:
        raise SystemExit(
            f"{env_name}: reference_mode=contractive needs xref_init_fast_min/max "
            f"in its ENV_CONFIG (rule.md Step 3) -- none defined.")
    return env


def _attach_cmg(env, env_name: str) -> None:
    """Put the SHIPPED metric on the env, so ``local_lambda`` has one.

    The grid in the npz comes from the SDP directly and never touches the env.
    """
    from contractionRL import cm_data

    try:
        cm_data.attach_cmg(env, env_name, tag=f"[minproj:{env_name}]")
    except FileNotFoundError as e:
        raise SystemExit(f"{env_name}: needs a CM dataset.\n{e}") from e


def _ceiling_grid(env, kw, dims, gx, gy, other, actuator_ok, grid):
    """The ORIGINAL estimator: re-solve a single-state SDP per cell and take the
    largest gated lam. Answers "how fast could you contract here if you designed a
    metric for this state alone" -- a ceiling, not the shipped controller's rate.

    Kept behind --estimator ceiling because that ceiling is a legitimate question,
    but it is not the one this figure is captioned with, and it carries the
    actuator gate's frame dependence into the value (see _certified_grid).
    """
    from lambda_subsets import jacobians, max_lambda  # noqa: PLC0415
    Z = np.empty((grid, grid), dtype=np.float64)
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            pts = other.copy()
            pts[:, dims[0]] = xv
            pts[:, dims[1]] = yv
            A, B = jacobians(env, pts.astype(np.float32))
            # min over the projected dims: each sample alone is one certificate,
            # and the slice inherits the worst of them.
            lams = [max_lambda(A[k:k + 1], B[k:k + 1], kw, gate=actuator_ok)[0]
                    for k in range(len(pts))]
            Z[j, i] = float(np.min(lams))
        print(f"[minproj]   column {i + 1}/{grid} done", flush=True)
    return Z


def _certified_grid(env, cfg, dims, gx, gy, other):
    """lam(x) of the CERTIFIED metric -- the quantity the certificate is about.

    The "ceiling" estimator this replaces re-solves a SEPARATE single-state SDP at
    every cell, so it answers "how fast COULD you contract here if you designed a
    metric just for this state". That is a real question but it is not the one the
    figure is captioned with: the deployed controller carries ONE certified metric
    field W(x), and the state-dependence a reader cares about is that field's rate.

    Three things follow from using the right object, all of them measured:

    * It is EXACT along the plant's symmetries. lam is a generalized eigenvalue of
      (Acl'M + M Acl, M), so a congruence x -> T x with T'MT the transported metric
      leaves it unchanged. On the car's yaw axis the ceiling estimator reported
      1.0398x spread; this reports 1.000000x, which is what a yaw-invariant plant
      must give. Nothing has to be denoised for that to hold.
    * It has no actuator gate in it, so it cannot inherit the gate's frame bug --
      the gate draws e from XE_INIT, a fixed WORLD-frame box, while K(x) rotates
      with the state, and a box is not rotation-invariant. Measured systematic
      ripple 3.3 sd at 200k draws, which no sample count removes. The actuator
      question is still worth asking; it belongs on top as a mask, not multiplied
      into the value.
    * ONE SDP instead of ~54000. The grid points are variables of the same program,
      so no cell gets a nearest-neighbour W substituted for its own.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from contractionRL.agents.skrl.ncm_synthesis import (  # noqa: I001, PLC0415
        cvstem_joint,
        drift_jacobians,
    )
    from find_x_init import local_rates  # noqa: E402, PLC0415

    pts = []
    for xv in gx:
        for yv in gy:
            q = other.copy()
            q[:, dims[0]] = xv
            q[:, dims[1]] = yv
            pts.append(q)
    pts = np.concatenate(pts, axis=0)                     # (grid*grid*n_other, x_dim)
    print(f"[minproj] certified metric: ONE joint SDP over {len(pts)} states "
          f"(lbd={cfg['lbd']}, r={cfg['r']}, eps={cfg['cm_eps']})", flush=True)
    A, B = drift_jacobians(env.get_f_and_B, pts.astype(np.float32))
    sol = cvstem_joint(A, B, lbd=cfg["lbd"], eps=cfg["cm_eps"], dt=1.0,
                       solver="MOSEK", r_scaler=cfg["r"],
                       w_lb=cfg["w_lb"], w_ub=cfg["w_ub"])
    if sol is None:
        raise SystemExit(
            f"joint SDP infeasible at the env's shipped lbd={cfg['lbd']} over the "
            f"plot grid. That is a rule.md Step 1 result about this env, not a "
            f"plotting failure -- settle the rate before drawing it.")
    lam = local_rates(env, pts, sol["W"], cfg["r"])
    # Same "min over the projected-out dims" semantics the ceiling estimator uses,
    # so the two panels are directly comparable.
    return lam.reshape(len(gx), len(gy), -1).min(axis=2).T   # Z[j, i]


def generate(env_name: str, *, grid: int, n_other: int, n_x0: int,
             n_traj: int, seed: int, ref_mode: str = "stabilizing",
             migrate_gain: float = 1.0,
             estimator: str = "certified") -> pathlib.Path:
    import contractionRL.tasks.direct.classic  # noqa: F401
    import torch
    from find_uniform_lambda import control_violation_rate, expand_box  # noqa: E402
    from lambda_subsets import active_dims_auto  # noqa: E402

    env = _make_env(env_name, n_traj, ref_mode, migrate_gain)
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
    #
    # Drawn ONLY over dims A(x)/B(x) actually depend on. Position is the case that
    # matters: the car's f = [v cos(th), v sin(th), 0, 0] ignores x and y outright,
    # so randomising them across the 8 projected samples cannot change lam and can
    # only add spread to the min taken over them -- variation in a figure about
    # the rate, sourced from coordinates the rate provably cannot see. Inactive
    # dims are pinned at the box centre instead. active_dims_auto already keeps
    # them off the plotted axes for the same reason; this applies the same rule to
    # what is projected out.
    act = set(active_dims_auto(env).tolist())
    other = np.tile(0.5 * (x_min + x_max), (n_other, 1))
    for d in sorted(act):
        other[:, d] = rng.uniform(x_min[d], x_max[d], size=n_other)

    # The ACTUATOR gate, same one find_uniform_lambda applies. Without it lam* is
    # the rate the LMI would allow if control were free, which on the car is 4.88
    # against a shipped 0.3902 -- and reaching 4.88 needs ‖K‖₂ ≈ 60 against a ±1.5
    # actuator, putting 99.8% of the commanded controls outside the box. That is a
    # ceiling, not a rate any trajectory can exhibit, and drawing it as the
    # background of a figure whose trajectories converge at 0.39 invites exactly
    # the wrong reading. With the gate, lam* is the fastest REALIZABLE rate.
    #
    # Errors come from the env's own define_initial_state, never from a named box
    # -- which box reset() reads depends on whether X_INIT is set.
    with torch.no_grad():
        _, xe_0, _ = env.define_initial_state(torch.arange(4096, device=env.device))
    e_pool = xe_0.detach().cpu().numpy().astype(np.float64)
    uref_lo = env.UREF_MIN.detach().cpu().numpy().astype(np.float64)
    uref_hi = env.UREF_MAX.detach().cpu().numpy().astype(np.float64)
    u_lo, u_hi = expand_box(uref_lo, uref_hi, 2.0)
    # 2000, not 64. The gate is called ONE STATE AT A TIME here (max_lambda hands
    # it W[k:k+1]), so n_draw IS the entire sample -- unlike find_uniform_lambda,
    # which pools num_samples x n_draws = 1000 x 100 and lands at sd 0.0006.
    #
    # At 64 the violation fraction has sd 0.023 against a 0.05 budget that the true
    # value (0.0350-0.0367 across every car cell) sits only 0.65 sd under, so ~26%
    # of cells gate out on RNG alone. And because the draws are SHARED across cells,
    # those errors are spatially correlated -- correlated noise renders as smooth
    # contiguous blobs, which read as physics. Measured damage on the car: 17.3% of
    # cells painted lam*=0, and the survivors split across two adjacent scan bins
    # for an apparent 1.46x state dependence, on a plant whose rate is flat to five
    # digits (local_rates over 2000 states: 0.44198-0.44199). 1.46x is exactly one
    # bin of the geomspace in max_lambda -- the figure was drawing the coin flip.
    # Quadrotor's 2.13x was the same flip two bins wide (1.4597^2).
    #
    # 2000 gives sd 0.0041, z = 3.65, i.e. 0.3 falsely-gated cells in a 2500-cell
    # figure. The cost is one (2000, u_dim) matmul per solve, against a MOSEK SDP.
    _n_draw = 2000
    _e_draws = e_pool[rng.integers(0, len(e_pool), size=(1, _n_draw))]
    _u_draws = rng.uniform(uref_lo, uref_hi, size=(1, _n_draw, uref_lo.size))

    def actuator_ok(W, Bk, r_scaler):
        """<= VIOL_FRAC of u = uref - K e inside the applied control box."""
        return control_violation_rate(W, Bk, r_scaler=r_scaler, e_draws=_e_draws,
                                      uref_draws=_u_draws,
                                      u_lo=u_lo, u_hi=u_hi) <= VIOL_FRAC

    print(f"[minproj] {env_name}: {grid}x{grid} cells, min over {n_other} "
          f"projected samples, dims={[env.state_names[d] for d in dims]}, "
          f"estimator={estimator}", flush=True)
    if estimator == "certified":
        Z = _certified_grid(env, cfg, dims, gx, gy, other)
    else:
        Z = _ceiling_grid(env, kw, dims, gx, gy, other, actuator_ok, grid)

    # x0 from the env's OWN reset, not a uniform box draw. reset() takes
    # x_0 = clamp(xref_0, box) + xe_0, and for car_v1 xref's velocity is drawn
    # from [0.3, 1.5] specifically so the plant sits in the weak-authority region
    # (sigma = min(1, v) = v < 1). Sampling uniformly over the box would erase the
    # very concentration this figure is supposed to show, and would also not match
    # the envs whose cached x0 was reset-drawn.
    x0 = _reset_x0(env, n_x0, seed=seed)

    # Trajectories: the env's own reference rollout, which is what the panel is
    # about — where the closed loop actually goes, not where the box allows.
    traj, traj_ref = _rollout(env, n_traj)

    out = DATA_DIR / f"minproj_{env_name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Z=Z, dims=np.asarray(dims), grid_x=gx, grid_y=gy,
             exact=np.asarray(True), lbd=np.asarray(cfg["lbd"]),
             r=np.asarray(cfg["r"]), w_lb=np.asarray(cfg["w_lb"]),
             w_ub=np.asarray(cfg["w_ub"]), traj=traj, traj_ref=traj_ref, x0=x0,
             state_names=np.asarray(list(env.state_names)),
             x_min=x_min, x_max=x_max,
             actuator_gated=np.asarray(True), viol_frac=np.asarray(VIOL_FRAC),
             reference_mode=np.asarray(ref_mode),
             cm_dataset=np.asarray(getattr(env, "_cm_dataset", "")),
             estimator=np.asarray(estimator))
    print(f"[minproj] wrote {out}")
    return out


def refresh_x0(env_name: str, *, n_x0: int, n_traj: int, seed: int,
               ref_mode: str = "stabilizing", migrate_gain: float = 1.0) -> pathlib.Path:
    """Rewrite x0/traj in the cached npz from the CURRENT env; keep Z.

    lambda* depends on the plant and the envelope, so a change to how episodes
    start cannot move it. Recomputing the grid to pick up a new x0 draw would burn
    thousands of SDP solves for an identical result.
    """
    import contractionRL.tasks.direct.classic  # noqa: F401

    src = DATA_DIR / f"minproj_{env_name}.npz"
    if not src.exists():
        raise SystemExit(f"nothing cached at {src} — use --generate")
    old = dict(np.load(src, allow_pickle=True))

    env = _make_env(env_name, n_traj, ref_mode, migrate_gain)
    lo = env.X_MIN.detach().cpu().numpy().astype(np.float64)
    hi = env.X_MAX.detach().cpu().numpy().astype(np.float64)
    x0 = _reset_x0(env, n_x0, seed=seed)
    out_frac = float(((x0 < lo - 1e-6) | (x0 > hi + 1e-6)).any(axis=1).mean())

    old["x0"] = x0
    old["traj"], old["traj_ref"] = _rollout(env, n_traj)
    old["x_min"], old["x_max"] = lo, hi
    old["reference_mode"] = np.asarray(ref_mode)
    old["cm_dataset"] = np.asarray(getattr(env, "_cm_dataset", ""))
    np.savez(src, **old)
    print(f"[minproj] refreshed x0/traj in {src}  "
          f"({out_frac:.2%} of x0 outside the box)")
    return src


def _reset_x0(env, n: int, *, seed: int = 0) -> np.ndarray:
    """``n`` initial states as the env actually produces them, via repeated reset.

    Batched: one reset yields ``num_envs`` states, so this loops until it has n.
    """
    out = []
    k = 0
    while sum(len(o) for o in out) < n:
        env.reset(seed=seed + k)
        out.append(env.x_t.detach().cpu().numpy().copy())
        k += 1
        if k > 4000:                     # a stuck env must not spin forever
            break
    return np.concatenate(out, axis=0)[:n].astype(np.float64)


CONST_REL_TOL = 1e-3          # 0.1% of the mean rate


def _is_constant(Z) -> bool:
    """Is this field constant, in a way that does not depend on its units?

    The test used to be ``Z.max() - Z.min() < 1e-6``, an ABSOLUTE tolerance, and
    it is scale-dependent by construction: a plant whose lambda is ten times
    larger needs ten times the slack to read as equally flat. car came out at
    max-min = 1.870e-06 against that 1e-6 -- constant to 4.2e-06 RELATIVE, i.e.
    0.0004%, and still judged "varying", so the panel fell through to viridis with
    matplotlib's autoscale and painted a full rainbow across a range of 1.9e-06.
    Every fix upstream was intact and the figure still showed a state-dependent
    rate, because the last step invented one.

    0.1% of the mean is far above the numerical floor (4e-06 here) and far below
    any real effect -- the smallest genuine state-dependence measured in this repo
    is car_v1's 2.7x.
    """
    import numpy as np
    Z = np.asarray(Z)
    pos = Z[Z > 0]
    if pos.size == 0:
        return True
    return float(pos.max() - pos.min()) <= CONST_REL_TOL * float(abs(pos.mean()))


def _cm_cfg(env_name: str) -> dict:
    """The env's shipped (lbd, r, envelope) — the figure must describe what ships."""
    import yaml
    p = (REPO / "source/contractionRL/contractionRL/tasks/direct/classic"
         / env_name / "agents/skrl_c2rl_ppo_cfg.yaml")
    cm = yaml.safe_load(p.read_text(encoding="utf-8"))["cm"]
    return {"lbd": float(cm["lbd"]), "r": float(cm["cvstem_r_scaler"]),
            "w_lb": float(cm["w_lb"]), "w_ub": float(cm["w_ub"]),
            "cm_eps": float(cm["cm_eps"])}


def _rollout(env, n_traj: int):
    """Open-loop-along-reference rollout. Returns ``(x(t), xref(t))``.

    Both, because the figure is about the pair: xref is where the task ASKS the
    plant to be, x is where it goes. With reference_mode="contractive" the two
    are visibly different objects -- the reference migrates from the low-rate
    region toward the high-rate one and x follows it -- and drawing only x hides
    what is being asked."""
    import torch
    obs, _ = env.reset(seed=0)
    T = int(env.max_episode_len)
    xs = [env.x_t.detach().cpu().numpy().copy()]
    for t in range(T):
        u = env.uref[:, min(t, env.uref.shape[1] - 1)]
        obs, *_ = env.step(u.detach().cpu().numpy()
                           if isinstance(u, torch.Tensor) else u)
        xs.append(env.x_t.detach().cpu().numpy().copy())
    xr = env.xref[:, :T + 1].detach().cpu().numpy().astype(np.float64)
    return (np.transpose(np.asarray(xs, dtype=np.float64), (1, 0, 2)), xr)


# ─────────────────────────────── plot ────────────────────────────────────── #

# Envs whose layout is pinned by hand rather than inferred. Only needed where
# lambda* is CONSTANT: with a flat field there is no "axis lambda* varies along"
# to detect, so marginal_axis' tie-break decides the layout arbitrarily. car is
# pinned to put vel on x, matching car_v1 -- the two are the same plant and
# read as a pair, so they must not come out transposed relative to each other.
AXIS_OVERRIDE = {"car": "y"}


def marginal_axis(Z: np.ndarray) -> str:
    """"x" or "y": the axis lambda* actually varies along.

    Z is indexed [y, x]. Comparing the peak-to-peak of the per-axis means, rather
    than the raw spread, keeps a single noisy cell from deciding the layout.
    A flat field falls through to "x", where the panel is wider and easier to read
    — see AXIS_OVERRIDE for the envs where that tie-break is not the wanted one.
    """
    var_x = float(np.ptp(Z.mean(axis=0)))
    var_y = float(np.ptp(Z.mean(axis=1)))
    return "y" if var_y > var_x else "x"


def plot(env_name: str, *, n_traj: int = N_TRAJ_SHOWN,
         axis: str | None = None, rate_only: bool = False) -> pathlib.Path:
    """The min-projected lambda* field for one env.

    ``rate_only`` drops everything that is not the state-dependent rate itself —
    no rollouts, no x0 marginal, no connectors — and writes ``rate_<env>.svg``
    instead of ``minproj_<env>.svg``, so the field can be shown on its own.
    """
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
    ex, ey = _edges(gx), _edges(gy)
    traj, x0 = z["traj"], z["x0"]
    traj_ref = z["traj_ref"] if "traj_ref" in z.files else None
    k_show = 0 if rate_only else min(int(n_traj), traj.shape[0])
    shown = traj[:k_show]

    # Always put the dim lambda* varies along on the X axis, transposing the slice
    # when it is the second one. That keeps ONE layout for every env -- field on
    # top, marginal below over x, vertical connectors -- while still giving each
    # env the marginal that carries information. car_v1 is why: its lambda* is
    # flat in yaw and swings 2.78 across vel, so vel becomes its x-axis.
    which = axis or AXIS_OVERRIDE.get(env_name) or marginal_axis(Z)
    if which == "y":
        Z = Z.T
        dx, dy = dy, dx
        gx, gy = gy, gx
        ex, ey = _edges(gx), _edges(gy)
    md = dx                                   # the marginal is always over x now

    # The colorbar gets its OWN column spanning only the top row, so both panels
    # keep an identical width. Attaching it to the top axes instead (colorbar(ax=ax))
    # steals width from that axes alone, and then a dotted vertical at the same
    # data value lands at a different PIXEL in each panel -- which defeats the
    # entire point of the shared axis.
    if rate_only:
        # Same panel width and colorbar column as the two-panel figure, so a
        # rate-only plot can sit beside a full one without rescaling.
        fig = plt.figure(figsize=(10.6, 7.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        axd = None
        cax = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=(10.6, 9.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], width_ratios=[1.0, 0.035],
                              hspace=0.06, wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        axd = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[0, 1])
        ax.tick_params(labelbottom=False)
    which = "x"                               # one code path from here down

    # A constant field gets a NEUTRAL flat colour, not a point on viridis. Mapping
    # a single value through the colormap put car and quadrotor at the very bottom
    # of the scale, so their panels rendered dark purple -- which reads as "low
    # lambda*" beside the varying-field panels where purple genuinely is low.
    if _is_constant(Z):
        mesh = ax.pcolormesh(ex, ey, np.zeros_like(Z), cmap="Greys",
                             vmin=0.0, vmax=1.0, shading="flat")
    else:
        mesh = ax.pcolormesh(ex, ey, Z, cmap="viridis", shading="flat")
    # A constant field gets NO colorbar. Normalizing a colormap around a single
    # value invents a range (car showed 4.4-5.3 for a field that is 4.884
    # everywhere) and invites reading a gradient that does not exist. The value is
    # stated on the panel instead. The colorbar COLUMN is still reserved and just
    # switched off, so all five figures keep identical panel geometry and can be
    # laid side by side.
    is_const = _is_constant(Z)
    if is_const:
        cax.axis("off")
        ax.text(0.5, 0.045,
                rf"$\lambda^*(x) = {float(Z.mean()):.4g}$  everywhere",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_LABEL, color="0.10",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55",
                          alpha=0.92), zorder=8)
    else:
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(r"$\lambda^*(x)$", fontsize=FS_CBAR)
        cb.ax.tick_params(labelsize=FS_TICK)
        cb.formatter.set_useOffset(False)
        cb.update_ticks()

    for k in range(k_show):
        # The REFERENCE first, under the state: xref is what the task asks for and
        # x is what the plant does, so the pair is the object of interest. Under
        # reference_mode="contractive" they are visibly different -- the reference
        # migrates out of the low-rate region and x follows it.
        if traj_ref is not None:
            rx, ry = traj_ref[k, :, dx], traj_ref[k, :, dy]
            ax.plot(rx, ry, "--", color="#1f77b4", lw=LINEWIDTH * 0.8, alpha=0.95,
                    zorder=3, label=r"$x_{ref}(t)$" if k == 0 else None)
            ax.plot(rx[0], ry[0], "s", ms=9, mfc="white", mec="#12496f", mew=2.2,
                    zorder=7, label=r"$x_{ref}(0)$" if k == 0 else None)
            ax.plot(rx[-1], ry[-1], "*", ms=17, mfc="#1f77b4", mec="#12496f",
                    mew=1.0, zorder=8,
                    label=r"$x_{ref}(T)$" if k == 0 else None)
        tx, ty = traj[k, :, dx], traj[k, :, dy]
        ax.plot(tx, ty, "-", color="#d62728", lw=LINEWIDTH, alpha=0.92,
                zorder=4, label=r"$x(t)$" if k == 0 else None)
        _arrows(ax, tx, ty)
        ax.plot(tx[0], ty[0], "o", ms=11, mfc="white", mec="#7f0f14", mew=2.4,
                zorder=7, label=r"$x_0$" if k == 0 else None)

    # Axes cover the union of the field and what is drawn on it: clipping to the
    # box hides every trajectory start outside it (on car, about half of them).
    if k_show:
        _all = ([shown] if traj_ref is None else [shown, traj_ref[:k_show]])
        lo_y = min([ey[0]] + [float(np.min(a[:, :, dy])) for a in _all])
        hi_y = max([ey[-1]] + [float(np.max(a[:, :, dy])) for a in _all])
        lo_x = min([ex[0]] + [float(np.min(a[:, :, dx])) for a in _all])
        hi_x = max([ex[-1]] + [float(np.max(a[:, :, dx])) for a in _all])
    else:
        # Nothing drawn on top of the field, so the field IS the extent.
        lo_x, hi_x, lo_y, hi_y = ex[0], ex[-1], ey[0], ey[-1]
    pad_y, pad_x = 0.04 * (hi_y - lo_y), 0.02 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
    ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
    # Mark the certified box, so the blank margin is not mistaken for field.
    for yv in (ey[0], ey[-1]):
        ax.axhline(yv, color="white", lw=1.6, ls="--", alpha=0.75, zorder=5)

    ax.set_ylabel(names[dy], fontsize=FS_LABEL)
    if which == "y" or rate_only:
        ax.set_xlabel(names[dx], fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    # Env name only. The synthesis parameters (lbd, r, w_lb, w_ub) are a property
    # of the shipped config, not of what the panel shows, and they crowded the
    # one thing a reader needs to identify the figure by.
    ax.set_title(pretty(env_name), fontsize=FS_TITLE, pad=14)
    if k_show:
        ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.92)

    if rate_only:
        out = FIG_DIR / f"rate_{env_name}.svg"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[rate] wrote {out}  ({names[dx]} x {names[dy]}, lambda* spread "
              f"{float(Z.max() - Z.min()):.3g})")
        return out

    # ── the marginal, over the dim lambda* varies along ─────────────────────
    lim = ax.get_ylim() if which == "y" else ax.get_xlim()
    m = x0[:, md]
    keep = (m >= lim[0]) & (m <= lim[1])
    dens, centres = _marginal(m[keep], *lim)
    if which == "y":
        axd.fill_betweenx(centres, 0.0, dens, color="0.45", alpha=0.55, zorder=2)
        axd.plot(dens, centres, color="0.15", lw=2.0, zorder=3)
        axd.set_xlabel(rf"$x_0$ density ({names[md]})", fontsize=FS_LABEL - 2)
        axd.set_xlim(left=0.0)
        axd.set_ylim(*lim)
    else:
        axd.fill_between(centres, 0.0, dens, color="0.45", alpha=0.55, zorder=2)
        axd.plot(centres, dens, color="0.15", lw=2.0, zorder=3)
        axd.set_xlabel(names[md], fontsize=FS_LABEL)
        axd.set_ylabel(r"$x_0$ density", fontsize=FS_LABEL)
        axd.set_ylim(bottom=0.0)
        axd.set_xlim(*lim)
    axd.tick_params(labelsize=FS_TICK)

    # Connectors, oriented to match: a line from each shown rollout's start to
    # its place in the distribution.
    for k in range(k_show):
        start_val = traj[k, 0, md]
        if which == "y":
            for a in (ax, axd):
                a.axhline(start_val, ls=":", lw=1.9, color="#7f0f14", alpha=0.85,
                          zorder=6)
            axd.plot(0.0, start_val, "o", ms=9, mfc="white", mec="#7f0f14",
                     mew=2.2, zorder=7, clip_on=False)
        else:
            for a in (ax, axd):
                a.axvline(start_val, ls=":", lw=1.9, color="#7f0f14", alpha=0.85,
                          zorder=6)
            axd.plot(start_val, 0.0, "o", ms=9, mfc="white", mec="#7f0f14",
                     mew=2.2, zorder=7, clip_on=False)

    out_box = ((x0[:, dx] < ex[0]) | (x0[:, dx] > ex[-1])
               | (x0[:, dy] < ey[0]) | (x0[:, dy] > ey[-1]))
    frac = float(out_box.mean())
    if frac > 0:
        # Upper RIGHT: the marginal's left shoulder is where the interesting mass
        # sits on the weak-authority envs, and the note was landing on top of it.
        axd.text(0.985, 0.86, f"{frac:.0%} of $x_0$ outside the certified box",
                 transform=axd.transAxes, fontsize=FS_LEGEND - 3, color="#7f0f14",
                 ha="right", va="top")

    out = FIG_DIR / f"minproj_{env_name}.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[minproj] wrote {out}  ({names[dx]} x {names[dy]}, lambda* spread "
          f"{float(Z.max() - Z.min()):.3g}, marginal over {names[md]} "
          f"({which}-axis), {k_show} trajectories, {frac:.0%} of x0 outside box)")
    return out


def _marginal(u: np.ndarray, lo: float, hi: float, bins: int = 42):
    """Smoothed 1D histogram of x0 along the shared axis.

    A histogram rather than a KDE: no bandwidth to defend, and the reset draw is
    a uniform box perturbation, so the honest shape is flat-topped with real
    edges — a KDE would round exactly the corners that matter.
    """
    H, edges = np.histogram(u, bins=bins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    num = np.convolve(H, k, mode="same")
    # Same edge correction as the 2D case: convolve zero-pads, which would sag
    # the two ends and invent a taper the samples do not have.
    den = np.convolve(np.ones_like(H), k, mode="same")
    return num / np.where(den > 0, den, 1.0), centres


def _edges(c: np.ndarray) -> np.ndarray:
    """Cell centres -> edges, so pcolormesh colours the region each cell covers."""
    step = (c[-1] - c[0]) / max(len(c) - 1, 1)
    return np.concatenate([c - step / 2.0, [c[-1] + step / 2.0]])


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
    p.add_argument("--envs", default="car,car_v1,segway,cartpole,quadrotor",
                   help="comma-separated short env names")
    p.add_argument("--generate", action="store_true",
                   help="recompute the lambda* grid (expensive) before plotting")
    p.add_argument("--refresh-x0", "--refresh_x0", action="store_true",
                   help="redraw x0/traj from the current env, keeping the cached "
                        "lambda* grid — the cheap fix when reset() changed but the "
                        "plant and envelope did not")
    p.add_argument("--grid", type=int, default=15)
    p.add_argument("--n-other", "--n_other", type=int, default=8,
                   help="samples of the projected-out dims per cell (the min is "
                        "taken over these)")
    p.add_argument("--n-x0", "--n_x0", type=int, default=2000)
    p.add_argument("--n-traj", "--n_traj", type=int, default=10)
    p.add_argument("--show-traj", "--show_traj", type=int, default=N_TRAJ_SHOWN,
                   help="rollouts to draw (the npz may hold more)")
    p.add_argument("--marginal-axis", "--marginal_axis", choices=["x", "y"],
                   default=None,
                   help="force which axis the x0 marginal is taken over "
                        "(default: whichever one lambda* varies along)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-mode", "--reference_mode",
                   choices=["stabilizing", "contractive"], default="stabilizing",
                   help="contractive: the reference starts in the low-rate "
                        "region and migrates toward XREF_INIT_FAST")
    p.add_argument("--migrate-gain", "--migrate_gain", type=float, default=1.0)
    p.add_argument("--estimator", choices=["certified", "ceiling"],
                   default="certified",
                   help="certified: lam(x) of the shipped metric field (exact, "
                        "one SDP, invariant along the plant's symmetries). "
                        "ceiling: the old per-cell re-solved max rate.")
    p.add_argument("--rate-only", "--rate_only", action="store_true",
                   help="also write rate_<env>.svg: the state-dependent "
                        "contraction rate alone, no rollouts or x0 marginal")
    a = p.parse_args()

    envs = [e.strip() for e in a.envs.split(",") if e.strip()]
    for e in envs:
        if a.generate or not (DATA_DIR / f"minproj_{e}.npz").exists():
            generate(e, grid=a.grid, n_other=a.n_other, n_x0=a.n_x0,
                     n_traj=a.n_traj, seed=a.seed,
                     ref_mode=a.reference_mode, migrate_gain=a.migrate_gain, estimator=a.estimator)
        elif a.refresh_x0:
            refresh_x0(e, n_x0=a.n_x0, n_traj=a.n_traj, seed=a.seed,
                       ref_mode=a.reference_mode, migrate_gain=a.migrate_gain)
        plot(e, n_traj=a.show_traj, axis=a.marginal_axis)
        if a.rate_only:
            plot(e, axis=a.marginal_axis, rate_only=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

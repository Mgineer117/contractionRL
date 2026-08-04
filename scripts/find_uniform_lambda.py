"""Largest contraction rate lambda feasible along the whole CLOSED-LOOP tube.

``build_cm_dataset`` solves the CV-STEM SDP per sampled state and, where a state
is infeasible at the requested ``lbd``, HALVES lambda for that state
(``_solve_cm_metric_with_backoff``). The result is a metric whose certified rate
varies state to state, and whose worst case is whatever the least-feasible
sample happened to accept — the aggregate summary reports it after the fact
rather than choosing it.

This utility answers the design question instead: what is the largest lambda
such that the SDP is feasible at EVERY sampled ``(x, u)``? That is the rate a
uniform certificate can actually claim.

Samples come from ``n`` CLOSED-LOOP rollouts under the CV-STEM control itself
(``--num-trajs`` x ``--horizon``), so lambda is certified where the controller
actually operates and consecutive states are one real ``dt`` apart -- which is
what makes the ``W_prev_bar`` chaining a true material derivative. Uniform
sampling over the whole state box was tried and removed: it is a far stronger
claim than the closed loop needs, it spends solves on physically unreachable
corners, and it admitted no lambda at any envelope on cartpole / segway /
turtlebot / quadrotor.

Feasibility here means BOTH conditions, because either alone is misleading:

1. the LMI solves, and
2. the control that gain implies, ``u = uref - (1/r) B^T W^-1 e``, lands inside
   the env's own control box -- an exact per-sample test, since the tube gives a
   concrete ``(x, e, uref)`` at every step.

Without (2), lambda can be pushed arbitrarily high just by shrinking
``r_scaler`` — the resulting "certificate" is bang-bang saturation, not a
realizable gain. Measured on the car at ``r_scaler=0.01``: lambda* >= 4 with the
check off, 0.97 with it on.

The two failure modes need OPPOSITE corrections, so they are distinguished and
handled separately. An actuator violation GROWS ``r_scaler`` by
``--step-factor`` (default 1.5) and retries, since a larger r shrinks K; an LMI
infeasibility does not, since raising r removes control authority and only makes
the LMI harder.

lambda is searched on the mirror-image ladder, SHRINKING from ``--lbd-max`` by
the same factor. It is not a plain bisection because feasibility is NOT monotone
in lambda: ``solve_cm_metric`` weights its objective by ``chi_weight = 1/lbd``,
so a small lambda inflates that weight, yields a W with a larger gain, and can
fail the actuator check that a larger lambda passes (on the car at
``r_scaler=0.01``: infeasible at 0.05, feasible at 0.97). The ladder finds the
feasible band and a bisection then refines only its top edge.

``--jacobian`` selects which ``A`` enters the LMI. DEFAULT ``drift``
(``A = df/dx``) matches what ``build_cm_dataset`` actually solves and what this
repo's CV-STEM docstring specifies (control enters ONLY through the Riccati
penalty on ``B``), so the answer transfers to real synthesis. ``generalized``
(``+ sum_k u_k dB_k/dx``) is the differential Jacobian a u-dependent rate would
need, but it is NOT the program this repo solves -- its lambda* does not
transfer.

Example::

    python scripts/find_uniform_lambda.py --task classic-car-v0 \\
        --w-lb 0.1 --w-ub 10 --num-trajs 4 --horizon 100
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

from contractionRL.agents.skrl.math_utils import b_jacobian, jacobian  # noqa: E402
from contractionRL.agents.skrl.ncm_synthesis import solve_cm_metric  # noqa: E402


def expand_2x(lo, hi):
    """The control box as a 2x expansion of the reference box, SIGN-AWARE.

    "2x" means each bound moves twice as far FROM ZERO, which is a plain
    doubling only when the bound already points away from zero:

        lower bound < 0 -> 2*lo   (more negative = wider)
        lower bound > 0 -> lo/2   (closer to zero = wider; 2*lo would SHRINK it)
        upper bound > 0 -> 2*hi
        upper bound < 0 -> hi/2

    Multiplying every bound by 2 would narrow the box from any side whose bound
    sits on the far side of zero (a positive lower bound, a negative upper one),
    i.e. silently tighten the actuator budget instead of widening it.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.where(lo < 0, 2.0 * lo, lo / 2.0), np.where(hi > 0, 2.0 * hi, hi / 2.0)


def _boxes(env):
    """The env's own state/control boxes as numpy, whatever it calls them."""
    x_lo = env.X_MIN.detach().cpu().numpy()
    x_hi = env.X_MAX.detach().cpu().numpy()
    u_lo = env.U_MIN.detach().cpu().numpy()
    u_hi = env.U_MAX.detach().cpu().numpy()
    return x_lo, x_hi, u_lo, u_hi


def generalized_jacobians(env, x_np, u_np, device="cpu", mode="generalized"):
    """``A(x,u) = df/dx + sum_k u_k dB_k/dx`` and ``B(x)`` for each sample.

    Returns ``(A, B)`` with shapes ``(n, x_dim, x_dim)`` and ``(n, x_dim, u_dim)``.
    """
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).requires_grad_()
    with torch.enable_grad():
        f, B, _ = env.get_f_and_B(x)
        DfDx = jacobian(f, x, create_graph=False)                    # (n, x, x)
        DBDx = b_jacobian(B, x, B.shape[-1], create_graph=False)     # (n, x, x, u)
    if mode == "drift":
        # EXACTLY what build_cm_dataset feeds the LMI (ncm_synthesis.py: DfDx =
        # jacobian(f, x)). Tsukamoto's CV-STEM as implemented here uses the DRIFT
        # Jacobian; control enters only through the Riccati penalty on B. Use
        # this when the answer must transfer to actual synthesis -- note u then
        # does not enter the LMI at all, so sampling it changes nothing.
        A = DfDx
    else:
        # sum_k u_k * dB_k/dx — the term the drift-only Jacobian drops.
        u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
        A = DfDx + torch.einsum("nxyu,nu->nxy", DBDx, u)
    return (A.detach().cpu().numpy().astype(np.float64),
            B.detach().cpu().numpy().astype(np.float64))


SDP_INFEASIBLE = "sdp"
ACTUATOR_INFEASIBLE = "actuator"


def cvstem_control(W, B, e, uref, r_scaler):
    """The control CV-STEM's own metric implies: ``u = uref - R^-1 B^T M e``.

    ``M = W^-1`` is the metric and ``K = (1/r) B^T M`` the gain the SDP's Riccati
    term was written for (see ncm_synthesis's module docstring: "M(x) = W(x)^-1
    doubles as a state-dependent Riccati solution with gain K(x) = R^-1 B(x)^T
    M(x)"). Using THIS control -- rather than a uniform draw -- makes the
    (x, u, x_next) triple self-consistent: the successor state is where the
    certified closed loop actually goes, and the actuator check becomes an exact
    per-sample test on u instead of a Monte-Carlo bound on |K e|.
    """
    M = np.linalg.inv(W)
    K = (1.0 / r_scaler) * B.T @ M
    return uref - K @ e, K


def feasible_tube(env, *, lbd, w_lb, w_ub, eps, solver, r_scaler, ctl_lo, ctl_hi,
                  horizon, use_wdot, jac_mode, check_u=True, seed=0):
    """Feasibility along CLOSED-LOOP rollouts under the CV-STEM control.

    ``n`` reference trajectories (the env's own), each rolled out from
    ``x0 = xref0 + xe0`` with ``u_t = uref_t - (1/r) B^T M(x_t) e_t``, giving
    ``n x horizon`` samples drawn from the TUBE the controller actually induces.

    Consecutive states along a rollout are genuinely one ``dt`` apart, so the
    ``W_prev_bar`` chaining is the real material derivative here.

    Returns ``(ok, (traj, step) of first failure, reason)``.
    """
    torch.manual_seed(seed)
    env.reset()
    n, u_dim = env.num_envs, int(env.num_dim_control)
    x = env.x_t.clone()
    dt = float(env.dt)
    Wbar_prev = [None] * n
    horizon = min(int(horizon), int(env.max_episode_len) - 1)
    for t in range(horizon):
        xref_t = env.xref[:, t]
        uref_t = env.uref[:, t]
        A, B = generalized_jacobians(env, x.numpy().astype(np.float64),
                                     uref_t.numpy().astype(np.float64), mode=jac_mode)
        us = np.empty((n, u_dim), dtype=np.float64)
        for i in range(n):
            got = solve_cm_metric(
                A[i], B[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
                r_scaler=r_scaler, return_wbar=True,
                W_prev_bar=(Wbar_prev[i] if use_wdot else None),
                dt=(dt if use_wdot else 0.0))
            W, Wbar = got if isinstance(got, tuple) else (got, None)
            if W is None:
                return False, (i, t), SDP_INFEASIBLE
            Wbar_prev[i] = Wbar
            e = env.wrap_angles((x[i] - xref_t[i]).unsqueeze(0))[0].numpy().astype(np.float64)
            u, _ = cvstem_control(np.asarray(W, dtype=np.float64), B[i], e,
                                  uref_t[i].numpy().astype(np.float64), r_scaler)
            if check_u and np.any((u < ctl_lo - 1e-9) | (u > ctl_hi + 1e-9)):
                return False, (i, t), ACTUATOR_INFEASIBLE
            us[i] = u
        # Propagate with the env's own dynamics (mirrors env_base.step).
        with torch.no_grad():
            f_t, B_t, _ = env.get_f_and_B(x)
            xdot = f_t + torch.bmm(B_t, torch.as_tensor(
                us, dtype=torch.float32).unsqueeze(-1)).squeeze(-1)
        x = torch.clamp(env.wrap_angles(x + dt * xdot), env.X_MIN, env.X_MAX)
    return True, None, ""


def feasible_with_r_backoff(_checker, *, lbd, r_scaler, max_r_steps, step_factor=1.5,
                            log=print, **kw):
    """An ``r_scaler`` on the ladder ``r0 * s^k``, k in [-N, N], feasible EVERYWHERE.

    r is searched in BOTH directions because the two constraints push opposite
    ways, and neither bound alone defines the answer:

    * SDP infeasible      -> r must SHRINK. Smaller r enlarges the ``B R^-1 B^T``
      term, i.e. adds control authority, which is what makes the LMI solvable.
    * actuator saturated   -> r must GROW. Larger r shrinks ``K = (1/r) B^T W^-1``,
      pulling the feedback back inside the control box.

    So feasibility in r is a BAND: too small saturates the actuator, too large
    kills the LMI. Growing only (the earlier behaviour) reported envs as
    infeasible whenever their band lay BELOW the starting r -- cartpole, segway
    and quadrotor all did at w_lb=0.1.

    r must be uniform across the box (it defines one metric family), so a single
    r has to work at every sample. Returns ``(r_used, None)`` or ``(None, reason)``.
    """
    # Ascending ladder: low r first (LMI-friendly), high r last (actuator-friendly).
    ladder = sorted({r_scaler * (step_factor ** k)
                     for k in range(-max_r_steps, max_r_steps + 1)})
    saw_sdp = saw_act = False
    for r in ladder:
        ok, bad, reason = _checker(lbd=lbd, r_scaler=r, **kw)
        if ok:
            if r != r_scaler:
                log(f"      r_scaler {r_scaler:g} -> {r:g} "
                    f"(x{step_factor:g}^{round(math.log(r / r_scaler, step_factor))})")
            return r, None
        saw_sdp |= reason == SDP_INFEASIBLE
        saw_act |= reason == ACTUATOR_INFEASIBLE
    span = f"[{ladder[0]:.4g}, {ladder[-1]:.4g}]"
    if saw_sdp and saw_act:
        why = "no r works: small r saturates the actuator, large r kills the LMI"
    elif saw_sdp:
        why = "SDP infeasible across the whole r ladder (envelope too tight)"
    else:
        why = "actuator saturated across the whole r ladder"
    return None, f"{why} over r in {span}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="classic-car-v0")
    p.add_argument("--w-lb", "--w_lb", type=float, default=0.1)
    p.add_argument("--w-ub", "--w_ub", type=float, default=10.0)
    p.add_argument("--cm-eps", "--cm_eps", type=float, default=0.1)
    p.add_argument("--r-scaler", "--r_scaler", type=float, default=1.0)
    p.add_argument("--solver", default="MOSEK")
    p.add_argument("--max-r-steps", "--max_r_steps", "--max-r-doublings",
                   "--max_r_doublings", dest="max_r_steps", type=int, default=16,
                   help="on actuator saturation, grow r_scaler by --step-factor up to "
                        "this many times")
    p.add_argument("--step-factor", "--step_factor", type=float, default=1.5,
                   help="multiplicative ladder step: r_scaler GROWS by this on actuator "
                        "saturation, lambda SHRINKS by it (default 1.5; 2.0 = doubling)")
    p.add_argument("--no-actuator-check", "--no_actuator_check", action="store_true",
                   help="skip the control-bound check (lambda* becomes optimistic)")
    p.add_argument("--no-wdot", "--no_wdot", dest="wdot", action="store_false",
                   help="drop the material derivative: solve the STATIC LMI instead of "
                        "the (x,u,x_next) form. Optimistic -- Wdot must be dominated")
    p.add_argument("--auto-envelope", "--auto_envelope", action="store_true",
                   help="if infeasible, widen the envelope (w_lb /= step, w_ub *= step) "
                        "up to --max-envelope-steps times until a lambda exists")
    p.add_argument("--max-envelope-steps", "--max_envelope_steps", type=int, default=6)
    p.add_argument("--num-trajs", "--num_trajs", type=int, default=4,
                   help="n reference trajectories rolled out closed-loop")
    p.add_argument("--horizon", type=int, default=60,
                   help="steps per rollout (num_trajs*horizon samples)")
    p.add_argument("--jacobian", choices=("drift", "generalized"), default="drift",
                   help="drift: A = df/dx, EXACTLY what build_cm_dataset solves, so the "
                        "answer transfers to synthesis (u does not enter the LMI). "
                        "generalized: A = df/dx + sum_k u_k dB_k/dx -- physically the "
                        "right differential Jacobian, but NOT the program this repo "
                        "(or the CV-STEM reference) actually solves")
    p.add_argument("--lbd-max", "--lbd_max", type=float, default=4.0)
    p.add_argument("--tol", type=float, default=0.01, help="bisection tolerance on lambda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    import contractionRL.tasks.direct.classic  # noqa: F401  (registers the ids)
    import gymnasium as gym
    env = gym.make(args.task, num_envs=args.num_trajs, device="cpu").unwrapped
    x_lo, x_hi, u_lo, u_hi = _boxes(env)
    x_dim, u_dim = int(env.num_dim_x), int(env.num_dim_control)
    ur_lo = env.UREF_MIN.detach().cpu().numpy()
    ur_hi = env.UREF_MAX.detach().cpu().numpy()

    print(f"[uniform-lambda] task={args.task}  x_dim={x_dim} u_dim={u_dim}")
    print(f"[uniform-lambda] state box  lo={np.round(x_lo, 3)}  hi={np.round(x_hi, 3)}")
    print(f"[uniform-lambda] control box lo={np.round(u_lo, 3)} hi={np.round(u_hi, 3)}")
    n_samples = args.num_trajs * args.horizon
    print(f"[uniform-lambda] TUBE: {args.num_trajs} reference trajectories x "
          f"{args.horizon} steps = {n_samples} closed-loop samples "
          f"(u = uref - (1/r)B^T M e at every step)")
    print(f"[uniform-lambda] w_lb={args.w_lb} w_ub={args.w_ub} eps={args.cm_eps} "
          f"r_scaler={args.r_scaler} solver={args.solver} jacobian={args.jacobian}")
    if args.wdot:
        print(f"[uniform-lambda] Wdot ON: LMI at x_next carries "
              f"-Wdot ~ (Wbar(x) - Wbar(x_next))/dt, dt={float(env.dt):g}")
    else:
        print("[uniform-lambda] Wdot OFF (--no-wdot): static LMI at x only")

    # Actuator feasibility is EXACT: u_cvstem = uref - (1/r)B^T M e is a
    # concrete vector per sample, so it is tested against the control box
    # directly. No Monte-Carlo over error directions and no rho, and no need to
    # subtract uref into a "feedback budget" -- uref is already inside u.
    check_u = not args.no_actuator_check
    ctl_lo, ctl_hi = expand_2x(ur_lo, ur_hi)
    if not (np.allclose(ctl_lo, u_lo, atol=1e-6) and np.allclose(ctl_hi, u_hi, atol=1e-6)):
        print(f"[uniform-lambda] WARNING: env control box "
              f"[{np.round(u_lo, 3)}, {np.round(u_hi, 3)}] != sign-aware 2x expansion of "
              f"UREF [{np.round(ctl_lo, 3)}, {np.round(ctl_hi, 3)}]; using the expansion.")
    if check_u:
        print(f"[uniform-lambda] actuator check ON (exact): u = uref - (1/r)B^T M e "
              f"must lie in [{np.round(ctl_lo, 3)}, {np.round(ctl_hi, 3)}]")
    else:
        print("[uniform-lambda] actuator check OFF: lambda* will be optimistic")

    r_used = {"r": args.r_scaler}
    env_wlb = {"w_lb": args.w_lb, "w_ub": args.w_ub}

    def _check(lbd):
        common = dict(w_lb=env_wlb["w_lb"], w_ub=env_wlb["w_ub"], eps=args.cm_eps,
                      solver=args.solver, ctl_lo=ctl_lo, ctl_hi=ctl_hi,
                      horizon=args.horizon, use_wdot=args.wdot,
                      jac_mode=args.jacobian, check_u=check_u, seed=args.seed)
        r, why = feasible_with_r_backoff(
            lambda **kw: feasible_tube(env, **kw), lbd=lbd, r_scaler=args.r_scaler,
            max_r_steps=(args.max_r_steps if check_u else 0),
            step_factor=args.step_factor,
            log=lambda s: print(f"[uniform-lambda] {s}"), **common)
        if r is None:
            print(f"[uniform-lambda]   lambda={lbd:9.4f} -> infeasible ({why})")
            return False
        r_used["r"] = r
        print(f"[uniform-lambda]   lambda={lbd:9.4f} -> FEASIBLE everywhere "
              f"(r_scaler={r:g}, {n_samples} samples)")
        return True

    # Feasibility floor for the bisection's lower bound. NOT 1e-3, and not 0:
    # solve_cm_metric's default chi_weight is 1/lbd, so a tiny lambda blows that
    # weight up (1e3 at lambda=1e-3) and the optimizer returns a W with a much
    # larger gain K=(1/r)B^T W^-1. Under the actuator check that makes very small
    # lambda HARDER, not easier -- measured: the probe demanded r=3 at
    # lambda=1e-3 while lambda=0.5 cleared the box at r=1.5. A floor that small
    # therefore aborts on a spurious "envelope infeasible" verdict.
    LBD_FLOOR = 0.05

    # LADDER, shrinking lambda by /step_factor from the ceiling, mirroring the
    # r_scaler ladder that grows by *step_factor. A plain bisection would be
    # unsound: it assumes feasibility is monotone in lambda, and with the
    # actuator check it is NOT. chi_weight defaults to 1/lbd, so a small lambda
    # inflates that weight, the optimizer returns a W with a larger gain
    # K=(1/r)B^T W^-1, and the control box gets HARDER to satisfy. Measured on
    # the car at r_scaler=0.01: infeasible at lambda=0.05, feasible at 0.97.
    grid = []
    g = args.lbd_max
    while g > LBD_FLOOR:
        grid.append(g)
        g /= args.step_factor
    grid.append(LBD_FLOOR)
    grid.reverse()                                    # ascending, like the old scan
    print(f"[uniform-lambda] lambda ladder: {len(grid)} values from {args.lbd_max:g} "
          f"down to {LBD_FLOOR:g} by /{args.step_factor:g} ...")
    ok_grid = [v for v in grid if _check(v)]
    # ENLARGE THE ENVELOPE rather than reporting "infeasible" and stopping: a
    # tight {w_lb, w_ub} is a modelling choice, not a property of the plant, and
    # w_lb=0.1 admits no uniform lambda on cartpole/segway/turtlebot/quadrotor.
    # Widen symmetrically (w_lb down, w_ub up) until some lambda exists.
    if not ok_grid and args.auto_envelope:
        for k in range(1, args.max_envelope_steps + 1):
            env_wlb["w_lb"] = args.w_lb / (args.step_factor ** k)
            env_wlb["w_ub"] = args.w_ub * (args.step_factor ** k)
            print(f"[uniform-lambda] widening envelope (step {k}): "
                  f"w_lb={env_wlb['w_lb']:.4g} w_ub={env_wlb['w_ub']:.4g}")
            ok_grid = [v for v in grid if _check(v)]
            if ok_grid:
                break
    if not ok_grid:
        print("\n[uniform-lambda] RESULT: no feasible lambda anywhere on the ladder"
              + (f" (envelope widened to w_lb={env_wlb['w_lb']:.4g}, "
                 f"w_ub={env_wlb['w_ub']:.4g})." if args.auto_envelope else ".")
              + "\n  The {w_lb, w_ub, eps, r_scaler} envelope is infeasible over this box."
              + ("" if args.auto_envelope else
                 "\n  Re-run with --auto-envelope to widen w_lb/w_ub automatically.")
              + ("\n  If the failures above are the ACTUATOR box, raise --max-r-steps,\n"
                 "  or check the WARNING above about a one-sided feedback budget."
                 if check_u else ""))
        return 2

    lo = max(ok_grid)
    if min(ok_grid) > grid[0]:
        print(f"[uniform-lambda] NOTE: lambda={grid[0]:g} is infeasible while "
              f"lambda={min(ok_grid):.4g} is feasible — feasibility is non-monotone "
              f"here (chi_weight=1/lambda); lambda* is the top of the feasible band.")
    if lo >= grid[-1]:
        print(f"\n[uniform-lambda] RESULT: feasible at the search ceiling lambda={lo:g}"
              f" (r_scaler={r_used['r']:g}).\n"
              f"  The true maximum is HIGHER — re-run with a larger --lbd-max.")
        return 0

    # Bisect between the top feasible grid point and the next one up.
    hi = min(g for g in grid if g > lo)
    while hi - lo > args.tol:      # invariant: lo feasible, hi infeasible
        mid = 0.5 * (lo + hi)
        if _check(mid):
            lo = mid
        else:
            hi = mid

    # Re-run at the final lambda so the reported r_scaler is the one that
    # actually certified lambda*, not whatever the last (rejected) probe set.
    _check(lo)
    print(f"\n[uniform-lambda] RESULT: uniform contraction rate lambda* = {lo:.4f}"
          f"  at r_scaler = {r_used['r']:g}"
          f"  (w_lb={env_wlb['w_lb']:.4g}, w_ub={env_wlb['w_ub']:.4g}"
          f"{', Wdot ON' if args.wdot else ', Wdot OFF'})")
    print(f"  Largest lambda feasible at EVERY sampled (x, u) with "
          f"w_lb={env_wlb['w_lb']:.4g}, w_ub={env_wlb['w_ub']:.4g}"
          + (", and whose CV-STEM control u = uref - (1/r)B^T M e stays inside the "
             "control box at every sampled (x, e, uref)."
             if check_u else " (actuator check DISABLED)."))
    print(f"  Set `lbd: {lo:.3f}`, `cvstem_r_scaler: {r_used['r']:g}`, "
          f"`w_lb: {env_wlb['w_lb']:.4g}`, `w_ub: {env_wlb['w_ub']:.4g}` to synthesize "
          f"with no per-state backoff.\n"
          f"  NOTE: this is a sampled certificate, not a proof — it holds over the\n"
          f"  {n_samples} points tested. Raise "
          "--num-trajs/--horizon to tighten it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

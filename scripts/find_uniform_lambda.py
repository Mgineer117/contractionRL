"""Largest contraction rate lambda that is feasible over the WHOLE state/control box.

``build_cm_dataset`` solves the CV-STEM SDP per sampled state and, where a state
is infeasible at the requested ``lbd``, HALVES lambda for that state
(``_solve_cm_metric_with_backoff``). The result is a metric whose certified rate
varies state to state, and whose worst case is whatever the least-feasible
sample happened to accept — the aggregate summary reports it after the fact
rather than choosing it.

This utility answers the design question instead: what is the largest lambda
such that the SDP is feasible at EVERY sampled ``(x, u)``? That is the rate a
uniform certificate can actually claim.

Feasibility here means BOTH conditions, because either alone is misleading:

1. the LMI solves, and
2. the implied gain ``K = (1/r) B^T W^-1`` keeps the feedback ``K·e`` inside the
   env's own control box at ``‖e‖ = rho`` (95% of directions).

Without (2), lambda can be pushed arbitrarily high just by shrinking
``r_scaler`` — the resulting "certificate" is bang-bang saturation, not a
realizable gain. Measured on the car at ``r_scaler=0.01``: lambda* >= 4 with the
check off, 0.97 with it on.

The two failure modes need OPPOSITE corrections, so they are distinguished and
handled separately. An actuator violation DOUBLES ``r_scaler`` and retries
(larger r shrinks K); an LMI infeasibility does not, since raising r removes
control authority and only makes the LMI harder.

One consequence: feasibility is NOT monotone in lambda. ``solve_cm_metric``
weights its objective by ``chi_weight = 1/lbd``, so a small lambda inflates that
weight, yields a W with a larger gain, and can fail the actuator check that a
larger lambda passes. Pure bisection would be unsound, so the search does a
coarse scan first and bisects only inside the feasible band.

Two differences from ``build_cm_dataset`` are deliberate:

* States and controls are drawn UNIFORMLY from the env's own ``[X_MIN, X_MAX]``
  and ``[U_MIN, U_MAX]`` boxes, not from rollouts. A uniform claim has to be
  tested on the whole box, not on the on-policy tube.
* The LMI uses the GENERALIZED Jacobian ``A(x,u) = df/dx + sum_k u_k dB_k/dx``.
  With the drift-only Jacobian the control never enters the LMI at all, so
  sampling ``u`` would be pure waste and the certificate would ignore exactly
  the term that makes a control-affine system's contraction rate u-dependent.

Sampling is ``n`` states x ``m`` controls = ``n*m`` LMI points.
``--u-mode uniform`` (default) draws the ``m`` controls INDEPENDENTLY for each
state, so the pairs cover the product box rather than a shared lattice of ``m``
control values. ``--u-mode vertices`` instead evaluates the ``2^u_dim`` u-box
CORNERS at every state: since the ``sum_k u_k dB_k/dx`` term is linear in ``u``
it attains its extremes there, so that mode is the one that certifies the whole
box, while ``uniform`` gives a cheaper interior estimate that can be optimistic.

Example::

    python scripts/find_uniform_lambda.py --task classic-car-v0 \\
        --w-lb 0.1 --w-ub 10 --num-states 300 --num-controls 8
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

from contractionRL.agents.skrl.math_utils import b_jacobian, jacobian  # noqa: E402
from contractionRL.agents.skrl.ncm_synthesis import solve_cm_metric  # noqa: E402


def _boxes(env):
    """The env's own state/control boxes as numpy, whatever it calls them."""
    x_lo = env.X_MIN.detach().cpu().numpy()
    x_hi = env.X_MAX.detach().cpu().numpy()
    u_lo = env.U_MIN.detach().cpu().numpy()
    u_hi = env.U_MAX.detach().cpu().numpy()
    return x_lo, x_hi, u_lo, u_hi


def generalized_jacobians(env, x_np, u_np, device="cpu"):
    """``A(x,u) = df/dx + sum_k u_k dB_k/dx`` and ``B(x)`` for each sample.

    Returns ``(A, B)`` with shapes ``(n, x_dim, x_dim)`` and ``(n, x_dim, u_dim)``.
    """
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).requires_grad_()
    with torch.enable_grad():
        f, B, _ = env.get_f_and_B(x)
        DfDx = jacobian(f, x, create_graph=False)                    # (n, x, x)
        DBDx = b_jacobian(B, x, B.shape[-1], create_graph=False)     # (n, x, x, u)
    u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
    # sum_k u_k * dB_k/dx — the term the drift-only Jacobian drops.
    A = DfDx + torch.einsum("nxyu,nu->nxy", DBDx, u)
    return (A.detach().cpu().numpy().astype(np.float64),
            B.detach().cpu().numpy().astype(np.float64))


SDP_INFEASIBLE = "sdp"
ACTUATOR_INFEASIBLE = "actuator"


def feasible_everywhere(A, B, *, lbd, w_lb, w_ub, eps, solver, r_scaler,
                        u_lo=None, u_hi=None, rho=0.0):
    """Is every sample feasible at this ``(lambda, r_scaler)``?

    Returns ``(ok, first_failure_index, reason)``. ``reason`` distinguishes the
    two failure modes, which need OPPOSITE corrections:

    * ``SDP_INFEASIBLE``      — the LMI itself has no solution. Raising r_scaler
      SHRINKS the ``B R^-1 B^T`` term, i.e. removes control authority, so it
      makes this strictly worse. Only a smaller lambda (or a wider envelope) helps.
    * ``ACTUATOR_INFEASIBLE`` — the LMI solves but the implied gain
      ``K = (1/r) B^T W^-1`` saturates the actuator box. Raising r_scaler shrinks
      K, so THIS is the one r-doubling fixes.

    Bails at the first failure: one is enough to disqualify the pair.
    """
    check_u = rho > 0 and u_lo is not None and u_hi is not None
    for i in range(A.shape[0]):
        W = solve_cm_metric(
            A[i], B[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps,
            solver=solver, r_scaler=r_scaler,
            u_lo=u_lo, u_hi=u_hi, rho=(rho if check_u else 0.0),
        )
        if W is not None:
            continue
        if not check_u:
            return False, i, SDP_INFEASIBLE
        # Re-solve WITHOUT the actuator check to tell the modes apart. Only on
        # failure, so the common path still costs one solve.
        W_plain = solve_cm_metric(
            A[i], B[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps,
            solver=solver, r_scaler=r_scaler,
        )
        return False, i, (ACTUATOR_INFEASIBLE if W_plain is not None else SDP_INFEASIBLE)
    return True, -1, ""


def feasible_with_r_backoff(A, B, *, lbd, r_scaler, max_doublings, log=print, **kw):
    """Smallest ``r_scaler`` in ``{r, 2r, 4r, ...}`` making EVERY sample feasible.

    r must be uniform across the box (it defines one metric family), so this
    searches for a single r that works everywhere rather than letting each state
    pick its own. Returns ``(r_used, None)`` on success, ``(None, reason)`` on
    failure. An SDP-infeasible verdict aborts immediately: doubling r only
    removes control authority, so it cannot recover.
    """
    r = r_scaler
    for k in range(max_doublings + 1):
        ok, bad, reason = feasible_everywhere(A, B, lbd=lbd, r_scaler=r, **kw)
        if ok:
            if k:
                log(f"      r_scaler {r_scaler:g} -> {r:g} ({k} doubling(s)) to clear the actuator box")
            return r, None
        if reason == SDP_INFEASIBLE:
            return None, f"SDP infeasible at sample {bad} (r={r:g}); raising r cannot help"
        r *= 2.0
    return None, (f"actuator box still saturated after {max_doublings} doublings "
                  f"(r={r / 2:g})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="classic-car-v0")
    p.add_argument("--num-states", "--num_states", "--num-samples", "--num_samples",
                   dest="num_states", type=int, default=300,
                   help="n: uniform STATE samples")
    p.add_argument("--num-controls", "--num_controls", dest="num_controls",
                   type=int, default=8,
                   help="m: uniform CONTROL samples drawn independently PER state "
                        "(--u-mode uniform); total LMI points = n*m")
    p.add_argument("--w-lb", "--w_lb", type=float, default=0.1)
    p.add_argument("--w-ub", "--w_ub", type=float, default=10.0)
    p.add_argument("--cm-eps", "--cm_eps", type=float, default=0.1)
    p.add_argument("--r-scaler", "--r_scaler", type=float, default=1.0)
    p.add_argument("--solver", default="MOSEK")
    p.add_argument("--u-mode", "--u_mode", choices=("uniform", "vertices"),
                   default="uniform",
                   help="uniform: m independent controls per state (n*m points). "
                        "vertices: the 2^u_dim u-box corners per state — the "
                        "worst case for a u-linear term, so it certifies the box")
    p.add_argument("--rho", type=float, default=None,
                   help="error magnitude ‖e‖ at which the implied feedback K·e is "
                        "checked against the control box (default: ‖XE_MAX‖)")
    p.add_argument("--max-r-doublings", "--max_r_doublings", type=int, default=8,
                   help="on actuator saturation, double r_scaler up to this many times")
    p.add_argument("--no-actuator-check", "--no_actuator_check", action="store_true",
                   help="skip the control-bound check (lambda* becomes optimistic)")
    p.add_argument("--lbd-max", "--lbd_max", type=float, default=4.0)
    p.add_argument("--scan-points", "--scan_points", type=int, default=9,
                   help="coarse-scan resolution before bisecting (feasibility is "
                        "non-monotone in lambda once the actuator check is on)")
    p.add_argument("--tol", type=float, default=0.01, help="bisection tolerance on lambda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    import contractionRL.tasks.direct.classic  # noqa: F401  (registers the ids)
    import gymnasium as gym
    env = gym.make(args.task, num_envs=1, device="cpu").unwrapped
    x_lo, x_hi, u_lo, u_hi = _boxes(env)
    x_dim, u_dim = int(env.num_dim_x), int(env.num_dim_control)

    # n uniform states, and for EACH of them m controls -> n*m LMI points.
    n = args.num_states
    xs_u = rng.uniform(x_lo, x_hi, size=(n, x_dim))
    if args.u_mode == "vertices":
        # The u-box corners: identical set per state by construction, since a
        # term linear in u attains its extremes there.
        per_state = np.array(list(itertools.product(*zip(u_lo, u_hi))), dtype=np.float64)
        m = per_state.shape[0]
        us = np.tile(per_state, (n, 1))
    else:
        # INDEPENDENT draws per state, not one shared set tiled n times: reusing
        # the same m controls everywhere would test a lattice of m control
        # values rather than the control box.
        m = args.num_controls
        us = rng.uniform(u_lo, u_hi, size=(n, m, u_dim)).reshape(n * m, u_dim)
    xs = np.repeat(xs_u, m, axis=0)

    print(f"[uniform-lambda] task={args.task}  x_dim={x_dim} u_dim={u_dim}")
    print(f"[uniform-lambda] state box  lo={np.round(x_lo, 3)}  hi={np.round(x_hi, 3)}")
    print(f"[uniform-lambda] control box lo={np.round(u_lo, 3)} hi={np.round(u_hi, 3)}")
    print(f"[uniform-lambda] n={n} states x m={m} controls ({args.u_mode}) "
          f"= {xs.shape[0]} LMI points")
    print(f"[uniform-lambda] w_lb={args.w_lb} w_ub={args.w_ub} eps={args.cm_eps} "
          f"r_scaler={args.r_scaler} solver={args.solver}")

    A, B = generalized_jacobians(env, xs, us)

    # Actuator feasibility. The certified gain is K = (1/r)·B^T·W^-1 and the
    # applied feedback is K·e, so a metric is only usable if that fits the env's
    # OWN control box at a realistic error magnitude. Without this, lambda can be
    # pushed arbitrarily high by shrinking r — which is exactly the bang-bang
    # saturation that made an earlier "best" (lbd=2, r=0.01) result meaningless.
    rho = args.rho if args.rho is not None else float(np.linalg.norm(
        env.XE_MAX.detach().cpu().numpy()))
    check_u = not args.no_actuator_check
    if check_u:
        print(f"[uniform-lambda] actuator check ON: |K·e| must sit in "
              f"[{np.round(u_lo, 3)}, {np.round(u_hi, 3)}] at ‖e‖=rho={rho:.3f} "
              f"(95% of directions); r_scaler doubles up to {args.max_r_doublings}x on violation")
    else:
        print("[uniform-lambda] actuator check OFF (--no-actuator-check): lambda* will be "
              "optimistic — a feasible metric may still saturate the actuator")

    common = dict(w_lb=args.w_lb, w_ub=args.w_ub, eps=args.cm_eps, solver=args.solver,
                  u_lo=(u_lo if check_u else None), u_hi=(u_hi if check_u else None),
                  rho=(rho if check_u else 0.0))
    r_used = {"r": args.r_scaler}

    def _check(lbd):
        r, why = feasible_with_r_backoff(
            A, B, lbd=lbd, r_scaler=args.r_scaler,
            max_doublings=(args.max_r_doublings if check_u else 0),
            log=lambda s: print(f"[uniform-lambda] {s}"), **common)
        if r is None:
            print(f"[uniform-lambda]   lambda={lbd:9.4f} -> infeasible ({why})")
            return False
        r_used["r"] = r
        print(f"[uniform-lambda]   lambda={lbd:9.4f} -> FEASIBLE everywhere "
              f"(r_scaler={r:g}, {xs.shape[0]} samples)")
        return True

    # Feasibility floor for the bisection's lower bound. NOT 1e-3, and not 0:
    # solve_cm_metric's default chi_weight is 1/lbd, so a tiny lambda blows that
    # weight up (1e3 at lambda=1e-3) and the optimizer returns a W with a much
    # larger gain K=(1/r)B^T W^-1. Under the actuator check that makes very small
    # lambda HARDER, not easier -- measured: the probe demanded r=3 at
    # lambda=1e-3 while lambda=0.5 cleared the box at r=1.5. A floor that small
    # therefore aborts on a spurious "envelope infeasible" verdict.
    LBD_FLOOR = 0.05

    # COARSE SCAN before bisecting. Pure bisection assumes feasibility is
    # monotone in lambda, and with the actuator check it is NOT: chi_weight
    # defaults to 1/lbd, so a small lambda inflates that weight, the optimizer
    # returns a W with a larger gain K=(1/r)B^T W^-1, and the control box is
    # HARDER to satisfy. Measured on the car at r_scaler=0.01: infeasible at
    # lambda=0.05, feasible at lambda=0.97. Probing a single low lambda as "the
    # easy case" would have reported the envelope as unusable.
    grid = list(np.linspace(LBD_FLOOR, args.lbd_max, args.scan_points))
    print(f"[uniform-lambda] coarse scan over {len(grid)} lambdas in "
          f"[{LBD_FLOOR:g}, {args.lbd_max:g}] ...")
    ok_grid = [g for g in grid if _check(g)]
    if not ok_grid:
        print("\n[uniform-lambda] RESULT: no feasible lambda anywhere on the scan.\n"
              "  The {w_lb, w_ub, eps, r_scaler} envelope itself is infeasible over this\n"
              "  box. Widen w_ub / lower w_lb / lower cm_eps rather than lowering lambda."
              + ("\n  If the failures above are the ACTUATOR box, raise --max-r-doublings:\n"
                 "  a very small starting r_scaler needs many doublings to reach a gain\n"
                 "  that fits (r=0.01 needed 6 to clear it on the car)."
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
          f"  at r_scaler = {r_used['r']:g}")
    print(f"  Largest lambda feasible at EVERY sampled (x, u) with w_lb={args.w_lb}, "
          f"w_ub={args.w_ub}"
          + (f", and whose gain K=(1/r)·B^T·W^-1 keeps |K·e| inside the control box "
             f"at ‖e‖={rho:.3f}." if check_u else " (actuator check DISABLED)."))
    print(f"  Set `lbd: {lo:.3f}` and `cvstem_r_scaler: {r_used['r']:g}` to synthesize "
          f"with no per-state backoff.\n"
          f"  NOTE: this is a sampled certificate, not a proof — it holds over the\n"
          f"  {xs.shape[0]} points tested. Raise --num-states/--num-controls to tighten it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

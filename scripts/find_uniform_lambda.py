"""Largest contraction rate lambda that is feasible over the WHOLE state/control box.

``build_cm_dataset`` solves the CV-STEM SDP per sampled state and, where a state
is infeasible at the requested ``lbd``, HALVES lambda for that state
(``_solve_cm_metric_with_backoff``). The result is a metric whose certified rate
varies state to state, and whose worst case is whatever the least-feasible
sample happened to accept — the aggregate summary reports it after the fact
rather than choosing it.

This utility answers the design question instead: what is the largest lambda
such that the SDP is feasible at EVERY sampled ``(x, u)``? That is the rate a
uniform certificate can actually claim. It bisects lambda and, at each
candidate, requires every sample to solve with NO backoff.

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


def feasible_everywhere(A, B, *, lbd, w_lb, w_ub, eps, solver, r_scaler, verbose=False):
    """True iff the SDP solves at EVERY sample at this lambda (no backoff).

    Returns ``(ok, n_checked, first_failure_index)``. Bails at the first
    infeasible sample: one failure already disqualifies lambda, and rejection is
    the common case during bisection.
    """
    for i in range(A.shape[0]):
        W = solve_cm_metric(
            A[i], B[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps,
            solver=solver, r_scaler=r_scaler,
        )
        if W is None:
            return False, i + 1, i
    return True, A.shape[0], -1


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
    p.add_argument("--lbd-max", "--lbd_max", type=float, default=4.0)
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

    def _check(lbd):
        ok, n, bad = feasible_everywhere(
            A, B, lbd=lbd, w_lb=args.w_lb, w_ub=args.w_ub, eps=args.cm_eps,
            solver=args.solver, r_scaler=args.r_scaler)
        status = "FEASIBLE everywhere" if ok else f"infeasible (first fail at sample {bad})"
        print(f"[uniform-lambda]   lambda={lbd:9.4f} -> {status}  [{n} solves]")
        return ok

    # Near-zero lambda is the trivially easiest case; if even that fails, the
    # envelope (w_lb/w_ub/eps/r_scaler) is wrong and no lambda will work. Say so
    # plainly instead of returning 0 as if it were a real answer.
    # Not exactly 0: solve_cm_metric's default chi_weight is 1/lbd.
    LBD_FLOOR = 1e-3
    print(f"[uniform-lambda] probing the envelope at lambda={LBD_FLOOR:g} ...")
    if not _check(LBD_FLOOR):
        print(f"\n[uniform-lambda] RESULT: infeasible even at lambda={LBD_FLOOR:g}.\n"
              "  The {w_lb, w_ub, eps, r_scaler} envelope itself is infeasible over this\n"
              "  box, so no uniform contraction rate exists here. Widen w_ub / lower w_lb\n"
              "  / lower cm_eps rather than lowering lambda.")
        return 2

    hi = args.lbd_max
    if _check(hi):
        print(f"\n[uniform-lambda] RESULT: feasible at the search ceiling lambda={hi:g}.\n"
              f"  The true maximum is HIGHER — re-run with a larger --lbd-max.")
        return 0

    lo = LBD_FLOOR                 # known feasible
    while hi - lo > args.tol:      # invariant: lo feasible, hi infeasible
        mid = 0.5 * (lo + hi)
        if _check(mid):
            lo = mid
        else:
            hi = mid

    print(f"\n[uniform-lambda] RESULT: uniform contraction rate lambda* = {lo:.4f}")
    print(f"  Largest lambda feasible at EVERY sampled (x, u) with w_lb={args.w_lb}, "
          f"w_ub={args.w_ub}.")
    print(f"  Set `lbd: {lo:.3f}` to synthesize with no per-state backoff.\n"
          f"  NOTE: this is a sampled certificate, not a proof — it holds over the\n"
          f"  {xs.shape[0]} points tested. Raise --num-states/--num-controls to tighten it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

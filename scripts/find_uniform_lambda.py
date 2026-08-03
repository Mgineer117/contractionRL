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

Two differences from ``build_cm_dataset`` are deliberate:

* States and controls are drawn UNIFORMLY from the env's own ``[X_MIN, X_MAX]``
  and ``[U_MIN, U_MAX]`` boxes, not from rollouts. A uniform claim has to be
  tested on the whole box, not on the on-policy tube.
* ``--jacobian`` selects which ``A`` enters the LMI. DEFAULT ``drift``
  (``A = df/dx``) matches what ``build_cm_dataset`` actually solves and what
  this repo's CV-STEM docstring specifies (control enters ONLY through the
  Riccati penalty on ``B``), so the answer transfers to real synthesis. Note
  that under ``drift`` the control does NOT enter the LMI, so the m control
  samples per state have no effect on lambda*. ``generalized``
  (``+ sum_k u_k dB_k/dx``) is the differential Jacobian a u-dependent rate
  would need, but it is NOT the program this repo solves -- its lambda* does
  not transfer.

Sampling is ``n`` states x ``m`` controls = ``n*m`` LMI points.
``--u-mode uniform`` (default) draws the ``m`` controls INDEPENDENTLY for each
state, so the pairs cover the product box rather than a shared lattice of ``m``
control values. ``--u-mode vertices`` instead evaluates the ``2^u_dim`` u-box
CORNERS at every state: since the ``sum_k u_k dB_k/dx`` term is linear in ``u``
it attains its extremes there, so that mode is the one that certifies the whole
box, while ``uniform`` gives a cheaper interior estimate that can be optimistic.

The tube (``--sampling tube``, the default)
------------------------------------------
``n`` reference trajectories x ``m`` initial errors on the ``rho``-shell x ``p``
rollouts under ``u = uref - K_lqr(xref) e + sigma*xi``. The gaussian ``sigma``
is what makes it a TUBE rather than a bundle of measure-zero sample paths; set
it to the POLICY's exploration std so the region certified is the region
training visits.

Three properties this construction buys, each fixing a real defect:

* the tube is generated under a NOMINAL LQR, so it is independent of
  ``(lbd, r_scaler, w_lb, w_ub)`` and can be built ONCE and reused by every
  probe. Generating it under CV-STEM instead put an SDP inside the rollout loop
  (12.6 ms x n*m*p*horizon = 6.7 h per ladder) AND made the sample set move with
  the controller under test.
* ``rho`` is an OUTPUT, searched downward, so the result is a ``(lambda*, rho*)``
  pair: the rate and the neighbourhood it holds on. Pinning ``rho`` at the whole
  ``XE_INIT`` box is why segway reported no feasible lambda at all -- that box's
  extremes are initial conditions that are already falling.
* feasibility is a violation RATE over a subsample, not bail-on-first-failure.
  Noise deliberately pushes the tube into its extremes, so an all-or-nothing
  gate gets STRICTLY stricter as the tube is made more realistic. This matches
  ``cvstem_lqr.py``, which reports a within-box rate rather than demanding zero.

Example::

    python scripts/find_uniform_lambda.py --task classic-car-v0 \\
        --w-lb 0.1 --w-ub 10 --num-states 300 --num-controls 8

    # tube: segway, noise at the policy's own exploration std
    python scripts/find_uniform_lambda.py --task classic-segway-v0 \\
        --num-trajs 1 --num-ics 8 --num-noisy 4 --sigma 1.0 --horizon 200
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


def feasible_everywhere_cvstem(A, B, xs, es, urefs, *, lbd, w_lb, w_ub, eps, solver,
                               r_scaler, ctl_lo, ctl_hi, env, dt, use_wdot,
                               jac_mode, check_u=True):
    """Feasibility over ``(x, u_cvstem, x_next)`` triples.

    Two batched passes, because ``x_next`` depends on ``u_cvstem``, which depends
    on ``W(x)``, which is the SDP solution -- so the successor Jacobians cannot
    be precomputed:

      1. solve the SDP at every ``x``  -> ``W(x)``, ``W̄(x)``
         then ``u = uref - (1/r) B^T W^-1 e`` and ``x_next = x + dt*(f + B u)``
      2. batch the Jacobians at ``x_next`` and re-solve there, carrying
         ``W̄(x)`` as ``W_prev_bar`` so the LMI includes ``-Ẇ``.

    Returns ``(ok, first_failure_index, reason)``.
    """
    n = A.shape[0]
    Ws, Wbars = [], []
    for i in range(n):
        got = solve_cm_metric(A[i], B[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps,
                              solver=solver, r_scaler=r_scaler, return_wbar=True)
        W, Wbar = got if isinstance(got, tuple) else (got, None)
        if W is None:
            return False, i, SDP_INFEASIBLE
        Ws.append(np.asarray(W, dtype=np.float64))
        Wbars.append(Wbar)

    # u_cvstem per sample, then the EXACT actuator test on the applied control.
    us = np.empty((n, B.shape[2]), dtype=np.float64)
    for i in range(n):
        us[i], _ = cvstem_control(Ws[i], B[i], es[i], urefs[i], r_scaler)
    if check_u:
        bad = np.where(np.any((us < ctl_lo - 1e-9) | (us > ctl_hi + 1e-9), axis=1))[0]
        if bad.size:
            return False, int(bad[0]), ACTUATOR_INFEASIBLE
    if not use_wdot:
        return True, -1, ""

    # x_next under the CERTIFIED control, then the Wdot-inclusive LMI there.
    x_t = torch.as_tensor(xs, dtype=torch.float32)
    u_t = torch.as_tensor(us, dtype=torch.float32)
    with torch.no_grad():
        f_t, B_t, _ = env.get_f_and_B(x_t)
        xdot = f_t + torch.bmm(B_t, u_t.unsqueeze(-1)).squeeze(-1)
    x_next = torch.clamp(env.wrap_angles(x_t + dt * xdot), env.X_MIN, env.X_MAX)
    A_n, B_n = generalized_jacobians(env, x_next.numpy().astype(np.float64), us,
                                     mode=jac_mode)
    for i in range(n):
        W2 = solve_cm_metric(A_n[i], B_n[i], lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps,
                             solver=solver, r_scaler=r_scaler,
                             W_prev_bar=Wbars[i], dt=dt)
        if W2 is None:
            return False, i, SDP_INFEASIBLE
    return True, -1, ""


def feasible_everywhere(A, B, *, lbd, w_lb, w_ub, eps, solver, r_scaler,
                        u_lo=None, u_hi=None, rho=0.0,
                        A_next=None, B_next=None, dt=0.0):
    """Is every sample feasible at this ``(lambda, r_scaler)``?

    Returns ``(ok, first_failure_index, reason)``. ``reason`` distinguishes the
    two failure modes, which need OPPOSITE corrections:

    * ``SDP_INFEASIBLE``      — the LMI itself has no solution. Raising r_scaler
      SHRINKS the ``B R^-1 B^T`` term, i.e. removes control authority, so it
      makes this strictly worse. Only a smaller lambda (or a wider envelope) helps.
    * ``ACTUATOR_INFEASIBLE`` — the LMI solves but the implied gain
      ``K = (1/r) B^T W^-1`` saturates the actuator box. Raising r_scaler shrinks
      K, so THIS is the one r-doubling fixes.

    With ``A_next``/``B_next``/``dt`` the check uses the (x, u, x_next) TRIPLE and
    includes the material derivative: solve at x for W̄(x), then require the LMI
    at x_next to hold WITH ``-Ẇ ≈ (W̄(x) - W̄(x_next))/dt`` folded in
    (``_add_wdot_term``). Without it the static LMI omits Ẇ entirely, which is
    not the contraction condition and is optimistic — Ẇ is a term the rest of
    the LMI has to dominate.

    Bails at the first failure: one is enough to disqualify the pair.
    """
    check_u = rho > 0 and u_lo is not None and u_hi is not None
    use_wdot = A_next is not None and dt > 0
    kw = dict(w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver, r_scaler=r_scaler)
    for i in range(A.shape[0]):
        if use_wdot:
            # Step 1: metric at x. Needs W̄ (normalized) to difference against.
            got = solve_cm_metric(A[i], B[i], lbd=lbd, return_wbar=True, **kw)
            W_prev, Wbar_prev = got if isinstance(got, tuple) else (got, None)
            if W_prev is None:
                return False, i, SDP_INFEASIBLE
            # Step 2: metric at x_next, with -Ẇ = (W̄_prev - W̄)/dt in the LMI.
            W = solve_cm_metric(
                A_next[i], B_next[i], lbd=lbd, W_prev_bar=Wbar_prev, dt=dt,
                u_lo=u_lo, u_hi=u_hi, rho=(rho if check_u else 0.0), **kw)
        else:
            W = solve_cm_metric(
                A[i], B[i], lbd=lbd,
                u_lo=u_lo, u_hi=u_hi, rho=(rho if check_u else 0.0), **kw)
        if W is not None:
            continue
        if not check_u:
            return False, i, SDP_INFEASIBLE
        # Re-solve WITHOUT the actuator check to tell the modes apart. Only on
        # failure, so the common path still costs one solve.
        if use_wdot:
            W_plain = solve_cm_metric(A_next[i], B_next[i], lbd=lbd,
                                      W_prev_bar=Wbar_prev, dt=dt, **kw)
        else:
            W_plain = solve_cm_metric(A[i], B[i], lbd=lbd, **kw)
        return False, i, (ACTUATOR_INFEASIBLE if W_plain is not None else SDP_INFEASIBLE)
    return True, -1, ""


def shell_errors(env, rho, n, gen):
    """``n`` errors on the ``rho``-shell of the ``XE_INIT`` ellipsoid.

    ``rho`` is DIMENSIONLESS: the error is ``rho * (v/|v|) * halfwidth`` with
    ``v`` isotropic gaussian, so ``rho=1`` is the ellipsoid inscribed in the
    env's own ``XE_INIT`` box and ``rho=0.5`` is half of it. Scaling by the box
    half-widths is what makes rho comparable across dimensions whose units
    differ by an order of magnitude (segway: pos +-1 m vs pitch_rate +-pi rad/s).

    The SHELL, not the ball: both the actuator check and the LMI are worst at
    the largest error, so interior draws only dilute the sample.
    """
    scale = 0.5 * (env.XE_INIT_MAX - env.XE_INIT_MIN)
    v = torch.randn(n, int(env.num_dim_x), generator=gen)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rho * v * scale


def lqr_reference_gains(env, xref, uref, jac_mode, r_lqr=0.1):
    """Nominal LQR gain at each reference state: ``K = R^-1 B^T P``, P from the CARE.

    The tube is generated under this NOMINAL controller, not under CV-STEM, for
    two reasons:

    * it removes the SDP from the rollout loop entirely (12.6 ms/solve x
      n*m*p*horizon states was 6.7 h for a full ladder), and
    * it makes the tube INDEPENDENT of (lbd, r_scaler, w_lb, w_ub), which is
      what licenses generating it ONCE and reusing it for every probe. A tube
      generated by the very controller under test changes with every probe and
      is part of why feasibility looked non-monotone.

    Mirrors ``sdlqr.py``'s state-dependent LQR (same ``solve_continuous_are``).
    """
    from scipy.linalg import solve_continuous_are
    n, T = xref.shape[0], xref.shape[1]
    x_dim, u_dim = int(env.num_dim_x), int(env.num_dim_control)
    K = np.zeros((n, T, u_dim, x_dim))
    for t in range(T):
        A, B = generalized_jacobians(env, xref[:, t].numpy().astype(np.float64),
                                     uref[:, t].numpy().astype(np.float64), mode=jac_mode)
        for i in range(n):
            P = solve_continuous_are(A[i], B[i], np.eye(x_dim), np.eye(u_dim) * r_lqr)
            K[i, t] = (P @ B[i] / r_lqr).T
    return K


def build_tube(env, *, rho, num_ics, num_noisy, horizon, sigma, jac_mode, seed=0):
    """Generate the tube ONCE: ``n`` refs x ``m`` ICs x ``p`` noisy rollouts.

    ``n = env.num_envs`` reference trajectories; from each, ``m`` initial errors
    on the ``rho``-shell; from each of those, ``p`` rollouts under

        u_t = uref_t - K_lqr(xref_t) e_t + sigma * xi_t,   xi_t ~ N(0, I)

    The gaussian term is what makes this a TUBE rather than a bundle of
    measure-zero sample paths: it fills the neighbourhood the stochastic policy
    actually explores. ``sigma`` should be the policy's own exploration std, so
    the region certified is the region training visits.

    Fully batched over ``n*m*p`` rollouts and free of SDP solves, so the cost is
    a few seconds regardless of the probe ladder that consumes it.

    Returns ``(x_prev, x, uref, e)`` stacked over all rollouts and steps. Each
    sample carries its PREDECESSOR state so the verifier can form the real
    material derivative ``Wdot ~ (Wbar(x_prev) - Wbar(x))/dt`` -- the pairing is
    why the tube stores x_prev instead of being a flat state list.
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    env.reset()
    n = env.num_envs
    T = min(int(horizon) + 1, int(env.max_episode_len) - 1)
    xref, uref_traj = env.xref[:, :T].clone(), env.uref[:, :T].clone()
    dt = float(env.dt)
    m, p = int(num_ics), int(num_noisy)

    K = lqr_reference_gains(env, xref, uref_traj, jac_mode)
    K_t = torch.as_tensor(K, dtype=torch.float32)

    # Tile refs -> one row per (ref, ic, noise) rollout.
    idx = torch.arange(n).repeat_interleave(m * p)
    xr, ur, Kb = xref[idx], uref_traj[idx], K_t[idx]
    e0 = shell_errors(env, rho, n * m, gen).repeat_interleave(p, dim=0)
    x = env.wrap_angles(xr[:, 0] + e0)

    xs_prev, xs, us_ref, es = [], [], [], []
    for t in range(T - 1):
        e = env.wrap_angles(x - xr[:, t])
        u = ur[:, t] - torch.bmm(Kb[:, t], e.unsqueeze(-1)).squeeze(-1)
        if sigma > 0:
            u = u + sigma * torch.randn(u.shape, generator=gen)
        with torch.no_grad():
            f_t, B_t, _ = env.get_f_and_B(x)
            xdot = f_t + torch.bmm(B_t, u.unsqueeze(-1)).squeeze(-1)
        x_next = torch.clamp(env.wrap_angles(x + dt * xdot), env.X_MIN, env.X_MAX)
        # Sample = x_next, with x as its one-dt predecessor.
        xs_prev.append(x)
        xs.append(x_next)
        us_ref.append(ur[:, t + 1])
        es.append(env.wrap_angles(x_next - xr[:, t + 1]))
        x = x_next
    cat = lambda L: torch.cat(L, dim=0).numpy().astype(np.float64)  # noqa: E731
    return cat(xs_prev), cat(xs), cat(us_ref), cat(es)


def tube_violation_rates(tube, env, *, lbd, w_lb, w_ub, eps, solver, r_scaler,
                         ctl_lo, ctl_hi, use_wdot, jac_mode, check_u=True,
                         subsample=400, seed=0):
    """SDP-infeasibility and actuator-violation RATES over a tube subsample.

    RATES, not bail-on-first-failure. Gaussian noise deliberately pushes the
    tube into its extremes, so an all-or-nothing gate gets STRICTLY stricter as
    the tube is made more realistic -- the opposite of the intent. Measured on
    segway: p99|u| ~ 10 against a +-6 box while the max is 57, so a single
    transient sample was disqualifying rates that are otherwise fine.

    This matches what the deployed controller already reports: ``cvstem_lqr.py``
    tallies a within-box RATE (``_u_within_count``), it does not require zero
    violations.

    Only ``subsample`` samples are verified (2 SDP solves each with Wdot), which
    is what keeps a probe at seconds rather than minutes. Returns
    ``(sdp_rate, act_rate, n_checked)``.
    """
    x_prev_all, x_all, uref_all, e_all = tube
    n_tot = x_all.shape[0]
    rng = np.random.default_rng(seed)
    sel = (rng.choice(n_tot, size=int(subsample), replace=False)
           if subsample and subsample < n_tot else np.arange(n_tot))
    dt = float(env.dt)

    A, B = generalized_jacobians(env, x_all[sel], uref_all[sel], mode=jac_mode)
    A_p, B_p = generalized_jacobians(env, x_prev_all[sel], uref_all[sel], mode=jac_mode)
    kw = dict(w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver, r_scaler=r_scaler)
    n_sdp = n_act = 0
    for k in range(len(sel)):
        Wbar_prev = None
        if use_wdot:
            got = solve_cm_metric(A_p[k], B_p[k], lbd=lbd, return_wbar=True, **kw)
            W_p, Wbar_prev = got if isinstance(got, tuple) else (got, None)
            if W_p is None:
                n_sdp += 1
                continue
        W = solve_cm_metric(A[k], B[k], lbd=lbd, W_prev_bar=Wbar_prev,
                            dt=(dt if use_wdot else 0.0), **kw)
        if W is None:
            n_sdp += 1
            continue
        if not check_u:
            continue
        u, _ = cvstem_control(np.asarray(W, dtype=np.float64), B[k], e_all[sel][k],
                              uref_all[sel][k], r_scaler)
        if np.any((u < ctl_lo - 1e-9) | (u > ctl_hi + 1e-9)):
            n_act += 1
    nc = max(len(sel), 1)
    return n_sdp / nc, n_act / nc, len(sel)


def transient_slice(tube, block, steps):
    """The first ``steps`` steps of every rollout.

    ``build_tube`` concatenates STEP-MAJOR (all ``block = n*m*p`` rollouts at
    step 0, then all at step 1, ...), so the transient is a leading slice.
    """
    return tuple(a[:int(block) * int(steps)] for a in tube)


def feasible_tube(tube, env, *, max_violation, block, transient_steps=30, **kw):
    """Rate-thresholded feasibility, in ``feasible_with_r_backoff``'s protocol.

    Both the WHOLE tube and its TRANSIENT are gated, because a rate over the
    whole tube hides the transient: the error decays, so most samples are
    near-converged and cheap to satisfy, and a budget meant for rare excursions
    silently pays for a systematically infeasible startup instead. Measured on
    segway at lambda=0.906: 2% SDP-infeasible overall, but 10% over the first 30
    steps and 0% after -- the overall figure was entirely transient, absorbed by
    a 2% budget. This is the same over-leniency that made the car's earlier 5%
    rollout check accept bang-bang gains.

    ``cvstem_lqr.py`` keeps exactly this split (``_u_within_count_transient``
    over its first ~30 steps, alongside the episode-wide tally).

    Returns ``(ok, None, reason)``. The reason still distinguishes the two
    failure modes, because they need opposite corrections to ``r_scaler``.
    """
    for sub in (tube, transient_slice(tube, block, transient_steps)):
        sdp_rate, act_rate, _ = tube_violation_rates(sub, env, **kw)
        if sdp_rate > max_violation:
            return False, None, SDP_INFEASIBLE
        if act_rate > max_violation:
            return False, None, ACTUATOR_INFEASIBLE
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
    p.add_argument("--num-states", "--num_states", "--num-samples", "--num_samples",
                   dest="num_states", type=int, default=300,
                   help="n: uniform STATE samples")
    p.add_argument("--num-controls", "--num_controls", dest="num_controls",
                   type=int, default=8,
                   help="m: (xref, uref) reference draws per state, each giving one "
                        "(x, u_cvstem, x_next) triple; total = n*m")
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
    p.add_argument("--no-feedback-budget", "--no_feedback_budget", dest="feedback_budget",
                   action="store_false",
                   help="check K.e against the FULL control box instead of the budget "
                        "U-UREF that is actually left for feedback (u = uref + K.e)")
    p.add_argument("--no-wdot", "--no_wdot", dest="wdot", action="store_false",
                   help="drop the material derivative: solve the STATIC LMI instead of "
                        "the (x,u,x_next) form. Optimistic -- Wdot must be dominated")
    p.add_argument("--auto-envelope", "--auto_envelope", action="store_true",
                   help="if infeasible, widen the envelope (w_lb /= step, w_ub *= step) "
                        "up to --max-envelope-steps times until a lambda exists")
    p.add_argument("--max-envelope-steps", "--max_envelope_steps", type=int, default=6)
    p.add_argument("--sampling", choices=("tube", "box"), default="tube",
                   help="tube: n closed-loop rollouts under the CV-STEM control "
                        "(n x horizon samples) -- certifies where the system actually "
                        "operates. box: uniform over the whole state box -- a far "
                        "stronger claim, unachievable on most envs")
    p.add_argument("--num-trajs", "--num_trajs", type=int, default=4,
                   help="tube mode: n REFERENCE trajectories. Pointless above 1 on "
                        "segway/cartpole, whose sample_reference_controls uses freqs=[] "
                        "with XREF_INIT=0, so every reference is the same one; spend the "
                        "budget on --num-ics/--num-noisy there")
    p.add_argument("--num-ics", "--num_ics", type=int, default=8,
                   help="tube mode: m initial errors per reference, on the rho-shell")
    p.add_argument("--num-noisy", "--num_noisy", type=int, default=4,
                   help="tube mode: p noisy rollouts per initial error")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="tube mode: std of the zero-mean gaussian added to the nominal "
                        "control. Set it to the POLICY's exploration std (segway c2rl: "
                        "initial_log_std=0 -> 1.0) so the tube is the region training "
                        "actually visits. 0 = noiseless sample paths")
    p.add_argument("--rho", type=float, default=1.0,
                   help="tube radius as a DIMENSIONLESS fraction of the XE_INIT box "
                        "(1.0 = the inscribed ellipsoid). Searched DOWNWARD, so the "
                        "result is a (lambda*, rho*) pair: the rate AND the region it "
                        "holds on. A fixed rho is why segway reported no lambda at all")
    p.add_argument("--rho-min", "--rho_min", type=float, default=0.05,
                   help="stop shrinking rho here and report failure")
    p.add_argument("--max-violation", "--max_violation", type=float, default=0.0,
                   help="tolerated SDP-infeasible / actuator-violation RATE over the "
                        "tube subsample (0 = none). cvstem_lqr.py reports a within-box "
                        "rate rather than demanding zero, so a small budget here matches "
                        "the deployed controller; with noise ON a 0 budget is stricter "
                        "than the noiseless check was")
    p.add_argument("--transient-steps", "--transient_steps", type=int, default=30,
                   help="the leading steps of each rollout gated SEPARATELY against "
                        "--max-violation. Without this split a tube-wide rate hides a "
                        "systematically infeasible startup behind mostly-converged "
                        "samples (segway: 10%% over the first 30 steps, 0%% after, 2%% "
                        "overall). Mirrors cvstem_lqr.py's transient tally")
    p.add_argument("--subsample", type=int, default=400,
                   help="tube samples verified per (lambda, r) probe. The tube itself is "
                        "generated once and is much larger; this is the per-probe cost "
                        "knob (each sample = 2 SDP solves with Wdot)")
    p.add_argument("--horizon", type=int, default=60,
                   help="tube mode: steps per rollout (n*m*p*horizon samples total)")
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
    rng = np.random.default_rng(args.seed)

    import contractionRL.tasks.direct.classic  # noqa: F401  (registers the ids)
    import gymnasium as gym
    _n_env = args.num_trajs if args.sampling == "tube" else 1
    env = gym.make(args.task, num_envs=_n_env, device="cpu").unwrapped
    x_lo, x_hi, u_lo, u_hi = _boxes(env)
    x_dim, u_dim = int(env.num_dim_x), int(env.num_dim_control)

    # DATA = (x, u_cvstem, x_next) triples. Instead of drawing u uniformly, the
    # control is the one CV-STEM's own metric implies, u = uref - (1/r)B^T M e,
    # so the triple lies on the CERTIFIED closed loop rather than on an arbitrary
    # input. n states x m reference draws (xref, uref) -> n*m triples.
    n, m = args.num_states, args.num_controls
    xs = np.repeat(rng.uniform(x_lo, x_hi, size=(n, x_dim)), m, axis=0)
    # Tracking error from the env's OWN initial-error box, not an arbitrary
    # xref: e is what the deployed controller actually sees at reset.
    # XE_INIT, *not* XE_MIN: XE_INIT is what define_initial_state() draws from at
    # every reset(), so it is the error the deployed controller actually meets,
    # while XE_MIN/XE_MAX only feeds get_rollout(mode="c3m")'s broad sampler.
    # Tube mode goes through env.reset() and so always used XE_INIT; box mode
    # used XE_MIN, which made the two modes certify DIFFERENT error sets and
    # their lambda* incomparable (segway: +-1 vs +-pi on pitch_rate).
    xe_lo = env.XE_INIT_MIN.detach().cpu().numpy()
    xe_hi = env.XE_INIT_MAX.detach().cpu().numpy()
    es = rng.uniform(xe_lo, xe_hi, size=(n * m, x_dim))
    ur_lo = env.UREF_MIN.detach().cpu().numpy()
    ur_hi = env.UREF_MAX.detach().cpu().numpy()
    urefs = rng.uniform(ur_lo, ur_hi, size=(n * m, u_dim))

    print(f"[uniform-lambda] task={args.task}  x_dim={x_dim} u_dim={u_dim}")
    print(f"[uniform-lambda] state box  lo={np.round(x_lo, 3)}  hi={np.round(x_hi, 3)}")
    print(f"[uniform-lambda] control box lo={np.round(u_lo, 3)} hi={np.round(u_hi, 3)}")
    n_samples = (args.num_trajs * args.num_ics * args.num_noisy * args.horizon
                 if args.sampling == "tube" else xs.shape[0])
    if args.sampling == "tube":
        print(f"[uniform-lambda] TUBE: n={args.num_trajs} refs x m={args.num_ics} ICs "
              f"on the rho-shell x p={args.num_noisy} noisy rollouts x "
              f"{args.horizon} steps = {n_samples} samples, generated ONCE under a "
              f"NOMINAL LQR (u = uref - K_lqr e + {args.sigma:g}*xi), then "
              f"{args.subsample} verified per probe")
        if args.num_trajs > 1 and float(env.xref.abs().max()) == 0.0:
            print(f"[uniform-lambda] WARNING: this env's reference is identically zero "
                  f"(freqs=[] and XREF_INIT=0), so all {args.num_trajs} 'different' "
                  f"references are the SAME trajectory. Use --num-trajs 1 and spend the "
                  f"budget on --num-ics/--num-noisy.")
    else:
        print(f"[uniform-lambda] BOX: n={n} states x m={m} (xref,uref) draws "
              f"= {n_samples} (x, u_cvstem, x_next) triples")
    print(f"[uniform-lambda] w_lb={args.w_lb} w_ub={args.w_ub} eps={args.cm_eps} "
          f"r_scaler={args.r_scaler} solver={args.solver} jacobian={args.jacobian}")
    if args.wdot:
        print(f"[uniform-lambda] Wdot ON: LMI at x_next carries "
              f"-Wdot ~ (Wbar(x) - Wbar(x_next))/dt, dt={float(env.dt):g}")
    else:
        print("[uniform-lambda] Wdot OFF (--no-wdot): static LMI at x only")

    # u is not known until W is solved, so the Jacobian at x uses the reference
    # control (it is ignored entirely under --jacobian drift, the default).
    A, B = generalized_jacobians(env, xs, urefs, mode=args.jacobian)

    # Actuator feasibility is now EXACT: u_cvstem = uref - (1/r)B^T M e is a
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
    # The tube is generated ONCE per rho (never per probe): it is built under a
    # nominal LQR, so it does not depend on lambda, r_scaler or the envelope.
    tube_box = {"tube": None, "rho": None}

    def _check(lbd):
        if args.sampling == "tube":
            common = dict(tube=tube_box["tube"], env=env,
                          w_lb=env_wlb["w_lb"], w_ub=env_wlb["w_ub"], eps=args.cm_eps,
                          solver=args.solver, ctl_lo=ctl_lo, ctl_hi=ctl_hi,
                          use_wdot=args.wdot, jac_mode=args.jacobian, check_u=check_u,
                          subsample=args.subsample, seed=args.seed,
                          max_violation=args.max_violation,
                          block=args.num_trajs * args.num_ics * args.num_noisy,
                          transient_steps=args.transient_steps)
        else:
            common = dict(w_lb=env_wlb["w_lb"], w_ub=env_wlb["w_ub"], eps=args.cm_eps,
                          solver=args.solver, ctl_lo=ctl_lo, ctl_hi=ctl_hi, env=env,
                          dt=float(env.dt), use_wdot=args.wdot, jac_mode=args.jacobian,
                          check_u=check_u, xs=xs, es=es, urefs=urefs)
        _checker = ((lambda **kw: feasible_tube(**kw)) if args.sampling == "tube"
                    else (lambda **kw: feasible_everywhere_cvstem(A, B, **kw)))
        r, why = feasible_with_r_backoff(
            _checker, lbd=lbd, r_scaler=args.r_scaler,
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
    def search_lambda():
        """Largest feasible lambda: ladder, then bisect its top edge. None if empty."""
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
            return None
        lo = max(ok_grid)
        if min(ok_grid) > grid[0]:
            print(f"[uniform-lambda] NOTE: lambda={grid[0]:g} is infeasible while "
                  f"lambda={min(ok_grid):.4g} is feasible — feasibility is non-monotone "
                  f"here (chi_weight=1/lambda); lambda* is the top of the feasible band.")
        if lo >= grid[-1]:
            print(f"[uniform-lambda] NOTE: feasible at the search CEILING lambda={lo:g}; "
                  f"the true maximum is HIGHER — re-run with a larger --lbd-max.")
            return lo
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
        return lo

    # RHO LADDER (tube mode). rho is the certified REGION, and shrinking it is
    # the correction that a fixed rho denied the search: segway reported "no
    # feasible lambda" only because rho was pinned at the whole XE_INIT box,
    # whose extremes are initial conditions that are already falling. Searching
    # rho downward turns that dead end into a (lambda*, rho*) PAIR -- the rate
    # AND the neighbourhood of the reference it actually holds on.
    if args.sampling == "tube":
        lo, rho = None, args.rho
        while rho >= args.rho_min:
            tube_box["tube"] = build_tube(
                env, rho=rho, num_ics=args.num_ics, num_noisy=args.num_noisy,
                horizon=args.horizon, sigma=args.sigma, jac_mode=args.jacobian,
                seed=args.seed)
            tube_box["rho"] = rho
            e_norm = np.linalg.norm(tube_box["tube"][3], axis=1)
            print(f"\n[uniform-lambda] === rho={rho:.4g} "
                  f"({tube_box['tube'][1].shape[0]} tube samples, "
                  f"|e| mean={e_norm.mean():.3f} max={e_norm.max():.3f}) ===")
            lo = search_lambda()
            if lo is not None:
                break
            rho /= args.step_factor
        rho_star = rho
    else:
        lo, rho_star = search_lambda(), None

    if lo is None:
        print("\n[uniform-lambda] RESULT: no feasible lambda anywhere on the ladder"
              + (f", down to rho={args.rho_min:g}" if args.sampling == "tube" else "")
              + (f" (envelope widened to w_lb={env_wlb['w_lb']:.4g}, "
                 f"w_ub={env_wlb['w_ub']:.4g})." if args.auto_envelope else ".")
              + "\n  The {w_lb, w_ub, eps, r_scaler} envelope is infeasible over this box."
              + ("" if args.auto_envelope else
                 "\n  Re-run with --auto-envelope to widen w_lb/w_ub automatically.")
              + ("\n  If the failures above are the ACTUATOR box, raise --max-r-steps\n"
                 "  or allow a small --max-violation (cvstem_lqr.py reports a RATE)."
                 if check_u else ""))
        return 2
    if rho_star is not None:
        rate_kw = dict(lbd=lo, w_lb=env_wlb["w_lb"], w_ub=env_wlb["w_ub"],
                       eps=args.cm_eps, solver=args.solver, r_scaler=r_used["r"],
                       ctl_lo=ctl_lo, ctl_hi=ctl_hi, use_wdot=args.wdot,
                       jac_mode=args.jacobian, check_u=check_u,
                       subsample=args.subsample, seed=args.seed)
        block = args.num_trajs * args.num_ics * args.num_noisy
        print(f"\n[uniform-lambda] rho* = {rho_star:.4g} of the XE_INIT box, "
              f"sigma={args.sigma:g}, budget {args.max_violation:.1%}:")
        for name, sub in (("whole tube", tube_box["tube"]),
                          (f"transient (first {args.transient_steps} steps)",
                           transient_slice(tube_box["tube"], block, args.transient_steps))):
            s_rate, a_rate, n_ck = tube_violation_rates(sub, env, **rate_kw)
            print(f"    {name:<38} SDP-infeasible {s_rate:6.1%}   "
                  f"actuator-violating {a_rate:6.1%}   (n={n_ck})")
    print(f"\n[uniform-lambda] RESULT: uniform contraction rate lambda* = {lo:.4f}"
          + (f"  on a tube of rho* = {rho_star:.4g}" if rho_star is not None else "")
          + f"  at r_scaler = {r_used['r']:g}"
          + f"  (w_lb={env_wlb['w_lb']:.4g}, w_ub={env_wlb['w_ub']:.4g}"
          + f"{', Wdot ON' if args.wdot else ', Wdot OFF'})")
    print(f"  Largest lambda feasible with "
          f"w_lb={env_wlb['w_lb']:.4g}, w_ub={env_wlb['w_ub']:.4g}"
          + (", and whose CV-STEM control u = uref - (1/r)B^T M e stays inside the "
             "control box on the tube."
             if check_u else " (actuator check DISABLED)."))
    print(f"  Set `lbd: {lo:.3f}`, `cvstem_r_scaler: {r_used['r']:g}`, "
          f"`w_lb: {env_wlb['w_lb']:.4g}`, `w_ub: {env_wlb['w_ub']:.4g}` to synthesize "
          f"with no per-state backoff.")
    if rho_star is not None and args.sigma > 0:
        print(f"  This is a PROBABILISTIC statement, not a certificate: it holds on the\n"
              f"  reachable tube induced by noise sigma={args.sigma:g}, at the measured\n"
              f"  violation rates above (budget --max-violation={args.max_violation:g}).")
    print(f"  NOTE: sampled, not a proof — it holds over the {n_samples} tube points\n"
          f"  ({args.subsample} verified per probe). Raise "
          + ("--num-ics/--num-noisy/--horizon/--subsample" if args.sampling == "tube"
             else "--num-states/--num-controls") + " to tighten it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

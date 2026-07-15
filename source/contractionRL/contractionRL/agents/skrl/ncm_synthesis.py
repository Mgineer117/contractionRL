"""Neural Contraction Metric (NCM) Synthesis.

This module implements ALL of C2RL's offline CMG synthesis (Phase A, always
run — see c2rl.py's module docstring): the convex (SDP) machinery from
Hiroyasu Tsukamoto's Neural Contraction Metric work (Tsukamoto, Chung &
Slotine, "Neural Contraction Metrics for Robust Estimation and Control: A
Convex Optimization Approach") and the two ``cmg_method`` training pipelines
built on top of it:

  * "cvstem": ``solve_cm_metric`` solves a convex feasibility SDP per sampled
    state, returning a dual contraction metric ``W``; ``build_cm_dataset``
    assembles the resulting ``{x -> W*}`` dataset over many states, and
    ``regress_cmg`` MSE-regresses the CMG network onto it.
  * "ccm" (default): ``train_cmg_ccm`` trains the CMG network directly with
    C1 and C2 differentiable contraction losses, bypassing the SDP entirely.
    Completely independent of the SDP machinery below — no shared state.

Convex-program formulation (``solve_cm_metric`` / ``build_cm_dataset``)
-------------------------------------------------------------------------
This is Tsukamoto's CV-STEM (``classncm.cvstem0``) applied pointwise, per
sampled state, to a control-affine system ``ẋ = f(x) + B(x)u`` with drift
Jacobian ``A(x) = ∂f/∂x``. Control enters ONLY through a Riccati penalty on
the control matrix ``B(x)`` — there is no control-box vertex enumeration and
no annihilator/killing condition; this module solves the single LMI::

    variables: W̄ ⪰ 0,  ν ≥ 0,  χ ≥ 0
    I ⪯ W̄ ⪯ χ·I
    A·W̄ + W̄·Aᵀ - 2ν·B R⁻¹Bᵀ + 2λ·W̄ ⪯ -ε·I,   R = r_scaler·I
    minimize  J = chi_weight·χ + nu_weight·ν
    deploy    W = W̄ / ν

The **factor 2** on the Riccati term is not decorative: the closed loop under
``u = u_d - R⁻¹BᵀM·e`` is ``A - B R⁻¹BᵀM``, so the primal carries
``-2·M B R⁻¹Bᵀ M`` and the congruence ``W = M⁻¹`` carries it through as
``-2·B R⁻¹Bᵀ``. Because ``B``/``R`` are baked in, ``M(x) = W(x)⁻¹`` doubles as
a state-dependent Riccati solution, so this comes with an explicit,
ready-to-use LQR-style gain ``K(x) = R⁻¹B(x)ᵀM(x)`` (not wired up by this
module; C2RL only consumes ``M`` for the Mahalanobis reward, see c2rl.py).

Control is *penalized* here, not *bounded* over an action range — there is no
``cm_u_lo``/``cm_u_hi`` control box and no vertex enumeration. This can be
infeasible for systems where the Riccati term alone can't dominate the drift
at the requested ``λ`` (e.g. driftless/underactuated systems near a
singularity) — if so, there is no control-box knob to narrow anymore; lower
``w_lb``/``cvstem_r_scaler`` or ``λ`` instead (see the objective section
below), or raise ``max_lambda_reductions`` for the per-state λ-backoff.

Feasibility is a real signal, not a nuisance
--------------------------------------------
If the SDP is infeasible at a state, that is correct math (no metric
contracts the system at that ``λ`` there), not a bug — see
``min_feasibility_rate``/``max_lambda_reductions`` below for how infeasible
states are handled.

The objective: ν and χ are DECISION VARIABLES (always — there is no other mode)
-------------------------------------------------------------------------------
Tsukamoto's ``classncm.cvstem0`` solves, per state (his ``Nsplit = 1`` — the
reference really is one SDP per sample, exactly like this module)::

    variables: W̄ ⪰ 0,  ν ≥ 0,  χ ≥ 0
    I ⪯ W̄ ⪯ χ·I
    Ẇ̄ + A·W̄ + W̄·Aᵀ - 2ν·B·R⁻¹·Bᵀ + 2λ·W̄ ⪯ -ε·I
    minimize  J = (d₁·b̄/λ)·χ + d₂·ν          ← steady-state tracking-error bound
    deploy    W = W̄ / ν                      ← classncm.py:535

So ``ν`` is the metric SCALE and ``χ`` the CONDITION NUMBER. In other words,
the two quantities CV-STEM *optimizes* are the two this module used to
*hard-code*. With them hard-coded there was nothing left to optimize, so the
program degenerated to ``Minimize(0)`` and returned an arbitrary feasible
point — whatever the interior-point solver happened to land on, which is not
even a continuous function of ``x``. **That mode is gone**: ``J`` is now
always minimized, so ``W*(x)`` is a well-defined optimum, which measurably
lowers the condition number and the state-to-state scatter of ``W`` (a likely
contributor to CMG regression noise).

One constraint the reference does NOT have: ``W`` must deploy inside
``[w_lb, w_ub]`` (the CMG's ``bound_W`` envelope and the reward's scale), which
here becomes ``ν ≤ 1/w_lb`` and ``χ ≤ ν·w_ub``. CV-STEM leaves ``ν`` unbounded
above and merely penalizes it. **That cap is what makes this infeasible on
segway/cartpole at ``w_lb=0.1``**: those systems need a higher-gain (i.e.
smaller-``w_lb``) metric than the envelope permits. Measured on segway: 0%
feasible at ``w_lb=0.1``, 100% at ``w_lb=0.001`` or at ``r_scaler=0.01``. If
the SDP is infeasible everywhere, lower ``w_lb`` or ``cvstem_r_scaler`` before
touching λ — it is a metric-envelope problem, not a contraction-rate one.

Remaining deliberate simplification: ``Ẇ`` is dropped BY DEFAULT
------------------------------------------------------------------
The material-derivative term ``Ẇ = ∂W/∂x·ẋ`` is dropped by default —
``∂W/∂x`` is undefined for a pointwise decision variable, and when states are
sampled i.i.d. (``_sample_cm_states``, the default) there is no neighbouring
sample to difference against (his ``(W̄-I)/dt`` differences consecutive states
of a *trajectory*). The condition is therefore exact for a constant metric and
approximate for the state-varying one ultimately deployed; the neural
regression (a smooth ``W(x)`` network) reintroduces spatial coherence across
samples, which is the whole point of learning an NCM rather than storing a
lookup table of independent pointwise solutions.

``wdot_dt > 0`` ports his ``(W̄ - I)/dt`` proxy anyway, but it is **off by
default because it is infeasible here**: at the envs' integration step
(``dt ≈ 0.03-0.05``) the term ``(W̄-I)/dt`` scales by ``20-33×`` and swamps every
other term. It is exposed for experimentation, not as a recommendation.

When states ARE trajectory-ordered — ``build_cm_dataset``'s ``traj_x``/
``traj_lengths``/``temporal_dt`` (driven by C2RL's ``cm_wdot_trajectory``
config, fed by ``dynamics_pretrain.load_offline_trajectories`` reading an
offline ``dynamics_data.npz``) — there IS a real neighbouring sample: the
per-state loop threads each solve's normalized ``W̄`` forward as the next
state's ``W_prev_bar`` (see ``_add_wdot_term``), so ``Ẇ ≈ (W̄_t −
W̄_{t−1})/temporal_dt`` is the ACTUAL material derivative along the reference
trajectory, not Tsukamoto's identity-proxy — strictly more informative than
either dropping ``Ẇ`` or the static proxy above, and the two are mutually
exclusive with this on (temporal supersedes ``wdot_dt`` per state, falling
back to it only at trajectory starts where there is genuinely no predecessor).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as _tqdm

from .math_utils import (
    EarlyStopper,
    bound_W,
    build_lr_scheduler,
    jacobian,
    train_val_split,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Solver setup (cvxpy / MOSEK license)
# ─────────────────────────────────────────────────────────────────────────── #

def _require_cvxpy():
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "C2RL's CMG synthesis (cmg_method='cvstem') needs cvxpy (with an SDP "
            "solver such as SCS). Install it with `pip install cvxpy`."
        ) from e
    return cp


_MOSEK_LICENSE_CONFIGURED = False


def _ensure_mosek_license() -> None:
    """Point MOSEK at this repo's ``mosek.lic`` if ``MOSEKLM_LICENSE_FILE`` isn't
    already set — no-op for every other solver (SCS/CLARABEL/...).

    MOSEK's own default search path is ``~/mosek/mosek.lic`` (see README's
    Installation section); this project's license instead ships at the repo
    root, so without this every ``cm_solver: MOSEK`` solve raises cvxpy's
    ``err_missing_license_file``. Runs the directory walk at most once per
    process (cached in ``_MOSEK_LICENSE_CONFIGURED``) and never overrides a
    ``MOSEKLM_LICENSE_FILE`` the user already exported themselves.
    """
    global _MOSEK_LICENSE_CONFIGURED
    if _MOSEK_LICENSE_CONFIGURED or os.environ.get("MOSEKLM_LICENSE_FILE"):
        return
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "mosek.lic"
        if candidate.is_file():
            os.environ["MOSEKLM_LICENSE_FILE"] = str(candidate)
            break
    _MOSEK_LICENSE_CONFIGURED = True


# Run at import time — MOSEK's license env var must be set BEFORE `import mosek`
# happens (transitively, the first time `import cvxpy` runs inside
# `_require_cvxpy`), not merely before `prob.solve(...)`. cvxpy's mosek backend
# reads the license path once at import; setting the env var afterwards is too
# late even on that very first solve call.
_ensure_mosek_license()


# ─────────────────────────────────────────────────────────────────────────── #
# One-time warnings (each fires at most once per process — see docstrings)
# ─────────────────────────────────────────────────────────────────────────── #

_LICENSE_ERROR_WARNED = False


def _warn_once_if_license_error(solver: str, exc: Exception) -> None:
    """Surface a solver license failure loudly, exactly once per process.

    ``solve_cm_metric`` deliberately swallows every per-state solve error and
    returns ``None`` (treated as "infeasible at this state") so one bad solve
    can't abort a whole batch — but a missing/misconfigured license (e.g.
    ``cm_solver: MOSEK`` without a valid ``mosek.lic``) fails on EVERY solve,
    and would otherwise silently show up as "0% feasible" with no clue why.
    """
    global _LICENSE_ERROR_WARNED
    if _LICENSE_ERROR_WARNED or "license" not in str(exc).lower():
        return
    _LICENSE_ERROR_WARNED = True
    print(
        f"[C2RL] WARNING: cm_solver={solver!r} raised a license error on its first solve — "
        f"every subsequent solve will likely also fail and be reported as infeasible. "
        f"See README.md's MOSEK installation section. Original error: {exc}"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Core pointwise SDP solve
# ─────────────────────────────────────────────────────────────────────────── #

def _sym(M):
    """Symmetrise a cvxpy/numpy matrix expression (0.5·(M + Mᵀ)).

    cvxpy's PSD (``>>``/``<<``) constraints require a provably-symmetric
    operand; the LMI is mathematically symmetric but cvxpy won't always
    deduce it, so we symmetrise explicitly.
    """
    return 0.5 * (M + M.T)


def _add_wdot_term(S, Wbar, I, *, wdot_dt: float, W_prev_bar: np.ndarray | None, dt: float):
    """Fold the ``-Ẇ`` material-derivative term into the (normalized) LMI operand ``S``.

    Two mutually-exclusive proxies for ``Ẇ``, both acting on the NORMALIZED ``W̄``
    (so they stay convex/linear in the decision variable and don't drag in the
    variable scale ``ν``):

    * **temporal** (``W_prev_bar`` given, ``dt>0``): the true material derivative
      along a trajectory, ``Ẇ ≈ (W̄ - W̄_prev)/dt``, so ``-Ẇ = (W̄_prev - W̄)/dt``.
      ``W̄_prev`` is the PREVIOUS step's normalized metric at the same state
      sequence; at a trajectory start / just after a reset it is ``None`` and the
      term is dropped (``Ẇ≈0`` there). This is Tsukamoto's ``(W̄-I)/dt``
      generalized from ``I`` to the actual predecessor — a strictly better
      estimate, since consecutive states of the SAME trajectory are differenced
      rather than differencing from an arbitrary identity. Driven by
      ``build_cm_dataset``'s ``traj_x``/``traj_lengths``/``temporal_dt`` (C2RL's
      ``cm_wdot_trajectory`` config) — also retained as a general library
      capability via ``solve_cm_metric``'s own ``W_prev_bar``/``dt``/
      ``return_wbar`` kwargs.
    * **static proxy** (``wdot_dt>0``, ``W_prev_bar`` None): Tsukamoto's literal
      ``(W̄-I)/dt``. Off by default (infeasible at the envs' small dt — see module
      docstring). Superseded by the temporal term whenever both are supplied.

    Returns ``S`` unchanged when neither is active.
    """
    if W_prev_bar is not None and dt > 0:
        return S + (np.asarray(W_prev_bar, dtype=np.float64) - Wbar) / dt
    if wdot_dt > 0:
        return S + (Wbar - I) / wdot_dt
    return S


def solve_cm_metric(
    A_f: np.ndarray,
    B: np.ndarray,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str = "SCS",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    W_prev_bar: np.ndarray | None = None,
    dt: float = 0.0,
    return_wbar: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, np.ndarray | None]:
    """Solve the pointwise CV-STEM contraction-metric SDP at one state. See module docstring.

    Args:
        A_f:   drift Jacobian ``∂f/∂x`` at the state, ``(x_dim, x_dim)``.
        B:     control matrix ``(x_dim, u_dim)`` at the state.
        lbd:   contraction rate λ.
        w_lb/w_ub: eigenvalue bounds for ``W`` (match the deployed metric's).
        eps:   strict-definiteness margin on the contraction LMI.
        solver: cvxpy solver name (default SCS).
        r_scaler: ``R = r_scaler·I`` in the ``B R⁻¹ Bᵀ`` term (mirrors
            ``sdlqr.py``'s ``R_scaler``).
        chi_weight/nu_weight: weights of the CV-STEM objective ``J = chi_weight·χ
            + nu_weight·ν`` (Tsukamoto's ``d₁·b̄/α`` and ``d₂``).
            ``chi_weight=None`` → ``1/lbd``, mirroring his ``chi/alp``.
        wdot_dt: if > 0, include Tsukamoto's ``Ẇ ≈ (W̄ - I)/dt`` proxy for the
            material derivative (``classncm.cvstem0``). ``0`` (default) omits it.
        W_prev_bar/dt/return_wbar: temporal ``Ẇ`` — see ``_add_wdot_term``.

    Returns:
        The symmetric feasible metric ``W`` ``(x_dim, x_dim)`` as float32, with
        eigenvalues inside ``[w_lb, w_ub]``, or ``None`` if the SDP is infeasible
        or the solver errors. If ``return_wbar``, returns ``(W, W̄)`` instead.
    """
    cp = _require_cvxpy()
    A_f = np.asarray(A_f, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    x_dim = A_f.shape[0]
    r = r_scaler + 1e-5  # strictly positive — mirrors sdlqr.py's R_scaler guard

    # CV-STEM's variable structure (classncm.cvstem0): solve for the NORMALIZED
    # dual metric W̄ with I ⪯ W̄ ⪯ χ·I, then deploy W = W̄/ν. χ is therefore
    # exactly the condition number and ν exactly the metric scale.
    Wbar = cp.Variable((x_dim, x_dim), symmetric=True)
    chi = cp.Variable(nonneg=True)
    nu = cp.Variable(nonneg=True)
    I = np.eye(x_dim)
    # Deployment envelope w_lb·I ⪯ W ⪯ w_ub·I becomes the two linear scalar
    # constraints below (λmin(W̄)≥1, λmax(W̄)≤χ), which also keep
    # ν ∈ [1/w_ub, 1/w_lb] — so ν can never collapse to 0.
    constraints = [Wbar >> I, Wbar << chi * I, nu <= 1.0 / w_lb, chi <= nu * w_ub]

    riccati = (2.0 / r) * (B @ B.T)
    S = A_f @ Wbar + Wbar @ A_f.T + 2.0 * lbd * Wbar - nu * riccati
    S = _add_wdot_term(S, Wbar, I, wdot_dt=wdot_dt, W_prev_bar=W_prev_bar, dt=dt)
    constraints.append(_sym(S) << -eps * I)

    # CV-STEM's objective J, always — there is no "feasibility only" mode. χ and ν
    # are the metric's condition number and scale; leaving them unpenalized
    # (Minimize(0)) would return an arbitrary point of the feasible set, not a
    # well-defined W*(x).
    cw = (1.0 / lbd) if chi_weight is None else chi_weight  # Tsukamoto's chi/alp
    obj = cp.Minimize(cw * chi + nu_weight * nu)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver)
    except Exception as e:  # noqa: BLE001 — one bad solve must not abort the whole batch
        _warn_once_if_license_error(solver, e)
        return (None, None) if return_wbar else None

    def _fail():
        return (None, None) if return_wbar else None

    if prob.status not in ("optimal", "optimal_inaccurate") or Wbar.value is None:
        return _fail()
    scale = float(nu.value)
    if not np.isfinite(scale) or scale <= 0:
        return _fail()
    Wbar_v = np.asarray(Wbar.value, dtype=np.float64)
    Wbar_v = 0.5 * (Wbar_v + Wbar_v.T)
    Wv = Wbar_v / scale
    if not np.all(np.isfinite(Wv)):
        return _fail()
    if return_wbar:
        # Normalized W̄ is what a temporal-Ẇ caller would cache as next step's
        # W_prev_bar (see _add_wdot_term) — return it alongside the deployed metric.
        return Wv.astype(np.float32), Wbar_v.astype(np.float32)
    return Wv.astype(np.float32)


def _lmi_residual(A_f: np.ndarray, B: np.ndarray, W: np.ndarray, lbd: float, *, r_scaler: float = 1.0) -> float:
    """Max eigenvalue of the contraction LMI at a SOLVED ``W`` — post-hoc numpy
    re-evaluation of the same expression ``solve_cm_metric`` constrains.

    Evaluated on the DEPLOYED metric ``W = W̄/ν``, whereas the SDP imposes
    ``<< -eps·I`` on the normalized ``W̄`` — the two differ by the scale, so the
    bound this must clear is ``-eps/ν``, not ``-eps``. What matters is that it
    stays comfortably NEGATIVE; a value at or above 0 flags a solver returning
    "optimal"/"optimal_inaccurate" that is numerically borderline — useful when
    comparing SDP solvers' accuracy (see cm_solver).
    """
    A_f = np.asarray(A_f, dtype=np.float64)
    r = r_scaler + 1e-5
    riccati = (2.0 / r) * (B @ B.T)
    S = _sym(A_f @ W + W @ A_f.T - riccati + 2.0 * lbd * W)
    return float(np.max(np.linalg.eigvalsh(S)))


_LAMBDA_BACKOFF_FACTOR = 0.5  # each retry halves λ — not exposed as a config knob, only the retry count is


def _solve_cm_metric_with_backoff(
    A_f: np.ndarray,
    B: np.ndarray,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str,
    r_scaler: float,
    max_lambda_reductions: int,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    W_prev_bar: np.ndarray | None = None,
    dt: float = 0.0,
    return_wbar: bool = False,
) -> tuple[np.ndarray | None, float, int] | tuple[np.ndarray | None, np.ndarray | None, float, int]:
    """Solve ``solve_cm_metric`` at ``lbd``; on infeasibility, retry the SAME
    state alone with λ halved, up to ``max_lambda_reductions`` times, before
    giving up.

    A per-state fallback, not a global one — the LMI can be infeasible at a
    "hard" state (e.g. near a kinematic singularity) purely because the
    requested contraction RATE is too aggressive there, even though a
    (slower-contracting) feasible metric still exists. Rather than dropping
    that state entirely (lowering ``feasibility_rate``) or silently living
    with a coarser certificate everywhere, relax λ ONLY for that state until
    it becomes feasible again, or the retry budget runs out.

    Returns ``(W or None, λ actually used, reductions applied)`` — callers
    should warn when ``reductions applied > 0`` (this function does not print
    itself: the offline dataset synthesis and the online per-step reward have
    very different tolerances for log volume, see call sites).

    Note λ-backoff cannot rescue a STRUCTURALLY infeasible LMI — one where no
    metric contracts the system at that state at any rate (e.g. a driftless
    system where the Riccati penalty can't dominate the drift). There the LMI
    stays infeasible down to λ→0 and the retries are pure cost; the fix is to
    lower ``w_lb``/``cvstem_r_scaler`` (see module docstring), not λ.
    """
    cur_lbd = lbd
    for attempt in range(max_lambda_reductions + 1):
        res = solve_cm_metric(
            A_f, B, lbd=cur_lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
            r_scaler=r_scaler, chi_weight=chi_weight, nu_weight=nu_weight, wdot_dt=wdot_dt,
            W_prev_bar=W_prev_bar, dt=dt, return_wbar=return_wbar,
        )
        Wv, Wbar_v = res if return_wbar else (res, None)
        if Wv is not None:
            return (Wv, Wbar_v, cur_lbd, attempt) if return_wbar else (Wv, cur_lbd, attempt)
        cur_lbd *= _LAMBDA_BACKOFF_FACTOR
    return (None, None, cur_lbd, max_lambda_reductions) if return_wbar else (None, cur_lbd, max_lambda_reductions)


# ─────────────────────────────────────────────────────────────────────────── #
# Offline dataset synthesis ("cvstem")
# ─────────────────────────────────────────────────────────────────────────── #

def _sample_cm_states(
    get_rollout, *, num_samples: int, x_dim: int,
    x_samples: np.ndarray | None, random_ratio: float, tag: str = "[C2RL]",
) -> np.ndarray:
    """Assemble the ``num_samples`` states the CMG SDP is solved over, mixing a
    ``random_ratio`` fraction of BROAD off-reference states with reference-structured
    ones — so the regressed CMG generalizes to the off-reference states an early,
    chaotic policy actually visits, not just the near-reference tube.

    * **reference states** (``1 - random_ratio``): ``get_rollout(·, "c3m")`` —
      states built around sampled reference trajectories (``x = xref + xe``).
    * **random states** (``random_ratio``): the offline ``x_samples`` pool when
      given (dynamics-pretrain data = states visited under random actions),
      otherwise ``get_rollout(·, "dynamics")`` — states drawn broadly/uniformly
      across the whole state space (the analytic proxy for random-action coverage).

    ``random_ratio=0`` reproduces the old behavior exactly: all reference states
    (or all of ``x_samples`` when that was supplied). ``random_ratio=1`` is all
    random/offline. The metric SDP depends only on the STATE, so only ``x`` is
    taken from each source.
    """
    random_ratio = float(np.clip(random_ratio, 0.0, 1.0))
    # Back-compat: an offline pool with no explicit mix request is used wholesale
    # as before (it already is the random-action distribution).
    if x_samples is not None and random_ratio == 0.0:
        return np.asarray(x_samples, dtype=np.float32)[:, :x_dim]

    n_rand = int(round(num_samples * random_ratio))
    n_ref = num_samples - n_rand
    parts: list[np.ndarray] = []
    if n_ref > 0:
        ref = np.asarray(get_rollout(n_ref, "c3m")["x"].cpu(), dtype=np.float32)[:, :x_dim]
        parts.append(ref)
    if n_rand > 0:
        if x_samples is not None:
            pool = np.asarray(x_samples, dtype=np.float32)[:, :x_dim]
            take = min(n_rand, pool.shape[0])
            if take < n_rand:
                print(f"{tag} WARNING: cmg_random_ratio wants {n_rand} random states "
                      f"but the offline pool has only {pool.shape[0]} — using {take}.")
            idx = np.random.choice(pool.shape[0], size=take, replace=False)
            parts.append(pool[idx])
        else:
            # "dynamics" mode tiles states by num_control_per_state; ask for 1 so we
            # get n_rand DISTINCT states (we only use x, not the paired controls).
            rand = np.asarray(
                get_rollout(n_rand, "dynamics", num_control_per_state=1)["x"].cpu(),
                dtype=np.float32,
            )[:, :x_dim]
            parts.append(rand)
    x_np = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return x_np


def _flatten_trajectory_states(
    traj_x: np.ndarray, traj_lengths: np.ndarray, *, x_dim: int, max_states: int, tag: str = "[C2RL]",
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a subset of offline reference TRAJECTORIES into one ordered
    ``(n, x_dim)`` array plus a boolean ``(n,)`` mask marking each kept
    trajectory's FIRST state — preserving within-trajectory time order, which
    ``build_cm_dataset``'s temporal ``Ẇ ≈ (W̄_t − W̄_{t−1})/dt`` term (see
    ``cm_wdot_trajectory``) needs to know where to reset the predecessor.

    Trajectories are visited in a SHUFFLED order (so a ``max_states`` cap
    doesn't always keep the same early trajectories run after run), but each
    kept trajectory's own valid steps stay fully contiguous and in order —
    never interleaved with another trajectory's states. Stops once
    ``max_states`` states are collected, truncating (never padding) the last
    trajectory to fit exactly; if the offline pool has fewer than
    ``max_states`` valid states in total, uses everything available and warns.
    """
    n_traj = traj_x.shape[0]
    order = np.random.permutation(n_traj)
    parts: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    total = 0
    for n in order:
        length = int(traj_lengths[n])
        if length <= 0 or total >= max_states:
            continue
        take = min(length, max_states - total)
        seg = traj_x[n, :take, :x_dim].astype(np.float32)
        mask = np.zeros(take, dtype=bool)
        mask[0] = True
        parts.append(seg)
        starts.append(mask)
        total += take
    if not parts:
        raise ValueError(
            f"{tag} build_cm_dataset: no valid offline trajectory states available "
            f"(traj_x has {n_traj} trajectories, max_states={max_states})."
        )
    if total < max_states:
        print(f"{tag} WARNING: cmg_memory_size={max_states} exceeds the "
              f"{total} available offline trajectory states ({n_traj} trajectories) — using {total}.")
    return np.concatenate(parts, axis=0), np.concatenate(starts, axis=0)


def build_cm_dataset(
    get_rollout,
    get_f_and_B,
    *,
    x_dim: int,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    num_samples: int,
    solver: str = "SCS",
    device="cpu",
    tag: str = "[C2RL]",
    x_samples: np.ndarray | None = None,
    min_feasibility_rate: float = 0.0,
    r_scaler: float = 1.0,
    max_lambda_reductions: int = 5,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    random_ratio: float = 0.0,
    traj_x: np.ndarray | None = None,
    traj_lengths: np.ndarray | None = None,
    temporal_dt: float = 0.0,
) -> dict:
    """Build the offline ``{x → W*(x)}`` NCM dataset. See module docstring.

    States come from ``x_samples`` when given (e.g. a subsample of an offline
    ``dynamics_data.npz`` — see ``C2RLAgent._sample_cmg_x``), from ``traj_x``/
    ``traj_lengths`` when given (trajectory-ordered — see below), else
    ``num_samples`` states are drawn fresh from ``get_rollout(num_samples,
    "c3m")`` (the classic env's analytic state space). Either way, autodiffs
    ``get_f_and_B`` for the drift Jacobian ``∂f/∂x`` (same pattern as
    C3M/SD-LQR), then solves one CV-STEM SDP per state. A state infeasible at
    ``lbd`` is retried with λ halved, up to ``max_lambda_reductions`` times
    (``0`` disables this and reverts to dropping infeasible states outright —
    see ``_solve_cm_metric_with_backoff``); each state that needed a reduction
    prints a warning, plus one aggregate warning after the loop. Returns
    ``{"x", "W", "feasibility_rate", "residual_mean", "residual_max",
    "lambda_reduced_count", "lambda_reduced_rate"}`` over the feasible
    states — the residuals are the post-hoc contraction-LMI slack at each
    solved ``W``, evaluated at the λ ACTUALLY used for that state (see
    ``_lmi_residual``).

    ``traj_x``/``traj_lengths``/``temporal_dt`` (offline reference-trajectory
    Ẇ — see ``dynamics_pretrain.load_offline_trajectories`` and C2RL's
    ``cm_wdot_trajectory`` config): when ``traj_x`` (``(N, T, x_dim)``) and
    ``traj_lengths`` (``(N,)``) are given, states are drawn via
    ``_flatten_trajectory_states`` INSTEAD of ``_sample_cm_states`` — a subset
    of full trajectories (up to ``num_samples`` total states), kept in their
    ORIGINAL time order. When ``temporal_dt > 0`` too, the per-state loop below
    threads each solve's normalized ``W̄`` forward as the NEXT state's
    ``W_prev_bar`` (resetting to ``None`` at each trajectory's first state, or
    after an infeasible/dropped state — see ``_add_wdot_term``), so the SDP's
    ``Ẇ`` term is the REAL material derivative ``(W̄_t − W̄_{t−1})/temporal_dt``
    along the actual reference trajectory, rather than dropped or approximated
    by Tsukamoto's static ``(W̄−I)/wdot_dt`` proxy (which ``temporal_dt``
    supersedes whenever both are set — see ``_add_wdot_term``). Mutually
    incompatible with ``random_ratio``/``x_samples`` (ignored when ``traj_x``
    is given): mixing in i.i.d. states would break trajectory continuity.
    """
    if traj_x is not None:
        if traj_lengths is None:
            raise ValueError("build_cm_dataset: traj_x given without traj_lengths.")
        x_np, traj_start = _flatten_trajectory_states(
            traj_x, traj_lengths, x_dim=x_dim, max_states=num_samples, tag=tag,
        )
    else:
        x_np = _sample_cm_states(
            get_rollout, num_samples=num_samples, x_dim=x_dim,
            x_samples=x_samples, random_ratio=random_ratio, tag=tag,
        )
        traj_start = None
    n = x_np.shape[0]
    use_temporal = traj_start is not None and temporal_dt > 0
    if use_temporal:
        print(f"{tag} NCM SDP synthesis: temporal Ẇ from offline reference trajectories "
              f"({int(traj_start.sum())} trajectories, {n} states, temporal_dt={temporal_dt}).")

    # Autodiff the drift Jacobian once for the whole batch, then loop states for
    # the (numpy/CPU) cvxpy solves. Mirrors c3m.py's _compute_loss Jacobian setup.
    x = torch.as_tensor(x_np).to(torch.float32).to(device).requires_grad_()
    with torch.enable_grad():
        f, B, _Bbot = get_f_and_B(x)
    f = f.to(torch.float32).to(device)
    DfDx = jacobian(f, x, create_graph=False).detach().cpu().numpy()  # (n, x, x)
    B_np = B.detach().cpu().numpy()  # (n, x, u)

    xs, Ws, residuals = [], [], []
    n_reduced = 0
    reduced_lbds: list[float] = []
    prev_Wbar = None  # only advanced when use_temporal — the previous state's W̄
    pbar = _tqdm.tqdm(range(n), desc=f"{tag} NCM SDP synthesis", file=sys.stdout)
    for i in pbar:
        if use_temporal and traj_start[i]:
            prev_Wbar = None  # new trajectory — no real predecessor yet
        result = _solve_cm_metric_with_backoff(
            DfDx[i], B_np[i],
            lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
            r_scaler=r_scaler, max_lambda_reductions=max_lambda_reductions,
            chi_weight=chi_weight, nu_weight=nu_weight, wdot_dt=wdot_dt,
            W_prev_bar=prev_Wbar, dt=temporal_dt, return_wbar=use_temporal,
        )
        if use_temporal:
            Wv, Wbar_v, lbd_used, reductions = result
            # Chain forward for the NEXT state in this trajectory. A dropped/
            # infeasible state (Wv is None) breaks the chain — the next state
            # is no longer truly consecutive (one dt) with any solved
            # predecessor, so treat it like a fresh trajectory start rather
            # than differencing against a stale or wrong-dt W̄.
            prev_Wbar = Wbar_v if Wv is not None else None
        else:
            Wv, lbd_used, reductions = result
        if reductions > 0 and Wv is not None:
            n_reduced += 1
            reduced_lbds.append(lbd_used)
            print(f"{tag} WARNING: state {i} infeasible at λ={lbd:.4g} — "
                  f"reduced to λ={lbd_used:.4g} ({reductions} halving step(s)) to reach feasibility.")
        if Wv is not None:
            xs.append(x_np[i])
            Ws.append(Wv)
            residuals.append(_lmi_residual(DfDx[i], B_np[i], Wv, lbd_used, r_scaler=r_scaler))
        if (i + 1) % 128 == 0:
            pbar.set_postfix(feasible=f"{len(xs)}/{i + 1}")
    pbar.close()

    feas_rate = len(xs) / max(1, n)
    infeasibility_hint = (
        "If every state is infeasible, this is a metric-envelope or contraction-rate "
        "problem, not a control-authority one — this module's LMI has no control-box "
        "vertices to narrow. Lower w_lb/cvstem_r_scaler before touching λ; see "
        "ncm_synthesis.py's module docstring."
    )
    if not xs:
        raise RuntimeError(
            f"{tag} NCM synthesis produced 0 feasible metrics out of {n} states — "
            f"check lbd={lbd}, eps={eps}, w_lb={w_lb}, w_ub={w_ub}, and the dynamics model. "
            + infeasibility_hint
        )
    if feas_rate < min_feasibility_rate:
        raise RuntimeError(
            f"{tag} NCM synthesis only {feas_rate:.1%} of {n} states feasible, below "
            f"min_feasibility_rate={min_feasibility_rate:.1%} — check lbd={lbd}, eps={eps}, "
            f"w_lb={w_lb}, w_ub={w_ub}, and the dynamics model, or lower min_feasibility_rate "
            f"(yaml `cm:` block) if this rate is expected for this env. " + infeasibility_hint
        )
    finite_residuals = [r for r in residuals if np.isfinite(r)]
    residual_mean = float(np.mean(finite_residuals)) if finite_residuals else float("nan")
    residual_max = float(np.max(finite_residuals)) if finite_residuals else float("nan")
    lambda_reduced_rate = n_reduced / max(1, n)
    if n_reduced:
        print(
            f"{tag} WARNING: {n_reduced}/{n} states ({lambda_reduced_rate:.1%}) required λ reduction "
            f"to reach feasibility (mean reduced λ={np.mean(reduced_lbds):.4g}, "
            f"min={np.min(reduced_lbds):.4g}, requested λ={lbd:.4g}) — those states' CMG targets "
            f"certify a SLOWER contraction rate than the rest of the dataset."
        )
    print(
        f"{tag} NCM synthesis: {len(xs)}/{n} states feasible ({feas_rate:.1%}), "
        f"LMI residual mean={residual_mean:.3g} max={residual_max:.3g}"
    )
    return {
        "x": np.stack(xs).astype(np.float32),
        "W": np.stack(Ws).astype(np.float32),
        "feasibility_rate": feas_rate,
        "residual_mean": residual_mean,
        "residual_max": residual_max,
        "lambda_reduced_count": n_reduced,
        "lambda_reduced_rate": lambda_reduced_rate,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset caching
# ─────────────────────────────────────────────────────────────────────────── #

def _same_weight(cached, requested: float | None) -> bool:
    """Compare a cached ``chi_weight`` against the requested one, treating the
    ``None`` sentinel (= "use 1/lbd", see ``solve_cm_metric``) as ``nan`` on disk
    — ``nan != nan``, so a plain ``==`` would re-solve every run for the default.
    """
    cached = float(cached)
    if requested is None:
        return bool(np.isnan(cached))
    return cached == requested


def cm_dataset_cache_path(dynamics_data_path: str) -> Path:
    """Where a synthesized CM dataset (``build_cm_dataset``'s ``{x, W}`` pairs)
    is cached for a given offline ``dynamics_data.npz`` — same directory, so the
    two caches (dynamics + CM) travel together per-env (see
    ``load_offline_dynamics_data``)."""
    return Path(dynamics_data_path).with_name("cm_data.npz")


def load_cached_cm_dataset(
    cache_path: Path,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str,
    num_samples: int,
    tag: str = "[C2RL]",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    random_ratio: float = 0.0,
    wdot_trajectory: bool = False,
    temporal_dt: float = 0.0,
) -> dict | None:
    """Load a previously-cached CM dataset (see ``save_cm_dataset``) if it was
    synthesized under the EXACT same SDP config requested now — the per-state
    solve is what's expensive, so any change to ``lbd``/``w_lb``/``w_ub``/``eps``/
    ``solver``/``num_samples``/``r_scaler`` invalidates it (returns ``None``,
    triggering a fresh ``build_cm_dataset`` solve) rather than silently reusing
    stale ``W`` targets that no longer match the requested contraction condition.
    """
    if not cache_path.is_file():
        return None
    npz = np.load(cache_path)
    matches = (
        "chi_weight" in npz
        and float(npz["lbd"]) == lbd
        and float(npz["w_lb"]) == w_lb
        and float(npz["w_ub"]) == w_ub
        and float(npz["eps"]) == eps
        and str(npz["solver"]) == solver
        and int(npz["num_samples"]) == num_samples
        and float(npz.get("r_scaler", 1.0)) == r_scaler
        # chi_weight=None means "1/lbd" (see solve_cm_metric); store it as nan so
        # a cached None and an explicit float never compare equal by accident.
        and _same_weight(npz["chi_weight"], chi_weight)
        and float(npz["nu_weight"]) == nu_weight
        and float(npz["wdot_dt"]) == wdot_dt
        # random_ratio changes the STATE distribution the CMG is fit over, so a
        # cache solved at a different mix must not be reused (.get for old caches).
        and float(npz.get("random_ratio", 0.0)) == random_ratio
        # wdot_trajectory/temporal_dt change BOTH the state distribution (a
        # trajectory-ordered subset, not _sample_cm_states' i.i.d. mix) AND the
        # LMI itself (a real Ẇ term) — .get for caches predating this feature.
        and bool(npz.get("wdot_trajectory", False)) == wdot_trajectory
        and float(npz.get("temporal_dt", 0.0)) == temporal_dt
    )
    if not matches:
        print(f"{tag} Cached CM dataset at {cache_path} was synthesized with a "
              f"different cm/cmg config — re-solving the SDP.")
        return None
    print(f"{tag} Loaded cached CM dataset ({npz['x'].shape[0]} states) from {cache_path} "
          f"— skipping the per-state SDP solve.")
    return {
        "x": npz["x"],
        "W": npz["W"],
        "feasibility_rate": float(npz["feasibility_rate"]),
        "residual_mean": float(npz["residual_mean"]),
        "residual_max": float(npz["residual_max"]),
        # .get(...) — older caches predate the λ-backoff mechanism.
        "lambda_reduced_count": int(npz.get("lambda_reduced_count", 0)),
        "lambda_reduced_rate": float(npz.get("lambda_reduced_rate", 0.0)),
    }


def save_cm_dataset(
    cache_path: Path,
    dataset: dict,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str,
    num_samples: int,
    tag: str = "[C2RL]",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    random_ratio: float = 0.0,
    wdot_trajectory: bool = False,
    temporal_dt: float = 0.0,
) -> None:
    """Persist a freshly-synthesized CM dataset (``build_cm_dataset``'s return
    value) alongside the SDP config it was solved under, so a later run with the
    same config (``load_cached_cm_dataset``) can skip the expensive per-state
    solve entirely."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x=dataset["x"],
        W=dataset["W"],
        feasibility_rate=dataset["feasibility_rate"],
        residual_mean=dataset["residual_mean"],
        residual_max=dataset["residual_max"],
        lambda_reduced_count=dataset.get("lambda_reduced_count", 0),
        lambda_reduced_rate=dataset.get("lambda_reduced_rate", 0.0),
        lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver, num_samples=num_samples,
        r_scaler=r_scaler,
        chi_weight=float("nan") if chi_weight is None else float(chi_weight),
        nu_weight=nu_weight, wdot_dt=wdot_dt, random_ratio=random_ratio,
        wdot_trajectory=wdot_trajectory, temporal_dt=temporal_dt,
    )
    print(f"{tag} Cached CM dataset ({dataset['x'].shape[0]} states) → {cache_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# CMG regression (for CV-STEM)
# ─────────────────────────────────────────────────────────────────────────── #

def regress_cmg(
    ccm_gen,
    dataset: dict,
    *,
    w_lb: float,
    x_dim: int,
    bounded: bool,
    epochs: int,
    lr: float,
    batch_size: int,
    lr_scheduler: str = "",
    lr_scheduler_kwargs: dict | None = None,
    device="cpu",
    tag: str = "[C2RL]",
    on_epoch: Callable[[int, float, float, float], None] | None = None,
    val_frac: float = 0.1,
    early_stop_patience: int = 10,
) -> dict:
    """Fit the CMG to the NCM dataset by MSE regression.

    The loss compares the CMG's DEPLOYED metric — ``bound_W(ccm_gen(x), w_lb,
    x_dim, bounded)`` — against the SDP targets ``W*``. Stopping is driven
    solely by the held-out validation loss (see ``EarlyStopper``) — if a
    held-out split is configured, the best-val-epoch weights are restored
    afterward (the post-loop ``stopper.restore_best`` below).
    """
    x = torch.as_tensor(dataset["x"]).to(torch.float32).to(device)
    W_target = torch.as_tensor(dataset["W"]).to(torch.float32).to(device)
    n = x.shape[0]

    train_idx, val_idx = train_val_split(n, val_frac, device=device)
    n_train = train_idx.shape[0]
    x_val, W_val = x[val_idx], W_target[val_idx]
    stopper = EarlyStopper(patience=early_stop_patience if val_idx.shape[0] > 0 else 0)

    opt = torch.optim.Adam(ccm_gen.parameters(), lr=lr)
    scheduler = build_lr_scheduler(opt, lr_scheduler, lr_scheduler_kwargs)
    ccm_gen.train()
    losses: list[float] = []
    val_losses: list[float] = []

    pbar = _tqdm.tqdm(range(epochs), desc=f"{tag} CMG regression", file=sys.stdout)
    for epoch in pbar:
        perm = train_idx[torch.randperm(n_train, device=device)]
        iters = max(1, n_train // batch_size)
        total = 0.0
        batch_pbar = _tqdm.tqdm(
            range(iters), desc=f"{tag} epoch {epoch + 1}/{epochs}",
            file=sys.stdout, leave=False,
        )
        for b in batch_pbar:
            idx = perm[b * batch_size : (b + 1) * batch_size]
            raw_W, _ = ccm_gen(x[idx])
            W_pred = bound_W(raw_W, w_lb, x_dim, bounded)
            loss = F.mse_loss(W_pred, W_target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batch_pbar.set_postfix(mse=f"{loss.item():.4g}")
        batch_pbar.close()
        epoch_loss = total / iters
        losses.append(epoch_loss)
        if scheduler is not None:
            scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]

        postfix = {"mse": f"{epoch_loss:.4g}", "lr": f"{cur_lr:.3g}"}
        stop_val = False
        val_loss = float("nan")
        if val_idx.shape[0] > 0:
            ccm_gen.eval()
            with torch.no_grad():
                raw_W_val, _ = ccm_gen(x_val)
                val_loss = F.mse_loss(bound_W(raw_W_val, w_lb, x_dim, bounded), W_val).item()
            ccm_gen.train()
            val_losses.append(val_loss)
            postfix["val"] = f"{val_loss:.4g}"
            stop_val = stopper.step(val_loss, ccm_gen, epoch)

        pbar.set_postfix(**postfix)
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss, cur_lr, val_loss)
        if stop_val:
            print(f"{tag} CMG regression early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val MSE {stopper.best:.4g} @ epoch {stopper.best_epoch + 1}).")
            pbar.close()
            break

    if val_idx.shape[0] > 0:
        stopper.restore_best(ccm_gen)
        final_loss = losses[stopper.best_epoch]
        final_val_loss = stopper.best
        print(f"{tag} CMG regression: using best-val epoch {stopper.best_epoch + 1}/{len(losses)} "
              f"(train MSE {final_loss:.4g}, val MSE {final_val_loss:.4g}).")
    else:
        final_loss = losses[-1] if losses else float("nan")
        final_val_loss = float("nan")
    ccm_gen.eval()
    return {
        "loss_history": losses,
        "final_loss": final_loss,
        "val_loss_history": val_losses,
        "final_val_loss": final_val_loss,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# CCM neural-network synthesis (C1 + C2 losses — no per-state SDP)
# ─────────────────────────────────────────────────────────────────────────── #

def train_cmg_ccm(
    ccm_gen,
    get_f_and_B,
    get_rollout,
    *,
    x_dim: int,
    u_dim: int,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    epochs: int,
    lr: float,
    batch_size: int,
    num_samples: int,
    lr_scheduler: str = "",
    lr_scheduler_kwargs: dict | None = None,
    device="cpu",
    tag: str = "[C2RL]",
    on_epoch: Callable[[int, float, float, float], None] | None = None,
    val_frac: float = 0.1,
    early_stop_patience: int = 10,
    x_samples: np.ndarray | None = None,
    random_ratio: float = 0.0,
    pd_loss_num_samples: int = 1024,
    orthonormalize_bbot: bool = False,
) -> dict:
    """Train the CMG network directly with C1 and C2 differentiable contraction
    losses — the Manchester CCM conditions satisfied pointwise via gradient
    descent, **no per-state SDP solve and no MSE regression**.

    This is the CCM formulation's alternative to the ``build_cm_dataset`` +
    ``regress_cmg`` pipeline: instead of solving a convex SDP at every state
    to get ``W*`` targets and then regressing the CMG onto them, the CMG network
    is optimized end-to-end so its output ``W(x)`` satisfies C1 and C2.

    Stopping is driven solely by the held-out validation loss (see
    ``EarlyStopper``) — with a held-out split configured, the best-val-epoch
    weights are restored afterward.
    """
    from .math_utils import (
        b_jacobian,
        loss_pos_matrix_random_sampling,
        weighted_gradients,
    )

    x_np = _sample_cm_states(
        get_rollout, num_samples=num_samples, x_dim=x_dim,
        x_samples=x_samples, random_ratio=random_ratio, tag=tag,
    )
    x_all = torch.as_tensor(x_np).to(torch.float32).to(device)
    n = x_all.shape[0]
    print(f"{tag} CCM neural synthesis: training CMG on {n} states "
          f"(C1+C2 losses, λ={lbd}, ε={eps}, w_lb={w_lb}, w_ub={w_ub}).")

    train_idx, val_idx = train_val_split(n, val_frac, device=device)
    n_train = train_idx.shape[0]
    stopper = EarlyStopper(patience=early_stop_patience if val_idx.shape[0] > 0 else 0)

    opt = torch.optim.Adam(ccm_gen.parameters(), lr=lr)
    scheduler = build_lr_scheduler(opt, lr_scheduler, lr_scheduler_kwargs)
    ccm_gen.train()

    bounded = getattr(ccm_gen, "bounded", False)
    I_xdim = torch.eye(x_dim, device=device)
    losses: list[float] = []
    c1_history: list[float] = []
    c2_history: list[float] = []
    val_losses: list[float] = []

    def _ccm_loss(x_batch: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        # The whole body needs autograd (jacobian/weighted_gradients both call
        # torch.autograd.grad internally) regardless of the CALLER's ambient
        # grad mode — the validation branch below calls this from inside
        # torch.no_grad(), where a bare `with torch.enable_grad():` around only
        # get_f_and_B isn't enough: torch.autograd.grad() itself checks
        # torch.is_grad_enabled() at call time, so jacobian()/weighted_gradients()
        # calls made AFTER that inner block closes (back under the outer
        # no_grad) raise "does not require grad and does not have a grad_fn"
        # even though their inputs do have a grad_fn.
        with torch.enable_grad():
            x = x_batch.detach().clone().requires_grad_(True)
            bs = x.shape[0]

            raw_W, _ = ccm_gen(x)
            W = bound_W(raw_W, w_lb, x_dim, bounded)

            f, B, Bbot = get_f_and_B(x)
            f = f.to(torch.float32).to(device)
            B = B.to(torch.float32).to(device)
            Bbot = Bbot.to(torch.float32).to(device)

            DfDx = jacobian(f, x, create_graph=False).detach()   # (bs, x, x)
            DBDx = b_jacobian(B, x, u_dim, create_graph=False).detach()  # (bs, x, x, u)
            f = f.detach(); B = B.detach(); Bbot = Bbot.detach()

            if orthonormalize_bbot:
                Bbot = torch.linalg.qr(Bbot).Q

            DfW = weighted_gradients(W, f, x)  # (bs, x, x)
            DfDxW = torch.matmul(DfDx, W)
            sym_DfDxW = 0.5 * (DfDxW + DfDxW.transpose(1, 2))
            C1_inner = -DfW + 2 * sym_DfDxW + 2 * lbd * W
            C1 = torch.matmul(torch.matmul(Bbot.transpose(1, 2), C1_inner), Bbot)
            nd = C1.shape[-1]
            C1_reg = C1 + eps * torch.eye(nd, device=device)
            c1_loss = loss_pos_matrix_random_sampling(-C1_reg, num_samples=pd_loss_num_samples)

            c2_loss = torch.zeros(1, device=device)
            for j in range(u_dim):
                DbW = weighted_gradients(W, B[:, :, j], x)  # (bs, x, x)
                DbDxW = torch.matmul(DBDx[:, :, :, j], W)
                sym_DbDxW = 0.5 * (DbDxW + DbDxW.transpose(1, 2))
                C2_inner = DbW - 2 * sym_DbDxW
                C2 = torch.matmul(torch.matmul(Bbot.transpose(1, 2), C2_inner), Bbot)
                c2_loss = c2_loss + (C2 ** 2).reshape(bs, -1).sum(1).mean()

            if not bounded:
                overshoot = W - w_ub * I_xdim
                os_loss = loss_pos_matrix_random_sampling(-overshoot, num_samples=pd_loss_num_samples)
            else:
                os_loss = torch.zeros((), device=device)

            loss = c1_loss + c2_loss + os_loss
            return loss, float(c1_loss.item()), float(c2_loss.item())

    pbar = _tqdm.tqdm(range(epochs), desc=f"{tag} CCM neural synthesis", file=sys.stdout)
    for epoch in pbar:
        perm = train_idx[torch.randperm(n_train, device=device)]
        iters = max(1, n_train // batch_size)
        total, total_c1, total_c2 = 0.0, 0.0, 0.0
        batch_pbar = _tqdm.tqdm(
            range(iters), desc=f"{tag} epoch {epoch + 1}/{epochs}",
            file=sys.stdout, leave=False,
        )
        for b in batch_pbar:
            idx = perm[b * batch_size : (b + 1) * batch_size]
            loss, c1_v, c2_v = _ccm_loss(x_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            total_c1 += c1_v
            total_c2 += c2_v
            batch_pbar.set_postfix(loss=f"{loss.item():.4g}")
        batch_pbar.close()
        epoch_loss = total / iters
        losses.append(epoch_loss)
        c1_history.append(total_c1 / iters)
        c2_history.append(total_c2 / iters)

        if scheduler is not None:
            scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]

        postfix = {"loss": f"{epoch_loss:.4g}", "c1": f"{c1_history[-1]:.4g}",
                   "c2": f"{c2_history[-1]:.4g}", "lr": f"{cur_lr:.3g}"}
        stop_val = False
        val_loss = float("nan")
        if val_idx.shape[0] > 0:
            ccm_gen.eval()
            with torch.no_grad():
                val_loss_val, _, _ = _ccm_loss(x_all[val_idx])
                val_loss = val_loss_val.item()
            ccm_gen.train()
            val_losses.append(val_loss)
            postfix["val"] = f"{val_loss:.4g}"
            stop_val = stopper.step(val_loss, ccm_gen, epoch)

        pbar.set_postfix(**postfix)
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss, cur_lr, val_loss)
        if stop_val:
            print(f"{tag} CCM neural synthesis early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val loss {stopper.best:.4g} @ epoch {stopper.best_epoch + 1}).")
            pbar.close()
            break

    if val_idx.shape[0] > 0:
        stopper.restore_best(ccm_gen)
        final_loss = losses[stopper.best_epoch]
        final_val_loss = stopper.best
        print(f"{tag} CCM neural synthesis: using best-val epoch {stopper.best_epoch + 1}/{len(losses)} "
              f"(train loss {final_loss:.4g}, val loss {final_val_loss:.4g}).")
    else:
        final_loss = losses[-1] if losses else float("nan")
        final_val_loss = float("nan")
    ccm_gen.eval()
    print(f"{tag} CCM neural synthesis: c1_loss {c1_history[0]:.4g} → {c1_history[-1]:.4g}, "
          f"c2_loss {c2_history[0]:.4g} → {c2_history[-1]:.4g}")
    return {
        "loss_history": losses,
        "c1_history": c1_history,
        "c2_history": c2_history,
        "final_loss": final_loss,
        "val_loss_history": val_losses,
        "final_val_loss": final_val_loss,
    }

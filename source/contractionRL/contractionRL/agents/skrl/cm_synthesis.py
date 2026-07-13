"""Contraction-metric (CM) synthesis by convex optimization — the pointwise SDP
behind C2RL's online (``use_cmg=False``) metric.

The convex program is the one from Hiroyasu Tsukamoto's Neural Contraction Metric
work (Tsukamoto, Chung & Slotine, "Neural Contraction Metrics for Robust
Estimation and Control: A Convex Optimization Approach"): a convex feasibility
SDP that returns a dual contraction metric ``W`` at a state.

**Naming note:** despite the paper's title, C2RL's default ``use_cmg=False`` path
is NOT "neural" — it solves the SDP directly at each state and uses ``W`` as-is,
with no network anywhere. That is just a *contraction metric* obtained by convex
optimization, hence "CM" (not "NCM") throughout this codebase. The "Neural" in
Tsukamoto's method refers specifically to a network that is REGRESSED onto the
pointwise SDP solutions so it generalizes across states — that is the OFFLINE
path here (``build_cm_dataset`` + ``regress_cmg``), retained as a library
capability but unused by the default online flow. (C2RL's ``use_cmg=True`` uses a
network too, but synthesized by C3M's differentiable loss, not by regression.)

C2RL with ``use_cmg=False`` (see c2rl.py) evaluates the SDP **online, per state**,
inverts to ``M = W⁻¹`` and turns it into the Mahalanobis tracking reward
(``compute_cm_reward_online``) — it keeps NO metric network.

Pieces:

  1. ``solve_cm_metric(A_f, Bbot, ...)`` — one convex feasibility SDP per
     state, returning a dual contraction metric ``W`` (or ``None`` when
     infeasible / the solver fails). Used by BOTH the online and offline paths.
  2. ``compute_cm_reward_online(observations, ...)`` — C2RL's online reward:
     solve the SDP at each state, ``M = W⁻¹``, ``-||e||²_M`` tracking cost.
  3. ``build_cm_dataset`` / ``regress_cmg`` — the offline dataset + CMG
     regression (library-only; see note above).

Convex-program formulation and its deliberate simplifications
------------------------------------------------------------
This mirrors C3M's ``C1`` dual weak-CCM condition (see c3m.py ``_compute_loss``),
solved pointwise as an LMI in the unknown metric ``W`` instead of being enforced
on a network by a differentiable loss:

    Bᗩᵀ (Aᶠ·W + W·Aᶠᵀ + 2λ·W) Bᗩ ⪯ -ε·I          (contraction)
    w_lb·I ⪯ W ⪯ w_ub·I                            (eigenvalue bounds)
    objective: minimize 0                          (feasibility only)

with ``Aᶠ = ∂f/∂x`` (drift Jacobian only, matching C3M's ``C1``) and
``Bᗩ = B_null`` the annihilator of the control matrix ``B``.

TWO simplifications, both intentional (see the c2rl.py design / project memory):

  * **The material-derivative term ``Ẇ = ∂W/∂x·ẋ`` is dropped.** C3M's ``C1``
    carries a ``-Ẇ_f`` term, but here ``W`` is a per-point DECISION VARIABLE —
    a single sampled state has no neighbouring samples from which to form a
    spatial gradient, so ``∂W/∂x`` is undefined inside the SDP. Dropping it
    makes the condition pointwise-convex. The subsequent neural regression
    (a smooth ``W(x)`` network) reintroduces spatial coherence across samples,
    which is the whole point of learning an NCM rather than storing a lookup
    table of independent pointwise solutions.
  * **The C3M ``C2`` "killing"/orthogonality equality is NOT enforced.** As a
    pointwise linear equality it frequently renders the small SDP infeasible for
    little practical gain here; the contraction LMI + bounds alone define the
    feasible metric set we sample from.

  * **Feasibility only** (``Minimize(0)``): the user opted out of the CV-STEM
    condition-number objective, so the solver returns *some* feasible ``W`` in
    the bounded set rather than the steady-state-error-optimal one.

Fully-actuated states (``B_null`` empty or all-zero, i.e. ``x_dim == u_dim``)
carry no nontrivial contraction LMI — every bounded ``W`` is a valid CCM there —
so the contraction constraint is skipped and only the eigenvalue bounds apply.
(``formulation="cvstem"`` below is unaffected by this — it never uses ``B_null``.)

``formulation`` (``solve_cm_metric`` / ``build_cm_dataset`` / ``compute_cm_reward_online``)
---------------------------------------------------------------------------------------
Two selectable LMIs, both still pointwise-convex-with-``Ẇ``-dropped (see above):

  * ``"ccm"`` (default) — the annihilator-projected condition above. Manchester-
    style: eliminates ``B`` entirely, so it only certifies that *some* contracting
    controller exists under ``W`` — it hands you no controller.
  * ``"cvstem"`` — Tsukamoto CV-STEM-style: keeps ``B`` directly via a Riccati
    term instead of eliminating it —

        Aᶠ·W + W·Aᶠᵀ - (1/R)·B·Bᵀ + 2λ·W ⪯ -ε·I,   R = r_scaler·I

    the same congruence-transform trick (``W = M⁻¹``) that turns the nonlinear
    ``M B R⁻¹ Bᵀ M`` Riccati term convex/affine in ``W``. Because ``B``/``R``
    are baked in, the resulting ``M(x) = W(x)⁻¹`` doubles as a state-dependent
    Riccati solution, so — unlike ``"ccm"`` — it comes with an explicit,
    ready-to-use LQR-style gain ``K(x) = R⁻¹B(x)ᵀM(x)`` (not wired up by this
    module; C2RL only consumes ``M`` for the Mahalanobis reward, see c2rl.py).
    No annihilator/``B_null`` handling needed — the LMI is always the full
    ``x_dim × x_dim`` inequality regardless of actuation.
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

from .angle_utils import wrap_diff
from .math_utils import EarlyStopper, bound_W, build_lr_scheduler, jacobian, spd_inverse, train_val_split


# ─────────────────────────────────────────────────────────────────────────── #
# Solver setup (cvxpy / MOSEK license)
# ─────────────────────────────────────────────────────────────────────────── #

def _require_cvxpy():
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "C2RL's online contraction metric (use_cmg=False) needs cvxpy (with an SDP "
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


_LAMBDA_REDUCTION_WARNED = False


def _warn_once_lambda_reduced(lbd: float) -> None:
    """Surface that ``compute_cm_reward_online`` needed the λ-reduction
    backoff at least once, exactly once per process — this function is called
    every training step/replay, so (unlike ``build_cm_dataset``'s per-state +
    aggregate prints) a warning on every occurrence would flood stdout.
    """
    global _LAMBDA_REDUCTION_WARNED
    if _LAMBDA_REDUCTION_WARNED:
        return
    _LAMBDA_REDUCTION_WARNED = True
    print(
        f"[C2RL] WARNING: online CM SDP infeasible at λ={lbd:.4g} for at least one state — "
        f"falling back to a reduced λ for that state only (see _solve_cm_metric_with_backoff). "
        f"This message prints once per process; it does not repeat on later occurrences."
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Formulation validation
# ─────────────────────────────────────────────────────────────────────────── #

_VALID_FORMULATIONS = ("ccm", "cvstem")


def _validate_formulation(formulation: str) -> None:
    """Reject anything but the two known LMIs.

    Every call site below branches as ``if formulation == "cvstem": ... else:
    (ccm)`` — with no validation, a typo (``"cv_stem"``, ``"ccm3"``, ...) would
    silently fall through to the ``ccm`` branch instead of erroring, quietly
    training on the wrong metric with no warning anywhere. Fail loudly instead.
    """
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"cm_formulation={formulation!r} is not one of {_VALID_FORMULATIONS} "
            f"— check the yaml `cm.cm_formulation` value (see cm_synthesis.py module docstring)."
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Core pointwise SDP solve
# ─────────────────────────────────────────────────────────────────────────── #

def _sym(M):
    """Symmetrise a cvxpy/numpy matrix expression (0.5·(M + Mᵀ)).

    cvxpy's PSD (``>>``/``<<``) constraints require a provably-symmetric
    operand; the congruence ``Bᗩᵀ S Bᗩ`` is mathematically symmetric but cvxpy
    won't always deduce it, so we symmetrise explicitly.
    """
    return 0.5 * (M + M.T)


def solve_cm_metric(
    A_f: np.ndarray,
    Bbot: np.ndarray | None,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str = "SCS",
    formulation: str = "ccm",
    B: np.ndarray | None = None,
    r_scaler: float = 1.0,
) -> np.ndarray | None:
    """Solve the pointwise contraction-metric feasibility SDP at one state. See module docstring.

    Args:
        A_f:   drift Jacobian ``∂f/∂x`` at the state, ``(x_dim, x_dim)``.
        Bbot:  control-annihilator basis ``B_null``, ``(x_dim, null_dim)`` — or
               ``None``/zero for fully-actuated states (contraction LMI skipped).
               Only used by ``formulation="ccm"``.
        lbd:   contraction rate λ.
        w_lb/w_ub: eigenvalue bounds for ``W`` (match the deployed metric's).
        eps:   strict-definiteness margin on the contraction LMI.
        solver: cvxpy solver name (default SCS).
        formulation: ``"ccm"`` (default) — Manchester-style: eliminate ``B``
            via the annihilator ``Bbot``, an existence-only certificate that
            *some* controller contracts under ``W`` (unchanged legacy
            behavior). ``"cvstem"`` — Tsukamoto CV-STEM-style: keep ``B``
            directly via a Riccati ``B R⁻¹ Bᵀ`` term instead of eliminating
            it, so the LMI reflects an actual control-cost tradeoff rather
            than "any controller might exist". Both drop the ``Ẇ``
            material-derivative term for the same pointwise-SDP tractability
            reason (see module docstring) — ``cvstem`` is not the full CV-STEM
            condition, just its Riccati-embedded convex core.
        B:     control matrix ``(x_dim, u_dim)`` at the state — required when
            ``formulation="cvstem"``, unused otherwise.
        r_scaler: ``R = r_scaler·I`` in the ``B R⁻¹ Bᵀ`` term (mirrors
            ``sdlqr.py``'s ``R_scaler``) — only used by ``"cvstem"``.

    Returns:
        The symmetric feasible metric ``W`` ``(x_dim, x_dim)`` as float32, or
        ``None`` if the SDP is infeasible or the solver errors.
    """
    _validate_formulation(formulation)
    cp = _require_cvxpy()
    A_f = np.asarray(A_f, dtype=np.float64)
    x_dim = A_f.shape[0]

    W = cp.Variable((x_dim, x_dim), symmetric=True)
    I = np.eye(x_dim)
    constraints = [W >> w_lb * I, W << w_ub * I]

    if formulation == "cvstem":
        B = np.asarray(B, dtype=np.float64)
        r = r_scaler + 1e-5  # strictly positive — mirrors sdlqr.py's R_scaler guard
        S = A_f @ W + W @ A_f.T - (1.0 / r) * (B @ B.T) + 2.0 * lbd * W
        constraints.append(_sym(S) << -eps * np.eye(x_dim))
    elif Bbot is not None:
        Bbot = np.asarray(Bbot, dtype=np.float64)
        if Bbot.ndim == 2 and Bbot.shape[1] > 0 and not np.allclose(Bbot, 0.0):
            S = A_f @ W + W @ A_f.T + 2.0 * lbd * W
            proj = _sym(Bbot.T @ S @ Bbot)
            nd = Bbot.shape[1]
            constraints.append(proj << -eps * np.eye(nd))

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=solver)
    except Exception as e:  # noqa: BLE001 — one bad solve must not abort the whole batch
        _warn_once_if_license_error(solver, e)
        return None

    if prob.status not in ("optimal", "optimal_inaccurate") or W.value is None:
        return None
    Wv = 0.5 * (np.asarray(W.value, dtype=np.float64) + np.asarray(W.value, dtype=np.float64).T)
    if not np.all(np.isfinite(Wv)):
        return None
    return Wv.astype(np.float32)


def _lmi_residual(
    A_f: np.ndarray, Bbot: np.ndarray | None, W: np.ndarray, lbd: float,
    *, formulation: str = "ccm", B: np.ndarray | None = None, r_scaler: float = 1.0,
) -> float:
    """Max eigenvalue of the contraction LMI projection at a SOLVED ``W`` —
    post-hoc numpy re-evaluation of the same expression ``solve_cm_metric``
    constrains ``<< -eps*I``. Should be <= -eps for a feasible solution; a
    logged value close to 0 (or positive) flags a solver returning
    "optimal"/"optimal_inaccurate" that's numerically borderline — useful when
    comparing SDP solvers' accuracy (see cm_solver). ``nan`` for fully-actuated
    ``"ccm"`` states (no contraction LMI to evaluate) — ``"cvstem"`` always has
    one since it never projects ``B`` away.
    """
    if formulation == "cvstem":
        r = r_scaler + 1e-5
        S = A_f @ W + W @ A_f.T - (1.0 / r) * (B @ B.T) + 2.0 * lbd * W
        S = 0.5 * (S + S.T)
        return float(np.max(np.linalg.eigvalsh(S)))
    if Bbot is None or Bbot.ndim != 2 or Bbot.shape[1] == 0 or np.allclose(Bbot, 0.0):
        return float("nan")
    S = A_f @ W + W @ A_f.T + 2.0 * lbd * W
    proj = Bbot.T @ S @ Bbot
    proj = 0.5 * (proj + proj.T)
    return float(np.max(np.linalg.eigvalsh(proj)))


_LAMBDA_BACKOFF_FACTOR = 0.5  # each retry halves λ — not exposed as a config knob, only the retry count is


def _solve_cm_metric_with_backoff(
    A_f: np.ndarray,
    Bbot: np.ndarray | None,
    *,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    solver: str,
    formulation: str,
    B: np.ndarray | None,
    r_scaler: float,
    max_lambda_reductions: int,
) -> tuple[np.ndarray | None, float, int]:
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
    """
    cur_lbd = lbd
    for attempt in range(max_lambda_reductions + 1):
        Wv = solve_cm_metric(
            A_f, Bbot, lbd=cur_lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
            formulation=formulation, B=B, r_scaler=r_scaler,
        )
        if Wv is not None:
            return Wv, cur_lbd, attempt
        cur_lbd *= _LAMBDA_BACKOFF_FACTOR
    return None, cur_lbd, max_lambda_reductions


# ─────────────────────────────────────────────────────────────────────────── #
# Offline dataset synthesis (NCM)
# ─────────────────────────────────────────────────────────────────────────── #

def build_cm_dataset(
    get_rollout,
    get_f_and_B,
    *,
    x_dim: int,
    u_dim: int,
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
    formulation: str = "ccm",
    r_scaler: float = 1.0,
    max_lambda_reductions: int = 5,
) -> dict:
    """Build the offline ``{x → W*(x)}`` NCM dataset. See module docstring.

    States come from ``x_samples`` when given (e.g. a subsample of an offline
    ``dynamics_data.npz`` — see ``C2RLAgent._sample_cmg_x``), else ``num_samples``
    states are drawn fresh from ``get_rollout(num_samples, "c3m")`` (the classic
    env's analytic state space). Either way, autodiffs ``get_f_and_B`` for the
    drift Jacobian ``Aᶠ = ∂f/∂x`` and annihilator ``B_null`` (same pattern as
    C3M/SD-LQR), then solves one SDP per state — ``formulation`` selects which
    LMI (see ``solve_cm_metric``). A state infeasible at ``lbd`` is retried
    with λ halved, up to ``max_lambda_reductions`` times (``0`` disables this
    and reverts to dropping infeasible states outright — see
    ``_solve_cm_metric_with_backoff``); each state that needed a reduction
    prints a warning, plus one aggregate warning after the loop. Returns
    ``{"x", "W", "feasibility_rate", "residual_mean", "residual_max",
    "lambda_reduced_count", "lambda_reduced_rate"}`` over the feasible
    states — the residuals are the post-hoc contraction-LMI slack at each
    solved ``W``, evaluated at the λ ACTUALLY used for that state (see
    ``_lmi_residual``); ``nan`` if every feasible state was fully-actuated (no
    LMI to evaluate, ``formulation="ccm"`` only).

    Raises ``RuntimeError`` if NO state is feasible (a misconfigured λ/ε/bounds
    or a broken dynamics model — silently returning an empty dataset would make
    the downstream regression fail obscurely instead), or if
    ``min_feasibility_rate`` is set and the feasible fraction falls below it —
    a low-but-nonzero rate would otherwise silently regress the CMG onto a
    small, likely spatially-biased subset of the sampled states with no
    warning at all.
    """
    _validate_formulation(formulation)
    if x_samples is not None:
        x_np = np.asarray(x_samples, dtype=np.float32)[:, :x_dim]
    else:
        data = get_rollout(num_samples, "c3m")
        x_np = np.asarray(data["x"], dtype=np.float32)[:, :x_dim]
    n = x_np.shape[0]

    # Autodiff the drift Jacobian + annihilator once for the whole batch, then
    # loop states for the (numpy/CPU) cvxpy solves. Mirrors c3m.py's
    # _compute_loss / sdlqr.py's _compute_action Jacobian setup.
    x = torch.from_numpy(x_np).to(torch.float32).to(device).requires_grad_()
    with torch.enable_grad():
        f, B, Bbot = get_f_and_B(x)
    f = f.to(torch.float32).to(device)
    DfDx = jacobian(f, x, create_graph=False).detach().cpu().numpy()  # (n, x, x)
    Bbot_np = Bbot.detach().cpu().numpy() if Bbot is not None else None  # (n, x, null)
    B_np = B.detach().cpu().numpy() if formulation == "cvstem" else None  # (n, x, u)

    xs, Ws, residuals = [], [], []
    n_reduced = 0
    reduced_lbds: list[float] = []
    pbar = _tqdm.tqdm(range(n), desc=f"{tag} NCM SDP synthesis", file=sys.stdout)
    for i in pbar:
        bbot_i = Bbot_np[i] if Bbot_np is not None else None
        b_i = B_np[i] if B_np is not None else None
        Wv, lbd_used, reductions = _solve_cm_metric_with_backoff(
            DfDx[i], bbot_i, lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
            formulation=formulation, B=b_i, r_scaler=r_scaler,
            max_lambda_reductions=max_lambda_reductions,
        )
        if reductions > 0 and Wv is not None:
            n_reduced += 1
            reduced_lbds.append(lbd_used)
            print(f"{tag} WARNING: state {i} infeasible at λ={lbd:.4g} — "
                  f"reduced to λ={lbd_used:.4g} ({reductions} halving step(s)) to reach feasibility.")
        if Wv is not None:
            xs.append(x_np[i])
            Ws.append(Wv)
            residuals.append(_lmi_residual(
                DfDx[i], bbot_i, Wv, lbd_used, formulation=formulation, B=b_i, r_scaler=r_scaler
            ))
        if (i + 1) % 128 == 0:
            pbar.set_postfix(feasible=f"{len(xs)}/{i + 1}")
    pbar.close()

    feas_rate = len(xs) / max(1, n)
    if not xs:
        raise RuntimeError(
            f"{tag} NCM synthesis produced 0 feasible metrics out of {n} states — "
            f"check lbd={lbd}, eps={eps}, w_lb={w_lb}, w_ub={w_ub}, and the dynamics model."
        )
    if feas_rate < min_feasibility_rate:
        raise RuntimeError(
            f"{tag} NCM synthesis only {feas_rate:.1%} of {n} states feasible, below "
            f"min_feasibility_rate={min_feasibility_rate:.1%} — check lbd={lbd}, eps={eps}, "
            f"w_lb={w_lb}, w_ub={w_ub}, and the dynamics model, or lower min_feasibility_rate "
            f"(yaml `cm:` block) if this rate is expected for this env."
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

def with_formulation_suffix(path: Path, formulation: str) -> Path:
    """Insert ``_{formulation}`` before a path's suffix — e.g.
    ``cm_data.npz`` → ``cm_data_ccm.npz`` / ``cm_data_cvstem.npz`` — so caches
    for different ``formulation``s (see ``solve_cm_metric``) get separate
    files instead of one colliding/overwriting the other, whether the path
    came from an explicit ``cm_data_path`` or an auto-derived default. Applied
    at every ``cm_data*.npz`` write/read site (``cm_dataset_cache_path`` and
    ``C2RLAgent.synthesize_cmg``'s explicit-path / per-run-fallback cases).
    """
    return path.with_name(f"{path.stem}_{formulation}{path.suffix}")


def cm_dataset_cache_path(dynamics_data_path: str, formulation: str = "ccm") -> Path:
    """Where a synthesized CM dataset (``build_cm_dataset``'s ``{x, W}`` pairs)
    is cached for a given offline ``dynamics_data.npz`` — same directory, so the
    two caches (dynamics + CM) travel together per-env (see
    ``load_offline_dynamics_data``). Filename is formulation-suffixed (see
    ``with_formulation_suffix``)."""
    return with_formulation_suffix(Path(dynamics_data_path).with_name("cm_data.npz"), formulation)


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
    formulation: str = "ccm",
    r_scaler: float = 1.0,
) -> dict | None:
    """Load a previously-cached CM dataset (see ``save_cm_dataset``) if it was
    synthesized under the EXACT same SDP config requested now — the per-state
    solve is what's expensive, so any change to ``lbd``/``w_lb``/``w_ub``/``eps``/
    ``solver``/``num_samples``/``formulation``/``r_scaler`` invalidates it
    (returns ``None``, triggering a fresh ``build_cm_dataset`` solve) rather
    than silently reusing stale ``W`` targets that no longer match the
    requested contraction condition — critically including ``formulation``,
    since a ``"ccm"``-cached dataset and a ``"cvstem"``-cached one are
    solutions to different LMIs and must never be swapped in for each other.
    """
    if not cache_path.is_file():
        return None
    npz = np.load(cache_path)
    matches = (
        float(npz["lbd"]) == lbd
        and float(npz["w_lb"]) == w_lb
        and float(npz["w_ub"]) == w_ub
        and float(npz["eps"]) == eps
        and str(npz["solver"]) == solver
        and int(npz["num_samples"]) == num_samples
        and str(npz.get("formulation", "ccm")) == formulation
        and float(npz.get("r_scaler", 1.0)) == r_scaler
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
    formulation: str = "ccm",
    r_scaler: float = 1.0,
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
        formulation=formulation, r_scaler=r_scaler,
    )
    print(f"{tag} Cached CM dataset ({dataset['x'].shape[0]} states) → {cache_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# CMG regression
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
    """Fit the CMG to the NCM dataset by MSE regression. See module docstring.

    The loss compares the CMG's DEPLOYED metric — ``bound_W(ccm_gen(x), w_lb,
    x_dim, bounded)``, the exact quantity the Mahalanobis reward and the plotted
    certificate use (see rl_glue.compute_mahalanobis_reward) — against the SDP
    targets ``W*``. Fitting the bounded output directly means it works for both
    the plain ``CCM_Generator`` (raw + w_lb·I) and the eigenvalue-bounded
    ``BoundedCCM_Generator`` with no per-generator target adjustment.

    ``lr_scheduler``/``lr_scheduler_kwargs`` build an optional per-epoch LR
    schedule via ``math_utils.build_lr_scheduler`` — the same helper
    ``dynamics_pretrain.py`` uses for the NeuralDynamics fit, so both
    pretraining loops anneal LR the same way (independently configured).

    ``val_frac``/``early_stop_patience`` hold out that fraction of the (fixed,
    once-sampled) SDP dataset and stop once its MSE hasn't improved for that
    many consecutive epochs, restoring the best-val-epoch weights — the same
    ``math_utils.EarlyStopper`` pattern ``dynamics_pretrain.pretrain_dynamics``
    uses for the NeuralDynamics fit. ``val_frac<=0`` disables both, always
    running the full ``epochs`` budget (old behavior).

    ``on_epoch(epoch, train_mse, lr, val_mse)`` — optional callback fired after
    every epoch (agent-agnostic: this module stays a pure library, see module
    docstring — ``C2RLAgent.synthesize_cmg`` supplies a callback that calls
    ``track_data``/``write_tracking_data`` so both loss curves stream to wandb
    as separate series). ``val_mse`` is ``nan`` when validation is disabled
    (``val_frac<=0``).
    """
    x = torch.from_numpy(dataset["x"]).to(torch.float32).to(device)
    W_target = torch.from_numpy(dataset["W"]).to(torch.float32).to(device)
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
        for b in range(iters):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            raw_W, _ = ccm_gen(x[idx])
            W_pred = bound_W(raw_W, w_lb, x_dim, bounded)
            loss = F.mse_loss(W_pred, W_target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        epoch_loss = total / iters
        losses.append(epoch_loss)
        if scheduler is not None:
            scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]

        postfix = {"mse": f"{epoch_loss:.4g}", "lr": f"{cur_lr:.3g}"}
        stop = False
        val_loss = float("nan")
        if val_idx.shape[0] > 0:
            ccm_gen.eval()
            with torch.no_grad():
                raw_W_val, _ = ccm_gen(x_val)
                val_loss = F.mse_loss(bound_W(raw_W_val, w_lb, x_dim, bounded), W_val).item()
            ccm_gen.train()
            val_losses.append(val_loss)
            postfix["val"] = f"{val_loss:.4g}"
            stop = stopper.step(val_loss, ccm_gen, epoch)

        pbar.set_postfix(**postfix)
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss, cur_lr, val_loss)
        if stop:
            print(f"{tag} CMG regression early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val MSE {stopper.best:.4g} @ epoch {stopper.best_epoch + 1}).")
            pbar.close()
            break

    if val_idx.shape[0] > 0:
        # Restore the best-val-epoch weights whether early-stopped or the loop
        # simply ran out its full epoch budget, and report THAT epoch's
        # train/val MSE — not the last epoch trained, which is a different
        # (possibly worse, already-overfit) set of weights than what's
        # actually loaded into ccm_gen after this returns.
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
# Online reward (use_cmg=False)
# ─────────────────────────────────────────────────────────────────────────── #

def compute_cm_reward_online(
    observations: torch.Tensor,
    actions: torch.Tensor | None,
    get_f_and_B,
    *,
    x_dim: int,
    u_dim: int,
    angle_idx,
    lbd: float,
    w_lb: float,
    w_ub: float,
    eps: float,
    tracking_scaler: float,
    control_scaler: float,
    solver: str = "SCS",
    formulation: str = "ccm",
    r_scaler: float = 1.0,
    max_lambda_reductions: int = 5,
) -> torch.Tensor:
    """C2RL's on-the-fly (use_cmg=False) Mahalanobis reward — **no CMG network**.

    Solves the contraction-metric feasibility SDP (``solve_cm_metric``) at each observation's
    state to get a dual metric ``W``, inverts it to ``M = W⁻¹``, and returns the
    same reward the use_cmg=True path gets from its frozen CMG:

        ``-(tracking_scaler·||e||²_M + control_scaler·||u - uref||²)``

    with ``e = wrap_diff(x - xref)``. This is the online counterpart of
    ``rl_glue.compute_mahalanobis_reward`` (which reads a pre-synthesized CMG):
    off-policy SAC replays each transition many times, so solving one SDP per
    transition — cached back into the replay buffer by the caller — is cheaper
    than pre-synthesizing a whole CMG network (see c2rl.py).

    On infeasibility, retries that state alone with λ halved (see
    ``_solve_cm_metric_with_backoff``), up to ``max_lambda_reductions`` times,
    before falling back to ``W = I`` (``M = I``, plain Euclidean tracking cost)
    so one bad state can't produce a NaN reward that would poison the buffer.
    Unlike ``build_cm_dataset`` this does NOT print a warning per reduced state
    — this runs every training step/replay, so per-call logging would flood
    stdout; see ``_warn_once_lambda_reduced`` (fires once per process instead).
    """
    _validate_formulation(formulation)
    dtype = torch.float32
    device = observations.device
    obs = observations.to(dtype)
    x = obs[:, :x_dim]
    xref = obs[:, x_dim : 2 * x_dim]
    e = wrap_diff(x - xref, angle_idx).unsqueeze(-1)

    # Drift Jacobian ∂f/∂x and control annihilator B_null at each state (same
    # autodiff path as build_cm_dataset), then one CPU SDP solve per state.
    xg = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f, B, Bbot = get_f_and_B(xg)
    DfDx = jacobian(f, xg, create_graph=False).detach().cpu().numpy()
    Bbot_np = Bbot.detach().cpu().numpy() if Bbot is not None else None
    B_np = B.detach().cpu().numpy() if formulation == "cvstem" else None

    n = x.shape[0]
    W = torch.empty((n, x_dim, x_dim), dtype=dtype, device=device)
    I = torch.eye(x_dim, dtype=dtype, device=device)
    for i in range(n):
        bbot_i = Bbot_np[i] if Bbot_np is not None else None
        b_i = B_np[i] if B_np is not None else None
        Wv, _lbd_used, reductions = _solve_cm_metric_with_backoff(
            DfDx[i], bbot_i, lbd=lbd, w_lb=w_lb, w_ub=w_ub, eps=eps, solver=solver,
            formulation=formulation, B=b_i, r_scaler=r_scaler,
            max_lambda_reductions=max_lambda_reductions,
        )
        if reductions > 0 and Wv is not None:
            _warn_once_lambda_reduced(lbd)
        W[i] = I if Wv is None else torch.as_tensor(Wv, dtype=dtype, device=device)

    M = spd_inverse(W)
    quad = (e.transpose(1, 2) @ M @ e).squeeze(-1)
    reward = -tracking_scaler * quad
    if actions is not None and control_scaler > 0:
        uref = obs[:, 2 * x_dim : 2 * x_dim + u_dim]
        feedback = actions.to(dtype) - uref
        control_cost = (feedback ** 2).sum(dim=-1, keepdim=True)
        reward = reward - control_scaler * control_cost
    return reward

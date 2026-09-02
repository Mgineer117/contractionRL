"""Neural Contraction Metric (NCM) Synthesis.

All of C2RL's offline CMG synthesis (Phase A, always run): the convex (SDP)
machinery from Tsukamoto, Chung & Slotine, "Neural Contraction Metrics for
Robust Estimation and Control: A Convex Optimization Approach", plus the two
the CV-STEM pipeline built on it:

  * "cvstem": ``cvstem_joint`` solves ONE feasibility SDP over all sampled states ->
    dual metric ``W``; ``build_cm_dataset`` assembles ``{x -> W*}``;
    ``regress_cmg`` MSE-regresses the CMG onto it.
    differentiable contraction losses, bypassing the SDP entirely. Shares no
    state with any of the above.

The convex program (``cvstem_joint``)
----------------------------------------
Tsukamoto's CV-STEM (``classncm.cvstem0``) applied pointwise to
``ẋ = f(x) + B(x)u`` with drift Jacobian ``A(x) = ∂f/∂x``. Control enters only
through a Riccati penalty on ``B(x)`` — no control-box vertices, no annihilator::

    variables: W̄ ⪰ 0,  ν ≥ 0,  χ ≥ 0
    I ⪯ W̄ ⪯ χ·I
    A·W̄ + W̄·Aᵀ - 2ν·B R⁻¹Bᵀ + 2λ·W̄ ⪯ -ε·I,   R = r_scaler·I
    minimize  J = chi_weight·χ + nu_weight·ν
    deploy    W = W̄ / ν

The **factor 2** on the Riccati term is load-bearing: the closed loop under
``u = u_d - R⁻¹BᵀM·e`` is ``A - B R⁻¹BᵀM``, so the primal carries
``-2·M B R⁻¹Bᵀ M`` and the congruence ``W = M⁻¹`` carries it through. Since
``B``/``R`` are baked in, ``M(x) = W(x)⁻¹`` doubles as a state-dependent Riccati
solution with gain ``K(x) = R⁻¹B(x)ᵀM(x)`` (not wired up here — C2RL consumes
only ``M``, for the Mahalanobis reward).

``ν`` is the metric scale and ``χ`` its condition number, and both are decision
variables — the two quantities CV-STEM optimizes. (Hard-coding them degenerated
the program to ``Minimize(0)``, returning an arbitrary interior-point solution
that wasn't even continuous in ``x``. That mode is gone.)

Feasibility is a signal, not a bug: infeasible at a state means no metric
contracts the system at that λ there. Handled by ``min_feasibility_rate`` /
``max_lambda_reductions`` (per-state λ-backoff).

One constraint the reference lacks: ``W`` must deploy inside ``[w_lb, w_ub]``
(the CMG's ``bound_W`` envelope), i.e. ``ν ≤ 1/w_lb`` and ``χ ≤ ν·w_ub``.
CV-STEM leaves ν unbounded above and merely penalizes it. **That cap is what
makes segway/cartpole infeasible at w_lb=0.1** — they need a higher-gain
(smaller-w_lb) metric than the envelope permits. Measured on segway: 0% feasible
at w_lb=0.1, 100% at w_lb=0.001 or r_scaler=0.01. So on 0% feasible, lower
``w_lb``/``cvstem_r_scaler`` before touching λ — it's an envelope problem, not a
rate one.

``Ẇ`` is dropped by default
---------------------------
``∂W/∂x`` is undefined for a pointwise decision variable, and i.i.d.-sampled
states (``_sample_cm_states``, the default) have no neighbour to difference
against. The condition is thus exact for a constant metric, approximate for the
state-varying one deployed; the smooth CMG regression reintroduces spatial
coherence, which is the point of learning an NCM over a lookup table.

``wdot_dt > 0`` ports Tsukamoto's ``(W̄-I)/dt`` proxy, but it is **off by
default because it is infeasible here**: at these envs' ``dt ≈ 0.03-0.05`` the
term scales by 20-33x and swamps everything else. Exposed for experiment only.

When states are trajectory-ordered (``build_cm_dataset``'s ``traj_x``/
``traj_lengths``/``temporal_dt``, driven by C2RL's ``cm_wdot_trajectory``) there
is a real neighbour: each solve's normalized ``W̄`` threads forward as the next
state's ``W_prev_bar``, giving the actual material derivative
``Ẇ ≈ (W̄_t − W̄_{t−1})/temporal_dt`` rather than the identity-proxy. Mutually
exclusive with ``wdot_dt``, which it supersedes per state (falling back to it
only at trajectory starts, where there genuinely is no predecessor).
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

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
            "C2RL's CMG synthesis needs cvxpy (with an SDP "
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


# Run at import time — MOSEK's license env var must be set before `import mosek`
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

    ``cvstem_joint`` returns ``None`` rather than raising on a solver error, and
    returns ``None`` (treated as "infeasible at this state") so one bad solve
    can't abort a whole batch — but a missing/misconfigured license (e.g.
    ``cm_solver: MOSEK`` without a valid ``mosek.lic``) fails on every solve,
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

    Two mutually-exclusive proxies for ``Ẇ``, both acting on the normalized ``W̄``
    (so they stay convex/linear in the decision variable and don't drag in the
    variable scale ``ν``):

    * **temporal** (``W_prev_bar`` given, ``dt>0``): the true material derivative
      along a trajectory, ``Ẇ ≈ (W̄ - W̄_prev)/dt``, so ``-Ẇ = (W̄_prev - W̄)/dt``.
      ``W̄_prev`` is the previous step's normalized metric at the same state
      sequence; at a trajectory start / just after a reset it is ``None`` and the
      term is dropped (``Ẇ≈0`` there). This is Tsukamoto's ``(W̄-I)/dt``
      generalized from ``I`` to the actual predecessor — a strictly better
      estimate, since consecutive states of the same trajectory are differenced
      rather than differencing from an arbitrary identity. Driven by
      ``build_cm_dataset``'s ``traj_x``/``traj_lengths``/``temporal_dt`` (C2RL's
      ``cm_wdot_trajectory`` config) — also retained as a general library
      capability via ``cvstem_joint``'s own ``pairs``/``dt``/
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


def _lmi_residual(A_f: np.ndarray, B: np.ndarray, W: np.ndarray, lbd: float, *, r_scaler: float = 1.0) -> float:
    """Max eigenvalue of the contraction LMI at a solved ``W`` — post-hoc numpy
    re-evaluation of what ``cvstem_joint`` constrains.

    Evaluated on the deployed ``W = W̄/ν`` while the SDP constrains the
    normalized ``W̄``, so the bound to clear is ``-eps/ν``, not ``-eps``. Should
    stay comfortably negative; >= 0 flags a solver reporting "optimal" that is
    numerically borderline (useful for comparing cm_solver choices).
    """
    A_f = np.asarray(A_f, dtype=np.float64)
    r = r_scaler + 1e-5
    riccati = (2.0 / r) * (B @ B.T)
    S = _sym(A_f @ W + W @ A_f.T - riccati + 2.0 * lbd * W)
    return float(np.max(np.linalg.eigvalsh(S)))


_LAMBDA_BACKOFF_FACTOR = 0.5  # each retry halves λ — not exposed as a config knob, only the retry count is


def _sample_cm_states(
    get_rollout, *, num_samples: int, x_dim: int,
    x_samples: np.ndarray | None, random_ratio: float, tag: str = "[C2RL]",
) -> np.ndarray:
    """Assemble the ``num_samples`` states the SDP is solved over, mixing a
    ``random_ratio`` fraction of broad off-reference states with
    reference-structured ones — so the regressed CMG generalizes to where an
    early chaotic policy actually goes, not just the near-reference tube.

    * reference (``1-random_ratio``): ``get_rollout(·, "c3m")``, x = xref + xe.
    * random (``random_ratio``): the offline ``x_samples`` pool if given (states
      visited under random actions), else ``get_rollout(·, "dynamics")`` (broad
      uniform draw — the analytic proxy for that coverage).

    ``random_ratio=0`` is all-reference, ``=1`` all-random. Only ``x`` is taken
    from each source; the metric SDP depends on nothing else.
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
            # get n_rand distinct states (we only use x, not the paired controls).
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
    """Flatten offline reference trajectories into one ordered ``(n, x_dim)``
    array plus a ``(n,)`` mask marking each kept trajectory's first state — the
    temporal Ẇ term needs both the time order and where to reset the predecessor.

    Trajectory order is shuffled (so a ``max_states`` cap doesn't always keep the
    same early ones), but each trajectory's own steps stay contiguous and in
    order, never interleaved. Truncates (never pads) the last trajectory to hit
    ``max_states`` exactly; warns and uses everything if the pool is smaller.
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
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    wdot_dt: float = 0.0,
    random_ratio: float = 0.0,
    traj_x: np.ndarray | None = None,
    traj_lengths: np.ndarray | None = None,
    temporal_dt: float = 0.0,
    u_bound: float | np.ndarray | None = None,
    rho: float = 0.0,
) -> dict:
    """Build the offline ``{x → W*(x)}`` NCM dataset. See module docstring.

    States come from ``x_samples`` if given, else ``traj_x``/``traj_lengths``
    (trajectory-ordered, see below), else a fresh ``get_rollout(·, "c3m")`` draw.
    Either way: autodiff ``get_f_and_B`` for ``∂f/∂x``, then one CV-STEM SDP per
    state. Infeasible states retry with λ halved up to ``max_lambda_reductions``
    times (0 = drop them outright instead); each reduction warns, plus one
    aggregate warning after the loop.

    Returns ``{"x", "W", "feasibility_rate", "residual_mean", "residual_max",
    "lambda_reduced_count", "lambda_reduced_rate"}`` over the feasible states.
    Residuals are the post-hoc LMI slack at each solved ``W``, evaluated at the
    λ actually used for that state.

    ``u_bound``/``rho``: a state whose implied gain would saturate the actuator
    counts as infeasible, with the same λ-backoff (and counted in
    ``lambda_reduced_rate``).

    ``traj_x``/``traj_lengths``/``temporal_dt`` (C2RL's ``cm_wdot_trajectory``):
    states are drawn by ``_flatten_trajectory_states`` in original time order,
    and with ``temporal_dt > 0`` the loop threads each solve's ``W̄`` forward as
    the next state's ``W_prev_bar`` (reset to None at trajectory starts and
    after a dropped state), making the SDP's Ẇ the real material derivative
    rather than Tsukamoto's static proxy. Ignores ``random_ratio``/``x_samples``
    — mixing in i.i.d. states would break trajectory continuity.
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

    # ── one joint SDP over all n samples, not one per state ─────────────── #
    # The pointwise program below solves an independent SDP per state with its own
    # ν/χ and no Ẇ term, so its "feasibility" is a per-state statement, not a
    # contraction certificate: it can report 100% at a λ the joint program cannot
    # certify at all (measured on car_v1). C2RL then regressed its frozen CMG
    # onto that, and every downstream rate claim inherited the weaker program.
    # This is the same cvstem_joint that find_uniform_lambda searches with, so the
    # (lbd, r) it certifies is the one actually solved here.
    #
    # dt: fixed at 1.0 everywhere, and no longer configurable on either side. It
    # was, and find_uniform_lambda defaulted it to the env's dt (0.03) while
    # generation defaulted to 1.0 -- a (W̄-I)/dt term 33x larger in the SEARCH than
    # in the program the search was picking a lambda for, forcing ν from ~4 to
    # ~140. The two must solve the same program; the only way to guarantee that
    # is for neither to have a knob.
    A_j = DfDx.astype(np.float64)
    B_j = B_np.astype(np.float64)
    joint_dt = float(wdot_dt or temporal_dt or 1.0)
    # Announced before the solve, and flushed. One joint program at n=10000 runs
    # ~15 h (cost ~n^1.95) inside a single cvxpy call with no progress of its own,
    # where the per-state loop it replaced had a tqdm bar. Without this the job is
    # indistinguishable from a hang for its entire life — and Python block-buffers
    # stdout to a file, so a wall-limit SIGTERM would discard even this. Run these
    # under `python -u`.
    # ~15 h AND ~17 GB resident at n=10000 (measured on cartpole, 2026-08-27). The
    # memory is why these have to run one at a time: two concurrent builds do not
    # fit in 30 GB, and the OOM killer takes whichever is further along.
    print(f"{tag} joint CV-STEM SDP: one program over {n} samples "
          f"(lbd={lbd:g}, eps={eps:g}, w=[{w_lb:g},{w_ub:g}], r={r_scaler:g}, "
          f"dt={joint_dt:g}, solver={solver}) — no per-sample progress to report, "
          f"expect ~15 h and ~17 GB at n=10000; run these SERIALLY ...", flush=True)
    sol = cvstem_joint(A_j, B_j, lbd=lbd, eps=eps, dt=joint_dt,
                       solver=solver, r_scaler=r_scaler, chi_weight=chi_weight,
                       nu_weight=nu_weight, w_lb=w_lb, w_ub=w_ub)
    if sol is None:
        raise RuntimeError(
            f"{tag} joint CV-STEM SDP INFEASIBLE at lbd={lbd:g}, eps={eps:g}, "
            f"w=[{w_lb:g},{w_ub:g}], r={r_scaler:g} over {n} samples. This is a real "
            "result, not a transient failure: no single metric family certifies that "
            "rate over this state box. Re-run scripts/find_uniform_lambda.py for this "
            "env and use what it certifies.")
    # The LMI residual, measured rather than reported by the solver. cvstem_joint
    # returns only {W, nu, chi, J}, so a `sol.get("residual")` here is silently
    # NaN — and residual_mean/residual_max are exactly the evidence that the
    # shipped metric satisfies the LMI strictly (8a64182 advertises "strictly
    # negative LMI residuals"). NaN would keep the field's name and drop its
    # meaning, so recompute the slack at the returned solution:
    #     S_k = (W̄-I)/dt + A W̄ + W̄ Aᵀ + 2λ W̄ - ν(2/r) B Bᵀ  ⪯  -eps·I
    # with W̄_k = ν·W_k (cvstem_joint returns the deployed w = W̄/ν).
    nu_v, r_v = float(sol["nu"]), r_scaler + 1e-5
    W_dep = np.asarray(sol["W"], dtype=np.float64)
    res = np.empty(n, dtype=np.float64)
    for k in range(n):
        Wb = nu_v * W_dep[k]
        S = ((Wb - np.eye(Wb.shape[0])) / joint_dt
             + A_j[k] @ Wb + Wb @ A_j[k].T + 2.0 * lbd * Wb
             - nu_v * (2.0 / r_v) * (B_j[k] @ B_j[k].T))
        res[k] = float(np.linalg.eigvalsh(0.5 * (S + S.T))[-1])
    if not (res.max() < 0.0):
        raise RuntimeError(
            f"{tag} joint CV-STEM SDP returned status optimal but its LMI is not "
            f"strictly negative: max residual {res.max():.6g} at lbd={lbd:g}, "
            f"eps={eps:g} over {n} samples. Treat as infeasible — a solver that "
            "reports optimal on a marginally-violated program certifies nothing.")
    print(f"{tag} joint CV-STEM SDP feasible over {n} samples: nu={nu_v:.4g}, "
          f"chi={sol['chi']:.4g}, J={sol['J']:.4g}, dt={joint_dt:g}, "
          f"LMI residual mean={res.mean():.4g} max={res.max():.4g}.")
    return {
        "x": x_np.astype(np.float32),
        "W": W_dep.astype(np.float32),
        # One program, so feasibility is all-or-nothing — there is no per-state
        # rate to average, and no per-state lambda backoff.
        "feasibility_rate": 1.0,
        "residual_mean": float(res.mean()),
        "residual_max": float(res.max()),
        "lambda_reduced_count": 0,
        "lambda_reduced_rate": 0.0,
    }

def _same_weight(cached, requested: float | None) -> bool:
    """Compare a cached ``chi_weight`` against the requested one, treating the
    ``None`` sentinel (= "use 1/lbd", see ``cvstem_joint``) as ``nan`` on disk
    — ``nan != nan``, so a plain ``==`` would re-solve every run for the default.
    """
    cached = float(cached)
    if requested is None:
        return bool(np.isnan(cached))
    return cached == requested


def _same_u_bound(cached, requested) -> bool:
    """Like ``_same_weight`` but for ``u_bound``, which may now be a scalar or
    a per-channel array (pairwise actuator check) — compares element-wise."""
    cached_arr = np.atleast_1d(np.asarray(cached, dtype=np.float64))
    if requested is None:
        return bool(np.all(np.isnan(cached_arr)))
    if np.any(np.isnan(cached_arr)):
        return False
    requested_arr = np.atleast_1d(np.asarray(requested, dtype=np.float64))
    if cached_arr.shape != requested_arr.shape:
        return False
    return bool(np.array_equal(cached_arr, requested_arr))


def cm_dataset_filename(lbd: float, w_lb: float, w_ub: float, r_scaler: float = 1.0,
                        stem: str = "cm_data") -> str:
    """Cache filename encoding the most-swept SDP knobs (lbd/w_lb/w_ub/r_scaler)
    so differing contraction configs cache side-by-side instead of clobbering
    each other. r_scaler belongs here for the same reason the others do — it is
    part of the LMI (the ``R = r_scaler·I`` Riccati term). The rest of the
    config is verified at load time by ``load_cached_cm_dataset``."""
    return f"{stem}_lbd{lbd:g}_wlb{w_lb:g}_wub{w_ub:g}_rs{r_scaler:g}.npz"


def cm_dataset_cache_path(dynamics_data_path: str, *, lbd: float, w_lb: float,
                          w_ub: float, r_scaler: float = 1.0) -> Path:
    """Where a CM dataset is cached for a given offline ``dynamics_data.npz``:
    same directory, so the dynamics and CM caches travel together per-env."""
    return Path(dynamics_data_path).with_name(cm_dataset_filename(lbd, w_lb, w_ub, r_scaler))


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
    u_bound: float | np.ndarray | None = None,
    rho: float = 0.0,
) -> dict | None:
    """Load a cached CM dataset only if synthesized under the exact same SDP
    config. Any change to lbd/w_lb/w_ub/eps/solver/num_samples/r_scaler returns
    None (forcing a fresh solve) rather than silently reusing stale ``W``
    targets that no longer match the requested contraction condition.
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
        # chi_weight=None means "1/lbd" (see cvstem_joint); store it as nan so
        # a cached None and an explicit float never compare equal by accident.
        and _same_weight(npz["chi_weight"], chi_weight)
        and float(npz["nu_weight"]) == nu_weight
        and float(npz["wdot_dt"]) == wdot_dt
        # random_ratio changes the state distribution the CMG is fit over, so a
        # cache solved at a different mix must not be reused (.get for old caches).
        and float(npz.get("random_ratio", 0.0)) == random_ratio
        # wdot_trajectory/temporal_dt change both the state distribution (a
        # trajectory-ordered subset, not _sample_cm_states' i.i.d. mix) and the
        # LMI itself (a real Ẇ term) — .get for caches predating this feature.
        and bool(npz.get("wdot_trajectory", False)) == wdot_trajectory
        and float(npz.get("temporal_dt", 0.0)) == temporal_dt
        # u_bound/rho add the actuator-feasibility check (removed with the per-state path) —
        # .get for caches predating this feature (None sentinel stored as nan,
        # same trick as chi_weight, since u_bound=None is a real "off" state).
        # u_bound may now be per-channel (pairwise check) — compare as arrays.
        and _same_u_bound(npz.get("u_bound", float("nan")), u_bound)
        and float(npz.get("rho", 0.0)) == rho
    )
    if not matches:
        print(f"{tag} Cached CM dataset at {cache_path} was synthesized with a "
              f"different cm/cmg config — re-solving the SDP.")
        return None
    print(f"{tag} Loaded cached CM dataset ({npz['x'].shape[0]} states) from {cache_path} "
          f"— skipping the per-state SDP solve.")
    # SDP-config keys stay out: they are the cache KEY, already checked above,
    # and returning them would invite a caller to re-read a setting instead of
    # its own config. Everything else in the npz is payload and comes through --
    # save_cm_dataset writes extras verbatim, so a whitelist here would drop them
    # on the way back in and reintroduce exactly the bug that fixed.
    cfg_keys = {"lbd", "w_lb", "w_ub", "eps", "solver", "num_samples", "r_scaler",
                "chi_weight", "nu_weight", "wdot_dt", "random_ratio",
                "wdot_trajectory", "temporal_dt", "u_bound", "rho"}
    out = {k: npz[k] for k in npz.files if k not in cfg_keys}
    out.update({
        "feasibility_rate": float(npz["feasibility_rate"]),
        "residual_mean": float(npz["residual_mean"]),
        "residual_max": float(npz["residual_max"]),
        # .get(...) — older caches predate the λ-backoff mechanism.
        "lambda_reduced_count": int(npz.get("lambda_reduced_count", 0)),
        "lambda_reduced_rate": float(npz.get("lambda_reduced_rate", 0.0)),
    })
    return out


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
    u_bound: float | np.ndarray | None = None,
    rho: float = 0.0,
) -> None:
    """Persist a freshly-synthesized CM dataset (``build_cm_dataset``'s return
    value) alongside the SDP config it was solved under, so a later run with the
    same config (``load_cached_cm_dataset``) can skip the expensive per-state
    solve entirely.

    Keys of ``dataset`` beyond the ones named below are written through verbatim.
    They used to be dropped in silence, which cost the SOS path its analytic
    coefficients: ``_build_via_sos`` returned ``sos_coeffs``, the npz simply did
    not contain them, and the only symptom was a KeyError much later in whatever
    tried to read them back."""
    known = {"x", "W", "feasibility_rate", "residual_mean", "residual_max",
             "lambda_reduced_count", "lambda_reduced_rate"}
    extra = {k: v for k, v in dataset.items() if k not in known}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **extra,
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
        u_bound=float("nan") if u_bound is None else np.asarray(u_bound, dtype=np.float64), rho=rho,
    )
    print(f"{tag} Cached CM dataset ({dataset['x'].shape[0]} states) → {cache_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# CMG regression (for CV-STEM)
# ─────────────────────────────────────────────────────────────────────────── #

def _metric_space_error(ccm_gen, dataset, w_lb, x_dim, device):
    """Median relative error of the DEPLOYED metric against the SDP's own.

    Reported, not enforced. An earlier version raised here, on the belief that
    regressing on W and inverting per step was destroying the certificate -- that
    was wrong: for cvstem the head sets ``outputs_metric=True`` and the block
    above already inverts ``W* -> M`` once, in float64, so the net is fit on M
    directly. Measured on the real path, lambda survives (car_v1 0.0% of states
    flipped sign, segway 3.0%); the 96% figure that motivated the guard came from
    a test harness that used a bare W-emitting CCM_Generator instead.

    Kept because the pair (cond(W*), deployed metric error) is a cheap and honest
    thing to print -- segway runs cond 826 against car_v1's 4.0, which is exactly
    why that float64 inversion is load-bearing.
    """
    import numpy as np  # noqa: I001, PLC0415
    import torch  # noqa: PLC0415
    from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse  # noqa: PLC0415

    x = torch.as_tensor(np.asarray(dataset["x"])[:512], dtype=torch.float32,
                        device=device)
    W_t = np.asarray(dataset["W"])[:512].astype(np.float64)
    with torch.no_grad():
        raw, _ = ccm_gen(x)
        out = bound_W(raw, w_lb, x_dim, getattr(ccm_gen, "bounded", False))
        M_hat = (out if getattr(ccm_gen, "outputs_metric", False)
                 else spd_inverse(out)).double().cpu().numpy()
    M_ref = np.linalg.inv(W_t)
    num = np.linalg.norm(M_hat - M_ref, axis=(1, 2))
    den = np.linalg.norm(M_ref, axis=(1, 2))
    return float(np.median(num / np.maximum(den, 1e-12))), float(
        np.linalg.cond(W_t).mean())


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

    Compares the CMG's deployed metric (``bound_W(ccm_gen(x), ...)``) against the
    SDP targets ``W*``. Stopping is driven solely by held-out validation loss,
    and the best-val-epoch weights are restored afterward.
    """
    # The dataset stays resident on the host; only the minibatch actually being
    # trained on is moved to the accelerator. Its footprint grows as
    # cmg_memory_size x x_dim^2 (plus the same again for the targets), so
    # holding it on the device charges every concurrent trial for the whole
    # dataset for the whole regression, while the optimizer only ever needs one
    # batch. Pinning makes the per-batch H2D copy async-capable.
    x = torch.as_tensor(dataset["x"]).to(torch.float32)
    W_target = torch.as_tensor(dataset["W"]).to(torch.float32)
    if getattr(ccm_gen, "outputs_metric", False):
        # The head emits M, so invert the SDP's W* Once here rather than
        # inverting the network's output on every env step. Done in float64:
        # W* spans [w_lb, w_ub] with a ratio up to 1e3, and inverting the small
        # end in float32 is where the fit would lose the stiff, weakly-actuated
        # states the certificate is tightest at.
        W_target = torch.linalg.inv(W_target.to(torch.float64)).to(torch.float32)
        W_target = 0.5 * (W_target + W_target.transpose(-1, -2))   # kill drift
        print(f"[CMG] regressing M = W^-1 directly ({tuple(W_target.shape)}); "
              f"eig(M) in [{float(torch.linalg.eigvalsh(W_target).min()):.4g}, "
              f"{float(torch.linalg.eigvalsh(W_target).max()):.4g}]")
    if torch.device(device).type == "cuda":
        x, W_target = x.pin_memory(), W_target.pin_memory()
    n = x.shape[0]

    # Indices live wherever the data does, so `x[idx]` gathers on the host.
    train_idx, val_idx = train_val_split(n, val_frac, device="cpu")
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
        perm = train_idx[torch.randperm(n_train)]
        iters = max(1, n_train // batch_size)
        total = 0.0
        batch_pbar = _tqdm.tqdm(
            range(iters), desc=f"{tag} epoch {epoch + 1}/{epochs}",
            file=sys.stdout, leave=False,
        )
        for b in batch_pbar:
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x_b = x[idx].to(device, non_blocking=True)
            W_b = W_target[idx].to(device, non_blocking=True)
            raw_W, _ = ccm_gen(x_b)
            W_pred = bound_W(raw_W, w_lb, x_dim, bounded)
            loss = F.mse_loss(W_pred, W_b)
            opt.zero_grad()
            loss.backward()
            # Targets W* span [w_lb, w_ub] with a ratio up to 1e3, so a long
            # regression can spike a gradient into non-finite weights, crashing
            # the next eigh/SVD in _to_bounded_spd. Clipping keeps the fit
            # stable without moving the converged metric.
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(ccm_gen.parameters(), max_norm=10.0)
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
                # Chunked by batch_size, like the training loop above — a single
                # unbatched forward over the whole validation split (the eigh()
                # inside bound_W/_to_bounded_spd scales with batch size) was the
                # single largest allocation in this function, repeated every
                # epoch, and OOM'd in practice well before training itself did.
                val_sq_err, val_n = 0.0, 0
                for vb in range(0, x_val.shape[0], batch_size):
                    x_vb = x_val[vb : vb + batch_size].to(device, non_blocking=True)
                    raw_W_val, _ = ccm_gen(x_vb)
                    W_pred_val = bound_W(raw_W_val, w_lb, x_dim, bounded)
                    chunk = W_val[vb : vb + batch_size].to(device, non_blocking=True)
                    val_sq_err += F.mse_loss(W_pred_val, chunk, reduction="sum").item()
                    val_n += chunk.numel()
                val_loss = val_sq_err / val_n
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
    m_err, cond_w = _metric_space_error(ccm_gen, dataset, w_lb, x_dim, device)
    print(f"{tag} CMG regression: cond(W*) {cond_w:.1f}, median relative error in "
          f"the DEPLOYED metric {m_err:.3%}.")
    return {
        "loss_history": losses,
        "final_loss": final_loss,
        "val_loss_history": val_losses,
        "final_val_loss": final_val_loss,
        "metric_rel_error": m_err,
        "cond_W": cond_w,
    }


# Removed 2026-07-30: hard-control-bound CV-STEM synthesis (solve_cm_metric_bounded /
# build_cm_dataset_bounded / regress_gain_net) — Boyd's bounded-peak-input LMI with the
# gain as a free SDP variable Y := K·W. Mathematically correct, but measured worse than
# the post-hoc actuator filter (solve_cm_metric's u_bound/rho) once deployed through
# regressed nets: 98.4% held-out actuator-violation rate vs the filter's 24.6%. Its
# trace(W)-minimizing objective rides the LMI boundary with no margin, so regression
# noise crosses it most of the time. Revisit only with an explicit safety margin.
# Full implementation: `git log -S solve_cm_metric_bounded -- <this file>`.


# ─────────────────────────────────────────────────────────────────────────── #
# CCM neural-network synthesis (C1 + C2 losses — no per-state SDP)
# ─────────────────────────────────────────────────────────────────────────── #

# Removed 2026-08-31: train_cmg_ccm — trained the CMG directly on the C1/C2
# contraction losses with no SDP (cmg_method="ccm"). Never used in practice,
# and it was the only reason the CMG head could emit W instead of M; dropping
# it makes outputs_metric unconditional and removes an SPD inverse from every
# env step. C3M is unaffected — its own pd/c1/c2 losses live in math_utils.
# Recover with `git log -S train_cmg_ccm -- <this file>`.


# ─────────────────────────────────────────────────────────────────────────── #
# Online CV-STEM metric (no offline synthesis)
# ─────────────────────────────────────────────────────────────────────────── #

def cvstem_joint(
    A: np.ndarray,
    B: np.ndarray,
    *,
    lbd: float,
    eps: float,
    dt: float,
    solver: str = "MOSEK",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    pairs: list | None = None,
    w_lb: float | None = None,
    w_ub: float | None = None,
) -> dict | None:
    """Tsukamoto's ``cvstem0``: One SDP over all ``n`` samples, ν/χ shared.

    Args:
        A: drift Jacobians ``∂f/∂x``, ``(n, x_dim, x_dim)``.
        B: control matrices, ``(n, x_dim, u_dim)``.
        lbd: contraction rate λ (the reference's ``alpha``).
        eps: strict-definiteness margin (the reference's ``epsilon``).
        dt: env step, for the ``Ẇ`` term. ``0`` is rejected rather than silently
            solving the different (static) program.
        pairs: optional ``[(i, j), ...]`` meaning "sample ``j`` is one ``dt`` after
            sample ``i``" — from ``(x, u, x_next)`` transitions. When given, sample
            ``i``'s LMI carries the real backward difference
            ``-Ẇ = (W̄_i - W̄_j)/dt`` (both endpoints are variables of this one
            program, so no sequential chaining and no solve-order dependence), and
            samples that start no pair carry no ``Ẇ`` term at all. When ``None``,
            every sample carries Tsukamoto's ``(W̄ - I)/dt`` proxy instead.

            Required with ``pairs``: every ``j`` must also be in the sample set, so
            it carries its own contraction LMI and its own ``W̄ ⪯ χI``. Otherwise
            ``W̄_j`` is not a metric, it is a free slack variable, and since
            ``-Ẇ`` rewards ``W̄_j > W̄_i`` the solver inflates it to manufacture
            margin: measured on the car at λ=0.5, dt=0.03, a 3.5% inflation buys
            ~1300 of LMI slack and the "certificate" comes back with
            ``ν = 1.4e-11``, ``‖K‖₂ = 1.3e-11`` — feasible, and open loop.
            With the endpoint constrained, the same run gives ν=2.96, ‖K‖₂=2.17
            and residual inflation 1.6% (≈2% optimistic on the gain, bounded by
            ``(χ-1)·growth/dt``). ``growth_med``/``growth_max`` in the result
            report it, so the residual is visible rather than assumed away.
        r_scaler: ``R = r_scaler·I``, shared with the deployed gain.
        chi_weight/nu_weight: ``J = chi_weight·χ + nu_weight·ν``; ``None`` →
            ``1/lbd`` (the reference's ``d₁b̄·χ/α`` with ``d₁b̄ = 1``).
        w_lb/w_ub: Optional deployment envelope ``w_lb·I ⪯ W ⪯ w_ub·I`` on the
            deployed ``W = W̄/ν``, the same two scalar caps the joint program
            applies: ``ν ≤ 1/w_lb`` and ``χ ≤ ν·w_ub``. ``None`` (both, the
            default) leaves ν and χ free, which is Tsukamoto's program exactly —
            pass neither and this function is byte-identical to before.

            ``w_lb`` is the direct gain cap: ``‖M‖₂ ≤ ν ≤ 1/w_lb``, so
            ``‖K‖₂ ≤ ‖B‖₂/(r·w_lb)``. That is the knob ``r`` only *looks* like
            (ν absorbs ``r`` whenever χ is pinned). ``w_ub`` caps the condition
            number relative to the scale and mostly buys or costs feasibility.

    Returns:
        ``{"W": (n,x,x), "nu", "chi", "J"}`` with ``W_k = W̄_k/ν`` — all on the
        same scale, which is what the shared ν buys — or ``None`` if the joint
        program is infeasible or the solver errors.
    """
    cp = _require_cvxpy()
    if not dt > 0:
        raise ValueError(
            "cvstem_joint: dt must be > 0 — the (W̄-I)/dt term is part of the "
            "reference program, and dt=0 would silently solve the static LMI instead."
        )
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n, x_dim = A.shape[0], A.shape[1]
    r = r_scaler + 1e-5
    eye = np.eye(x_dim)

    Wbars = [cp.Variable((x_dim, x_dim), PSD=True) for _ in range(n)]
    nu = cp.Variable(nonneg=True)
    chi = cp.Variable(nonneg=True)
    successor = {}
    for i, j in (pairs or []):
        if not 0 <= j < n:
            raise ValueError(
                f"cvstem_joint: pair ({i}, {j}) points outside the sample set — every "
                f"x_next must itself be a sample, or W̄_{j} is a free slack variable "
                f"the solver inflates to fake margin (see the `pairs` docstring)."
            )
        successor[i] = j
    cons = []
    # Deployment envelope on W = W̄/ν, as two scalar caps on the shared nu/chi —
    # global, not per-sample. Absent by default: that is the reference program.
    if w_lb is not None:
        cons.append(nu <= 1.0 / w_lb)
    if w_ub is not None:
        cons.append(chi <= nu * w_ub)
    for k in range(n):
        Wb = Wbars[k]
        cons += [Wb >> eye, Wb << chi * eye]
        if pairs is None:
            wdot = (Wb - eye) / dt                    # Tsukamoto's proxy
        elif k in successor:
            wdot = (Wb - Wbars[successor[k]]) / dt    # -Ẇ from the real transition
        else:
            wdot = 0.0                                # endpoint: no successor sampled
        S = (wdot + A[k] @ Wb + Wb @ A[k].T + 2.0 * lbd * Wb
             - nu * ((2.0 / r) * (B[k] @ B[k].T)))
        cons.append(_sym(S) << -eps * eye)
    cw = (1.0 / lbd) if chi_weight is None else chi_weight
    prob = cp.Problem(cp.Minimize(cw * chi + nu_weight * nu), cons)
    try:
        prob.solve(solver=solver)
    except Exception as e:  # noqa: BLE001 — a solver blow-up is an infeasible point
        _warn_once_if_license_error(solver, e)
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or nu.value is None:
        return None
    scale = float(nu.value)
    if not np.isfinite(scale) or scale <= 0:
        return None
    W = np.empty((n, x_dim, x_dim), dtype=np.float64)
    for k, Wb in enumerate(Wbars):
        if Wb.value is None or not np.all(np.isfinite(Wb.value)):
            return None
        W[k] = 0.5 * (Wb.value + Wb.value.T) / scale
    out = {"W": W, "nu": scale, "chi": float(chi.value), "J": float(prob.value)}
    if successor:
        # How much the metric grew along each transition. 1.0 = no inflation, so
        # no margin was manufactured; well above 1 means the certificate is
        # leaning on -Ẇ instead of on control authority.
        g = [float(np.linalg.eigvalsh(W[j])[-1] / np.linalg.eigvalsh(W[i])[-1])
             for i, j in successor.items()]
        out["growth_med"], out["growth_max"] = float(np.median(g)), float(np.max(g))
    return out


def _as_np(v) -> np.ndarray:
    """Array-like or a torch tensor (possibly on the GPU) -> float64 numpy.

    The env's boxes (``X_MIN``/``U_MIN``/...) are tensors on the env's device, and
    cvxpy is numpy/CPU-only, so every box entering this module goes through here.
    ``np.asarray`` alone raises on a CUDA tensor.
    """
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy().astype(np.float64)
    return np.asarray(v, dtype=np.float64)


def sample_state_box(x_lo, x_hi, *, n: int, seed: int | None = None) -> np.ndarray:
    """``np.random.uniform(xlims[0], xlims[1], (n, x_dim))`` — the reference's draw.

    Not the rollout/tube sampling ``_sample_cm_states`` does: the reference
    covers the whole declared state box, which is a stronger claim (the metric
    must certify unreachable corners too) and the reason its ``W`` regression
    generalizes off-trajectory.
    """
    rng = np.random.default_rng(seed)
    x_lo, x_hi = _as_np(x_lo).ravel(), _as_np(x_hi).ravel()
    return rng.uniform(x_lo, x_hi, size=(int(n), x_lo.size)).astype(np.float32)


def drift_jacobians(get_f_and_B, x_np: np.ndarray, device="cpu"):
    """``(A = ∂f/∂x, B)`` for a batch of states, as float64 numpy.

    The reference's ``Afun`` is a user-supplied SDC matrix; every solver in this
    repo (``build_cm_dataset``, ``cvstem_lqr``) feeds the drift Jacobian, so this
    is the one place that choice is made for the joint path too.
    """
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).requires_grad_()
    with torch.enable_grad():
        f, B, _ = get_f_and_B(x)
    A = jacobian(f.to(torch.float32), x, create_graph=False)
    return (A.detach().cpu().numpy().astype(np.float64),
            B.detach().cpu().numpy().astype(np.float64))


def M_to_cholvec(W: np.ndarray) -> np.ndarray:
    """Regression labels: ``M = W⁻¹``, ``R = chol(M)ᵀ`` (upper), vectorized.

    ``(n, x, x) -> (n, x(x+1)/2)`` in row-major upper-triangular order.
    Tsukamoto's ``M2cholM`` walks the diagonals instead; the ordering is a
    bijection either way and only matters if you compare raw weights with his.
    Mirrors ``nn_modules.CholMetric``, which must use the same order.
    """
    W = np.asarray(W, dtype=np.float64)
    x_dim = W.shape[-1]
    iu = np.triu_indices(x_dim)
    out = np.empty((W.shape[0], iu[0].size), dtype=np.float32)
    for k in range(W.shape[0]):
        R = np.linalg.cholesky(np.linalg.inv(W[k])).T   # upper, M = RᵀR
        out[k] = R[iu]
    return out


def cvstem_linesearch(
    A: np.ndarray,
    B: np.ndarray,
    *,
    lbd_lo: float,
    lbd_hi: float,
    da: float,
    eps: float,
    dt: float,
    solver: str = "MOSEK",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    pairs: list | None = None,
    tag: str = "[CVSTEM-LQR]",
    w_lb: float | None = None,
    w_ub: float | None = None,
) -> tuple[float, dict | None]:
    """Tsukamoto's ``linesearch``: walk λ up, stop when ``J`` stops improving.

    His exact loop — λ from ``lbd_lo`` in steps of ``da``, solve the joint SDP at
    each, and the first time ``J`` fails to improve, back off one ``da`` and take
    that λ. ``J`` is the steady-state-error bound ``d₁b̄·χ/λ + d₂·ν``, so this is a
    rate-vs-error trade-off, not the largest feasible rate: a faster λ costs a
    worse bound. (``find_uniform_lambda.py`` answers the other question — the
    largest λ whose implied control still fits the actuator.)

    Infeasibility ends the walk the same way a rising ``J`` does: λ has gone past
    what this sample set admits, so the previous λ stands.

    Returns ``(lbd_opt, solution_at_lbd_opt)``; the solution is ``None`` only if
    even ``lbd_lo`` is infeasible.
    """
    lbd = lbd_lo
    best_lbd, best_sol, best_J = lbd_lo, None, float("inf")
    while lbd <= lbd_hi + 1e-12:
        sol = cvstem_joint(A, B, lbd=lbd, eps=eps, dt=dt, solver=solver,
                           r_scaler=r_scaler, chi_weight=chi_weight,
                           nu_weight=nu_weight, pairs=pairs, w_lb=w_lb, w_ub=w_ub)
        if sol is None:
            print(f"{tag} linesearch: lbd={lbd:.4g} infeasible — stopping")
            break
        print(f"{tag} linesearch: lbd={lbd:.4g}  J={sol['J']:.6g} "
              f"(chi={sol['chi']:.4g}, nu={sol['nu']:.4g})")
        if sol["J"] >= best_J:
            break
        best_lbd, best_sol, best_J = lbd, sol, sol["J"]
        lbd += da
    print(f"{tag} linesearch: optimal lbd = {best_lbd:.4g} (J={best_J:.6g})")
    return best_lbd, best_sol



def cvstem_metric_dataset(
    get_f_and_B,
    x_lo,
    x_hi,
    *,
    n_samples: int,
    lbd: float,
    eps: float,
    dt: float,
    solver: str = "MOSEK",
    r_scaler: float = 1.0,
    chi_weight: float | None = None,
    nu_weight: float = 1.0,
    seed: int | None = None,
    device="cpu",
    tag: str = "[CVSTEM-LQR]",
    linesearch: tuple | None = None,
    wdot: str = "proxy",
    u_lo=None,
    u_hi=None,
    w_lb: float | None = None,
    w_ub: float | None = None,
) -> dict | None:
    """Steps 1-3 of the reference pipeline: sample the box, one joint SDP, label.

    Returns ``{"x", "W", "cholM", "nu", "chi", "J"}`` — regression inputs, the
    certified metrics, and the ``chol(M)`` labels ``CholMetric`` is fit to — or
    ``None`` if the joint SDP is infeasible at this λ.

    There is no partial success and no feasibility rate: one program covers every
    sample, so it either certifies λ over the whole draw or it does not.

    ``linesearch=(lo, hi, da)`` runs ``cvstem_linesearch`` over the same samples
    first and uses its ``argmin J`` λ instead of ``lbd`` — his own λ selection.
    The returned dict carries the λ actually used under ``"lbd"``.

    ``wdot`` picks the ``Ẇ`` term (see ``cvstem_joint``):

    * ``"proxy"`` (default, the reference): ``+(W̄-I)/dt`` at every sample.
    * ``"transition"``: propagate each sample one ``dt`` under a uniformly random
      ``u ∈ [u_lo, u_hi]`` (the env's actuator box, required for this mode) and
      use the real difference ``-Ẇ = (W̄(x) - W̄(x_next))/dt``, with both endpoints
      as constrained samples of the same program. Doubles the program size and
      makes ``W`` a metric at ``2·n_samples`` states, of which the first
      ``n_samples`` are the ones the regression is fit to.

    The regression set is the joint program's sample set — the reference fits its
    network to exactly the states it certified, so scaling the training data means
    raising ``n_samples``, not labelling extra states after the fact.
    """
    x_np = sample_state_box(x_lo, x_hi, n=n_samples, seed=seed)
    pairs = None
    if wdot == "transition":
        if u_lo is None or u_hi is None:
            raise ValueError(
                "cvstem_metric_dataset: wdot='transition' needs the env's actuator box "
                "(u_lo/u_hi) — the Ẇ term is measured along transitions driven by a "
                "uniformly random control drawn from it."
            )
        rng = np.random.default_rng(seed)
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            f, B_t, _ = get_f_and_B(xt)
            u = torch.as_tensor(
                rng.uniform(_as_np(u_lo).ravel(), _as_np(u_hi).ravel(),
                            size=(n_samples, B_t.shape[-1])),
                dtype=torch.float32, device=device)
            xn = xt + float(dt) * (f + torch.bmm(B_t, u.unsqueeze(-1)).squeeze(-1))
            xn = torch.clamp(
                xn,
                torch.as_tensor(_as_np(x_lo), dtype=torch.float32, device=device),
                torch.as_tensor(_as_np(x_hi), dtype=torch.float32, device=device))
        # One sample set: [x | x_next]. x_next must carry its own LMI or the -Ẇ
        # term degenerates into free slack (see cvstem_joint's `pairs` docstring).
        x_all = np.concatenate([x_np, xn.cpu().numpy().astype(np.float32)], axis=0)
        pairs = [(k, n_samples + k) for k in range(n_samples)]
    else:
        x_all = x_np
    A, B = drift_jacobians(get_f_and_B, x_all, device=device)
    print(f"{tag} CV-STEM joint SDP: {n_samples} uniform samples over the state box "
          f"(wdot={wdot}, {len(x_all)} states in the program), "
          f"lbd={lbd}, eps={eps}, r_scaler={r_scaler}, dt={dt}, solver={solver} ...",
          flush=True)
    if linesearch is not None:
        lo, hi, da = (float(v) for v in linesearch)
        lbd, sol = cvstem_linesearch(A, B, lbd_lo=lo, lbd_hi=hi, da=da, eps=eps, dt=dt,
                                     solver=solver, r_scaler=r_scaler,
                                     chi_weight=chi_weight, nu_weight=nu_weight,
                                     pairs=pairs, tag=tag, w_lb=w_lb, w_ub=w_ub)
    else:
        sol = cvstem_joint(A, B, lbd=lbd, eps=eps, dt=dt, solver=solver,
                           r_scaler=r_scaler, chi_weight=chi_weight,
                           nu_weight=nu_weight, pairs=pairs, w_lb=w_lb, w_ub=w_ub)
    if sol is None:
        return None
    print(f"{tag} CV-STEM joint SDP feasible: nu={sol['nu']:.4g} (metric scale, "
          f"max eig M), chi={sol['chi']:.4g} (condition number), J={sol['J']:.4g}.")
    W_fit = sol["W"][:len(x_np)]     # x_next entered to pin Ẇ, not to be regressed
    return {"x": x_np, "W": W_fit.astype(np.float32),
            "cholM": M_to_cholvec(W_fit),
            "nu": sol["nu"], "chi": sol["chi"], "J": sol["J"], "lbd": lbd}


def cvstem_cache_path(data_path: str, *, lbd: float, r_scaler: float, w_lb, w_ub,
                      eps: float, dt: float, n_samples: int) -> Path:
    """Where ``cvstem_metric_dataset``'s result is cached.

    Separate from ``cm_dataset_cache_path`` because these are different programs:
    this one is the single joint SDP (shared ν/χ, ``Ẇ`` proxy at ``dt``) and its
    result carries ``nu``/``chi``/``J``/``cholM``, none of which the pointwise
    cache stores. Sharing a filename would let one silently load the other's
    metric. The knobs most often swept go in the NAME so configs cache
    side-by-side; the rest are verified at load time.
    """
    w = f"{'none' if w_lb is None else f'{w_lb:g}'}_{'none' if w_ub is None else f'{w_ub:g}'}"
    stem = (f"cvstem_joint_lbd{lbd:g}_rs{r_scaler:g}_w{w}"
            f"_eps{eps:g}_dt{dt:g}_n{n_samples}.npz")
    return Path(data_path).with_name(stem)


def save_cvstem_dataset(cache_path: Path, dataset: dict, **cfg) -> None:
    """Persist a joint-SDP solve alongside the config it was solved under.

    Config keys are stored under a ``cfg_`` prefix: ``lbd`` is both an input and
    a result here (``lbd_linesearch`` makes the solve pick its own), and an
    unprefixed collision would silently overwrite one with the other.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, x=dataset["x"], W=dataset["W"], cholM=dataset["cholM"],
             nu=dataset["nu"], chi=dataset["chi"], J=dataset["J"], lbd=dataset["lbd"],
             **{f"cfg_{k}": ("none" if v is None else v) for k, v in cfg.items()})
    print(f"[CVSTEM-LQR] Cached joint SDP ({dataset['x'].shape[0]} states) → {cache_path}")


def load_cvstem_dataset(cache_path: Path, *, tag: str = "[CVSTEM-LQR]", **cfg) -> dict | None:
    """Reload a cached joint solve, or None if it is missing or was solved under
    a different config. Every knob that enters the program is checked — a stale
    hit here would deploy a metric certifying a different rate than the yaml
    claims, which no downstream check would catch."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return None
    try:
        d = np.load(cache_path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001 — a corrupt cache must re-solve, not crash
        print(f"{tag} Ignoring unreadable cache {cache_path}: {type(e).__name__}: {e}")
        return None
    for key, want in cfg.items():
        stored = f"cfg_{key}"
        if stored not in d.files:
            print(f"{tag} Cache {cache_path.name} predates '{key}' — re-solving.")
            return None
        got = d[stored].item() if d[stored].ndim == 0 else d[stored]
        want_c = "none" if want is None else want
        same = (got == want_c) if isinstance(want_c, str) else np.allclose(
            float(got), float(want_c), rtol=1e-9, atol=0.0)
        if not same:
            print(f"{tag} Cache {cache_path.name} has {key}={got} but config wants "
                  f"{want_c} — re-solving.")
            return None
    print(f"{tag} Loaded cached joint SDP ({d['x'].shape[0]} states) from {cache_path} "
          f"— skipping the solve.")
    return {"x": d["x"], "W": d["W"], "cholM": d["cholM"],
            "nu": float(d["nu"]), "chi": float(d["chi"]),
            "J": float(d["J"]), "lbd": float(d["lbd"])}


def regress_cholm(
    net,
    dataset: dict,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    lr_scheduler: str = "",
    lr_scheduler_kwargs: dict | None = None,
    device="cpu",
    tag: str = "[CVSTEM-LQR]",
    val_frac: float = 0.1,
    early_stop_patience: int = 10,
) -> dict:
    """MSE-fit ``CholMetric`` to the ``chol(M)`` labels — the reference's ``model.fit``.

    Loss is on the cholesky vector, not on ``M``: that is what the reference
    regresses, and it keeps the objective linear in the network output (no
    ``eigh``/inverse in the loop, unlike ``regress_cmg``'s bounded-``W`` fit).
    """
    x = torch.as_tensor(dataset["x"]).to(torch.float32)
    y = torch.as_tensor(dataset["cholM"]).to(torch.float32)
    n = x.shape[0]
    train_idx, val_idx = train_val_split(n, val_frac, device="cpu")
    n_train = train_idx.shape[0]
    x_val, y_val = x[val_idx].to(device), y[val_idx].to(device)
    stopper = EarlyStopper(patience=early_stop_patience if val_idx.shape[0] > 0 else 0)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = build_lr_scheduler(opt, lr_scheduler, lr_scheduler_kwargs)
    net.train()
    losses: list[float] = []
    pbar = _tqdm.tqdm(range(epochs), desc=f"{tag} chol(M) regression", file=sys.stdout)
    for epoch in pbar:
        perm = train_idx[torch.randperm(n_train)]
        iters = max(1, n_train // batch_size)
        total = 0.0
        for b in range(iters):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            loss = F.mse_loss(net.net(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / iters)
        if scheduler is not None:
            scheduler.step()
        postfix = {"mse": f"{losses[-1]:.4g}"}
        stop_val = False
        if val_idx.shape[0] > 0:
            net.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(net.net(x_val), y_val).item()
            net.train()
            postfix["val"] = f"{val_loss:.4g}"
            stop_val = stopper.step(val_loss, net, epoch)
        pbar.set_postfix(**postfix)
        if stop_val:
            print(f"{tag} chol(M) regression early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val MSE {stopper.best:.4g}).")
            break
    pbar.close()
    if val_idx.shape[0] > 0:
        stopper.restore_best(net)
    net.eval()
    return {"loss_history": losses, "final_loss": losses[-1] if losses else float("nan"),
            "final_val_loss": stopper.best if val_idx.shape[0] > 0 else float("nan")}

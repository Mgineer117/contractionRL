"""Shared episode-level evaluation metrics for path- and vel-tracking environments.

Used by both contractionRL (skrl, Isaac Sim) and mjrl (classic envs).
All functions operate on plain numpy arrays so they are backend-agnostic.
"""
from __future__ import annotations

import numpy as np


def episode_metrics(
    error_norms: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Compute standard tracking metrics from a 1-D per-step error norm array.

    Args:
        error_norms: shape (T,) — L2 norm of tracking error at each step.
        dt: simulation step duration in seconds.

    Returns:
        auc               – cumulative (sum) L2 error over episode
        lambda_emp        – empirical contraction rate (positive = contracting)
        contraction_flag  – fraction of steps where error strictly decreased
        performance_score – negative mean error (higher = better)
    """
    T = len(error_norms)
    if T == 0:
        return {"auc": 0.0, "lambda_emp": 0.0, "contraction_flag": 0.0, "performance_score": 0.0}

    auc = float(np.sum(error_norms))
    mean_err = auc / T
    performance_score = -mean_err

    # Empirical contraction rate: λ = -(log(e_T) - log(e_0)) / (T * dt)
    e0 = float(error_norms[0])
    eT = float(error_norms[-1])
    if e0 > 1e-8 and eT > 1e-8:
        lambda_emp = -(np.log(eT) - np.log(e0)) / max(T * dt, 1e-8)
    else:
        lambda_emp = 0.0

    # Fraction of steps where error strictly decreased (skip first step)
    if T > 1:
        decreasing = np.sum(error_norms[1:] < error_norms[:-1])
        contraction_flag = float(decreasing) / (T - 1)
    else:
        contraction_flag = 0.0

    return {
        "auc": auc,
        "lambda_emp": lambda_emp,
        "contraction_flag": contraction_flag,
        "performance_score": performance_score,
    }


def batch_episode_metrics(
    error_norms_batch: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Average episode_metrics over a batch of episodes.

    Args:
        error_norms_batch: shape (N, T) — N episodes of T steps each.
        dt: simulation step duration in seconds.

    Returns: dict of mean values across episodes.
    """
    results = [episode_metrics(row, dt) for row in error_norms_batch]
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def mean_confidence_interval(data, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and 95% CI half-width (1.96 * standard error of the mean)."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    mean = float(np.mean(data)) if n > 0 else 0.0
    if n < 2:
        return mean, 0.0
    sem = float(np.std(data, ddof=1) / np.sqrt(n))
    return mean, 1.96 * sem


def fit_exponential_envelope(
    error_trajectories: list[np.ndarray],
    dt: float,
    num_c_candidates: int = 100,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Convergence rate (lambda) and overshoot (C), exactly per the paper.

    Each error curve is first normalized by its initial value e(0) so it starts
    at 1 and C >= 1 is a pure overshoot factor (xe(t) <= C * exp(-lambda * t)).
    Then, following the paper's "Convergence rate" procedure verbatim:

      (1) The curve with the HIGHEST OVERSHOOT (largest peak of the normalized
          error) is selected. On that curve ALONE we search the convergence
          rate lambda > 0 and overshoot C >= 1 such that xe(t) <= C*exp(-lambda*t)
          for all t in [0, T] and the AUC of C*exp(-lambda*t) over [0, T] is
          minimized. This fixes C = C*.
      (2) With C* fixed, the convergence rate lambda is computed for EACH curve
          as the tightest rate that keeps it under the C* envelope:
              lambda_j = min_t (ln C* - ln xe_j(t)) / t.

    Kept identical to agents/skrl/eval_metrics.fit_exponential_envelope (the env
    layer duplicates it so it need not import from the agent layer).

    Args:
        error_trajectories: list of 1-D arrays of RAW tracking-error norms
            (normalization by e(0) is done here); sample i is at time (i+1)*dt.
        dt: environment step time [s].

    Returns:
        (C_star, lambdas). C_star is the single fixed overshoot (>= 1). lambdas
        is a 1-D array with one convergence rate per input curve (0.0 where no
        positive decaying rate bounds that curve). If no curve carries usable
        error information, returns (1.0, zeros(len(error_trajectories))).
    """
    # Normalize each curve by e(0); keep (t, e) with e > eps at t = (i+1)*dt.
    norm_curves: list[tuple[np.ndarray, np.ndarray]] = []
    peaks = []  # peak normalized error (overshoot) per curve; -inf if unusable
    for traj in error_trajectories:
        traj = np.asarray(traj, dtype=np.float64).reshape(-1)
        if traj.size == 0 or traj[0] <= eps:
            norm_curves.append((np.empty(0), np.empty(0)))
            peaks.append(-np.inf)
            continue
        e = traj / traj[0]
        t = dt * np.arange(1, traj.size + 1)  # (i+1)*dt: avoids divide-by-zero
        keep = np.isfinite(e) & (e > eps)  # drop NaN/Inf from any diverged step
        t, e = t[keep], e[keep]
        norm_curves.append((t, e))
        peaks.append(float(e.max()) if e.size else -np.inf)

    peaks = np.asarray(peaks)
    if not np.any(np.isfinite(peaks)):
        return 1.0, np.zeros(len(error_trajectories))

    # (1) Highest-overshoot curve → search C* minimizing the envelope's
    #     [0, T] AUC = C*(1 - exp(-lambda*T)) / lambda.
    t_star, e_star = norm_curves[int(np.argmax(peaks))]
    log_e_star = np.log(e_star)
    T_star = float(t_star.max())
    peak_star = float(e_star.max())
    start_C = max(1.0, peak_star)
    c_candidates = np.linspace(start_C, start_C * 10.0, num=num_c_candidates)

    C_star, min_auc = start_C, float("inf")
    for C_test in c_candidates:
        if peak_star > C_test:  # envelope must sit above the curve's peak
            continue
        lbd = float(np.min((np.log(C_test) - log_e_star) / t_star))
        if lbd <= 0:
            continue
        auc = C_test * (1.0 - np.exp(-lbd * T_star)) / lbd
        if auc < min_auc:
            min_auc, C_star = auc, float(C_test)

    # (2) Fix C*, compute the tightest lambda for each curve.
    log_C = np.log(C_star)
    lambdas = np.zeros(len(error_trajectories))
    for j, (t, e) in enumerate(norm_curves):
        if t.size == 0:
            continue
        lambdas[j] = max(float(np.min((log_C - np.log(e)) / t)), 0.0)

    return C_star, lambdas

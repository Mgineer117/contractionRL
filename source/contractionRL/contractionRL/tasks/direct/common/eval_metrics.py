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
) -> tuple[float, float]:
    """Fit C, lambda minimizing AUC (= C/lambda) of the bounding exponential.

    Ported from CAC-dev's evaluator (see agents/skrl/eval_metrics.py, which
    historically carried this same function for the post-training evaluator —
    duplicated here so the env layer (this module) doesn't have to import
    from the agent layer).

    For a candidate C, the tightest valid rate is
        lambda(C) = min over all samples of (ln C - ln e_i) / t_i
    (the bound must hold at every point of every trajectory). C is searched on
    a grid from the peak observed error to 10x that peak, and the (C, lambda)
    pair with the smallest C/lambda wins.

    Args:
        error_trajectories: list of 1-D arrays of (typically normalized)
            tracking errors; sample i of a trajectory is at time (i+1)*dt.
        dt: environment step time [s].

    Returns:
        (C, lambda). lambda == 0.0 signals that no decaying envelope fits
        (error never decays); C then falls back to the peak error.
    """
    ts, es = [], []
    for traj in error_trajectories:
        traj = np.asarray(traj, dtype=np.float64).reshape(-1)
        t = dt * np.arange(1, len(traj) + 1)  # (i+1)*dt: avoids divide-by-zero
        keep = traj > eps
        ts.append(t[keep])
        es.append(traj[keep])
    if not ts or sum(len(t) for t in ts) == 0:
        return 1.0, 0.0
    t_all = np.concatenate(ts)
    e_all = np.concatenate(es)

    global_max_err = float(e_all.max())
    start_C = max(1.0, global_max_err)
    c_candidates = np.linspace(start_C, start_C * 10.0, num=num_c_candidates)

    log_e = np.log(e_all)
    best_C, best_lbd, min_auc = start_C, 0.0, float("inf")
    for C_test in c_candidates:
        if global_max_err > C_test:
            continue
        lbd = float(np.min((np.log(C_test) - log_e) / t_all))
        if lbd <= 0:
            continue
        auc = C_test / lbd
        if auc < min_auc:
            min_auc, best_C, best_lbd = auc, float(C_test), lbd

    return best_C, best_lbd

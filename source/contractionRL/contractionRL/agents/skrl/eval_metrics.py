"""Evaluation metrics for tracking policies.

Ports the CAC-dev evaluator's contraction analysis (CAC-dev/trainer/evaluator.py)
to a vectorized, dependency-free form:

  * fit_exponential_envelope — find (C, lambda) such that every error sample
    satisfies e(t) <= C * exp(-lambda * t) while minimizing the envelope's
    AUC = C / lambda ("tightest exponential that still bounds the data").
  * mean_confidence_interval — mean and 95% CI half-width (1.96 * SEM).
"""
from __future__ import annotations

import numpy as np


def fit_exponential_envelope(
    error_trajectories: list[np.ndarray],
    dt: float,
    num_c_candidates: int = 100,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Fit C, lambda minimizing AUC (= C/lambda) of the bounding exponential.

    For a candidate C, the tightest valid rate is
        lambda(C) = min over all samples of (ln C - ln e_i) / t_i
    (the bound must hold at every point of every trajectory). C is searched on
    a grid from the peak observed error to 10x that peak, and the (C, lambda)
    pair with the smallest C/lambda wins — exactly the CAC-dev procedure, but
    vectorized over samples instead of a Python double-loop.

    Args:
        error_trajectories: list of 1-D arrays of (typically normalized)
            tracking errors; sample i of a trajectory is at time (i+1)*dt.
        dt: environment step time [s].

    Returns:
        (C, lambda). lambda == 0.0 signals that no decaying envelope fits
        (error never decays); C then falls back to the peak error.
    """
    # Flatten to (t_i, e_i) samples, dropping ~zero errors (log-undefined)
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
        # C must sit above every sample for the envelope to be feasible
        if global_max_err > C_test:
            continue
        lbd = float(np.min((np.log(C_test) - log_e) / t_all))
        if lbd <= 0:
            continue
        auc = C_test / lbd
        if auc < min_auc:
            min_auc, best_C, best_lbd = auc, float(C_test), lbd

    return best_C, best_lbd


def mean_confidence_interval(data, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and 95% CI half-width (1.96 * standard error of the mean)."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    mean = float(np.mean(data))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(data, ddof=1) / np.sqrt(n))
    return mean, 1.96 * sem

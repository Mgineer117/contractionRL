"""Unified contraction/tracking metrics + wandb logging — single source of truth.

Every algorithm (PPO / SAC / C3M / C2RL / LQR / SD-LQR) and every path-tracking
environment reports the SAME four stability metrics, computed the SAME
memory-efficient *streaming* way — from per-env running accumulators only
(e0, e_last, e_max, running sum of error norms, step count) — never storing the
full ``(num_envs, T)`` error tensor:

  * ``auc``                — area under the NORMALIZED error curve e(t)/e(0),
                             dt-weighted trapezoidal rule.  Lower is better.
  * ``contraction_rate``   — empirical exponential rate ``lambda`` from the
                             endpoints: ``e(T) = e(0)·exp(-lambda·T·dt)`` ⇒
                             ``lambda = -ln(e_T / e_0) / (T·dt)`` (clamped ≥ 0).
                             Logged both as ``contraction_rate`` and the
                             user-facing alias ``lambda``.
  * ``overshoot``          — peak normalized error ``e_max / e(0)`` (≥ 1 in
                             theory; a pure overshoot factor).
  * ``contraction_score``  — ``lambda / overshoot`` (higher = fast contraction
                             with little overshoot).

Each per-env quantity is reduced across the env population to a mean and a 95%
CI half-width (``1.96·SEM``, see :func:`mean_confidence_interval`).

Why streaming (and why the math is exactly the trapezoid): if ``err_sum`` is the
running sum ``Σ_{k=0}^{T-1} e_k`` of the raw error norms recorded at times
``t_k = k·dt``, then the trapezoidal integral of the sequence over
``[t_0, t_{T-1}]`` is

    ∫ e dt ≈ dt·(e_0/2 + e_1 + … + e_{T-2} + e_{T-1}/2)
           = dt·(Σ_k e_k − e_0/2 − e_{T-1}/2)
           = dt·(err_sum − 0.5·e_0 − 0.5·e_last),

so the NORMALIZED AUC ``∫ (e/e_0) dt = dt/e_0·(err_sum − 0.5·e_0 − 0.5·e_last)``
needs only ``err_sum``, ``e_0`` and ``e_last`` — no full curve.  (Both endpoints
get half weight; subtracting only ``e_0`` — or, worse, *adding* ``e_last`` — is
a trapezoid-rule error that earlier per-algorithm copies of this code had.)
"""

from __future__ import annotations

import io
import sys

import numpy as np
import torch

from .eval_metrics import mean_confidence_interval

# The four stability metrics every path-tracking algorithm/env must log.
METRIC_NAMES = ("auc", "contraction_rate", "overshoot", "contraction_score")


# ─────────────────────────────────────────────────────────────────────────── #
# Streaming metric math
# ─────────────────────────────────────────────────────────────────────────── #

def per_env_metrics(
    *,
    e0: torch.Tensor,
    e_last: torch.Tensor,
    e_max: torch.Tensor,
    err_sum: torch.Tensor,
    steps: torch.Tensor,
    dt: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Per-env metric tensors from streaming accumulators (see module docstring).

    All accumulators are ``(N,)`` or ``(N, 1)`` tensors on the same device;
    ``dt`` is the physical step time [s]. ``err_sum`` is ``Σ_k e_k`` over the
    episode's recorded steps, ``steps`` is that episode's length ``T`` (≥ 1).
    """
    e0c = e0.clamp(min=eps)
    eTc = e_last.clamp(min=eps)
    # T recorded samples (t = 0, dt, …, (T-1)·dt) span (T-1) intervals, i.e.
    # (T-1)·dt of elapsed time — the same interval count the trapezoid AUC uses.
    # clamp(min=1): the T==1 case has e_last==e0 so the rate is 0 regardless, and
    # this just avoids a 0·dt divisor.
    elapsed = (steps - 1).clamp(min=1) * dt
    # Normalized dt-weighted trapezoidal AUC (both endpoints half-weighted).
    auc = ((dt / e0c) * (err_sum - 0.5 * e0 - 0.5 * e_last)).clamp(min=0.0)
    # Empirical contraction rate from the endpoints; ≥ 0 (a negative raw value
    # just means the error grew — no contraction observed — not a valid rate).
    contraction_rate = (-(torch.log(eTc) - torch.log(e0c)) / elapsed).clamp(min=0.0)
    overshoot = (e_max.clamp(min=eps) / e0c).clamp(min=1e-6)
    contraction_score = contraction_rate / overshoot
    return {
        "auc": auc,
        "contraction_rate": contraction_rate,
        "overshoot": overshoot,
        "contraction_score": contraction_score,
    }


def summarize(
    per_env: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Reduce per-env metric tensors to ``{name}_mean`` / ``{name}_ci95`` floats.

    ``mask`` (bool, ``(N,)``) restricts the reduction to the envs that carry
    valid information (e.g. finished their episode); ``None`` uses all envs.
    """
    out: dict[str, float] = {}
    for name, vals in per_env.items():
        v = vals.reshape(-1)
        if mask is not None:
            v = v[mask.reshape(-1)]
        arr = v.detach().cpu().numpy()
        m, ci = mean_confidence_interval(arr, confidence)
        out[f"{name}_mean"] = float(m)
        out[f"{name}_ci95"] = float(ci)
    return out


class StreamingErrorStats:
    """Per-env streaming accumulators for the four contraction metrics.

    Feed one env step at a time via :meth:`update`, then call :meth:`summary`
    (mean/ci95 dict) or :meth:`metrics` (per-env tensors).  ``active`` freezes
    the accumulators of envs that already finished their (first) episode, so a
    non-terminating eval loop over ``max_episode_length`` steps measures exactly
    one episode per env.
    """

    def __init__(self, num_envs: int, device) -> None:
        self._device = device
        z = lambda: torch.zeros((num_envs, 1), device=device)
        self.e0 = z()
        self.e_last = z()
        self.e_max = z()
        self.err_sum = z()
        self.steps = z()

    def update(self, error: torch.Tensor, active: torch.Tensor | None = None) -> None:
        error = error.reshape(-1, 1).float()
        active = torch.ones_like(error) if active is None else active.reshape(-1, 1).float()
        is_first = (self.steps == 0) & (active > 0)
        self.e0 = torch.where(is_first, error, self.e0)
        self.e_last = torch.where(active > 0, error, self.e_last)
        self.e_max = torch.where(active > 0, torch.maximum(self.e_max, error), self.e_max)
        self.err_sum = self.err_sum + error * active
        self.steps = self.steps + active

    def metrics(self, dt: float) -> dict[str, torch.Tensor]:
        return per_env_metrics(
            e0=self.e0, e_last=self.e_last, e_max=self.e_max,
            err_sum=self.err_sum, steps=self.steps, dt=dt,
        )

    def summary(self, dt: float, mask: torch.Tensor | None = None) -> dict[str, float]:
        return summarize(self.metrics(dt), mask)


def reward_summary(
    total_reward: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """``total_reward_{max,min,mean,ci95}`` of per-env episodic total reward."""
    v = total_reward.reshape(-1)
    if mask is not None:
        v = v[mask.reshape(-1)]
    arr = v.detach().cpu().numpy()
    if arr.size == 0:
        return {f"total_reward_{s}": 0.0 for s in ("max", "min", "mean", "ci95")}
    m, ci = mean_confidence_interval(arr)
    return {
        "total_reward_max": float(np.max(arr)),
        "total_reward_min": float(np.min(arr)),
        "total_reward_mean": m,
        "total_reward_ci95": ci,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Unified wandb logging keys
# ─────────────────────────────────────────────────────────────────────────── #

def track_stability_summary(agent, summary: dict[str, float], *, tab: str = "Stability") -> None:
    """Push a :func:`summarize` dict onto ``agent`` under ``{tab}/...``.

    Emits, for every metric, ``{tab}/{name}_mean``, ``{tab}/{name}_ci95`` and a
    bare ``{tab}/{name}`` alias (== the mean; kept because several existing
    dashboards/checkpoint patches read the bare key).  ``contraction_rate`` is
    additionally mirrored to the user-facing name ``lambda``.
    """
    for k, v in summary.items():
        agent.track_data(f"{tab}/{k}", float(v))
        if k.endswith("_mean"):
            agent.track_data(f"{tab}/{k[:-len('_mean')]}", float(v))
    if "contraction_rate_mean" in summary:
        agent.track_data(f"{tab}/lambda", float(summary["contraction_rate_mean"]))
        agent.track_data(f"{tab}/lambda_mean", float(summary["contraction_rate_mean"]))
        if "contraction_rate_ci95" in summary:
            agent.track_data(f"{tab}/lambda_ci95", float(summary["contraction_rate_ci95"]))


def track_reward_summary(agent, summary: dict[str, float], *, tab: str = "Reward") -> None:
    """Push a :func:`reward_summary` dict onto ``agent`` under ``{tab}/...``."""
    for k, v in summary.items():
        agent.track_data(f"{tab}/{k}", float(v))


def reward_log_dict(summary: dict[str, float], device, *, tab: str = "Reward") -> dict:
    """``extras['log']``-style dict of scalar tensors mirroring
    :func:`track_reward_summary` (``{tab}/total_reward_{max,min,mean,ci95}``).

    For environments that log per-episode reward through Isaac Lab's
    ``extras['log']`` (surfaced to skrl as scalar tensors via
    ``environment_info: log``), so PPO/SAC/C2RL land on the SAME
    ``Reward/total_reward_mean`` key — at the SAME per-episode-reset cadence as
    :func:`stability_log_dict` — that C3M's eval loop emits via
    :func:`track_reward_summary`. Without this, that key was only ever written
    once, by the post-training evaluator in train.py.
    """
    return {f"{tab}/{k}": torch.tensor(float(v), device=device) for k, v in summary.items()}


def stability_log_dict(summary: dict[str, float], device, *, tab: str = "Stability") -> dict:
    """``extras['log']``-style dict of scalar tensors mirroring
    :func:`track_stability_summary` (``{tab}/{name}_mean``, ``_ci95``, a bare
    ``{tab}/{name}`` alias and the ``lambda`` alias).

    For environments that log per-episode metrics through Isaac Lab's
    ``extras['log']`` (surfaced to skrl as scalar tensors), so PPO/SAC/LQR/SD-LQR
    land on the exact same wandb keys the contraction trainers emit via
    :func:`track_stability_summary`.
    """
    out: dict = {}
    for k, v in summary.items():
        out[f"{tab}/{k}"] = torch.tensor(float(v), device=device)
        if k.endswith("_mean"):
            out[f"{tab}/{k[:-len('_mean')]}"] = torch.tensor(float(v), device=device)
    if "contraction_rate_mean" in summary:
        out[f"{tab}/lambda"] = torch.tensor(float(summary["contraction_rate_mean"]), device=device)
        out[f"{tab}/lambda_mean"] = torch.tensor(float(summary["contraction_rate_mean"]), device=device)
        if "contraction_rate_ci95" in summary:
            out[f"{tab}/lambda_ci95"] = torch.tensor(float(summary["contraction_rate_ci95"]), device=device)
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Unified trajectory / normalized-error plots
# ─────────────────────────────────────────────────────────────────────────── #

def _wandb_run():
    """Return the active wandb run (or None) without importing wandb eagerly."""
    if "wandb" not in sys.modules:
        return None
    return getattr(sys.modules["wandb"], "run", None)


def log_tracking_plots(
    traj_x: dict,
    traj_xref: dict,
    traj_error: dict,
    *,
    dt: float,
    prefix: str = "train",
    step: int | None = None,
    title: str | None = None,
) -> None:
    """Push ``{prefix}/normalized_error`` and ``{prefix}/path_tracking`` to wandb.

    ``traj_x`` / ``traj_xref`` map an env index → list of per-step position
    arrays; ``traj_error`` maps env index → list of per-step scalar error norms.
    Dimensionality of the position vectors selects a 1-D (vs time), 2-D or 3-D
    trajectory plot.  No-op when wandb is inactive or nothing was collected.
    ``prefix`` is the full leading key path — e.g. ``"train"`` for single-policy
    algorithms, ``"train/con"`` / ``"train/opt"`` for C2RL's two policies.
    """
    if _wandb_run() is None:
        return
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image

    label = title or prefix

    def _push(fig, key):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        try:
            payload = {f"{prefix}/{key}": wandb.Image(Image.open(buf))}
            if step is not None:
                payload["global_step"] = step
            wandb.log(payload)
        except Exception:
            pass

    # ── Normalized error curve(s) ─────────────────────────────────────────── #
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    fig_err, ax_err = plt.subplots(figsize=(6, 4))
    err_plotted = False
    for i, errs in traj_error.items():
        if not errs:
            continue
        errs_arr = np.asarray(errs, dtype=np.float64)
        e0 = max(float(errs_arr[0]), 1e-8)
        norm = errs_arr / e0
        auc = float(_trapz(norm, dx=float(dt)))
        ax_err.plot(norm, label=f"Env {i} (AUC: {auc:.2f})")
        err_plotted = True
    if err_plotted:
        ax_err.set_title(f"{label} Normalized Error")
        ax_err.set_xlabel("Step")
        ax_err.set_ylabel("Normalized Error")
        ax_err.legend(fontsize="small")
        _push(fig_err, "normalized_error")
    else:
        plt.close(fig_err)

    # ── Position trajectory vs reference ──────────────────────────────────── #
    fig = plt.figure(figsize=(6, 5))
    ax = None
    pos_plotted = False
    for i in traj_x:
        xs = traj_x.get(i) or []
        refs = traj_xref.get(i) or []
        if len(xs) == 0 or len(refs) == 0:
            continue
        tx = np.asarray(xs, dtype=np.float64)
        txref = np.asarray(refs, dtype=np.float64)
        d = min(tx.shape[-1], txref.shape[-1])
        if d < 1:
            continue
        if ax is None:
            ax = fig.add_subplot(111, projection="3d") if d >= 3 else fig.add_subplot(111)
        t = np.arange(len(tx))
        if d == 1:
            ax.scatter(t, tx[:, 0], c=t, cmap="viridis", s=10, label=f"x (env {i})")
            ax.plot(t, txref[:, 0], "--", color="red", label=f"x_ref (env {i})")
            ax.set_xlabel("Step"); ax.set_ylabel("Position")
        elif d == 2:
            ax.scatter(tx[:, 0], tx[:, 1], c=t, cmap="viridis", s=10, label=f"x (env {i})")
            ax.plot(txref[:, 0], txref[:, 1], "--", color="red", label=f"x_ref (env {i})")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        else:
            ax.scatter(tx[:, 0], tx[:, 1], tx[:, 2], c=t, cmap="viridis", s=10, label=f"x (env {i})")
            ax.plot(txref[:, 0], txref[:, 1], txref[:, 2], "--", color="red", label=f"x_ref (env {i})")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        pos_plotted = True
    if pos_plotted:
        ax.set_title(f"{label} Path Tracking")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        _push(fig, "path_tracking")
    else:
        plt.close(fig)

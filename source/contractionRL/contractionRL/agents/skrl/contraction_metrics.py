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
import warnings

import numpy as np
import torch

from contractionRL.tasks.direct.common.eval_metrics import mean_confidence_interval

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


class StatManager:
    """Unified, memory-bounded, trajectory-wise path-tracking metric collector.

    One class for every algorithm (C3M/C2RL/PPO/SAC/LQR/SD-LQR) and both env
    backends (classic ``SyncVectorEnv`` eval rollouts and batched Isaac envs).
    Fed one env-step at a time via :meth:`update`, it stores each *tracked* env's
    per-step error TRAJECTORY for the current episode, then computes the four
    contraction metrics (``auc``/``contraction_rate``/``overshoot``/
    ``contraction_score``, mean + 95% CI) via the SAME :func:`per_env_metrics` /
    :func:`summarize` used everywhere else — so keys, formulas and CI are
    unchanged.

    Memory: only the first ``num_envs_for_eval`` of ``num_envs`` envs are tracked
    (PPO uses ~4096 envs; keeping full trajectories for all of them is wasteful).
    A request larger than ``num_envs`` is clamped to ``num_envs`` with a warning.

    Robustness (the point of this class): a per-env trajectory is only
    ``available`` once anchored at a **detected initialization** — the caller must
    pass ``initialized`` sourced from the true per-env episode counter
    (``time_steps == 0`` for classic envs, ``episode_length_buf == 0`` for Isaac),
    NOT "the first update". If an env is never seen to initialize (e.g. the vector
    env is left parked at a converged terminal state — the skrl reset-cache +
    gymnasium 1.x next-step-autoreset failure where ``x ≈ xref`` so ``e0 ≈ 0`` and
    ``auc = Σe/e0`` explodes to ~1e7), :meth:`summary` still returns a best-effort
    value BUT emits a warning and reports ``available=False`` — the failure is
    loud instead of a silent 1e7. ``episode_length`` (fixed for path-tracking) is
    stored so an unanchored/partial episode can still be length-normalized.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device,
        *,
        episode_length: int | None = None,
        num_envs_for_eval: int | None = None,
        keep_trajectories: bool = True,
    ) -> None:
        if num_envs_for_eval is None:
            num_envs_for_eval = num_envs
        elif num_envs_for_eval > num_envs:
            warnings.warn(
                f"StatManager: num_envs_for_eval ({num_envs_for_eval}) > num_envs "
                f"({num_envs}); clamping to {num_envs}.",
                stacklevel=2,
            )
            num_envs_for_eval = num_envs
        self.n = int(num_envs_for_eval)
        self.dt = float(dt)
        self.device = device
        self.episode_length = int(episode_length) if episode_length is not None else None
        self.keep_trajectories = keep_trajectories
        # Track the first ``n`` envs (deterministic, cheap). For a random subset,
        # pass a pre-permuted env order upstream.
        self._idx = torch.arange(self.n, device=device)
        self.reset()

    def reset(self) -> None:
        self._buf: list[list[float]] = [[] for _ in range(self.n)]        # current episode
        self._done: list[list[float] | None] = [None] * self.n           # last completed episode
        self._started = [False] * self.n                                 # current buf anchored at a real init?
        self._done_started = [False] * self.n                            # was _done anchored at a real init?
        self._traj_x: list[list] = [[] for _ in range(self.n)]
        self._traj_xref: list[list] = [[] for _ in range(self.n)]
        self._done_x: list[list | None] = [None] * self.n
        self._done_xref: list[list | None] = [None] * self.n

    def _sel(self, t: torch.Tensor) -> torch.Tensor:
        """Take the tracked subset from a per-env tensor (accepts (N,) or (N,1))."""
        return t.reshape(t.shape[0], -1)[self._idx] if t.dim() > 1 else t.reshape(-1)[self._idx]

    def update(
        self,
        error: torch.Tensor,
        initialized: torch.Tensor | None = None,
        finished: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
        xref: torch.Tensor | None = None,
    ) -> None:
        """Record one env-step for the tracked subset.

        error       : (N,) or (N,1) per-env scalar tracking error at the CURRENT state.
        initialized : (N,) bool — env just started a fresh episode (episode counter == 0).
                      Anchors ``e0``. Source from real env state, not "first update".
        finished    : (N,) bool — env's episode ended AFTER this step; its buffer is
                      snapshotted as the reportable completed episode.
        active      : (N,) bool — env is still being recorded (skip frozen/finished envs).
                      Defaults to "all active".
        x, xref     : optional (N, pos_dim) current/reference positions for plotting.
        """
        err = self._sel(error).detach().reshape(-1).float().cpu()
        init = self._sel(initialized).detach().reshape(-1).bool().cpu() if initialized is not None else None
        fin = self._sel(finished).detach().reshape(-1).bool().cpu() if finished is not None else None
        act = self._sel(active).detach().reshape(-1).bool().cpu() if active is not None else None
        xs = self._sel(x).detach().cpu() if (x is not None and self.keep_trajectories) else None
        xr = self._sel(xref).detach().cpu() if (xref is not None and self.keep_trajectories) else None

        for i in range(self.n):
            if act is not None and not bool(act[i]):
                continue  # env frozen (already finished its first episode) — record nothing
            if init is not None and bool(init[i]):
                # Fresh episode start → (re)anchor the buffer on a real init.
                self._buf[i] = []
                self._traj_x[i] = []
                self._traj_xref[i] = []
                self._started[i] = True
            self._buf[i].append(float(err[i]))
            if xs is not None:
                self._traj_x[i].append(xs[i].numpy())
                if xr is not None:
                    self._traj_xref[i].append(xr[i].numpy())
            if fin is not None and bool(fin[i]) and self._buf[i]:
                # Episode ended → snapshot as the reportable completed episode.
                self._done[i] = self._buf[i]
                self._done_started[i] = self._started[i]
                self._done_x[i] = self._traj_x[i]
                self._done_xref[i] = self._traj_xref[i]
                self._buf[i] = []
                self._traj_x[i] = []
                self._traj_xref[i] = []
                self._started[i] = False

    def _episodes(self) -> tuple[list[list[float]], list[bool]]:
        """Per-tracked-env reportable trajectory (completed if present, else current)."""
        trajs, started = [], []
        for i in range(self.n):
            if self._done[i] is not None:
                trajs.append(self._done[i]); started.append(self._done_started[i])
            else:
                trajs.append(self._buf[i]); started.append(self._started[i])
        return trajs, started

    @property
    def available(self) -> bool:
        """True iff at least one tracked env has an init-anchored trajectory."""
        _, started = self._episodes()
        trajs, _ = self._episodes()
        return any(s and len(t) > 0 for s, t in zip(started, trajs))

    def metrics(self) -> dict[str, torch.Tensor]:
        """Per-tracked-env metric tensors (see :func:`per_env_metrics`)."""
        trajs, _ = self._episodes()
        dev = self.device
        z = lambda vals: torch.tensor(vals, device=dev, dtype=torch.float32).reshape(-1, 1)
        # Fixed-length fallback: an env with no recorded data contributes a
        # neutral, non-exploding entry (masked out by summary()).
        e0 = z([t[0] if t else 1.0 for t in trajs])
        e_last = z([t[-1] if t else 1.0 for t in trajs])
        e_max = z([max(t) if t else 1.0 for t in trajs])
        err_sum = z([sum(t) if t else 0.0 for t in trajs])
        steps = z([len(t) if t else (self.episode_length or 1) for t in trajs])
        return per_env_metrics(e0=e0, e_last=e_last, e_max=e_max, err_sum=err_sum, steps=steps, dt=self.dt)

    def summary(self) -> dict[str, float]:
        """Mean/ci95 over tracked envs; warns and reports over ALL non-empty envs
        (unanchored included) when no env has an init-anchored trajectory."""
        trajs, started = self._episodes()
        anchored = torch.tensor([s and len(t) > 0 for s, t in zip(started, trajs)], device=self.device)
        if bool(anchored.any()):
            mask = anchored
        else:
            nonempty = torch.tensor([len(t) > 0 for t in trajs], device=self.device)
            warnings.warn(
                "StatManager.summary(): no env has an init-anchored trajectory — the "
                "eval env was likely never (re)initialized (parked terminal state / "
                "reset-cache bug). Reporting best-effort metrics over non-anchored "
                "episodes; e0 may be unreliable.",
                stacklevel=2,
            )
            mask = nonempty if bool(nonempty.any()) else None
        return summarize(self.metrics(), mask)

    def trajectories(self) -> tuple[dict, dict, dict]:
        """``(traj_x, traj_xref, traj_error)`` for :func:`log_tracking_plots`
        (env-index → per-step list), over the reportable episodes."""
        trajs, _ = self._episodes()
        traj_error = {i: list(trajs[i]) for i in range(self.n) if trajs[i]}
        tx = {i: (self._done_x[i] if self._done[i] is not None else self._traj_x[i])
              for i in range(self.n)}
        txr = {i: (self._done_xref[i] if self._done[i] is not None else self._traj_xref[i])
               for i in range(self.n)}
        traj_x = {i: v for i, v in tx.items() if v}
        traj_xref = {i: v for i, v in txr.items() if v}
        return traj_x, traj_xref, traj_error


class StatManagerEnvWrapper:
    """Env wrapper that auto-collects path-tracking metrics via :class:`StatManager`.

    Drop-in wrapper around a (skrl-wrapped) vector env whose observation is
    ``[x, xref, uref]``. It removes the need for every algorithm's eval loop to
    hand-thread ``stats.update(...)``: ``reset()`` starts a fresh metric window,
    ``step()`` computes each env's tracking error ``‖x−xref‖`` from the returned
    obs and feeds a :class:`StatManager`, anchoring ``e0`` on the env's REAL
    episode counter (``time_steps == 0``) and freezing each env after its first
    episode end — so one ``reset()`` + rollout yields one clean per-env episode.
    Read results with :meth:`stability_summary` / :meth:`trajectories`. Every
    other attribute/method forwards to the inner env, so it is transparent to the
    trainer, skrl, and ``WandbPlotWrapper``.

    Config (``x_dim``/``dt``/``episode_length``/``num_envs``) is derived from the
    env on the first reset via ``get_attr``; recording silently no-ops if that
    fails, so non-path-tracking envs keep working. Only ``num_envs_for_eval`` envs
    are tracked (trajectory memory bound; default 64).
    """

    def __init__(self, env, *, num_envs_for_eval: int = 64):
        self.env = env
        self._num_envs_for_eval = num_envs_for_eval
        self._stats: StatManager | None = None
        self._finished: torch.Tensor | None = None
        self._x_dim: int | None = None
        self._pos_dim: int | None = None

    def __getattr__(self, name):
        # Fires only for attributes not found on the wrapper itself. Guard the
        # inner-env handle to avoid infinite recursion before __init__ sets it.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _first_attr(self, *names, default=None):
        for n in names:
            try:
                v = self.env.get_attr(n)
                return v[0] if isinstance(v, (list, tuple)) else v
            except Exception:
                continue
        return default

    def _device(self):
        return getattr(self.env, "device", "cpu")

    def _ensure_stats(self) -> bool:
        if self._stats is not None:
            return True
        x_dim = self._first_attr("num_dim_x")
        dt = self._first_attr("step_dt", "dt")
        ep = self._first_attr("max_episode_length", "max_episode_len")
        if x_dim is None or dt is None:
            return False
        self._x_dim = int(x_dim)
        pd = self._first_attr("pos_dimension", default=x_dim)
        self._pos_dim = int(pd) if pd is not None else int(x_dim)
        num_envs = int(getattr(self.env, "num_envs", 1))
        self._stats = StatManager(
            num_envs, float(dt), self._device(),
            episode_length=(int(ep) + 1 if ep is not None else None),
            num_envs_for_eval=self._num_envs_for_eval, keep_trajectories=True,
        )
        self._finished = torch.zeros(num_envs, dtype=torch.bool, device=self._device())
        return True

    def _init_flags(self):
        try:
            ts = self.env.get_attr("time_steps")
            return torch.tensor([int(t) == 0 for t in ts], device=self._device())
        except Exception:
            return None

    def _record(self, obs: torch.Tensor) -> None:
        if not self._ensure_stats():
            return
        xd, pd = self._x_dim, self._pos_dim
        error = torch.norm(obs[:, :xd] - obs[:, xd:2 * xd], dim=-1, keepdim=True)
        self._stats.update(
            error, initialized=self._init_flags(), active=~self._finished,
            x=obs[:, :pd], xref=obs[:, xd:xd + pd],
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start a fresh metric window and record e0.
        self._stats = None
        self._finished = None
        if torch.is_tensor(obs):
            self._record(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if torch.is_tensor(obs):
            self._record(obs)  # active mask uses finished from PRIOR steps
            if self._finished is not None:
                done = (terminated | truncated).reshape(-1).to(self._finished.device).bool()
                self._finished |= done
        return obs, reward, terminated, truncated, info

    def stability_summary(self) -> dict[str, float]:
        return self._stats.summary() if self._stats is not None else {}

    def trajectories(self):
        return self._stats.trajectories() if self._stats is not None else ({}, {}, {})

    def all_finished(self) -> bool:
        return bool(self._finished.all()) if self._finished is not None else False


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

    Emits ``{tab}/{name}_mean`` and ``{tab}/{name}_ci95`` for every metric —
    no bare ``{tab}/{name}`` alias, since that would be a byte-for-byte
    duplicate of ``{name}_mean``.  ``contraction_rate`` is additionally
    mirrored to the user-facing name ``lambda`` (single ``{tab}/lambda`` +
    ``{tab}/lambda_ci95``, not also ``lambda_mean`` — same reasoning).
    """
    for k, v in summary.items():
        agent.track_data(f"{tab}/{k}", float(v))
    if "contraction_rate_mean" in summary:
        agent.track_data(f"{tab}/lambda", float(summary["contraction_rate_mean"]))
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
    :func:`track_stability_summary` (``{tab}/{name}_mean``, ``_ci95`` and the
    ``lambda`` alias — no bare ``{tab}/{name}`` duplicate, see that function).

    For environments that log per-episode metrics through Isaac Lab's
    ``extras['log']`` (surfaced to skrl as scalar tensors), so PPO/SAC/LQR/SD-LQR
    land on the exact same wandb keys the contraction trainers emit via
    :func:`track_stability_summary`.
    """
    out: dict = {}
    for k, v in summary.items():
        out[f"{tab}/{k}"] = torch.tensor(float(v), device=device)
    if "contraction_rate_mean" in summary:
        out[f"{tab}/lambda"] = torch.tensor(float(summary["contraction_rate_mean"]), device=device)
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


def log_raw_config(raw_cfg: dict | None) -> None:
    """Push the pre-filter agent/trainer/models YAML dict to wandb.config.

    skrl's own ``Agent.init()`` already logs ``dataclasses.asdict(self.cfg)``
    (i.e. only fields the algorithm's Cfg dataclass declares) — any YAML key
    ``_filter_cfg_fields`` silently dropped (typo'd sweep param, stray key
    left over from copy-pasting another algorithm's config, ...) never makes
    it into that log. This logs the *complete* dict as loaded from YAML
    (post CLI overrides, pre dataclass-filtering) under ``raw_yaml/*`` so a
    dropped key is still visible on the run instead of silently vanishing.
    No-op if wandb is inactive.
    """
    run = _wandb_run()
    if run is None or not raw_cfg:
        return
    run.config.update({"raw_yaml": raw_cfg}, allow_val_change=True)


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
        payload = {f"{prefix}/{key}": wandb.Image(Image.open(buf))}
        if step is not None:
            payload["global_step"] = step
        wandb.log(payload)

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

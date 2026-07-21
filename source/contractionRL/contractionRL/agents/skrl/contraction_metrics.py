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
import math
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



class StatManagerEnvWrapper:
    """Env wrapper that computes paper-style batched C and lambda globally across a sampled
    subset of environments by storing the full trajectories (first-come, first-eval).
    """

    def __init__(self, env, *, num_envs_for_eval: int = 64):
        self.env = env
        self._num_envs_for_eval = num_envs_for_eval
        
        self._initialized = False
        self._x_dim: int | None = None
        self._pos_dim: int | None = None
        self._dt: float | None = None
        self._max_ep_len: int | None = None

        self._recent_auc_mean: float = 1e2
        self._recent_auc_ci95: float = 0.0
        self._recent_lambda_mean: float = 0.0
        self._recent_lambda_ci95: float = 0.0
        self._recent_C: float = 1e2
        self._recent_score_mean: float = 0.0
        self._recent_score_ci95: float = 0.0
        # Bumped each time a full eval buffer is reduced to metrics — lets
        # callers (e.g. C3M's eval loop) detect that a rollout actually
        # produced FRESH numbers instead of silently re-reading stale ones.
        self._compute_count: int = 0

        # Buffer states
        self._eval_buffer = None
        self._time_buffer = None
        self._tracking_env_ids = None
        self._tracking_steps = None
        self._completed_slots = None
        self._e0 = None

        self._traj_x_buf = None
        self._traj_xref_buf = None
        self._recent_trajs = ({}, {}, {})

        # Action volatility (see _track_action_volatility). Lazily allocated on
        # the first step, from the action tensor's own shape/device — the action
        # dimension is not among the attributes _ensure_stats can discover.
        self._prev_action = None
        self._prev_done = None
        self._vol_sum = None
        self._vol_count = None
        self._episode_volatility = None
        self._episode_volatility_seen = None

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _first_attr(self, *names, default=None):
        for n in names:
            # Direct access first (skrl wrappers forward attributes; also try the
            # raw unwrapped env for Isaac Lab, whose skrl wrapper may not forward).
            for target in (self.env, getattr(self.env, "unwrapped", None)):
                if target is not None and hasattr(target, n):
                    return getattr(target, n)
            # Fallback to get_attr for standard Gymnasium VectorEnvs
            try:
                v = self.env.get_attr(n)
                return v[0] if isinstance(v, (list, tuple)) else v
            except Exception:
                continue
        return default

    def _device(self):
        return getattr(self.env, "device", "cpu")

    def _ensure_stats(self) -> bool:
        if self._initialized:
            return True
        # "num_dim_x" is the classic BaseEnv name; "x_dim" is the Isaac
        # path-tracking property (path_tracking_base.py). Both env families
        # share the [x, xref, uref] observation layout this wrapper slices.
        x_dim = self._first_attr("num_dim_x", "x_dim")
        dt = self._first_attr("step_dt", "dt")
        ep = self._first_attr("max_episode_length", "max_episode_len")
        if x_dim is None or dt is None:
            return False

        self._x_dim = int(x_dim)
        pd = self._first_attr("pos_dimension")
        self._pos_dim = int(pd) if pd is not None else min(3, int(x_dim))
        self._dt = float(dt)
        self._max_ep_len = int(ep) if ep is not None else 1000
        
        num_envs = int(getattr(self.env, "num_envs", 1))
        self._num_envs_for_eval = min(num_envs, self._num_envs_for_eval)
        N = self._num_envs_for_eval
        T = self._max_ep_len
        dev = self._device()

        self._eval_buffer = torch.zeros((N, T), dtype=torch.float32, device=dev)
        self._time_buffer = torch.zeros((N, T), dtype=torch.float32, device=dev)
        self._tracking_env_ids = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._tracking_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._completed_slots = torch.zeros(N, dtype=torch.bool, device=dev)
        self._e0 = torch.zeros(N, dtype=torch.float32, device=dev)
        
        self._traj_x_buf = [[] for _ in range(N)]
        self._traj_xref_buf = [[] for _ in range(N)]

        self._initialized = True
        return True

    def _init_flags(self):
        """Per-env bool: episode counter == 0, i.e. the env just (auto-)reset.

        Both env families reset done envs INSIDE step() and return the fresh
        episode's first observation, so counter == 0 marks exactly the obs a
        new slot must anchor e0 on. "time_steps" is the classic BaseEnv
        counter; "episode_length_buf" is Isaac Lab's.
        """
        try:
            ts = self._first_attr("time_steps", "episode_length_buf")
            if isinstance(ts, torch.Tensor):
                return (ts == 0).reshape(-1).to(self._device())
            if ts is not None:
                return torch.tensor([int(t) == 0 for t in ts], device=self._device())
        except Exception:
            pass
        return None

    def _compute_batched_metrics(self):
        N = self._num_envs_for_eval
        # Clamp once: a stored error of exactly 0 would otherwise produce
        # 0 * inf = NaN in the C search below (exp(lambda*t) overflows to inf
        # in float32 once lambda*t ≳ 88) and -inf in the log for lambda.
        errs = torch.clamp(self._eval_buffer, min=1e-8)

        # 1. AUC per env (trapezoid over the true per-slot time base)
        dt_array = self._time_buffer[:, 1:] - self._time_buffer[:, :-1]
        auc_vec = torch.sum(dt_array * 0.5 * (errs[:, :-1] + errs[:, 1:]), dim=1)

        # 2. Find curve with highest overshoot
        max_overshoots = torch.max(errs, dim=1).values
        worst_idx = torch.argmax(max_overshoots)
        x_worst = errs[worst_idx]
        t_worst = self._time_buffer[worst_idx]

        # 3. Find optimal C for worst curve: C(lambda) = max_t x(t)·e^{lambda·t}
        #    over lambda in (0, 10], keeping the C whose envelope AUC is minimal.
        lambdas = torch.linspace(0.01, 10.0, steps=1000, device=self._device())
        exp_term = torch.exp(lambdas.unsqueeze(1) * t_worst.unsqueeze(0))
        C_lambdas = torch.max(x_worst.unsqueeze(0) * exp_term, dim=1).values
        C_lambdas = torch.clamp(C_lambdas, min=1.0)

        T_max = t_worst[-1]
        auc_bounds = (C_lambdas / lambdas) * (1.0 - torch.exp(-lambdas * T_max))
        best_idx = torch.argmin(auc_bounds)
        best_C = C_lambdas[best_idx]

        # 4. With C fixed, per-env lambda = min_t (ln C - ln x(t)) / t  (t > 0)
        t_pos = torch.clamp(self._time_buffer[:, 1:], min=1e-8)  # (N, T-1)
        x_pos = errs[:, 1:]                                      # (N, T-1)

        lambda_vals = (torch.log(best_C) - torch.log(x_pos)) / t_pos
        min_lambdas = torch.min(lambda_vals, dim=1).values
        lambda_vec = torch.clamp(min_lambdas, min=0.0, max=10.0)
        score_vec = lambda_vec / torch.clamp(best_C, min=1e-6)

        auc_m, auc_ci = mean_confidence_interval(auc_vec.detach().cpu().numpy(), 0.95)
        self._recent_auc_mean = float(auc_m)
        self._recent_auc_ci95 = float(auc_ci)

        lambda_m, lambda_ci = mean_confidence_interval(lambda_vec.detach().cpu().numpy(), 0.95)
        self._recent_lambda_mean = float(lambda_m)
        self._recent_lambda_ci95 = float(lambda_ci)

        score_m, score_ci = mean_confidence_interval(score_vec.detach().cpu().numpy(), 0.95)
        self._recent_score_mean = float(score_m)
        self._recent_score_ci95 = float(score_ci)

        self._recent_C = best_C.item()
        self._compute_count += 1

        # Save trajectories + per-slot normalized error curves (consumed by
        # log_tracking_plots via trajectories()). Keyed by SLOT index — an env
        # can legitimately own two slots in one buffer round (early termination
        # + re-init), so env-id keys would silently collide.
        err_rows = self._eval_buffer.detach().cpu()
        res_x, res_xref, res_err = {}, {}, {}
        for j in range(N):
            res_x[j] = self._traj_x_buf[j]
            res_xref[j] = self._traj_xref_buf[j]
            res_err[j] = err_rows[j].tolist()
        self._recent_trajs = (res_x, res_xref, res_err)

    def _record(self, obs: torch.Tensor, info: dict | None = None) -> None:
        if not self._ensure_stats():
            return

        xd, pd = self._x_dim, self._pos_dim

        unwrapped = getattr(self.env, "unwrapped", self.env)
        if isinstance(info, dict) and "tracking_error" in info:
            # Classic BaseEnv (env_base.py): "tracking_error" is the TRUE
            # per-env squared error computed BEFORE reset_idx() overwrites
            # x_t/xref for any env whose episode just ended — same ordering
            # fix as the reward computation. Reading it from `obs` instead
            # would, for an auto-reset env, measure the fresh reset (new
            # episode) state against the OLD episode's e0/window, spiking the
            # normalized error at the episode boundary.
            error = torch.sqrt(torch.clamp(info["tracking_error"].reshape(-1).to(self._device()), min=0.0))
        elif hasattr(unwrapped, "get_tracking_error"):
            # Isaac path-tracking envs: angle-wrapped ||x - x_ref|| per env.
            err = unwrapped.get_tracking_error()
            if isinstance(err, torch.Tensor):
                error = err.reshape(-1, 1).to(self._device())
            else:
                error = torch.tensor(err, dtype=torch.float32, device=self._device()).reshape(-1, 1)
            error = error.reshape(-1)
        else:
            diff = obs[:, :xd] - obs[:, xd:2 * xd]
            angle_idx = self._first_attr("angle_idx")
            if angle_idx:
                for idx in angle_idx:
                    diff[:, idx] = (diff[:, idx] + math.pi) % (2 * math.pi) - math.pi
            error = torch.norm(diff, dim=-1, keepdim=True)
            error = error.reshape(-1)
        init_flags = self._init_flags()
        if init_flags is None:
            init_flags = torch.zeros(error.shape[0], dtype=torch.bool, device=self._device())

        # Extract values for tracking
        err_vals = error.detach()

        # At an auto-reset step the classic BaseEnv's info["tracking_error"] is
        # the OLD episode's terminal error (computed before reset_idx), while
        # obs/counters are already the NEW episode's. A slot opened here would
        # anchor e0 on a near-zero stale value, inflating every normalized
        # error in the window. Substitute the env's post-reset initial error
        # (squared norm, hence sqrt) for the freshly reset envs.
        if init_flags.any():
            init_err_sq = self._first_attr("init_tracking_error")
            if isinstance(init_err_sq, torch.Tensor):
                fresh = torch.sqrt(torch.clamp(
                    init_err_sq.reshape(-1).to(err_vals), min=0.0))
                err_vals = torch.where(init_flags.reshape(-1), fresh, err_vals)
        obs_x = obs[:, :pd].detach().cpu().numpy()
        obs_xref = obs[:, xd:xd + pd].detach().cpu().numpy()

        N = self._num_envs_for_eval

        # For each env that is initializing, complete its old slot if any, and start a new one
        init_indices = torch.nonzero(init_flags, as_tuple=True)[0]
        for env_idx in init_indices:
            # If this env was already being tracked, complete its old slot FIRST!
            old_slots = torch.nonzero(self._tracking_env_ids == env_idx, as_tuple=True)[0]
            for old_slot in old_slots:
                if not self._completed_slots[old_slot]:
                    self._completed_slots[old_slot] = True
                    # Early termination: extend the final recorded error to the
                    # end of the horizon so every slot has a full-length curve.
                    # The slot keeps its env id — a completed episode must stay
                    # in the buffer until the WHOLE buffer is reduced; freeing
                    # it here would let the very next init reuse and overwrite
                    # it, and (ids != -1).all() below could then never fire.
                    step = self._tracking_steps[old_slot]
                    if step < self._max_ep_len:
                        last_val = self._eval_buffer[old_slot, step-1] if step > 0 else 1.0
                        self._eval_buffer[old_slot, step:] = last_val
                        time_steps_pad = torch.arange(step, self._max_ep_len, device=self._device(), dtype=torch.float32)
                        self._time_buffer[old_slot, step:] = time_steps_pad * self._dt

            # Now assign a new slot (first come, first eval — no free slot means
            # this episode simply isn't tracked this round)
            empty_slots = torch.nonzero(self._tracking_env_ids == -1, as_tuple=True)[0]
            if len(empty_slots) > 0:
                slot = empty_slots[0]
                self._tracking_env_ids[slot] = env_idx
                self._tracking_steps[slot] = 0
                self._completed_slots[slot] = False
                self._traj_x_buf[slot] = []
                self._traj_xref_buf[slot] = []
                    
        # Update active slots
        active_slots = torch.nonzero((self._tracking_env_ids != -1) & (~self._completed_slots), as_tuple=True)[0]
        
        for slot in active_slots:
            env_id = self._tracking_env_ids[slot]
            step = self._tracking_steps[slot]
            
            if step == 0:
                self._e0[slot] = err_vals[env_id].clamp(min=1e-8)
            
            if step < self._max_ep_len:
                val = err_vals[env_id] / self._e0[slot]
                self._eval_buffer[slot, step] = val
                self._time_buffer[slot, step] = step * self._dt
                
                self._traj_x_buf[slot].append(obs_x[env_id])
                self._traj_xref_buf[slot].append(obs_xref[env_id])
            
            step += 1
            self._tracking_steps[slot] = step
            
            # If reached max length, pad to end
            if step >= self._max_ep_len:
                self._completed_slots[slot] = True

        # Check if all slots are completed
        if (self._tracking_env_ids != -1).all() and self._completed_slots.all():
            self._compute_batched_metrics()
            # Clear slots
            self._tracking_env_ids.fill_(-1)
            self._completed_slots.fill_(False)

    @staticmethod
    def _obs_tensor(obs):
        """Batched obs tensor from a step/reset return (Isaac wrappers may
        return an observation dict keyed by group, e.g. {"policy": ...})."""
        if isinstance(obs, dict):
            obs = obs.get("policy")
        return obs if torch.is_tensor(obs) else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # An EXTERNAL reset (e.g. C3M's per-eval reset) starts a fresh window:
        # every in-flight episode is invalidated, so drop all slots. Without
        # this, leftover partial slots would be "completed" by padding at the
        # next init detection and reduced into garbage metrics.
        if self._initialized:
            self._tracking_env_ids.fill_(-1)
            self._completed_slots.fill_(False)
        self._reset_action_volatility()
        o = self._obs_tensor(obs)
        if o is not None:
            self._record(o, info if isinstance(info, dict) else None)
        return obs, info

    def _reset_action_volatility(self) -> None:
        """Drop every in-flight action-volatility accumulator.

        Mirrors what reset() does to the eval slots: an external reset breaks
        the action sequence, so any partially accumulated episode would splice
        pre- and post-reset actions into one average. Completed per-episode
        values are KEPT — they are already-finished measurements, and the
        summary would otherwise go empty (and the key silently vanish from
        wandb) after every external reset.
        """
        self._prev_action = None
        self._prev_done = None
        if self._vol_sum is not None:
            self._vol_sum.zero_()
            self._vol_count.zero_()

    def _track_action_volatility(self, action, terminated, truncated) -> None:
        """Accumulate per-step ``||u_t - u_{t-1}||_2`` into per-episode means.

        This is a MEASUREMENT of the deployed action sequence, not a penalty —
        it is the executed action (exploration noise included), whereas CAPS'
        temporal term regularizes the policy MEAN over sampled states. The two
        are deliberately different quantities: this one is what the actuator
        actually sees, so it stays meaningful for algorithms with no CAPS term
        at all and is the thing to read when asking whether a policy is
        physically deployable.

        AUTORESET. Both env families reset done envs INSIDE ``step()`` and
        return the fresh episode's first observation (see ``_init_flags``), so
        the action taken on the step AFTER a done is computed from an unrelated
        initial state. Differencing across that boundary would report a large
        spurious jump whose size depends on the reset distribution rather than
        on the policy. Pairs are therefore skipped whenever the PREVIOUS step
        terminated or truncated for that env — which is also why the previous
        done flags are carried on ``_prev_done`` rather than read from the
        current step.

        Units are raw action units per step (NOT per second): dividing by dt
        would make the number depend on the integrator step, and dt is fixed
        within an env anyway. Compare across configs of one env, not across
        envs with different dt.
        """
        if not torch.is_tensor(action):
            return
        a = action.detach().reshape(action.shape[0], -1).float()
        n = a.shape[0]

        done = (terminated | truncated) if torch.is_tensor(terminated) else None
        done = torch.zeros(n, dtype=torch.bool, device=a.device) if done is None \
            else done.detach().reshape(-1).to(a.device)

        if self._vol_sum is None or self._vol_sum.shape[0] != n:
            z = torch.zeros(n, dtype=torch.float32, device=a.device)
            self._vol_sum, self._vol_count = z.clone(), z.clone()
            self._episode_volatility = z.clone()
            self._episode_volatility_seen = torch.zeros(n, dtype=torch.bool, device=a.device)
            self._prev_action, self._prev_done = None, None

        if self._prev_action is not None and self._prev_action.shape == a.shape:
            valid = ~self._prev_done                      # skip pairs across a reset
            step_delta = (a - self._prev_action).norm(dim=-1)
            self._vol_sum += torch.where(valid, step_delta, torch.zeros_like(step_delta))
            self._vol_count += valid.float()

        # Finalize the episodes that ended on THIS step, then clear their
        # accumulators so the next episode starts from zero.
        if bool(done.any()):
            finished = done & (self._vol_count > 0)
            if bool(finished.any()):
                mean_delta = self._vol_sum / self._vol_count.clamp(min=1.0)
                self._episode_volatility = torch.where(
                    finished, mean_delta, self._episode_volatility)
                self._episode_volatility_seen |= finished
            self._vol_sum = torch.where(done, torch.zeros_like(self._vol_sum), self._vol_sum)
            self._vol_count = torch.where(done, torch.zeros_like(self._vol_count), self._vol_count)

        self._prev_action = a.clone()
        self._prev_done = done

    def _action_volatility_summary(self) -> dict[str, float]:
        """``action_volatility_{mean,ci95,max}`` over envs with a finished episode.

        Returns {} until at least one episode has completed, so the key is
        ABSENT rather than zero — same rule the step() gate applies to the
        other stability metrics, and for the same reason: skrl's write_interval
        averages its window, so a placeholder would blend into the real value.
        """
        if self._episode_volatility_seen is None or not bool(self._episode_volatility_seen.any()):
            return {}
        v = self._episode_volatility[self._episode_volatility_seen].detach().cpu().numpy()
        m, ci = mean_confidence_interval(v)
        return {
            "action_volatility_mean": m,
            "action_volatility_ci95": ci,
            "action_volatility_max": float(np.max(v)),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._track_action_volatility(action, terminated, truncated)
        o = self._obs_tensor(obs)
        if o is not None:
            self._record(o, info if isinstance(info, dict) else None)

            # Inject metrics into info["log"] so skrl's trainer (cfg
            # `environment_info: log`, scalar tensors only) tracks them.
            #
            # Gated on _compute_count: before the first buffer round completes,
            # stability_summary() returns the CONSTRUCTOR SENTINELS (auc_mean
            # and C = 1e2), and the buffer only completes at the END of an
            # episode. Logging them from step 0 meant ~498 of a 500-step
            # episode reported 1e2, and skrl's write_interval averages its
            # window — so the value that reached wandb (and the sweep) was a
            # BLEND of sentinel and truth, not the truth:
            #     Stability/auc_mean ≈ 60 + 0.4·true_auc
            # (measured, classic-cartpole-v0 + lqr: true 7.52 → logged 63.01;
            # overshoot 3.95 → logged 61.58, the same 60/40 mix). Worse, the
            # blend ratio depends on where buffer completion lands inside the
            # write window, so it varied run to run — noise of the same order
            # as the real differences the sweeps are trying to resolve.
            # Emitting nothing until there is a real value keeps the key
            # absent instead of wrong; skrl simply has no datapoint to average.
            if self._initialized and self._compute_count > 0 and isinstance(info, dict):
                if "log" not in info or not isinstance(info["log"], dict):
                    info["log"] = {}
                info["log"].update(stability_log_dict(self.stability_summary(), self._device()))

        return obs, reward, terminated, truncated, info

    def stability_summary(self) -> dict[str, float]:
        if not self._initialized:
            return {}
        # Measured from actions alone, so it needs none of the buffer machinery
        # below — merged in only once an episode has finished (see
        # _action_volatility_summary).
        volatility = self._action_volatility_summary()
        # Every metric carries the "{name}_mean"/"{name}_ci95" key shape that
        # track_stability_summary documents and patch_auc_checkpoint
        # (agent_patches.py) looks up. C is a single shared scalar, so its
        # ci95 is 0 by construction.
        return {
            "auc_mean": self._recent_auc_mean,
            "auc_ci95": self._recent_auc_ci95,
            "contraction_rate_mean": self._recent_lambda_mean,
            "contraction_rate_ci95": self._recent_lambda_ci95,
            "overshoot_mean": self._recent_C,
            "overshoot_ci95": 0.0,
            "contraction_score_mean": self._recent_score_mean,
            "contraction_score_ci95": self._recent_score_ci95,
            **volatility,
        }

    def trajectories(self):
        return self._recent_trajs

    def all_finished(self) -> bool:
        # We can just return False or True depending on usage.
        # This was previously used by ContractionRunner's eval loop.
        # In this new logic, the buffer fills up and resets automatically.
        return False


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
        ax_err.grid(True, alpha=0.3)
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
        ax.grid(True, alpha=0.3)
        _push(fig, "path_tracking")
    else:
        plt.close(fig)

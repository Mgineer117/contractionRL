"""Base class for path-tracking environments.

A path-tracking environment gives the agent:
    obs = [x_current, x_ref, u_ref]

and rewards it with the quadratic contraction cost:
    r = -||x_current - x_ref||_I^2   (identity weighting)

Subclasses must implement:
    _setup_scene(), _apply_action(),
    _get_physical_state() -> (N, state_dim),
    _get_dones() -> (terminated, time_out),
    _set_robot_state_from_ref(env_ids, x_ref_init)  — reset robot to match ref[0]
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from contractionRL.agents.skrl.angle_utils import wrap_diff

from .eval_metrics import fit_exponential_envelope, mean_confidence_interval
from .state_guard import carry_forward_nonfinite
from .traj_buffer import TrajectoryBuffer

# wandb is optional; only used when a run is active
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

# matplotlib is optional; only used to render the PathTracking wandb figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
except ImportError:
    _plt = None

_WANDB_PLOT_INTERVAL = 20   # log trajectory plot every N completed episodes (env 0)
_VIZ_MAX_ENVS = 1         # cap on how many envs' trajectories go into the plot


class PathTrackingBase(DirectRLEnv):
    """Abstract base for path-tracking environments.

    Subclass must define:
        cfg.traj_path   : str — path to .npz reference trajectory file
        cfg.action_space, observation_space

    Subclass may override:
        angle_idx : list[int] — indices within the (raw, per-block) physical
            state that hold a wrapping angle (e.g. yaw). Default () — most
            path-tracking states (proj_gravity + joint pos/vel) have no raw
            angle. Networks embed these continuously ((cos, sin), see
            agents/skrl/angle_utils.py); the tracking ERROR computed here stays
            in raw coordinates but has these dims shortest-angle WRAPPED before
            any norm/reward/metric — otherwise a wraparound (e.g. a yaw U-turn)
            would spike the error by ~2*pi.
    """
    angle_idx: "Sequence[int]" = ()

    # ── Divergence guard ────────────────────────────────────────────────── #
    # An initially unstable policy (random init, or C2RL's con_policy early on)
    # can drive the sim to a non-finite / exploded physical state. Rather than
    # terminating the episode to recover (which would truncate the very
    # trajectory the contraction metrics are fit over), any non-finite element
    # is replaced by the same element from the last known-finite state and the
    # episode keeps running — see carry_forward_nonfinite(). This env family
    # NEVER terminates on divergence (see _get_dones() in each subclass);
    # falling (terminate_on_fall) is the only termination path, and it is off
    # by default for path tracking.
    _ERROR_CLAMP = 1.0e3              # cap on ||error|| feeding reward/AUC/metrics

    def _sanitize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Replace non-finite elements of the physical state with the same
        element from the last known-finite state (carry-forward), so a
        diverging rollout can't propagate NaN/Inf downstream — without ever
        needing to reset/terminate the env to do so.

        Emits a throttled warning (once per physics step, regardless of how many
        call sites sanitize) whenever it actually has to scrub a non-finite
        state — this should be rare, so a silent guard would hide a real problem.
        """
        nonfinite = ~torch.isfinite(x)
        if nonfinite.any():
            step = int(getattr(self, "common_step_counter", 0))
            if step != getattr(self, "_last_nonfinite_warn_step", -1):
                self._last_nonfinite_warn_step = step
                n_bad = int(nonfinite.any(dim=-1).sum())
                print(
                    f"[PathTracking] WARNING: {n_bad}/{x.shape[0]} env(s) produced a "
                    f"non-finite (NaN/Inf) physical state at step {step} — carrying "
                    f"the last finite value forward (divergence guard).",
                    flush=True,
                )
        if self._last_valid_state is None:
            self._last_valid_state = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        sanitized = carry_forward_nonfinite(x, self._last_valid_state)
        self._last_valid_state = sanitized
        return sanitized

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._traj_buf = TrajectoryBuffer(cfg.traj_path, self.device)

        n = self.num_envs
        self._traj_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        self._x_ref = torch.zeros(n, self._traj_buf.state_dim, device=self.device)
        self._u_ref = torch.zeros(n, self._traj_buf.action_dim, device=self.device)
        self._actions = torch.zeros(n, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # last known-finite physical state, for the divergence guard (see
        # _sanitize_state) — populated lazily on first use.
        self._last_valid_state: torch.Tensor | None = None

        # --- episode-level eval metric accumulators ---
        # Streaming (memory-efficient) accumulators for the unified contraction
        # metrics — running sum of error norms (_episode_auc), first/last/peak
        # error and step count — so auc/contraction_rate/overshoot/
        # contraction_score are computed for EVERY finished env without storing
        # the full per-step error curve (see contraction_metrics.per_env_metrics;
        # the same streaming math C3M/C2RL eval use).
        self._episode_auc = torch.zeros(n, device=self.device)
        self._episode_e0 = torch.zeros(n, device=self.device)
        self._episode_emax = torch.zeros(n, device=self.device)
        self._episode_contraction_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._prev_error_norm = torch.full((n,), float("inf"), device=self.device)
        self._episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        # per-episode undiscounted return, for the Reward/total_reward_* tab
        # (same cadence/keys as Stability, see _reset_idx).
        self._episode_reward = torch.zeros(n, device=self.device)

        # --- PathTracking wandb figure: position + error^2 across up to
        # _VIZ_MAX_ENVS envs (CPU, pre-allocated). "live" buffers accumulate the
        # in-progress episode per viz env; "hist" holds each viz env's most
        # recently COMPLETED episode (snapshotted at that env's own reset,
        # since viz envs can terminate/reset at different times) — the plot
        # renders from "hist", not "live".
        self._viz_n = min(_VIZ_MAX_ENVS, n)
        self._viz_error_live = np.zeros((self._viz_n, self.max_episode_length), dtype=np.float32)
        self._viz_pos_live = np.zeros((self._viz_n, self.max_episode_length, 3), dtype=np.float32)
        self._viz_state_live = np.zeros((self._viz_n, self.max_episode_length, self._traj_buf.state_dim), dtype=np.float32)
        self._viz_error_hist = np.zeros((self._viz_n, self.max_episode_length), dtype=np.float32)
        self._viz_pos_hist = np.zeros((self._viz_n, self.max_episode_length, 3), dtype=np.float32)
        self._viz_state_hist = np.zeros((self._viz_n, self.max_episode_length, self._traj_buf.state_dim), dtype=np.float32)
        self._viz_len_hist = np.zeros((self._viz_n,), dtype=np.int64)
        self._env0_episode_count = 0

        # Contraction interface — dynamics model injected by ContractionRunner
        self._dynamics_model = None
        # Certified/target contraction rate + metric-conditioning inputs for
        # the THEORETICAL exponential bound on the PathTracking figure. All
        # None by default (curve omitted) — set via set_contraction_certificate().
        self.target_lambda: float | None = None
        self._rl_discount_factor: float | None = None
        self._static_metric_bounds: tuple[float, float] | None = None
        self._cmg_bounds_fn: Callable[[torch.Tensor], tuple[float, float]] | None = None

    # ------------------------------------------------------------------ #
    # Contraction algorithm interface (C3M / LQR / SD-LQR / C2RL)
    # ------------------------------------------------------------------ #

    def set_dynamics_model(self, model) -> None:
        """Inject a NeuralDynamics model for get_f_and_B (required for Isaac envs)."""
        self._dynamics_model = model

    def set_contraction_certificate(
        self,
        lbd: float | None,
        *,
        discount_factor: float | None = None,
        static_metric_bounds: tuple[float, float] | None = None,
        cmg_bounds_fn: Callable[[torch.Tensor], tuple[float, float]] | None = None,
    ) -> None:
        """Configure the two exponential bounds drawn on the PathTracking
        figure, both of the same shape:

            e(t) <= sqrt(m_bar / m_underbar) * [1/(1-gamma)] * e(0) * exp(-lbd*t)

        differing only in where (m_bar, m_underbar) come from:
          - THEORETICAL: the CMG's configured hard limits (w_lb/w_ub), i.e.
            what the metric is *guaranteed* to satisfy by construction,
            regardless of what the network has actually learned so far.
          - EMPIRICAL: the CURRENT metric's eigenvalue extremes, *measured* on
            the actually-visited states — reflects the real (possibly looser
            or tighter) network today, not the worst-case design limit.

        Called by ContractionRunner for algorithms that have a certified rate
        (C3M/C2RL's cfg.lbd); left at defaults (both curves omitted) for
        PPO/SAC/LQR/SD-LQR, which have neither a CMG nor a certified rate.

        Args:
            lbd: certified/target contraction rate (e.g. cfg.lbd).
            discount_factor: RL discount gamma of the policy actually being
                deployed/plotted — inflates BOTH bounds by 1/(1-gamma) since a
                *discounted* objective enforces the certificate on average
                rather than as a hard per-step constraint. None (or 0) for
                C3M, which has no discounting at all — gamma > 0 only for an
                RL-trained policy (C2RL's con_policy/opt_policy). For C2RL use
                gamma_opt unless running con_only, in which case
                gamma_con — i.e. whichever policy's rollout the figure
                is actually showing.
            static_metric_bounds: (m_bar, m_underbar) from the CMG's configured
                w_lb/w_ub (m_bar=1/w_lb, m_underbar=1/w_ub) — the THEORETICAL
                bound's fixed conditioning factor.
            cmg_bounds_fn: callable (x_batch) -> (m_bar, m_underbar), evaluating
                the CURRENT contraction metric's eigenvalue extremes on a batch
                of states — the EMPIRICAL bound's measured conditioning factor.
        """
        self.target_lambda = None if lbd is None else float(lbd)
        self._rl_discount_factor = discount_factor
        self._static_metric_bounds = static_metric_bounds
        self._cmg_bounds_fn = cmg_bounds_fn

    def _get_visualization_position(self) -> torch.Tensor:
        """Returns (N, 3) world-frame position for the PathTracking figure's
        position subplot. Default: the robot's root position — meaningful for
        locomotion (humanoid/quadruped), but degenerates to a single fixed
        point for a fixed-base arm (manipulator), since its root never moves.
        Override in a subclass (e.g. to return end-effector position) for a
        more informative plot there.
        """
        robot = getattr(self, "_robot", None)
        if robot is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return robot.data.root_pos_w

    def _log_pathtracking_figure(self) -> None:
        """Render the "PathTracking" wandb figure from the viz envs' most
        recently completed episodes:
          left  — attempted-trajectory position (world xy) per env
          right — ||error||_I^2 per step, plus THREE reference curves (each
                  only drawn if its inputs are available):
            1. fitted envelope — fit_exponential_envelope (CAC-dev style),
               a pure data fit of the observed error trajectories: no CMG,
               no gamma, just the tightest C*exp(-lambda*t) that bounds them.
            2. theoretical bound — sqrt(m_bar/m_underbar) * [1/(1-gamma)] *
               e(0) * exp(-lbd*t), where m_bar/m_underbar come from the CMG's
               CONFIGURED hard limits (w_lb/w_ub) — the worst-case guarantee
               by construction, regardless of what the network currently does.
            3. empirical bound — same shape, but m_bar/m_underbar are
               MEASURED (max/min eigenvalues of the CURRENT CMG evaluated on
               the actually-visited states) — reflects the real network today.
          Both (2)/(3) use set_contraction_certificate()'s lbd/gamma; gamma=0
          (no inflation) for non-discounted certificates like C3M, gamma>0 for
          an RL-trained policy (C2RL).
        """
        valid = np.nonzero(self._viz_len_hist > 0)[0]
        if len(valid) == 0:
            return
        dt = self.step_dt

        # (i+1)*dt indexing, matching fit_exponential_envelope's own convention
        error_traces = [self._viz_error_hist[i, : self._viz_len_hist[i]] for i in valid]
        pos_traces = [self._viz_pos_hist[i, : self._viz_len_hist[i]] for i in valid]

        # Plot 1: Attempted trajectories
        fig_pos, ax_pos = _plt.subplots(figsize=(6, 5))
        for p in pos_traces:
            ax_pos.plot(p[:, 0], p[:, 1], linewidth=0.8, alpha=0.6)
        ax_pos.set_xlabel("x [m]")
        ax_pos.set_ylabel("y [m]")
        ax_pos.set_title(f"Attempted trajectories (n={len(valid)})")
        ax_pos.set_aspect("equal", adjustable="datalim")
        fig_pos.tight_layout()
        _wandb.log({"train/tracking_trajectory": _wandb.Image(fig_pos), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
        _plt.close(fig_pos)

        # Plot 2: Tracking error with contraction bounds
        fig_err, ax_err = _plt.subplots(figsize=(6, 5))
        max_len = max(len(e) for e in error_traces)
        t_axis = np.arange(1, max_len + 1) * dt
        for e in error_traces:
            t = np.arange(1, len(e) + 1) * dt
            ax_err.plot(t, e ** 2, linewidth=0.6, alpha=0.5, color="tab:blue")

        # 1. fitted envelope — data only, no CMG/gamma involved.
        C_fit, lbds_fit = fit_exponential_envelope(error_traces, dt)
        pos_lbds = lbds_fit[lbds_fit > 0]
        lambda_fit = float(pos_lbds.mean()) if pos_lbds.size else 0.0
        if lambda_fit > 0:
            e0_mean = float(np.mean([e[0] for e in error_traces if len(e) and e[0] > 0]))
            ax_err.plot(
                t_axis, (e0_mean * C_fit * np.exp(-lambda_fit * t_axis)) ** 2, "r--", linewidth=2,
                label=f"fitted envelope (C={C_fit:.2g}·e₀, λ̄={lambda_fit:.2g})",
            )

        if self.target_lambda is not None:
            e0_mean = float(np.mean([e[0] for e in error_traces]))
            gamma_factor = 1.0
            if self._rl_discount_factor is not None:
                gamma_factor = 1.0 / max(1e-6, 1.0 - self._rl_discount_factor)

            # 2. theoretical — CMG's configured hard limits (w_lb/w_ub).
            if self._static_metric_bounds is not None:
                m_bar, m_underbar = self._static_metric_bounds
                cond = float(np.sqrt(max(m_bar, 1e-12) / max(m_underbar, 1e-12)))
                C_theory = cond * gamma_factor * e0_mean
                ax_err.plot(
                    t_axis, (C_theory * np.exp(-self.target_lambda * t_axis)) ** 2, "g:", linewidth=2,
                    label=f"theoretical bound (C={C_theory:.2g}, λ={self.target_lambda:.2g})",
                )

            # 3. empirical — CURRENT CMG's eigenvalues, measured on visited states.
            if self._cmg_bounds_fn is not None:
                state_traces = [self._viz_state_hist[i, : self._viz_len_hist[i]] for i in valid]
                states = np.concatenate(state_traces, axis=0)
                if len(states) > 2000:
                    states = states[np.random.choice(len(states), 2000, replace=False)]
                x_batch = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                m_bar, m_underbar = self._cmg_bounds_fn(x_batch)
                cond = float(np.sqrt(max(m_bar, 1e-12) / max(m_underbar, 1e-12)))
                C_emp = cond * gamma_factor * e0_mean
                ax_err.plot(
                    t_axis, (C_emp * np.exp(-self.target_lambda * t_axis)) ** 2, "m-.", linewidth=2,
                    label=f"empirical bound (C={C_emp:.2g}, λ={self.target_lambda:.2g})",
                )

        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel(r"$\|e\|_I^2$")
        ax_err.set_title("Tracking error² with contraction bounds")
        ax_err.set_yscale("log")
        ax_err.legend(fontsize=8)
        fig_err.tight_layout()
        _wandb.log({"train/normalized_error": _wandb.Image(fig_err), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
        _plt.close(fig_err)

    def get_f_and_B(self, x):
        """Return (f, B, B_null) for contraction agents.

        Delegates to the injected NeuralDynamics model — call
        ``set_dynamics_model(model)`` before using C3M/LQR/SDLQR/C2RL.
        """
        if self._dynamics_model is None:
            raise RuntimeError(
                "get_f_and_B requires a NeuralDynamics model. "
                "Load one with --dynamics_checkpoint and pass it to ContractionRunner."
            )
        return self._dynamics_model.get_f_and_B(x)

    def get_rollout(self, buffer_size: int, mode: str) -> dict:
        """Sample random (x, xref, uref) triples for C3M-style contraction synthesis.

        Samples reference states from the trajectory buffer, adding bounded
        Gaussian noise to produce actual states that deviate from the reference.
        Returns numpy float32 arrays (required by mjrl's to_tensor()).
        """
        if mode == "dynamics":
            return self._get_dynamics_rollout(buffer_size)
        if mode != "c3m":
            raise ValueError(f"PathTrackingBase.get_rollout: unsupported mode '{mode}'")

        buf = self._traj_buf
        # Random trajectories and random timesteps within them
        traj_ids = torch.randint(0, buf.num_trajs, (buffer_size,), device=self.device)
        steps = torch.randint(0, buf.traj_len, (buffer_size,), device=self.device)
        xref, uref = buf.get(traj_ids, steps)  # (N, state_dim), (N, action_dim)

        # Small Gaussian noise to create x ≠ xref (contraction error)
        noise = torch.randn_like(xref) * 0.05
        x = xref + noise

        return {
            "x":    x.cpu().numpy().astype(np.float32),
            "xref": xref.cpu().numpy().astype(np.float32),
            "uref": uref.cpu().numpy().astype(np.float32),
        }

    def _get_dynamics_rollout(self, buffer_size: int) -> dict:
        """Sample (x, u, x_dot) pairs for NeuralDynamics training.

        Uses consecutive (t, t+1) pairs from the reference trajectory buffer
        and approximates x_dot via finite differences: (x_{t+1} - x_t) / step_dt.
        This avoids requiring additional env interaction beyond what's in the buffer.
        """
        buf = self._traj_buf
        traj_ids = torch.randint(0, buf.num_trajs, (buffer_size,), device=self.device)
        # Avoid last step so t+1 is always valid
        steps = torch.randint(0, buf.traj_len - 1, (buffer_size,), device=self.device)
        x_t, u_t = buf.get(traj_ids, steps)
        x_next, _ = buf.get(traj_ids, steps + 1)
        x_dot = (x_next - x_t) / self.step_dt
        return {
            "x":     x_t.cpu().numpy().astype(np.float32),
            "u":     u_t.cpu().numpy().astype(np.float32),
            "x_dot": x_dot.cpu().numpy().astype(np.float32),
        }

    def get_tracking_error(self) -> torch.Tensor:
        """Current ||x - x_ref|| per env, (N,).

        Shared across all path-tracking envs (quadruped/humanoid/manipulator) —
        used by the post-training evaluator to fit the exponential contraction
        envelope C * exp(-lambda * k * dt) bounding the error curve. angle_idx
        dims (e.g. yaw) are shortest-angle wrapped before the norm — see
        _get_rewards for why an unwrapped difference would be wrong.
        """
        return torch.norm(wrap_diff(self._get_physical_state() - self._x_ref, self.angle_idx), dim=-1)

    @property
    def x_dim(self) -> int:
        return self._traj_buf.state_dim

    @property
    def u_dim(self) -> int:
        return self._traj_buf.action_dim

    # ------------------------------------------------------------------ #
    # Interface to implement in subclasses
    # ------------------------------------------------------------------ #

    def _get_physical_state(self) -> torch.Tensor:
        """Returns (N, state_dim) current robot physical state."""
        raise NotImplementedError

    def _set_robot_state_from_ref(
        self, env_ids: torch.Tensor, x_ref_init: torch.Tensor
    ) -> None:
        """Set robot state (joints + base vel) to match x_ref_init."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # PathTracking logic (shared)
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        step = self.episode_length_buf.long()
        self._x_ref, self._u_ref = self._traj_buf.get(self._traj_ids, step)
        # sanitize so the policy never receives NaN/Inf from a diverging env
        x = self._sanitize_state(self._get_physical_state())
        obs = torch.cat([x, self._x_ref, self._u_ref], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # sanitize + clamp so a diverging env yields a large-but-FINITE penalty
        # (never NaN/Inf) into the reward, AUC, and Stability curves — the
        # episode is never reset/terminated because of it (see _sanitize_state).
        x = self._sanitize_state(self._get_physical_state())
        # angle_idx dims (e.g. yaw) are wrapped to the shortest-angle difference
        # BEFORE any norm/reward/metric — a raw difference would spike by ~2*pi
        # whenever the physical angle wraps (e.g. a yaw U-turn). Non-angle dims
        # are unaffected (wrap_diff is a no-op there, and a no-op entirely when
        # angle_idx is empty).
        error = wrap_diff(x - self._x_ref, self.angle_idx)
        error_norm = torch.norm(error, dim=-1).clamp(max=self._ERROR_CLAMP)   # (N,)

        # accumulate streaming metric state: running error sum (AUC integrand),
        # initial error e0 (on the episode's first step), and peak error e_max.
        self._episode_auc += error_norm
        is_first = (self._episode_steps == 0)
        self._episode_e0 = torch.where(is_first, error_norm, self._episode_e0)
        self._episode_emax = torch.maximum(self._episode_emax, error_norm)

        # contraction flag: count steps where error strictly decreased (skip step 0)
        contracting = (~is_first) & (error_norm < self._prev_error_norm)
        self._episode_contraction_steps += contracting.long()

        self._prev_error_norm = error_norm.clone()
        self._episode_steps += 1

        # collect the first _viz_n envs' trajectories for the PathTracking figure
        viz_n = self._viz_n
        viz_steps = self._episode_steps[:viz_n]  # already incremented above; use step-1 as index
        in_range = viz_steps <= self.max_episode_length
        if in_range.any():
            idx = (viz_steps[in_range] - 1).cpu().numpy()
            rows = in_range.nonzero(as_tuple=True)[0].cpu().numpy()
            self._viz_error_live[rows, idx] = error_norm[:viz_n][in_range].detach().cpu().numpy()
            pos = self._get_visualization_position()[:viz_n]
            self._viz_pos_live[rows, idx] = pos[in_range].detach().cpu().numpy()
            self._viz_state_live[rows, idx] = x[:viz_n][in_range].detach().cpu().numpy()

        # -||error||_I^2, using the sanitized+clamped norm so a diverged step
        # can't emit a huge/NaN reward that wrecks the value fn / normalizers.
        reward = -(error_norm * error_norm)
        self._episode_reward += reward
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # --- log episode metrics for envs that actually ran ---
        finished = env_ids[self._episode_steps[env_ids] > 0]
        env0_finished = bool((finished == 0).any()) if len(finished) > 0 else False
        if len(finished) > 0:
            self.extras.setdefault("log", {})

            dt = self.step_dt

            # --- contraction flag (fraction of steps the error decreased) ---
            steps = self._episode_steps[finished].float()
            valid_steps = (steps - 1).clamp(min=1)
            contraction_flag = self._episode_contraction_steps[finished].float() / valid_steps
            self.extras["log"]["Stability/contraction_flag"] = contraction_flag.mean()

            # --- Unified streaming contraction metrics for EVERY finished env ---
            # auc (normalized dt-weighted trapezoid), contraction_rate (lambda),
            # overshoot (e_max/e0) and contraction_score — each mean + 95% CI,
            # using the SAME streaming computation and wandb keys as C3M/C2RL
            # (contraction_metrics), so PPO/SAC/LQR/SD-LQR share the same
            # Stability/* tab, formula and CI as the contraction algorithms.
            # Computed over ALL finished envs (not just the few viz envs the old
            # fit_exponential_envelope path used), for tighter statistics.
            from contractionRL.agents.skrl.contraction_metrics import (
                per_env_metrics, stability_log_dict, summarize,
            )
            per_env = per_env_metrics(
                e0=self._episode_e0[finished],
                e_last=self._prev_error_norm[finished],
                e_max=self._episode_emax[finished],
                err_sum=self._episode_auc[finished],
                steps=steps,
                dt=dt,
            )
            self.extras["log"].update(stability_log_dict(summarize(per_env), self.device))

            # --- Reward/total_reward_* — same keys/cadence as the Stability
            # tab above, and as C3M's eval loop (track_reward_summary), so
            # PPO/SAC/C2RL don't rely on a once-only post-training eval point.
            from contractionRL.agents.skrl.contraction_metrics import reward_log_dict, reward_summary
            self.extras["log"].update(
                reward_log_dict(reward_summary(self._episode_reward[finished]), self.device)
            )

        # --- PathTracking figure: snapshot completed episodes for viz envs ---
        # (env_ids may include indices >= _viz_n, which we simply don't track)
        env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.asarray(env_ids)
        viz_ids = env_ids_np[env_ids_np < self._viz_n]
        if len(viz_ids) > 0:
            ep_lens = self._episode_steps[torch.as_tensor(viz_ids, device=self.device)].cpu().numpy()
            for local_i, ep_len in zip(viz_ids, ep_lens):
                if ep_len > 0:
                    self._viz_error_hist[local_i] = self._viz_error_live[local_i]
                    self._viz_pos_hist[local_i] = self._viz_pos_live[local_i]
                    self._viz_state_hist[local_i] = self._viz_state_live[local_i]
                    self._viz_len_hist[local_i] = ep_len
            self._viz_error_live[viz_ids] = 0.0
            self._viz_pos_live[viz_ids] = 0.0
            self._viz_state_live[viz_ids] = 0.0

        # --- wandb PathTracking figure (position + error^2 with bounds) ---
        # Cadence still keyed off env 0's reset — a simple periodic heartbeat,
        # not a synchronization requirement (viz envs snapshot independently
        # above, so the figure renders whatever's most recently completed for
        # each of them, which may span slightly different wall-clock episodes).
        if env0_finished:
            self._env0_episode_count += 1
            if (
                _wandb is not None and _plt is not None
                and self._env0_episode_count % _WANDB_PLOT_INTERVAL == 0
                and getattr(_wandb, "run", None) is not None
                and self._viz_len_hist.max() > 0
            ):
                self._log_pathtracking_figure()

        # --- reset per-env buffers ---
        self._episode_auc[env_ids] = 0.0
        self._episode_e0[env_ids] = 0.0
        self._episode_emax[env_ids] = 0.0
        self._episode_contraction_steps[env_ids] = 0
        self._prev_error_norm[env_ids] = float("inf")
        self._episode_steps[env_ids] = 0
        self._episode_reward[env_ids] = 0.0

        # sample reference trajectories
        self._traj_ids[env_ids] = self._traj_buf.sample_traj_ids(len(env_ids))

        # initialise robot to match x_ref at step 0
        x_ref_init = self._traj_buf.initial_state(self._traj_ids[env_ids])  # (n, state_dim)
        self._set_robot_state_from_ref(env_ids, x_ref_init)

        # resync the divergence guard's baseline to the freshly-reset physical
        # state, so it doesn't carry forward a stale pre-reset value into the
        # new episode.
        if self._last_valid_state is not None:
            self._last_valid_state[env_ids] = self._get_physical_state()[env_ids]

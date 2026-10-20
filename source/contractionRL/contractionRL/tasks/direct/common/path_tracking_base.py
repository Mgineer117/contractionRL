"""Base class for path-tracking environments.

Observation: s = (x, x_ref, reference parameters/time). Plant gusts are hidden.
Reward: r = E_current - E_next, with identity or an installed contraction metric.
Failures terminate; the horizon truncates. Kinematic observations are clipped.

Subclasses must implement:
    _initialize_robot_indices(),
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

from contractionRL.agents.utils.system_dynamics.state import wrap_diff, unwrap_state_angles, clamp_physical_state
from contractionRL.agents.utils.system_dynamics.reference import RefWindow
from contractionRL.agents.utils.contraction.math_utils import tracking_energy, energy_decrease
from contractionRL.agents.utils.RL.logging import log_metrics

from .eval_metrics import fit_exponential_envelope
from .state_guard import carry_forward_nonfinite
from .termination_box import TerminationBoxMixin
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


class PathTrackingBase(TerminationBoxMixin, DirectRLEnv):
    """Abstract base for path-tracking environments.

    Subclass must define:
        cfg.traj_path   : str — path to .npz reference trajectory file
        cfg.action_space, observation_space

    Subclass may override:
        angle_idx : list[int] — indices within the (raw, per-block) physical
            state that hold a wrapping angle (e.g. yaw). Default () — most
            path-tracking states (proj_gravity + joint pos/vel) have no raw
            angle. Networks embed these continuously ((cos, sin), see
            agents/utils/system_dynamics/state.py); the tracking error computed here stays
            in raw coordinates but has these dims shortest-angle wrapped before
            any norm/reward/metric — otherwise a wraparound (e.g. a yaw U-turn)
            would spike the error by ~2*pi.
    """
    has_analytical_dynamics = False
    angle_idx: Sequence[int] = ()

    # One name per physical state dim (see agents/utils/system_dynamics/state.py for the
    # vocabulary). Declaring names rather than raw indices is what lets the
    # networks quotient out the symmetry of the tracking problem: which dims are
    # pure translation directions, which is the planar heading, and which co-
    # rotate under a yaw rotation. Crucially the frame suffix matters --
    # root_lin_vel_b / root_ang_vel_b / projected_gravity_b are body frame and so
    # already yaw invariant, whereas a world-frame velocity would have to be
    # co-rotated with the position pair. Left empty -> angle_idx below is used as
    # before and no quotient is applied.
    #
    # Subclasses whose width depends on the robot (joint count) should override
    # the ``state_names`` property instead of this attribute.
    _state_names: Sequence[str] = ()

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self._state_names)

    @property
    def pos_dimension(self) -> int:
        """Number of pure translation directions, derived from state_names."""
        names = self.state_names
        if not names:
            return 0
        from contractionRL.agents.utils.system_dynamics.state import StateSymmetry
        return StateSymmetry.from_names(names).pos_dimension

    # Keep terminal observations finite; the raw state still triggers failure.
    _ERROR_CLAMP = 1.0e3              # cap on ||error|| feeding reward/AUC/metrics

    def _unwrap_state_angles(self, x):
        state = unwrap_state_angles(x, getattr(self, '_continuous_state', None), self.angle_idx)
        self._continuous_state = state.detach().clone()
        return state

    def _sanitize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Replace non-finite elements of the physical state with the same
        element from the last known-finite state (carry-forward), so a
        diverging rollout can't propagate NaN/Inf downstream — while the raw non-finite state triggers termination.

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
        if hasattr(self, "ref_window"):
            lo, hi = self._state_bounds()
            clipped = clamp_physical_state(sanitized, lo, hi, self.angle_idx)
            if self.ref_window.u_dim == 7:
                clipped[:, 14:17] = sanitized[:, 14:17]
            else:
                clipped[:, :2] = sanitized[:, :2]
            sanitized = clipped
        self._last_valid_state = sanitized
        return sanitized

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._initialize_robot_indices()

        self._traj_buf = TrajectoryBuffer(cfg.traj_path, self.device)
        self._traj_buf.require_nominal(self.step_dt)
        self._traj_buf.unwrap_angles(self.angle_idx)

        # Compact reference generator state; no future reference sequence.
        from contractionRL.agents.utils.system_dynamics.reference import ReferenceState
        from .disturbance import BoundedGust
        # World-frame wind force [N] and torque [Nm], applied to base or hand.
        force = 1.0 if self._traj_buf.action_dim == 7 else (2.0 if self._traj_buf.action_dim == 12 else 15.0)
        force = float(getattr(cfg, "disturbance_force_bound", force))
        torque = float(getattr(cfg, "disturbance_torque_bound", .1))
        self.disturbance = BoundedGust(self.num_envs, [force] * 3 + [torque] * 3, self.step_dt, self.device)
        self.disturbance.reset(torch.arange(self.num_envs, device=self.device))
        self.ref_window = ReferenceState(self._traj_buf.state_dim, self._traj_buf.action_dim, 1,
                                         self._reference_control,
                                         parameter_bounds=([0.], [self._traj_buf.num_trajs - 1.]),
                                         time_bound=self.cfg.episode_length_s,
                                         state_names=self.state_names, angle_idx=self.angle_idx)
        self._rebuild_observation_space()
        low, high = self._state_bounds()
        margin = 0.05 * (high - low)
        if self._traj_buf.action_dim != 7:
            margin[2:6] = 0  # periodic heading and unit projected-gravity vector
            margin[-1] = 0.03  # clearance above the robot's failure height [m]
        self.reference_margin = margin
        control_low, control_high = self._control_bounds()
        self._traj_buf.validate(low, high, margin, control_low, control_high,
                                self.max_episode_length + 1, upright_index=None if self._traj_buf.action_dim == 7 else 5)

        n = self.num_envs
        self._traj_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        # See set_fix_ref_trajectories. Off by default: every reset redraws.
        self.fix_ref_trajectories = False
        self._ref_fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._x_ref = torch.zeros(n, self._traj_buf.state_dim, device=self.device)
        self._prev_energy = torch.zeros(n, device=self.device)
        self._u_ref = torch.zeros(n, self._traj_buf.action_dim, device=self.device)
        self._actions = torch.zeros(n, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # last known-finite physical state, for the divergence guard (see
        # _sanitize_state) — populated lazily on first use.
        self._last_valid_state: torch.Tensor | None = None

        # Finite physical domains; position excursions and robot failures terminate.
        _bounds_lo, _bounds_hi = self._state_bounds()
        self._init_termination_box(
            getattr(self.cfg, "x_termination_min", None) or _bounds_lo,
            getattr(self.cfg, "x_termination_max", None) or _bounds_hi,
            clamp_box=None,                       # no clamp on this side
            armed=bool(getattr(self.cfg, "terminate_out_of_box", True)),
            terminal=bool(getattr(self.cfg, "x_termination_terminal", False)),
            tag="PathTracking")
        # Every subclass owns its own _get_dones (fall detection, task-specific
        # limits), so the box is folded in by wrapping the bound method once
        # here rather than editing each subclass — the same post-construction
        # patching idiom agent_patches.py uses for skrl, and for the same
        # reason: one place to change, and no subclass can forget to call it.
        self._get_dones = self._with_termination_box(self._get_dones)

        # --- episode-level eval metric accumulators ---
        # Streaming (memory-efficient) accumulators for the unified contraction
        # metrics — running sum of error norms (_episode_auc), first/last/peak
        # error and step count — so auc/contraction_rate/overshoot/
        # contraction_score are computed for every finished env without storing
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
        # recently completed episode (snapshotted at that env's own reset,
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
        # Frozen CMG (contraction metric generator) — injected via set_ccm()
        # for C2RL's Phase B. None for PPO/SAC/LQR/SD-LQR, which never call
        # set_ccm and so keep the plain -(error_norm**2) reward in
        # _get_rewards unchanged. Defaults match that plain reward's implicit
        # weights (identity tracking weight, no control penalty) so enabling
        # the CCM branch doesn't also silently reweight the objective.
        self.ccm_gen = None
        self.tracking_scaler = 1.0
        self.control_scaler = 0.0
        # Certified/target contraction rate + metric-conditioning inputs for
        # the theoretical exponential bound on the PathTracking figure. All
        # None by default (curve omitted) — set via set_contraction_certificate().
        self.target_lambda: float | None = None
        self._rl_discount_factor: float | None = None
        self._static_metric_bounds: tuple[float, float] | None = None
        self._cmg_bounds_fn: Callable[[torch.Tensor], tuple[float, float]] | None = None

    # ------------------------------------------------------------------ #
    # Contraction algorithm interface (C3M / LQR / SD-LQR / C2RL)
    # ------------------------------------------------------------------ #

    def _initialize_robot_indices(self) -> None:
        """Resolve articulation indices before building reference state names."""
        raise NotImplementedError

    def set_dynamics_model(self, model) -> None:
        """Inject a NeuralDynamics model for get_f_and_B (required for Isaac envs)."""
        self._dynamics_model = model

    def _metric_from_cmg(self, x, xref=None):
        """M(x) from the CMG, inverting only when the head emits W.

        Mirrors classic env_base._metric_from_cmg exactly — see
        tests/test_isaac_parity.py, which requires both families to convert the
        CMG output the same way or their Mahalanobis rewards stop being
        comparable. C2RL builds the CMG with outputs_metric=True
        and its forward already returns M (so this is a pass-through and the
        per-step SPD inverse disappears); "ccm" emits W and is inverted here.
        """
        from contractionRL.agents.utils.contraction.math_utils import bound_W, spd_inverse
        raw, _ = self.ccm_gen(x)
        out = bound_W(raw, self.w_lb, x.shape[-1],
                      getattr(self.ccm_gen, "bounded", False))
        if getattr(self.ccm_gen, "outputs_metric", False):
            return out
        return spd_inverse(out)

    def set_ccm(self, ccm_gen, w_lb, device, tracking_scaler=None, control_scaler=None,
                reward_euclidean=False, reward_level=False,
                residual_anchor_scale=0.0, cvstem_r_scaler=1.0) -> None:
        """Inject the frozen CMG for C2RL's Phase B Mahalanobis reward.

        Mirrors common/analytical_env.py's BaseEnv.set_ccm exactly (same
        signature) so C2RLSkrlTrainer.train() can call it identically for
        both env families — see _get_rewards for the reward this enables.

        The scalers default to None = keep the plain-reward weights set in
        __init__ (1.0 / 0.0).

        The last four are classic-only reward variants (residual-base RL over
        CV-STEM-LQR); _get_rewards here has no branch for them, so a non-default
        value raises rather than silently training on the Mahalanobis reward.
        """
        unsupported = [n for n, v in (("reward_euclidean", reward_euclidean),
                                      ("reward_level", reward_level),
                                      ("residual_anchor_scale", residual_anchor_scale),
                                      ("cvstem_r_scaler", cvstem_r_scaler))
                       if v not in (False, 0.0) and not (n == "cvstem_r_scaler" and v == 1.0)]
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__}._get_rewards implements only the Mahalanobis "
                f"reward; classic-only knobs requested: {', '.join(unsupported)}."
            )
        self.ccm_gen = ccm_gen
        self.w_lb = w_lb
        self.ccm_device = device
        if tracking_scaler is not None:
            self.tracking_scaler = float(tracking_scaler)
        if control_scaler is not None:
            self.control_scaler = float(control_scaler)

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
          - Theoretical: the CMG's configured hard limits (w_lb/w_ub), i.e.
            what the metric is *guaranteed* to satisfy by construction,
            regardless of what the network has actually learned so far.
          - Empirical: the current metric's eigenvalue extremes, *measured* on
            the actually-visited states — reflects the real (possibly looser
            or tighter) network today, not the worst-case design limit.

        Called by ContractionRunner for algorithms that have a certified rate
        (C3M/C2RL's cfg.lbd); left at defaults (both curves omitted) for
        PPO/SAC/LQR/SD-LQR, which have neither a CMG nor a certified rate.

        Args:
            lbd: certified/target contraction rate (e.g. cfg.lbd).
            discount_factor: RL discount gamma of the policy actually being
                deployed/plotted — inflates both bounds by 1/(1-gamma) since a
                *discounted* objective enforces the certificate on average
                rather than as a hard per-step constraint. None (or 0) for
                C3M, which has no discounting at all — gamma > 0 only for an
                RL-trained policy (C2RL's deployed policy uses its
                discount_factor — i.e. the policy whose rollout the figure
                is actually showing).
            static_metric_bounds: (m_bar, m_underbar) from the CMG's configured
                w_lb/w_ub (m_bar=1/w_lb, m_underbar=1/w_ub) — the theoretical
                bound's fixed conditioning factor.
            cmg_bounds_fn: callable (x_batch) -> (m_bar, m_underbar), evaluating
                the current contraction metric's eigenvalue extremes on a batch
                of states — the empirical bound's measured conditioning factor.
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
          right — ||error||_I^2 per step, plus three reference curves (each
                  only drawn if its inputs are available):
            1. fitted envelope — fit_exponential_envelope (CAC-dev style),
               a pure data fit of the observed error trajectories: no CMG,
               no gamma, just the tightest C*exp(-lambda*t) that bounds them.
            2. theoretical bound — sqrt(m_bar/m_underbar) * [1/(1-gamma)] *
               e(0) * exp(-lbd*t), where m_bar/m_underbar come from the CMG's
               configured hard limits (w_lb/w_ub). These bound the spectral
               factor; decay still requires a separately verified controller.
            3. empirical bound — same shape, but m_bar/m_underbar are
               measured (max/min eigenvalues of the current CMG evaluated on
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
        log_metrics({"train/tracking_trajectory": _wandb.Image(fig_pos), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
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
                    label=f"configured-rate reference (C={C_theory:.2g}, λ={self.target_lambda:.2g})",
                )

            # 3. empirical — current CMG's eigenvalues, measured on visited states.
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
                    label=f"sampled-spectrum reference (C={C_emp:.2g}, λ={self.target_lambda:.2g})",
                )

        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel(r"$\|e\|_I^2$")
        ax_err.set_title("Tracking error² with contraction bounds")
        ax_err.set_yscale("log")
        ax_err.legend(fontsize=8)
        fig_err.tight_layout()
        log_metrics({"train/normalized_error": _wandb.Image(fig_err), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
        _plt.close(fig_err)

    def get_f_and_B(self, x, *, need_null: bool = True):
        """Return (f, B, B_null) for contraction agents.

        Delegates to the injected NeuralDynamics model — call
        ``set_dynamics_model(model)`` before using C3M/LQR/SDLQR/C2RL.

        ``need_null`` exists for signature parity with ``BaseEnv.get_f_and_B``
        (a contraction agent calls whichever it was handed). It is accepted and
        ignored here: the NeuralDynamics forward pass produces all three heads
        together, so there is nothing to skip.
        """
        del need_null
        if self._dynamics_model is None:
            raise RuntimeError(
                "get_f_and_B requires a NeuralDynamics model — Isaac Sim envs have no "
                "analytical dynamics. C3M/C2RL build and pretrain one themselves; "
                "lqr/sdlqr/cvstem-lqr need one injected via set_dynamics_model(), which "
                "ContractionRunner does from --dynamics_checkpoint (a dynamics.pt saved "
                "by a previous C3M/C2RL run)."
            )
        return self._dynamics_model.get_f_and_B(x)

    def get_rollout(self, buffer_size: int, mode: str, num_control_per_state: int | None = None) -> dict:
        """Sample random (x, xref, uref) triples for C3M-style contraction synthesis.

        Samples independent states, references and controls over the full boxes,
        using the same distribution as C2RL's CCM synthesis.
        Returns numpy float32 arrays (required by mjrl's to_tensor()).

        ``num_control_per_state`` is a classic-env-only knob (see
        ``env_base.get_rollout``) accepted here purely so callers can pass it
        uniformly across both env families — Isaac's dynamics rollout draws
        real consecutive-step transitions from the trajectory buffer, so there
        is no "controls per state" tiling to configure; the argument is ignored.
        """
        if mode == "dynamics":
            return self._get_dynamics_rollout(buffer_size)
        if mode != "c3m":
            raise ValueError(f"PathTrackingBase.get_rollout: unsupported mode '{mode}'")

        from contractionRL.agents.utils.contraction.ccm_synthesis import sample_ccm_domain
        low, high = self._state_bounds()
        u_low, u_high = self._control_bounds()
        data = sample_ccm_domain(buffer_size, low, high, u_low, u_high, device=self.device)
        return {key: value.cpu().numpy().astype(np.float32) for key, value in data.items()}

    def _get_dynamics_rollout(self, buffer_size: int) -> dict:
        """Sample (x, u, x_dot) pairs for NeuralDynamics training.

        Uses consecutive (t, t+1) pairs from the reference trajectory buffer
        and approximates x_dot via finite differences: (x_{t+1} - x_t) / step_dt.
        This avoids requiring additional env interaction beyond what's in the buffer.
        """
        buf = self._traj_buf
        traj_ids = buf.sample_traj_ids(buffer_size)
        # Avoid last step so t+1 is always valid
        steps = (torch.rand(buffer_size, device=self.device) * (buf._lengths[traj_ids] - 1)).long()
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

    def set_fix_ref_trajectories(self, flag: bool) -> None:
        """Freeze (or unfreeze) the per-env reference trajectories.

        The classic twin is ``BaseEnv.set_fix_ref_trajectories``; both exist so
        the training entry point can set this uniformly (see
        tests/test_isaac_parity.py). Isaac references come from a fixed dataset,
        so "fixing" here means pinning each env slot to one sampled traj id for
        the whole run rather than redrawing every episode. Turning it on clears
        the assignment so the next reset re-mints it.
        """
        self.fix_ref_trajectories = bool(flag)
        if not hasattr(self, "_ref_fixed"):
            self._ref_fixed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ref_fixed[:] = False
        print(f"[PathTrackingBase] fix_ref_trajectories={self.fix_ref_trajectories}"
              + (f" ({self.num_envs} fixed reference trajectories)"
                 if self.fix_ref_trajectories else " (resampled every episode)"))

    def get_reference_trajectory(self) -> torch.Tensor:
        """whole reference path per env: ``(num_envs, T, x_dim)``.

        The classic twin is ``BaseEnv.get_reference_trajectory``; both exist so
        logging code plots the complete reference against the policy rollout
        instead of rebuilding a reference from the (offset-subsampled,
        horizon-truncated) observation window. See tests/test_isaac_parity.py.
        """
        return self._traj_buf.full(self._traj_ids)

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

    def _rebuild_observation_space(self) -> None:
        """Declare ``{x, xrefs, urefs}`` as the observation space.

        Isaac's DirectRLEnv takes an integer width from the env cfg, so the
        flat width is what it is told; the Dict is exposed alongside it for
        ``RefWindow.from_space``. Both agree by construction (``flat_dim``).
        """
        lo, hi = self._state_bounds()
        u_lo, u_hi = self._control_bounds()
        self.observation_space = self.ref_window.space(lo, hi, u_lo, u_hi)
        self.cfg.observation_space = self.ref_window.flat_dim
        self.num_observations = self.ref_window.flat_dim
        declared = getattr(self, "single_observation_space", None)
        if hasattr(declared, "spaces"):
            declared.spaces["policy"] = self.observation_space
        elif isinstance(declared, dict):
            declared["policy"] = self.observation_space

    def _with_termination_box(self, get_dones):
        """Wrap a subclass ``_get_dones`` so the box folds into its result.

        Returns ``(terminated, time_out)`` exactly as Isaac Lab expects; the
        raw domain excursions are terminal failures.
        Also publishes ``episode_ended_early`` on ``extras`` so
        StatManagerEnvWrapper can drop those episodes from AUC/lambda instead of
        measuring a curve that was cut — which is precisely the objection that
        kept this env family from terminating on divergence before.
        """
        def wrapped():
            terminated, time_out = get_dones()
            x = self._get_physical_state()
            self.extras["dynamics_x_next"] = x.detach().clone()
            lo, hi = self._state_bounds()
            pos = slice(14, 17) if self.ref_window.u_dim == 7 else slice(0, 2)
            low, high = torch.as_tensor(lo, device=x.device), torch.as_tensor(hi, device=x.device)
            failed = ((x[:, pos] < low[pos]) | (x[:, pos] > high[pos])).any(-1)
            # Domain excursions in the other coordinates are failures too.
            # Clipping only their terminal observation avoids hidden overspeed
            # states with identical observations but different future motion.
            canonical = wrap_diff(x, self.angle_idx)
            failed |= ((canonical < low - 1e-5) | (canonical > high + 1e-5)).any(-1)
            failed |= (~torch.isfinite(x)).any(-1)
            if self.ref_window.u_dim != 7:
                failed |= self._robot.data.root_pos_w[:, 2] > 3.
            self.extras["episode_ended_early"] = terminated | failed
            done = terminated | failed | time_out
            self.extras.pop("final_observation", None)
            self.extras["_final_observation"] = done.clone()
            if done.any():
                self.extras["final_observation"] = self._get_observations()["policy"].clone()
            return terminated | failed, time_out
        return wrapped

    def _reference_control(self, xref, parameters, time):
        _, control = self._traj_buf.get(parameters[:, 0].long().clamp(0, self._traj_buf.num_trajs - 1),
                                        (time / self.step_dt).round().long())
        return control

    def step(self, action):
        low, high = self._control_bounds()
        action = torch.as_tensor(action, device=self.device).nan_to_num().clamp(
            torch.as_tensor(low, device=self.device), torch.as_tensor(high, device=self.device))
        if getattr(self.ref_window, "history_length", 0):
            current = self._sanitize_state(self._get_physical_state())
            row = torch.cat((current, action, current.new_ones(self.num_envs, 1)), -1)
            self._tracking_history = torch.cat((self._tracking_history[:, 1:], row[:, None]), 1)
        gust = self.disturbance.advance()
        body = self._ee_id[:1] if self.ref_window.u_dim == 7 else [0]
        self._robot.set_external_force_and_torque(gust[:, None, :3], gust[:, None, 3:],
                                                   body_ids=body, is_global=True)
        return super().step(action)

    def _state_bounds(self):
        """Finite physical domains shared with nominal reference validation."""
        n = self.ref_window.x_dim
        from .reference_domain import robot_domain
        low, high, _ = robot_domain(self._traj_buf.action_dim, getattr(self.cfg, "base_height_min", None))
        if len(low) != n:
            raise ValueError(f"Reference state width {n} does not match robot layout {len(low)}")
        return low, high

    def _control_bounds(self):
        space = getattr(self, "single_action_space", self.action_space)
        lo = np.asarray(space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return lo, hi

    def configure_encoder_context(self, horizon: int):
        """Expose nominal bank futures and measured history without changing the task."""
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            raise ValueError("Encoder context requires a nonnegative integer window")
        self.ref_window.preview_length = self.ref_window.history_length = horizon
        self.ref_window.preview_start = self.ref_window.preview_stride = 1
        self.ref_window.temporal_context = bool(horizon)
        self._tracking_history = torch.zeros(self.num_envs, horizon,
            self.ref_window.x_dim + self.ref_window.u_dim + 1, device=self.device)
        self._rebuild_observation_space()

    def configure_ref_window(self, length: int, offset: int = 1,
                             gamma: float | None = None) -> None:
        """Resize the reference window and rebuild ``observation_space`` —
        the Isaac twin of ``BaseEnv.configure_ref_window``. Must be called
        before the agent/memory are built. Idempotent."""
        if getattr(self.ref_window, "compact", False):
            return
        self.ref_window = RefWindow(x_dim=self.ref_window.x_dim, u_dim=self.ref_window.u_dim,
                                    length=int(length), offset=int(offset))
        self._rebuild_observation_space()
        print(f"[PathTrackingBase] reference window: length={self.ref_window.length} "
              f"offset={self.ref_window.offset} (spans "
              f"{(self.ref_window.length - 1) * self.ref_window.offset} steps) -> "
              f"obs {self.ref_window.flat_dim}-wide")
        if gamma is not None:
            warn = self.ref_window.check_markov(float(gamma), int(self.max_episode_length))
            if warn:
                print(warn)

    def _get_observations(self) -> dict:
        step = self.episode_length_buf.long()
        if getattr(self.ref_window, "compact", False):
            self._x_ref, self._u_ref = self._traj_buf.get(self._traj_ids, step)
            x = self._sanitize_state(self._get_physical_state())
            xi = torch.cat((self._traj_ids.float().unsqueeze(-1),
                            (step * self.step_dt).unsqueeze(-1),
                            x.new_zeros(self.num_envs, self.ref_window.u_dim)), -1)
            future = None
            if self.ref_window.preview_length:
                points = [self._traj_buf.get(self._traj_ids,
                          (step + offset).clamp(max=self.max_episode_length))[0]
                          for offset in range(1, self.ref_window.preview_length + 1)]
                future = torch.stack(points, 1)
            history = self._tracking_history if self.ref_window.history_length else None
            return {"policy": self.ref_window.flatten(x, self._x_ref, None, xi=xi,
                                                       preview=future, history=history)}
        x_refs, u_refs = self._traj_buf.get_window(self._traj_ids, step, self.ref_window)
        # Slot 0 is the current reference — what the reward and every metric use.
        self._x_ref, self._u_ref = x_refs[:, 0], u_refs[:, 0]
        # sanitize so the policy never receives NaN/Inf from a diverging env
        x = self._sanitize_state(self._get_physical_state())
        # Flat, in skrl's sorted-key Dict order (RefWindow.flatten) — Isaac's
        # DirectRLEnv passes the "policy" tensor straight through without a
        # Dict-space tensorize step, so flattening happens here.
        return {"policy": self.ref_window.flatten(x, x_refs, u_refs)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh _x_ref/_u_ref for this step here rather than relying on
        # _get_observations() to have set them: DirectRLEnv.step() calls
        # _get_rewards() before _get_observations() (after incrementing
        # episode_length_buf), so without this, _x_ref/_u_ref would still hold
        # the previous step's values — pairing the post-transition physical
        # state x below against a one-step-stale reference. _traj_buf.get is a
        # pure lookup (no side effects), so recomputing it again in
        # _get_observations() with the same episode_length_buf is harmless.
        step = self.episode_length_buf.long()
        self._x_ref, self._u_ref = self._traj_buf.get(self._traj_ids, step)

        # sanitize + clamp so a diverging env yields a large-but-Finite penalty
        # (never NaN/Inf) into the reward, AUC, and Stability curves — the
        # episode is never reset/terminated because of it (see _sanitize_state).
        x = self._sanitize_state(self._get_physical_state())
        # angle_idx dims (e.g. yaw) are wrapped to the shortest-angle difference
        # Before any norm/reward/metric — a raw difference would spike by ~2*pi
        # whenever the physical angle wraps (e.g. a yaw U-turn). Non-angle dims
        # are unaffected (wrap_diff is a no-op there, and a no-op entirely when
        # angle_idx is empty).
        error = wrap_diff(x - self._x_ref, self.angle_idx)
        error_norm = torch.norm(error, dim=-1).clamp(max=self._ERROR_CLAMP)   # (N,)
        self.extras["terminal_tracking_error"] = error_norm.clone()
        self.extras["tracking_error_vector"] = error.detach().clone()
        self.extras["physical_tracking_error_vector"] = (x - self._x_ref).detach().clone()
        self.extras["tracking_error"] = error.square().sum(-1).detach()

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

        if self.ccm_gen is not None:
            # C2RL Phase B: Lyapunov-decrease reward tracking_scaler*(V - next_V)
            # - control_scaler*||u||^2, V = error^T M error, M = the frozen
            # CMG's metric at the corresponding state — mirrors classic
            # analytical_env.py's get_rewards exactly. _get_rewards only ever sees
            # the post-transition state (no separate "prev x" lookup buffer
            # like the reference trajectory has), so the previous step's
            # error vector + metric are cached explicitly in
            # _prev_error/_prev_M (seeded at _reset_idx) rather than
            # recomputed from a stored trajectory.
            with torch.no_grad():
                next_M = self._metric_from_cmg(x, self._x_ref)

                V = tracking_energy(self._prev_error, self._prev_M, self.angle_idx)
                next_V = tracking_energy(error, next_M, self.angle_idx)

                control_effort = torch.norm(self._actions, p=2, dim=-1) ** 2
                reward = energy_decrease(V, next_V)

                self._prev_error = error.clone()
                self._prev_M = next_M
        else:
            # -||error||_I^2, using the sanitized+clamped norm so a diverged step
            # can't emit a huge/NaN reward that wrecks the value fn / normalizers.
            next_energy = tracking_energy(error, angle_idx=self.angle_idx)
            reward = energy_decrease(self._prev_energy, next_energy)
            self._prev_energy = next_energy
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

            # --- Unified streaming contraction metrics for every finished env ---
            # auc (normalized dt-weighted trapezoid), contraction_rate (lambda),
            # overshoot (e_max/e0) and contraction_score — each mean + 95% CI,
            # using the same streaming computation and wandb keys as C3M/C2RL
            # (contraction_metrics), so PPO/SAC/LQR/SD-LQR share the same
            # Stability/* tab, formula and CI as the contraction algorithms.
            # Computed over all finished envs (not just the few viz envs the old
            # fit_exponential_envelope path used), for tighter statistics.
            from contractionRL.agents.utils.contraction.metrics import per_env_metrics, stability_log_dict, summarize
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
            from contractionRL.agents.utils.contraction.metrics import reward_log_dict, reward_summary
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

        # sample reference trajectories — unless fix_ref_trajectories pinned a
        # permanent one per env slot on its first reset (see
        # set_fix_ref_trajectories); then only the not-yet-assigned slots draw.
        if getattr(self, "fix_ref_trajectories", False):
            fresh = ~self._ref_fixed[env_ids]
            if bool(fresh.any()):
                new_ids = env_ids[fresh]
                self._traj_ids[new_ids] = self._traj_buf.sample_traj_ids(len(new_ids))
                self._ref_fixed[new_ids] = True
        else:
            self._traj_ids[env_ids] = self._traj_buf.sample_traj_ids(len(env_ids))

        # initialise robot to match x_ref at step 0
        x_ref_init = self._traj_buf.initial_state(self._traj_ids[env_ids])  # (n, state_dim)
        self._set_robot_state_from_ref(env_ids, x_ref_init)
        if hasattr(self, '_continuous_state'):
            self._continuous_state[env_ids] = x_ref_init
        self.disturbance.reset(env_ids)
        initial_error = wrap_diff(self._get_physical_state()[env_ids] - x_ref_init, self.angle_idx)
        self._prev_energy[env_ids] = initial_error.square().sum(-1)

        # resync the divergence guard's baseline to the freshly-reset physical
        # state, so it doesn't carry forward a stale pre-reset value into the
        # new episode.
        if self._last_valid_state is not None:
            self._last_valid_state[env_ids] = self._get_physical_state()[env_ids]

        # seed the CCM Lyapunov cache (_get_rewards) from the freshly-reset
        # state, so the first step of the new episode compares against a
        # meaningful V rather than a stale value from the env's previous
        # episode — mirrors classic analytical_env.py's reset_idx M-seeding.
        if self.ccm_gen is not None:
            if not hasattr(self, "_prev_M"):
                state_dim = self._traj_buf.state_dim
                self._prev_error = torch.zeros(self.num_envs, state_dim, device=self.device)
                self._prev_M = torch.zeros(self.num_envs, state_dim, state_dim, device=self.device)
            with torch.no_grad():
                x0 = self._get_physical_state()[env_ids]
                error0 = wrap_diff(x0 - x_ref_init, self.angle_idx)
                self._prev_error[env_ids] = error0
                self._prev_M[env_ids] = self._metric_from_cmg(x0, x_ref_init)
        if getattr(self.ref_window, "history_length", 0):
            self._tracking_history[env_ids] = 0

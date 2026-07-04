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

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from .eval_metrics import mean_confidence_interval
from .traj_buffer import TrajectoryBuffer

# wandb is optional; only used when a run is active
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

_WANDB_PLOT_INTERVAL = 20   # log trajectory plot every N completed episodes (env 0)


class PathTrackingBase(DirectRLEnv):
    """Abstract base for path-tracking environments.

    Subclass must define:
        cfg.traj_path   : str — path to .npz reference trajectory file
        cfg.action_space, observation_space
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._traj_buf = TrajectoryBuffer(cfg.traj_path, self.device)

        n = self.num_envs
        self._traj_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        self._x_ref = torch.zeros(n, self._traj_buf.state_dim, device=self.device)
        self._u_ref = torch.zeros(n, self._traj_buf.action_dim, device=self.device)
        self._actions = torch.zeros(n, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # --- episode-level eval metric accumulators ---
        self._episode_auc = torch.zeros(n, device=self.device)
        self._episode_contraction_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._episode_first_error = torch.zeros(n, device=self.device)
        self._episode_last_error = torch.zeros(n, device=self.device)
        self._episode_max_error = torch.zeros(n, device=self.device)
        self._prev_error_norm = torch.full((n,), float("inf"), device=self.device)
        self._episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        # trajectory buffer for env 0 wandb plot (CPU, pre-allocated)
        self._env0_error_buf = np.zeros(self.max_episode_length, dtype=np.float32)
        self._env0_step = 0
        self._env0_episode_count = 0

        # Contraction interface — dynamics model injected by ContractionRunner
        self._dynamics_model = None

    # ------------------------------------------------------------------ #
    # Contraction algorithm interface (C3M / LQR / SD-LQR / TEMP)
    # ------------------------------------------------------------------ #

    def set_dynamics_model(self, model) -> None:
        """Inject a NeuralDynamics model for get_f_and_B (required for Isaac envs)."""
        self._dynamics_model = model

    def get_f_and_B(self, x):
        """Return (f, B, B_null) for contraction agents.

        Delegates to the injected NeuralDynamics model — call
        ``set_dynamics_model(model)`` before using C3M/LQR/SDLQR/TEMP.
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
        envelope C * exp(-lambda * k * dt) bounding the error curve.
        """
        return torch.norm(self._get_physical_state() - self._x_ref, dim=-1)

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
        x = self._get_physical_state()
        obs = torch.cat([x, self._x_ref, self._u_ref], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        x = self._get_physical_state()
        error = x - self._x_ref
        error_norm = torch.norm(error, dim=-1)   # (N,)

        # accumulate AUC
        self._episode_auc += error_norm

        # record first / last error per env
        is_first = (self._episode_steps == 0)
        self._episode_first_error = torch.where(is_first, error_norm, self._episode_first_error)
        self._episode_last_error = error_norm
        self._episode_max_error = torch.maximum(self._episode_max_error, error_norm)

        # contraction flag: count steps where error strictly decreased (skip step 0)
        contracting = (~is_first) & (error_norm < self._prev_error_norm)
        self._episode_contraction_steps += contracting.long()

        self._prev_error_norm = error_norm.clone()
        self._episode_steps += 1

        # collect env-0 trajectory for wandb plot
        if self._env0_step < self.max_episode_length:
            self._env0_error_buf[self._env0_step] = error_norm[0].item()
            self._env0_step += 1

        return -torch.sum(error * error, dim=-1)   # -||error||_I^2

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # --- log episode metrics for envs that actually ran ---
        finished = env_ids[self._episode_steps[env_ids] > 0]
        if len(finished) > 0:
            self.extras.setdefault("log", {})

            auc = self._episode_auc[finished].float()
            steps = self._episode_steps[finished].float().clamp(min=1)
            e0 = self._episode_first_error[finished].float().clamp(min=1e-8)
            eT = self._episode_last_error[finished].float().clamp(min=1e-8)
            e_max = self._episode_max_error[finished].float().clamp(min=1e-8)
            T = steps
            dt = self.step_dt

            # empirical contraction rate: e(T) = e(0) * exp(-lambda * T*dt)
            lambda_emp = -(torch.log(eT) - torch.log(e0)) / (T * dt)

            # overshoot: peak error relative to the initial error — a contracting
            # trajectory should never exceed e(0), so overshoot > 1 flags excursions
            overshoot = e_max / e0

            # contraction flag: fraction of contracting steps (skip step 0)
            valid_steps = (steps - 1).clamp(min=1)
            contraction_flag = self._episode_contraction_steps[finished].float() / valid_steps

            performance_score = -(auc / steps)

            self.extras["log"]["Episode/contraction_flag"] = contraction_flag.mean()
            self.extras["log"]["Episode/performance_score"] = performance_score.mean()

            auc_mean, auc_ci95 = mean_confidence_interval(auc.cpu().numpy())
            lbd_mean, lbd_ci95 = mean_confidence_interval(lambda_emp.cpu().numpy())
            os_mean, os_ci95 = mean_confidence_interval(overshoot.cpu().numpy())

            self.extras["log"]["Stability/auc"] = torch.tensor(auc_mean, device=self.device)
            self.extras["log"]["Stability/auc_ci95"] = torch.tensor(auc_ci95, device=self.device)
            self.extras["log"]["Stability/contraction_rate"] = torch.tensor(lbd_mean, device=self.device)
            self.extras["log"]["Stability/contraction_rate_ci95"] = torch.tensor(lbd_ci95, device=self.device)
            self.extras["log"]["Stability/overshoot"] = torch.tensor(os_mean, device=self.device)
            self.extras["log"]["Stability/overshoot_ci95"] = torch.tensor(os_ci95, device=self.device)

        # --- wandb trajectory plot for env 0 ---
        env0_done = env_ids is not None and any(int(e) == 0 for e in env_ids)
        if env0_done and self._env0_step > 0:
            self._env0_episode_count += 1
            if (
                _wandb is not None
                and self._env0_episode_count % _WANDB_PLOT_INTERVAL == 0
                and getattr(_wandb, "run", None) is not None
            ):
                traj = self._env0_error_buf[:self._env0_step].tolist()
                data = [[t, e] for t, e in enumerate(traj)]
                table = _wandb.Table(data=data, columns=["step", "tracking_error"])  # type: ignore[attr-defined]
                _wandb.log({  # type: ignore[attr-defined]
                    "PathTracking/error_trajectory": _wandb.plot.line(  # type: ignore[attr-defined]
                        table, "step", "tracking_error",
                        title="Tracking Error Over Episode (env 0)"
                    )
                })
            self._env0_step = 0

        # --- reset per-env buffers ---
        self._episode_auc[env_ids] = 0.0
        self._episode_contraction_steps[env_ids] = 0
        self._episode_first_error[env_ids] = 0.0
        self._episode_last_error[env_ids] = 0.0
        self._episode_max_error[env_ids] = 0.0
        self._prev_error_norm[env_ids] = float("inf")
        self._episode_steps[env_ids] = 0

        # sample reference trajectories
        self._traj_ids[env_ids] = self._traj_buf.sample_traj_ids(len(env_ids))

        # initialise robot to match x_ref at step 0
        x_ref_init = self._traj_buf.initial_state(self._traj_ids[env_ids])  # (n, state_dim)
        self._set_robot_state_from_ref(env_ids, x_ref_init)

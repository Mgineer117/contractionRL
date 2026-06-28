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

import torch

from isaaclab.envs import DirectRLEnv

from .traj_buffer import TrajectoryBuffer


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
        return -torch.sum(error * error, dim=-1)   # -||error||_I^2

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # sample reference trajectories
        self._traj_ids[env_ids] = self._traj_buf.sample_traj_ids(len(env_ids))

        # initialise robot to match x_ref at step 0
        x_ref_init = self._traj_buf.initial_state(self._traj_ids[env_ids])  # (n, state_dim)
        self._set_robot_state_from_ref(env_ids, x_ref_init)

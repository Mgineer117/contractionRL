from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .manipulator_env_cfg import ManipulatorEnvCfg


class ManipulatorEnv(DirectRLEnv):
    cfg: ManipulatorEnvCfg

    def __init__(self, cfg: ManipulatorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # arm joint indices (excludes finger joints)
        self._arm_joint_ids, _ = self._robot.find_joints(self.cfg.arm_joint_names)
        # EE body index
        self._ee_body_id, _ = self._robot.find_bodies(self.cfg.ee_body_name)

        self._actions = torch.zeros(self.num_envs, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        # target positions in world frame (per env)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        # map [-1,1] actions to joint position targets within joint limits
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._arm_joint_ids, :]
        joint_pos_mid = 0.5 * (joint_pos_limits[..., 0] + joint_pos_limits[..., 1])
        joint_pos_range = 0.5 * (joint_pos_limits[..., 1] - joint_pos_limits[..., 0])
        self._joint_targets = joint_pos_mid + joint_pos_range * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._arm_joint_ids)

    def _get_observations(self) -> dict:
        # EE position in world frame → subtract env origin for local coords
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id[0], :]  # (N, 3)
        ee_pos_local = ee_pos_w - self.scene.env_origins

        # target is already stored in world frame; convert to local
        target_local = self._target_pos - self.scene.env_origins

        obs = torch.cat(
            [
                self._robot.data.joint_pos[:, self._arm_joint_ids],   # (N, 7)
                self._robot.data.joint_vel[:, self._arm_joint_ids],   # (N, 7)
                ee_pos_local,                                          # (N, 3)
                target_local,                                          # (N, 3)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id[0], :]
        dist = torch.norm(ee_pos_w - self._target_pos, dim=-1)

        rew_reach = dist * self.cfg.rew_reach_scale
        rew_success = (dist < self.cfg.success_threshold).float() * self.cfg.rew_success_scale
        rew_action_rate = (
            torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate_scale
        )

        return rew_reach + rew_success + rew_action_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # only timeout termination; no fall/collision detection for reach task
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # sample new target positions (world frame = local + env_origin)
        n = len(env_ids)
        target_local = torch.stack(
            [
                sample_uniform(*self.cfg.target_pos_range_x, (n,), self.device),
                sample_uniform(*self.cfg.target_pos_range_y, (n,), self.device),
                sample_uniform(*self.cfg.target_pos_range_z, (n,), self.device),
            ],
            dim=-1,
        )
        self._target_pos[env_ids] = target_local + self.scene.env_origins[env_ids]

        # reset to default joint configuration
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

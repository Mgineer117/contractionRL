from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .quadruped_env_cfg import QuadrupedEnvCfg


class QuadrupedEnv(DirectRLEnv):
    cfg: QuadrupedEnvCfg

    def __init__(self, cfg: QuadrupedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        # velocity commands: [lin_vel_x, lin_vel_y, yaw_rate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

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
        # position targets = default + scaled deviation
        self._joint_targets = self._robot.data.default_joint_pos + self.cfg.action_scale * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,           # (N, 3)
                self._robot.data.root_ang_vel_b,           # (N, 3)
                self._robot.data.projected_gravity_b,      # (N, 3)
                self._commands,                            # (N, 3)
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # (N, 12)
                self._robot.data.joint_vel,                # (N, 12)
                self._actions,                             # (N, 12)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking (xy)
        lin_vel_err = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rew_lin_vel = torch.exp(-lin_vel_err / 0.25) * self.cfg.rew_lin_vel_scale

        # yaw rate tracking
        yaw_err = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        rew_yaw = torch.exp(-yaw_err / 0.25) * self.cfg.rew_yaw_rate_scale

        # penalise vertical base velocity
        rew_z_vel = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.rew_z_vel_scale

        # penalise lateral base angular velocity
        rew_ang_vel_xy = (
            torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
            * self.cfg.rew_ang_vel_xy_scale
        )

        # penalise joint torques
        rew_torques = (
            torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.rew_torques_scale
        )

        # penalise action rate
        rew_action_rate = (
            torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate_scale
        )

        # alive bonus
        rew_alive = (1.0 - self.reset_terminated.float()) * self.cfg.rew_alive_scale

        return rew_lin_vel + rew_yaw + rew_z_vel + rew_ang_vel_xy + rew_torques + rew_action_rate + rew_alive

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # terminate if base height drops below threshold
        base_height = self._robot.data.root_pos_w[:, 2]
        fell = base_height < self.cfg.base_height_min
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # sample new velocity commands
        lo, hi = self.cfg.cmd_lin_vel_range
        self._commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(lo, hi)
        self._commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device).uniform_(lo, hi)
        lo, hi = self.cfg.cmd_yaw_range
        self._commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(lo, hi)

        # reset robot state to defaults
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..common.vel_commands import VelCommands
from .env_cfg import QuadrupedVelTrackingEnvCfg


class QuadrupedVelTrackingEnv(DirectRLEnv):
    """Unitree Go2 velocity-tracking environment.

    Commands: constant (vx, vy) + sinusoidal yaw rate.
    State for path-tracking export:
        base_lin_vel_b(3) + base_ang_vel_b(3) + proj_gravity_b(3)
        + joint_pos_rel(12) + joint_vel(12)  → 33 dims
    """

    cfg: QuadrupedVelTrackingEnvCfg

    STATE_DIM = 33   # dims recorded as reference state for path tracking

    def __init__(self, cfg: QuadrupedVelTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, _ = self._robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
        self._actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)

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
        default_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets = default_pos + self.cfg.action_scale * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    # ------------------------------------------------------------------ #
    # State extraction — also called by generate_ref_traj.py
    # ------------------------------------------------------------------ #

    def get_physical_state(self) -> torch.Tensor:
        """Returns (N, 33) physical state without commands or actions.

        Layout: [base_lin_vel_b(3), base_ang_vel_b(3), proj_gravity_b(3),
                 joint_pos_rel(12), joint_vel(12)]
        """
        return torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)                  # (N, 4)
        state = self.get_physical_state()         # (N, 33)
        obs = torch.cat([state, cmds, self._prev_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)

        # linear velocity tracking (xy)
        lin_err = torch.sum(
            torch.square(cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rew_lin = torch.exp(-lin_err / 0.25) * self.cfg.rew_lin_vel

        # yaw rate tracking
        yaw_err = torch.square(cmds[:, 3] - self._robot.data.root_ang_vel_b[:, 2])
        rew_yaw = torch.exp(-yaw_err / 0.25) * self.cfg.rew_yaw_rate

        rew_z = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.rew_z_vel
        rew_axy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.rew_ang_vel_xy
        rew_tau = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.rew_torques
        rew_ar = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate
        rew_alive = (1.0 - self.reset_terminated.float()) * self.cfg.rew_alive

        return rew_lin + rew_yaw + rew_z + rew_axy + rew_tau + rew_ar + rew_alive

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._cmd.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root = self._robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------ #
    # Debug visualisation: blue = command vel, green = current vel
    # ------------------------------------------------------------------ #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_cmd_vel_marker"):
                cmd_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/QuadrupedVelCmd")
                cur_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/QuadrupedVelCur")
                self._cmd_vel_marker = VisualizationMarkers(cmd_cfg)
                self._cur_vel_marker = VisualizationMarkers(cur_cfg)
            self._cmd_vel_marker.set_visibility(True)
            self._cur_vel_marker.set_visibility(True)
        else:
            if hasattr(self, "_cmd_vel_marker"):
                self._cmd_vel_marker.set_visibility(False)
                self._cur_vel_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # event stream can fire before the scene is ready or during teardown
        if not hasattr(self, "scene") or not self._robot.is_initialized:
            return
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 4): [vx_b, vy_b, vz, yaw]

        base_pos = self._robot.data.root_pos_w  # (N, 3)
        cmd_pos = base_pos.clone(); cmd_pos[:, 2] += 1.5
        cur_pos = base_pos.clone(); cur_pos[:, 2] += 1.1

        cmd_quat, cmd_spd = self._vel_body_xy_to_arrow(cmds[:, :2])
        cur_quat, cur_spd = self._vel_body_xy_to_arrow(self._robot.data.root_lin_vel_b[:, :2])

        # length grows with speed (gain); width fixed so arrows stay visible at low speed
        len_gain, width = 2.5, 0.4
        cmd_scale = torch.cat([cmd_spd * len_gain, torch.full_like(cmd_spd, width), torch.full_like(cmd_spd, width)], dim=-1)
        cur_scale = torch.cat([cur_spd * len_gain, torch.full_like(cur_spd, width), torch.full_like(cur_spd, width)], dim=-1)

        proto = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._cmd_vel_marker.visualize(translations=cmd_pos, orientations=cmd_quat, scales=cmd_scale, marker_indices=proto)
        self._cur_vel_marker.visualize(translations=cur_pos, orientations=cur_quat, scales=cur_scale, marker_indices=proto)

    def _vel_body_xy_to_arrow(self, vel_body_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-frame XY velocity → world-frame arrow quaternion (wxyz) + speed scale."""
        q = self._robot.data.root_quat_w  # (N, 4) w,x,y,z
        yaw = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
        )
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx_w = cy * vel_body_xy[:, 0] - sy * vel_body_xy[:, 1]
        vy_w = sy * vel_body_xy[:, 0] + cy * vel_body_xy[:, 1]
        angle = torch.atan2(vy_w, vx_w)
        ha = angle * 0.5
        quat = torch.stack([torch.cos(ha), torch.zeros_like(ha), torch.zeros_like(ha), torch.sin(ha)], dim=-1)
        speed = vel_body_xy.norm(dim=-1).clamp(0.1, 2.0).unsqueeze(-1)
        return quat, speed

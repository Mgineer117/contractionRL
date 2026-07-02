from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..common.vel_commands import VelCommands
from .env_cfg import HumanoidVelTrackingEnvCfg


class HumanoidVelTrackingEnv(DirectRLEnv):
    """Unitree H1 humanoid velocity-tracking environment.

    State for path-tracking export (41 dims):
        proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19)
    """

    cfg: HumanoidVelTrackingEnvCfg

    STATE_DIM = 41

    def __init__(self, cfg: HumanoidVelTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, _ = self._robot.find_joints([
            ".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso",
            ".*_ankle",
            ".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow",
        ])
        self._actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)
        self._episode_vel_auc = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # Explicit, strongly-contrasting ground color (dark blue-teal) instead of relying on
        # GroundPlaneCfg's default grid-texture tint — the robot is light-colored, so a dark,
        # saturated (non-gray) ground reads clearly against it regardless of scene brightness.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        default_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets = default_pos + self.cfg.action_scale * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def get_physical_state(self) -> torch.Tensor:
        """Returns (N, 41) physical state without commands/actions.

        Layout: [proj_gravity_b(3), joint_pos_rel(19), joint_vel(19)]
        """
        return torch.cat(
            [
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 4)
        # Full 47-dim state for the locomotion policy (needs velocities to track commands)
        full_state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )
        obs = torch.cat([full_state, cmds, self._prev_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)

        lin_err = torch.sum(torch.square(cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        rew_lin = torch.exp(-lin_err / 0.1) * self.cfg.rew_lin_vel

        yaw_err = torch.square(cmds[:, 3] - self._robot.data.root_ang_vel_b[:, 2])
        rew_yaw = torch.exp(-yaw_err / 0.1) * self.cfg.rew_yaw_rate

        rew_z = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.rew_z_vel
        rew_axy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.rew_ang_vel_xy
        rew_up = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) * self.cfg.rew_upright
        rew_tau = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.rew_torques
        rew_ar = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate
        
        vel_err_vec = torch.cat([
            cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2],
            (cmds[:, 3] - self._robot.data.root_ang_vel_b[:, 2]).unsqueeze(-1),
        ], dim=-1)
        self._episode_vel_auc += torch.norm(vel_err_vec, dim=-1)

        return rew_lin + rew_yaw + rew_z + rew_axy + rew_up + rew_tau + rew_ar + rew_alive

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        auc_vals = self._episode_vel_auc[env_ids]
        if hasattr(self, "_episode_discounted_returns"):
            disc_returns = self._episode_discounted_returns[env_ids]
            undisc_returns = self._episode_undiscounted_returns[env_ids]
            lengths = self._episode_lengths_custom[env_ids]
            if (auc_vals > 0).any():
                self.extras.setdefault("log", {})
                self.extras["log"]["Episode/auc"] = auc_vals[auc_vals > 0].mean().item()
                self.extras["log"]["Episode/discounted_return"] = disc_returns[auc_vals > 0].mean().item()
                self.extras["log"]["Episode/undiscounted_return"] = undisc_returns[auc_vals > 0].mean().item()
                self.extras["log"]["Episode/avg_reward_per_step"] = (undisc_returns[auc_vals > 0] / lengths[auc_vals > 0]).mean().item()
            
            self._episode_discounted_returns[env_ids] = 0.0
            self._episode_undiscounted_returns[env_ids] = 0.0
            self._episode_lengths_custom[env_ids] = 0.0
            self._current_discounts[env_ids] = 1.0
            
        self._episode_vel_auc[env_ids] = 0.0

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._cmd.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]

        if self.cfg.randomize_init:
            n = len(env_ids)
            # Random yaw: sample θ ∈ [0, 2π], build wxyz quaternion around Z
            theta = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
            cos_h, sin_h = torch.cos(theta * 0.5), torch.sin(theta * 0.5)
            root[:, 3:7] = torch.stack(
                [cos_h, torch.zeros_like(cos_h), torch.zeros_like(cos_h), sin_h], dim=-1
            )
            # Random x,y offset (spread robots across scene)
            root[:, :2] += torch.empty(n, 2, device=self.device).uniform_(
                -self.cfg.init_pos_range, self.cfg.init_pos_range
            )
            # Joint noise: perturb around default pose for trajectory diversity
            joint_pos[:, self._joint_ids] += torch.empty(
                n, len(self._joint_ids), device=self.device
            ).uniform_(-self.cfg.init_joint_noise, self.cfg.init_joint_noise)

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------ #
    # Debug visualisation: blue = command vel, green = current vel
    # ------------------------------------------------------------------ #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_cmd_vel_marker"):
                _arrow = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"
                cmd_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/HumanoidVelCmd",
                    markers={"arrow": sim_utils.UsdFileCfg(
                        usd_path=_arrow, scale=(0.6, 0.06, 0.06),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    )},
                )
                cur_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/HumanoidVelCur",
                    markers={"arrow": sim_utils.UsdFileCfg(
                        usd_path=_arrow, scale=(0.6, 0.06, 0.06),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )},
                )
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

        cmd_quat = self._vel_body_xy_to_arrow(cmds[:, :2])
        cur_quat = self._vel_body_xy_to_arrow(self._robot.data.root_lin_vel_b[:, :2])

        fixed_scale = torch.ones(self.num_envs, 3, device=self.device)
        proto = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._cmd_vel_marker.visualize(translations=cmd_pos, orientations=cmd_quat, scales=fixed_scale, marker_indices=proto)
        self._cur_vel_marker.visualize(translations=cur_pos, orientations=cur_quat, scales=fixed_scale, marker_indices=proto)

    def _vel_body_xy_to_arrow(self, vel_body_xy: torch.Tensor) -> torch.Tensor:
        """Body-frame XY velocity → world-frame arrow quaternion (wxyz)."""
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
        return torch.stack([torch.cos(ha), torch.zeros_like(ha), torch.zeros_like(ha), torch.sin(ha)], dim=-1)

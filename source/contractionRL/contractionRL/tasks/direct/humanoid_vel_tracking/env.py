from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from ..common.eval_metrics import mean_confidence_interval
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
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
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
        rew_rp = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.rew_roll_pitch
        rew_flat = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) * self.cfg.rew_flat_orientation
        rew_tau = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.rew_torque
        rew_ar = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate

        vel_err_vec = cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2]
        self._episode_vel_auc += torch.norm(vel_err_vec, dim=-1)

        total_reward = self.cfg.rew_alive + rew_lin + rew_yaw + rew_flat + rew_z + rew_rp + rew_tau + rew_ar

        if not hasattr(self, "_episode_discounted_returns"):
            self._episode_discounted_returns = torch.zeros(self.num_envs, device=self.device)
            self._current_discounts = torch.ones(self.num_envs, device=self.device)
            self._episode_undiscounted_returns = torch.zeros(self.num_envs, device=self.device)
            self._episode_lengths_custom = torch.zeros(self.num_envs, device=self.device)

        self._episode_discounted_returns += self._current_discounts * total_reward
        self._episode_undiscounted_returns += total_reward
        self._episode_lengths_custom += 1.0
        self._current_discounts *= 0.99

        return total_reward

    def get_tracking_error(self) -> torch.Tensor:
        """Current velocity-tracking error norm per env, (N,).

        Same integrand as the Episode/auc metric: body-frame xy velocity error
        against the commanded velocity. Used by the post-training evaluator to
        fit the exponential contraction envelope.
        """
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)
        return torch.norm(cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Fall = base dropped too low OR body tilted past the limit. projected
        # gravity z is -1 upright and rises toward 0/+1 as the base tilts, so
        # ``> fall_grav_z_max`` (default -0.71) fires at ~>45 deg from upright.
        too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        tilted = self._robot.data.projected_gravity_b[:, 2] > self.cfg.fall_grav_z_max
        fell = too_low | tilted
        if not getattr(self.cfg, "terminate_on_fall", True):
            fell = torch.zeros_like(time_out)
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
                mask = auc_vals > 0
                self.extras["log"]["Episode/auc"] = auc_vals[mask].mean()
                self.extras["log"]["Reward/discounted_return"] = disc_returns[mask].mean()
                self.extras["log"]["Reward/avg_reward_per_step"] = (undisc_returns[mask] / lengths[mask]).mean()
                # undiscounted_return is dropped here — it's the same quantity skrl
                # already tracks as "Reward / Total reward (mean)"; only its 95% CI
                # (not available from skrl's tracker) is worth adding.
                _, reward_ci95 = mean_confidence_interval(undisc_returns[mask].cpu().numpy())
                self.extras["log"]["Reward/total_reward_ci95"] = torch.tensor(reward_ci95, device=self.device)
            
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
    # Debug visualisation: blue = command vel, green = current vel, yellow = yaw rate
    #
    # Arrows are built from two 3D primitives (cylinder shaft + cone head,
    # both axis="X") rather than a flat arrow mesh — a flat mesh looks like a
    # thin line (direction ambiguous) from many camera angles, while a
    # cylinder+cone is a real 3D solid, recognizable from any angle.
    # ------------------------------------------------------------------ #

    _ARROW_SHAFT_LEN = 0.4
    _ARROW_SHAFT_RADIUS = 0.02
    _ARROW_HEAD_LEN = 0.2
    _ARROW_HEAD_RADIUS = 0.05

    def _make_arrow_markers_cfg(self, prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        return VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "shaft": sim_utils.CylinderCfg(
                    radius=self._ARROW_SHAFT_RADIUS, height=self._ARROW_SHAFT_LEN,
                    axis="X", visual_material=material,
                ),
                "head": sim_utils.ConeCfg(
                    radius=self._ARROW_HEAD_RADIUS, height=self._ARROW_HEAD_LEN,
                    axis="X", visual_material=material,
                ),
            },
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_cmd_vel_marker"):
                cmd_cfg = self._make_arrow_markers_cfg("/Visuals/HumanoidVelCmd", (0.0, 0.0, 1.0))
                cur_cfg = self._make_arrow_markers_cfg("/Visuals/HumanoidVelCur", (0.0, 1.0, 0.0))
                yaw_cfg = self._make_arrow_markers_cfg("/Visuals/HumanoidYawCmd", (1.0, 0.9, 0.0))
                self._cmd_vel_marker = VisualizationMarkers(cmd_cfg)
                self._cur_vel_marker = VisualizationMarkers(cur_cfg)
                self._yaw_cmd_marker = VisualizationMarkers(yaw_cfg)
            self._cmd_vel_marker.set_visibility(True)
            self._cur_vel_marker.set_visibility(True)
            self._yaw_cmd_marker.set_visibility(True)
        else:
            if hasattr(self, "_cmd_vel_marker"):
                self._cmd_vel_marker.set_visibility(False)
                self._cur_vel_marker.set_visibility(False)
                self._yaw_cmd_marker.set_visibility(False)

    def _arrow_parts(self, base_pos: torch.Tensor, quat: torch.Tensor, scale_len: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (translations, orientations, marker_indices) for a shaft+head arrow at base_pos, pointing along quat."""
        n = base_pos.shape[0]
        if scale_len is None:
            scale_len = torch.ones(n, device=self.device)

        s = scale_len.unsqueeze(-1)
        shaft_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN / 2, 0.0, 0.0], device=self.device
        ).expand(n, -1) * s
        head_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN, 0.0, 0.0], device=self.device
        ).expand(n, -1) * s + torch.tensor(
            [self._ARROW_HEAD_LEN / 2, 0.0, 0.0], device=self.device
        ).expand(n, -1)

        shaft_pos = base_pos + quat_apply(quat, shaft_offset)
        head_pos = base_pos + quat_apply(quat, head_offset)

        translations = torch.cat([shaft_pos, head_pos], dim=0)
        orientations = torch.cat([quat, quat], dim=0)
        marker_indices = torch.cat([
            torch.zeros(n, dtype=torch.int32, device=self.device),  # 0 -> "shaft"
            torch.ones(n, dtype=torch.int32, device=self.device),   # 1 -> "head"
        ], dim=0)
        return translations, orientations, marker_indices

    def _debug_vis_callback(self, event):
        # event stream can fire before the scene is ready or during teardown
        if not hasattr(self, "scene") or not self._robot.is_initialized:
            return
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 4): [vx_b, vy_b, vz, yaw]

        cmd_pos = self._robot.data.root_pos_w.clone(); cmd_pos[:, 2] += 1.5
        cur_pos = self._robot.data.root_pos_w.clone(); cur_pos[:, 2] += 1.1

        # commands are body-frame: rotate into world (yaw only) for display
        w, x, y, z = self._robot.data.root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cmd_xy_w = torch.stack(
            [cos_y * cmds[:, 0] - sin_y * cmds[:, 1],
             sin_y * cmds[:, 0] + cos_y * cmds[:, 1]], dim=-1
        )
        cmd_quat = self._vel_world_xy_to_arrow(cmd_xy_w)
        cur_quat = self._vel_world_xy_to_arrow(self._robot.data.root_lin_vel_w[:, :2])

        cmd_mag = torch.clamp(torch.norm(cmd_xy_w, dim=-1), min=0.01)
        cur_mag = torch.clamp(torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=-1), min=0.01)

        cmd_translations, cmd_orientations, cmd_indices = self._arrow_parts(cmd_pos, cmd_quat, cmd_mag)
        cur_translations, cur_orientations, cur_indices = self._arrow_parts(cur_pos, cur_quat, cur_mag)

        cmd_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cmd_scale[:self.num_envs, 0] = cmd_mag  # scale shaft length
        cmd_scale[:, 1:] = 1.5  # Make command arrow 50% thicker

        cur_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cur_scale[:self.num_envs, 0] = cur_mag  # scale shaft length
        cur_scale[:, 1:] = 0.7  # Make current arrow thinner

        self._cmd_vel_marker.visualize(
            translations=cmd_translations, orientations=cmd_orientations, scales=cmd_scale, marker_indices=cmd_indices
        )
        self._cur_vel_marker.visualize(
            translations=cur_translations, orientations=cur_orientations, scales=cur_scale, marker_indices=cur_indices
        )

        # Yaw-rate arrow (yellow): tangential to heading, showing which way the
        # nose is commanded to swing. Anchored at the nose, above the vel arrows.
        yaw_rate = cmds[:, 3]
        yaw_vec_w = torch.stack([-sin_y * yaw_rate, cos_y * yaw_rate], dim=-1)
        yaw_quat = self._vel_world_xy_to_arrow(yaw_vec_w)
        yaw_mag = torch.clamp(yaw_rate.abs() * 2.0, min=0.05)
        nose = self._robot.data.root_pos_w.clone()
        nose[:, 0] += 0.3 * cos_y
        nose[:, 1] += 0.3 * sin_y
        nose[:, 2] += 1.9
        yaw_translations, yaw_orientations, yaw_indices = self._arrow_parts(nose, yaw_quat, yaw_mag)
        yaw_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        yaw_scale[:self.num_envs, 0] = yaw_mag
        yaw_scale[:, 1:] = 1.2
        self._yaw_cmd_marker.visualize(
            translations=yaw_translations, orientations=yaw_orientations, scales=yaw_scale, marker_indices=yaw_indices
        )

    def _vel_world_xy_to_arrow(self, vel_world_xy: torch.Tensor) -> torch.Tensor:
        """World-frame XY velocity → world-frame arrow quaternion (wxyz)."""
        angle = torch.atan2(vel_world_xy[:, 1], vel_world_xy[:, 0])
        ha = angle * 0.5
        return torch.stack([torch.cos(ha), torch.zeros_like(ha), torch.zeros_like(ha), torch.sin(ha)], dim=-1)

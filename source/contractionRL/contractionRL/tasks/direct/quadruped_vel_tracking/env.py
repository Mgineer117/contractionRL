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

from ..common.vel_commands import VelCommands
from .env_cfg import QuadrupedVelTrackingEnvCfg


class QuadrupedVelTrackingEnv(DirectRLEnv):
    """Unitree Go2 velocity-tracking environment.

    Commands: constant (vx, vy) + sinusoidal yaw rate.
    State for path-tracking export:
        proj_gravity_b(3) + joint_pos_rel(12) + joint_vel(12)  → 27 dims
    """

    cfg: QuadrupedVelTrackingEnvCfg

    STATE_DIM = 27   # dims recorded as reference state for path tracking

    def __init__(self, cfg: QuadrupedVelTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, _ = self._robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
        self._actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)
        self._episode_vel_auc = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # Standard Isaac Sim ground plane (grey grid texture)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        # Distant light gives directional sunlight with shadows — the standard
        # Isaac Sim look. DomeLightCfg was washing everything to flat white.
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
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
        """Returns (N, 27) physical state without commands or actions.

        Layout: [proj_gravity_b(3), joint_pos_rel(12), joint_vel(12)]
        Linear and angular body velocities are excluded so the state is
        purely proprioceptive and matches the path-tracking observation.
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
        # Full 33-dim state for the locomotion policy (needs velocities to track commands)
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

        # linear velocity tracking (xy) in world frame
        lin_err = torch.sum(
            torch.square(cmds[:, :2] - self._robot.data.root_lin_vel_w[:, :2]), dim=1
        )
        rew_lin = torch.exp(-lin_err / 0.1) * self.cfg.rew_lin_vel

        # heading bonus: align robot's yaw with the direction of the commanded velocity
        cmd_yaw = torch.atan2(cmds[:, 1], cmds[:, 0])
        # root_quat_w is (w, x, y, z)
        w, x, y, z = self._robot.data.root_quat_w[:, 0], self._robot.data.root_quat_w[:, 1], self._robot.data.root_quat_w[:, 2], self._robot.data.root_quat_w[:, 3]
        robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        heading_err = cmd_yaw - robot_yaw
        heading_err = torch.remainder(heading_err + torch.pi, 2 * torch.pi) - torch.pi
        rew_heading = torch.exp(-torch.square(heading_err) / 0.1) * self.cfg.rew_heading

        # yaw rate tracking
        yaw_err = torch.square(cmds[:, 3] - self._robot.data.root_ang_vel_b[:, 2])
        rew_yaw = torch.exp(-yaw_err / 0.1) * self.cfg.rew_yaw_rate

        vel_err_vec = cmds[:, :2] - self._robot.data.root_lin_vel_w[:, :2]
        self._episode_vel_auc += torch.norm(vel_err_vec, dim=-1)

        rew_z = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.rew_z_vel
        rew_rp = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.rew_roll_pitch
        rew_tau = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.rew_torque
        rew_act = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate
        total_reward = rew_lin + rew_heading + rew_yaw + rew_z + rew_rp + rew_tau + rew_act
        
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
                mask = auc_vals > 0
                self.extras["log"]["Episode/auc"] = auc_vals[mask].mean()
                self.extras["log"]["Episode/discounted_return"] = disc_returns[mask].mean()
                self.extras["log"]["Episode/undiscounted_return"] = undisc_returns[mask].mean()
                self.extras["log"]["Episode/avg_reward_per_step"] = (undisc_returns[mask] / lengths[mask]).mean()
            
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
    #
    # Arrows are built from two 3D primitives (cylinder shaft + cone head,
    # both axis="X") rather than the flat arrow_x.usd mesh — a flat mesh
    # looks like a thin line (direction ambiguous) from many camera angles,
    # while a cylinder+cone is a real 3D solid, recognizable from any angle.
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
                cmd_cfg = self._make_arrow_markers_cfg("/Visuals/QuadrupedVelCmd", (0.0, 0.0, 1.0))
                cur_cfg = self._make_arrow_markers_cfg("/Visuals/QuadrupedVelCur", (0.0, 1.0, 0.0))
                self._cmd_vel_marker = VisualizationMarkers(cmd_cfg)
                self._cur_vel_marker = VisualizationMarkers(cur_cfg)
            self._cmd_vel_marker.set_visibility(True)
            self._cur_vel_marker.set_visibility(True)
        else:
            if hasattr(self, "_cmd_vel_marker"):
                self._cmd_vel_marker.set_visibility(False)
                self._cur_vel_marker.set_visibility(False)

    def _arrow_parts(self, base_pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (translations, orientations, marker_indices) for a shaft+head arrow at base_pos, pointing along quat.

        Cylinder/cone primitives are centered at their own local origin, so
        each part is offset along local +X by half its own length plus
        whatever precedes it — this reproduces the old arrow_x.usd mesh's
        convention of "origin at the tail, extending outward" for the
        combined shaft+head, with the cone's base flush against the shaft's tip.
        """
        n = base_pos.shape[0]
        shaft_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN / 2, 0.0, 0.0], device=self.device
        ).expand(n, -1)
        head_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN + self._ARROW_HEAD_LEN / 2, 0.0, 0.0], device=self.device
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

        base_pos = self._robot.data.root_pos_w  # (N, 3)
        cmd_pos = self._robot.data.root_pos_w.clone()
        cmd_pos[:, 2] += 0.5  # Same height
        cur_pos = self._robot.data.root_pos_w.clone()
        cur_pos[:, 2] += 0.5  # Same height

        cmd_quat = self._vel_body_xy_to_arrow(cmds[:, :2])
        cur_quat = self._vel_body_xy_to_arrow(self._robot.data.root_lin_vel_b[:, :2])

        cmd_translations, cmd_orientations, cmd_indices = self._arrow_parts(cmd_pos, cmd_quat)
        cur_translations, cur_orientations, cur_indices = self._arrow_parts(cur_pos, cur_quat)

        cmd_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cmd_scale[:, 1:] = 1.5  # Make command arrow 50% thicker
        cur_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cur_scale[:, 1:] = 0.7  # Make current arrow thinner

        self._cmd_vel_marker.visualize(
            translations=cmd_translations, orientations=cmd_orientations, scales=cmd_scale, marker_indices=cmd_indices
        )
        self._cur_vel_marker.visualize(
            translations=cur_translations, orientations=cur_orientations, scales=cur_scale, marker_indices=cur_indices
        )

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

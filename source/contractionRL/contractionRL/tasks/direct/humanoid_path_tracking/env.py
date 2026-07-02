from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..common.path_tracking_base import PathTrackingBase
from .env_cfg import HumanoidPathTrackingEnvCfg


class HumanoidPathTrackingEnv(PathTrackingBase):
    """Unitree H1 path-tracking environment.

    obs  = [x(41), x_ref(41), u_ref(19)] = 101
    reward = -||x - x_ref||^2

    x layout: proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19)
    """

    cfg: HumanoidPathTrackingEnvCfg

    def __init__(self, cfg: HumanoidPathTrackingEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, _ = self._robot.find_joints([
            ".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso",
            ".*_ankle",
            ".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow",
        ])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
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

    def _get_physical_state(self) -> torch.Tensor:
        return torch.cat(
            [
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        return fell, time_out

    def _set_robot_state_from_ref(self, env_ids: torch.Tensor, x_ref_init: torch.Tensor) -> None:
        # x layout: [gravity(3), joint_pos_rel(3:22), joint_vel(22:41)]
        joint_pos_rel = x_ref_init[:, 3:22]   # (n, 19)
        joint_vel_ref = x_ref_init[:, 22:41]  # (n, 19)

        n = len(env_ids)
        full_pos = self._robot.data.default_joint_pos[env_ids].clone()
        full_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # Reference joint state + small random offset so the controller has something to correct
        noise = torch.empty(n, len(self._joint_ids), device=self.device).uniform_(
            -self.cfg.init_noise_scale, self.cfg.init_noise_scale
        )
        full_pos[:, self._joint_ids] = full_pos[:, self._joint_ids] + joint_pos_rel + noise
        full_vel[:, self._joint_ids] = joint_vel_ref

        # Root: default pose (upright) + env origin — velocity zeroed (not in state)
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root_vel = torch.zeros(n, 6, device=self.device)

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(full_pos, full_vel, None, env_ids)

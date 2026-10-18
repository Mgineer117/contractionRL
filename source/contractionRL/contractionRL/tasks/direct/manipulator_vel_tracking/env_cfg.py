from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ..common.vel_commands import VelCmdCfg


@configclass
class ManipulatorVelTrackingEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    decimation = 2
    episode_length_s = 15.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 36
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)

    arm_joint_names = ["panda_joint[1-7]"]
    ee_body_name = "panda_hand"

    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-0.3, 0.3),
        vy_range=(-0.3, 0.3),
        vz_range=(-0.2, 0.2),
        yaw_A_range=(0.1, 0.5),
        yaw_omega_range=(0.3, 1.5),
    )

    action_scale_shoulder = 0.20
    action_scale_forearm = 0.025

    rew_ee_vel = -1.0
    rew_ee_yaw = -0.5
    rew_action_rate = -0.01
    rew_joint_limits = -0.1

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..common.vel_commands import VelCmdCfg


@configclass
class ManipulatorVelTrackingEnvCfg(DirectRLEnvCfg):
    # env — EE Cartesian velocity tracking task
    decimation = 2
    episode_length_s = 5.0

    # Franka Panda: 7 arm joints (finger joints excluded from control)
    # state:  joint_pos(7) + joint_vel(7) + ee_pos_local(3)
    #         + ee_lin_vel(3) + ee_yaw_vel(1)  = 21
    # obs:    state(21) + commands(4) + prev_actions(7) = 32
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 32
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    arm_joint_names = ["panda_joint[1-7]"]
    ee_body_name = "panda_hand"

    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-0.3, 0.3),   # EE linear velocity [m/s]
        vy_range=(-0.3, 0.3),
        vz_range=(-0.2, 0.2),
        yaw_A_range=(0.1, 0.5),  # EE yaw rate [rad/s]
        yaw_omega_range=(0.3, 1.5),
    )

    # action maps [-1,1] to joint positions within soft limits
    # (handled inside the env via midpoint + range scaling)

    # reward scales (tentative)
    rew_ee_vel = -1.0       # -||ee_vel - cmd_vel||^2
    rew_ee_yaw = -0.5       # -||ee_yaw_vel - cmd_yaw||^2
    rew_action_rate = -0.01
    rew_joint_limits = -0.1  # penalty for approaching joint limits

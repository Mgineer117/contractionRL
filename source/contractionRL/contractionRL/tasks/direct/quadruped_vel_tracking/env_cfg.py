from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..common.vel_commands import VelCmdCfg


@configclass
class QuadrupedVelTrackingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0

    # Unitree Go2: 12 joints (3 per leg × 4 legs)
    # state:   base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3)
    #          + joint_pos_rel(12) + joint_vel(12)  = 33
    # obs:     state(33) + commands(4) + prev_actions(12) = 49
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 49
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)

    # velocity commands
    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-1.0, 1.5),
        vy_range=(-0.5, 0.5),
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.3, 1.0),
        yaw_omega_range=(0.5, 2.0),
    )

    # action: deviation from default joint positions [rad]
    action_scale = 0.25

    # termination
    base_height_min = 0.20  # [m]

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = False
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales (tentative)
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_z_vel = -0.5
    rew_ang_vel_xy = -0.05
    rew_torques = -1e-5
    rew_action_rate = -0.01
    rew_alive = 0.5

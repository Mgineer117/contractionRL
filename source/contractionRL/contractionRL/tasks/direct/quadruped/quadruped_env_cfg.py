from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # spaces: base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + cmd_vel(3)
    #         + joint_pos(12) + joint_vel(12) + actions(12) = 48
    # action: normalised joint position deviation in [-1, 1], scaled by action_scale [rad]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # action scale: deviation from default joint positions (rad)
    action_scale = 0.25

    # termination
    base_height_min = 0.2  # [m] terminate if base falls below this

    # reward scales (tentative)
    rew_lin_vel_scale = 2.0
    rew_yaw_rate_scale = 0.5
    rew_z_vel_scale = -0.5
    rew_ang_vel_xy_scale = -0.05
    rew_torques_scale = -1e-5
    rew_action_rate_scale = -0.01
    rew_alive_scale = 0.5

    # velocity command range [m/s] and [rad/s]
    cmd_lin_vel_range = [-1.0, 1.0]
    cmd_yaw_range = [-1.0, 1.0]

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import H1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # H1 joints: 2 hip_yaw + 2 hip_roll + 2 hip_pitch + 2 knee + 1 torso  (legs=9)
    #            + 2 ankle (feet=2)
    #            + 2 shoulder_pitch + 2 shoulder_roll + 2 shoulder_yaw + 2 elbow (arms=8)
    #            = 19 total
    # obs: base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + cmd_vel(3)
    #      + joint_pos(19) + joint_vel(19) + actions(19) = 69
    # action: normalised joint position deviation in [-1, 1], scaled by action_scale [rad]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 69
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # action scale (rad) — deviation from default joint positions
    action_scale = 0.25

    # termination: terminate if base falls below this height
    base_height_min = 0.5  # [m]

    # reward scales (tentative)
    rew_lin_vel_scale = 2.0
    rew_yaw_rate_scale = 0.5
    rew_z_vel_scale = -0.5
    rew_ang_vel_xy_scale = -0.05
    rew_upright_scale = -1.0    # penalise tilt via projected gravity xy
    rew_torques_scale = -1e-5
    rew_action_rate_scale = -0.01
    rew_alive_scale = 1.0

    # velocity command range
    cmd_lin_vel_range = [-1.0, 1.0]
    cmd_yaw_range = [-1.0, 1.0]

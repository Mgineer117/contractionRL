from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ManipulatorEnvCfg(DirectRLEnvCfg):
    # env — reach task: move EE to a randomly sampled target position
    decimation = 2
    episode_length_s = 5.0
    # obs: joint_pos(7) + joint_vel(7) + ee_pos(3) + target_pos(3) = 20
    # action: normalised joint position targets in [-1, 1] (7 arm joints)
    #         mapped to soft joint limits inside _pre_physics_step
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 20
    state_space = 0

    # simulation (gravity disabled via HIGH_PD cfg)
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot — high PD gains, gravity disabled for stable position control
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # arm joint names (excludes finger joints)
    arm_joint_names = ["panda_joint[1-7]"]

    # end-effector body name
    ee_body_name = "panda_hand"

    # target sampling range (relative to robot base, in metres)
    target_pos_range_x = [0.3, 0.7]
    target_pos_range_y = [-0.3, 0.3]
    target_pos_range_z = [0.2, 0.6]

    # reward scales (tentative)
    rew_reach_scale = -1.0        # -dist(ee, target)
    rew_success_scale = 5.0       # bonus if dist < success_threshold
    rew_action_rate_scale = -0.01
    success_threshold = 0.05      # [m]

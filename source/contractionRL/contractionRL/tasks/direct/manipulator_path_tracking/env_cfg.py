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
class ManipulatorPathTrackingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0

    # state_dim = 21, action_dim = 7
    # obs: x(21) + x_ref(21) + u_ref(7) = 49
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 49
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    arm_joint_names = ["panda_joint[1-7]"]
    ee_body_name = "panda_hand"

    traj_path: str = "logs/ref_trajs/manipulator.npz"

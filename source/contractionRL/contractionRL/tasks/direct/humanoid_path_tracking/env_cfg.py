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
class HumanoidPathTrackingEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0

    # state_dim = 47, action_dim = 19
    # obs: x(47) + x_ref(47) + u_ref(19) = 113
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 113
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    action_scale = 0.25
    traj_path: str = "logs/ref_trajs/humanoid.npz"
    base_height_min = 0.50

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class QuadrupedPathTrackingEnvCfg(DirectRLEnvCfg):

    num_envs = 4096
    decimation = 4
    episode_length_s = 40.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 126
    state_space = 0
    disturbance_force_bound: float = 2.0
    disturbance_torque_bound: float = 0.1

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    action_scale = 0.25

    traj_path: str = "data/quadruped/dynamics_data.npz"

    terminate_on_fall: bool = True
    base_height_min = 0.20

    init_noise_scale: float = 0.05
    init_xy_noise_scale: float = 0.1
    init_yaw_noise_scale: float = 0.1

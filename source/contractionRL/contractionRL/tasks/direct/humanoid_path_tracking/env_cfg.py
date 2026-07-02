from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import H1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass


@configclass
class HumanoidPathTrackingEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    decimation = 4
    episode_length_s = 10.0

    # state_dim = 41: proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19)
    # obs: x(41) + x_ref(41) + u_ref(19) = 101
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 101
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    action_scale = 0.25
    traj_path: str = "logs/humanoid/ref_trajs.npz"
    base_height_min = 0.50

    # uniform joint-position noise added to x_ref[0] at reset [rad]
    init_noise_scale: float = 0.05

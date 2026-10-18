import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):

    num_envs = 4096
    decimation = 2
    episode_length_s = 5.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))

    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    action_scale = 100.0
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    initial_pole_angle_range = [-0.25, 0.25]
    max_cart_pos = 3.0

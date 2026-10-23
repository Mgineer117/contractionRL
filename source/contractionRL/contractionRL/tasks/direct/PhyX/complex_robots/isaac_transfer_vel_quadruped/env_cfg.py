from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from contractionRL.tasks.direct.common.vel_commands import VelCmdCfg


@configclass
class QuadrupedVelTrackingEnvCfg(DirectRLEnvCfg):

    num_envs = 4096
    decimation = 4
    episode_length_s = 40.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 55
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(-5.0, 8.66, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1920, 1080),
    )

    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(0.5, 1.5),
        vy_range=(0.0, 0.0),
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.0, 0.5),
        yaw_omega_range=(0.0, 2 * math.pi / episode_length_s),
        yaw_omega_binary=True,
        yaw_phase_range=(0.0, 0.0),
    )

    action_scale = 0.25

    terminate_on_fall: bool = True
    target_base_height = 0.34
    base_height_min = 0.20
    fall_grav_z_max = -0.71

    randomize_init: bool = True
    init_pos_range: float = 0.3
    init_joint_noise: float = 0.05

    rew_alive = 0.1
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_flat_orientation = -0.5
    rew_z_vel = -0.5
    rew_roll_pitch = -0.05
    rew_action_rate = -0.01
    rew_hip = -0.3
    rew_base_height = -10.0

    gait_freq = 1.0
    gait_duty = 0.8
    rew_gait = 1.0
    gait_transition_band = 0.2

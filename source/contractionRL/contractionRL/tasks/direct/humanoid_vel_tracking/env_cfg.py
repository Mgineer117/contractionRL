from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import H1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from ..common.vel_commands import VelCmdCfg


@configclass
class HumanoidVelTrackingEnvCfg(DirectRLEnvCfg):
    # env

    num_envs = 4096
    decimation = 4
    episode_length_s = 10.0

    # Unitree H1: 19 joints
    #   legs:  2 hip_yaw + 2 hip_roll + 2 hip_pitch + 2 knee + 1 torso = 9
    #   feet:  2 ankle = 2
    #   arms:  2 shoulder_pitch + 2 shoulder_roll + 2 shoulder_yaw + 2 elbow = 8
    # state (path-tracking export): proj_gravity(3) + joint_pos_rel(19) + joint_vel(19) = 41
    # obs:     full_state(47) + commands(4) + prev_actions(19) = 70
    #   (full_state still includes lin_vel_b+ang_vel_b for the locomotion policy to use)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 70
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    # Position-delta control needs the actuators' real PD gains (H1_CFG's own
    # per-joint stiffness/damping in the "legs" [incl. torso], "feet", "arms"
    # groups) — zero gains produce zero torque regardless of position target.
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-1.0, 1.5),
        vy_range=(-0.5, 0.5),
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.2, 0.8),
        yaw_omega_range=(0.4, 1.5),
    )

    # action: deviation from default joint positions [rad] — matches quadruped_vel_tracking's convention
    action_scale = 0.25

    base_height_min = 0.50  # [m]

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = False
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales (tentative)
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_z_vel = -0.5
    rew_ang_vel_xy = -0.05
    rew_upright = -1.0       # penalise tilt via projected_gravity_b[:, :2]
    rew_torques = -1e-5
    rew_action_rate = -0.01
    rew_alive = 1.0

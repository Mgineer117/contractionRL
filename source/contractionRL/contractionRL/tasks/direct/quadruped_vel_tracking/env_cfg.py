from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from ..common.vel_commands import VelCmdCfg


@configclass
class QuadrupedVelTrackingEnvCfg(DirectRLEnvCfg):
    # env

    num_envs = 4096
    decimation = 4
    episode_length_s = 10.0

    # Unitree Go2: 12 joints (3 per leg × 4 legs)
    # state:   base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3)
    #          + joint_pos_rel(12) + joint_vel(12)  = 33
    # obs:     state(33) + commands(4) + prev_actions(12) = 49
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 49
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    # Position-delta control needs the actuator's real PD gains (stiffness=25.0,
    # damping=0.5) — zero gains produce zero torque regardless of position target.
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    # video/viewport camera: the default ViewerCfg is a *static, world-fixed*
    # camera far from any particular robot (eye=(7.5,7.5,7.5), lookat=(0,0,0)),
    # so as the robot walks away during the 10s episode it shrinks into the
    # distance and looks blurry/low-res in recorded videos even though the
    # capture resolution itself is fine. Track env 0's robot instead, with a
    # closer chase-cam framing and a higher capture resolution.
    #
    # eye/lookat give a ~38 degree downward angle (vs. the previous ~18 degrees),
    # since a shallow angle put most of the frame above the horizon — showing the
    # DomeLight's untextured "sky" fill (a flat, uniform light-gray, since it has
    # no texture_file) instead of the actual (dark) ground plane below the horizon.
    # No ground-color or lighting change can fix that if the ground isn't in frame.
    viewer: ViewerCfg = ViewerCfg(
        eye=(-5.0, 8.66, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1920, 1080),
    )

    # velocity commands
    # yaw_omega_binary: each episode gets EXACTLY 0.0 (constant yaw rate,
    # sin(phi) with omega=0 → no oscillation, just a random-but-fixed turn
    # rate) or EXACTLY 2*pi/episode_length_s (one full, gentle S-turn sine
    # cycle over the whole episode) — never an in-between frequency.
    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-0.2, 0.2),
        vy_range=(-0.2, 0.2),
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.0, 0.0),
        # yaw_A_range=(0.1, 0.5),
        yaw_omega_range=(0.0, 2 * math.pi / episode_length_s),
        yaw_omega_binary=True,
    )

    # action: deviation from default joint positions [rad]
    action_scale = 0.25

    # termination
    base_height_min = 0.20     # [m] terminate if base drops below this
    fall_grav_z_max = -0.5     # terminate if projected_gravity_b z rises above this (~>60 deg tilt)

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = True
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales — legged_gym-style recipe: body-frame exp-tracking terms
    # (sigma 0.25) dominate; flat-orientation replaces fall termination as the
    # thing that makes "fallen" strictly worse than any upright behavior.
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_flat_orientation = -2.5  # on sum(projected_gravity_b[:, :2]**2)
    rew_z_vel = -0.5
    rew_roll_pitch = -0.05
    rew_torque = -1e-5
    rew_action_rate = -0.01

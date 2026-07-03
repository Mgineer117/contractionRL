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
    episode_length_s = 20.0

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

    # velocity commands: forward speed along the *current heading* + a sinusoidal
    # yaw rate. The command distribution has exactly two sampled dimensions —
    # forward speed (vx) and yaw-rate amplitude (yaw_A). vy/vz are pinned to 0 so
    # the linear command is purely along heading, and the yaw frequency is fixed
    # to EXACTLY one full sine cycle per episode (omega = 2*pi/T), phase 0. Over
    # one cycle the yaw integrates to zero, so the robot weaves left-then-right
    # (an S) and returns to its initial heading.
    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(0.0, 0.5),        # forward speed [m/s] — sampled
        vy_range=(0.0, 0.0),        # no lateral component: velocity is along heading
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.0, 0.3),     # yaw-rate amplitude [rad/s] — sampled
        yaw_omega_range=(2 * math.pi / episode_length_s, 2 * math.pi / episode_length_s),  # fixed: one cycle/episode
        yaw_omega_binary=False,
        yaw_phase_range=(0.0, 0.0),  # start each episode at zero yaw rate
    )

    # action: deviation from default joint positions [rad]
    action_scale = 0.25

    # termination
    # terminate_on_fall: training uses fall termination; the post-training
    # evaluator flips this to False at runtime so episodes always run the full
    # length (metrics comparable across policies regardless of fall behavior).
    terminate_on_fall: bool = True
    base_height_min = 0.20     # [m] terminate if base drops below this
    # -0.71 ≈ -cos(45°): terminate beyond ~45° tilt. The previous -0.5 (~60°)
    # left a loophole — a robot crouched at 40-55° never terminated and sat
    # accumulating negative reward for the whole episode (observed as episode
    # returns of ~-433), fattening the advantage tails that spike PPO's KL.
    fall_grav_z_max = -0.71

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = True
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales — legged_gym-style recipe: body-frame exp-tracking terms
    # (sigma 0.25) dominate. Falls are punished primarily by termination (lost
    # future reward); the alive bonus keeps per-step reward positive in any
    # reasonable alive state so terminating early is never advantageous, and
    # the small flat-orientation term is smooth shaping toward upright — kept
    # mild because a large one creates reward cliffs near falls, whose
    # heavy-tailed advantages spike PPO's KL and crash the adaptive LR.
    rew_alive = 0.5
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_flat_orientation = -0.5  # on sum(projected_gravity_b[:, :2]**2)
    rew_z_vel = -0.5
    rew_roll_pitch = -0.05
    rew_torque = -1e-5
    rew_action_rate = -0.01

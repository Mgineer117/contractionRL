"""Analytical defaults; backend selection configures the PhysX control interval."""

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..classic.car.env import ENV_CONFIG as CAR
from ..classic.quadrotor.env import ENV_CONFIG as QUADROTOR
from ..classic.cartpole.env import ENV_CONFIG as CARTPOLE
from ..classic.segway.env import ENV_CONFIG as SEGWAY


@configclass
class CarEnvCfg(DirectRLEnvCfg):
    classic_task: str = "classic-car-v0"
    mode: str = "analytical"
    physx_reference_path: str = ""
    reference_mode: str = "stabilizing"
    disturbance_enabled: bool = True
    disturbance_scales: list[float] = [0., 0., .02, .05]
    reference_margin_fraction: float = .05
    decimation = 1
    episode_length_s = CAR["time_bound"]
    action_space = CAR["num_dim_control"]
    observation_space = 1
    state_space = 0
    sim = SimulationCfg(dt=CAR["dt"], render_interval=1)
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=100.0, replicate_physics=False)
    viewer = ViewerCfg(eye=(30.0, 30.0, 30.0), lookat=(0.0, 0.0, 0.0))


@configclass
class CarV1EnvCfg(CarEnvCfg):
    classic_task: str = "classic-car-v1"


@configclass
class QuadrotorEnvCfg(CarEnvCfg):
    classic_task: str = "classic-quadrotor-v0"
    disturbance_scales: list[float] = [0., 0., 0., .2, .2, .2, 0., 0., 0., 0.]
    episode_length_s = QUADROTOR["time_bound"]
    action_space = QUADROTOR["num_dim_control"]
    sim = SimulationCfg(dt=QUADROTOR["dt"], render_interval=1)


@configclass
class CartpoleEnvCfg(CarEnvCfg):
    classic_task: str = "classic-cartpole-v0"
    disturbance_scales: list[float] = [0., 0., .05, .05]
    episode_length_s = CARTPOLE["time_bound"]
    action_space = 1
    sim = SimulationCfg(dt=CARTPOLE["dt"], render_interval=1)
    viewer = ViewerCfg(eye=(8., 8., 5.), lookat=(0., 0., 1.))


@configclass
class SegwayEnvCfg(CartpoleEnvCfg):
    classic_task: str = "classic-segway-v0"
    episode_length_s = SEGWAY["time_bound"]

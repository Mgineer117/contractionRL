"""Timing and dimensions come directly from the classic plant configurations."""

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..classic.car.env import ENV_CONFIG as CAR
from ..classic.quadrotor.env import ENV_CONFIG as QUADROTOR


@configclass
class CarEnvCfg(DirectRLEnvCfg):
    classic_task: str = "classic-car-v0"
    reference_mode: str = "stabilizing"
    decimation = 1
    episode_length_s = CAR["time_bound"]
    action_space = CAR["num_dim_control"]
    observation_space = 1  # replaced with the classic Dict space before initialization
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
    episode_length_s = QUADROTOR["time_bound"]
    action_space = QUADROTOR["num_dim_control"]
    sim = SimulationCfg(dt=QUADROTOR["dt"], render_interval=1)

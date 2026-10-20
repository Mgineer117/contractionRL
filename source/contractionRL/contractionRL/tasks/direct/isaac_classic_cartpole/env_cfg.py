"""Isaac-Classic-Cartpole-v0 configuration."""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
from ..common.isaac.env_cfg import TrackingEnvCfg
from .dynamics import ENV_CONFIG as CARTPOLE


@configclass
class CartpoleEnvCfg(TrackingEnvCfg):
    classic_task: str = "classic-cartpole-v0"
    disturbance_scales: list[float] = [0., 0., .05, .05]
    episode_length_s = CARTPOLE["time_bound"]
    action_space = 1
    sim = SimulationCfg(dt=CARTPOLE["dt"], render_interval=1)
    viewer = ViewerCfg(eye=(8., 8., 5.), lookat=(0., 0., 1.))

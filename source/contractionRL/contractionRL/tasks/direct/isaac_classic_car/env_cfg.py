"""Isaac-Classic-Car-v0 configuration."""

from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from ..common.isaac.env_cfg import TrackingEnvCfg
from .dynamics import ENV_CONFIG as CAR


@configclass
class CarEnvCfg(TrackingEnvCfg):
    classic_task: str = "classic-car-v0"
    disturbance_scales: list[float] = [0., 0., .02, .05]
    episode_length_s = CAR["time_bound"]
    action_space = CAR["num_dim_control"]
    sim = SimulationCfg(dt=CAR["dt"], render_interval=1)

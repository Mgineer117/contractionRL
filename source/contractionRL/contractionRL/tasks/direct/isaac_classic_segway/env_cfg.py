"""Isaac-Classic-Segway-v0 configuration."""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
from ..common.isaac.env_cfg import TrackingEnvCfg
from .dynamics import ENV_CONFIG as SEGWAY


@configclass
class SegwayEnvCfg(TrackingEnvCfg):
    classic_task: str = "classic-segway-v0"
    episode_length_s = SEGWAY["time_bound"]
    # Same nominal feedback in both backends; suppress upright numerical drift.
    reference_feedback_gain: list[float] = [-1., -19.28069421348111, -13.785420462054004, -6.748228111445539]
    disturbance_scales: list[float] = [0., 0., .05, .05]
    action_space = 1
    sim = SimulationCfg(dt=SEGWAY["dt"], render_interval=1)
    viewer = ViewerCfg(eye=(8., 8., 5.), lookat=(0., 0., 1.))

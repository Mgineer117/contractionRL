"""Isaac-Classic-Quadrotor-v0 configuration."""

from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from contractionRL.tasks.direct.common.isaac.env_cfg import TrackingEnvCfg
from .dynamics import ENV_CONFIG as QUADROTOR


@configclass
class QuadrotorEnvCfg(TrackingEnvCfg):
    classic_task: str = "classic-quadrotor-v0"
    disturbance_scales: list[float] = [0., 0., 0., .2, .2, .2, 0., 0., 0., 0.]
    xe_init_min: list[float] = [.5, .5, .5] + QUADROTOR["xe_init_min"][3:]
    xe_init_max: list[float] = [1.5, 1.5, 1.] + QUADROTOR["xe_init_max"][3:]
    xe_init_sign_dims: list[int] = [0, 1, 2]
    episode_length_s = QUADROTOR["time_bound"]
    action_space = QUADROTOR["num_dim_control"]
    sim = SimulationCfg(dt=QUADROTOR["dt"], render_interval=1)

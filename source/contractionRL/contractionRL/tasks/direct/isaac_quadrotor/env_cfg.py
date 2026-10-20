"""Isaac-Quadrotor-v0 configuration."""

from isaaclab.utils import configclass
from ..isaac_classic_quadrotor.env_cfg import QuadrotorEnvCfg


@configclass
class QuadrotorPhysXEnvCfg(QuadrotorEnvCfg):
    pass

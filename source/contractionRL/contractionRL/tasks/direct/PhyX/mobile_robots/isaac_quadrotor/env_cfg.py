"""Isaac-Quadrotor-v0 configuration."""

from isaaclab.utils import configclass
from contractionRL.tasks.direct.Classic.isaac_classic_quadrotor.env_cfg import QuadrotorEnvCfg


@configclass
class QuadrotorPhysXEnvCfg(QuadrotorEnvCfg):
    pass

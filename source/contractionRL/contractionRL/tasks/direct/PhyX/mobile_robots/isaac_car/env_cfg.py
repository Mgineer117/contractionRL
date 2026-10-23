"""Isaac-Car-v0 configuration."""

from isaaclab.utils import configclass
from contractionRL.tasks.direct.Classic.isaac_classic_car.env_cfg import CarEnvCfg


@configclass
class CarPhysXEnvCfg(CarEnvCfg):
    pass

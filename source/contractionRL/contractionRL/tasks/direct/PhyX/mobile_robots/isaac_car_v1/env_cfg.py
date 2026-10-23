"""Isaac-Car-v1 configuration."""

from isaaclab.utils import configclass
from contractionRL.tasks.direct.Classic.isaac_classic_car_v1.env_cfg import CarV1EnvCfg


@configclass
class CarV1PhysXEnvCfg(CarV1EnvCfg):
    pass

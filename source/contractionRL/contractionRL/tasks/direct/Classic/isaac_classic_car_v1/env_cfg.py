"""Isaac-Classic-Car-v1 configuration."""

from isaaclab.utils import configclass
from contractionRL.tasks.direct.Classic.isaac_classic_car.env_cfg import CarEnvCfg


@configclass
class CarV1EnvCfg(CarEnvCfg):
    classic_task: str = "classic-car-v1"

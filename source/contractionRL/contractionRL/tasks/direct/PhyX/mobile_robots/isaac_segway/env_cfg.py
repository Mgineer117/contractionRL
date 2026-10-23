"""Isaac-Segway-v0 configuration."""

from isaaclab.utils import configclass
from contractionRL.tasks.direct.Classic.isaac_classic_segway.env_cfg import SegwayEnvCfg


@configclass
class SegwayPhysXEnvCfg(SegwayEnvCfg):
    pass

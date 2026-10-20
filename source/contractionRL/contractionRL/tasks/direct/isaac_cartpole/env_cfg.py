"""Isaac-Cartpole-v0 configuration."""

from isaaclab.utils import configclass
from ..isaac_classic_cartpole.env_cfg import CartpoleEnvCfg


@configclass
class CartpolePhysXEnvCfg(CartpoleEnvCfg):
    pass

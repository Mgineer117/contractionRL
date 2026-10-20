"""Isaac-Car-v0 environment."""

from ..common.isaac.physx_env import PhysXTrackingEnv
from ..common.isaac.backend import configure_backend


class IsaacCarEnv(PhysXTrackingEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        configure_backend(cfg, task="Isaac-Car-v0")
        super().__init__(cfg, render_mode, **kwargs)

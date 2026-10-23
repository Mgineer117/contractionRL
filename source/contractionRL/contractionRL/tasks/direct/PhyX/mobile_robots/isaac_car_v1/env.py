"""Isaac-Car-v1 environment."""

from contractionRL.tasks.direct.common.isaac.physx_env import PhysXTrackingEnv
from contractionRL.tasks.direct.common.isaac.backend import configure_backend


class IsaacCarV1Env(PhysXTrackingEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        configure_backend(cfg, task="Isaac-Car-v1")
        super().__init__(cfg, render_mode, **kwargs)

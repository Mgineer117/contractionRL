"""Isaac-Cartpole-v0 environment."""

from contractionRL.tasks.direct.common.isaac.planar_env import PlanarTrackingEnv
from contractionRL.tasks.direct.common.isaac.backend import configure_backend


class IsaacCartpoleEnv(PlanarTrackingEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        configure_backend(cfg, task="Isaac-Cartpole-v0")
        super().__init__(cfg, render_mode, **kwargs)

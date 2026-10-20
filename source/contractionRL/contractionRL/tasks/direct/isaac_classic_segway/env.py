"""Isaac-Classic-Segway-v0 environment."""

from ..common.isaac.env import ClassicIsaacEnv
from ..common.isaac.backend import configure_backend


class IsaacClassicSegwayEnv(ClassicIsaacEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        configure_backend(cfg, task="Isaac-Classic-Segway-v0")
        super().__init__(cfg, render_mode, **kwargs)

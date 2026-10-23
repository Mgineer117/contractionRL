"""Isaac-Classic-Car-v1 environment."""

from contractionRL.tasks.direct.common.isaac.env import ClassicIsaacEnv
from contractionRL.tasks.direct.common.isaac.backend import configure_backend


class IsaacClassicCarV1Env(ClassicIsaacEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        configure_backend(cfg, task="Isaac-Classic-Car-v1")
        super().__init__(cfg, render_mode, **kwargs)

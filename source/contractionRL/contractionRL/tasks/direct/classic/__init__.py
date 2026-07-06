"""Classic (non-Isaac) analytical tracking environments.

Each subpackage registers a gym environment so it is discoverable by
``scripts/list_envs.py`` and runnable with ``scripts/skrl/train.py --classic``.

Importing this package registers every classic env. This works both as
``contractionRL.tasks.direct.classic`` (under Isaac) and as a standalone
``classic`` package on sys.path (no Isaac) — the registration
f-strings adapt to whichever package name they are imported under.
"""

from . import car  # noqa: F401  registers classic-car-v0
from . import cartpole  # noqa: F401 registers classic-cartpole-v0
from . import turtlebot  # noqa: F401 registers classic-turtlebot-v0
from . import segway  # noqa: F401 registers classic-segway-v0
from . import quadrotor  # noqa: F401 registers classic-quadrotor-v0

__all__ = ["car", "cartpole", "turtlebot", "segway", "quadrotor"]

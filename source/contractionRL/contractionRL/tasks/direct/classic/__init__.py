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
from . import ball_and_beam  # noqa: F401  registers classic-ball_and_beam-v0
from . import two_link_arm  # noqa: F401  registers classic-two_link_arm-v0
from . import aircraft  # noqa: F401  registers classic-aircraft-v0
from . import auv  # noqa: F401  registers classic-auv-v0
from . import pvtol  # noqa: F401  registers classic-pvtol-v0
from . import tora  # noqa: F401  registers classic-tora-v0

__all__ = ["car", "cartpole", "turtlebot", "segway", "quadrotor", "ball_and_beam", "two_link_arm", "aircraft", "auv", "pvtol", "tora"]

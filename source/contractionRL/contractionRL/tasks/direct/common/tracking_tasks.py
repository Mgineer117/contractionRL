"""Register paired tracking tasks without importing the Isaac runtime."""

import warnings

warnings.filterwarnings("ignore", message=r".*classic-car-v0 is out of date.*", category=DeprecationWarning)

from .. import isaac_classic_car, isaac_car  # noqa: E402,F401
from .. import isaac_classic_car_v1, isaac_car_v1  # noqa: E402,F401
from .. import isaac_classic_quadrotor, isaac_quadrotor  # noqa: E402,F401
from .. import isaac_classic_cartpole, isaac_cartpole  # noqa: E402,F401
from .. import isaac_classic_segway, isaac_segway  # noqa: E402,F401

ENV_IDS = {"car_v1": "classic-car-v1"}


def env_id(name: str) -> str:
    """Resolve a plant short name to its CPU-only analytical task ID."""
    return ENV_IDS.get(name, f"classic-{name}-v0")

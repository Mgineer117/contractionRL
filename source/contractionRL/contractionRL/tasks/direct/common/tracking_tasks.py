"""Register Isaac tracking tasks and construct their internal analytical plants."""

from importlib import import_module

from ..Classic import isaac_classic_car, isaac_classic_car_v1, isaac_classic_quadrotor, isaac_classic_cartpole, isaac_classic_segway  # noqa: F401
from ..PhyX.mobile_robots import isaac_car, isaac_car_v1, isaac_quadrotor, isaac_cartpole, isaac_segway  # noqa: F401

_PLANTS = {"car": "CarEnv", "car_v1": "CarV1Env", "cartpole": "CartPoleEnv", "segway": "SegwayEnv", "quadrotor": "QuadrotorEnv"}


def make_analytical_plant(name, **kwargs):
    """Construct a plant directly; standalone CPU Gym environments are not registered."""
    name = name.removeprefix("Isaac-Classic-").removeprefix("classic-")
    name = "car_v1" if name.lower() == "car-v1" else name.removesuffix("-v0").lower()
    if name not in _PLANTS:
        raise ValueError(f"Unknown analytical plant: {name}")
    module = import_module(f"contractionRL.tasks.direct.Classic.isaac_classic_{name}.dynamics")
    return getattr(module, _PLANTS[name])(**kwargs)

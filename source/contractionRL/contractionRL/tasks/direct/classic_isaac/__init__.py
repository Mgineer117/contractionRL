"""Isaac visualization of the exact classic analytical plants."""

import gymnasium as gym

from .. import classic  # noqa: F401 — registers the source tasks


def _register(task, classic_task, cfg_class):
    # Reuse the registrations, including car-v1's shared/car-specific configs.
    kwargs = dict(gym.spec(classic_task).kwargs)
    kwargs["env_cfg_entry_point"] = f"{__name__}.env_cfg:{cfg_class}"
    gym.register(
        id=task,
        entry_point=f"{__name__}.env:ClassicIsaacEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_register("Isaac-Car-v0", "classic-car-v0", "CarEnvCfg")
_register("Isaac-Car-v1", "classic-car-v1", "CarV1EnvCfg")
_register("Isaac-Quadrotor-v0", "classic-quadrotor-v0", "QuadrotorEnvCfg")

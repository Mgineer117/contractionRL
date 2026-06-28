"""Classic Car tracking environment — trained with mjrl (LQR / C3M).

Non-Isaac, analytical control-affine env. Registered with gym so it appears in
``scripts/list_envs.py`` alongside the Isaac Sim tasks.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Car-Direct-v0",
    entry_point=f"{__name__}.env:CarEnv",
    disable_env_checker=True,
    kwargs={
        "mjrl_lqr_cfg_entry_point": f"{agents.__name__}:mjrl_lqr_cfg.yaml",
        "mjrl_c3m_cfg_entry_point": f"{agents.__name__}:mjrl_c3m_cfg.yaml",
    },
)

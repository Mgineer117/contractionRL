"""Duffing toy environment."""

import gymnasium as gym

from . import agents

gym.register(
    id="toy-duff-v0",
    entry_point=f"{__name__}.env:DuffingEnv",
    disable_env_checker=True,
    kwargs={
        # ONE reference, 64 distinct initial conditions -- one global-optimum
        # task per env slot. train.py uses this when --num_envs is not given, so
        # the run matches the task set scripts/precompute_global.py solved.
        "default_num_envs": 64,
        "skrl_cfg_entry_point":       f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point":   f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_c3m_cfg_entry_point":   f"{agents.__name__}:skrl_c3m_cfg.yaml",
        "skrl_lqr_cfg_entry_point":   f"{agents.__name__}:skrl_lqr_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{agents.__name__}:skrl_sdlqr_cfg.yaml",
        "skrl_cvstem_lqr_cfg_entry_point": f"{agents.__name__}:skrl_cvstem_lqr_cfg.yaml",
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_ppo_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_sac_cfg.yaml",
        # Toy-only solver. No classic or Isaac env registers this key, so
        # pointing them at one fails on a missing entry point rather than
        # silently doing something meaningless.
    },
)

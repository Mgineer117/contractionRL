"""Isaac-Classic-Quadrotor-v0 tracking task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Classic-Quadrotor-v0",
    entry_point=f"{__name__}.env:IsaacClassicQuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadrotorEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_cfg",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:ppo_cfg",
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:c2rl_ppo_cfg",
        "skrl_contraction_ppo_cfg_entry_point": f"{agents.__name__}:contraction_ppo_cfg",
        "skrl_mbpo_cfg_entry_point": "contractionRL.tasks.direct.common:skrl_mbpo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_c3m_cfg_entry_point": f"{agents.__name__}:skrl_c3m_cfg.yaml",
        "skrl_lqr_cfg_entry_point": f"{agents.__name__}:skrl_lqr_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{agents.__name__}:skrl_sdlqr_cfg.yaml",
        "skrl_cvstem_lqr_cfg_entry_point": f"{agents.__name__}:skrl_cvstem_lqr_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_sac_cfg.yaml",
    },
)

# CPU-only access to the same analytical plant, used by metric tools and tests.
gym.register(
    id="classic-quadrotor-v0",
    entry_point=f"{__name__}.dynamics:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_cfg",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:ppo_cfg",
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:c2rl_ppo_cfg",
        "skrl_contraction_ppo_cfg_entry_point": f"{agents.__name__}:contraction_ppo_cfg",
        "skrl_mbpo_cfg_entry_point": "contractionRL.tasks.direct.common:skrl_mbpo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_c3m_cfg_entry_point": f"{agents.__name__}:skrl_c3m_cfg.yaml",
        "skrl_lqr_cfg_entry_point": f"{agents.__name__}:skrl_lqr_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{agents.__name__}:skrl_sdlqr_cfg.yaml",
        "skrl_cvstem_lqr_cfg_entry_point": f"{agents.__name__}:skrl_cvstem_lqr_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_sac_cfg.yaml",
    },
)

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Transfer-path-Quadruped-v0",
    entry_point=f"{__name__}.env:QuadrupedPathTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.env_cfg:QuadrupedPathTrackingEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:ppo_cfg",
        "skrl_contraction_ppo_cfg_entry_point": f"{agents.__name__}:contraction_ppo_cfg",
        "skrl_mbpo_cfg_entry_point": f"{agents.__name__}:mbpo_cfg",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:ppo_cfg",
        "skrl_sac_cfg_entry_point":    f"{agents.__name__}:sac_cfg",
        "skrl_c3m_cfg_entry_point":   f"{agents.__name__}:c3m_cfg",
        "skrl_sdlqr_cfg_entry_point": f"{agents.__name__}:sdlqr_cfg",
        "skrl_lqr_cfg_entry_point":   f"{agents.__name__}:lqr_cfg",
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:c2rl_ppo_cfg",
        "skrl_c2rl_sac_cfg_entry_point": f"{agents.__name__}:c2rl_sac_cfg",
    },
)

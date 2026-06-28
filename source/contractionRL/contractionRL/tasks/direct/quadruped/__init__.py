import gymnasium as gym

from . import agents

gym.register(
    id="Quadruped-Direct-v0",
    entry_point=f"{__name__}.quadruped_env:QuadrupedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:QuadrupedEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

import gymnasium as gym

from . import agents

gym.register(
    id="Manipulator-Direct-v0",
    entry_point=f"{__name__}.manipulator_env:ManipulatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulator_env_cfg:ManipulatorEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

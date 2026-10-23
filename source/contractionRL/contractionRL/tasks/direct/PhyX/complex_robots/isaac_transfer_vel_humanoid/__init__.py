import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Transfer-vel-Humanoid-v0",
    entry_point=f"{__name__}.env:HumanoidVelTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HumanoidVelTrackingEnvCfg",
        "skrl_mbpo_cfg_entry_point": "contractionRL.tasks.direct.common:skrl_mbpo_cfg.yaml",
        "skrl_cfg_entry_point":     f"{agents.__name__}:ppo_cfg",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:ppo_cfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

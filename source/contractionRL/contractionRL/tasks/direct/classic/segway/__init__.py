import gymnasium as gym

gym.register(
    id="classic-segway-v0",
    entry_point=f"{__name__}.env:SegwayEnv",
    disable_env_checker=True,
    kwargs={
        "skrl_c3m_cfg_entry_point": f"{__name__}.agents:skrl_c3m_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

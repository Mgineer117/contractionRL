from contractionRL.agents.utils.RL.config import config_entrypoints

ppo_cfg, c2rl_ppo_cfg, contraction_ppo_cfg, sac_cfg, c2rl_sac_cfg, c3m_cfg, lqr_cfg, sdlqr_cfg, mbpo_cfg = config_entrypoints(
    __file__, ('ppo', 'c2rl_ppo', 'contraction_ppo', 'sac', 'c2rl_sac', 'c3m', 'lqr', 'sdlqr', 'mbpo'))

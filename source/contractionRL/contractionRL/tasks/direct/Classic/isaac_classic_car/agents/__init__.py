"""skrl agent configurations for Isaac-Classic-Car-v0."""

from contractionRL.agents.utils.RL.config import config_entrypoints
ppo_cfg, c2rl_ppo_cfg, contraction_ppo_cfg = config_entrypoints(__file__)

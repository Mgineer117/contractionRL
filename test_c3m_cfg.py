import yaml
from contractionRL.agents.skrl.c3m import C3MCfg

with open("source/contractionRL/contractionRL/tasks/direct/classic/cartpole/agents/skrl_c3m_cfg.yaml") as f:
    cfg_dict = yaml.safe_load(f)

cfg_dict["agent"]["experiment"].setdefault("wandb_kwargs", {})["sync_tensorboard"] = False

cfg = C3MCfg(**{k: v for k, v in cfg_dict["agent"].items() if k in C3MCfg.__dataclass_fields__})
print("Experiment dict:", cfg.experiment)

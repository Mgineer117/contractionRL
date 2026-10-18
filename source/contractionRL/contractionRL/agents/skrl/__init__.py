"""Algorithm agents, with compatibility imports for utilities moved to agents.utils."""

from contractionRL.agents.utils.RL.models import CLActorModel, MetricModel
from contractionRL.agents.skrl.c3m import C3MAgent, C3MCfg, C3MSkrlTrainer
from contractionRL.agents.skrl.sdlqr import SDLQRAgent, LQRAgent, SDLQRCfg, LQRCfg
from contractionRL.agents.skrl.cvstem_lqr import CVSTEMLQRAgent, CVSTEMLQRCfg
from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLPPOCfg, C2RLSACCfg, C2RLSkrlTrainer
from contractionRL.agents.utils.RL.runner import CLActorRunner
from contractionRL.agents.skrl.contraction_ppo import ContractionPPOAgent, ContractionPPOCfg
from contractionRL.agents.skrl.mbpo import MBPOAgent, MBPOCfg

__all__ = [
    "MBPOAgent",
    "MBPOCfg",
    "ContractionPPOAgent",
    "ContractionPPOCfg",
    "CLActorModel",
    "MetricModel",
    "C3MAgent",
    "C3MCfg",
    "C3MSkrlTrainer",
    "SDLQRAgent",
    "LQRAgent",
    "SDLQRCfg",
    "LQRCfg",
    "CVSTEMLQRAgent",
    "CVSTEMLQRCfg",
    "C2RLAgent",
    "C2RLPPOCfg",
    "C2RLSACCfg",
    "C2RLSkrlTrainer",
    "CLActorRunner",
]


def _install_legacy_utility_imports():
    """Keep historical scripts and pickled class paths working without stub files."""
    import importlib
    import sys
    import types

    aliases = {
        "ncm_synthesis": "contraction.ncm_synthesis",
        "ccm_synthesis": "contraction.ccm_synthesis",
        "math_utils": "contraction.math_utils",
        "contraction_metrics": "contraction.metrics",
        "models": "RL.models",
        "runner": "RL.runner",
        "rl_glue": "RL.config",
        "agent_patches": "RL.agent_patches",
        "wandb_logging": "RL.logging",
        "env_wrapper": "RL.env",
        "preprocessors": "RL.env",
        "angle_utils": "system_dynamics.state",
        "state_symmetry": "system_dynamics.state",
        "ref_window": "system_dynamics.reference",
        "reference_state": "system_dynamics.reference",
        "dynamics_pretrain": "system_dynamics.dynamics",
    }
    for old, new in aliases.items():
        module = importlib.import_module(f"contractionRL.agents.utils.{new}")
        sys.modules[f"{__name__}.{old}"] = globals()[old] = module

    legacy = types.ModuleType(f"{__name__}.nn_modules")
    for location, names in {
        "RL.networks": ("MLP", "CLActor", "PreviewSequenceEncoder", "_ACT_MAP", "_MIN_LOG_STD", "_zero_last_linear"),
        "contraction.metric_models": ("CCM_Generator", "CholMetric", "BoundedCCM_Generator", "AnalyticSOSMetric"),
        "system_dynamics.dynamics": ("NeuralDynamics", "_sym_from_names"),
    }.items():
        module = importlib.import_module(f"contractionRL.agents.utils.{location}")
        for name in names:
            setattr(legacy, name, getattr(module, name))
    sys.modules[legacy.__name__] = globals()["nn_modules"] = legacy


_install_legacy_utility_imports()

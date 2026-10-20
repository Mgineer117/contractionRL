"""Shared glue used across contraction agents (C3M/SDLQR/LQR/C2RL) and the
ContractionRunner: raw-dict -> dataclass config filtering, plus the
CMG-derived Mahalanobis reward machinery C2RL always reads its reward from.

Extracted verbatim from C2RLAgent (see c2rl.py's module docstring for the full
normalization rationale) so algorithms call the same code instead of copies
that can silently drift apart.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Sequence


def load_agent_config(path):
    """Resolve YAML inheritance before applying CLI/sweep overrides.

    Shared ancestors in a diamond are applied once; later parents and the
    child override earlier values. Lists are replaced, dictionaries merged.
    """
    import copy
    from pathlib import Path
    import yaml
    result, applied, active = {}, set(), set()

    def merge(target, values):
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def visit(filename):
        filename = Path(filename).resolve()
        if filename in active:
            raise ValueError(f"Cyclic config inheritance: {filename}")
        if filename in applied:
            return
        active.add(filename)
        with filename.open() as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"Agent config must be a mapping: {filename}")
        parents = values.pop("extends", [])
        parents = [parents] if isinstance(parents, str) else parents
        if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
            raise ValueError(f"extends must be a path or list of paths: {filename}")
        for parent in parents:
            visit(filename.parent / parent)
        merge(result, values)
        active.remove(filename)
        applied.add(filename)

    visit(path)
    if str(result.get("agent", {}).get("class", "")).upper() == "C2RL-PPO":
        models = result.get("models", {})
        if "value" in models:
            inherited = models.pop("value")
            merge(inherited, models.get("critic", {}))
            models["critic"] = inherited
    return result


def config_entrypoints(source_file):
    """Isaac Lab calls these loaders before Hydra applies trial overrides."""
    from pathlib import Path
    directory = Path(source_file).parent
    def ppo_cfg():
        return load_agent_config(directory / "skrl_ppo_cfg.yaml")
    def c2rl_ppo_cfg():
        return load_agent_config(directory / "skrl_c2rl_ppo_cfg.yaml")
    def contraction_ppo_cfg():
        return load_agent_config(directory / "skrl_contraction_ppo_cfg.yaml")
    return ppo_cfg, c2rl_ppo_cfg, contraction_ppo_cfg


def load_registered_agent_config(task, key):
    import gymnasium as gym
    import importlib
    from pathlib import Path
    entry = gym.spec(task.split(":")[-1]).kwargs[key]
    if callable(entry):
        return entry()
    if ":" not in entry:
        return load_agent_config(entry)
    module, name = entry.split(":")
    package = importlib.import_module(module)
    return load_agent_config(Path(package.__file__).parent / name) if name.endswith(".yaml") else getattr(package, name)()


ISAAC_RL_TIMESTEPS = {
    "Isaac-Classic-Car-v0": 500_000,
    "Isaac-Classic-Car-v1": 500_000,
    "Isaac-Classic-Quadrotor-v0": 2_000_000,
    "Isaac-Car-v0": 500_000,
    "Isaac-Car-v1": 500_000,
    "Isaac-Classic-Cartpole-v0": 1_000_000,
    "Isaac-Classic-Segway-v0": 1_000_000,
    "Isaac-Cartpole-v0": 1_000_000,
    "Isaac-Segway-v0": 1_000_000,
    "Isaac-Quadrotor-v0": 2_000_000,
    "Manipulator-PathTracking-v0": 3_000_000,
    "Manipulator-VelTracking-v0": 3_000_000,
    "Quadruped-PathTracking-v0": 4_000_000,
    "Quadruped-VelTracking-v0": 4_000_000,
    "Humanoid-PathTracking-v0": 5_000_000,
    "Humanoid-VelTracking-v0": 5_000_000,
}


def default_isaac_rl_timesteps(task, algorithm):
    """Task budgets count vector steps; offline controllers keep their own horizons."""
    if algorithm.lower().replace("_", "-") in {"c3m", "lqr", "sdlqr", "sd-lqr", "cvstem-lqr"}:
        return None
    return ISAAC_RL_TIMESTEPS.get(str(task).rsplit(":", 1)[-1])


def apply_isaac_rl_training_default(agent_cfg, task, algorithm, *, overrides=()):
    """Apply task defaults before CLI/sweep overrides; Hydra has already resolved its overrides."""
    budget = default_isaac_rl_timesteps(task, algorithm)
    explicit = any(arg.split("=", 1)[0].removeprefix("--").lstrip("+~")
                   in {"trainer.timesteps", "agent.trainer.timesteps"} for arg in overrides)
    if budget is not None and not explicit:
        agent_cfg.setdefault("trainer", {})["timesteps"] = budget


CONTEXT_ROLES = ("policy", "critic", "value", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
REMOVED_CONTEXT_OPTIONS = frozenset({"backbone", "preview_steps", "temporal_context", "use_history", "encoder_stride",
                                     "encoder_hidden", "embed_dim", "feature_dim", "encoder_window"})


def validate_experiment_config(cfg):
    """Reject removed selectors rather than silently running another experiment."""
    if 'extends' in cfg:
        raise ValueError('Resolve config inheritance with load_agent_config before constructing an algorithm')
    if 'gru' in cfg:
        raise ValueError('gru settings moved to models.encoder')
    for name, block in (("config", cfg), ("agent", cfg.get("agent", {})),
                        ("empirical_dynamics", cfg.get("empirical_dynamics", {}))):
        for key in ("mode", "use_empirical_dynamics", "use_learned_disturbance"):
            if key in block:
                raise ValueError(f"{name}.{key} was removed; the environment determines exact or learned dynamics")
    for role in CONTEXT_ROLES:
        block = cfg.get("models", {}).get(role, {})
        removed = REMOVED_CONTEXT_OPTIONS.intersection(block)
        if removed:
            raise ValueError(f"models.{role}: removed context options {sorted(removed)}; use encoder and models.encoder")
        if block.get("encoder") not in (None, "gru"):
            raise ValueError(f"models.{role}.encoder must be null (no GRU) or gru")
    if str(cfg.get("agent", {}).get("class", "")).lower() == "c3m" and any(
            cfg.get("models", {}).get(role, {}).get("encoder") == "gru" for role in CONTEXT_ROLES):
        raise ValueError("C3M offline state sampling has no measured history; GRU context requires a rollout-trained policy/critic")
    critic = cfg.get("models", {}).get("critic", cfg.get("models", {}).get("value", cfg.get("models", {}).get("critic_1", {})))
    for role in CONTEXT_ROLES[1:]:
        block = cfg.get("models", {}).get(role)
        if block is not None and "encoder" in block and block["encoder"] != critic.get("encoder"):
            raise ValueError("All critic and target critic encoders must match")
    from contractionRL.agents.utils.RL.config_usage import validate_algorithm_config
    validate_algorithm_config(cfg)
    shared = cfg.get("models", {}).get("encoder", {})
    if not isinstance(shared, dict) or set(shared) - {"feature_dim", "encoder_window"}:
        raise ValueError("models.encoder accepts only feature_dim and encoder_window")
    for key in ("feature_dim", "encoder_window"):
        value = shared.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"models.encoder.{key} must be a positive integer or null")


def configure_encoder_context(env, cfg):
    """Build one raw context bank; each network independently chooses whether to encode it."""
    import math
    from contractionRL.runners.contraction_runner import _unwrap_env
    validate_experiment_config(cfg)
    raw = _unwrap_env(env)
    from contractionRL.agents.utils.RL.config_usage import validate_optimization_parameters
    validate_optimization_parameters(cfg, cfg.get('_tuned_parameters', []), raw)
    roles = cfg.get("models", {})
    enabled = any(roles.get(role, {}).get("encoder") == "gru" for role in CONTEXT_ROLES)
    window = getattr(raw, "ref_window", None)
    if enabled:
        if window is None or not getattr(window, "compact", False) or not hasattr(raw, "configure_encoder_context"):
            raise ValueError("GRU context requires compact tracking observations s=(x, xref, xi)")
        shared = dict(roles.get("encoder", {}))
        count = shared.get("encoder_window")
        if count is None:
            gamma = float(cfg.get("agent", {}).get("discount_factor", .99))
            if not math.isfinite(gamma) or not 0 <= gamma < 1:
                raise ValueError("A null encoder_window requires 0 <= discount_factor < 1")
            count = max(1, math.ceil(round(1 / (1 - gamma), 10)))
        dim = shared.get("feature_dim")
        dim = 3 * window.x_dim if dim is None else dim
        roles["encoder"] = {"feature_dim": dim, "encoder_window": count}
        if window.preview_length != count or window.history_length != count:
            raw.configure_encoder_context(count)
    elif window is not None and getattr(window, "compact", False) and (window.preview_length or window.history_length):
        raw.configure_encoder_context(0)
    return roles.get("encoder", {})


def model_context_config(cfg):
    """Expand shared GRU parameters only for internal model construction."""
    import copy
    models = copy.deepcopy(cfg.get("models", {}))
    shared = models.pop("encoder", {})
    critic = models.get("critic", models.get("value", models.get("critic_1", {})))
    critic_encoder = critic.get("encoder")
    for role in CONTEXT_ROLES:
        block = models.get(role)
        if isinstance(block, dict):
            block.setdefault("encoder", None if role == "policy" else critic_encoder)
            # Existing model constructors call the context-free path "mlp".
            if block["encoder"] is None:
                block["encoder"] = "mlp"
            if block["encoder"] == "gru":
                block["feature_dim"] = shared.get("feature_dim")
    return models


def filter_cfg_fields(cfg_dict: dict, dataclass_type, *, context: str, forwarded_fields=()) -> dict:
    """Keep only keys that are declared fields of ``dataclass_type``.

    Any other key is *not applied* to the agent/trainer — so instead of
    dropping it silently (which is how config typos and stale sweep parameter
    names went unnoticed), warn loudly with the ignored keys. ``class`` is
    expected to be stripped by the caller and is never reported.
    """
    fields = dataclass_type.__dataclass_fields__
    ignored = sorted(k for k in cfg_dict if k not in fields and k != "class" and k not in forwarded_fields)
    if ignored:
        warnings.warn(
            f"[{context}] ignoring config key(s) not in "
            f"{dataclass_type.__name__} (NOT applied to the algorithm): {ignored}",
            stacklevel=2,
        )
    return {k: v for k, v in cfg_dict.items() if k in fields}


def migrate_c2rl_metric_cfg(cfg: dict) -> dict:
    """Resolve CMG section keys and saved-config aliases for the flat agent namespace."""
    cfg = cfg.copy()
    names = {
        "ccm_epochs": "cmg_ccm_epochs",
        "ccm_lr": "cmg_ccm_lr",
        "ccm_batch_size": "cmg_ccm_batch_size",
        "cvstem_batch_size": "cmg_regress_batch_size",
        "regress_epochs": "cmg_regress_epochs",
        "regress_lr": "cmg_regress_lr",
        "regress_lr_scheduler": "cmg_regress_lr_scheduler",
        "regress_lr_scheduler_kwargs": "cmg_regress_lr_scheduler_kwargs",
        "val_frac": "cmg_val_frac",
        "early_stop_patience": "cmg_early_stop_patience",
        "random_ratio": "cmg_random_ratio",
    }
    for name, internal in names.items():
        if name in cfg:
            cfg[internal] = cfg.pop(name)
    if "cmg_memory_size" in cfg:
        value = cfg.pop("cmg_memory_size")
        cfg.setdefault("ccm_memory_size", value)
        cfg.setdefault("cvstem_memory_size", value)
    for key in ("w_lb", "w_ub", "lbd"):
        if key in cfg:
            value = cfg.pop(key)
            for method in ("ccm", "cvstem"):
                cfg.setdefault(f"{method}_{key}", value)
    old_margin = cfg.pop("ccm_margin", None)
    if old_margin is not None and cfg.get("cmg_method", "cvstem") == "ccm":
        cfg["cm_eps"] = old_margin
    return cfg





def validate_observation_normalization(cfg, *, allow_features=False):
    """Keep physical observations intact; models scale only their neural inputs."""
    if (cfg.get("use_state_norm", False) and not allow_features) or any(cfg.get(key) is not None for key in
            ("observation_preprocessor", "state_preprocessor")):
        raise ValueError("use_state_norm and external observation/state preprocessors are unsupported for physical-coordinate tracking; use the model's tracking features instead")


def make_base_rl_cfg(
    raw_cfg: dict,
    *,
    base_algorithm: str,
    gamma: float,
    name: str,
    experiment_dir: str,
    device,
    observation_space,
    angle_idx: Sequence[int],
    x_dim: int,
    u_dim: int,
) -> dict:
    """Project a raw C2RL-style config dict down to a real PPO_CFG/SAC_CFG dict.

    Passing C2RL/CMG-specific keys (W_lr, lbd, cm_eps, ...) to
    PPO_CFG(**cfg) / SAC_CFG(**cfg) would raise TypeError, since those are
    kw_only dataclasses that reject unknown kwargs. Also rebuilds `experiment`
    as a plain dict (the raw value may be an ExperimentCfg object, which is not
    subscriptable).
    """
    validate_observation_normalization(raw_cfg, allow_features=base_algorithm.upper() == 'PPO')
    if base_algorithm.upper() == "SAC":
        from skrl.agents.torch.sac import SAC_CFG as _BaseCfg
    else:
        from skrl.agents.torch.ppo import PPO_CFG as _BaseCfg
    valid = _BaseCfg.__dataclass_fields__
    d = {k: v for k, v in raw_cfg.items() if k in valid and k != "experiment"}
    d["discount_factor"] = gamma

    # YAML 1.1 (PyYAML) parses unquoted scientific notation without a
    # decimal point (e.g. `1e-5`) as a str, not a float. skrl's Runner
    # normally rescues this via _process_cfg's eval(), but C2RL's inner
    # PPO/SAC sub-agent bypasses Runner entirely, so a `learning_rate: 1e-5`
    # would reach torch.optim.Adam as the string "1e-5" and blow up with
    # "'<=' not supported between instances of 'float' and 'str'".
    # Coerce any numeric-looking string scalar back to float here (the
    # `learning_rate_scheduler` name string is handled just below and is
    # non-numeric, so float() leaves it untouched via the try/except).
    for _k, _v in list(d.items()):
        if isinstance(_v, str):
            try:
                d[_k] = float(_v)
            except ValueError:
                pass

    # Resolve a string "learning_rate_scheduler" (e.g. "KLAdaptiveLR",
    # yaml's usual way of naming it) to the real class — skrl's Runner
    # does this via _process_cfg's eval(), which doesn't run here since
    # C2RL's inner PPO/SAC sub-agent bypasses Runner entirely.
    if isinstance(d.get("learning_rate_scheduler"), str):
        import torch
        from skrl.resources.schedulers import torch as schedulers
        scheduler_name = d["learning_rate_scheduler"].removeprefix("torch.optim.lr_scheduler.")
        scheduler = getattr(schedulers, scheduler_name, None) or getattr(torch.optim.lr_scheduler, scheduler_name, None)
        if scheduler is None:
            raise ValueError(f"Unsupported learning_rate_scheduler: {scheduler_name}")
        d["learning_rate_scheduler"] = scheduler
    if d.get("learning_rate_scheduler_kwargs") is None:
        d["learning_rate_scheduler_kwargs"] = {}

    # "rewards_shaper_scale" is a yaml convenience (same as skrl's own
    # Runner._process_cfg) for the real PPO_CFG/SAC_CFG field
    # "rewards_shaper", a Callable — translate it here since C2RL's inner
    # PPO/SAC sub-agent bypasses Runner entirely. 1.0 (or unset) is a no-op.
    rewards_shaper_scale = raw_cfg.get("rewards_shaper_scale")
    # use_reward_norm: non-biasing running-std reward normalization (r/std, no
    # mean subtraction — preserves the optimal policy). Installed through the
    # Same rewards_shaper hook and, when both are set, absorbs rewards_shaper_scale
    # as its post-normalization scale (giving reward variance scale²). See
    # preprocessors.RunningRewardScaler for why value_norm alone leaves the
    # metric-bound-dependent (w_lb/w_ub) reward-scale instability unaddressed.
    if raw_cfg.get("use_reward_norm", False):
        from contractionRL.agents.utils.RL.env import RunningRewardScaler
        d["rewards_shaper"] = RunningRewardScaler(
            scale=(rewards_shaper_scale if rewards_shaper_scale is not None else 1.0),
            device=device,
        )
    elif rewards_shaper_scale is not None and rewards_shaper_scale != 1.0:
        d["rewards_shaper"] = lambda rewards, *a, scale=rewards_shaper_scale, **kw: rewards * scale

    if base_algorithm.upper() == "PPO" and raw_cfg.get("use_value_norm", True):
        from contractionRL.agents.utils.RL.env import RunningStandardScaler
        d["value_preprocessor"] = RunningStandardScaler
        d["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # Match the outer agent's logging cadence; it owns W&B and checkpoints.
    experiment = raw_cfg.get("experiment", {})
    write_interval = (experiment.get("write_interval", "auto") if isinstance(experiment, dict)
                      else experiment.write_interval)
    d["experiment"] = {
        "directory": os.path.join(experiment_dir, name),
        "write_interval": write_interval,
        "checkpoint_interval": 0,
    }
    return d



def normalize_agent_cfg(agent_cfg: dict, *, algorithm: str, device=None) -> None:
    """Strip this repo's non-skrl ``agent:`` keys and resolve the preprocessors.

    Shared by both of train.py's routes so a yaml key means the same thing on
    each. Policy exploration standard deviations are learned by the algorithm.

    External observation/state preprocessors are rejected to preserve physical
    coordinates. C2RL-PPO can normalize neural features internally; value
    normalization independently rescales the critic target.
    """
    a = agent_cfg.setdefault("agent", {})
    c2rl = algorithm in ("c2rl-ppo", "c2rl_ppo")
    validate_observation_normalization(a, allow_features=algorithm in ("ppo", "contraction_ppo", "contraction-ppo", "contractionppo", "c2rl-ppo", "c2rl_ppo"))
    ppo_family = algorithm in ("ppo", "contraction_ppo", "contraction-ppo", "contractionppo", "c2rl-ppo", "c2rl_ppo")
    if not ppo_family:
        if a.pop("use_value_expansion", False) or any(key in a for key in ("value_expansion_horizon", "value_expansion_batch_size")):
            raise ValueError("use_value_expansion is supported only by PPO-family algorithms")
        a.pop("use_state_norm", None)
    use_value_norm = a.pop("use_value_norm", True)
    # C2RL builds its own PPO/SAC sub-agent from the raw yaml dict via
    # rl_glue.make_base_rl_cfg, which reads ``use_value_norm`` itself (defaulting
    # to True when absent). Popping it here therefore made the key invisible to
    # C2RL: the `algorithm == "ppo"` branch below never fires for "c2rl-ppo", so
    # the flag was consumed, discarded, and then silently re-defaulted to True —
    # i.e. `use_value_norm: false` / `--use_value_norm false` did nothing at all
    # on any c2rl run, and the critic always got a RunningStandardScaler.
    # Put it back for the contraction algorithms, which are the ones that read it
    # downstream. (Plain ppo/sac still consume it here, as before.)
    if algorithm in ("c2rl-ppo", "c2rl_ppo"):
        a["use_value_norm"] = use_value_norm
    for key in ("state_preprocessor", "state_preprocessor_kwargs",
                "observation_preprocessor", "observation_preprocessor_kwargs"):
        a.pop(key, None)

    if use_value_norm and algorithm in ("ppo", "contraction_ppo", "contraction-ppo", "contractionppo"):
        a["value_preprocessor"] = "RunningStandardScaler"
        a["value_preprocessor_kwargs"] = None
    else:
        a.pop("value_preprocessor", None)
        a.pop("value_preprocessor_kwargs", None)

    # Reward switches are consumed here only for skrl-native algorithms.
    if algorithm in ("ppo", "sac", "mbpo", "contraction_ppo", "contraction-ppo", "contractionppo"):
        if a.pop("use_reward_norm", False):
            from contractionRL.agents.utils.RL.env import RunningRewardScaler
            a["rewards_shaper"] = RunningRewardScaler(
                scale=float(a.pop("rewards_shaper_scale", 1.0)), device=device or "cpu")
        a.pop("reward_euclidean", None)
        a.pop("reward_level", None)

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


ISAAC_RL_TIMESTEPS = {
    "Isaac-Car-v0": 500_000,
    "Isaac-Car-v1": 500_000,
    "Isaac-Classic-Cartpole-v0": 1_000_000,
    "Isaac-Classic-Segway-v0": 1_000_000,
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


EXPERIMENT_MODES = ("analytical", "sim_to_real")


def migrate_dynamics_mode(cfg: dict, *, default="analytical") -> dict:
    """Read historical dynamics flags once, retaining only the canonical mode."""
    cfg = cfg.copy()
    learned = cfg.pop("use_empirical_dynamics", None)
    cfg.pop("use_learned_disturbance", None)
    mode = cfg.get("mode", "sim_to_real" if learned else default)
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"mode must be one of {EXPERIMENT_MODES}, got {mode!r}")
    cfg["mode"] = mode
    return cfg


def normalize_experiment_mode(cfg: dict, *, physx_only=False, default="analytical") -> str:
    """Resolve a full config before models, logging or checkpoint setup."""
    agent = cfg.setdefault("agent", {})
    dynamics = cfg.get("empirical_dynamics", {})
    merged = {**dynamics, **agent}
    if "mode" in cfg:
        merged["mode"] = cfg["mode"]
    mode = migrate_dynamics_mode(merged, default=default)["mode"]
    if physx_only:
        mode = "sim_to_real"
    cfg["mode"] = mode
    for block in (agent, dynamics):
        block.pop("use_empirical_dynamics", None)
        block.pop("use_learned_disturbance", None)
        block.pop("mode", None)
    return mode


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
    validate_observation_normalization(a, allow_features=c2rl)
    if algorithm not in ("c2rl-ppo", "c2rl_ppo"):
        if a.pop("use_value_expansion", False) or any(
                key in a for key in ("value_expansion_horizon", "value_expansion_batch_size")):
            raise ValueError("use_value_expansion is supported only by C2RL-PPO")
    if not c2rl:
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

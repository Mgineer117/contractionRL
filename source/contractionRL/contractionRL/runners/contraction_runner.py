"""ContractionRunner — single unified Runner for all algorithms × all envs.

Mirrors the skrl Runner interface::

    runner = ContractionRunner(env, cfg, task_id="Car-Direct-v0",
                               num_envs=4, is_classic=True)
    runner.run()

Algorithm routing
-----------------
PPO / SAC (and any other skrl-native algo):
    → skrl Runner  (skrl.utils.runner.torch.Runner)

C3M / LQR / SDLQR / C2RL-PPO / C2RL-SAC:
    → native skrl Agent subclasses (C3MAgent, LQRAgent, SDLQRAgent, C2RLAgent)
    with custom skrl Trainers (C3MSkrlTrainer, C2RLSkrlTrainer, or
    SequentialTrainer for eval-only analytical agents). C2RL-PPO/C2RL-SAC are
    the SAME C2RLAgent class — the algo string just selects which
    base_algorithm (PPO or SAC) its con_policy/opt_policy are built on; see
    _setup_c2rl.

Classic env: internally creates SyncVectorEnv + wrap_env.
Isaac env:   env is already a SkrlVecEnvWrapper — passed through.

Custom YAML flags handled here (stripped before passing to skrl)
-----------------------------------------------------------------
use_state_norm: true/false  →  observation_preprocessor
use_value_norm: true/false  →  value_preprocessor (PPO only)
anneal_log_std: true/false  →  exponential log_std schedule 0 → -20
"""

from __future__ import annotations

import copy
import importlib
import os
import warnings

import gymnasium as gym

_SKRL_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "amp", "ippo", "mappo"})
_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "c2rl-ppo", "c2rl-sac"})


# ────────────────────────────────────────────────────────────────────────── #
# Helper: load agent YAML from gym registry entry point
# ────────────────────────────────────────────────────────────────────────── #

_ENTRY_KEY: dict[str, str] = {
    "ppo":   "skrl_cfg_entry_point",
    "sac":   "skrl_sac_cfg_entry_point",
    "c3m":   "skrl_c3m_cfg_entry_point",
    "lqr":   "skrl_lqr_cfg_entry_point",
    "sdlqr": "skrl_sdlqr_cfg_entry_point",
    "c2rl-ppo": "skrl_c2rl_ppo_cfg_entry_point",
    "c2rl-sac": "skrl_c2rl_sac_cfg_entry_point",
}


def _filter_cfg_fields(cfg_dict: dict, dataclass_type, *, context: str) -> dict:
    """Keep only keys that are declared fields of ``dataclass_type``.

    Any other key is *not applied* to the agent/trainer — so instead of dropping
    it silently (which is how config typos and stale sweep parameter names went
    unnoticed), warn loudly with the ignored keys. ``class`` is expected to be
    stripped by the caller and is never reported.
    """
    fields = dataclass_type.__dataclass_fields__
    ignored = sorted(k for k in cfg_dict if k not in fields and k != "class")
    if ignored:
        warnings.warn(
            f"[ContractionRunner] {context}: ignoring config key(s) not in "
            f"{dataclass_type.__name__} (NOT applied to the algorithm): {ignored}",
            stacklevel=2,
        )
    return {k: v for k, v in cfg_dict.items() if k in fields}


def _make_cmg_bounds_fn(cmg_model, w_lb: float):
    """Build the (x_batch) -> (m_bar, m_underbar) closure a path-tracking env's
    PathTracking figure uses to draw its theoretical exponential bound (see
    PathTrackingBase.set_contraction_certificate) — the CURRENT contraction
    metric's eigenvalue extremes over a batch of states, not the static
    w_lb/w_ub config bounds.
    """
    from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse

    ccm_gen = cmg_model.ccm_gen
    x_dim = cmg_model.x_dim

    def _bounds_fn(x_batch):
        import torch
        with torch.no_grad():
            raw_W, _ = ccm_gen(x_batch)
            # Honor the CMG's `bounded` flag so a BoundedCCM_Generator (already
            # eigenvalue-bounded) isn't double-bounded by an extra +w_lb·I —
            # otherwise the plotted contraction envelope would use a different
            # metric than the deployed one. Mirrors c2rl.py's bound_W calls.
            W = bound_W(raw_W, w_lb, x_dim, getattr(ccm_gen, "bounded", False))
            M = spd_inverse(W)
            eigvals = torch.linalg.eigvalsh(M)
        return float(eigvals.max().item()), float(eigvals.min().clamp(min=1e-12).item())

    return _bounds_fn


def load_agent_cfg(task_id: str, algorithm: str, custom_path: str | None = None) -> dict:
    """Load agent YAML config from the gym registry or a custom path."""
    import yaml
    if custom_path:
        with open(custom_path) as f:
            return yaml.safe_load(f)
    entry_key = _ENTRY_KEY.get(algorithm.lower())
    if entry_key is None:
        raise ValueError(f"No known entry-point key for algorithm '{algorithm}'")
    spec = gym.spec(task_id)
    ep = (spec.kwargs or {}).get(entry_key)
    if ep is None:
        available = [k for k in (spec.kwargs or {}) if k.endswith("_entry_point")]
        raise ValueError(
            f"No '{entry_key}' registered for {task_id}. Available: {available}"
        )
    pkg, _, fname = ep.partition(":")
    mod = importlib.import_module(pkg)
    cfg_path = os.path.join(os.path.dirname(mod.__file__), fname)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────────────────────────────── #
# Helper: translate custom YAML flags to skrl format
# ────────────────────────────────────────────────────────────────────────── #

def _prepare_skrl_cfg(cfg: dict, algo: str) -> tuple[dict, bool]:
    """Pop ContractionRL-specific keys and translate to skrl conventions."""
    cfg = copy.deepcopy(cfg)
    a = cfg.setdefault("agent", {})

    use_state = a.pop("use_state_norm", True)
    use_value = a.pop("use_value_norm", True)
    anneal = a.pop("anneal_log_std", False)

    if use_state:
        a["observation_preprocessor"] = "RunningStandardScaler"
        a["observation_preprocessor_kwargs"] = None
        if algo == "ppo":
            a["state_preprocessor"] = "RunningStandardScaler"
            a["state_preprocessor_kwargs"] = None
    else:
        for k in ("state_preprocessor", "state_preprocessor_kwargs",
                  "observation_preprocessor", "observation_preprocessor_kwargs"):
            a.pop(k, None)

    if use_value and algo == "ppo":
        a["value_preprocessor"] = "RunningStandardScaler"
        a["value_preprocessor_kwargs"] = None
    else:
        a.pop("value_preprocessor", None)
        a.pop("value_preprocessor_kwargs", None)

    return cfg, anneal


# ────────────────────────────────────────────────────────────────────────── #
# Helper: PPO log_std annealing hook
# ────────────────────────────────────────────────────────────────────────── #

def _patch_ppo_annealing(runner) -> None:
    """Exponential schedule: log_std = 0*(1-t^5) + (-20)*t^5, t=progress."""
    import torch
    _orig = runner.agent.post_interaction

    def _hook(timestep: int, timesteps: int) -> None:
        _orig(timestep=timestep, timesteps=timesteps)
        ratio = (timestep / max(1, timesteps)) ** 5.0
        val = max(-20.0, min(2.0, float(-20.0 * ratio)))
        with torch.no_grad():
            runner.agent.policy.log_std_parameter.fill_(val)

    runner.agent.post_interaction = _hook


# ────────────────────────────────────────────────────────────────────────── #
# Helper: build skrl env wrapper for contraction agents
# ────────────────────────────────────────────────────────────────────────── #

def _make_skrl_env(env, task_id: str | None, num_envs: int, is_classic: bool):
    """Return a skrl-wrapped env suitable for contraction agents."""
    if is_classic:
        if task_id is None and hasattr(env, "spec") and env.spec:
            task_id = env.spec.id
        if task_id is None:
            raise ValueError("task_id is required for classic envs")
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        vec_env = SyncVectorEnv([lambda: gym.make(task_id)] * num_envs)
        # Do NOT force vec_env.device = "cpu" here. The underlying physics step
        # is numpy/CPU-bound either way, but GymnasiumWrapper.step/reset already
        # bridges numpy <-> torch at the right device via tensorize_space /
        # untensorize_space (actions are .cpu().numpy()'d before being handed to
        # the numpy env; observations are re-tensorized onto `device` after).
        # Leaving vec_env.device unset lets GymnasiumWrapper fall back to the
        # global skrl device (cuda:0 when available) — matching Isaac envs and
        # letting C3M/C2RL's batched Jacobian/backprop math run on GPU instead
        # of silently pinning the whole classic pipeline (including the neural
        # networks) to CPU.
        return wrap_env(vec_env, wrapper="gymnasium")
    return env  # Isaac: already wrapped


# ────────────────────────────────────────────────────────────────────────── #
# ContractionRunner
# ────────────────────────────────────────────────────────────────────────── #

class ContractionRunner:
    """Unified runner — same interface as ``skrl.utils.runner.torch.Runner``.

    Args:
        env: gymnasium environment (raw single env or pre-wrapped).
        cfg: agent + trainer config dict (loaded from YAML).
        task_id: gym env id; needed for SyncVectorEnv on classic envs.
        num_envs: number of parallel envs (classic envs only).
        is_classic: True for classic (no-Isaac) envs.
        ml_framework: ``"torch"`` or ``"jax"`` (skrl-native algos only).
        dynamics_model: optional NeuralDynamics for Isaac envs. Injected into
            the env via ``set_dynamics_model()`` so ``get_f_and_B`` works for
            C3M / LQR / SD-LQR / C2RL. Classic envs have analytical dynamics
            and do not need this.
    """

    def __init__(
        self,
        env,
        cfg: dict,
        *,
        task_id: str | None = None,
        num_envs: int = 1,
        is_classic: bool = True,
        ml_framework: str = "torch",
        dynamics_model=None,
    ) -> None:
        algo = cfg.get("agent", {}).get("class", "ppo").lower()

        if algo in _SKRL_ALGOS:
            self._setup_skrl(env, cfg, algo, task_id, num_envs, is_classic, ml_framework)
        elif algo in _CONTRACTION_ALGOS:
            self._setup_contraction(env, cfg, algo, task_id, num_envs, is_classic, dynamics_model)
        else:
            raise ValueError(
                f"Unknown algorithm '{algo}'. "
                f"skrl: {sorted(_SKRL_ALGOS)}, contraction: {sorted(_CONTRACTION_ALGOS)}"
            )

    # ── skrl-native algos (PPO/SAC/etc.) ───────────────────────────────── #

    def _setup_skrl(self, env, cfg, algo, task_id, num_envs, is_classic, ml_framework):
        skrl_cfg, anneal = _prepare_skrl_cfg(cfg, algo)
        skrl_cfg["trainer"]["close_environment_at_exit"] = False

        # Unwrap to the deepest raw env to check for angle_idx
        raw_env = env
        seen = set()
        while True:
            rid = id(raw_env)
            if rid in seen:
                break
            seen.add(rid)
            inner = getattr(raw_env, "unwrapped", None) or getattr(raw_env, "env", None)
            if inner is None or inner is raw_env:
                break
            raw_env = inner

        angle_idx = list(getattr(raw_env, "angle_idx", []) or [])
        if not angle_idx and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            angle_idx = list(getattr(first_env, "angle_idx", []) or [])

        if skrl_cfg.get("agent", {}).get("observation_preprocessor") == "RunningStandardScaler" and angle_idx:
            raise ValueError(
                "use_state_norm=True (RunningStandardScaler) is fundamentally incompatible "
                "with environments that expose angle_idx. Normalizing raw angles linearly "
                "destroys the periodicity required by embed_angles(cos, sin). "
                "Disable use_state_norm or use a custom scaler."
            )

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)

        if ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            from skrl.utils.runner.jax import Runner

        self._runner = Runner(skrl_env, skrl_cfg)
        self._env = skrl_env
        self._mode = "skrl"

        if anneal and algo == "ppo":
            _patch_ppo_annealing(self._runner)

    # ── contraction agents (C3M/LQR/SDLQR/C2RL) ───────────────────────── #

    def _setup_contraction(self, env, cfg, algo, task_id, num_envs, is_classic, dynamics_model=None):
        from contractionRL.agents.skrl.models import ControllerNetwork, MetricNetwork

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)
        # Use the wrapped env's device for both classic and Isaac envs. The
        # classic env's own physics step is numpy/CPU-bound regardless (see
        # _make_skrl_env), but the agent's neural networks and gradient math
        # (C3M/C2RL Jacobians, batched contraction losses) benefit hugely from
        # GPU — hardcoding CPU here was pinning that compute off the GPU even
        # when one was available. SD-LQR/LQR are unaffected: they pin their own
        # internal `_compute_device = "cpu"` regardless of this value, since
        # scipy's CARE solver is CPU-only anyway.
        device = skrl_env.device
        obs_space = skrl_env.observation_space
        act_space = skrl_env.action_space
        state_space = getattr(skrl_env, "state_space", None)

        # Unwrap to the deepest raw env that exposes get_rollout / get_f_and_B.
        # For classic envs this is the single gymnasium env; for Isaac envs it
        # is the DirectRLEnv subclass buried under SkrlVecEnvWrapper + gym layers.
        raw_env = env
        seen = set()
        while True:
            rid = id(raw_env)
            if rid in seen:
                break
            seen.add(rid)
            inner = getattr(raw_env, "unwrapped", None) or getattr(raw_env, "env", None)
            if inner is None or inner is raw_env:
                break
            raw_env = inner

        # For Isaac Sim envs: inject the neural dynamics model so the env can
        # provide get_f_and_B. Classic envs have analytical dynamics built-in.
        if not is_classic and dynamics_model is not None:
            if hasattr(raw_env, "set_dynamics_model"):
                raw_env.set_dynamics_model(dynamics_model)

        get_f_and_B = getattr(raw_env, "get_f_and_B", None)
        get_rollout = getattr(raw_env, "get_rollout", None)
        if get_f_and_B is None and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            get_f_and_B = getattr(first_env, "get_f_and_B", None)
            get_rollout = getattr(first_env, "get_rollout", None)

        agent_cfg = copy.deepcopy(cfg.get("agent", {}))
        agent_cfg.pop("class", None)
        trainer_cfg = copy.deepcopy(cfg.get("trainer", {}))
        trainer_cfg.pop("class", None)
        trainer_cfg.setdefault("close_environment_at_exit", False)

        if agent_cfg.get("use_analytical_dynamics") and not is_classic:
            raise ValueError(
                "use_analytical_dynamics=True is only valid for classic envs that expose "
                "analytical get_f_and_B. Isaac Sim envs have no analytical dynamics — "
                "remove --use_analytical_dynamics and let C3M train NeuralDynamics online."
            )
            
        x_dim = getattr(raw_env, "x_dim", None)
        u_dim = getattr(raw_env, "u_dim", None)
        # angle_idx: indices within an x-block that hold a raw (wrapping) angle.
        # Every network built below embeds these as (cos, sin) at its input —
        # see angle_utils.py / models.py — while the env/loss/error math keeps
        # using the RAW state. Defaults to [] (no angle dims) for envs that
        # don't declare it.
        angle_idx = list(getattr(raw_env, "angle_idx", []) or [])
        if x_dim is None and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            x_dim = getattr(first_env, "x_dim", None)
            u_dim = getattr(first_env, "u_dim", None)
            if not angle_idx:
                angle_idx = list(getattr(first_env, "angle_idx", []) or [])

        models_cfg = copy.deepcopy(cfg.get("models", {}))
        memory_cfg = copy.deepcopy(cfg.get("memory", {}))

        if algo in ("c3m",):
            self._setup_c3m(skrl_env, device, obs_space, state_space, act_space,
                            agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                            angle_idx=angle_idx)
        elif algo in ("lqr", "sdlqr"):
            # LQR/SD-LQR build no networks (get_f_and_B is either analytical or
            # an externally-loaded NeuralDynamics whose angle_idx was already
            # baked in when IT was trained) — nothing here to embed.
            self._setup_sdlqr(skrl_env, device, obs_space, state_space, act_space,
                              agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr=(algo == "lqr"), x_dim=x_dim, u_dim=u_dim, angle_idx=angle_idx)
        elif algo in ("c2rl-ppo", "c2rl-sac"):
            # base_algorithm is derived from the algo string itself (i.e. which
            # entry point / yaml you used), not a config toggle — see
            # C2RLAgent's base_algorithm constructor kwarg.
            self._setup_c2rl(skrl_env, device, obs_space, state_space, act_space,
                             agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                             base_algorithm=("PPO" if algo == "c2rl-ppo" else "SAC"), angle_idx=angle_idx)

    def _setup_c3m(self, env, device, obs_space, state_space, act_space,
                   agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                   angle_idx=None):
        angle_idx = list(angle_idx or [])
        from contractionRL.agents.skrl.models import ControllerNetwork, MetricNetwork
        from contractionRL.agents.skrl.c3m import C3MAgent, C3MCfg, C3MSkrlTrainer, C3MTrainerCfg
        import dataclasses

        hd_policy = models_cfg.get("policy", {}).get("network", [{}])[0].get("layers", [128, 128])
        hd_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("layers", [128, 128])
        act_policy = models_cfg.get("policy", {}).get("network", [{}])[0].get("activations", "tanh")
        act_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("activations", "tanh")

        # agent.actor_architecture: sweep-friendly override of the *policy* network
        # spec (the CLActor's w1/w2 MLPs share these hidden layers; only their
        # input/output sizes differ). Accepts a bare hidden-layer list
        # (e.g. [256, 256]) or a dict {layers: [...], activations: "tanh"}.
        # Popped here so it reaches model construction and does not trip the
        # C3MCfg field filter. Only the policy net is affected — not the CMG.
        actor_arch = agent_cfg.pop("actor_architecture", None)
        if isinstance(actor_arch, dict):
            hd_policy = actor_arch.get("layers", hd_policy)
            act_policy = actor_arch.get("activations", act_policy)
        elif actor_arch is not None:
            hd_policy = list(actor_arch)

        # C3M is a certificate-based controller synthesis (not a stochastic-policy
        # RL method), so the CLActor should be deterministic by default — no
        # Gaussian log_std / sampling machinery. A config may still opt into the
        # GaussianMixin wrapper explicitly.
        policy_class_str = models_cfg.get("policy", {}).get("class", "DeterministicMixin")
        if policy_class_str == "GaussianMixin":
            from contractionRL.agents.skrl.models import CLActorModel
            policy_cls = CLActorModel
        else:
            from contractionRL.agents.skrl.models import CLDeterministicActorModel
            policy_cls = CLDeterministicActorModel

        policy_kwargs = models_cfg.get("policy", {}).copy()
        policy_kwargs.pop("class", None)
        policy_kwargs.pop("network", None)
        policy_kwargs.pop("backbone", None)
        policy_kwargs.pop("angle_idx", None)  # angle_idx below is the single source of truth

        constrain_eigenvalues = models_cfg.get("cmg", {}).get("network", [{}])[0].get("constrain_eigenvalues", False)
        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)

        models = {
            "policy": policy_cls(obs_space, act_space, device, hidden_dim=hd_policy, activation=act_policy, x_dim=x_dim, angle_idx=angle_idx, **policy_kwargs),
            "cmg": MetricNetwork(obs_space, act_space, device, hidden_dim=hd_cmg, activation=act_cmg, x_dim=x_dim, angle_idx=angle_idx, constrain_eigenvalues=constrain_eigenvalues, w_lb=w_lb, w_ub=w_ub),
        }

        if "dynamics" in models_cfg:
            from contractionRL.agents.skrl.nn_modules import NeuralDynamics
            dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
            dyn_hd = dyn_net.get("layers", [256, 256])
            dyn_act = dyn_net.get("activations", "relu")
            models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx)

        agent = C3MAgent(
            cfg=C3MCfg(**_filter_cfg_fields(agent_cfg, C3MCfg, context="agent")),
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        trainer_cfg.setdefault("timesteps", 30000)
        trainer = C3MSkrlTrainer(
            cfg=C3MTrainerCfg(**_filter_cfg_fields(trainer_cfg, C3MTrainerCfg, context="trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "c3m"

        # C3M has no discounting (no_op discount_factor=None) — its certificate
        # is a hard per-step LMI, not a discounted-objective approximation.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=None,
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(models["cmg"], w_lb),
            )

    def _setup_sdlqr(self, env, device, obs_space, state_space, act_space,
                     agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr: bool = False, x_dim=None, u_dim=None, angle_idx=None):
        from contractionRL.agents.skrl.sdlqr import SDLQRAgent, LQRAgent, SDLQRCfg, LQRCfg
        from skrl.trainers.torch import SequentialTrainer
        import dataclasses
        from skrl.trainers.torch.base import TrainerCfg

        cfg_cls = LQRCfg if lqr else SDLQRCfg
        agent_cls = LQRAgent if lqr else SDLQRAgent
        cfg_fields = cfg_cls.__dataclass_fields__

        agent = agent_cls(
            cfg=cfg_cls(**{k: v for k, v in agent_cfg.items() if k in cfg_fields}),
            models={},
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
            angle_idx=angle_idx,
        )
        # SequentialTrainer for eval — no gradient updates
        valid_keys = TrainerCfg.__dataclass_fields__
        tcfg = dataclasses.asdict(TrainerCfg(**{
            k: v for k, v in {
                "timesteps": trainer_cfg.get("timesteps", 10000),
                "close_environment_at_exit": False,
            }.items() if k in valid_keys
        }))
        trainer = SequentialTrainer(cfg=tcfg, env=env, agents=agent)
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "lqr" if lqr else "sdlqr"

    def _setup_c2rl(self, env, device, obs_space, state_space, act_space,
                    agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                    base_algorithm: str = "PPO", angle_idx=None):
        angle_idx = list(angle_idx or [])
        from contractionRL.agents.skrl.models import MetricNetwork
        from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer, C2RLTrainerCfg
        from contractionRL.agents.skrl.runner import _gaussian_factory

        base_algorithm = base_algorithm.upper()
        con_only = bool(agent_cfg.get("con_only", False))

        # For PPO, con_memory/opt_memory hold exactly one rollout epoch's transitions
        # and are reset every epoch. For SAC, they act as persistent replay buffers.
        mem_size = memory_cfg.get("memory_size", -1)
        rollouts = agent_cfg.get("rollouts", 300)
        if base_algorithm == "PPO" and mem_size not in (-1, rollouts):
            raise ValueError(
                f"[C2RL] memory.memory_size={mem_size} must be -1 (automatically "
                f"determined, same as agent:rollouts) or exactly match agent.rollouts "
                f"({rollouts}) for PPO."
            )
        
        # We pass mem_size via memory_cfg into C2RLAgent constructor

        # con_policy/opt_policy: same backbone dispatch PPO/SAC path-tracking
        # configs use (control -> CLActorModel, mlp -> MLPResidualActorModel,
        # both u = uref + feedback) — see runner.py's _gaussian_factory.
        def _build_policy(block_name):
            spec = models_cfg.get(block_name, {}).copy()
            spec.pop("class", None)
            backbone = spec.pop("backbone", "control")
            spec.setdefault("clip_log_std", True)
            spec.setdefault("min_log_std", -4.605)
            spec.setdefault("max_log_std", 2.0)
            spec.setdefault("angle_idx", angle_idx)
            # x_dim is already known from raw_env.x_dim (see top of _setup_c2rl) —
            # pass it explicitly so backbones like mlp-squashed, which see both
            # path-tracking and vel-tracking layouts, don't have to guess the
            # [x, xref, uref] split from obs_dim/act_dim parity alone.
            spec.setdefault("x_dim", x_dim)
            # agent_class lets _gaussian_factory reject unbounded backbones for
            # C2RL-SAC (divergence) and squashed backbones for C2RL-PPO.
            return _gaussian_factory(obs_space, state_space, act_space, device,
                                     backbone=backbone, agent_class=f"C2RL-{base_algorithm}",
                                     **spec)

        con_policy = _build_policy("con_policy")
        opt_policy = None if con_only else _build_policy("opt_policy")

        cmg_net = models_cfg.get("cmg", {}).get("network", [{}])
        cmg_hd = cmg_net[0].get("layers", [256, 256]) if cmg_net else [256, 256]
        cmg_act = cmg_net[0].get("activations", "tanh") if cmg_net else "tanh"
        constrain_eigenvalues = cmg_net[0].get("constrain_eigenvalues", False) if cmg_net else False
        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)
        cmg_model = MetricNetwork(
            obs_space, act_space, device, hidden_dim=cmg_hd, activation=cmg_act,
            x_dim=x_dim, angle_idx=angle_idx, constrain_eigenvalues=constrain_eigenvalues, w_lb=w_lb, w_ub=w_ub,
        )

        # con_critic/opt_critic: algorithm-agnostic network spec — used as a
        # single value function V(obs) for PPO, or as the architecture shared
        # by all 4 twin-Q networks for SAC (critic_1/critic_2 + their targets).
        def _critic_spec(block_name):
            net = models_cfg.get(block_name, {}).get("network", [{}])
            hd = net[0].get("layers", [256, 256]) if net else [256, 256]
            act = net[0].get("activations", "tanh") if net else "tanh"
            return hd, act

        # con_value/opt_value (PPO) and con_critic_*/opt_critic_* (SAC) all see
        # obs with the SAME angle-bearing x/xref blocks as the policy — embed
        # their input identically (EmbeddedDeterministicModel; see models.py).
        # Q-models additionally concatenate the RAW action (never angle-valued).
        from contractionRL.agents.skrl.models import EmbeddedDeterministicModel

        def _make_value_model(hidden_dim, activation):
            """V(obs) -> scalar (PPO critic)."""
            return EmbeddedDeterministicModel(
                obs_space, act_space, device, hidden_dim=hidden_dim, activation=activation,
                x_dim=x_dim, angle_idx=angle_idx, use_actions=False,
            )

        def _make_q_model(hidden_dim, activation):
            """Q(obs, action) -> scalar (SAC critic_1/critic_2/targets)."""
            return EmbeddedDeterministicModel(
                obs_space, act_space, device, hidden_dim=hidden_dim, activation=activation,
                x_dim=x_dim, angle_idx=angle_idx, use_actions=True,
            )

        models = {"con_policy": con_policy, "cmg": cmg_model}
        if opt_policy is not None:
            models["opt_policy"] = opt_policy

        # NeuralDynamics (learned ẋ = f(x) + B(x)·u) — same mechanism as C3M.
        # Built unless analytical dynamics is requested (classic-only). For Isaac
        # envs it is the ONLY source of f/B; C2RLAgent uses it whenever
        # use_analytical_dynamics is false.
        if "dynamics" in models_cfg and not agent_cfg.get("use_analytical_dynamics"):
            from contractionRL.agents.skrl.nn_modules import NeuralDynamics
            dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
            dyn_hd = dyn_net.get("layers", [256, 256])
            dyn_act = dyn_net.get("activations", "relu")
            models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx)

        if base_algorithm == "PPO":
            con_hd, con_act = _critic_spec("con_critic")
            models["con_value"] = _make_value_model(con_hd, con_act)
            if opt_policy is not None:
                opt_hd, opt_act = _critic_spec("opt_critic")
                models["opt_value"] = _make_value_model(opt_hd, opt_act)
        elif base_algorithm == "SAC":
            con_hd, con_act = _critic_spec("con_critic")
            for key in ("con_critic_1", "con_critic_2", "con_target_critic_1", "con_target_critic_2"):
                models[key] = _make_q_model(con_hd, con_act)
            if opt_policy is not None:
                opt_hd, opt_act = _critic_spec("opt_critic")
                for key in ("opt_critic_1", "opt_critic_2", "opt_target_critic_1", "opt_target_critic_2"):
                    models[key] = _make_q_model(opt_hd, opt_act)
        else:
            raise ValueError(f"[C2RL] Unsupported base_algorithm: {base_algorithm}")

        # Pass the RAW agent_cfg dict (not a pre-parsed C2RLPPOCfg/C2RLSACCfg) —
        # C2RLAgent keeps the full dict in self._raw_cfg, which is what
        # _make_base_cfg() later filters against PPO_CFG/SAC_CFG. Pre-parsing
        # first would silently drop any PPO- or SAC-specific field not ALSO
        # declared on the matching C2RL cfg dataclass (this bit us for
        # base_algorithm before it became an explicit constructor kwarg — see
        # C2RLAgent.__init__).
        agent_cfg["memory_size"] = mem_size
        agent = C2RLAgent(
            cfg=agent_cfg,
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            base_algorithm=base_algorithm,
            x_dim=x_dim,
            u_dim=u_dim,
            num_envs=env.num_envs,
            angle_idx=angle_idx,
        )
        trainer_cfg.setdefault("timesteps", 300000)
        trainer_cfg.setdefault("rollouts", agent_cfg.get("rollouts", 300))
        trainer = C2RLSkrlTrainer(
            cfg=C2RLTrainerCfg(**_filter_cfg_fields(trainer_cfg, C2RLTrainerCfg, context="trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "c2rl"

        # C2RL's certificate is trained against con_policy (gamma_contracting);
        # the PathTracking figure shows whichever policy act() actually deploys
        # — opt_policy (gamma_optimal) unless con_only.
        deployed_gamma = agent_cfg.get("gamma_contracting", 0.0) if con_only else agent_cfg.get("gamma_optimal", 0.99)
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=deployed_gamma,
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(cmg_model, w_lb),
            )

    # ── public interface ────────────────────────────────────────────────── #

    @property
    def agent(self):
        if self._mode == "skrl":
            return self._runner.agent
        return self._agent

    @property
    def trainer(self):
        if self._mode == "skrl":
            return getattr(self._runner, "_trainer", None) or getattr(self._runner, "trainer", None)
        return self._trainer

    def load(self, path: str) -> None:
        if self._mode == "skrl":
            self._runner.agent.load(path)
        else:
            self._agent.load(path)

    def run(self):
        if self._mode == "skrl":
            return self._runner.run()
        return self._trainer.train()

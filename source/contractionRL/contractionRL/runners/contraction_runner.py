"""ContractionRunner — single unified Runner for all algorithms × all envs.

Mirrors the skrl Runner interface::

    runner = ContractionRunner(env, cfg, task_id="Car-Direct-v0",
                               num_envs=4, is_classic=True)
    runner.run()

Algorithm routing
-----------------
PPO / SAC (and any other skrl-native algo):
    → skrl Runner  (skrl.utils.runner.torch.Runner)

C3M / LQR / SDLQR / TEMP:
    → native skrl Agent subclasses (C3MAgent, LQRAgent, SDLQRAgent, TEMPAgent)
    with custom skrl Trainers (C3MSkrlTrainer, TEMPSkrlTrainer, or
    SequentialTrainer for eval-only analytical agents).

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

import gymnasium as gym

_SKRL_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "amp", "ippo", "mappo"})
_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "temp"})


# ────────────────────────────────────────────────────────────────────────── #
# Helper: load agent YAML from gym registry entry point
# ────────────────────────────────────────────────────────────────────────── #

_ENTRY_KEY: dict[str, str] = {
    "ppo":   "skrl_cfg_entry_point",
    "sac":   "skrl_sac_cfg_entry_point",
    "c3m":   "skrl_c3m_cfg_entry_point",
    "lqr":   "skrl_lqr_cfg_entry_point",
    "sdlqr": "skrl_sdlqr_cfg_entry_point",
    "temp":  "skrl_temp_cfg_entry_point",
}


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
        # letting C3M/TEMP's batched Jacobian/backprop math run on GPU instead
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
            C3M / LQR / SD-LQR / TEMP. Classic envs have analytical dynamics
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

    # ── contraction agents (C3M/LQR/SDLQR/TEMP) ───────────────────────── #

    def _setup_contraction(self, env, cfg, algo, task_id, num_envs, is_classic, dynamics_model=None):
        from contractionRL.agents.skrl.models import CLActorModel, CMGModel

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)
        # Use the wrapped env's device for both classic and Isaac envs. The
        # classic env's own physics step is numpy/CPU-bound regardless (see
        # _make_skrl_env), but the agent's neural networks and gradient math
        # (C3M/TEMP Jacobians, batched contraction losses) benefit hugely from
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
        if x_dim is None and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            x_dim = getattr(first_env, "x_dim", None)
            u_dim = getattr(first_env, "u_dim", None)

        models_cfg = copy.deepcopy(cfg.get("models", {}))

        if algo in ("c3m",):
            self._setup_c3m(skrl_env, device, obs_space, state_space, act_space,
                            agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim, u_dim)
        elif algo in ("lqr", "sdlqr"):
            self._setup_sdlqr(skrl_env, device, obs_space, state_space, act_space,
                              agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr=(algo == "lqr"), x_dim=x_dim, u_dim=u_dim)
        elif algo == "temp":
            self._setup_temp(skrl_env, device, obs_space, state_space, act_space,
                             agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim, u_dim)

    def _setup_c3m(self, env, device, obs_space, state_space, act_space,
                   agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None):
        from contractionRL.agents.skrl.models import CLActorModel, CMGModel
        from contractionRL.agents.skrl.c3m import C3MAgent, C3MCfg, C3MSkrlTrainer, C3MTrainerCfg
        import dataclasses

        hd_policy = models_cfg.get("policy", {}).get("network", [{}])[0].get("layers", [128, 128])
        hd_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("layers", [128, 128])
        act_policy = models_cfg.get("policy", {}).get("network", [{}])[0].get("activations", "tanh")
        act_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("activations", "tanh")

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

        constrain_eigenvalues = models_cfg.get("cmg", {}).get("network", [{}])[0].get("constrain_eigenvalues", False)
        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)

        models = {
            "policy": policy_cls(obs_space, act_space, device, hidden_dim=hd_policy, activation=act_policy, x_dim=x_dim, **policy_kwargs),
            "cmg": CMGModel(obs_space, act_space, device, hidden_dim=hd_cmg, activation=act_cmg, x_dim=x_dim, constrain_eigenvalues=constrain_eigenvalues, w_lb=w_lb, w_ub=w_ub),
        }
        
        if "dynamics" in models_cfg:
            from contractionRL.agents.skrl.nn_modules import NeuralDynamics
            dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
            dyn_hd = dyn_net.get("layers", [256, 256])
            dyn_act = dyn_net.get("activations", "relu")
            models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device)

        cfg_fields = C3MCfg.__dataclass_fields__
        agent = C3MAgent(
            cfg=C3MCfg(**{k: v for k, v in agent_cfg.items() if k in cfg_fields}),
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
        tcfg_fields = C3MTrainerCfg.__dataclass_fields__
        trainer = C3MSkrlTrainer(
            cfg=C3MTrainerCfg(**{k: v for k, v in trainer_cfg.items() if k in tcfg_fields}),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "c3m"

    def _setup_sdlqr(self, env, device, obs_space, state_space, act_space,
                     agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr: bool = False, x_dim=None, u_dim=None):
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

    def _setup_temp(self, env, device, obs_space, state_space, act_space,
                    agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None):
        from contractionRL.agents.skrl.models import CLActorModel, CMGModel
        from contractionRL.agents.skrl.temp import TEMPAgent, TEMPCfg, TEMPSkrlTrainer, TEMPTrainerCfg
        from skrl.models.torch import DeterministicMixin, Model
        import dataclasses

        hd = agent_cfg.pop("hidden_dim", [128, 128])
        cmg_hd = agent_cfg.pop("cmg_hidden_dim", hd)

        def _make_value_model():
            """Standard MLP value function for PPO sub-agents."""
            import torch.nn as nn
            from skrl.models.torch import DeterministicMixin, Model

            class _ValueModel(DeterministicMixin, Model):
                def __init__(self, obs_space, act_space, dev):
                    Model.__init__(self, observation_space=obs_space, action_space=act_space, device=dev)
                    DeterministicMixin.__init__(self, clip_actions=False)
                    n_obs = int(obs_space.shape[0])
                    self._net = nn.Sequential(
                        nn.Linear(n_obs, 256), nn.Tanh(),
                        nn.Linear(256, 256), nn.Tanh(),
                        nn.Linear(256, 1),
                    )
                def compute(self, inputs, role="value"):
                    return self._net(inputs["observations"]), {}

            return _ValueModel(obs_space, act_space, device)

        models = {
            "con_policy": CLActorModel(obs_space, act_space, device, hidden_dim=hd),
            "con_value": _make_value_model(),
            "opt_policy": CLActorModel(obs_space, act_space, device, hidden_dim=hd),
            "opt_value": _make_value_model(),
            "cmg": CMGModel(obs_space, act_space, device, hidden_dim=cmg_hd),
        }

        cfg_fields = TEMPCfg.__dataclass_fields__
        agent = TEMPAgent(
            cfg=TEMPCfg(**{k: v for k, v in agent_cfg.items() if k in cfg_fields}),
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            num_envs=env.num_envs,
        )
        trainer_cfg.setdefault("timesteps", 300000)
        trainer_cfg.setdefault("rollouts", agent_cfg.get("rollouts", 300))
        tcfg_fields = TEMPTrainerCfg.__dataclass_fields__
        trainer = TEMPSkrlTrainer(
            cfg=TEMPTrainerCfg(**{k: v for k, v in trainer_cfg.items() if k in tcfg_fields}),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "temp"

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

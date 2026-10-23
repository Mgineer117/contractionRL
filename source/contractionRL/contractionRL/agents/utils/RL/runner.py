"""Tracking-aware skrl models with independent policy and critic context encoders."""
from __future__ import annotations

from skrl.utils.runner.torch import Runner

def _is_window_space(space) -> bool:
    """True for a ``{x, xrefs, urefs}`` path-tracking observation. Velocity-
    tracking envs keep a flat Box obs and route to skrl's stock models."""
    import gymnasium as gym
    return isinstance(space, gym.spaces.Dict) and {"x", "xrefs", "urefs"} <= set(space.spaces)


def _gaussian_factory(observation_space, state_space, action_space, device, agent_class=None, **kwargs):
    """Select the controller by control_mode and the action distribution by algorithm."""
    is_sac = "SAC" in str(agent_class).upper() or str(agent_class).upper() == "MBPO"
    control_mode = kwargs.pop("control_mode", "structured")
    network = kwargs.pop("network", [{}]) or [{}]
    kwargs.pop("output", None)
    spec = network[0]
    hidden, activation = spec.get("layers", [128, 128]), spec.get("activations", "tanh")
    if hasattr(observation_space, "reference_state"):
        from contractionRL.agents.utils.RL.models import FullControlActorModel, FullControlSquashedActorModel
        model = FullControlSquashedActorModel if is_sac else FullControlActorModel
        return model(observation_space, action_space, device, hidden_dim=hidden,
                     activation=activation, control_mode=control_mode, **kwargs)
    if _is_window_space(observation_space):
        from contractionRL.agents.utils.RL.models import CLActorModel, SquashedCLActorModel, MLPResidualActorModel, SquashedMLPActorModel
        if control_mode == "structured":
            model = SquashedCLActorModel if is_sac else CLActorModel
        elif control_mode == "residual":
            model = SquashedMLPActorModel if is_sac else MLPResidualActorModel
        else:
            raise ValueError("full/factorized control requires compact tracking observations")
        return model(observation_space, action_space, device, hidden_dim=hidden, activation=activation, **kwargs)
    if control_mode == "residual":
        raise ValueError("Residual control requires a tracking reference")
    if is_sac:
        from contractionRL.agents.utils.RL.models import SquashedMLPActorModel
        kwargs.pop("initial_log_std", None)
        kwargs.pop("clip_actions", None)
        return SquashedMLPActorModel(observation_space=observation_space, action_space=action_space,
                                    device=device, hidden_dim=hidden, activation=activation, **kwargs)
    from skrl.utils.model_instantiators.torch import gaussian_model
    for key in ("angle_idx", "sym", "x_dim", "encoder", "feature_dim"):
        kwargs.pop(key, None)
    return gaussian_model(observation_space=observation_space, state_space=state_space,
                          action_space=action_space, device=device, network=network, output="ACTIONS", **kwargs)


def _deterministic_factory(observation_space, state_space, action_space, device, **kwargs):
    """Build an independent V/Q network with optional tracking context."""
    network = kwargs.pop("network", [{}])
    kwargs.pop("output", None)
    net_spec = network[0] if network else {}
    use_actions = "ACTIONS" in str(net_spec.get("input", "OBSERVATIONS"))
    hidden_dim = net_spec.get("layers", [256, 256])
    activation = net_spec.get("activations", "tanh")

    if hasattr(observation_space, "reference_state"):
        from contractionRL.agents.utils.RL.models import FullControlValueModel
        return FullControlValueModel(observation_space, action_space, device,
                                     hidden_dim=hidden_dim, activation=activation,
                                     use_actions=use_actions, **kwargs)

    if not _is_window_space(observation_space):
        from skrl.utils.model_instantiators.torch import deterministic_model
        for _k in ("angle_idx", "sym", "x_dim", "encoder", "encoder_hidden",
                   "encoder_stride", "combine", "analytic_potential", "w_lb"):
            kwargs.pop(_k, None)
        return deterministic_model(
            observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
            network=network or [{}], output="ONE", **kwargs)

    from contractionRL.agents.utils.RL.models import RefWindowValueModel

    if "embed_dim" in kwargs:
        kwargs.setdefault("encoder_hidden", int(kwargs.pop("embed_dim")))

    return RefWindowValueModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        hidden_dim=hidden_dim,
        activation=activation,
        use_actions=use_actions,
        **kwargs,
    )


class CLActorRunner(Runner):
    def run(self, mode='train'):
        from contractionRL.runners.contraction_runner import _unwrap_env
        from contractionRL.tasks.direct.common.transfer_dynamics import is_transfer_path, evaluate_backend_pair
        from contractionRL.agents.utils.RL.logging import evaluation_section
        raw = _unwrap_env(self._env)
        transfer = is_transfer_path(raw)
        if mode == 'eval' and transfer:
            from contractionRL.agents.utils.RL.evaluation import rollout_stability
            from contractionRL.agents.utils.contraction.metrics import StatManagerEnvWrapper
            env = self._env if hasattr(self._env, 'stability_summary') else StatManagerEnvWrapper(self._env)
            return evaluate_backend_pair(env, lambda: rollout_stability(env, self.agent, max_steps=raw.max_episode_length + 1))
        with evaluation_section('Simulation Evaluation' if transfer else None):
            return super().run(mode)

    @staticmethod
    def load_cfg_from_yaml(path):
        from contractionRL.agents.utils.RL.config import load_agent_config
        return load_agent_config(path)

    def __init__(self, env, cfg):
        from contractionRL.runners.contraction_runner import _unwrap_env
        from contractionRL.tasks.direct.common.transfer_dynamics import is_transfer_path
        raw = _unwrap_env(env)
        if is_transfer_path(raw) and getattr(raw, '_dynamics_model', None) is None:
            raise RuntimeError('Prepare the transfer simulator before constructing the runner')
        from contractionRL.agents.utils.RL.config import configure_encoder_context, validate_observation_normalization
        import copy
        cfg = copy.deepcopy(cfg)
        if 'use_value_norm' in cfg.get('agent', {}):
            from contractionRL.agents.utils.RL.config import normalize_agent_cfg
            normalize_agent_cfg(cfg, algorithm=str(cfg['agent']['class']).lower(), device=getattr(env, 'device', None))
        configure_encoder_context(env, cfg)
        validate_observation_normalization(cfg.get("agent", {}))
        if hasattr(env, "set_discount_factor"):
            env.set_discount_factor(cfg.get("agent", {}).get("discount_factor", .99))
        # Select the policy distribution before model construction.
        import copy
        self._feature_cfg = copy.deepcopy(cfg)
        self._agent_class = str((cfg.get("agent") or {}).get("class", "")).strip()
        self._mbpo = self._agent_class.lower() == "mbpo"
        self._contractionppo = self._agent_class.lower().replace("_", "").replace("-", "") == "contractionppo"
        if self._mbpo:
            import copy
            cfg = copy.deepcopy(cfg)
            cfg["agent"]["class"] = "SAC"
        if self._contractionppo:
            import copy
            cfg = copy.deepcopy(cfg)
            value_norm = cfg["agent"].pop("use_value_norm", False)
            if value_norm:
                cfg["agent"]["value_preprocessor"] = "RunningStandardScaler"
                cfg["agent"]["value_preprocessor_kwargs"] = None
            cfg["agent"]["class"] = "PPO"  # Use skrl's PPO construction path with our subclass.
        super().__init__(env, cfg)

    def _generate_models(self, env, cfg):
        from contractionRL.agents.utils.RL.config import model_context_config
        cfg["models"] = model_context_config(cfg)
        cfg["models"].pop("dynamics", None)
        return super()._generate_models(env, cfg)

    def _generate_agent(self, env, cfg, models):
        for key in ('use_value_expansion','value_expansion_horizon','value_expansion_batch_size','action_bound_penalty'):
            cfg['agent'].pop(key, None)
        agent = super()._generate_agent(env, cfg, models)
        from contractionRL.runners.contraction_runner import _unwrap_env
        from contractionRL.tasks.direct.common.transfer_dynamics import attach_transfer_agent
        attach_transfer_agent(_unwrap_env(env), agent)
        from contractionRL.agents.utils.RL.agent_patches import patch_kl_logging
        patch_kl_logging(agent)
        if self._contractionppo or self._mbpo:
            agent.bind_env(env)
        if 'action_bound_penalty' in self._feature_cfg.get('agent', {}):
            from contractionRL.runners.contraction_runner import _unwrap_env
            raw = _unwrap_env(env)
            beta = self._feature_cfg['agent']['action_bound_penalty']
            if hasattr(raw, 'set_action_bound_penalty'):
                raw.set_action_bound_penalty(beta)
            elif beta:
                raise ValueError('Environment does not implement action_bound_penalty')
        if 'PPO' in self._agent_class.upper() and self._feature_cfg.get('agent', {}).get('use_value_expansion', False):
            from contractionRL.agents.utils.RL.ppo_features import PPOFeatures
            agent._ppo_features = PPOFeatures(agent, env, self._feature_cfg)
        return agent

    def _process_cfg(self, cfg):
        cfg = super()._process_cfg(cfg)
        from skrl.resources.preprocessors.torch import RunningStandardScaler as StockScaler
        if cfg.get("value_preprocessor") is StockScaler:
            from contractionRL.agents.utils.RL.env import RunningStandardScaler
            cfg["value_preprocessor"] = RunningStandardScaler
        return cfg

    def _component(self, name: str):
        if getattr(self, "_mbpo", False):
            from contractionRL.agents.skrl.mbpo import MBPOAgent, MBPOCfg
            if name.lower() == "sac": return MBPOAgent
            if name.lower() == "sac_cfg": return MBPOCfg
        from contractionRL.agents.skrl.contraction_ppo import ContractionPPOAgent, ContractionPPOCfg, ContractionPPOPolicy, ContractionPPOValue
        if name.lower() == "contractionppopolicy":
            return ContractionPPOPolicy
        if name.lower() == "contractionppovalue":
            return ContractionPPOValue
        if getattr(self, "_contractionppo", False):
            if name.lower() == "ppo": return ContractionPPOAgent
            if name.lower() == "ppo_cfg": return ContractionPPOCfg
        if name.lower() == "gaussianmixin":
            agent_class = getattr(self, "_agent_class", "")

            def _factory(*args, **kw):
                return _gaussian_factory(*args, agent_class=agent_class, **kw)

            return _factory
        if name.lower() == "deterministicmixin":
            return _deterministic_factory
        return super()._component(name)

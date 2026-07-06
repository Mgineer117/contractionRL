"""CLActorRunner — backbone-aware GaussianMixin factory.

YAML policy config:

    policy:
      class: GaussianMixin
      backbone: control           # CLActor (W2 @ tanh(W1 @ (x-x_ref))); alias: contraction
      clip_log_std: True
      min_log_std: -4.605
      max_log_std: 2.0
      network:                    # layers used for w1 and w2 MLPs
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: mlp               # standard skrl MLP (default)
      network:
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: control-squashed  # tanh-squashed CLActor (SquashedCLActorModel) — same
                                   # bilinear feedback architecture as "control", but bounded
                                   # + log_prob-corrected; requires the SAME [x, xref, uref]
                                   # layout as "control". See models.py's SquashedCLActorModel /
                                   # _TanhSquashMixin docstrings for why this is required for
                                   # SAC (skrl's stock GaussianMixin + clip_actions is NOT
                                   # equivalent — clip never reaches log_prob).
      clip_log_std: True
      min_log_std: -20.0
      max_log_std: 2.0
      network:
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: mlp-squashed      # tanh-squashed plain-MLP actor (SquashedGaussianActorModel).
                                   # Adds uref like "mlp" does IF the observation layout has one
                                   # ([x, xref, uref]); otherwise (e.g. velocity-tracking) it's a
                                   # plain squashed MLP over the full observation. State-dependent
                                   # log_std, so no initial_log_std/clip_actions (silently
                                   # ignored — squashing already bounds every sampled action).
      clip_log_std: True
      min_log_std: -20.0
      max_log_std: 2.0
      network:
        - name: net
          input: OBSERVATIONS
          layers: [256, 256]
          activations: relu
      output: ACTIONS

When backbone is ``contraction``, ``layers`` is extracted from the first
network entry and passed as ``hidden_dim`` to CLActorModel.  All other
MLP-specific keys (``output``, ``initial_log_std``) are dropped.
"""
from __future__ import annotations

from skrl.utils.runner.torch import Runner


# Backbone → action-distribution family. Unbounded backbones sample from a raw
# Normal (valid only for PPO-family trust-region methods); squashed backbones
# tanh-bound the action + correct log_prob (valid only for SAC-family off-policy
# entropy tuning). Mixing them across families silently mistrains — see
# _assert_backbone_algo_compatible.
_UNBOUNDED_BACKBONES = frozenset({"mlp", "control", "contraction"})
_SQUASHED_BACKBONES = frozenset({"mlp-squashed", "control-squashed"})


def _assert_backbone_algo_compatible(backbone: str, agent_class: str | None) -> None:
    """Raise on (algorithm, backbone) pairings that would silently mistrain.

    - SAC / C2RL-SAC need a BOUNDED (tanh-squashed) action distribution: an
      unbounded Gaussian's log_prob is unbounded below, so SAC's automatic
      entropy-coefficient tuning has no fixed point and diverges. Unbounded
      backbones (mlp / control / contraction) are rejected for SAC.
    - PPO / C2RL-PPO need a closed-form entropy (entropy bonus) and clean
      trust-region log_prob ratios; the squashed backbones deliberately raise
      from get_entropy() and have zero-gradient action boundaries, so they are
      rejected for PPO.

    Other agents (C3M, LQR, SD-LQR) are unconstrained — they don't build their
    policy through this factory with a squashed backbone and don't do SAC-style
    entropy tuning. ``agent_class=None`` also skips the check (nothing to
    validate against).
    """
    if not agent_class:
        return
    algo = agent_class.upper()
    is_sac = "SAC" in algo
    is_ppo = "PPO" in algo
    if is_sac and backbone in _UNBOUNDED_BACKBONES:
        suggested = "mlp-squashed" if backbone == "mlp" else "control-squashed"
        raise ValueError(
            f"backbone: {backbone!r} is an UNBOUNDED Gaussian and is not valid for "
            f"agent class {agent_class!r} (SAC family): SAC's automatic entropy tuning "
            f"has no fixed point with an unbounded log_prob and will diverge. Use "
            f"backbone: {suggested!r} (tanh-squashed, bounded) for SAC."
        )
    if is_ppo and backbone in _SQUASHED_BACKBONES:
        suggested = "mlp" if backbone == "mlp-squashed" else "control"
        raise ValueError(
            f"backbone: {backbone!r} is tanh-squashed and has no closed-form entropy "
            f"(get_entropy raises), so it is not valid for agent class {agent_class!r} "
            f"(PPO family), whose entropy bonus / analytic entropy require it. Use "
            f"backbone: {suggested!r} (unbounded Gaussian) for PPO."
        )


def _gaussian_factory(observation_space, state_space, action_space, device,
                       backbone: str = "mlp", agent_class: str | None = None, **kwargs):
    # Hard guard: reject algorithm/backbone pairs that silently mistrain
    # (unbounded + SAC → divergence; squashed + PPO → no analytic entropy).
    _assert_backbone_algo_compatible(backbone, agent_class)

    # "control" is the preferred spelling for the CLActor backbone;
    # "contraction" is kept as a backward-compatible alias.
    if backbone in ("control", "contraction"):
        import gymnasium
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # CLActorModel requires obs layout [x, x_ref, u_ref]: obs_dim == 2*x_dim + u_dim
        # and x_dim must be an integer
        remainder = obs_dim - act_dim
        if remainder <= 0 or remainder % 2 != 0:
            raise ValueError(
                f"backbone: contraction requires obs_dim = 2*x_dim + u_dim, but got "
                f"obs_dim={obs_dim}, act_dim={act_dim} (remainder={remainder} is not even). "
                f"Use backbone: mlp for velocity-tracking environments."
            )
        from contractionRL.agents.skrl.models import CLActorModel
        network = kwargs.pop("network", [{}])
        hidden_dim = network[0].get("layers", [128, 128]) if network else [128, 128]
        kwargs.pop("output", None)
        kwargs.pop("initial_log_std", None)

        # Try to extract x_dim if available
        x_dim = kwargs.pop("x_dim", None)

        return CLActorModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            **kwargs,
        )

    if backbone == "control-squashed":
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        remainder = obs_dim - act_dim
        if remainder <= 0 or remainder % 2 != 0:
            raise ValueError(
                f"backbone: control-squashed requires obs_dim = 2*x_dim + u_dim, but got "
                f"obs_dim={obs_dim}, act_dim={act_dim} (remainder={remainder} is not even). "
                f"Use backbone: mlp-squashed for velocity-tracking environments."
            )
        from contractionRL.agents.skrl.models import SquashedCLActorModel
        network = kwargs.pop("network", [{}])
        hidden_dim = network[0].get("layers", [128, 128]) if network else [128, 128]
        kwargs.pop("output", None)
        x_dim = kwargs.pop("x_dim", None)
        return SquashedCLActorModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            **kwargs,
        )

    if backbone == "mlp-squashed":
        from contractionRL.agents.skrl.models import SquashedGaussianActorModel
        network = kwargs.pop("network", [{}])
        hidden_dim = network[0].get("layers", [256, 256]) if network else [256, 256]
        activation = network[0].get("activations", "relu") if network else "relu"
        kwargs.pop("output", None)
        # log_std is state-dependent (network head) for this backbone, and
        # squashing already bounds every sampled action — these two yaml
        # keys (meaningful for the other backbones above) don't apply here.
        kwargs.pop("initial_log_std", None)
        kwargs.pop("clip_actions", None)
        return SquashedGaussianActorModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dim=hidden_dim,
            activation=activation,
            **kwargs,
        )

    if backbone == "mlp":
        # Path-tracking envs give obs = [x, xref, uref], i.e. obs_dim = 2*x_dim +
        # u_dim (remainder even) — same layout check as the "control" backbone
        # above. When it holds, "mlp" still adds uref to its output (mu = uref +
        # MLP(obs)), just via a plain MLP over the full observation instead of
        # CLActor's (x-xref)-only bilinear structure. Vel-tracking envs have no
        # uref (remainder is odd for every current env), so they fall through
        # to the stock stateless MLP below, unaffected.
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        remainder = obs_dim - act_dim
        if remainder > 0 and remainder % 2 == 0:
            from contractionRL.agents.skrl.models import MLPResidualActorModel
            network = kwargs.pop("network", [{}])
            hidden_dim = network[0].get("layers", [128, 128]) if network else [128, 128]
            kwargs.pop("output", None)
            initial_log_std = kwargs.pop("initial_log_std", 0.0)
            return MLPResidualActorModel(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                hidden_dim=hidden_dim,
                initial_log_std=initial_log_std,
                **kwargs,
            )

    from skrl.utils.model_instantiators.torch import gaussian_model
    kwargs.pop("angle_idx", None)
    return gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )


def _deterministic_factory(observation_space, state_space, action_space, device, **kwargs):
    """DeterministicMixin (value/critic) factory — the EmbeddedDeterministicModel
    counterpart to ``_gaussian_factory`` above.

    Value/critic networks see the SAME observation as the policy (including any
    angle-bearing x/xref blocks), so they need the same continuous embedding —
    skrl's stock ``deterministic_model`` instantiator has no notion of this
    (it's vendored library code). This factory reads the same yaml ``network``/
    ``angle_idx`` keys the policy factories do and builds
    ``EmbeddedDeterministicModel`` instead. When ``angle_idx`` is empty (the
    common case for envs with no wrapping angle in their state), this reduces
    to a plain MLP over the observation (+ actions, for a Q-model) — the same
    architecture stock ``deterministic_model`` would have built.
    """
    from contractionRL.agents.skrl.models import EmbeddedDeterministicModel

    network = kwargs.pop("network", [{}])
    kwargs.pop("output", None)
    net_spec = network[0] if network else {}
    use_actions = "ACTIONS" in str(net_spec.get("input", "OBSERVATIONS"))
    hidden_dim = net_spec.get("layers", [256, 256])
    activation = net_spec.get("activations", "tanh")

    return EmbeddedDeterministicModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        hidden_dim=hidden_dim,
        activation=activation,
        use_actions=use_actions,
        **kwargs,
    )


class CLActorRunner(Runner):
    def __init__(self, env, cfg):
        # Capture the agent class up front so the gaussian factory can reject
        # (algorithm, backbone) combinations that silently mistrain (see
        # _assert_backbone_algo_compatible). Set before super().__init__ because
        # that's what triggers model construction via _component below.
        self._agent_class = str((cfg.get("agent") or {}).get("class", "")).strip()
        super().__init__(env, cfg)

    def _component(self, name: str):
        if name.lower() == "gaussianmixin":
            agent_class = getattr(self, "_agent_class", "")

            def _factory(*args, **kw):
                return _gaussian_factory(*args, agent_class=agent_class, **kw)

            return _factory
        if name.lower() == "deterministicmixin":
            return _deterministic_factory
        return super()._component(name)

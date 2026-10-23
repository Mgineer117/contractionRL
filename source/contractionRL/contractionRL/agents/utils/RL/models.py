"""Actors and critics for compact tracking and legacy reference-window observations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from contractionRL.agents.utils.contraction.math_utils import rescale_residual
from contractionRL.agents.utils.RL.networks import MLP, CLActor, PreviewSequenceEncoder, _ACT_MAP
from contractionRL.agents.utils.contraction.metric_models import BoundedCCM_Generator, CCM_Generator
from contractionRL.agents.utils.system_dynamics.reference import Feats, RefWindow
from contractionRL.agents.utils.system_dynamics.state import StateSymmetry

_MIN_LOG_STD = math.log(0.001)  # Positive exploration standard-deviation floor.


def _layout(observation_space, x_dim=None, angle_idx=None, sym=None, ref_offset=1):
    """Read the declared observation layout and validate the caller's state dimension."""
    window = RefWindow.from_space(observation_space, offset=ref_offset)
    if x_dim is not None and int(x_dim) != window.x_dim:
        raise ValueError(
            f"x_dim mismatch: caller says {int(x_dim)}, observation space declares "
            f"{window.x_dim}. The env and the model disagree about the state layout.")
    return window, Feats(window.x_dim, sorted(set(angle_idx or []) | set(getattr(window, "angle_idx", ()))), sym)


class _AnalyticPotentialMixin:
    """Critic potential V(s) = error_energy(s) + learned_residual(s).

    The frozen metric supplies the known shaping potential. This parameterization
    requires value normalization to be disabled.
    """

    def _init_potential(self, window, feats, w_lb):
        self._pot_ccm_gen = None          # set by C2RLAgent._attach_critic_potential
        self._pot_window = window
        self._pot_feats = feats
        self._pot_w_lb = float(w_lb)

    def _analytic_potential(self, inputs):
        """``||e||^2_M`` from the observation, where ``e = x - xrefs[0]``."""
        gen = getattr(self, "_pot_ccm_gen", None)
        obs = inputs.get("observations")
        if gen is None or obs is None:
            return None
        from contractionRL.agents.utils.contraction.math_utils import bound_W, spd_inverse, tracking_energy
        x, xrefs, _ = self._pot_window.split(obs)
        with torch.no_grad():
            e = x - xrefs[:, 0]
            W = bound_W(gen(x)[0], self._pot_w_lb, self._pot_window.x_dim,
                        getattr(gen, "bounded", False))
            M = W if getattr(gen, "outputs_metric", False) else spd_inverse(W)
            return tracking_energy(e, M, self._pot_feats.angle_idx).unsqueeze(-1)


# Critic

class RefWindowValueModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
    """Legacy reference-window critic combining state/error and reference embeddings.

    With use_actions=True, append the action to produce a SAC Q-function.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        network: list | None = None,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        use_actions: bool = False,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        combine: str = "concat",
        analytic_potential: bool = False,
        w_lb: float = 0.01,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        net_spec = (network or [{}])[0] if network else {}
        hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
        activation = net_spec.get("activations", activation)
        act_module = _ACT_MAP[activation.lower()]() \
            if isinstance(activation, str) else activation

        if combine != "concat":
            raise ValueError(
                f"RefWindowValueModel requires combine='concat', got {combine!r}")
        self.combine = combine

        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        act_dim = int(self.action_space.shape[0])
        self._use_actions = use_actions

        self.phi = MLP(self.feats.single_dim + self.feats.error_dim, list(hidden_dim),
                       encoder_hidden, activation=act_module)
        self.psi = PreviewSequenceEncoder(
            preview_dim=self.window.length * self.feats.pair_dim,
            point_dim=self.feats.pair_dim,
            hidden=encoder_hidden, out_dim=encoder_hidden,
            mode=encoder, stride=encoder_stride)
        self.net = MLP(2 * encoder_hidden + (act_dim if use_actions else 0),
                       list(hidden_dim), 1, activation=act_module)

        self.use_analytic_potential = bool(analytic_potential)
        self._init_potential(self.window, self.feats, w_lb)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "value"):
        obs = inputs["observations"]
        n = obs.shape[0]
        x, xrefs, _ = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        phi = self.phi(torch.cat([self.feats.single(x), e], dim=-1))
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        parts = [phi, psi]
        if self._use_actions:
            parts.append(inputs["taken_actions"])
        value = self.net(torch.cat(parts, dim=-1))
        if self.use_analytic_potential:
            pot = self._analytic_potential(inputs)
            if pot is not None:
                value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
        return value, {}


# Back-compat aliases: the runner/config layer still refers to the critic by its
# old names. One class serves both roles now (the privileged `states` channel is
# gone — the critic reads the same observation as the actor).
EmbeddedDeterministicModel = RefWindowValueModel
TrajectoryAwareValueModel = RefWindowValueModel


# Metric generator

def _observation_scaling(model, space, device):
    from gymnasium.spaces import flatten_space
    from contractionRL.agents.utils.system_dynamics.reference import affine_observation_scaling
    box = flatten_space(space)
    center, scale = affine_observation_scaling(box.low, box.high)
    model.register_buffer("obs_center", center.to(device))
    model.register_buffer("obs_scale", scale.to(device))


class FullControlActorModel(GaussianMixin, Model):
    """Compact tracking actor with structured, factorized, or plain MLP control."""
    def __init__(self, observation_space, action_space, device, hidden_dim=None,
                 activation="tanh", angle_idx=(), initial_log_std=0.0,
                 min_log_std=-6.908, max_log_std=2.0, control_mode="structured",
                 clip_actions=False, clip_log_std=True, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        if clip_actions:
            raise ValueError("Clip applied controls in the environment; PPO must score raw Gaussian samples")
        if activation.lower() not in _ACT_MAP:
            raise ValueError(f"Unsupported activation: {activation}")
        preview = getattr(RefWindow.from_space(observation_space), 'preview_length', 0)
        if not preview and (kwargs.get("encoder", "mlp") != "mlp" or kwargs.get("encoder_stride", 1) != 1):
            raise ValueError("Compact sequence encoders require an explicit reference preview")
        temporal = getattr(RefWindow.from_space(observation_space), 'temporal_context', False)
        # PPO scores/stores the raw Gaussian sample; the plant clips applied controls.
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        if control_mode not in ("full", "residual", "structured", "factorized"):
            raise ValueError(f"Unknown policy control mode: {control_mode}")
        self.control_mode = control_mode
        self.window = RefWindow.from_space(observation_space)
        self.angle_idx = list(angle_idx)
        _observation_scaling(self, observation_space, device)
        if control_mode == "structured":
            _, feats = _layout(observation_space, angle_idx=angle_idx, sym=kwargs.get("sym"))
            self.cl_actor = CLActor(self.window, feats, hidden_dim=hidden_dim, activation=activation,
                                   encoder=kwargs.get('encoder', 'mlp'), encoder_hidden=kwargs.get('encoder_hidden', 64),
                                   encoder_stride=kwargs.get('encoder_stride', 1), feature_dim=kwargs.get('feature_dim'))
            self.cl_actor.logstd.requires_grad_(False)
        else:
            from contractionRL.agents.utils.system_dynamics.reference import TrackingFeatures
            if kwargs.get('encoder', 'mlp') == 'gru':
                from contractionRL.agents.utils.RL.networks import TemporalTrackingFeatures
                self.tracking_features = TemporalTrackingFeatures(self.window, self.angle_idx,
                    encoder=kwargs.get('encoder', 'mlp'), hidden=kwargs.get('encoder_hidden', 64),
                    stride=kwargs.get('encoder_stride', 1), feature_dim=kwargs.get('feature_dim'))
            else:
                self.tracking_features = TrackingFeatures(self.window, self.angle_idx)
            self.register_buffer("_uses_tracking_features", torch.tensor(True))
            hidden = list(hidden_dim or [128, 128])
            if control_mode == "factorized":
                self.latent_dim = 3 * self.window.x_dim
                self.w1 = MLP(self.tracking_features.output_dim, hidden, self.latent_dim,
                              activation=_ACT_MAP[activation.lower()]())
                self.w2 = MLP(self.tracking_features.output_dim, hidden,
                              self.window.u_dim * self.latent_dim, activation=_ACT_MAP[activation.lower()]())
            else:
                self.net = MLP(self.tracking_features.output_dim, hidden, action_space.shape[0],
                               activation=_ACT_MAP[activation.lower()]())
        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(initial_log_std)))
        self.register_buffer("control_low", torch.as_tensor(action_space.low, device=device))
        self.register_buffer("control_high", torch.as_tensor(action_space.high, device=device))
        self.to(device)

    def features(self, obs):
        if not bool(self._uses_tracking_features):
            return (obs - self.obs_center) / self.obs_scale
        return self.tracking_features(obs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.control_mode != "structured" and "_uses_tracking_features" not in state_dict:
            raise ValueError("Checkpoint predates periodic tracking features; load it with its archived source snapshot")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def compute(self, inputs, role="policy"):
        if self.control_mode == "structured":
            mean = self.cl_actor.mean_control(inputs["observations"])
            return mean, {"log_std": self.log_std_parameter}
        if self.control_mode == "factorized":
            features = self.features(inputs["observations"])
            weights = self.w2(features).reshape(-1, self.window.u_dim, self.latent_dim)
            mean = (weights @ self.w1(features).tanh().unsqueeze(-1)).squeeze(-1)
            return mean, {"log_std": self.log_std_parameter}
        mean = self.net(self.features(inputs["observations"]))
        mean = (self.control_low + self.control_high) / 2 + (self.control_high - self.control_low) / 2 * mean.tanh()
        if self.control_mode == "residual":
            mean = mean + self.window.split(inputs["observations"])[2][:, 0]
        return mean, {"log_std": self.log_std_parameter}


class FullControlValueModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
    """Unstructured MLP V or Q over the compact tracking state."""
    def __init__(self, observation_space, action_space, device, hidden_dim=None,
                 activation="tanh", use_actions=False, analytic_potential=False, w_lb=.01, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        preview = getattr(RefWindow.from_space(observation_space), 'preview_length', 0)
        if (not preview and (kwargs.get("encoder", "mlp") != "mlp" or kwargs.get("encoder_stride", 1) != 1
                or kwargs.get("combine", "concat") != "concat" or kwargs.get("embed_dim", 64) != 64
                or kwargs.get("encoder_hidden", 64) != 64)):
            raise ValueError("Compact critics use a plain MLP; sequence/embedding options require the legacy window layout")
        DeterministicMixin.__init__(self, clip_actions=False)
        self.window = RefWindow.from_space(observation_space)
        self.use_actions = use_actions
        _observation_scaling(self, observation_space, device)
        from contractionRL.agents.utils.system_dynamics.reference import TrackingFeatures
        if kwargs.get('encoder', 'mlp') == 'gru':
            from contractionRL.agents.utils.RL.networks import TemporalTrackingFeatures
            self.tracking_features = TemporalTrackingFeatures(self.window, kwargs.get('angle_idx', ()),
                encoder=kwargs.get('encoder', 'mlp'), hidden=kwargs.get('encoder_hidden', 64),
                stride=kwargs.get('encoder_stride', 1), feature_dim=kwargs.get('feature_dim'))
        else:
            self.tracking_features = TrackingFeatures(self.window, kwargs.get("angle_idx", ()))
        self.use_analytic_potential = bool(analytic_potential)
        self._init_potential(self.window, self.tracking_features, w_lb)
        self.net = MLP(self.tracking_features.output_dim + (action_space.shape[0] if use_actions else 0),
                       list(hidden_dim or [128, 128]), 1,
                       activation=_ACT_MAP[activation.lower()]())
        self.to(device)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Earlier compact critics saved these shared feature buffers only once.
        state_dict = dict(state_dict)
        for key, value in list(state_dict.items()):
            if key.startswith("tracking_features."):
                state_dict.setdefault("_pot_feats." + key.removeprefix("tracking_features."), value)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def compute(self, inputs, role="value"):
        obs = self.tracking_features(inputs["observations"])
        if self.use_actions:
            obs = torch.cat((obs, inputs["taken_actions"]), -1)
        value = self.net(obs)
        if self.use_analytic_potential:
            potential = self._analytic_potential(inputs)
            if potential is not None:
                value = value + potential
        return value, {}


class MetricModel(Model):
    """CCM_Generator wrapped as a skrl Model for checkpointing.

    The underlying CCM_Generator is accessed via ``self.ccm_gen`` by C3MAgent
    and C2RLAgent for the contraction loss computation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mode: str = "deterministic",
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        **kwargs,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        self.x_dim, self.u_dim = self.window.x_dim, self.window.u_dim
        relative_mask = torch.zeros(self.x_dim, dtype=torch.bool, device=self.device)
        # Retain the zero mask solely to reject incompatible historical checkpoints.
        self.register_buffer('_relative_position_mask', relative_mask)

        common = dict(x_dim=self.x_dim, hidden_dim=hidden_dim or [256, 256],
                      activation=activation, mode=mode,
                      device=str(device) if not isinstance(device, str) else device,
                      angle_idx=list(angle_idx or []), sym=sym)
        if kwargs.get("constrain_eigenvalues", False):
            # outputs_metric: C2RL's CMG emits M directly (its SDP
            # targets can be inverted once offline, and the reward wants M), so
            # every env step drops a batched SPD inverse. C2RL's direct CCM
            # losses also use M; C3M keeps the W-emitting default.
            self.ccm_gen = BoundedCCM_Generator(
                w_lb=kwargs.get("w_lb", 0.1), w_ub=kwargs.get("w_ub", 10.0),
                outputs_metric=bool(kwargs.get("outputs_metric", False)), **common)
        else:
            self.ccm_gen = CCM_Generator(**common)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + '_relative_position_mask'
        recorded = state_dict.get(key, torch.zeros_like(self._relative_position_mask))
        if not torch.equal(recorded.to(self._relative_position_mask), self._relative_position_mask):
            raise ValueError('Checkpoint metric coordinates differ; load it with its archived source snapshot')
        state_dict.setdefault(key, torch.zeros_like(self._relative_position_mask))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def compute(self, inputs: dict, role: str = "cmg"):
        x, _, _ = self.window.split(inputs["observations"])
        W, _ = self.ccm_gen(x)
        return W.reshape(W.shape[0], -1), {}

    def act(self, inputs: dict, role: str = "cmg"):
        output, extra = self.compute(inputs, role)
        return output, None, extra

    def forward(self, inputs: dict, role: str = "cmg"):
        output, _ = self.compute(inputs, role)
        return output


# Actors — control backbone (CLActor)

def _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                    hidden_dim, activation,
                    encoder, encoder_hidden, encoder_stride, feature_dim=None):
    window, feats = _layout(observation_space, x_dim, angle_idx, sym)
    return CLActor(window=window, feats=feats,
                   hidden_dim=hidden_dim or [128, 128], activation=activation,
                   encoder=encoder, encoder_hidden=encoder_hidden,
                   encoder_stride=encoder_stride, feature_dim=feature_dim)


class CLDeterministicActorModel(DeterministicMixin, Model):
    """CLActor as a skrl Deterministic policy — C3M's default policy class
    (``class: DeterministicMixin`` in each ``skrl_c3m_cfg.yaml``). Unbounded;
    see ``SquashedCLDeterministicActorModel`` for ``backbone: control-squashed``.
    """

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False,
                 hidden_dim: list | None = None, activation: str = "tanh",
                 x_dim: int | None = None, angle_idx: list | None = None,
                 sym: StateSymmetry | None = None, encoder: str = "mlp",
                 encoder_hidden: int = 64, encoder_stride: int = 1, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation,
                                        encoder, encoder_hidden, encoder_stride, kwargs.get('feature_dim'))
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        return self.cl_actor.mean_control(inputs["observations"]), {}


class SquashedCLDeterministicActorModel(DeterministicMixin, Model):
    """Tanh-squashed CLActor, deterministic — ``backbone: control-squashed`` for C3M.

    ``compute()`` returns ``cl_actor.mean_control_squashed`` (feedback
    tanh-bounded into the action space, ``urefs[0]`` added after squashing).

    C3M needs this rather than a hard ``clip_actions`` clamp: ``torch.clamp``'s
    gradient is exactly zero at saturation, so ``K = jacobian(u, x)`` collapses
    there too and the certificate silently degrades to checking open-Loop drift
    in exactly the states that most need feedback — a confirmed real AUC
    divergence. ``tanh``'s gradient never hits exact zero.
    """

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False,
                 hidden_dim: list | None = None, activation: str = "tanh",
                 x_dim: int | None = None, angle_idx: list | None = None,
                 sym: StateSymmetry | None = None, encoder: str = "mlp",
                 encoder_hidden: int = 64, encoder_stride: int = 1, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation,
                                        encoder, encoder_hidden, encoder_stride, kwargs.get('feature_dim'))
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        return self.cl_actor.mean_control_squashed(
            inputs["observations"], self._d_min_actions, self._d_max_actions), {}


class CLActorModel(GaussianMixin, Model):
    """CLActor as a skrl Gaussian policy — ``backbone: control``.

    ``u = urefs[0] + W2(xrefs) @ tanh(W1(x) @ e)``. Unbounded; see
    ``SquashedCLActorModel`` for ``backbone: control-squashed``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = _MIN_LOG_STD,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation,
                                        encoder, encoder_hidden, encoder_stride, kwargs.get('feature_dim'))
        self.log_std_parameter = self.cl_actor.logstd

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        # u = urefs[0] + pi(s). The analytic-base warm-start (u = cvstem-lqr + pi)
        # has been removed, so the reference feedforward is the only base.
        feedback, uref = self.cl_actor._feedback(state)
        return uref + feedback, {"log_std": self.log_std_parameter}


# Actors — plain-MLP backbone

class MLPResidualActorModel(GaussianMixin, Model):
    """``backbone: mlp`` — ``mu = urefs[0] + MLP([single(x), e, psi(xrefs)])``.

    Same control law as CLActor, but one generic MLP over the whole (encoded)
    observation instead of the bilinear W1/W2 factorization — the deliberate
    architectural contrast. Reuses the same ``Feats`` maps and sequence encoder,
    so the two backbones differ only in how they combine the same inputs.

    Log-prob stays correct: the uref shift happens inside ``compute()``, so the
    mean GaussianMixin samples from already includes it.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = _MIN_LOG_STD,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        u_dim = self.window.u_dim
        act_module = _ACT_MAP[activation.lower()]() \
            if isinstance(activation, str) else activation
        self.psi = PreviewSequenceEncoder(
            preview_dim=self.window.length * self.feats.pair_dim,
            point_dim=self.feats.pair_dim, hidden=encoder_hidden, out_dim=encoder_hidden,
            mode=encoder, stride=encoder_stride)
        self.net = MLP(self.feats.single_dim + self.feats.error_dim + encoder_hidden,
                       list(hidden_dim or [128, 128]), u_dim, activation=act_module)
        self.log_std_parameter = nn.Parameter(torch.full((u_dim,), float(initial_log_std)))
        self.to(self.device)

    def _net_in(self, obs):
        n = obs.shape[0]
        x, xrefs, urefs = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        return torch.cat([self.feats.single(x), e, psi], dim=-1), urefs[:, 0]

    def compute(self, inputs: dict, role: str = "policy"):
        net_in, uref = self._net_in(inputs["observations"])
        return uref + self.net(net_in), {"log_std": self.log_std_parameter}


class _TanhSquashMixin:
    """Bounded Gaussian actions with the tanh change-of-variables correction.

    Initialize Model and GaussianMixin, then call _init_tanh_squash_bounds().
    compute() supplies the pre-squash mean, log_std and optional reference residual.
    Add the reference after squashing so zero feedback preserves feedforward.

    rescale_residual uses separate positive/negative half-scales around the reference;
    include the selected half-scale in log_prob. The stable tanh Jacobian is
    2 * (log(2) - u - softplus(-2*u)); see Haarnoja et al. (2018), Eq. 21.

    Use sampled log_prob for SAC entropy. Squashed entropy has no closed form,
    so this mixin does not support an analytic PPO entropy bonus.
    """

    _RESCALE_EPS = 1e-6  # keeps half-scales/atanh args away from 0 / +-1 (log(0)/atanh divergence)

    def _init_tanh_squash_bounds(self) -> None:
        if self._g_min_actions is None or self._g_max_actions is None:
            raise ValueError(
                f"{type(self).__name__} requires a fully-bounded action space "
                "(every dimension needs a finite low/high) — tanh-squashing has "
                "nothing to rescale into otherwise."
            )
        self.register_buffer("_action_low", self._g_min_actions.clone())
        self.register_buffer("_action_high", self._g_max_actions.clone())
        if (not torch.isfinite(self._action_low).all() or not torch.isfinite(self._action_high).all()
                or not (self._action_low < self._action_high).all()):
            raise ValueError("Squashed controls need finite, strictly ordered action bounds")

    def _rescale(self, tanh_u: torch.Tensor) -> torch.Tensor:
        """(-1, 1) -> [low, high]."""
        return self._action_low + 0.5 * (tanh_u + 1.0) * (self._action_high - self._action_low)

    def _unrescale(self, action: torch.Tensor) -> torch.Tensor:
        """[low, high] -> (-1, 1), clamped away from the boundary (atanh diverges there)."""
        frac = (action - self._action_low) / (self._action_high - self._action_low)
        return torch.clamp(2.0 * frac - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)

    def _rescale_residual(self, tanh_u: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(-1, 1) -> (low, high) via an asymmetric, residual-centered rescale.

        Returns ``(action, scale)`` where ``scale`` is the per-sample,
        per-side half-width used (needed for the log-det correction in
        ``act()``). See the class docstring for why this is exact and
        clamp-free, unlike a fixed-scale rescale-then-add. Delegates to
        ``math_utils.rescale_residual`` — shared with ``CLActor.mean_control_squashed``.
        """
        return rescale_residual(tanh_u, residual, self._action_low, self._action_high)

    def _unrescale_residual(self, action: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of ``_rescale_residual``: action -> (tanh_u in (-1, 1), scale)."""
        feedback = action - residual
        s_hi = torch.clamp(self._action_high - residual, min=self._RESCALE_EPS)
        s_lo = torch.clamp(residual - self._action_low, min=self._RESCALE_EPS)
        scale = torch.where(feedback >= 0, s_hi, s_lo)
        tanh_u = torch.clamp(feedback / scale, -1.0 + self._RESCALE_EPS, 1.0 - self._RESCALE_EPS)
        return tanh_u, scale

    def act(self, inputs: dict, *, role: str = "") -> tuple[torch.Tensor, dict]:
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        self._g_distribution = Normal(mean, log_std.exp())

        # Optional post-squash residual (e.g. uref for the [x, xref, uref]
        # path-tracking layout). Added to the action after squashing — see the
        # class docstring for why (residual law preservation + exact,
        # clamp-free log_prob). Popped so it doesn't leak into the returned
        # outputs dict.
        residual = outputs.pop("residual", None)

        taken_actions = inputs.get("taken_actions")
        if taken_actions is not None:
            # Recompute log_prob for an already-taken (post-squash, post-rescale,
            # post-residual) action — e.g. an on-policy update replaying stored
            # actions. SAC itself never hits this path (it always samples fresh).
            actions = taken_actions
            if residual is None:
                tanh_u = self._unrescale(actions)
                scale = 0.5 * (self._action_high - self._action_low)
            else:
                tanh_u, scale = self._unrescale_residual(actions, residual)
            u = torch.atanh(tanh_u)
        else:
            u = self._g_distribution.rsample()
            tanh_u = torch.tanh(u)
            if residual is None:
                actions = self._rescale(tanh_u)
                scale = 0.5 * (self._action_high - self._action_low)
            else:
                actions, scale = self._rescale_residual(tanh_u, residual)

        log_prob = self._g_distribution.log_prob(u)
        # Change-of-variables correction for y = tanh(u), stable form of
        # log(1 - tanh(u)^2) (Haarnoja et al. 2018, eq. 21).
        log_prob = log_prob - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        # Second correction for the rescale tanh_u -> action: d(action)/d(tanh_u)
        # is `scale` per dimension — a constant half-width (0.5*(high-low)) with
        # no residual, or the residual-dependent half-width from
        # _rescale_residual/_unrescale_residual above. Either way this is exact
        # for the action actually returned below (no clamping follows).
        log_prob = log_prob - torch.log(scale)

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        tanh_mean = torch.tanh(mean)
        mean_action = self._rescale(tanh_mean) if residual is None else self._rescale_residual(tanh_mean, residual)[0]

        outputs["mean_actions"] = mean_action
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form entropy (the squashed "
            "distribution isn't Gaussian) — it must only be used with algorithms "
            "that rely on the sampled log_prob (SAC), not an analytic entropy "
            "bonus (e.g. PPO's entropy_loss_scale)."
        )



class FullControlSquashedActorModel(_TanhSquashMixin, FullControlActorModel):
    """SAC MLP with a correctly normalized bounded full-control distribution."""
    def __init__(self, *args, **kwargs):
        FullControlActorModel.__init__(self, *args, **kwargs)
        if self.control_mode == "residual":
            raise ValueError("Residual compact policies are supported for PPO only")
        self._init_tanh_squash_bounds()

    def compute(self, inputs, role="policy"):
        if self.control_mode == "structured":
            feedback, uref = self.cl_actor._feedback(inputs["observations"])
            return feedback, {"log_std": self.log_std_parameter, "residual": uref}
        if self.control_mode == "factorized":
            return FullControlActorModel.compute(self, inputs, role)
        return self.net(self.features(inputs["observations"])), {"log_std": self.log_std_parameter}


class SquashedMLPActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed plain-MLP actor — ``backbone: mlp-squashed``.

    ``action = urefs[0] + rescale(tanh(MLP(...) + noise))``, uref added after
    squashing (see ``_TanhSquashMixin`` — before tanh would saturate uref).

    log_std is state-Dependent (the network outputs both), the SAC convention —
    unlike the other actors' single global log_std_parameter, which is fine for
    PPO's trust-region updates but not for SAC's entropy tuning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        hidden_dim: list | None = None,
        activation: str = "relu",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self._init_tanh_squash_bounds()

        act_dim = int(self.action_space.shape[0])
        act_module = _ACT_MAP[activation.lower()]() \
            if isinstance(activation, str) else activation
        if hasattr(observation_space, "spaces") and {"x", "xrefs", "urefs"} <= set(observation_space.spaces):
            self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
            self.psi = PreviewSequenceEncoder(
                preview_dim=self.window.length * self.feats.pair_dim,
                point_dim=self.feats.pair_dim, hidden=encoder_hidden, out_dim=encoder_hidden,
                mode=encoder, stride=encoder_stride)
            input_dim = self.feats.single_dim + self.feats.error_dim + encoder_hidden
        else:
            self.window = None
            if encoder != "mlp" or encoder_stride != 1:
                raise ValueError("Sequence encoders require a tracking reference window")
            input_dim = self.num_observations
        self.net = MLP(input_dim,
                       list(hidden_dim or [256, 256]), output_dim=None, activation=act_module)
        trunk_dim = self.net.output_dim
        self.mean_head = nn.Linear(trunk_dim, act_dim)
        self.log_std_head = nn.Linear(trunk_dim, act_dim)
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        self.mean_head.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        self.log_std_head.bias.data.fill_(0.0)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        obs = inputs["observations"]
        if self.window is None:
            features = self.net(obs)
            return self.mean_head(features), {"log_std": self.log_std_head(features)}
        n = obs.shape[0]
        x, xrefs, urefs = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        features = self.net(torch.cat([self.feats.single(x), e, psi], dim=-1))
        mean = self.mean_head(features)  # pre-squash feedback mean (no uref)
        # urefs[0] is added after squashing (the mixin consumes "residual"), so
        # the action is uref + rescale(tanh(feedback)).
        return mean, {"log_std": self.log_std_head(features), "residual": urefs[:, 0]}


class SquashedCLActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed CLActor — ``backbone: control-squashed``.

    ``action = urefs[0] + rescale(tanh(feedback + noise))`` — the feedback is
    squashed, uref added after, preserving ``u = uref + feedback`` exactly.

    SAC learns log_std through its policy loss and entropy tuning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self._init_tanh_squash_bounds()
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation,
                                        encoder, encoder_hidden, encoder_stride, kwargs.get('feature_dim'))
        self.log_std_parameter = self.cl_actor.logstd
        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        feedback, uref = self.cl_actor._feedback(inputs["observations"])
        return feedback, {"log_std": self.log_std_parameter, "residual": uref}

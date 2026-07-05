"""skrl-compatible model wrappers for contractionRL custom actors.

CLActorModel: wraps CLActor (C3M_U contracting controller) in the skrl
GaussianMixin interface so it can be passed to skrl PPO / C2RL runners.

Observation layout assumed: [x (x_dim), xref (x_dim), uref (u_dim)]
  → obs_dim = 2*x_dim + u_dim  →  x_dim = (obs_dim - u_dim) / 2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
except ImportError:
    raise ImportError("skrl is required. Install it or use the local developer copy.")

from .nn_modules import CCM_Generator, BoundedCCM_Generator, CLActor, MLP, NeuralDynamics

_MIN_LOG_STD = math.log(0.01)  # ≈ -4.605; matches CLActor annealing floor


class CLDeterministicActorModel(DeterministicMixin, Model):
    """Contracting C3M_U actor wrapped as a skrl Deterministic policy model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            mode="deterministic",
            anneal_stddev=False,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
        )

        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        return self.cl_actor.mean_control(state), {}


class CMGModel(Model):
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
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        self.obs_dim = int(observation_space.shape[0])
        self.u_dim = int(action_space.shape[0])
        if x_dim is None:
            x_dim = (self.obs_dim - self.u_dim) // 2
        self.x_dim = x_dim

        constrain_eigenvalues = kwargs.get("constrain_eigenvalues", False)
        
        if constrain_eigenvalues:
            w_lb = kwargs.get("w_lb", 0.1)
            w_ub = kwargs.get("w_ub", 10.0)
            self.ccm_gen = BoundedCCM_Generator(
                x_dim=x_dim,
                hidden_dim=hidden_dim or [256, 256],
                activation=activation,
                mode=mode,
                w_lb=w_lb,
                w_ub=w_ub,
                device=str(device) if not isinstance(device, str) else device,
            )
        else:
            self.ccm_gen = CCM_Generator(
                x_dim=x_dim,
                hidden_dim=hidden_dim or [256, 256],
                activation=activation,
                mode=mode,
                device=str(device) if not isinstance(device, str) else device,
            )

    def compute(self, inputs: dict, role: str = "cmg"):
        x = inputs["observations"][:, : self.x_dim]
        W, info = self.ccm_gen(x)
        return W.reshape(W.shape[0], -1), {}

    def act(self, inputs: dict, role: str = "cmg"):
        output, extra = self.compute(inputs, role)
        return output, None, extra

    def forward(self, inputs: dict, role: str = "cmg"):
        output, _ = self.compute(inputs, role)
        return output


class CLActorModel(GaussianMixin, Model):
    """Contracting C3M_U actor wrapped as a skrl Gaussian policy model.

    Args:
        observation_space: gymnasium observation space ([x, xref, uref]).
        action_space:      gymnasium action space (u).
        device:            torch device.
        clip_actions:      whether to clip sampled actions to action space bounds.
        clip_log_std:      whether to clip log_std to [min_log_std, max_log_std].
        min_log_std:       lower clip bound for log_std.
        max_log_std:       upper clip bound for log_std.
        initial_log_std:   initial value for the global log_std parameter.
        hidden_dim:        hidden layer sizes for the CLActor weight generators.
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
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            mode="stochastic",
            anneal_stddev=True,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
        )
        self.log_std_parameter = self.cl_actor.logstd

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)

        # Model.__init__ moves to self.device before cl_actor exists; re-sync here.
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        mean = self.cl_actor.mean_control(state)
        return mean, {"log_std": self.log_std_parameter}


class MLPResidualActorModel(GaussianMixin, Model):
    """Plain-MLP actor whose output is a residual added to u_ref: mu = uref + MLP(obs).

    Unlike CLActorModel (a specific bilinear w1/w2 architecture that only sees
    (x - xref)), this runs a standard MLP over the FULL observation
    [x, xref, uref] and adds uref to its raw output — same "u = uref +
    feedback" control law as CLActor, just a more generic architecture/inductive
    bias. This is what ``backbone: mlp`` resolves to for path-tracking envs
    (see runner.py's ``_gaussian_factory``).

    Log-prob stays correct for PPO/SAC: the uref shift happens INSIDE
    ``compute()``, so the mean GaussianMixin samples from and computes
    log_prob against already includes it — nothing shifts the action *after*
    it's sampled, so there's no mismatch between the logged action/log_prob
    and the action actually applied to the environment.
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
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        remainder = obs_dim - u_dim
        if remainder <= 0 or remainder % 2 != 0:
            raise ValueError(
                f"MLPResidualActorModel requires obs_dim = 2*x_dim + u_dim (path-tracking "
                f"layout [x, xref, uref]), got obs_dim={obs_dim}, act_dim={u_dim}."
            )
        self._u_dim = u_dim

        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        self.net = MLP(obs_dim, list(hidden_dim or [128, 128]), u_dim, activation=act_module)

        self.log_std_parameter = nn.Parameter(torch.full((u_dim,), float(initial_log_std)))

        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        obs = inputs["observations"]
        uref = obs[:, -self._u_dim:]  # last u_dim entries of [x, xref, uref]
        mean = uref + self.net(obs)
        return mean, {"log_std": self.log_std_parameter}


class _TanhSquashMixin:
    """Shared tanh-squash act()/log_prob machinery for bounded-action Gaussian actors.

    Factored out so every squashed backbone (plain MLP, CLActor-bilinear, ...)
    gets the EXACT same log_prob correction — this math is easy to get subtly
    wrong, so it lives in exactly one place rather than being copy-pasted per
    backbone.

    Necessary for SAC (and any off-policy method with an entropy term) on a
    bounded action space. skrl's stock ``GaussianMixin.act()`` samples from
    an UNBOUNDED ``Normal(mean, std)`` and, if ``clip_actions`` is set, only
    hard-clamps the sample afterwards — ``log_prob`` is still computed from
    the unclamped ``Normal``, so the clamp never reaches it. SAC's entropy
    term ``-alpha * log_prob`` appears in both the Bellman target and the
    policy loss, so an unbounded log_prob (which grows without limit as
    log_std shrinks) gives the automatic entropy-coefficient tuning no fixed
    point — this is what causes divergence, not any single hyperparameter.

    Including classes reparameterize ``u ~ Normal(mean, std)``, squash it
    through tanh, rescale to the action space's ``[low, high]`` bounds, and
    get the standard SAC change-of-variables correction to log_prob applied
    (Haarnoja et al. 2018, "Soft Actor-Critic", eq. 21)::

        a       = low + (high - low)/2 * (tanh(u) + 1)
        log pi(a) = log Normal(u) - sum_i log(1 - tanh(u_i)^2) - sum_i log((high-low)_i / 2)

    using the numerically stable identity (avoids log(0) as tanh(u) -> +-1)::

        log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))

    Requires the including class to (in this order):
      1. call ``Model.__init__`` then ``GaussianMixin.__init__`` normally
         (this mixin reads ``self._g_clip_log_std`` / ``_g_min_log_std`` /
         ``_g_max_log_std`` / ``_g_reduction``, all set there);
      2. register buffers ``self._action_low`` / ``self._action_high`` from a
         FULLY BOUNDED action space (call ``self._init_tanh_squash_bounds()``
         below to do both the bounds check and the registration);
      3. implement ``compute(inputs, role) -> (mean, {"log_std": ..., [
         "residual": ...]})`` where ``mean`` is the location of the PRE-squash
         Normal — i.e. the *feedback* only, NOT including uref.

    Residual (uref) handling — the ``residual`` key in ``compute``'s output
    dict, if present, is added to the action AFTER squashing:

        action = residual + rescale(tanh(u)),   u ~ Normal(mean, std)

    so the residual control law ``u = uref + bounded_feedback`` is preserved
    exactly. Adding uref to ``mean`` (i.e. BEFORE tanh) instead would give
    ``rescale(tanh(uref + feedback))`` — a *saturated* uref, not ``uref +
    feedback`` (at zero feedback it returns ``rescale(tanh(uref))``, not
    ``uref``), silently destroying the reference-tracking structure that
    ``control`` / ``mlp`` are built on. Because the residual is constant with
    respect to the sampled noise ``u``, the change-of-variables Jacobian for
    ``action = residual + f(u)`` is the same as for ``f(u)`` alone (unit shift),
    so log_prob is UNCHANGED by it — SAC's entropy fixed-point argument is
    unaffected (the tanh still bounds log_prob regardless of uref).

    act() is overridden entirely (not just compute()): the squash-then-
    correct-log_prob step happens between sampling and log_prob, which is not
    expressible by overriding compute() alone under skrl's own
    GaussianMixin.act().

    get_entropy() is intentionally NOT overridden to something meaningful:
    the squashed distribution has no closed-form entropy, so any including
    class must only be used with algorithms that rely on the sampled
    log_prob (SAC), never an analytic-entropy bonus (e.g. PPO's
    entropy_loss_scale > 0).
    """

    def _init_tanh_squash_bounds(self) -> None:
        if self._g_min_actions is None or self._g_max_actions is None:
            raise ValueError(
                f"{type(self).__name__} requires a fully-bounded action space "
                "(every dimension needs a finite low/high) — tanh-squashing has "
                "nothing to rescale into otherwise."
            )
        self.register_buffer("_action_low", self._g_min_actions.clone())
        self.register_buffer("_action_high", self._g_max_actions.clone())

    def _rescale(self, tanh_u: torch.Tensor) -> torch.Tensor:
        """(-1, 1) -> [low, high]."""
        return self._action_low + 0.5 * (tanh_u + 1.0) * (self._action_high - self._action_low)

    def _unrescale(self, action: torch.Tensor) -> torch.Tensor:
        """[low, high] -> (-1, 1), clamped away from the boundary (atanh diverges there)."""
        frac = (action - self._action_low) / (self._action_high - self._action_low)
        return torch.clamp(2.0 * frac - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)

    def act(self, inputs: dict, *, role: str = "") -> tuple[torch.Tensor, dict]:
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        self._g_distribution = Normal(mean, log_std.exp())

        # Optional post-squash residual (e.g. uref for the [x, xref, uref]
        # path-tracking layout). Added to the action AFTER squashing — see the
        # class docstring for why (residual law preservation + log_prob
        # invariance). Popped so it doesn't leak into the returned outputs dict.
        residual = outputs.pop("residual", None)

        taken_actions = inputs.get("taken_actions")
        if taken_actions is not None:
            # Recompute log_prob for an already-taken (post-squash, post-rescale,
            # post-residual) action — e.g. an on-policy update replaying stored
            # actions. SAC itself never hits this path (it always samples fresh).
            actions = taken_actions
            feedback = actions if residual is None else actions - residual
            u = torch.atanh(self._unrescale(feedback))
        else:
            u = self._g_distribution.rsample()
            feedback = self._rescale(torch.tanh(u))
            actions = feedback if residual is None else residual + feedback

        log_prob = self._g_distribution.log_prob(u)
        # Change-of-variables correction for y = tanh(u), stable form of
        # log(1 - tanh(u)^2) (Haarnoja et al. 2018, eq. 21).
        log_prob = log_prob - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        # Second correction for the affine rescale (-1, 1) -> [low, high]:
        # d(action)/d(tanh_u) = (high - low)/2 per dimension. (The residual
        # shift is unit-Jacobian, so it needs no correction here.)
        log_prob = log_prob - torch.log(0.5 * (self._action_high - self._action_low))

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        mean_action = self._rescale(torch.tanh(mean))
        outputs["mean_actions"] = mean_action if residual is None else residual + mean_action
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form entropy (the squashed "
            "distribution isn't Gaussian) — it must only be used with algorithms "
            "that rely on the sampled log_prob (SAC), not an analytic entropy "
            "bonus (e.g. PPO's entropy_loss_scale)."
        )


class SquashedGaussianActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed plain-MLP actor — ``backbone: mlp-squashed``.

    Same MLP-over-full-observation architecture as ``MLPResidualActorModel``,
    with the SAME "add uref when the layout has one" behavior: if
    ``obs_dim = 2*x_dim + u_dim`` (path-tracking's ``[x, xref, uref]``
    layout), the action is ``uref + rescale(tanh(MLP(obs) + noise))`` — i.e.
    a bounded feedback added to uref, matching ``MLPResidualActorModel``'s
    ``u = uref + feedback`` control law, just with the feedback squashed
    instead of left unbounded. uref is added AFTER squashing (see
    ``_TanhSquashMixin``'s residual handling — adding it before tanh would
    saturate uref and break the residual law). If the layout has no uref
    (e.g. velocity-tracking), the action is plain ``rescale(tanh(MLP(obs) +
    noise))``, matching what stock skrl's ``gaussian_model`` fallback would
    have squashed.

    log_std is STATE-DEPENDENT (the network outputs both mean and log_std),
    the standard SAC convention — unlike this repo's other actors
    (CLActorModel/MLPResidualActorModel), which use one global
    log_std_parameter (fine for PPO's trust-region updates, not for SAC's
    off-policy entropy tuning, which needs the policy to shrink/widen std
    per-state).

    See ``_TanhSquashMixin`` for the squashing math and its rationale.
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
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,  # squashing already bounds actions; clamping would be redundant
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        self._init_tanh_squash_bounds()

        obs_dim = int(self.observation_space.shape[0])
        act_dim = int(self.action_space.shape[0])
        remainder = obs_dim - act_dim
        # Same layout check as MLPResidualActorModel: [x, xref, uref] means
        # obs_dim = 2*x_dim + u_dim, i.e. remainder is even and positive.
        self._u_dim = act_dim if (remainder > 0 and remainder % 2 == 0) else None

        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        self.net = MLP(obs_dim, list(hidden_dim or [256, 256]), output_dim=None, activation=act_module)
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
        features = self.net(obs)
        mean = self.mean_head(features)  # pre-squash FEEDBACK mean (no uref)
        outputs = {"log_std": self.log_std_head(features)}
        if self._u_dim is not None:
            # path-tracking [x, xref, uref]: uref is added AFTER squashing (the
            # mixin consumes this "residual" key) so the action is
            # uref + rescale(tanh(feedback)), preserving u = uref + feedback.
            outputs["residual"] = obs[:, -self._u_dim:]  # last u_dim entries
        return mean, outputs


class SquashedCLActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed CLActor (bilinear feedback) actor — ``backbone: control-squashed``.

    Same bilinear feedback ``W2(x,xref) @ tanh(W1(x,xref) @ (x - xref))``
    architecture as ``CLActorModel`` (requires the path-tracking ``[x, xref,
    uref]`` observation layout). The action is
    ``uref + rescale(tanh(feedback + noise))``: the *feedback* is squashed
    (bounded) and uref is added AFTER squashing, so the control law
    ``u = uref + feedback`` is preserved exactly (adding uref before tanh, as
    a naive port of ``CLActorModel.mean_control`` — which returns
    ``uref + feedback`` — would do, saturates uref and breaks the law; see
    ``_TanhSquashMixin``'s residual handling). Squashed instead of left
    unbounded — same rationale as ``SquashedGaussianActorModel``.

    Unlike ``CLActorModel``, ``anneal_stddev`` is always ``False`` here: this
    backbone is meant for SAC, which learns log_std by gradient descent
    through the policy loss (automatic entropy tuning) rather than an
    external annealing schedule — CLActor's built-in ``anneal_stddev``
    freezes ``logstd`` (``requires_grad=False``) and expects something else
    to call ``.anneal_stddev()`` on it, which would fight SAC's own entropy
    optimizer. If you do want annealed exploration on top of PPO with this
    backbone, use the repo's existing ``std_dev_annealing`` yaml flag /
    ``patch_ppo_std_annealing`` (see agent_patches.py) — it writes directly
    to ``log_std_parameter.data`` and works regardless of this class's
    ``anneal_stddev=False`` default.

    log_std is a single GLOBAL parameter (``cl_actor.logstd``, shape
    ``(u_dim,)``), not state-dependent — same as ``CLActorModel``. Squashing
    bounds the log_prob either way (that's what fixes SAC's divergence); a
    state-dependent head isn't required for correctness, and reusing
    ``CLActor`` as-is keeps this backbone architecturally identical to plain
    ``control`` apart from the squash.
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
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,  # squashing already bounds actions; clamping would be redundant
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        self._init_tanh_squash_bounds()

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            mode="stochastic",
            anneal_stddev=False,  # see class docstring — SAC learns log_std via gradients
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
        )
        self.log_std_parameter = self.cl_actor.logstd

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)

        # Model.__init__ moves to self.device before cl_actor exists; re-sync here.
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        _, _, uref = self.cl_actor.trim_state(state)
        # mean_control returns uref + feedback; strip uref so the FEEDBACK is
        # what gets squashed, and hand uref to the mixin as a post-squash
        # residual → action = uref + rescale(tanh(feedback)). (uref cancels
        # exactly in the subtraction, leaving W2 @ tanh(W1 @ (x - xref)).)
        feedback = self.cl_actor.mean_control(state) - uref
        return feedback, {"log_std": self.log_std_parameter, "residual": uref}

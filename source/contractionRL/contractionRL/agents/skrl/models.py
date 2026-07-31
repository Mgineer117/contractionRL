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
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
except ImportError:
    raise ImportError("skrl is required. Install it or use the local developer copy.")

from .angle_utils import embed_angles, embedded_dim, wrap_diff
from .math_utils import rescale_residual
from .nn_modules import (
    MLP,
    PREVIEW_RESIDUAL_VARIANTS,
    BoundedCCM_Generator,
    CCM_Generator,
    CLActor,
    PreviewSequenceEncoder,
)

_MIN_LOG_STD = math.log(0.001)  # ≈ -6.908; matches CLActor annealing floor

# Sentinel distinguishing "caller didn't pass x_dim" (fall back to the
# dimension-parity guess) from "caller passed x_dim=None" (an env that
# reliably reports no x_dim, e.g. vel-tracking, meaning definitely NOT
# path-tracking — see SquashedMLPActorModel / EmbeddedDeterministicModel).
_X_DIM_UNSET = object()


def _preview_width(obs_dim: int, x_dim: int, u_dim: int) -> int:
    """Width of the reference-preview tail in a ``[x, xref, uref, preview]``
    observation (0 for the historic ``[x, xref, uref]`` layout). Preview holds
    future ``uref`` rows; see ``BaseEnv.configure_preview`` /
    ``PathTrackingBase.configure_preview``. When ``x_dim`` was inferred by the
    obs_dim/2 parity guess this is 0 by construction, so a caller that does not
    know the layout never accidentally slices a preview block."""
    return max(0, int(obs_dim) - 2 * int(x_dim) - int(u_dim))



class _AnalyticPotentialMixin:
    """Critic parameterization V(s) = ||e||^2_M + f_theta(s)  (O6).

    The Mahalanobis reward is the DECREMENT form r_t = Phi(s_t+1) - Phi(s_t),
    Phi(s) = -||e||^2_M. Telescoping gives V_shaped = (1-gamma) V_orig - Phi.
    So the decrement form makes the REWARD O(dt) but does NOT remove the O(1)
    potential from the problem — it moves it into the critic's regression
    target, where -Phi(s) = ||e||^2_M is analytically known and (M's eigenvalues
    spanning w_ub/w_lb = 1000) dominates what the network must fit. Classical
    shaping/value-initialization equivalence, Wiewiora 2003.

    Adding the closed form back lets f_theta represent only the (1-gamma) V_orig
    part — the genuinely O(dt) quantity the shaping meant to isolate.

    Needs the frozen Phase-A CMG (attached by C2RLAgent post-synthesis); until
    then the term is absent and the model behaves as before.

    SCALE: the added term is in REAL value units, so this is consistent only
    with the value preprocessor off (use_value_norm=false) — otherwise the
    network output is in normalized space. train.py's arm sets it; __init__ warns.
    """

    def _init_potential(self, x_dim, angle_idx, w_lb):
        self._pot_ccm_gen = None          # set by C2RLAgent._attach_critic_potential
        self._pot_x_dim = int(x_dim)
        self._pot_angle_idx = list(angle_idx or [])
        self._pot_w_lb = float(w_lb)

    def _analytic_potential(self, inputs):
        """||e||^2_M from the RAW observation channel [x, xref, ...]. Read from
        `observations` (not `states`): skrl hands every model both, and the
        privileged state channel carries future-xref RELATIVE to x, not xref
        itself, so e = x - xref is only recoverable from the observation."""
        gen = getattr(self, "_pot_ccm_gen", None)
        if gen is None:
            return None
        obs = inputs.get("observations")
        if obs is None:
            return None
        from .math_utils import bound_W, spd_inverse
        n = self._pot_x_dim
        x, xref = obs[:, :n], obs[:, n:2 * n]
        with torch.no_grad():
            e = wrap_diff(x - xref, self._pot_angle_idx).unsqueeze(-1)
            W = bound_W(gen(x)[0], self._pot_w_lb, n, getattr(gen, "bounded", False))
            M = spd_inverse(W)
            return torch.bmm(torch.bmm(e.transpose(1, 2), M), e).squeeze(-1)


class CLDeterministicActorModel(DeterministicMixin, Model):
    """Contracting C3M_U actor wrapped as a skrl Deterministic policy model.

    This is C3M's DEFAULT policy class (``class: DeterministicMixin`` in the
    per-env ``skrl_c3m_cfg.yaml`` files, built by
    ``contraction_runner._build_c3m_models``) — not the generic backbone
    dispatch in ``runner.py``. UNBOUNDED — for the tanh-squashed, bounded
    counterpart (``backbone: control-squashed``), see
    ``SquashedCLDeterministicActorModel`` below.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
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
            anneal_stddev=False,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
            angle_idx=angle_idx or [],
            sym=sym,
            preview_dim=_preview_width(obs_dim, x_dim, u_dim),
        )

        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        return self.cl_actor.mean_control(state), {}


class SquashedCLDeterministicActorModel(DeterministicMixin, Model):
    """Tanh-squashed CLActor, deterministic — ``backbone: control-squashed`` for C3M.

    Same bilinear feedback architecture as ``CLDeterministicActorModel``, but
    ``compute()`` returns ``cl_actor.mean_control_squashed`` (feedback
    tanh-bounded into the action space, ``uref`` added AFTER squashing — see
    ``CLActor.mean_control_squashed`` / ``math_utils.rescale_residual``)
    instead of the raw unbounded ``mean_control``.

    Plain deterministic, unlike SAC's ``SquashedCLActorModel`` counterpart —
    C3M certifies ``mean_control`` directly, never a noisy sample, so it needs
    no ``_TanhSquashMixin`` machinery beyond a bounded ``compute()``.

    C3M needs this rather than a hard ``clip_actions`` clamp: ``torch.clamp``'s
    gradient is exactly zero at saturation, so ``K = jacobian(u, x)`` collapses
    there too and the certificate silently degrades to checking OPEN-LOOP drift
    alone in exactly the states that most need feedback — a confirmed real AUC
    divergence. ``tanh``'s gradient never hits exact zero.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
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
            anneal_stddev=False,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
            angle_idx=angle_idx or [],
            sym=sym,
            preview_dim=_preview_width(obs_dim, x_dim, u_dim),
        )

        # DeterministicMixin.__init__ already computed _d_min_actions/_d_max_actions
        # unconditionally (regardless of clip_actions) — reuse those.
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        return self.cl_actor.mean_control_squashed(state, self._d_min_actions, self._d_max_actions), {}


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
        angle_idx = angle_idx or []

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
                angle_idx=angle_idx,
                sym=sym,
            )
        else:
            self.ccm_gen = CCM_Generator(
                x_dim=x_dim,
                hidden_dim=hidden_dim or [256, 256],
                activation=activation,
                mode=mode,
                device=str(device) if not isinstance(device, str) else device,
                angle_idx=angle_idx,
                sym=sym,
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

    UNBOUNDED — for the tanh-squashed, bounded counterpart (``backbone:
    control-squashed``), see ``SquashedCLActorModel``.
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
        # FiLMResidual's γ1/γ2 gate sources (see nn_modules.FiLMResidual): each
        # independently "preview" (default, future-reference window), "xref"
        # (invariant features of the current reference point), or "uref" (current
        # reference control). Set via YAML models.policy.film_gate1_source, e.g.
        # to test gating on the current reference rather than the future preview.
        film_gate1_source: str = "preview",
        film_gate2_source: str = "preview",
        # How a "preview"-sourced FiLM gate turns its P-point window into the
        # γ-network input: "mlp" (flatten), "gru", or "attn" — all three route
        # through nn_modules.PreviewSequenceEncoder and share film_gate_stride.
        # Set via YAML models.policy.film_gate_encoder.
        film_gate_encoder: str = "mlp",
        # Stride applied UNIFORMLY to film_gate_encoder's "mlp"/"gru"/"attn"
        # (see nn_modules.PreviewSequenceEncoder): keep every stride-th
        # preview point, nearest-first, instead of the full sequence. 1
        # (default) = dense, every point — the original behavior. Set via
        # YAML models.policy.film_gate_stride.
        film_gate_stride: int = 1,
        # Must agree with the env's configure_preview(..., include_xref=...) —
        # see env_base.BaseEnv.construct_state. False (default) means the
        # preview tail is uref-only, exactly as before this option existed.
        # Set via YAML models.policy.preview_includes_xref /
        # --preview_include_xref (train.py wires both together).
        preview_includes_xref: bool = False,
        # Must agree with the env's configure_preview(..., include_uref=...).
        # True (default) reproduces the historic layout exactly. False means
        # the preview tail carries NO uref block (xref-only — needs
        # preview_includes_xref=True or there is nothing in the tail). Set via
        # YAML models.policy.preview_includes_uref / --preview_no_uref
        # (train.py wires both together).
        preview_includes_uref: bool = True,
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
        # Declared explicitly (not swallowed by **kwargs): callers that know the
        # layout — contraction_runner passes the env's own x_dim — must win over
        # the dimension-parity guess, as they already do for every other
        # backbone here.
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            anneal_stddev=True,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
            angle_idx=angle_idx or [],
            sym=sym,
            preview_dim=_preview_width(obs_dim, x_dim, u_dim),
        )
        self.log_std_parameter = self.cl_actor.logstd

        # Optional analytic baseline for residual RL. When set (by C2RLAgent after
        # Phase A, see CVSTEMLQRBase), the actor mean becomes
        # ``u = u_cvstem-lqr(x,xref) + bilinear_feedback`` instead of
        # ``uref + bilinear_feedback`` — the policy starts at the certified
        # contraction controller and PPO learns only the correction. None keeps
        # the historic behaviour exactly.
        self.base_controller = None

        # Anticipatory FEEDFORWARD residual head (the π-network variant that ADDS
        # value to the CV-STEM-LQR base instead of degrading it). LQ-preview theory:
        # the optimal preview-tracking law is feedback(−K·e) + feedforward(future
        # reference). The base supplies the certified feedback −K·e; the myopic
        # analytic controller structurally LACKS the feedforward. This head learns
        # exactly that missing term as a function of ONLY the reference preview
        # (current + future uref) — independent of the error e, so it can never
        # reshape (and corrupt) the base's certified feedback gain, unlike the
        # bilinear W₂·tanh(W₁·e) residual (∝ e) which PPO drifts off the optimum
        # (→ AUC 1.19). Zero-initialized output ⇒ training STARTS exactly at the
        # analytic base and can only add helpful anticipation. Built only when the
        # obs carries a preview tail; enabled by C2RLAgent (residual_feedforward).
        self._preview_dim = _preview_width(obs_dim, x_dim, u_dim)
        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        self.ff_head = None
        # Structured preview-residual variants (nn_modules: DecoupledResidual /
        # LatentPreviewResidual / FiLMResidual). Built up front — even the unused
        # ones — so ALL their params are registered before the PPO optimizer is
        # constructed over policy.parameters(); compute() dispatches to the one
        # C2RLAgent selects (residual_variant), and the rest simply receive no
        # gradient. Need a preview to have a P role.
        self.residual_modules = nn.ModuleDict()
        # When the preview tail also carries relative-future-xref points (see
        # env_base.BaseEnv.construct_state), it is [uref-block, xref-block] —
        # each preview_includes_xref=False -> 0 (whole tail stays uref-only,
        # unchanged). n_pts must divide evenly; the env and model configs
        # agreeing on include_xref/num_preview/x_dim/u_dim guarantees that.
        self._xref_preview_dim = 0
        if preview_includes_xref and self._preview_dim > 0:
            if preview_includes_uref:
                n_pts = self._preview_dim // (u_dim + x_dim)
                self._xref_preview_dim = n_pts * x_dim
            else:
                # xref-only preview: no uref block at all, so the WHOLE tail
                # is the xref block (see env_base.construct_state).
                self._xref_preview_dim = self._preview_dim
        if self._preview_dim > 0:
            self.ff_head = MLP(u_dim + self._preview_dim, [64, 64], u_dim,
                               activation=act_module)
            self._zero_last_linear(self.ff_head)
            for _name, _cls in PREVIEW_RESIDUAL_VARIANTS.items():
                _extra = {}
                if _name == "film":
                    # See FiLMResidual: independently-selectable γ1/γ2 gate
                    # sources; xref_feat_dim needed when either gate reads "xref".
                    _extra = dict(xref_feat_dim=self.cl_actor.xref_feat_dim,
                                  gate1_source=film_gate1_source,
                                  gate2_source=film_gate2_source,
                                  gate_encoder=film_gate_encoder,
                                  gate_stride=film_gate_stride,
                                  xref_preview_dim=self._xref_preview_dim)
                self.residual_modules[_name] = _cls(
                    x_dim=x_dim, u_dim=u_dim,
                    state_dim=self.cl_actor.state_feat_dim,
                    preview_dim=self._preview_dim,
                    hidden=[128, 128], activation=act_module, **_extra)
        # "bilinear" (original CLActor feedback) | "feedforward" (preview-only) |
        # "decoupled" | "latent_bias" | "film". Set by C2RLAgent after Phase A.
        self.residual_variant = "bilinear"
        self.residual_feedforward = False  # legacy alias (True => "feedforward")

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)

        # Model.__init__ moves to self.device before cl_actor exists; re-sync here.
        self.to(self.device)

    @staticmethod
    def _zero_last_linear(mlp: MLP) -> None:
        """Zero the final Linear of an MLP so its output starts at exactly 0
        (residual begins at the analytic base; PPO grows it from there)."""
        last = None
        for m in mlp.net:
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        x_dim, u_dim = self.cl_actor.x_dim, self.cl_actor.u_dim
        if self.base_controller is not None and getattr(self, "_eval_base_only", False):
            # Controlled-comparison eval: bypass the residual to measure the PURE
            # CV-STEM-LQR base on the IDENTICAL frozen CMG the trained residual
            # used (eliminates the CMG-regression nondeterminism that separate
            # base/residual runs suffer). Set by train.py --eval_base_too.
            return self.base_controller(state), {"log_std": self.log_std_parameter}
        # The control law is u = base + π(s). The base is the REFERENCE FEEDFORWARD
        # uref by default (u = uref + π — π learns the whole feedback, the deployed
        # policy standing on its own), OR the analytic CV-STEM-LQR controller when
        # one is explicitly attached (cvstem_residual_base warm-start; off by default).
        if self.base_controller is not None:
            base = self.base_controller(state)
        else:
            base = state[:, 2 * x_dim : 2 * x_dim + u_dim]  # uref
        # π(s): architecture selected by residual_variant (see __init__/nn_modules).
        v = self.residual_variant
        if v == "feedforward" and self.ff_head is not None:
            # Preview-only, e-independent — provides NO feedback, so meaningful only
            # on a base that already closes the loop (cvstem_residual_base).
            pi = self.ff_head(state[:, 2 * x_dim:])  # [uref, preview] tail
        elif v in self.residual_modules:
            sf = self.cl_actor._state_feats(state)
            e = self.cl_actor._error_vec(state)
            pv = self.cl_actor._preview(state)
            xf = self.cl_actor._xref_feats(state)
            uref = state[:, 2 * x_dim : 2 * x_dim + u_dim]
            pi = self.residual_modules[v](sf, e, pv, xref_feats=xf, uref=uref)
        else:  # "bilinear" — original CLActor feedback W2·tanh(W1·e)
            pi, _uref = self.cl_actor._feedback(state)
        return base + pi, {"log_std": self.log_std_parameter}


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
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
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
        self._u_dim = u_dim
        # Explicit x_dim (from the env) wins over the parity guess — see CLActorModel.
        self._x_dim = x_dim if x_dim is not None else (obs_dim - u_dim) // 2
        # Optional future-uref preview tail (0 for the plain [x, xref, uref]
        # layout). When x_dim is explicit it absorbs the remainder exactly, so
        # the check below only rejects a degenerate/parity-guessed non-layout.
        self._preview_dim = _preview_width(obs_dim, self._x_dim, u_dim)
        if self._x_dim <= 0 or 2 * self._x_dim + u_dim + self._preview_dim != obs_dim:
            raise ValueError(
                f"MLPResidualActorModel requires obs_dim = 2*x_dim + u_dim (+preview) "
                f"(path-tracking layout [x, xref, uref, preview]), got obs_dim={obs_dim}, "
                f"act_dim={u_dim}, x_dim={self._x_dim}."
            )
        self._angle_idx = angle_idx or []
        self._sym = sym

        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        # net sees the x/xref blocks CONTINUOUSLY embedded + the raw uref tail
        # (+ the raw future-uref preview tail, which is symmetry-invariant).
        net_in_dim = (sym.pair_dim() if sym is not None else 2 * embedded_dim(self._x_dim, self._angle_idx)) + u_dim + self._preview_dim
        self.net = MLP(net_in_dim, list(hidden_dim or [128, 128]), u_dim, activation=act_module)

        self.log_std_parameter = nn.Parameter(torch.full((u_dim,), float(initial_log_std)))

        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        obs = inputs["observations"]
        x = obs[:, : self._x_dim]
        xref = obs[:, self._x_dim : 2 * self._x_dim]
        uref = obs[:, 2 * self._x_dim : 2 * self._x_dim + self._u_dim]
        preview = obs[:, 2 * self._x_dim + self._u_dim:]  # future-uref tail (width _preview_dim)
        net_in = torch.cat(
            [(self._sym.pair_features(x, xref) if self._sym is not None else torch.cat([embed_angles(x, self._angle_idx), embed_angles(xref, self._angle_idx)], -1)), uref, preview], dim=-1
        )
        mean = uref + self.net(net_in)
        return mean, {"log_std": self.log_std_parameter}


class _TanhSquashMixin:
    """Shared tanh-squash act()/log_prob machinery for bounded-action Gaussian actors.

    One copy for every squashed backbone: this correction is easy to get subtly
    wrong, so it is not copy-pasted per backbone.

    Necessary for SAC (any off-policy method with an entropy term) on a bounded
    action space. skrl's stock ``GaussianMixin.act()`` samples an UNBOUNDED
    ``Normal(mean, std)`` and merely hard-clamps the sample afterwards —
    ``log_prob`` still comes from the unclamped Normal. SAC's ``-alpha *
    log_prob`` appears in both the Bellman target and the policy loss, so an
    unbounded log_prob (growing without limit as log_std shrinks) leaves the
    automatic entropy tuning with no fixed point. That is the divergence
    mechanism — not any single hyperparameter.

    Including classes reparameterize ``u ~ Normal(mean, std)``, squash through
    tanh, rescale to ``[low, high]``, and get the standard change-of-variables
    correction applied (Haarnoja et al. 2018, eq. 21)::

        a       = low + (high - low)/2 * (tanh(u) + 1)
        log pi(a) = log Normal(u) - sum_i log(1 - tanh(u_i)^2) - sum_i log((high-low)_i / 2)

    using the numerically stable identity (avoids log(0) as tanh(u) -> +-1)::

        log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))

    Including classes must, in this order:
      1. call ``Model.__init__`` then ``GaussianMixin.__init__`` (this mixin
         reads the ``_g_*`` attributes set there);
      2. call ``self._init_tanh_squash_bounds()`` (checks the action space is
         fully bounded and registers ``_action_low``/``_action_high``);
      3. implement ``compute(inputs, role) -> (mean, {"log_std", ["residual"]})``
         where ``mean`` locates the PRE-squash Normal — the *feedback* only,
         NOT including uref.

    RESIDUAL (uref) is added AFTER squashing::

        action = residual + rescale_residual(tanh(u), residual)

    preserving ``u = uref + bounded_feedback`` exactly. Adding uref to ``mean``
    (before tanh) would give ``rescale(tanh(uref + feedback))`` — a SATURATED
    uref that returns ``rescale(tanh(uref))``, not ``uref``, at zero feedback,
    silently destroying the reference-tracking structure.

    ``rescale_residual`` is ASYMMETRIC: ``tanh(u) >= 0`` maps into
    ``[0, high - residual]`` and ``tanh(u) < 0`` into ``[-(residual - low), 0]``,
    a different per-sample scale each side (residual varies per state). For
    every residual in [low, high] this guarantees ``tanh(u) == 0 => action ==
    residual`` exactly, and ``action`` inside ``(low, high)`` for every ``u``
    with NO post-hoc clamping. A single constant rescale-then-add-then-clamp
    would push the sum out of bounds whenever residual is off-center, silently
    decoupling the clamped action from the log_prob of the unclamped one — SAC's
    entropy term would score the wrong sample. Both half-scales are positive
    constants w.r.t. ``u``, so the change-of-variables Jacobian is as simple as
    the fixed-scale case (just swap in the applicable half-scale): an EXACT
    closed-form log_prob for the applied action, not an approximation.

    act() is overridden entirely, not just compute(): squash-then-correct
    happens between sampling and log_prob, which overriding compute() alone
    can't express under skrl's GaussianMixin.act().

    get_entropy() is deliberately left meaningless — the squashed distribution
    has no closed-form entropy. Use only with algorithms relying on the sampled
    log_prob (SAC), never an analytic entropy bonus (PPO's entropy_loss_scale).
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
        # path-tracking layout). Added to the action AFTER squashing — see the
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
        # is `scale` per dimension — a CONSTANT half-width (0.5*(high-low)) with
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


class SquashedMLPActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed plain-MLP actor — ``backbone: mlp-squashed``.

    Same architecture and "add uref when the layout has one" behavior as
    ``MLPResidualActorModel``: on the ``[x, xref, uref]`` layout the action is
    ``uref + rescale(tanh(MLP(obs) + noise))``, uref added AFTER squashing (see
    ``_TanhSquashMixin`` — before tanh would saturate uref). With no uref in the
    layout (vel-tracking) it is plain ``rescale(tanh(MLP(obs) + noise))``.

    log_std is STATE-DEPENDENT (the network outputs both), the SAC convention —
    unlike this repo's other actors' single global log_std_parameter, which is
    fine for PPO's trust-region updates but not for SAC's entropy tuning, which
    needs per-state std.
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
        x_dim: int | None = _X_DIM_UNSET,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
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
        if x_dim is _X_DIM_UNSET:
            # Caller passed nothing (e.g. the classic CLActorRunner/yaml path,
            # which has no such signal) — fall back to the dimension-parity
            # guess. Same layout check as MLPResidualActorModel: [x, xref,
            # uref] means obs_dim = 2*x_dim + u_dim, i.e. remainder is even
            # and positive. This is a heuristic: an env whose obs_dim - act_dim
            # happens to be even and positive for reasons unrelated to a
            # uref layout (e.g. a flat velocity-tracking obs) would be
            # misclassified — pass x_dim explicitly (even x_dim=None, for "no,
            # definitely not path-tracking") to avoid this.
            remainder = obs_dim - act_dim
            is_path_tracking = remainder > 0 and remainder % 2 == 0
            x_dim = remainder // 2 if is_path_tracking else None
        else:
            # Caller already knows the layout (e.g. contraction_runner.py's
            # raw_env.x_dim, which is None for envs like vel-tracking that
            # never declare it) — trust it instead of guessing from dimensions.
            is_path_tracking = x_dim is not None
        self._u_dim = act_dim if is_path_tracking else None
        self._x_dim = x_dim if is_path_tracking else None
        # angle_idx only makes sense (and is only ever passed) when the layout
        # has a known x/xref split; vel-tracking's flat obs has none.
        self._angle_idx = (angle_idx or []) if is_path_tracking else []
        # Same gate as angle_idx: only the [x, xref, uref] layout has a
        # position block to quotient out; a flat obs leaves pos_dim at 0.
        self._sym = sym if is_path_tracking else None
        # Future-uref preview tail (0 without preview / for a flat obs).
        self._preview_dim = _preview_width(obs_dim, self._x_dim, act_dim) if is_path_tracking else 0

        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        net_in_dim = (
            (sym.pair_dim() if sym is not None else 2 * embedded_dim(self._x_dim, self._angle_idx)) + self._u_dim + self._preview_dim
            if is_path_tracking else obs_dim
        )
        self.net = MLP(net_in_dim, list(hidden_dim or [256, 256]), output_dim=None, activation=act_module)
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
        if self._u_dim is not None:
            x = obs[:, : self._x_dim]
            xref = obs[:, self._x_dim : 2 * self._x_dim]
            uref = obs[:, 2 * self._x_dim : 2 * self._x_dim + self._u_dim]
            preview = obs[:, 2 * self._x_dim + self._u_dim:]  # future-uref tail
            net_in = torch.cat(
                [(self._sym.pair_features(x, xref) if self._sym is not None else torch.cat([embed_angles(x, self._angle_idx), embed_angles(xref, self._angle_idx)], -1)), uref, preview], dim=-1
            )
        else:
            net_in = obs
        features = self.net(net_in)
        mean = self.mean_head(features)  # pre-squash FEEDBACK mean (no uref)
        outputs = {"log_std": self.log_std_head(features)}
        if self._u_dim is not None:
            # path-tracking [x, xref, uref]: uref is added AFTER squashing (the
            # mixin consumes this "residual" key) so the action is
            # uref + rescale(tanh(feedback)), preserving u = uref + feedback.
            outputs["residual"] = obs[:, 2 * self._x_dim : 2 * self._x_dim + self._u_dim]
        return mean, outputs


class SquashedCLActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed CLActor (bilinear feedback) actor — ``backbone: control-squashed``.

    Same bilinear feedback as ``CLActorModel`` (needs the ``[x, xref, uref]``
    layout). Action is ``uref + rescale(tanh(feedback + noise))`` — the FEEDBACK
    is squashed, uref added after, preserving ``u = uref + feedback`` exactly.
    (A naive port of ``CLActorModel.mean_control``, which already returns
    uref + feedback, would squash uref too and break the law.)

    ``anneal_stddev`` is always False here: this backbone is for SAC, which
    learns log_std through the policy loss. CLActor's annealing freezes
    ``logstd`` and expects an external caller to step it, which would fight
    SAC's entropy optimizer. Same reason ``"control-squashed"`` is deliberately
    NOT in ``runner.CONTROL_BACKBONES`` — that set auto-enables
    ``patch_ppo_std_annealing``, which only makes sense for a frozen log_std.

    log_std is a single GLOBAL parameter, as in ``CLActorModel``. Squashing
    bounds the log_prob either way (that's the SAC fix); a state-dependent head
    isn't needed for correctness, and reusing CLActor as-is keeps this backbone
    identical to plain ``control`` apart from the squash.
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
            anneal_stddev=False,  # see class docstring — SAC learns log_std via gradients
            angle_idx=angle_idx or [],
            sym=sym,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
            preview_dim=_preview_width(obs_dim, x_dim, u_dim),
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


class EmbeddedDeterministicModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
    """Value/critic MLP with angle-embedded input — the DeterministicMixin
    counterpart to MLPResidualActorModel/SquashedMLPActorModel.

    skrl's own model-instantiator DSL (``deterministic_model``, driven by the
    yaml ``network: input: OBSERVATIONS`` / ``concatenate([OBSERVATIONS,
    ACTIONS])`` keys) is vendored library code — it has no notion of an
    angle-bearing state, so a value/critic built through it would see the RAW
    (discontinuous-at-+-pi) angle. This class is this repo's drop-in
    replacement, wired in by CLActorRunner._component for "deterministicmixin"
    (see runner.py) whenever the env carries a non-empty angle_idx.

    Only the path-tracking observation layout ([x, xref, uref], obs_dim =
    2*x_dim + u_dim) has a known x/xref split to embed; for any other layout
    (e.g. velocity-tracking's flat obs) this reduces to a plain MLP over the
    raw observation (+ actions, for a critic) — identical to what skrl's stock
    deterministic_model would have built.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        network: list | None = None,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = _X_DIM_UNSET,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        use_actions: bool = False,
        analytic_potential: bool = False,
        w_lb: float = 0.01,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        net_spec = (network or [{}])[0] if network else {}
        hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
        activation = net_spec.get("activations", activation)

        obs_dim = int(self.observation_space.shape[0])
        act_dim = int(self.action_space.shape[0])
        if x_dim is _X_DIM_UNSET:
            # Caller passed nothing (e.g. the classic CLActorRunner/yaml path) —
            # fall back to the dimension-parity guess. Heuristic: an env whose
            # obs_dim - act_dim happens to be even and positive for reasons
            # unrelated to a uref layout (e.g. a flat velocity-tracking obs)
            # would be misclassified — pass x_dim explicitly (even x_dim=None,
            # for "no, definitely not path-tracking") to avoid this.
            remainder = obs_dim - act_dim
            is_path_tracking = remainder > 0 and remainder % 2 == 0
            x_dim = remainder // 2 if is_path_tracking else None
        else:
            # Caller already knows the layout (e.g. contraction_runner.py's
            # raw_env.x_dim, which is None for envs like vel-tracking that
            # never declare it) — trust it instead of guessing from dimensions.
            is_path_tracking = x_dim is not None
        self._x_dim = x_dim if is_path_tracking else None
        self._u_dim = act_dim if is_path_tracking else None
        self._angle_idx = (angle_idx or []) if is_path_tracking else []
        # Same gate as angle_idx: only the [x, xref, uref] layout has a
        # position block to quotient out; a flat obs leaves pos_dim at 0.
        self._sym = sym if is_path_tracking else None
        self._use_actions = use_actions
        # Future-uref preview tail (0 without preview / for a flat obs). The
        # critic is the ESSENTIAL consumer: V^pi depends on the future
        # reference, so previewing it is what makes a high discount_factor a
        # well-posed (Markov) value target rather than a POMDP.
        self._preview_dim = _preview_width(obs_dim, self._x_dim, act_dim) if is_path_tracking else 0
        # O6 analytic-potential critic (see _AnalyticPotentialMixin). Needs a
        # path-tracking [x, xref, ...] layout to recover e = x - xref at all.
        self.use_analytic_potential = bool(analytic_potential) and is_path_tracking
        self._init_potential(self._x_dim or 0, self._angle_idx, w_lb)

        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation

        obs_in_dim = (
            (sym.pair_dim() if sym is not None else 2 * embedded_dim(self._x_dim, self._angle_idx)) + self._u_dim + self._preview_dim
            if is_path_tracking else obs_dim
        )
        net_in_dim = obs_in_dim + (act_dim if use_actions else 0)
        self.net = MLP(net_in_dim, list(hidden_dim), 1, activation=act_module)

        self.to(self.device)

    def _embed_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._u_dim is None:
            return obs
        x = obs[:, : self._x_dim]
        xref = obs[:, self._x_dim : 2 * self._x_dim]
        uref = obs[:, 2 * self._x_dim : 2 * self._x_dim + self._u_dim]
        preview = obs[:, 2 * self._x_dim + self._u_dim:]  # future-uref tail
        return torch.cat(
            [(self._sym.pair_features(x, xref) if self._sym is not None else torch.cat([embed_angles(x, self._angle_idx), embed_angles(xref, self._angle_idx)], -1)), uref, preview], dim=-1
        )

    def compute(self, inputs: dict, role: str = "value"):
        obs_emb = self._embed_obs(inputs["observations"])
        if self._use_actions:
            net_in = torch.cat([obs_emb, inputs["taken_actions"]], dim=-1)
        else:
            net_in = obs_emb
        value = self.net(net_in)
        if getattr(self, "use_analytic_potential", False):
            pot = self._analytic_potential(inputs)
            if pot is not None:
                value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
        return value, {}


class TrajectoryAwareValueModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
    """Asymmetric (privileged) PPO critic: V(x, future-xref trajectory), read
    from skrl's separate ``states``/``state_space`` channel — see
    ``env_base.BaseEnv.configure_value_state``/``state()`` — completely
    DECOUPLED from whatever preview the ACTOR's own ``observations`` happens
    to carry. skrl's PPO already passes both ``observations`` AND ``states``
    to every model's ``inputs`` dict (``ppo.py``'s ``act()``); this is simply
    the first model here that reads ``inputs["states"]`` instead of
    ``inputs["observations"]``. The actor is completely unaffected either way.

    Input layout (see ``BaseEnv.state()``): ``[x (x_dim), future-xref-relative
    points (P * x_dim)]`` — P is whatever ``configure_value_state`` was set up
    with (a handful of geometric offsets, or every remaining step for
    ``full_trajectory=True``), independent of the actor's own preview length.

    ``encoder`` picks how the P-point trajectory block is turned into a fixed
    vector: "mlp" (flatten), "gru", or "attn" — all three route through
    ``PreviewSequenceEncoder`` and share ``encoder_stride`` (keep every
    stride-th point, nearest-first; 1 = dense/every point). Stride is what
    bounds "mlp"'s input width on a long/full-trajectory window — otherwise
    it scales with episode length.

    ``combine`` picks how the state embedding phi(x) and the goal/trajectory
    embedding psi(traj) are turned into a scalar value:

      "concat" (default) — cat([phi, psi]) through one joint MLP. Free to mix
      the embeddings however it likes, so it CAN (and empirically does) fit
      specific (x, traj) pairs jointly rather than learning independently
      meaningful phi/psi — not actually factorized despite the separate encoders.

      "bilinear" — true UVFA factorization (Schaul et al. 2015):
      V = w^T(phi ⊙ psi) + u^T phi + v^T psi + b. Never concatenated, so the two
      interact only through the elementwise product and each embedding must be
      useful on its own (the additive terms cover state-only / goal-only value).
      This is what should generalize to a NEW (x, traj) pair resembling two
      different training points — concat+MLP has no structural pressure to.

      "film" — middle ground, mirroring the actor's FiLMResidual: psi modulates
      phi BEFORE a small joint head, V = head(phi * softplus(gamma(psi)) +
      beta(psi)). Keeps a nonlinear head after the interaction (no
      encoder_hidden rank ceiling, unlike bilinear) while still FORCING the
      interaction through an explicit trajectory-conditioned gate (unlike
      concat). Untested at scale.
    """

    _COMBINE_MODES = ("concat", "bilinear", "film")

    def __init__(
        self,
        state_space,
        action_space,
        device,
        network: list | None = None,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        combine: str = "concat",
        analytic_potential: bool = False,
        w_lb: float = 0.01,
        **kwargs,
    ):
        Model.__init__(self, observation_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        if x_dim is None:
            raise ValueError("TrajectoryAwareValueModel needs x_dim — the state layout is "
                              "[x, P future-xref-relative points], not self-describing without it.")
        self.x_dim = int(x_dim)
        self.angle_idx = list(angle_idx or [])
        if combine not in self._COMBINE_MODES:
            raise ValueError(f"TrajectoryAwareValueModel: combine must be one of "
                             f"{self._COMBINE_MODES}, got {combine!r}")
        self.combine = combine

        net_spec = (network or [{}])[0] if network else {}
        hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
        activation = net_spec.get("activations", activation)
        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation

        state_dim = int(self.observation_space.shape[0])
        self._traj_dim = state_dim - self.x_dim
        if encoder not in PreviewSequenceEncoder.MODES:
            raise ValueError(f"TrajectoryAwareValueModel: encoder must be one of "
                             f"{PreviewSequenceEncoder.MODES}, got {encoder!r}")
        self.traj_encoder = PreviewSequenceEncoder(
            self._traj_dim, point_dim=self.x_dim, hidden=encoder_hidden,
            out_dim=encoder_hidden, mode=encoder, stride=encoder_stride)

        if self.combine == "concat":
            self.net = MLP(embedded_dim(self.x_dim, self.angle_idx) + encoder_hidden,
                           list(hidden_dim), 1, activation=act_module)
        elif self.combine == "film":
            # phi(x) gets the same depth as "bilinear"'s phi; psi(traj) (already
            # produced by traj_encoder above) drives a FiLM gate on phi BEFORE a
            # small joint head -- see class docstring's "film" entry.
            self.phi = MLP(embedded_dim(self.x_dim, self.angle_idx),
                           list(hidden_dim), encoder_hidden, activation=act_module)
            self.film_gamma = nn.Linear(encoder_hidden, encoder_hidden)
            self.film_beta = nn.Linear(encoder_hidden, encoder_hidden)
            self.film_head = MLP(encoder_hidden, [encoder_hidden], 1, activation=act_module)
        else:
            # phi(x): state embedding, same width as psi(traj) (encoder_hidden)
            # so they can be combined elementwise.
            #
            # EMBEDDING WIDTH MATTERS HERE, much more than it does for "concat".
            # The value this head can represent is
            #     V = w^T(phi ⊙ psi) + u^T phi + v^T psi + b,
            # i.e. a DIAGONAL bilinear form whose rank is bounded by
            # encoder_hidden — that width is the entire capacity of the
            # state x goal interaction, exactly the quantity UVFA (Schaul et al.
            # 2015) treats as a primary hyperparameter. "concat" has no such
            # bottleneck: it feeds both embeddings into a full hidden_dim MLP.
            # So comparing the two at a shared, small encoder_hidden confounds
            # "factorization" with "starved factorization" — the reason
            # encoder_hidden is plumbed through to the CLI/YAML rather than left
            # at its 64 default. phi gets hidden_dim depth too, so the two modes
            # are matched on everything except the interaction structure.
            self.phi = MLP(embedded_dim(self.x_dim, self.angle_idx),
                           list(hidden_dim), encoder_hidden, activation=act_module)
            self.bilinear_head = nn.Linear(encoder_hidden, 1)
            self.phi_head = nn.Linear(encoder_hidden, 1)
            self.psi_head = nn.Linear(encoder_hidden, 1)
        self.use_analytic_potential = bool(analytic_potential)
        self._init_potential(self.x_dim, self.angle_idx, w_lb)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "value"):
        state = inputs["states"]
        x = state[:, : self.x_dim]
        traj = state[:, self.x_dim:]
        x_emb = embed_angles(x, self.angle_idx)
        traj_emb = self.traj_encoder(traj)
        if self.combine == "concat":
            value = self.net(torch.cat([x_emb, traj_emb], dim=-1))
        elif self.combine == "film":
            phi = self.phi(x_emb)
            gamma = F.softplus(self.film_gamma(traj_emb))  # positive scale, mirrors actor's FiLM gain
            beta = self.film_beta(traj_emb)
            value = self.film_head(phi * gamma + beta)
        else:
            phi = self.phi(x_emb)
            value = self.bilinear_head(phi * traj_emb) + self.phi_head(phi) + self.psi_head(traj_emb)
        if self.use_analytic_potential:
            pot = self._analytic_potential(inputs)
            if pot is not None:
                value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
        return value, {}

"""Contraction metrics: spectral bounds, Cholesky regression, and analytic metrics."""
from __future__ import annotations

import math
from collections.abc import Sequence
import torch
import torch.nn as nn
from torch.distributions import Normal
from contractionRL.agents.utils.RL.networks import MLP, _ACT_MAP
from contractionRL.agents.utils.system_dynamics.state import StateSymmetry, embed_angles, embedded_dim



class CCM_Generator(nn.Module):
    """Contraction-metric generator W(x) = VᵀV (symmetric PSD by construction).

    In stochastic mode, samples from a diagonal Gaussian over the matrix entries
    and reports entropy for optional regularisation (used in C2RL).
    """
    bounded = False

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list[int],
        activation: str | nn.Module = "tanh",
        mode: str = "stochastic",
        device: str = "cpu",
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.mode = mode
        self.angle_idx = list(angle_idx)
        # Translation quotient: W depends only on the non-position dims (f and B
        # do, so the metric should too). pos_dim=0 keeps the absolute behaviour.
        self.sym = sym

        if isinstance(activation, str):
            activation = _ACT_MAP[activation.lower()]()

        # Network sees the continuous (cos, sin) embedding of any angle dims, with
        # the translation directions dropped; W(x) itself is still indexed/shaped
        # by the raw x_dim (see forward).
        self.backbone = MLP((sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx)),
                            list(hidden_dim), activation=activation)
        h = self.backbone.output_dim
        self.mu_head = nn.Linear(h, x_dim * x_dim)
        self.logstd_head = nn.Linear(h, x_dim * x_dim)
        self.to(device)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        n = x.shape[0]
        h = self.backbone(self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        mu = self.mu_head(h)

        if self.mode == "deterministic" or deterministic:
            W_flat = mu
            info = {
                "entropy": torch.zeros(n, 1, device=x.device, dtype=x.dtype),
                "logprobs": torch.zeros(n, 1, device=x.device, dtype=x.dtype),
            }
        else:
            logstd = torch.clamp(self.logstd_head(h), -5, 2)
            dist = Normal(mu, logstd.exp())
            W_flat = dist.rsample()
            info = {
                "entropy": dist.entropy().sum(-1, keepdim=True),
                "logprobs": dist.log_prob(W_flat).sum(-1, keepdim=True),
            }

        W = W_flat.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)  # symmetric PSD: VᵀV
        return W, info


class CholMetric(nn.Module):
    """Tsukamoto's NCM network (``classncm.train`` + ``ncm``/``cholM2M``).

    A plain MLP ``x ↦ vec(R)`` with ``R`` upper-triangular and ``M = RᵀR``, so
    the deployed metric is SPD by construction at every state — including ones
    the SDP never sampled — with no eigenvalue clamp anywhere. That is the whole
    reason the reference regresses ``chol(M)`` rather than ``W``: ``ν`` is free in
    its SDP, so no ``[w_lb, w_ub]`` envelope exists to clamp to, and clamping a
    regressed ``W`` would silently deploy a metric the SDP never certified.

    Deliberately not ``CCM_Generator``: that one outputs ``W`` and relies on
    ``bound_W`` for definiteness, which is the repo's envelope-bearing variant
    (still what C2RL uses). This one is only for CV-STEM-LQR.

    Raw ``x`` in, no feature map and no angle embedding — the reference feeds the
    state vector directly. Angles therefore enter unwrapped; a state box that
    spans ±π has a discontinuity the network must fit through, which is the
    reference's behaviour, not an oversight here.

    Output layer is plain linear, so ``R``'s diagonal may be zero or negative and
    ``M`` is PSD rather than strictly PD. Nothing downstream inverts ``M`` (the
    gain is ``K = R⁻¹BᵀM``), so a singular ``M`` degrades feedback smoothly
    instead of raising.
    """

    def __init__(self, x_dim: int, hidden_dims: Sequence[int] = (128, 128),
                 activation: nn.Module | None = None) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.n_out = self.x_dim * (self.x_dim + 1) // 2
        self.net = MLP(self.x_dim, list(hidden_dims), self.n_out,
                       activation=activation if activation is not None else nn.ReLU())
        iu = torch.triu_indices(self.x_dim, self.x_dim)
        # Same row-major upper-triangular order as ncm_synthesis.M_to_cholvec —
        # the labels and the reconstruction must agree on it.
        self.register_buffer("_iu", iu, persistent=False)

    def chol(self, x: torch.Tensor) -> torch.Tensor:
        """Upper-triangular ``R`` with ``M = RᵀR``, ``(b, x_dim, x_dim)``."""
        vec = self.net(x)
        R = x.new_zeros(x.shape[0], self.x_dim, self.x_dim)
        R[:, self._iu[0], self._iu[1]] = vec
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.chol(x)
        return torch.bmm(R.transpose(1, 2), R)


class BoundedCCM_Generator(nn.Module):
    """CMG with hard eigenvalue bounds baked into the forward pass.

    ``outputs_metric`` selects which of the two mutually-inverse matrices the
    forward pass produces. C2RL sets it True; C3M leaves it False:

      * C2RL (both synthesis methods) -> ``outputs_metric=True``: emits ``M`` directly,
        with eigenvalues squashed into ``[1/w_ub, 1/w_lb]``. The reward needs
        ``e^T M e``, so emitting ``M`` removes a batched SPD inverse from every
        env step (env_base.get_rewards used to call ``spd_inverse(W)`` per step
        per env, on top of the eigh this forward already does). The regression
        target is inverted once offline instead.
        The direct CCM loss is also written in M.
      * C3M -> ``outputs_metric=False``: emits ``W`` for its dual C1/C2 losses.

    Fitting ``M`` directly is also better aligned with its use: inverting a
    fitted ``W`` amplifies relative error exactly where ``W`` is smallest, which
    is the stiff, weakly-actuated region the certificate cares most about.
    """
    bounded = True

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list,
        activation: str | nn.Module = "tanh",
        mode: str = "deterministic",
        w_lb: float = 0.1,
        w_ub: float = 10.0,
        device: str = "cpu",
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
        outputs_metric: bool = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.mode = mode
        self.device = device
        if not (math.isfinite(w_lb) and math.isfinite(w_ub) and 0 < w_lb < w_ub):
            raise ValueError("Metric bounds require finite 0 < w_lb < w_ub")
        self.w_lb = w_lb
        self.w_ub = w_ub
        self.outputs_metric = bool(outputs_metric)
        # The eigenvalue box the forward pass squashes into. For M the envelope
        # simply inverts: w_lb <= eig(W) <= w_ub  <=>  1/w_ub <= eig(M) <= 1/w_lb.
        self._eig_lo = (1.0 / w_ub) if self.outputs_metric else w_lb
        self._eig_hi = (1.0 / w_lb) if self.outputs_metric else w_ub
        self.angle_idx = list(angle_idx)
        self.sym = sym

        if isinstance(activation, str):
            activation = _ACT_MAP[activation.lower()]()
        # Network sees the continuous embedding minus the translation directions;
        # W(x) itself stays x_dim x x_dim.
        self.model = MLP(
            input_dim=(sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx)),
            hidden_dims=hidden_dim, activation=activation,
        )

        self.model.to(device)

        out_dim = x_dim * x_dim
        self.mu = nn.Linear(self.model.output_dim, out_dim).to(device)
        self.logstd = nn.Linear(self.model.output_dim, out_dim).to(device)

    def _to_bounded_spd(self, flat: torch.Tensor) -> torch.Tensor:
        """The spectral sigmoid, without singular eigenvector derivatives."""
        raw = flat.reshape(-1, self.x_dim, self.x_dim)
        S = .5 * (raw + raw.mT)
        if not torch.isfinite(S).all():
            raise FloatingPointError("Nonfinite metric logits")
        if not S.requires_grad:
            eigenvalues, Q = torch.linalg.eigh(S)
            bounded = self._eig_lo + (self._eig_hi - self._eig_lo) * eigenvalues.sigmoid()
            return Q @ torch.diag_embed(bounded) @ Q.mT
        # Scale to avoid exp overflow, then use sigmoid(2S) = P²/(P²+(I-P)²).
        # This is the same matrix function Q sigmoid(eig(S)) Q', including at
        # repeated eigenvalues, and supports the second derivatives CCM needs.
        norm = float(S.detach().abs().sum(-1).amax())
        squarings = max(0, math.ceil(math.log2(max(norm, 1.))))
        I = torch.eye(self.x_dim, dtype=S.dtype, device=S.device).expand_as(S)
        P = torch.linalg.solve(I + torch.matrix_exp(-S / (2 ** squarings)), I)
        for _ in range(squarings):
            P2, complement = P @ P, I - P
            P = torch.linalg.solve(P2 + complement @ complement, P2)
            P = .5 * (P + P.mT)
        P = .5 * (P + P.mT)
        return self._eig_lo * I + (self._eig_hi - self._eig_lo) * P

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        logits = self.model(self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        mu = self.mu(logits)

        # Return-dict keys mirror CCM_Generator so the two are drop-in compatible.
        if self.mode == "deterministic" or deterministic:
            W = self._to_bounded_spd(mu)
            logprobs = torch.zeros(x.shape[0], 1, device=x.device)
            return W, {
                "dist": None,
                "probs": torch.ones_like(logprobs),
                "logprobs": logprobs,
                "entropy": torch.zeros_like(logprobs),
            }

        logstd = self.logstd(logits).clamp(-5, 2)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        sample = dist.rsample()
        W = self._to_bounded_spd(sample)
        logprobs = dist.log_prob(sample).sum(-1, keepdim=True)
        return W, {
            "dist": dist,
            "probs": torch.exp(logprobs),
            "logprobs": logprobs,
            "entropy": dist.entropy().sum(-1, keepdim=True),
        }


class AnalyticSOSMetric(nn.Module):
    """M(x) = W(x)^-1 evaluated from the SOS polynomial itself.

    Drop-in for ``BoundedCCM_Generator`` wherever a frozen metric is wanted, but
    with no network and nothing fitted: ``solvers.sos_cm`` certifies W(x) as an
    algebraic identity over the whole box, so regressing it into an MLP only adds
    error. Measured on toy-mg: the regression's median relative error in the
    DEPLOYED metric is 2.5%, which is not what "exact contraction metric" means,
    and it is the same M the offline V* solve uses -- any gap between them would
    show up as apparent suboptimality that is really a metric mismatch.

    Exposes ``bounded=True`` and ``outputs_metric=True`` so ``bound_W`` is a
    pass-through and ``BaseEnv._metric_from_cmg`` returns this forward directly.
    Being a polynomial it is exactly differentiable in x, which ``local_lambda``
    and the ``lambda_grad`` reference law both need.
    """

    def __init__(self, coeffs: dict, w_degree: int, x_dim: int = 2,
                 w_lb: float = 0.0, box=None):
        super().__init__()
        if x_dim != 2:
            raise ValueError(f"AnalyticSOSMetric is 2-state only, got {x_dim}; "
                             f"sos_cm's polynomial basis is 2-variable.")
        self.x_dim, self.w_lb = x_dim, float(w_lb)
        self.bounded, self.outputs_metric = True, True

        exps = [e for a in range(w_degree + 1) for b in range(w_degree + 1 - a)
                for e in ((a, b),)]
        self.register_buffer("exps", torch.tensor(exps, dtype=torch.float32))
        c = torch.zeros(3, len(exps))
        for k, (a, b) in enumerate(exps):
            for j, (i0, i1) in enumerate(((0, 0), (0, 1), (1, 1))):
                c[j, k] = float(coeffs[f"w_{i0}{i1}_{a}{b}"])
        self.register_buffer("coef", c)

        # W is PD on the box by certificate, so the closed-form 2x2 inverse below
        # needs no runtime guard -- but only if this really is that certificate.
        # Check once, here, rather than clamping a determinant every step and
        # hiding a mismatched or truncated coefficient set.
        if box is not None:
            lo, hi = box
            g = torch.stack(torch.meshgrid(
                torch.linspace(float(lo[0]), float(hi[0]), 65),
                torch.linspace(float(lo[1]), float(hi[1]), 65),
                indexing="ij"), -1).reshape(-1, 2)
            w00, w01, w11 = self._entries(g)
            det = w00 * w11 - w01 ** 2
            if float(det.min()) <= 0 or float(w00.min()) <= 0:
                raise ValueError(
                    f"the supplied SOS coefficients do not give a positive-"
                    f"definite W on the box (min det {float(det.min()):.3e}, "
                    f"min W00 {float(w00.min()):.3e}). M = W^-1 would be garbage "
                    f"and every Mahalanobis reward with it.")

    def _entries(self, x):
        basis = (x.unsqueeze(1).abs().clamp_min(1e-30) ** self.exps.unsqueeze(0)).prod(-1)
        # abs+clamp only guards 0**0; restore the sign the exponents imply.
        sign = torch.where((x.unsqueeze(1) < 0) & (self.exps.unsqueeze(0) % 2 == 1),
                           -1.0, 1.0).prod(-1)
        basis = basis * sign
        w = basis @ self.coef.T                              # (N, 3)
        return w[:, 0], w[:, 1], w[:, 2]

    def forward(self, x):
        w00, w01, w11 = self._entries(x)
        det = w00 * w11 - w01 ** 2
        M = torch.stack([torch.stack([w11, -w01], -1),
                         torch.stack([-w01, w00], -1)], -2) / det[:, None, None]
        return M, None

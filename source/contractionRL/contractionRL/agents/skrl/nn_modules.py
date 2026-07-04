"""Self-contained neural network modules for contractionRL.

Provides all network building blocks needed by C3M, SD-LQR, LQR, and C2RL with
no mjrl dependency.
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_MIN_LOG_STD = math.log(0.01)  # ≈ -4.605; annealing floor


# ─────────────────────────────────────────────────────────────────────────── #
# MLP
# ─────────────────────────────────────────────────────────────────────────── #

class MLP(nn.Module):
    """Standard feedforward MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        try:
            gain = nn.init.calculate_gain("tanh")
        except Exception:
            gain = 1.0
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(in_d, out_d)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.1)
            layers += [lin, activation]
        self.output_dim = dims[-1]
        if output_dim is not None:
            lin = nn.Linear(dims[-1], output_dim)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.0)
            layers.append(lin)
            self.output_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────── #
# CCM_Generator
# ─────────────────────────────────────────────────────────────────────────── #

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
    ):
        super().__init__()
        self.x_dim = x_dim
        self.mode = mode

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()]

        self.backbone = MLP(x_dim, list(hidden_dim), activation=activation)
        h = self.backbone.output_dim
        self.mu_head = nn.Linear(h, x_dim * x_dim)
        self.logstd_head = nn.Linear(h, x_dim * x_dim)
        self.to(device)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        n = x.shape[0]
        h = self.backbone(x)
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


# ─────────────────────────────────────────────────────────────────────────── #
# CLActor
# ─────────────────────────────────────────────────────────────────────────── #

class CLActor(nn.Module):
    """C3M_U contracting controller: u = W2(x,xref) @ tanh(W1(x,xref) @ e).

    The controller is differentiable in x so that K = du/dx feeds the contraction
    condition without needing a separate Jacobian network.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        mode: str = "deterministic",
        anneal_stddev: bool = False,
        hidden_dim: list[int] | None = None,
        activation: nn.Module | str = nn.Tanh(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.mode = mode
        assert mode in ("deterministic", "stochastic")

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()]

        hidden = list(hidden_dim) if hidden_dim else [128, 128]
        self.c = 3 * x_dim           # latent multiplier (matches CAC-dev)
        input_dim = 2 * x_dim        # [x, xref]

        self.w1 = MLP(input_dim, hidden, self.c * x_dim, activation=activation)
        self.w2 = MLP(input_dim, hidden, self.c * u_dim, activation=activation)

        self.anneal = anneal_stddev
        self.logstd = nn.Parameter(torch.zeros(1, u_dim), requires_grad=not anneal_stddev)
        self._init_logstd = 0.0

    def anneal_stddev(self, progress: float, mode: str = "exponential") -> None:
        """Anneal log_std from 0 to log(0.01) (≈-4.6) — prevents KL collapse."""
        if not self.anneal:
            return
        progress = float(max(0.0, min(1.0, progress)))
        ratio = progress ** 5.0 if mode == "exponential" else progress
        new_logstd = self._init_logstd * (1.0 - ratio) + _MIN_LOG_STD * ratio
        with torch.no_grad():
            self.logstd.data.fill_(float(max(_MIN_LOG_STD, min(2.0, new_logstd))))

    def trim_state(self, state: torch.Tensor):
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.u_dim]
        return x, xref, uref

    def mean_control(self, state: torch.Tensor) -> torch.Tensor:
        x, xref, uref = self.trim_state(state)
        x_xref = torch.cat((x, xref), dim=-1)
        n = x.shape[0]
        e = (x - xref).unsqueeze(-1)
        w1 = self.w1(x_xref).reshape(n, self.c, self.x_dim)
        w2 = self.w2(x_xref).reshape(n, self.u_dim, self.c)
        l1 = F.tanh(torch.matmul(w1, e))
        return uref + torch.matmul(w2, l1).squeeze(-1)

    def forward(self, state: torch.Tensor):
        x, xref, uref = self.trim_state(state)
        x_xref = torch.cat((x, xref), dim=-1)
        n = x.shape[0]
        e = (x - xref).unsqueeze(-1)
        w1 = self.w1(x_xref).reshape(n, self.c, self.x_dim)
        w2 = self.w2(x_xref).reshape(n, self.u_dim, self.c)
        l1 = F.tanh(torch.matmul(w1, e))
        mu = uref + torch.matmul(w2, l1).squeeze(-1)

        if self.mode == "deterministic":
            zeros = torch.zeros(n, 1, device=state.device)
            return mu, {"dist": None, "logprobs": zeros, "entropy": zeros}

        logstd = torch.clamp(self.logstd, _MIN_LOG_STD, 2)
        std = logstd.exp().expand_as(mu)
        dist = Normal(mu, std)
        u = dist.rsample()
        logprobs = dist.log_prob(u).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return u, {"dist": dist, "logprobs": logprobs, "entropy": entropy}

    def log_prob(self, dist, controls):
        return dist.log_prob(controls).sum(-1, keepdim=True)

    def entropy(self, dist):
        return dist.entropy().sum(-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────── #
# NeuralDynamics
# ─────────────────────────────────────────────────────────────────────────── #

_ACT_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class NeuralDynamics(nn.Module):
    """Control-affine neural dynamics  ẋ = f(x) + B(x)·u.

    Trained online by C3MAgent from trajectory buffer transition data and then
    loaded by SDLQRAgent / LQRAgent / C2RLAgent via ``NeuralDynamics.load()``.
    """

    _dtype = torch.float32

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: Sequence[int] = (256, 256),
        activation: str = "relu",
        device: str | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.null_dim = x_dim - u_dim
        self._hidden_dim = list(hidden_dim)
        self._activation_str = activation
        self.device = torch.device(device or "cpu")

        act = _ACT_MAP.get(activation, nn.ReLU)()
        self.f_net = MLP(x_dim, list(hidden_dim), x_dim, activation=act)
        self.B_net = MLP(x_dim, list(hidden_dim), x_dim * u_dim, activation=act)
        self.to(self.device)

    def get_f_and_B(self, x: torch.Tensor):
        """Return (f, B, B_null) — autodiff-compatible for C3M Jacobians."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self._dtype).to(self.device)
        f = self.f_net(x)
        B = self.B_net(x).reshape(-1, self.x_dim, self.u_dim)
        B_null = self._compute_B_null(B)
        return f, B, B_null

    def _compute_B_null(self, B: torch.Tensor) -> torch.Tensor:
        n = B.shape[0]
        if self.null_dim <= 0:
            return torch.zeros(n, self.x_dim, 1, device=B.device, dtype=B.dtype)
        with torch.no_grad():
            U, _, _ = torch.linalg.svd(B.detach(), full_matrices=True)
        return U[:, :, self.u_dim:].contiguous()

    def predict_x_dot(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        f, B, _ = self.get_f_and_B(x)
        return f + (B @ u.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0)
        return self.get_f_and_B(x)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "x_dim": self.x_dim,
                "u_dim": self.u_dim,
                "hidden_dim": self._hidden_dim,
                "activation": self._activation_str,
                "state_dict": self.state_dict(),
            },
            path,
        )
        print(f"[NeuralDynamics] saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "NeuralDynamics":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            x_dim=ckpt["x_dim"],
            u_dim=ckpt["u_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
            device=device,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[NeuralDynamics] loaded ← {path}  (x_dim={ckpt['x_dim']}, u_dim={ckpt['u_dim']})")
        return model

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            result.device = torch.device(args[0])
        return result

class BoundedCCM_Generator(nn.Module):
    """CMG with hard eigenvalue bounds baked into the forward pass."""
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
    ):
        super().__init__()

        self.x_dim = x_dim
        self.mode = mode
        self.device = device
        self.w_lb = w_lb
        self.w_ub = w_ub

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}.get(
                activation.lower(), nn.Tanh()
            )
        self.model = MLP(
            input_dim=x_dim, hidden_dims=hidden_dim,
            activation=activation,
        )
        
        self.model.to(device)

        out_dim = x_dim * x_dim
        self.mu = nn.Linear(self.model.output_dim, out_dim).to(device)
        self.logstd = nn.Linear(self.model.output_dim, out_dim).to(device)

    def _to_bounded_spd(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape → symmetrise → sigmoid-on-eigenvalues → bounded SPD."""
        n = flat.shape[0]
        S_raw = flat.view(n, self.x_dim, self.x_dim)
        S = 0.5 * (S_raw + S_raw.mT)           # symmetric; eigenvalues span ℝ
        if S.device.type == "mps":             # eigh unsupported on MPS → run on CPU
            lam, V = torch.linalg.eigh(S.cpu())
            lam, V = lam.to(S.device), V.to(S.device)
        else:                                  # CUDA/CPU: keep it on-device
            lam, V = torch.linalg.eigh(S)
        lam = self.w_lb + (self.w_ub - self.w_lb) * torch.sigmoid(lam)
        return V @ torch.diag_embed(lam) @ V.mT  # SPD, λ ∈ (w_lb, w_ub)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        logits = self.model(x)
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

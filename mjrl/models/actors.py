"""Contracting controller network for C3M (ported from CAC-dev ``policy_networks.py``).

``CLActor`` is the C3M_U controller: it builds state-dependent weight matrices
W1(x,xref), W2(x,xref) and maps the tracking error e = x - xref through
u = W2 · tanh(W1 · e), giving a controller that is differentiable in x (so its
Jacobian K = du/dx feeds the contraction condition).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from mjrl.models.building_blocks import MLP


def get_activation(activation):
    """Resolve an activation given an nn.Module or a string name."""
    if isinstance(activation, nn.Module):
        return activation
    name = str(activation).lower()
    table = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {activation}")
    return table[name]


def get_u_model(x_dim: int, u_dim: int, hidden_dims: list = None, activation=nn.Tanh()):
    """Two weight-generator MLPs (w1, w2) for the C3M_U controller."""
    if hidden_dims is None:
        hidden_dims = [128]
    input_dim = 2 * x_dim   # [x, xref]
    c = 3 * x_dim           # latent multiplier
    activation = get_activation(activation)
    w1 = MLP(input_dim, list(hidden_dims), c * x_dim, activation=activation)
    w2 = MLP(input_dim, list(hidden_dims), c * u_dim, activation=activation)
    return w1, w2


class CLActor(nn.Module):
    """C3M_U deterministic/stochastic contracting controller."""

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        mode: str = "deterministic",
        anneal_stddev: bool = False,
        hidden_dim: list = None,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.mode = mode
        assert mode in ("deterministic", "stochastic")

        self.w1, self.w2 = get_u_model(x_dim, u_dim, hidden_dims=hidden_dim, activation=activation)
        self.anneal = anneal_stddev
        self.logstd = nn.Parameter(torch.zeros(1, u_dim), requires_grad=not anneal_stddev)

    def trim_state(self, state: torch.Tensor):
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.u_dim]
        return x, xref, uref

    def forward(self, state: torch.Tensor):
        x, xref, _ = self.trim_state(state)
        x_xref = torch.cat((x, xref), dim=-1)
        n = x.shape[0]
        e = (x - xref).unsqueeze(-1)

        w1 = self.w1(x_xref).reshape(n, -1, self.x_dim)
        w2 = self.w2(x_xref).reshape(n, self.u_dim, -1)
        l1 = F.tanh(torch.matmul(w1, e))
        mu = torch.matmul(w2, l1).squeeze(-1)

        if self.mode == "deterministic":
            u = mu
            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
        else:
            logstd = torch.clip(self.logstd, -20, 2)
            std = torch.exp(logstd.expand_as(mu))
            dist = Normal(loc=mu, scale=std)
            u = dist.rsample()
            logprobs = dist.log_prob(u).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().sum(1)

        return u, {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy}

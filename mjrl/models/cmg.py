"""Contraction-metric generator W(x) (ported from CAC-dev ``CMG_networks.py``).

Generates a state-dependent SPD matrix W(x) = WᵀW. ``deterministic`` mode uses
the network mean; stochastic mode samples and reports log-prob/entropy (used by
the entropy regulariser in TEMP/CARL-style synthesis).
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch.distributions import Normal

from mjrl.models.building_blocks import MLP


class CCM_Generator(nn.Module):
    bounded = False

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list,
        activation: Union[str, nn.Module] = "tanh",
        mode: str = "stochastic",
        device: str = "cpu",
    ):
        super().__init__()
        self.x_dim = x_dim
        self.mode = mode
        self.device = device

        if isinstance(activation, str):
            name = activation.lower()
            if name == "tanh":
                activation = nn.Tanh()
            elif name == "relu":
                activation = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.model = MLP(input_dim=x_dim, hidden_dims=list(hidden_dim), activation=activation, device=device)
        self.mu = nn.Linear(hidden_dim[-1], x_dim * x_dim)
        self.logstd = nn.Linear(hidden_dim[-1], x_dim * x_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        n = x.shape[0]
        logits = self.model(x)
        mu = self.mu(logits)

        if self.mode == "deterministic" and deterministic:
            W = mu
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
            dist = None
        else:
            logstd = torch.clamp(self.logstd(logits), min=-5, max=2)
            std = torch.exp(logstd)
            dist = Normal(loc=mu, scale=std)
            W = dist.rsample()
            logprobs = dist.log_prob(W).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().unsqueeze(-1).sum(1)

        W = W.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)  # ensure symmetry / PSD
        return W, {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy}

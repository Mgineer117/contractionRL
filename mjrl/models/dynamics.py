"""Neural control-affine dynamics model for mjrl.

Learns  ẋ = f_net(x) + B_net(x) @ u
from (x, u, x_dot) transition data so that any environment without
analytical dynamics can still be used with LQR and C3M.

``get_f_and_B(x)`` is the primary interface consumed by agents:
  * x: torch.Tensor (n, x_dim) — may carry requires_grad for C3M Jacobians
  * returns: (f, B, B_null) all as tensors, autodiff-compatible

``forward(x)`` is the hook installed as ``env.learned_dynamics_model``:
  * accepts numpy or tensor (classic env simulation path)
  * returns (f, B, B_null) as tensors (env_base converts to numpy)
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from mjrl.models.building_blocks import MLP

_ACT_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}


class NeuralDynamics(nn.Module):
    """Control-affine neural dynamics  ẋ = f(x) + B(x)·u.

    ``f_net`` outputs the drift vector f(x) ∈ ℝ^{x_dim}.
    ``B_net`` outputs the input matrix B(x) ∈ ℝ^{x_dim × u_dim} (flattened).

    B_null (the orthogonal complement of col(B)) is computed via full SVD on
    demand — it is detached from the graph so it does not block gradient flow
    through f and B.
    """

    _dtype = torch.float32

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: Sequence[int] = (256, 256, 256),
        activation: str = "relu",
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.null_dim = x_dim - u_dim
        self._hidden_dim = list(hidden_dim)
        self._activation_str = activation

        act = _ACT_MAP.get(activation, nn.ReLU())

        self.f_net = MLP(
            input_dim=x_dim,
            hidden_dims=list(hidden_dim),
            output_dim=x_dim,
            activation=act,
        )
        self.B_net = MLP(
            input_dim=x_dim,
            hidden_dims=list(hidden_dim),
            output_dim=x_dim * u_dim,
            activation=act,
        )

        self.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # primary interface for agents (autodiff-compatible)
    # ------------------------------------------------------------------ #
    def get_f_and_B(self, x: torch.Tensor):
        """Return (f, B, B_null) as tensors; respects the autograd graph.

        x: (n, x_dim) tensor, possibly with requires_grad for C3M Jacobians.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self._dtype).to(self.device)

        f = self.f_net(x)                                      # (n, x_dim)
        B = self.B_net(x).reshape(-1, self.x_dim, self.u_dim)  # (n, x_dim, u_dim)
        B_null = self._compute_B_null(B)                        # (n, x_dim, null_dim)
        return f, B, B_null

    # ------------------------------------------------------------------ #
    # forward — hook for env.learned_dynamics_model (simulation path)
    # ------------------------------------------------------------------ #
    def forward(self, x):
        """Called by env_base.py's get_f_and_B under torch.no_grad()."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0)
        return self.get_f_and_B(x)

    # ------------------------------------------------------------------ #
    # training utility
    # ------------------------------------------------------------------ #
    def predict_x_dot(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """x_dot = f(x) + B(x) @ u   (used for MSE training loss)."""
        f, B, _ = self.get_f_and_B(x)
        return f + (B @ u.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ #
    def _compute_B_null(self, B: torch.Tensor) -> torch.Tensor:
        """Orthogonal complement of col(B) via full SVD (detached)."""
        n = B.shape[0]
        if self.null_dim <= 0:
            return torch.zeros(n, self.x_dim, 1, device=B.device, dtype=B.dtype)
        with torch.no_grad():
            # full SVD: U columns beyond u_dim span null(B^T)
            U, _, _ = torch.linalg.svd(B.detach(), full_matrices=True)  # (n, x, x)
        return U[:, :, self.u_dim:].contiguous()  # (n, x, null_dim)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
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
        )
        model.load_state_dict(ckpt["state_dict"])
        if device:
            model = model.to(torch.device(device))
            model.device = torch.device(device)
        model.eval()
        print(f"[NeuralDynamics] loaded ← {path}  (x_dim={ckpt['x_dim']}, u_dim={ckpt['u_dim']})")
        return model

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            result.device = torch.device(args[0])
        return result

"""LQR — analytical state-feedback tracking controller.

Linearises the dynamics about the reference (A = Df/Dx + Σ uref_j ∂B_j/∂x, B = B(xref)),
solves the continuous-time algebraic Riccati equation, and applies
u = uref - K (x - xref). It has no learnable parameters: ``learn`` is a no-op and
the trainer just evaluates it.

Ported from CAC-dev ``policy/lqr.py``; now inherits the mjrl ``Base`` so the
autograd Jacobian helpers and ``_dtype`` are available.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch.linalg import solve

from mjrl.models.base import Base


class LQR(Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        get_f_and_B: Callable,
        Q_scaler: float = 1.0,
        R_scaler: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = "LQR"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.action_dim = u_dim

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.get_f_and_B = get_f_and_B

        # contraction-bound metadata (consumed by env.render diagnostics)
        self.lbd = 0.0
        self.gamma = 1.0
        self.w_ub = 1.0
        self.w_lb = 1.0

        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def _trim(self, state: torch.Tensor):
        x = state[:, : self.x_dim].requires_grad_()
        xref = state[:, self.x_dim : 2 * self.x_dim].requires_grad_()
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.u_dim].requires_grad_()
        return x, xref, uref

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x, xref, uref = self._trim(state)
        xref = xref.requires_grad_()

        with torch.enable_grad():
            f_xref, B_xref, _ = self.get_f_and_B(xref)
            if isinstance(f_xref, np.ndarray):
                f_xref = torch.from_numpy(f_xref).to(self._dtype).to(self.device)
            if isinstance(B_xref, np.ndarray):
                B_xref = torch.from_numpy(B_xref).to(self._dtype).to(self.device)
            if f_xref.dim() == 1:
                f_xref = f_xref.unsqueeze(0)
            if B_xref.dim() == 2:
                B_xref = B_xref.unsqueeze(0)

            DfDx = self.Jacobian(f_xref, xref)        # (1, x, x)
            DBDx = self.B_Jacobian(B_xref, xref)      # (1, x, x, u)

        A = DfDx.clone().squeeze(0)
        for j in range(self.u_dim):
            A = A + uref[0, j] * DBDx[0, :, :, j]
        B = B_xref.to(self._dtype).squeeze(0)

        Q = (self.Q_scaler + 1e-5) * torch.eye(self.x_dim, dtype=self._dtype, device=self.device)
        R = (self.R_scaler + 1e-5) * torch.eye(self.u_dim, dtype=self._dtype, device=self.device)

        P_np = solve_continuous_are(
            A.detach().cpu().numpy(), B.detach().cpu().numpy(),
            Q.detach().cpu().numpy(), R.detach().cpu().numpy(),
        )
        P = torch.from_numpy(P_np).to(A)
        K = solve(R, B.T @ P)  # (u, x)

        # Return the control DEVIATION -K·e. The env applies uref + action, so the
        # realised control is the standard LQR tracking law uref - K·e. (uref is
        # still used above to linearise the dynamics for the Riccati solve.)
        e = x - xref
        u = -(K @ e.unsqueeze(-1)).squeeze(-1)

        return u, {"probs": self.dummy, "logprobs": self.dummy, "entropy": self.dummy}

    def learn(self, *args, **kwargs):
        """LQR has no trainable parameters."""
        return {}, {}, 0.0

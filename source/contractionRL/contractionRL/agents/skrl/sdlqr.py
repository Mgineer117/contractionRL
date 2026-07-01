"""SD-LQR / LQR — skrl-native analytical tracking agents.

Both agents are analytical (no learnable parameters) — ``update()`` is a no-op.
Use with skrl's SequentialTrainer.

SDLQRAgent: linearises at the *current* state x.
  A(x) = ∂f/∂x|_x + Σⱼ uref_j ∂Bⱼ/∂x|_x,  B = B(x)

LQRAgent: linearises at the *reference* xref.
  A(xref) = ∂f/∂x|_{xref} + Σⱼ uref_j ∂Bⱼ/∂x|_{xref},  B = B(xref)

Both solve the continuous-time algebraic Riccati equation (CARE) to get the
optimal gain K and apply u = uref - K·(x - xref).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch.linalg import solve

from skrl.agents.torch.base import Agent, AgentCfg

from .math_utils import b_jacobian, jacobian


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SDLQRCfg(AgentCfg):
    Q_scaler: float = 1.0
    R_scaler: float = 0.0


LQRCfg = SDLQRCfg


# ─────────────────────────────────────────────────────────────────────────── #
# Shared CARE solver
# ─────────────────────────────────────────────────────────────────────────── #

def _care_gain(A: torch.Tensor, B_mat: torch.Tensor,
               Q_scaler: float, R_scaler: float, x_dim: int, u_dim: int) -> torch.Tensor:
    """Solve CARE and return gain K = R⁻¹BᵀP → (u_dim, x_dim)."""
    dtype = A.dtype
    Q = (Q_scaler + 1e-5) * torch.eye(x_dim, dtype=dtype, device=A.device)
    R = (R_scaler + 1e-5) * torch.eye(u_dim, dtype=dtype, device=A.device)
    P_np = solve_continuous_are(
        A.detach().cpu().numpy(),
        B_mat.detach().cpu().numpy(),
        Q.detach().cpu().numpy(),
        R.detach().cpu().numpy(),
    )
    P = torch.from_numpy(P_np).to(A)
    return solve(R, B_mat.T @ P)  # (u_dim, x_dim)


# ─────────────────────────────────────────────────────────────────────────── #
# SD-LQR Agent
# ─────────────────────────────────────────────────────────────────────────── #

class SDLQRAgent(Agent):
    """State-Dependent LQR wrapped as a native skrl Agent.

    Linearises at the current state x, solves CARE per step, applies
    u = uref - K(x)·(x - xref).  No learnable parameters.

    Extra constructor kwarg:
      ``get_f_and_B``: ``(x) -> (f, B, Bbot)``
    """

    def __init__(
        self,
        *,
        cfg: SDLQRCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_f_and_B: Callable,
        x_dim: int | None = None,
        u_dim: int | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = SDLQRCfg(**{k: v for k, v in cfg.items() if k in SDLQRCfg.__dataclass_fields__})
        super().__init__(
            cfg=cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        obs_dim = int(observation_space.shape[0])
        u_dim_inferred = int(action_space.shape[0])
        
        if u_dim is None:
            u_dim = u_dim_inferred
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._cfg = cfg
        self._get_f_and_B = get_f_and_B
        self._compute_device = "cpu"  # Jacobians must be on CPU for scipy CARE

    def _compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Linearise at x, solve CARE, return u = uref - K·(x - xref)."""
        x_dim, u_dim = self._x_dim, self._u_dim
        cfg = self._cfg
        batch_size = obs.shape[0]

        x    = obs[:, :x_dim].float().to(self._compute_device).requires_grad_()
        xref = obs[:, x_dim : 2 * x_dim].float().to(self._compute_device)
        uref = obs[:, 2 * x_dim : 2 * x_dim + u_dim].float().to(self._compute_device)

        with torch.enable_grad():
            f, B, _ = self._get_f_and_B(x)
        f = f.float().to(self._compute_device)
        B = B.float().to(self._compute_device)

        DfDx = jacobian(f, x, create_graph=False)                      # (batch, x, x)
        DBDx = b_jacobian(B, x, u_dim, create_graph=False)             # (batch, x, x, u)
        
        actions = torch.zeros(batch_size, u_dim, device=self._compute_device)
        
        for i in range(batch_size):
            A = DfDx[i].clone()
            for j in range(u_dim):
                A = A + uref[i, j] * DBDx[i, :, :, j]
            B_mat = B[i]

            K = _care_gain(A, B_mat, cfg.Q_scaler, cfg.R_scaler, x_dim, u_dim)
            e = x[i] - xref[i]
            u = uref[i] - K @ e
            actions[i] = u

        return actions

    def act(self, observations, states, *, timestep: int, timesteps: int):
        orig_device = observations.device
        # Jacobians need grad enabled; scipy CARE is non-differentiable so safe.
        with torch.enable_grad():
            actions = self._compute_action(observations)
        actions = actions.detach().to(orig_device)
        log_prob = torch.zeros(actions.shape[0], 1, device=orig_device)
        return actions, {"log_prob": log_prob}

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def record_transition(
        self, *, observations, states, actions, rewards, next_observations,
        next_states, terminated, truncated, infos, timestep, timesteps,
    ) -> None:
        super().record_transition(
            observations=observations, states=states, actions=actions,
            rewards=rewards, next_observations=next_observations,
            next_states=next_states, terminated=terminated, truncated=truncated,
            infos=infos, timestep=timestep, timesteps=timesteps,
        )

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass  # analytical — no gradient updates


# ─────────────────────────────────────────────────────────────────────────── #
# LQR Agent (linearises at xref, not x)
# ─────────────────────────────────────────────────────────────────────────── #

class LQRAgent(Agent):
    """LQR wrapped as a native skrl Agent — linearises at reference trajectory.

    Analytical → ``update()`` is a no-op.

    Extra constructor kwarg:
      ``get_f_and_B``: ``(x) -> (f, B, Bbot)``
    """

    def __init__(
        self,
        *,
        cfg: LQRCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_f_and_B: Callable,
        x_dim: int | None = None,
        u_dim: int | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = LQRCfg(**{k: v for k, v in cfg.items() if k in LQRCfg.__dataclass_fields__})
        super().__init__(
            cfg=cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        obs_dim = int(observation_space.shape[0])
        u_dim_inferred = int(action_space.shape[0])
        
        if u_dim is None:
            u_dim = u_dim_inferred
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._cfg = cfg
        self._get_f_and_B = get_f_and_B
        self._compute_device = "cpu"

    def _compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Linearise at xref, solve CARE, return u = uref - K·(x - xref)."""
        x_dim, u_dim = self._x_dim, self._u_dim
        cfg = self._cfg
        batch_size = obs.shape[0]

        x     = obs[:, :x_dim].float().to(self._compute_device)
        xref  = obs[:, x_dim : 2 * x_dim].float().to(self._compute_device).requires_grad_()
        uref  = obs[:, 2 * x_dim : 2 * x_dim + u_dim].float().to(self._compute_device)

        with torch.enable_grad():
            f_xref, B_xref, _ = self._get_f_and_B(xref)
        f_xref = f_xref.float().to(self._compute_device)
        B_xref = B_xref.float().to(self._compute_device)

        DfDx = jacobian(f_xref, xref, create_graph=False)              # (batch, x, x)
        DBDx = b_jacobian(B_xref, xref, u_dim, create_graph=False)     # (batch, x, x, u)
        
        actions = torch.zeros(batch_size, u_dim, device=self._compute_device)

        for i in range(batch_size):
            A = DfDx[i].clone()
            for j in range(u_dim):
                A = A + uref[i, j] * DBDx[i, :, :, j]
            B_mat = B_xref[i]

            K = _care_gain(A, B_mat, cfg.Q_scaler, cfg.R_scaler, x_dim, u_dim)
            e = x[i] - xref[i]
            u = uref[i] - K @ e
            actions[i] = u
            
        return actions

    def act(self, observations, states, *, timestep: int, timesteps: int):
        orig_device = observations.device
        with torch.enable_grad():
            actions = self._compute_action(observations)
        actions = actions.detach().to(orig_device)
        log_prob = torch.zeros(actions.shape[0], 1, device=orig_device)
        return actions, {"log_prob": log_prob}

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def record_transition(
        self, *, observations, states, actions, rewards, next_observations,
        next_states, terminated, truncated, infos, timestep, timesteps,
    ) -> None:
        super().record_transition(
            observations=observations, states=states, actions=actions,
            rewards=rewards, next_observations=next_observations,
            next_states=next_states, terminated=terminated, truncated=truncated,
            infos=infos, timestep=timestep, timesteps=timesteps,
        )

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass

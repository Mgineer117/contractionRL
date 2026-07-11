"""SD-LQR / LQR — skrl-native analytical tracking agents.

Both agents are analytical (no learnable parameters) — ``update()`` is a no-op.
Use with skrl's SequentialTrainer.

SDLQRAgent: linearises at the *current* state x.
  A(x) = ∂f/∂x|_x + Σⱼ uref_j ∂Bⱼ/∂x|_x,  B = B(x)

LQRAgent: linearises at the *reference* xref.
  A(xref) = ∂f/∂x|_{xref} + Σⱼ uref_j ∂Bⱼ/∂x|_{xref},  B = B(xref)

Both solve the continuous-time algebraic Riccati equation (CARE) to get the
optimal gain K and apply u = uref - K·(x - xref).

Per-``act()`` workflow (there is no training loop — every step is this,
see ``_compute_action``):

  1. Split the raw observation into ``(x, xref, uref)``.
  2. Autodiff ``get_f_and_B`` (analytical, classic envs, or a loaded
     ``NeuralDynamics.get_f_and_B`` pretrained by C3M — see
     ``C3MAgent.save_dynamics``) at the linearization point (``x`` for
     SD-LQR, ``xref`` for LQR) to get ``f, B`` and their Jacobians
     ``∂f/∂x, ∂B/∂x``.
  3. Form ``A = ∂f/∂x + Σⱼ uref_j·∂Bⱼ/∂x`` per-environment (batched).
  4. Solve the CARE (``scipy.linalg.solve_continuous_are``, CPU-only, one
     linear system per environment in a Python loop — see ``_care_gain``) for
     the gain ``K = R⁻¹BᵀP``; falls back to zero feedback (``u = uref``) if
     ``(A, B)`` isn't stabilizable at that linearization point rather than
     aborting the batch.
  5. Apply ``u = uref - K·(x - xref)``.

Normalization: **none**, and none is meaningful here — there are no learned
weights whose input distribution could drift, only a per-step closed-form
solve on the raw physical state. ``_compute_device = "cpu"`` is unrelated to
normalization: it exists only because ``scipy``'s CARE solver requires CPU
numpy arrays, so the Jacobian computation for this step is done on CPU
regardless of the environment's own device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch.linalg import solve

from skrl.agents.torch.base import Agent, AgentCfg

from .angle_utils import wrap_diff
from .math_utils import b_jacobian, jacobian
from .rl_glue import filter_cfg_fields


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SDLQRCfg(AgentCfg):
    Q_scaler: float = 1.0
    # R must be strictly positive: R_scaler = 0 → R ≈ 1e-5·I → huge gains
    # K = R⁻¹BᵀP → discrete-dt divergence. Default to 1.0.
    R_scaler: float = 1.0


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
    try:
        P_np = solve_continuous_are(
            A.detach().cpu().numpy(),
            B_mat.detach().cpu().numpy(),
            Q.detach().cpu().numpy(),
            R.detach().cpu().numpy(),
        )
    except (np.linalg.LinAlgError, ValueError):
        # (A, B) not stabilizable / no finite CARE solution at this linearization
        # point → fall back to zero feedback (u = uref) rather than aborting the
        # entire batched rollout for one bad env.
        return torch.zeros(u_dim, x_dim, dtype=dtype, device=A.device)
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
        angle_idx: list | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = SDLQRCfg(**filter_cfg_fields(cfg, SDLQRCfg, context="SDLQRAgent"))
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
        self._angle_idx = angle_idx or []
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

        DfDx = jacobian(f, x, create_graph=False)
        DBDx = b_jacobian(B, x, u_dim, create_graph=False)
        f = f.detach(); B = B.detach()
        
        A_batch = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, uref)
        
        actions = torch.zeros(batch_size, u_dim, device=self._compute_device)
        e_batch = wrap_diff(x - xref, self._angle_idx)
        
        for i in range(batch_size):
            A = A_batch[i]
            B_mat = B[i]

            K = _care_gain(A, B_mat, cfg.Q_scaler, cfg.R_scaler, x_dim, u_dim)
            e = e_batch[i]
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
        angle_idx: list | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = LQRCfg(**filter_cfg_fields(cfg, LQRCfg, context="LQRAgent"))
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
        self._angle_idx = angle_idx or []
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

        A_batch = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, uref)
        
        actions = torch.zeros(batch_size, u_dim, device=self._compute_device)
        e_batch = wrap_diff(x - xref, self._angle_idx)

        for i in range(batch_size):
            A = A_batch[i]
            B_mat = B_xref[i]

            K = _care_gain(A, B_mat, cfg.Q_scaler, cfg.R_scaler, x_dim, u_dim)
            e = e_batch[i]
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

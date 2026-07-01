"""Pure-PyTorch math utilities for contraction RL algorithms.

These standalone functions provide the Jacobian computations, positive-definiteness
losses, and metric utilities needed by C3M, SD-LQR, LQR, and TEMP — with no mjrl
dependency.
"""
from __future__ import annotations

import torch
from torch import matmul, transpose
from torch.autograd import grad


def jacobian(f: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Batched autograd Jacobian ∂f/∂x → (n, f_dim, x_dim)."""
    n, f_dim, x_dim = x.shape[0], f.shape[-1], x.shape[-1]
    v = torch.eye(f_dim, device=x.device, dtype=x.dtype).unsqueeze(1).expand(f_dim, n, f_dim)
    J = grad(f, x, grad_outputs=v, create_graph=create_graph, retain_graph=True, is_grads_batched=True)[0]
    return J.transpose(0, 1)


def b_jacobian(B: torch.Tensor, x: torch.Tensor, u_dim: int, create_graph: bool = True) -> torch.Tensor:
    """Jacobian of each column of B: ∂B_j/∂x → (n, x_dim, x_dim, u_dim)."""
    n, x_dim, _ = B.shape
    B_flat = B.reshape(n, x_dim * u_dim)
    J_flat = jacobian(B_flat, x, create_graph=create_graph)
    J = J_flat.reshape(n, x_dim, u_dim, x_dim)
    return J.transpose(2, 3)


def weighted_gradients(W: torch.Tensor, v: torch.Tensor, x: torch.Tensor,
                       detach: bool = False, create_graph: bool = True) -> torch.Tensor:
    """Material derivative Ẇ = Σᵢ (∂W/∂xᵢ) vᵢ → (n, d, d)."""
    assert v.size() == x.size()
    z = torch.ones_like(W, requires_grad=True)
    g = grad(W, x, grad_outputs=z, create_graph=True, retain_graph=True)[0]
    S = (g * v).sum()
    dot_W = grad(S, z, create_graph=create_graph)[0]
    if detach:
        dot_W = dot_W.detach()
    return dot_W


def loss_pos_matrix_random_sampling(A: torch.Tensor, reg: bool = True):
    """PD loss via random projections: relu(-zᵀAz).mean() → (loss, reg_loss)."""
    n, d, _ = A.shape
    device, dtype = A.device, A.dtype
    z = torch.randn(n, d, device=device, dtype=dtype)
    z = z / z.norm(dim=-1, keepdim=True)
    z = z.unsqueeze(-1)
    zTAz = (transpose(z, 1, 2) @ A @ z)
    loss_eigen = torch.relu(-zTAz).mean()
    loss_reg = torch.relu(zTAz - 200).mean() if reg else torch.zeros(1, device=device, dtype=dtype)
    return loss_eigen, loss_reg


def loss_pos_matrix_eigen(A: torch.Tensor, reg: bool = True):
    """PD loss via eigenvalues: relu(-λ).sum() → (loss, reg_loss)."""
    device = A.device
    if device.type == "mps":
        lambdas = torch.linalg.eigvalsh(A.cpu()).to(device)
    else:
        lambdas = torch.linalg.eigvalsh(A)
    loss_eigen = torch.relu(-lambdas).sum(dim=-1).mean()
    loss_reg = torch.zeros(1, device=device, dtype=A.dtype)
    if reg:
        loss_reg = loss_reg + torch.relu(lambdas[:, -1] - 500).mean()
    return loss_eigen, loss_reg


def bound_W(raw_W: torch.Tensor, w_lb: float, x_dim: int, bounded: bool = False) -> torch.Tensor:
    """Add w_lb * I to raw CCM output to ensure strict positive definiteness."""
    if bounded:
        return raw_W
    I = torch.eye(x_dim, device=raw_W.device, dtype=raw_W.dtype)
    return raw_W + w_lb * I

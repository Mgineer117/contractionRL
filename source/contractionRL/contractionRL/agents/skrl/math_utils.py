"""Pure-PyTorch math utilities for contraction RL algorithms.

These standalone functions provide the Jacobian computations, positive-definiteness
losses, and metric utilities needed by C3M, SD-LQR, LQR, and C2RL — with no mjrl
dependency.
"""
from __future__ import annotations

import torch
from torch.autograd import grad


def jacobian(f: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Batched autograd Jacobian ∂f/∂x → (n, f_dim, x_dim)."""
    n, f_dim, x_dim = x.shape[0], f.shape[-1], x.shape[-1]
    if not f.requires_grad:
        return torch.zeros((n, f_dim, x_dim), device=x.device, dtype=x.dtype)
    J = torch.zeros(n, f_dim, x_dim, device=x.device, dtype=x.dtype)
    for i in range(f_dim):
        g = grad(f[:, i].sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            J[:, i, :] = g
    return J


def b_jacobian(B: torch.Tensor, x: torch.Tensor, u_dim: int, create_graph: bool = True) -> torch.Tensor:
    """Jacobian of each column of B: ∂B_j/∂x → (n, x_dim, x_dim, u_dim)."""
    n, x_dim, _ = B.shape
    B_flat = B.reshape(n, x_dim * u_dim)
    J_flat = jacobian(B_flat, x, create_graph=create_graph)
    J = J_flat.reshape(n, x_dim, u_dim, x_dim)
    return J.transpose(2, 3)


def Jacobian_Matrix(M: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n, device=x.device, dtype=x.dtype)
    for i in range(m):
        for j in range(m):
            g = grad(M[:, i, j].sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                J[:, i, j, :] = g
    return J


def weighted_gradients(W: torch.Tensor, v: torch.Tensor, x: torch.Tensor,
                       detach: bool = False, create_graph: bool = True) -> torch.Tensor:
    """Material derivative Ẇ = Σᵢ (∂W/∂xᵢ) vᵢ → (n, d, d)."""
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x, create_graph).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x, create_graph) * v.view(bs, 1, 1, -1)).sum(dim=3)


def loss_pos_matrix_random_sampling(A: torch.Tensor, num_samples: int = 1024) -> torch.Tensor:
    """PD-violation hinge loss via random directional sampling (no eigendecomposition).

    Projects ``A`` onto ``num_samples`` random unit vectors and penalizes only the
    directions where ``zᵀAz < 0``. Unlike an exact eigenvalue loss, this never
    differentiates through ``eigh``/``eigvalsh``, so it stays numerically stable
    even when double-differentiated (as happens via ``weighted_gradients``).
    """
    device, dtype = A.device, A.dtype
    d = A.size(-1)
    z = torch.randn(num_samples, d, device=device, dtype=dtype)
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1, num_samples, -1)).sum(dim=2).reshape(-1)
    negative = zTAz < 0
    if negative.any():
        return -zTAz[negative].mean()
    return torch.zeros((), device=device, dtype=dtype, requires_grad=True)


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


def spd_inverse(W: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of a batch of SPD matrices via Cholesky.

    ``W`` is SPD by construction (``VᵀV + w_lb·I`` or eigenvalue-bounded), so a
    Cholesky solve is both faster and more stable than the general LU-based
    ``torch.linalg.solve`` — and it won't silently return garbage if ``W`` drifts
    ill-conditioned.
    """
    n, d, _ = W.shape
    I = torch.eye(d, device=W.device, dtype=W.dtype).expand(n, d, d).contiguous()
    L = torch.linalg.cholesky(W)
    return torch.cholesky_solve(I, L)

"""Base class with the autograd/linear-algebra utilities shared by mjrl algorithms.

Ported and trimmed from CAC-dev ``policy/base.py``. Provides:
  * tensor/device helpers (``to_tensor``, ``to_device``)
  * the canonical LR schedule (``lr_decay_lambda`` driven by ``self.progress``)
  * batched Jacobians (``Jacobian``, ``Jacobian_Matrix``, ``B_Jacobian``)
  * ``weighted_gradients`` for contraction-metric derivatives
  * positive-definiteness losses (``loss_pos_matrix_*``)
  * eigenvalue recording/plotting for diagnostics
  * the contraction-metric lower-bound map ``_bound_W``
  * ``trim_state`` splitting an observation into (x, xref, uref, t)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul, transpose
from torch.autograd import grad


class Base(nn.Module, ABC):
    """Math/utility base for every mjrl contraction algorithm."""

    def __init__(self):
        super().__init__()

        self._dtype = torch.float32
        self.device = None  # algorithms set their own device

        # 0->1 fraction of training completed, consumed by lr_decay_lambda.
        self.progress = 0.0

        # diagnostics
        self.Cu_eigenvalues_records = []
        self.dot_M_eigenvalues_records = []
        self.sym_mabk_eigenvalues_records = []
        self.overshoot_records = []

        # common losses
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

    # ------------------------------------------------------------------ #
    # tensor / device helpers
    # ------------------------------------------------------------------ #
    def to_tensor(self, data) -> torch.Tensor:
        return torch.from_numpy(data).to(self._dtype).to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)
        # move optimizer state too (matters on MPS / multiprocessing)
        for attr_value in self.__dict__.values():
            if isinstance(attr_value, torch.optim.Optimizer):
                for _, state in attr_value.state.items():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

    # ------------------------------------------------------------------ #
    # LR schedule (shared)
    # ------------------------------------------------------------------ #
    def lr_decay_lambda(self, _):
        """Linear decay 1.0 -> 0.0 driven by ``self.progress`` (arg ignored)."""
        return max(0.0, 1.0 - self.progress)

    # ------------------------------------------------------------------ #
    # state / metric helpers
    # ------------------------------------------------------------------ #
    def trim_state(self, state: torch.Tensor):
        """Split an observation into (x, xref, uref, t)."""
        x = state[:, : self.x_dim].requires_grad_()
        xref = state[:, self.x_dim : 2 * self.x_dim].requires_grad_()
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.u_dim].requires_grad_()
        t = state[:, -1].unsqueeze(-1)
        return x, xref, uref, t

    def _bound_W(self, raw_W: torch.Tensor) -> torch.Tensor:
        """Map a raw CMG output to a usable contraction-metric inverse W.

        Standard CCM_Generator returns a PSD matrix (WᵀW); add w_lb*I for a
        strict positive-definite lower bound. Bounded generators already encode
        the bound and are returned as-is.
        """
        if getattr(self.CMG, "bounded", False):
            return raw_W
        I = torch.eye(self.x_dim, device=self.device, dtype=raw_W.dtype)
        return raw_W + self.w_lb * I

    # ------------------------------------------------------------------ #
    # gradient / weight norms
    # ------------------------------------------------------------------ #
    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is None:
                continue
            total_norm = torch.tensor(0.0, device=device)
            try:
                for param in model.parameters():
                    if param.grad is not None:
                        total_norm += torch.norm(param.grad, p=norm_type) ** norm_type
            except Exception:
                try:
                    total_norm += torch.norm(model.grad, p=norm_type) ** norm_type
                except Exception:
                    pass
            total_norm = total_norm ** (1.0 / norm_type)
            grad_dict[dir + "/grad/" + names[i]] = total_norm.item()
        return grad_dict

    # ------------------------------------------------------------------ #
    # eigenvalue diagnostics
    # ------------------------------------------------------------------ #
    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            A_sym = 0.5 * (A + A.transpose(-1, -2))
            try:
                if A_sym.device.type == "mps":
                    eigvals = torch.linalg.eigvalsh(A_sym.cpu())
                else:
                    eigvals = torch.linalg.eigvalsh(A_sym)
            except torch.linalg.LinAlgError:
                eigvals = torch.full(A_sym.shape[:-1], float("nan"), device=A.device)
        return eigvals.mean(0).cpu().numpy()

    def record_eigenvalues(self, Cu, dot_M, sym_MABK, overshoot):
        with torch.no_grad():
            self.Cu_eigenvalues_records.append(self.get_matrix_eig(Cu))
            self.dot_M_eigenvalues_records.append(self.get_matrix_eig(dot_M))
            self.sym_mabk_eigenvalues_records.append(self.get_matrix_eig(sym_MABK))
            self.overshoot_records.append(self.get_matrix_eig(overshoot))

    def get_eigenvalue_plot(self):
        num = 10
        if len(self.Cu_eigenvalues_records) < num:
            return None
        x = list(range(0, len(self.Cu_eigenvalues_records), num))

        def _stats(records):
            arr = np.asarray(records[::num])
            return arr.mean(axis=1), arr.max(axis=1), arr.min(axis=1)

        Cu_mean, Cu_max, Cu_min = _stats(self.Cu_eigenvalues_records)
        dM_mean, dM_max, dM_min = _stats(self.dot_M_eigenvalues_records)
        sm_mean, sm_max, sm_min = _stats(self.sym_mabk_eigenvalues_records)
        os_mean, os_max, os_min = _stats(self.overshoot_records)

        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0, 0].plot(x, Cu_mean); ax[0, 0].fill_between(x, Cu_max, Cu_min, alpha=0.2)
        ax[0, 0].set_title("Cu Eigenvalues")
        ax[0, 1].plot(x, dM_mean); ax[0, 1].fill_between(x, dM_max, dM_min, alpha=0.2)
        ax[0, 1].set_title("Dot M Eigenvalues")
        ax[1, 0].plot(x, sm_mean); ax[1, 0].fill_between(x, sm_max, sm_min, alpha=0.2)
        ax[1, 0].set_title("Sym MABK Eigenvalues")
        ax[1, 1].plot(x, os_mean); ax[1, 1].fill_between(x, os_max, os_min, alpha=0.2)
        ax[1, 1].set_title("Overshoot Eigenvalues")
        for a in ax.flat:
            a.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------ #
    # positive-definiteness losses
    # ------------------------------------------------------------------ #
    def loss_pos_matrix_random_sampling(self, A: torch.Tensor, reg: bool = True):
        n, A_dim, _ = A.shape
        z = torch.randn((n, A_dim)).to(dtype=self._dtype, device=self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)
        zTAz = matmul(matmul(zT, A), z)
        loss_eigen = torch.relu(-zTAz).mean()
        loss_reg = torch.relu(zTAz - 200).mean()
        return loss_eigen, (loss_reg if reg else 0)

    def loss_pos_matrix_eigen(self, A: torch.Tensor, reg: bool = True):
        if A.device.type == "mps":
            lambdas = torch.linalg.eigvalsh(A.cpu()).to(A.device)
        else:
            lambdas = torch.linalg.eigvalsh(A)
        loss_eigen = torch.relu(-lambdas).sum(dim=-1).mean()
        loss_reg = torch.tensor(0.0, device=self.device)
        if reg:
            loss_reg = loss_reg + torch.relu(lambdas[:, -1] - 500).mean()
        return loss_eigen, loss_reg

    # ------------------------------------------------------------------ #
    # autograd Jacobians
    # ------------------------------------------------------------------ #
    def Jacobian(self, f: torch.Tensor, x: torch.Tensor, create_graph: bool = True):
        f = f + 0.0 * x.sum()
        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]
        J = torch.zeros(n, f_dim, x_dim).to(dtype=self._dtype, device=self.device)
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=create_graph, retain_graph=True)[0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor, create_graph: bool = True):
        n = x.shape[0]
        matrix_dim = M.shape[-1]
        x_dim = x.shape[-1]
        J = torch.zeros(n, matrix_dim, matrix_dim, x_dim).to(dtype=self._dtype, device=self.device)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=create_graph, retain_graph=True)[0]
        return J

    def B_Jacobian(self, B: torch.Tensor, x: torch.Tensor, create_graph: bool = True):
        n = x.shape[0]
        x_dim = x.shape[-1]
        DBDx = torch.zeros(n, x_dim, x_dim, self.u_dim).to(dtype=self._dtype, device=self.device)
        for i in range(self.u_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i], x, create_graph=create_graph)
        return DBDx

    def weighted_gradients(self, W, v, x, detach: bool = False, create_graph: bool = True):
        assert v.size() == x.size()
        bs = x.shape[0]
        JM = self.Jacobian_Matrix(W, x, create_graph=create_graph)
        if detach:
            JM = JM.detach()
        return (JM * v.view(bs, 1, 1, -1)).sum(dim=3)

    # ------------------------------------------------------------------ #
    @abstractmethod
    def learn(self, *args, **kwargs):
        """Main training step for the algorithm."""
        raise NotImplementedError

"""C3M — Control Contraction Metric synthesis.

Jointly learns a contraction metric W(x) (via the CMG) and a contracting
controller u(x, xref) (via ``CLActor``) so that the closed loop satisfies the
control-contraction conditions:

  * Cu  = Ṁ + 2·sym(M(A+BK)) + 2λM  ≺ 0       (closed-loop contraction)
  * C1  = Bᗩᵀ(-Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bᗩ ≺ 0   (orthogonal / drift)
  * C2  = Bᗩᵀ(Ẇ_b - 2·sym(∂B/∂x·W))Bᗩ = 0           (compatibility)

Training data are random (x, xref, uref) triples sampled directly by
``env.get_rollout(buffer_size, "c3m")`` — no environment rollouts are needed.

Ported from CAC-dev ``policy/c3m.py``.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from mjrl.models.base import Base


class C3M(Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        CMG: nn.Module,
        actor: nn.Module,
        data: dict,
        get_f_and_B: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        cmg_updates_per_policy_update: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.nupdates = nupdates
        self.cmg_updates_per_policy_update = max(1, int(cmg_updates_per_policy_update))

        self.CMG = CMG
        self.actor = actor
        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            self.get_f_and_B.eval()

        self.eps = eps
        self.w_ub = w_ub
        self.w_lb = w_lb
        self.lbd = lbd
        self.gamma = 1.0  # for render diagnostics (C3M bound has no 1/(1-γ))

        self.W_optimizer = torch.optim.Adam(self.CMG.parameters(), lr=W_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=u_lr)
        self.W_lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.lr_decay_lambda)
        self.actor_lr_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.lr_decay_lambda)

        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    # ------------------------------------------------------------------ #
    # deployment (sampler / evaluator interface)
    # ------------------------------------------------------------------ #
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        a, _ = self.actor(state)
        return a, {"probs": self.dummy, "logprobs": self.dummy, "entropy": self.dummy}

    # ------------------------------------------------------------------ #
    # contraction loss
    # ------------------------------------------------------------------ #
    def compute_loss(self):
        I = torch.eye(self.x_dim, device=self.device)

        buffer_size = self.data["x"].shape[0]
        batch_size = min(1024, buffer_size)
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        batch = {key: self.data[key][indices] for key in self.data.keys()}

        x = self.to_tensor(batch["x"]).requires_grad_()
        xref = self.to_tensor(batch["xref"])
        uref = self.to_tensor(batch["uref"])

        raw_W, _ = self.CMG(x)
        W = self._bound_W(raw_W)
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        f, B, Bbot = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)
        B = B.to(self._dtype).to(self.device)
        Bbot = Bbot.to(self._dtype).to(self.device)

        DfDx = self.Jacobian(f, x)
        DBDx = self.B_Jacobian(B, x)
        f = f.detach(); B = B.detach(); Bbot = Bbot.detach()

        state = torch.cat([x, xref, uref], dim=1)
        u, _ = self.actor(state)
        K = self.Jacobian(u, x)

        A = DfDx + sum(
            u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(self.u_dim)
        )
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M

        DWDx = self.Jacobian_Matrix(W, x)
        DfW = (DWDx * f.view(batch_size, 1, 1, -1)).sum(dim=3)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2s = []
        for j in range(self.u_dim):
            DbW = (DWDx * B[:, :, j].view(batch_size, 1, 1, -1)).sum(dim=3)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
            C2_inner = DbW - 2 * sym_DbDxW
            C2s.append(matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot))

        overshoot = W - self.w_ub * I
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)
        C1 = C1 + self.eps * torch.eye(C1.shape[-1], device=self.device)
        c2_loss = sum([(C2 ** 2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])

        pd_loss, pd_reg = self.loss_pos_matrix_random_sampling(-Cu)
        c1_loss, c1_reg = self.loss_pos_matrix_random_sampling(-C1)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_random_sampling(-overshoot)

        self.record_eigenvalues(Cu, dot_M, sym_MABK, overshoot)

        loss = overshoot_loss + pd_loss + c1_loss + c2_loss + pd_reg + c1_reg + overshoot_reg
        return loss, {"pd_loss": pd_loss, "c1_loss": c1_loss, "c2_loss": c2_loss}

    def optimize_params(self, loss, optimizer, module, name: str):
        self.W_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
            self.W_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            return {}
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm([module], [name], dir="C3M", device=self.device)
        optimizer.step()
        return grad_dict

    def learn(self, *args, **kwargs):
        self.train()
        t0 = time.time()
        self.progress = min(1.0, self.num_updates / max(1, self.nupdates))

        grad_dict = {}
        for _ in range(self.cmg_updates_per_policy_update):
            loss, infos = self.compute_loss()
            grad_dict.update(self.optimize_params(loss, self.W_optimizer, self.CMG, "CMG"))

        loss, infos = self.compute_loss()
        grad_dict.update(self.optimize_params(loss, self.actor_optimizer, self.actor, "actor"))

        self.W_lr_scheduler.step()
        self.actor_lr_scheduler.step()

        n_epochs = self.cmg_updates_per_policy_update
        prev_updates = self.num_updates
        self.num_updates += n_epochs

        supp_dict = {}
        if prev_updates // 500 != self.num_updates // 500:
            fig = self.get_eigenvalue_plot()
            if fig is not None:
                supp_dict["C3M/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/lr/W_lr": self.W_lr_scheduler.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.actor_lr_scheduler.get_last_lr()[0],
        }
        loss_dict.update(grad_dict)
        self.eval()
        return loss_dict, supp_dict, time.time() - t0

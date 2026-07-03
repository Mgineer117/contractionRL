"""C3M — skrl-native Control Contraction Metric agent.

Jointly learns:
  * W(x)        — contraction-metric generator (CCM_Generator / CMGModel)
  * u(x,xref)   — contracting controller (CLActor / CLActorModel)

Training uses random (x, xref, uref) triples from ``env.get_rollout`` — no
environment rollouts needed.  Use with C3MSkrlTrainer.

Contraction conditions verified jointly:
  Cu = Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0     (closed-loop)
  C1 = Bᗩᵀ(-Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bᗩ ≺ 0
  C2 = Bᗩᵀ(Ẇ_b - 2·sym(∂B/∂x·W))Bᗩ = 0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tqdm as _tqdm
from torch import matmul, transpose

from skrl.agents.torch.base import Agent, AgentCfg
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .math_utils import (
    b_jacobian,
    bound_W,
    jacobian,
    loss_pos_matrix_random_sampling,
    spd_inverse,
    weighted_gradients,
)
from .nn_modules import NeuralDynamics


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class C3MCfg(AgentCfg):
    batch_size: int = 1024
    W_lr: float = 3e-4
    u_lr: float = 3e-4
    lbd: float = 1e-2
    eps: float = 1e-2
    w_ub: float = 10.0
    w_lb: float = 1e-1
    # Fraction of training (by progress, 0-1) during which the metric M is held
    # fixed (detached) in the contraction term Cu, mirroring the reference C3M
    # script's `detach=True if epoch < lr_step else False` warmup (5 of 15
    # epochs ≈ 1/3). Lets the policy start becoming contracting w.r.t. a frozen
    # initial metric before the double-backward Ṁ term (numerically the most
    # fragile part of the loss) starts shaping both networks jointly.
    detach_warmup_frac: float = 1.0 / 3.0
    use_analytical_dynamics: bool = False
    learning_rate_scheduler: str = ""
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C3MTrainerCfg(TrainerCfg):
    timesteps: int = 30000
    eval_interval: int = 1000
    environment_info: str = "default"


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C3MAgent(Agent):
    """Control Contraction Metric agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict:
      ``"policy"`` — CLActorModel (contracting C3M_U controller)
      ``"cmg"``    — CMGModel     (contraction-metric generator)

    Extra constructor kwargs:
      ``get_rollout``:    ``(batch_size, mode) -> dict``
        mode="c3m"      → {"x", "xref", "uref"}
        mode="dynamics" → {"x", "u", "x_dot"}
      ``get_f_and_B``:   ``(x) -> (f, B, Bbot)`` — required when
        ``cfg.use_analytical_dynamics=True``; Isaac envs must leave this None
        (NeuralDynamics is trained online from trajectory buffer data).
    """

    def __init__(
        self,
        *,
        cfg: C3MCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_rollout: Callable,
        get_f_and_B: Callable | None = None,
        x_dim: int | None = None,
        u_dim: int | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = C3MCfg(**{k: v for k, v in cfg.items() if k in C3MCfg.__dataclass_fields__})
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
        self._device = device
        self._cfg = cfg

        dev_str = str(device) if not isinstance(device, str) else device

        # ── Dynamics: analytical or learned online ──────────────────────── #
        if cfg.use_analytical_dynamics:
            if get_f_and_B is None:
                raise ValueError(
                    "C3M: use_analytical_dynamics=True requires a get_f_and_B callable "
                    "(classic envs only). Isaac Sim envs have no analytical dynamics."
                )
            self._get_f_and_B = get_f_and_B
            self._neural_dynamics = None
            self._dynamics_optimizer = None
        else:
            self._neural_dynamics = models.get("dynamics", None)
            if self._neural_dynamics is None:
                raise ValueError("C3M requires 'dynamics' model in models dictionary when use_analytical_dynamics=False")
                
            self._get_f_and_B = self._neural_dynamics.get_f_and_B
            self._dynamics_optimizer = torch.optim.Adam(
                self._neural_dynamics.parameters(), lr=cfg.dynamics_lr
            )
            if hasattr(cfg, "dynamics_lr_scheduler") and cfg.dynamics_lr_scheduler:
                scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.dynamics_lr_scheduler)
                self._dynamics_lr_scheduler = scheduler_cls(self._dynamics_optimizer, **getattr(cfg, "dynamics_lr_scheduler_kwargs", {}))
            else:
                self._dynamics_lr_scheduler = None

        # ── Extract underlying nn.Module objects from skrl model wrappers ── #
        self._ccm_gen = models["cmg"].ccm_gen
        self._cl_actor = models["policy"].cl_actor

        # ── Optimizers + LR schedulers ──────────────────────────────────── #
        # Separated optimizers matching actual CAC-dev behavior to allow alternating updates
        # and gradient NaN-filtering.
        self._w_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=cfg.W_lr)
        self._u_optimizer = torch.optim.Adam(self._cl_actor.parameters(), lr=cfg.u_lr)
        self._progress = 0.0
        
        if getattr(cfg, "learning_rate_scheduler", None):
            scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.learning_rate_scheduler)
            kwargs = getattr(cfg, "learning_rate_scheduler_kwargs", {})
            self._w_lr_scheduler = scheduler_cls(self._w_optimizer, **kwargs)
            self._u_lr_scheduler = scheduler_cls(self._u_optimizer, **kwargs)
        else:
            self._w_lr_scheduler = None
            self._u_lr_scheduler = None

        # ── Data buffer (numpy; static for analytical, or pre-generated) ────── #
        self._memory_size = getattr(self.memory, "memory_size", 131072) if self.memory is not None else 131072
        self._data = get_rollout(self._memory_size, "c3m")
        self._get_rollout = get_rollout
        self._batch_size = cfg.batch_size
        self._num_updates = 0

        modules = {"policy": models["policy"], "cmg": models["cmg"]}
        if self._neural_dynamics is not None:
            modules["dynamics"] = self._neural_dynamics
        self.checkpoint_modules.update(modules)

    # ── skrl Agent interface ────────────────────────────────────────────── #

    def act(self, observations, states, *, timestep: int, timesteps: int):
        with torch.no_grad():
            result = self.models["policy"].act({"observations": observations}, role="policy")
            actions = result[0]
            outputs = result[-1] if len(result) > 2 else result[1]
        return actions, outputs

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
        self._progress = float(timestep) / max(1, timesteps)

        # 1. Online NeuralDynamics training (skipped when analytical)
        if self._neural_dynamics is not None:
            dyn_data = self._get_rollout(self._cfg.dynamics_batch_size, "dynamics")
            dyn_loss = self._train_dynamics(dyn_data)
            self.track_data("Loss / C3M/dynamics/mse", dyn_loss)

        # 2. Anneal CLActor log_std
        self.models["policy"].cl_actor.anneal_stddev(self._progress)

        # 3. Full epoch update (looping entire buffer in batch_size chunks)
        loss_dict = self._learn()
        for k, v in loss_dict.items():
            self.track_data(f"Loss / {k}", v)

    # ── Contraction math (inlined from mjrl C3M) ───────────────────────── #

    def _to_tensor(self, arr) -> torch.Tensor:
        dev = self._device
        return torch.from_numpy(arr).to(torch.float32).to(dev)

    def _compute_loss(self, idx):
        cfg = self._cfg
        device = self._device
        x_dim, u_dim = self._x_dim, self._u_dim
        I = torch.eye(x_dim, device=device)

        buf = self._data
        batch_size = len(idx)

        x     = self._to_tensor(buf["x"][idx]).requires_grad_()
        xref  = self._to_tensor(buf["xref"][idx])
        uref  = self._to_tensor(buf["uref"][idx])

        raw_W, _ = self._ccm_gen(x)
        bounded = getattr(self._ccm_gen, "bounded", False)
        W = bound_W(raw_W, cfg.w_lb, x_dim, bounded)
        M = spd_inverse(W)

        with torch.enable_grad():
            f, B, Bbot = self._get_f_and_B(x)
        f    = f.to(torch.float32).to(device)
        B    = B.to(torch.float32).to(device)
        Bbot = Bbot.to(torch.float32).to(device)

        DfDx = jacobian(f, x, create_graph=False).detach()
        DBDx = b_jacobian(B, x, u_dim, create_graph=False).detach()
        f = f.detach(); B = B.detach(); Bbot = Bbot.detach()

        # Certify the *deterministic* controller: use the mean control, not a
        # noisy rsample. Exploration noise would otherwise perturb A and dot_x
        # (and hence the contraction condition Cu) — this matches TEMP's CMG loss.
        state = torch.cat([x, xref, uref], dim=1)
        u = self._cl_actor.mean_control(state)
        K = jacobian(u, x)

        A = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, u)
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)

        # Early-training warmup: freeze the metric M inside the contraction term
        # so the double-backward Ṁ path (the most numerically fragile part of the
        # loss) doesn't jointly destabilize the CMG while it's still poorly
        # conditioned right after init. The actor still gets gradient signal via
        # K (below), just w.r.t. a momentarily-fixed metric. See detach_warmup_frac.
        detach = self._progress < cfg.detach_warmup_frac
        dot_M = weighted_gradients(M, dot_x, x, detach=detach)
        M_eff = M.detach() if detach else M

        ABK = A + matmul(B, K)
        MABK = matmul(M_eff, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * cfg.lbd * M_eff

        DfW = weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * cfg.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        c2_loss = torch.zeros(1, device=device)
        for j in range(u_dim):
            DbW = weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
            C2_inner = DbW - 2 * sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)
            c2_loss = c2_loss + (C2 ** 2).reshape(batch_size, -1).sum(1).mean()
            
        Cu_reg = Cu + cfg.eps * I
        C1_reg = C1 + cfg.eps * torch.eye(C1.shape[-1], device=device)

        # Random-projection PD loss (reference C3M script): never differentiates
        # through an eigendecomposition, unlike an exact eigenvalue loss — keeps
        # the (already double-differentiated, via weighted_gradients) contraction
        # term numerically stable instead of NaN-ing out.
        pd_loss = loss_pos_matrix_random_sampling(-Cu_reg)
        c1_loss = loss_pos_matrix_random_sampling(-C1_reg)

        if bounded:
            os_loss = torch.zeros((), device=device)
        else:
            overshoot = W - cfg.w_ub * I
            os_loss = loss_pos_matrix_random_sampling(-overshoot)

        loss = os_loss + pd_loss + c1_loss + c2_loss
        return loss, {"pd_loss": pd_loss.item(), "c1_loss": c1_loss.item(), "c2_loss": c2_loss.item()}

    def _optimize_params(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> bool:
        self._w_optimizer.zero_grad()
        self._u_optimizer.zero_grad()
        loss.backward()

        # CAC-dev protection: filter out NaN/Inf gradients occasionally caused by eigvalsh
        # or numerical edge cases in the contraction metric.
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
            self._w_optimizer.zero_grad()
            self._u_optimizer.zero_grad()
            return False

        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        return True

    def _learn(self) -> dict:
        cfg = self._cfg
        self._ccm_gen.train(); self._cl_actor.train()

        buf = self._data
        n = buf["x"].shape[0]
        batch_size = cfg.batch_size

        # shuffle indices for the epoch; one full pass in batch_size chunks
        indices = np.random.permutation(n)
        iters = max(1, n // batch_size)
        total_pd = total_c1 = total_c2 = total_loss = 0.0

        pbar = tqdm(range(iters), desc=f"Epoch C3M Update", leave=False)
        for b in pbar:
            idx = indices[b * batch_size : (b + 1) * batch_size]
            
            # CMG (metric) updates holding controller fixed
            for _ in range(getattr(cfg, "cmg_updates_per_policy_update", 1)):
                loss, infos = self._compute_loss(idx)
                self._optimize_params(loss, self._w_optimizer, self._ccm_gen)
                
            # Controller (policy) update holding metric fixed
            loss, infos = self._compute_loss(idx)
            self._optimize_params(loss, self._u_optimizer, self._cl_actor)

            total_loss += loss.item()
            total_pd += infos["pd_loss"]
            total_c1 += infos["c1_loss"]
            total_c2 += infos["c2_loss"]

        # Step LR scheduler once per epoch
        if self._w_lr_scheduler is not None:
            self._w_lr_scheduler.step()
        if self._u_lr_scheduler is not None:
            self._u_lr_scheduler.step()

        self._ccm_gen.eval(); self._cl_actor.eval()

        return {
            "C3M/loss/loss":    total_loss / iters,
            "C3M/loss/pd_loss": total_pd / iters,
            "C3M/loss/c1_loss": total_c1 / iters,
            "C3M/loss/c2_loss": total_c2 / iters,
            "C3M/lr/lr":        self._u_lr_scheduler.get_last_lr()[0] if self._u_lr_scheduler else cfg.u_lr,
        }

    def _train_dynamics(self, data: dict) -> float:
        """MSE training of NeuralDynamics on (x, u, x_dot) data."""
        dev = self._neural_dynamics.device
        x     = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        u     = torch.as_tensor(data["u"], dtype=torch.float32, device=dev)
        x_dot = torch.as_tensor(data["x_dot"], dtype=torch.float32, device=dev)
        
        pred = self._neural_dynamics.predict_x_dot(x, u)
        loss = F.mse_loss(pred, x_dot)
        
        self._dynamics_optimizer.zero_grad()
        loss.backward()
        
        # Prevent NaNs from massive MSE losses by checking gradients and clipping
        if all(torch.isfinite(p.grad).all() for p in self._neural_dynamics.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._neural_dynamics.parameters(), 1.0)
            self._dynamics_optimizer.step()
            
        return loss.item()
    def save_dynamics(self, path: str) -> None:
        """Save NeuralDynamics checkpoint for SDLQR/LQR/TEMP to load."""
        if self._neural_dynamics is not None:
            self._neural_dynamics.save(path)
            print(f"[C3M] Saved NeuralDynamics → {path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C3MSkrlTrainer(Trainer):
    """skrl Trainer for C3M — no env interaction during training.

    Calls ``agent.update()`` in a tight loop.  Evaluation runs a few env episodes.
    """

    def train(self) -> None:
        agent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        timesteps = self.cfg.timesteps
        log_interval = getattr(agent, "write_interval", 100)
        if str(log_interval).lower() == "auto": log_interval = 100
        log_interval = int(log_interval)
        eval_interval = getattr(self.cfg, "eval_interval", 0)

        agent.init(trainer_cfg=self.cfg)
        agent.enable_training_mode(True)

        # Pretrain dynamics if needed
        epochs = getattr(agent.cfg, "dynamics_pretrain_epochs", 5)
        data_path = getattr(agent.cfg, "dynamics_pretrain_data_path", None)

        if agent._neural_dynamics is not None and epochs > 0:
            dev = agent._neural_dynamics.device
            if data_path:
                print(f"[C3M] Loading dynamics pretrain data from {data_path}")
                npz = np.load(data_path)
                
                # Check for NaNs in current offline data
                nan_mask = np.isnan(npz["x"]).any(axis=(1, 2)) | np.isnan(npz["u"]).any(axis=(1, 2)) | np.isnan(npz["x_dot"]).any(axis=(1, 2))
                if nan_mask.any():
                    num_nans = nan_mask.sum()
                    print(f"[C3M] WARNING: Found NaNs in {num_nans} offline episodes! Filtering them out...")
                    valid_mask = ~nan_mask
                    x_arr = npz["x"][valid_mask]
                    u_arr = npz["u"][valid_mask]
                    x_dot_arr = npz["x_dot"][valid_mask]
                else:
                    x_arr = npz["x"]
                    u_arr = npz["u"]
                    x_dot_arr = npz["x_dot"]
                    
                x = torch.from_numpy(x_arr).reshape(-1, x_arr.shape[-1]).to(torch.float32).to(dev)
                u = torch.from_numpy(u_arr).reshape(-1, u_arr.shape[-1]).to(torch.float32).to(dev)
                x_dot = torch.from_numpy(x_dot_arr).reshape(-1, x_dot_arr.shape[-1]).to(torch.float32).to(dev)
                n = x.shape[0]
                
                dbz = agent._cfg.dynamics_batch_size
                batches_per_epoch = max(1, n // dbz)
            else:
                x = u = x_dot = n = None
                batches_per_epoch = 1

            dyn_pbar = _tqdm.tqdm(range(epochs), desc="Pretraining dynamics", file=sys.stdout)
            for epoch in dyn_pbar:
                if x is not None:
                    # Offline pretraining: iterate over all batches in the epoch
                    for _ in range(batches_per_epoch):
                        dbz = min(agent._cfg.dynamics_batch_size, n)
                        idx = torch.randint(0, n, (dbz,), device=dev)
                        batch_data = {
                            "x": x[idx],
                            "u": u[idx],
                            "x_dot": x_dot[idx]
                        }
                        loss_val = agent._train_dynamics(batch_data)
                else:
                    # Online pretraining using rolling data (1 rollout = 1 epoch)
                    dyn_data = agent._get_rollout(agent._cfg.dynamics_batch_size, "dynamics")
                    loss_val = agent._train_dynamics(dyn_data)
                    
                dyn_pbar.set_postfix(loss=f"{loss_val:.3g}")
                
                # Step the LR scheduler every epoch
                if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
                    agent._dynamics_lr_scheduler.step()
                
                # Log to wandb
                agent.track_data("Loss / Pretrain/dynamics_mse", loss_val)
                if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
                    agent.track_data("C3M/lr/dynamics_lr", agent._dynamics_lr_scheduler.get_last_lr()[0])
                else:
                    agent.track_data("C3M/lr/dynamics_lr", agent._dynamics_optimizer.param_groups[0]["lr"])
                    
                if (epoch + 1) % log_interval == 0:
                    agent.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        pbar = _tqdm.tqdm(range(timesteps), desc="C3M training", file=sys.stdout)
        for t in pbar:
            agent.pre_interaction(timestep=t, timesteps=timesteps)
            agent.update(timestep=t, timesteps=timesteps)
            agent.post_interaction(timestep=t, timesteps=timesteps)

            # Evaluate metrics occasionally
            if eval_interval > 0 and (t + 1) % eval_interval == 0:
                eval_metrics = self.eval()
                for k, v in eval_metrics.items():
                    agent.track_data(f"Eval / {k}", v)

            if log_interval and (t + 1) % log_interval == 0:
                def _last(key):
                    v = agent.tracking_data.get(key, [float("nan")])
                    return v[-1] if isinstance(v, list) else float(v)
                postfix = dict(
                    loss=f"{_last('Loss / C3M/loss/loss'):.3g}",
                    pd=f"{_last('Loss / C3M/loss/pd_loss'):.3g}",
                )
                if agent._neural_dynamics is not None:
                    postfix["dyn"] = f"{_last('Loss / C3M/dynamics/mse'):.3g}"
                pbar.set_postfix(**postfix)
                
                # Write tracking data to wandb/tensorboard
                agent.write_tracking_data(timestep=t, timesteps=timesteps)

        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            agent.save_dynamics(dyn_path)

    def eval(self) -> dict:
        agent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        agent.enable_training_mode(False)
        observations, infos = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None
        
        x_dim = agent._x_dim
        auc_sum = torch.zeros((self.env.num_envs, 1), device=self.env.device)
        total_reward = torch.zeros((self.env.num_envs, 1), device=self.env.device)
        steps_count = torch.zeros((self.env.num_envs, 1), device=self.env.device)

        done = False
        while not done:
            with torch.no_grad():
                actions, _ = agent.act(observations, states, timestep=0, timesteps=1)
                
            x_curr = observations[:, :x_dim]
            x_ref = observations[:, x_dim:2*x_dim]
            error = torch.norm(x_curr - x_ref, dim=-1, keepdim=True)
            auc_sum += error
            
            observations, rewards, terminated, truncated, _ = self.env.step(actions)
            total_reward += rewards.view(self.env.num_envs, 1)
            steps_count += 1
            # Only stop once *every* env has finished, so per-env metrics are not
            # truncated by whichever env happens to terminate first.
            done = bool((terminated | truncated).all())

        agent.enable_training_mode(True)
        return {
            "reward_mean": total_reward.mean().item(),
            "auc": auc_sum.mean().item()
        }

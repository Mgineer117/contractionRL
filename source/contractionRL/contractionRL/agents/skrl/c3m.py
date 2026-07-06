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

Per-``update()`` workflow (called once per ``timestep`` by C3MSkrlTrainer,
see ``update()`` and ``_learn()`` below):

  1. **Dynamics** (skipped when ``use_analytical_dynamics``) — draw a fresh
     ``(x, u, x_dot)`` batch via ``get_rollout(..., "dynamics")`` and take one
     MSE gradient step on ``NeuralDynamics`` (``ẋ = f_net(x) + B_net(x)·u``).
     Isaac envs always use this path (no closed-form dynamics available);
     classic envs may instead set ``use_analytical_dynamics: true`` to use the
     env's own exact ``get_f_and_B(x)``, skipping this step entirely.
  2. Anneal ``policy.cl_actor``'s exploration ``log_std`` by training progress.
  3. **``_learn()``** — one full pass over ``self._data`` (a static/periodically
     -refreshed buffer of ``(x, xref, uref)`` triples from
     ``get_rollout(..., "c3m")``) in ``batch_size`` chunks. For each chunk:
       a. ``cmg_updates_per_policy_update`` gradient steps on the CMG
          (``_ccm_gen``) ALONE, controller held fixed (``_optimize_params``
          zeroes both optimizers but only steps ``_w_optimizer``).
       b. One gradient step on the controller (``_cl_actor``) ALONE, metric
          held fixed (steps ``_u_optimizer`` only).
     Both directions optimise the SAME combined loss (``pd_loss + c1_loss +
     c2_loss (+ os_loss)``, see ``_compute_loss``) — alternating which
     parameter group actually receives the gradient step is what keeps the
     joint (metric, controller) optimization stable (mirrors CAC-dev).

Normalization: **none**. Unlike C2RL, C3M's policy/CMG are bare
``nn.Module``s wrapped directly in skrl ``Model``s — they are never wrapped in
a ``PPO``/``SAC`` base agent, so there is no ``observation_preprocessor``/
``value_preprocessor`` anywhere in this file. Every network call
(``_compute_loss``, ``_train_dynamics``) sees the SAME raw ``(x, xref, uref)``
physical-unit values everywhere, at both training and eval time (``eval()``
below, and ``C3MSkrlTrainer.eval()``) — there is no train/eval or
loss-vs-inference input-distribution gap to worry about (contrast with
C2RL's CMG-loss vs. policy-training normalization mismatch, documented in
c2rl.py).
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
    # Number of CMG (metric) gradient steps per controller step within each
    # batch — read in _learn(). Must be a declared field or the runner's
    # dataclass-field filter silently drops it (always defaulting to 1).
    cmg_updates_per_policy_update: int = 1
    # Alias for u_lr: some configs/sweeps name the controller LR "actor_lr".
    # When set (non-None), __post_init__ copies it into u_lr so both spellings
    # take effect instead of being silently ignored.
    actor_lr: float | None = None
    lbd: float = 1e-2
    eps: float = 1e-2
    w_ub: float = 10.0
    w_lb: float = 0.1
    # Fraction of training (by progress, 0-1) during which the metric M is held
    # fixed (detached) in the contraction term Cu, mirroring the reference C3M
    # script's `detach=True if epoch < lr_step else False` warmup (5 of 15
    # epochs ≈ 1/3). Lets the policy start becoming contracting w.r.t. a frozen
    # initial metric before the double-backward Ṁ term (numerically the most
    # fragile part of the loss) starts shaping both networks jointly.
    detach_warmup_frac: float = 1.0 / 3.0
    # Random directions sampled per loss_pos_matrix_random_sampling() call (the
    # PD-violation hinge loss for pd_loss/c1_loss/os_loss). This is a pure
    # statistical-coverage knob — it does NOT affect numerical stability
    # (unlike an eigenvalue-decomposition loss, this method never
    # differentiates through an eigendecomposition at ANY sample count).
    # The reference script used 1024, but that's overkill for a loss called
    # every SGD step (not a one-shot certificate): for low-dimensional systems
    # (x_dim ~4-6, e.g. classic car/cartpole/segway/turtlebot) 1024 directions
    # measured ~13x slower per call than the old eigvalsh-based loss it
    # replaced, for negligible extra coverage. 128 stays comfortably above
    # every x_dim in this repo (~4-33) while costing only ~2x eigvalsh.
    pd_loss_num_samples: int = 128
    use_analytical_dynamics: bool = False
    learning_rate_scheduler: str = ""
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""

    def __post_init__(self):
        # "actor_lr" is an accepted alias for the controller learning rate u_lr.
        if self.actor_lr is not None:
            self.u_lr = self.actor_lr


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
        dyn_loss = None
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
        # Keep the latest losses on the agent so the trainer's progress-bar
        # postfix can read them even after post_interaction → write_tracking_data
        # clears tracking_data (otherwise the bar shows a spurious "nan").
        self._last_metrics = dict(loss_dict)
        if dyn_loss is not None:
            self._last_metrics["C3M/dynamics/mse"] = dyn_loss

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
        # (and hence the contraction condition Cu) — this matches C2RL's CMG loss.
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
        n_samp = cfg.pd_loss_num_samples
        pd_loss = loss_pos_matrix_random_sampling(-Cu_reg, num_samples=n_samp)
        c1_loss = loss_pos_matrix_random_sampling(-C1_reg, num_samples=n_samp)

        if bounded:
            # BoundedCCM_Generator already hard-bounds W's eigenvalues into
            # [w_lb, w_ub] via sigmoid in its forward pass — no soft penalty needed.
            os_loss = torch.zeros((), device=device)
        else:
            # Only an upper-bound penalty is needed: W = VᵀV + w_lb·I is bounded
            # BELOW by construction (bound_W), so a lower-bound loss on it would
            # be a permanent zero-gradient no-op (zᵀ(VᵀV)z = ‖Vz‖² ≥ 0 always).
            overshoot = W - cfg.w_ub * I
            os_loss = loss_pos_matrix_random_sampling(-overshoot, num_samples=n_samp)

        loss = os_loss + pd_loss + c1_loss + c2_loss
        return loss, {"pd_loss": pd_loss.item(), "c1_loss": c1_loss.item(), "c2_loss": c2_loss.item(), "os_loss": os_loss.item() if not bounded else 0.0}

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
        total_pd = total_c1 = total_c2 = total_os = total_loss = 0.0

        pbar = tqdm(range(iters), desc=f"Epoch C3M Update", leave=False)
        for b in pbar:
            idx = indices[b * batch_size : (b + 1) * batch_size]
            
            # CMG (metric) updates holding controller fixed
            for _ in range(cfg.cmg_updates_per_policy_update):
                loss, infos = self._compute_loss(idx)
                self._optimize_params(loss, self._w_optimizer, self._ccm_gen)
                
            # Controller (policy) update holding metric fixed
            loss, infos = self._compute_loss(idx)
            self._optimize_params(loss, self._u_optimizer, self._cl_actor)

            total_loss += loss.item()
            total_pd += infos["pd_loss"]
            total_c1 += infos["c1_loss"]
            total_c2 += infos["c2_loss"]
            total_os += infos["os_loss"]

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
            "C3M/loss/os_loss": total_os / iters,
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
        """Save NeuralDynamics checkpoint for SDLQR/LQR/C2RL to load."""
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
        eval_interval = getattr(self.cfg, "eval_interval", 0)
        log_interval = getattr(agent, "write_interval", "auto")
        if str(log_interval).lower() == "auto":
            log_interval = eval_interval if eval_interval > 0 else 200
        log_interval = int(log_interval)

        agent.init(trainer_cfg=self.cfg)
        agent.enable_training_mode(True)

        # Pretrain the learned NeuralDynamics before RL (same mechanism as
        # C2RL — see dynamics_pretrain.py; kept in one place so the two agents'
        # pretraining behavior can't drift apart). No-op for analytical
        # dynamics or dynamics_pretrain_epochs<=0.
        from .dynamics_pretrain import pretrain_dynamics
        pretrain_dynamics(
            agent,
            epochs=getattr(agent.cfg, "dynamics_pretrain_epochs", 5),
            data_path=getattr(agent.cfg, "dynamics_pretrain_data_path", None),
            timesteps=timesteps,
            log_interval=log_interval,
            tag="[C3M]",
        )

        pbar = _tqdm.tqdm(range(timesteps), desc="C3M training", file=sys.stdout)
        for t in pbar:
            agent.pre_interaction(timestep=t, timesteps=timesteps)
            agent.update(timestep=t, timesteps=timesteps)

            # Evaluate metrics occasionally
            if eval_interval > 0 and (t + 1) % eval_interval == 0:
                eval_metrics = self.eval()
                _stability_keys = {"auc", "contraction_rate", "overshoot", "contraction_score"}
                for k, v in eval_metrics.items():
                    tab = "Stability" if k in _stability_keys else "Reward"
                    # No space around "/" — must match path_tracking_base.py's
                    # own "Stability/..."/"Reward/..." keys exactly, or PPO/SAC
                    # (whose Stability metrics come from the env) and C3M
                    # (whose Stability metrics come from here) end up on two
                    # different wandb tabs ("Stability" vs "Stability ") for
                    # what should be the same metric.
                    agent.track_data(f"{tab}/{k}", v)

            agent.post_interaction(timestep=t, timesteps=timesteps)

            if log_interval and (t + 1) % log_interval == 0:
                # Read the losses captured on the agent by update(); tracking_data
                # may already be cleared by post_interaction → write_tracking_data,
                # which would otherwise make _last() return a spurious nan.
                metrics = getattr(agent, "_last_metrics", {})
                postfix = dict(
                    loss=f"{metrics.get('C3M/loss/loss', float('nan')):.3g}",
                    pd=f"{metrics.get('C3M/loss/pd_loss', float('nan')):.3g}",
                    os=f"{metrics.get('C3M/loss/os_loss', float('nan')):.3g}",
                )
                if agent._neural_dynamics is not None:
                    postfix["dyn"] = f"{metrics.get('C3M/dynamics/mse', float('nan')):.3g}"
                pbar.set_postfix(**postfix)

                # Write tracking data to wandb/tensorboard. The writer only
                # exists when skrl resolved write_interval > 0 (auto =
                # timesteps//100, so 0 for <100-step runs); guard so short runs
                # and writer-less configs log to the progress bar without crashing.
                if getattr(agent, "writer", None) is not None:
                    agent.write_tracking_data(timestep=t, timesteps=timesteps)

        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            agent.save_dynamics(dyn_path)

    def _env_scalar_attr(self, *names):
        """Fetch a scalar env attribute across both env backends.

        Isaac envs expose it directly on the skrl-wrapped env (shared across
        the batched sim, e.g. ``max_episode_length``/``step_dt``). Classic
        envs are N separate Python instances behind a gymnasium
        ``SyncVectorEnv`` (attr named e.g. ``max_episode_len``/``dt``), which
        does NOT forward arbitrary attribute access — only ``get_attr(name)``
        reaches the underlying sub-envs.
        """
        for name in names:
            val = getattr(self.env, name, None)
            if val is not None:
                return val
        if hasattr(self.env, "get_attr"):
            for name in names:
                try:
                    return self.env.get_attr(name)[0]
                except Exception:
                    continue
        raise AttributeError(f"none of {names} found on env {self.env!r}")

    def eval(self) -> dict:
        agent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        agent.enable_training_mode(False)
        observations, infos = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None

        x_dim = agent._x_dim
        num_envs = self.env.num_envs
        auc_sum = torch.zeros((num_envs, 1), device=self.env.device)
        total_reward = torch.zeros((num_envs, 1), device=self.env.device)
        steps_count = torch.zeros((num_envs, 1), device=self.env.device)
        e0 = torch.zeros((num_envs, 1), device=self.env.device)
        e_last = torch.zeros((num_envs, 1), device=self.env.device)
        e_max = torch.zeros((num_envs, 1), device=self.env.device)

        # Envs auto-reset individually on done (DirectRLEnv._reset_idx runs
        # inside step()), and termination is per-env (e.g. a quadruped falling
        # early) — so envs desynchronize and `(terminated | truncated).all()`
        # on a single step may never be true again once even one env has
        # reset off-cycle, hanging this loop indefinitely. Instead, run a
        # bounded number of steps and track which envs have finished their
        # FIRST episode, masking further accumulation for them once they have
        # (mirrors train.py's `_evaluate_best_model` quality-gate loop).
        max_steps = int(self._env_scalar_attr("max_episode_length", "max_episode_len")) + 1
        finished = torch.zeros(num_envs, dtype=torch.bool, device=self.env.device)

        for _ in range(max_steps):
            with torch.no_grad():
                actions, _ = agent.act(observations, states, timestep=0, timesteps=1)

            active = (~finished).unsqueeze(-1).float()
            x_curr = observations[:, :x_dim]
            x_ref = observations[:, x_dim:2*x_dim]
            error = torch.norm(x_curr - x_ref, dim=-1, keepdim=True)
            auc_sum += error * active
            is_first = (steps_count == 0) & (active > 0)
            e0 = torch.where(is_first, error, e0)
            e_last = torch.where(active > 0, error, e_last)
            e_max = torch.where(active > 0, torch.maximum(e_max, error), e_max)

            observations, rewards, terminated, truncated, _ = self.env.step(actions)
            total_reward += rewards.view(num_envs, 1) * active
            steps_count += active

            finished |= (terminated | truncated).view(num_envs)
            if finished.all():
                break

        agent.enable_training_mode(True)

        dt = float(self._env_scalar_attr("step_dt", "dt"))
        e0c = e0.clamp(min=1e-8)
        eTc = e_last.clamp(min=1e-8)
        T = steps_count.clamp(min=1)
        # Empirical contraction rate: e(T) = e(0) * exp(-lambda * T*dt). Clamped
        # to >= 0 — a negative raw value just means the error grew instead of
        # decaying (no contraction observed), not a valid "rate".
        lambda_emp = (-(torch.log(eTc) - torch.log(e0c)) / (T * dt)).clamp(min=0.0)
        # Overshoot: peak error relative to the initial error.
        overshoot = (e_max.clamp(min=1e-8) / e0c).clamp(min=1e-6)
        # Contraction score: contraction rate per unit of overshoot — higher
        # is better (fast contraction, little to no overshoot).
        contraction_score = lambda_emp / overshoot

        return {
            "reward_mean": total_reward.mean().item(),
            "auc": auc_sum.mean().item(),
            "contraction_rate": lambda_emp.mean().item(),
            "overshoot": overshoot.mean().item(),
            "contraction_score": contraction_score.mean().item(),
        }

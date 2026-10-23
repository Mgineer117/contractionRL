"""C3M jointly learns a physical-state metric and tracking controller.

Independent (x, xref, uref) samples cover the physical state/control boxes.
Analytical tasks supply exact f and B; PhysX tasks fit NeuralDynamics.
Both networks update from the same contraction-loss backward pass.

Cu = M_dot + 2 sym(M(A + BK)) + 2 lambda M < 0
C1 = B_perp.T(-W_dot_f + 2 sym(Df/Dx W) + 2 lambda W)B_perp < 0
C2 = B_perp.T(W_dot_b - 2 sym(DB/Dx W))B_perp = 0

Dynamics, metrics and their Jacobians use physical coordinates. The controller
uses shared TrackingFeatures, as in C2RL. There is no critic or running scaler.
"""

from __future__ import annotations

import os
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import tqdm as _tqdm
from skrl.agents.torch.base import Agent, AgentCfg
from skrl.trainers.torch.base import Trainer, TrainerCfg
from torch import matmul, transpose
from tqdm import tqdm

from contractionRL.agents.utils.contraction.math_utils import (
    b_jacobian,
    bound_W,
    jacobian,
    loss_pos_matrix_random_sampling,
    spd_inverse,
    weighted_gradients,
)
from contractionRL.agents.utils.system_dynamics.reference import RefWindow
from contractionRL.agents.utils.RL.config import filter_cfg_fields

# Configuration

@dataclass
class C3MCfg(AgentCfg):
    memory_size: int = 131072
    batch_size: int = 1024
    W_lr: float = 3e-4
    u_lr: float = 3e-4
    # Alias for u_lr: some configs/sweeps name the controller LR "actor_lr".
    # When set (non-None), __post_init__ copies it into u_lr so both spellings
    # take effect instead of being silently ignored.
    actor_lr: float | None = None
    lbd: float = 1e-2
    eps: float = 1e-2
    w_ub: float = 10.0
    w_lb: float = 0.1
    # Orthonormalize B_null so the epsilon margin and PD loss do not depend on
    # the scale of the supplied annihilator basis.
    orthonormalize_bbot: bool = False
    # Scale C1/C2 feasibility losses independently of the PD and controller terms.
    c1_c2_scale: float = 1.0
    # Detach M in Cu during the initial training fraction to stabilize metric learning.
    detach_warmup_frac: float = 1.0 / 3.0
    # Number of random directions used to estimate PD violations; more samples
    # improve statistical coverage without differentiating eigenvalues.
    pd_loss_num_samples: int = 128
    # Anneal learning rates while repeatedly fitting the offline buffer.
    learning_rate_scheduler: str = "ExponentialLR"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {"gamma": 0.9999})
    dynamics_lr: float = 1e-3
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""
    # Sample a fixed dynamics buffer, then fit it over multiple epochs.
    emp_dynamics_memory_size: int = 8192
    # Classic envs only: how many distinct control vectors get paired with each
    # sampled state in a "dynamics" rollout (env_base.get_rollout); replaces the
    # old hardcoded 3. Used both for pretraining and this agent's per-step
    # online dynamics refinement (see update()).
    num_controls_per_state: int = 3
    # Hold out dynamics validation data and restore the best validation checkpoint.
    dynamics_val_frac: float = 0.1
    dynamics_early_stop_patience: int = 10

    def __post_init__(self):
        # "actor_lr" is an accepted alias for the controller learning rate u_lr.
        if self.actor_lr is not None:
            self.u_lr = self.actor_lr


@dataclass
class C3MTrainerCfg(TrainerCfg):
    timesteps: int = 30000
    eval_interval: int = 1000
    environment_info: str = "default"


# Agent

class C3MAgent(Agent):
    """Control Contraction Metric agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict:
      ``"policy"`` — CLActorModel (contracting C3M_U controller)
      ``"cmg"``    — MetricModel     (contraction-metric generator)

    Extra constructor kwargs:
      ``get_rollout``:    ``(batch_size, mode) -> dict``
        mode="c3m"      → {"x", "xref", "uref"}
        mode="dynamics" → {"x", "u", "x_dot"}
      ``get_f_and_B``:   ``(x) -> (f, B, Bbot)`` — exact callback for analytical tasks;
        PhysX environments must leave this None
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
            cfg = C3MCfg(**filter_cfg_fields(cfg, C3MCfg, context="C3MAgent"))
        super().__init__(
            cfg=cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        # The observation space declares its layout ({x, xrefs, urefs}), so
        # x_dim/u_dim are read, never guessed from obs_dim (the old
        # (obs_dim - u_dim)//2 parity rule silently mis-split any other layout).
        self._window = RefWindow.from_space(observation_space)


        if u_dim is None:
            u_dim = self._window.u_dim
        if x_dim is None:
            x_dim = self._window.x_dim

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._device = device
        self._cfg = cfg


        # Dynamics: analytical or learned online
        if get_f_and_B is not None:
            self._get_f_and_B = get_f_and_B
            self._neural_dynamics = None
            self._dynamics_optimizer = None
        else:
            self._neural_dynamics = models.get("dynamics")
            if self._neural_dynamics is None:
                raise ValueError("C3M requires 'dynamics' model in models dictionary for environments without exact dynamics")

            self._get_f_and_B = self._neural_dynamics.get_f_and_B
            self._dynamics_optimizer = torch.optim.Adam(
                self._neural_dynamics.parameters(), lr=cfg.dynamics_lr
            )

        # Extract underlying nn.Module objects from project model wrappers
        self._ccm_gen = models["cmg"].ccm_gen
        self._cl_actor = models["policy"].cl_actor

        # Keep separate optimizers for CMG/controller gradient filtering,
        # with one configured scheduler type shared by the optimizers.
        self._w_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=cfg.W_lr)
        self._u_optimizer = torch.optim.Adam(self._cl_actor.parameters(), lr=cfg.u_lr)
        self._progress = 0.0

        if getattr(cfg, "learning_rate_scheduler", None):
            scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.learning_rate_scheduler)
            kwargs = getattr(cfg, "learning_rate_scheduler_kwargs", {})
            self._w_lr_scheduler = scheduler_cls(self._w_optimizer, **kwargs)
            self._u_lr_scheduler = scheduler_cls(self._u_optimizer, **kwargs)
            self._dynamics_lr_scheduler = (
                scheduler_cls(self._dynamics_optimizer, **kwargs)
                if self._dynamics_optimizer is not None else None
            )
        else:
            self._w_lr_scheduler = None
            self._u_lr_scheduler = None
            self._dynamics_lr_scheduler = None

        # Data buffer (numpy; static for analytical, or pre-generated)
        self._memory_size = getattr(self.memory, "memory_size", cfg.memory_size) if self.memory is not None else cfg.memory_size
        if self._memory_size < 1:
            raise ValueError("C3M memory.memory_size must be positive")
        self._data = get_rollout(self._memory_size, "c3m")
        self._get_rollout = get_rollout
        self._batch_size = cfg.batch_size
        self._num_updates = 0

        modules = {"policy": models["policy"], "cmg": models["cmg"]}
        if self._neural_dynamics is not None:
            modules["dynamics"] = self._neural_dynamics
        self.checkpoint_modules.update(modules)

    # skrl Agent interface

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
            stored = getattr(self, "_dynamics_rollout_data", None)
            if stored is not None:
                indices = np.random.randint(len(stored["x"]), size=self._cfg.dynamics_batch_size)
                dyn_data = {key: stored[key][indices] for key in ("x", "u", "x_dot")}
            else:
                dyn_data = self._get_rollout(
                    self._cfg.dynamics_batch_size, "dynamics",
                    num_control_per_state=getattr(self._cfg, "num_controls_per_state", None))
            dyn_loss = self._train_dynamics(dyn_data)
            self.track_data("Loss / C3M/dynamics/nll", dyn_loss)

        # 2. Full epoch update (looping entire buffer in batch_size chunks)
        loss_dict = self._learn()
        for k, v in loss_dict.items():
            # LR metrics (k starts with "C3M/lr/") get their own "Learning" tab,
            # matching the project's "Learning / Learning rate" convention —
            # mixing them into "Loss / ..." would group scalar LR values with
            # actual loss curves in wandb/tensorboard.
            tab = "Learning" if k.startswith("C3M/lr/") else "Loss"
            self.track_data(f"{tab} / {k}", v)
        # Keep the latest losses on the agent so the trainer's progress-bar
        # postfix can read them even after post_interaction → write_tracking_data
        # clears tracking_data (otherwise the bar shows a spurious "nan").
        self._last_metrics = dict(loss_dict)
        if dyn_loss is not None:
            self._last_metrics["C3M/dynamics/mse"] = dyn_loss

    # Contraction math (inlined from mjrl C3M)

    def _to_tensor(self, arr):
        dev = self.device
        if isinstance(arr, torch.Tensor):
            return arr.to(torch.float32).to(dev)
        return torch.as_tensor(arr).to(torch.float32).to(dev)

    def _c3m_pools(self):
        """(xref, uref) pools for the future window slots — reuses ``self._data``,
        the same random-triple draw the loss already batches over, so no extra
        rollout is needed. ``None`` when the window is a single point (nothing to
        fill) — ``RefWindow.synth_obs`` then holds the current reference."""
        if self._window.length <= 1:
            return None, None
        return self._to_tensor(self._data["xref"]), self._to_tensor(self._data["uref"])

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
        if getattr(cfg, "orthonormalize_bbot", False):
            # Bbot is already detached, so this is a plain constant re-basis (no
            # second-order-grad concern). Makes the ε margin + PD-loss weighting
            # uniform across states — see C3MCfg.orthonormalize_bbot.
            Bbot = torch.linalg.qr(Bbot).Q

        # Certify the deterministic policy mean, including its own squashing behavior.
        # Hard clipping would erase the controller Jacobian at saturation.
        # Sample reference windows from the same distribution seen by the actor.
        state = self._window.synth_obs(x, xref, uref, *self._c3m_pools())
        # Disable cuDNN here: contraction differentiates through du/dx and requires
        # double backward, which cuDNN GRUs do not support.
        with torch.backends.cudnn.flags(enabled=False):
            u = self.models["policy"].compute({"observations": state}, role="policy")[0]
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
            # Below by construction (bound_W), so a lower-bound loss on it would
            # be a permanent zero-gradient no-op (zᵀ(VᵀV)z = ‖Vz‖² ≥ 0 always).
            overshoot = W - cfg.w_ub * I
            os_loss = loss_pos_matrix_random_sampling(-overshoot, num_samples=n_samp)

        loss = os_loss + pd_loss + cfg.c1_c2_scale * (c1_loss + c2_loss)
        return loss, {"pd_loss": pd_loss.item(), "c1_loss": c1_loss.item(), "c2_loss": c2_loss.item(), "os_loss": os_loss.item() if not bounded else 0.0}

    def _optimize_params(self, loss: torch.Tensor) -> bool:
        """One joint gradient step on CMG + controller together.

        Matches the reference C3M script's single ``optimizer.step()`` over
        both networks' parameters simultaneously — see module docstring for
        why this replaced an earlier alternating (CMG-step, controller-step)
        scheme.
        """
        self._w_optimizer.zero_grad()
        self._u_optimizer.zero_grad()
        loss.backward()

        # CAC-dev protection: filter out NaN/Inf gradients occasionally caused by eigvalsh
        # or numerical edge cases in the contraction metric.
        params = list(self._ccm_gen.parameters()) + list(self._cl_actor.parameters())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            self._w_optimizer.zero_grad()
            self._u_optimizer.zero_grad()
            return False

        torch.nn.utils.clip_grad_norm_(self._ccm_gen.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self._cl_actor.parameters(), 1.0)
        self._w_optimizer.step()
        self._u_optimizer.step()
        return True

    def _learn(self) -> dict:
        cfg = self._cfg
        self._ccm_gen.train(); self._cl_actor.train()

        buf = self._data
        n = buf["x"].shape[0]
        batch_size = cfg.batch_size

        # shuffle indices for the epoch; one full pass in batch_size chunks
        indices = np.random.permutation(n)
        iters = math.ceil(n / batch_size)
        total_pd = total_c1 = total_c2 = total_os = total_loss = 0.0

        pbar = tqdm(range(iters), desc="Epoch C3M Update", leave=False)
        for b in pbar:
            idx = indices[b * batch_size : (b + 1) * batch_size]

            # Single joint gradient step on CMG + controller together.
            loss, infos = self._compute_loss(idx)
            self._optimize_params(loss)

            total_loss += loss.item()
            total_pd += infos["pd_loss"]
            total_c1 += infos["c1_loss"]
            total_c2 += infos["c2_loss"]
            total_os += infos["os_loss"]

        # Step LR schedulers once per epoch (shared type/kwargs across W, u, dynamics)
        if self._w_lr_scheduler is not None:
            self._w_lr_scheduler.step()
        if self._u_lr_scheduler is not None:
            self._u_lr_scheduler.step()
        if self._dynamics_lr_scheduler is not None:
            self._dynamics_lr_scheduler.step()

        self._ccm_gen.eval(); self._cl_actor.eval()

        metrics = {
            "C3M/loss/loss":    total_loss / iters,
            "C3M/loss/pd_loss": total_pd / iters,
            "C3M/loss/c1_loss": total_c1 / iters,
            "C3M/loss/c2_loss": total_c2 / iters,
            "C3M/loss/os_loss": total_os / iters,
            "C3M/lr/u_lr": self._u_lr_scheduler.get_last_lr()[0] if self._u_lr_scheduler else cfg.u_lr,
            "C3M/lr/w_lr": self._w_lr_scheduler.get_last_lr()[0] if self._w_lr_scheduler else cfg.W_lr,
        }
        if self._dynamics_lr_scheduler is not None:
            metrics["C3M/lr/dynamics_lr"] = self._dynamics_lr_scheduler.get_last_lr()[0]
        elif self._dynamics_optimizer is not None:
            metrics["C3M/lr/dynamics_lr"] = cfg.dynamics_lr
        return metrics

    def _train_dynamics(self, data: dict) -> float:
        """Conditional Gaussian likelihood training on (x, u, x_dot) data."""
        dev = self._neural_dynamics.device
        x     = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        u     = torch.as_tensor(data["u"], dtype=torch.float32, device=dev)
        x_dot = torch.as_tensor(data["x_dot"], dtype=torch.float32, device=dev)

        loss = self._neural_dynamics.negative_log_likelihood(x, u, x_dot)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite stochastic dynamics likelihood")

        self._dynamics_optimizer.zero_grad()
        loss.backward()

        # Prevent NaNs from massive MSE losses by checking gradients and clipping
        if all(torch.isfinite(p.grad).all() for p in self._neural_dynamics.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._neural_dynamics.parameters(), 1.0)
            self._dynamics_optimizer.step()
            self._neural_dynamics.disturbance_fitted.fill_(True)

        return loss.item()
    def save_dynamics(self, path: str) -> None:
        """Save NeuralDynamics checkpoint for SDLQR/LQR/C2RL to load."""
        if self._neural_dynamics is not None:
            self._neural_dynamics.save(path)
            print(f"[C3M] Saved NeuralDynamics → {path}")


# Trainer

class C3MSkrlTrainer(Trainer):
    """skrl Trainer for C3M with optional dynamics-data collection before fitting.

    Calls ``agent.update()`` in a tight loop.  Evaluation runs a few env episodes.
    """

    def train(self) -> None:
        agent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        timesteps = self.cfg.timesteps
        eval_interval = getattr(self.cfg, "eval_interval", 1)

        # Set write_interval before init(), which conditionally creates the writer.
        agent.write_interval = eval_interval
        agent.init(trainer_cfg=self.cfg)

        from contractionRL.agents.utils.contraction.metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))
        agent.enable_training_mode(True)

        # Pretrain the learned NeuralDynamics before RL (same mechanism as
        # C2RL — see dynamics_pretrain.py; kept in one place so the two agents'
        # pretraining behavior can't drift apart). No-op for analytical
        # dynamics or dynamics_pretrain_epochs<=0.
        from contractionRL.agents.utils.system_dynamics.dynamics import (
            collect_stochastic_rollouts, load_offline_dynamics_data, pretrain_dynamics,
        )
        data_path = getattr(agent.cfg, "dynamics_pretrain_data_path", "") or None
        dynamics_epochs = getattr(agent.cfg, "dynamics_pretrain_epochs", 5)
        disturbance_enabled = False
        if agent._neural_dynamics is not None:
            from contractionRL.runners.contraction_runner import _unwrap_env
            raw = _unwrap_env(self.env)
            disturbance_enabled = bool(getattr(getattr(raw, "disturbance", None), "enabled", False))
        if agent._neural_dynamics is not None and data_path and dynamics_epochs <= 0:
            agent._dynamics_rollout_data = load_offline_dynamics_data(
                data_path, tag="[C3M]", require_disturbed=disturbance_enabled)
        if agent._neural_dynamics is not None and not data_path:
            from contractionRL.agents.utils.RL.logging import metric_batch
            dt = getattr(raw, "dt", None) or raw.step_dt
            action_space = getattr(raw, "single_action_space", raw.action_space)
            with metric_batch(step=-getattr(agent.cfg, "dynamics_pretrain_epochs", 5) - 1):
                agent._dynamics_rollout_data = collect_stochastic_rollouts(
                    raw, agent.cfg.emp_dynamics_memory_size, low=action_space.low,
                    high=action_space.high, device=agent.device, dt=dt,
                    angle_idx=getattr(raw, "angle_idx", ()),
                    path=os.path.join(agent.experiment_dir, "stochastic_dynamics_data.npz"))
        pretrain_dynamics(
            agent,
            epochs=dynamics_epochs,
            data_path=data_path,
            timesteps=timesteps,
            memory_size=getattr(agent.cfg, "emp_dynamics_memory_size", None),
            num_controls_per_state=getattr(agent.cfg, "num_controls_per_state", None),
            log_interval=eval_interval,
            tag="[C3M]",
            val_frac=getattr(agent.cfg, "dynamics_val_frac", 0.1),
            early_stop_patience=getattr(agent.cfg, "dynamics_early_stop_patience", 10),
            require_disturbed=disturbance_enabled,
        )

        pbar = _tqdm.tqdm(range(timesteps), desc="C3M training", file=sys.stdout)
        for t in pbar:
            agent.pre_interaction(timestep=t, timesteps=timesteps)
            agent.update(timestep=t, timesteps=timesteps)

            # Evaluate metrics occasionally. track_*_summary writes the same
            # "Stability/..."/"Reward/..." keys (no space around "/") that
            # path_tracking_base.py emits for PPO/SAC, so every algorithm lands
            # on the same wandb tabs for the same metric.
            if eval_interval > 0 and (t + 1) % eval_interval == 0:
                from contractionRL.agents.utils.contraction.metrics import track_reward_summary, track_stability_summary
                ev = self.eval(timestep=t)
                for key in tuple(agent.tracking_data):
                    if key.startswith("Stability/"):
                        agent.tracking_data.pop(key)
                if 'sim_to_real' in ev:
                    for section, values in (('Simulation Evaluation', ev), ('Sim-to-Real Evaluation', ev['sim_to_real'])):
                        track_stability_summary(agent, values['stability'], tab=f'{section}/Stability')
                        track_reward_summary(agent, values['reward'], tab=f'{section}/Reward')
                        for key, value in values.get('control', {}).items():
                            agent.track_data(f'{section}/Control/{key}', value)
                else:
                    track_stability_summary(agent, ev['stability'])
                    track_reward_summary(agent, ev['reward'])
                    for key, value in ev.get('control', {}).items():
                        agent.track_data(f'Evaluation/Control/{key}', value)
                # C3M's trainer discards info["log"]; forward clamp metrics explicitly.
                _clamp = getattr(self.env, "clamp_summary", None)
                if callable(_clamp):
                    _cs = _clamp()
                    if _cs:
                        track_stability_summary(agent, _cs, tab="Clamp")

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

            agent.post_interaction(timestep=t, timesteps=timesteps)

        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            agent.save_dynamics(dyn_path)

    def _env_scalar_attr(self, *names):
        """Fetch a scalar env attribute across both env backends.

        Isaac envs expose it directly on the skrl-wrapped env (shared across
        the batched sim, e.g. ``max_episode_length``/``step_dt``). Classic
        envs are N separate Python instances behind a gymnasium
        ``SyncVectorEnv`` (attr named e.g. ``max_episode_len``/``dt``), which
        does not forward arbitrary attribute access — only ``get_attr(name)``
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

    def _force_env_reset(self) -> None:
        """Force skrl to genuinely re-reset the underlying vector env.

        skrl's ``GymnasiumWrapper.reset()`` resets the vector env only once and
        thereafter returns a cached ``self._observation`` (it assumes vector envs
        autoreset inside ``step()``). Under **gymnasium 1.x next-step autoreset**
        (Isaac Sim 5.1 pins gymnasium 1.2.1), that cache holds the previous
        episode's parked terminal observation — where the controller has already
        converged, so ``x ≈ xref`` and the initial tracking error ``e0 ≈ 0``.
        The eval AUC is ``(dt / e0)·Σe``, so a near-zero e0 makes AUC explode to
        ~1e7 even though the controller is perfectly stable (a metric artifact,
        not divergence). gymnasium 0.29 (the ``base`` env) autoresets in the same
        step, so its cache is a fresh reset and e0 is correct — which is why this
        only bites under env_isaaclab. Flipping ``_reset_once`` back on makes the
        next ``reset()`` actually re-randomize the sub-envs (``time_steps→0``).
        Project-side only — does not modify skrl.
        """
        w = self.env
        for _ in range(8):
            if isinstance(w, object) and "_reset_once" in vars(w):
                vars(w)["_reset_once"] = True
                return
            w = getattr(w, "env", None) or getattr(w, "_env", None)
            if w is None:
                return

    def eval(self, timestep: int = 0) -> dict:
        from contractionRL.tasks.direct.common.transfer_dynamics import is_transfer_path, evaluate_backend_pair
        raw = self.env.unwrapped
        if is_transfer_path(raw) and hasattr(raw, 'set_transition_backend'):
            pair = evaluate_backend_pair(self.env, lambda: self._eval_single(timestep))
            return {**pair['simulation'], 'sim_to_real': pair['sim_to_real']}
        return self._eval_single(timestep)

    def _eval_single(self, timestep: int = 0) -> dict:
        from contractionRL.agents.utils.RL.evaluation import EvaluationControl

        agent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        agent.enable_training_mode(False)
        self._force_env_reset()
        observations, infos = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None

        from contractionRL.agents.utils.contraction.metrics import reward_summary

        num_envs = self.env.num_envs
        device = self.env.device

        # Metric collection is now handled by StatManagerEnvWrapper on every
        # reset()/step() (auto e0-anchoring off the true episode counter, one clean
        # episode per env, memory-bounded to num_envs_for_eval). The eval loop just
        # drives the rollout and reads self.env.stability_summary() / trajectories()
        # at the end. reward masking stays here (agent-side).
        max_steps = int(self._env_scalar_attr("max_episode_length", "max_episode_len")) + 1
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        total_reward = torch.zeros((num_envs, 1), device=device)
        reward_steps = torch.zeros((num_envs, 1), device=device)
        # Snapshot the wrapper's compute counter so we can tell whether this
        # rollout actually produced fresh metrics (see check after the loop).
        computes_before = getattr(self.env, "_compute_count", None)

        eval_pbar = tqdm(range(max_steps), desc="C3M eval", leave=False, file=sys.stdout)
        with EvaluationControl(self.env, agent) as control:
            for step in eval_pbar:
                eval_pbar.set_postfix(timestep=f"{step + 1}/{max_steps}")
                active = ~finished
                actions = control.act(observations, states, active=active)
                observations, rewards, terminated, truncated, _ = self.env.step(actions)
                states = self.env.state() if hasattr(self.env, "state") else None
                total_reward += rewards.view(num_envs, 1) * active.unsqueeze(-1).float()
                reward_steps += active.unsqueeze(-1)
                done = (terminated | truncated).view(num_envs)
                control.after_step(done)
                finished |= done
                if finished.all():
                    break
            control_metrics = control.summary()
        eval_pbar.close()

        agent.enable_training_mode(True)

        fresh = computes_before is None or getattr(self.env, "_compute_count", None) != computes_before
        if not fresh:
            import warnings
            warnings.warn(
                "C3M eval: the StatManagerEnvWrapper buffer did not complete during "
                "this rollout (an env may have terminated early before the eval "
                "horizon elapsed) — withholding the previous evaluation's stability metrics.",
                stacklevel=2,
            )

        # Only envs that finished their (first) episode carry valid reward info.
        f_mask = finished if finished.any() else torch.ones_like(finished)

        # WandbPlotWrapper already captures these steps; avoid duplicate plots.
        return {
            "stability": self.env.stability_summary() if fresh else {},
            "reward": reward_summary(total_reward, f_mask, steps=reward_steps),
            "control": control_metrics,
        }

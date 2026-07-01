"""TEMP — Two-policy contraction-metric synthesis, native skrl Agent.

TEMPAgent alternates between a contracting rollout phase (γ→0) and an optimal
rollout phase (high γ).  Both PPO-backed policies share a CMG trained with the
contraction pd-loss.  The Mahalanobis reward ``-||e||²_M / std`` replaces the
environment reward for both policies.

Training loop (handled by TEMPSkrlTrainer):
  1. Collect ``rollout_steps`` transitions with *con_policy* (γ→0).
  2. Inject Mahalanobis rewards → PPO update for con_policy.
  3. Collect ``rollout_steps`` transitions with *opt_policy* (high γ).
  4. Inject Mahalanobis rewards → PPO update for opt_policy.
  5. CMG update with contraction pd-loss (random data from get_rollout).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from skrl.agents.torch.base import Agent, AgentCfg
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .math_utils import (
    b_jacobian,
    bound_W,
    jacobian,
    loss_pos_matrix_eigen,
    weighted_gradients,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class TEMPCfg(AgentCfg):
    # PPO shared config
    rollouts: int = 300
    learning_epochs: int = 8
    mini_batches: int = 4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 1.0
    kl_threshold: float = 0.0
    grad_norm_clip: float = 1.0
    # TEMP-specific
    gamma_contracting: float = 0.0
    gamma_optimal: float = 0.99
    con_only: bool = False
    # CMG
    W_lr: float = 3e-4
    w_ub: float = 10.0
    w_lb: float = 0.1
    lbd: float = 1e-2
    eps: float = 1e-2
    W_entropy_scaler: float = 1e-3
    cmg_updates_per_iter: int = 1
    cmg_minibatch_size: int = 1024
    buffer_size: int = 2048
    # reward normalisation
    reward_norm_beta: float = 0.99
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0


@dataclass
class TEMPTrainerCfg(TrainerCfg):
    timesteps: int = 300000
    rollouts: int = 300
    con_only: bool = False


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class TEMPAgent(Agent):
    """TEMP agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict:
      ``"con_policy"`` — CLActorModel (contracting controller, γ→0)
      ``"con_value"``  — DeterministicMixin Model (value fn for con PPO)
      ``"opt_policy"`` — CLActorModel (optimal controller, high γ)
      ``"opt_value"``  — DeterministicMixin Model (value fn for opt PPO)
      ``"cmg"``        — CMGModel wrapping CCM_Generator

    Extra constructor kwargs:
      ``get_rollout``:  ``(buffer_size, mode) -> dict(x, xref, uref)``
      ``get_f_and_B``:  ``(x) -> (f, B, Bbot)``
    """

    def __init__(
        self,
        *,
        cfg: TEMPCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_rollout: Callable,
        get_f_and_B: Callable,
        x_dim: int | None = None,
        u_dim: int | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = TEMPCfg(**{k: v for k, v in cfg.items() if k in TEMPCfg.__dataclass_fields__})
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
        self._con_only = cfg.con_only

        # ── Build two PPO agents (con + opt) ─────────────────────────────── #
        num_envs = 1
        con_memory = RandomMemory(memory_size=cfg.rollouts, num_envs=num_envs, device=device)
        opt_memory = RandomMemory(memory_size=cfg.rollouts, num_envs=num_envs, device=device)

        import dataclasses as _dc
        ppo_common = dict(
            learning_epochs=cfg.learning_epochs,
            mini_batches=cfg.mini_batches,
            gae_lambda=cfg.gae_lambda,
            learning_rate=cfg.learning_rate,
            ratio_clip=cfg.ratio_clip,
            value_clip=cfg.value_clip,
            entropy_loss_scale=cfg.entropy_loss_scale,
            value_loss_scale=cfg.value_loss_scale,
            kl_threshold=cfg.kl_threshold,
            grad_norm_clip=cfg.grad_norm_clip,
            rollouts=cfg.rollouts,
        )

        def _make_ppo_cfg(gamma: float) -> dict:
            d = _dc.asdict(PPO_CFG(**{k: v for k, v in ppo_common.items() if k in PPO_CFG.__dataclass_fields__}))
            d["discount_factor"] = gamma
            d["experiment"]["write_interval"] = 0
            d["experiment"]["checkpoint_interval"] = 0
            return d

        ppo_kwargs = dict(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self._con_ppo = PPO(
            cfg=_make_ppo_cfg(cfg.gamma_contracting),
            models={"policy": models["con_policy"], "value": models["con_value"]},
            memory=con_memory,
            **ppo_kwargs,
        )
        self._opt_ppo = PPO(
            cfg=_make_ppo_cfg(cfg.gamma_optimal),
            models={"policy": models["opt_policy"], "value": models["opt_value"]},
            memory=opt_memory,
            **ppo_kwargs,
        ) if not cfg.con_only else None

        self._con_memory = con_memory
        self._opt_memory = opt_memory

        # PPO.init() lazily creates memory tensors — call it here since sub-PPOs
        # are driven manually (not by a standard Trainer).
        self._con_ppo.init()
        if self._opt_ppo is not None:
            self._opt_ppo.init()

        # ── CMG + TEMP math ───────────────────────────────────────────────── #
        self._ccm_gen = models["cmg"].ccm_gen
        self._con_cl_actor = models["con_policy"].cl_actor
        self._get_f_and_B = get_f_and_B
        self._data = get_rollout(cfg.buffer_size, "c3m")
        self._get_rollout = get_rollout
        self._buffer_size = cfg.buffer_size

        self._W_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=cfg.W_lr)
        self._temp_progress = 0.0
        self._W_lr_scheduler = LambdaLR(
            self._W_optimizer, lr_lambda=lambda _: max(0.0, 1.0 - self._temp_progress)
        )

        self._running_reward_var: float | None = None

        self.checkpoint_modules.update({
            "con_policy": models["con_policy"],
            "con_value":  models["con_value"],
            "opt_policy": models.get("opt_policy"),
            "opt_value":  models.get("opt_value"),
            "cmg":        models["cmg"],
        })

    # ── skrl Agent interface ────────────────────────────────────────────── #

    def act(self, observations, states, *, timestep: int, timesteps: int):
        """During inference uses the optimal policy (or con_policy if con_only)."""
        ppo = self._opt_ppo if (self._opt_ppo and not self._con_only) else self._con_ppo
        with torch.no_grad():
            result = ppo.models["policy"].act({"observations": observations}, role="policy")
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
        # TEMPSkrlTrainer calls update_con / update_opt / update_cmg directly.
        pass

    # ── Mahalanobis reward ─────────────────────────────────────────────── #

    def _compute_mahalanobis_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute -||e||²_M / std  (normalised Riemannian tracking energy)."""
        cfg = self._cfg
        x_dim = self._x_dim
        device = self._device
        dtype = torch.float32
        I = torch.eye(x_dim, device=device, dtype=dtype)

        x    = observations[:, :x_dim].to(dtype)
        xref = observations[:, x_dim : 2 * x_dim].to(dtype)
        e = (x - xref).unsqueeze(-1)

        with torch.no_grad():
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, cfg.w_lb, x_dim)
            M = torch.linalg.solve(W, I.unsqueeze(0).expand_as(W))
            quad = (e.transpose(1, 2) @ M @ e).squeeze(-1)

            batch_var = quad.var(unbiased=False).item()
            beta = cfg.reward_norm_beta
            if self._running_reward_var is None:
                self._running_reward_var = batch_var + 1e-8
            else:
                self._running_reward_var = beta * self._running_reward_var + (1 - beta) * batch_var
            std = self._running_reward_var ** 0.5 + 1e-8
            reward = -cfg.tracking_scaler * quad / std
        return reward

    # ── CMG update (inlined from mjrl TEMP.compute_cmg_loss / update_cmg) ─ #

    def _compute_cmg_loss(self):
        cfg = self._cfg
        device = self._device
        x_dim, u_dim = self._x_dim, self._u_dim
        I = torch.eye(x_dim, device=device)

        buf = self._data
        n = buf["x"].shape[0]
        batch_size = min(cfg.cmg_minibatch_size, n)
        idx = np.random.choice(n, size=batch_size, replace=False)

        x    = torch.from_numpy(buf["x"][idx]).float().to(device).requires_grad_()
        xref = torch.from_numpy(buf["xref"][idx]).float().to(device)
        uref = torch.from_numpy(buf["uref"][idx]).float().to(device)

        state = torch.cat([x, xref, uref], dim=1)
        u = self._con_cl_actor.mean_control(state)
        K = jacobian(u, x, create_graph=False)
        u = u.detach()
        K = K.detach()

        raw_W, info_W = self._ccm_gen(x)
        W = bound_W(raw_W, cfg.w_lb, x_dim)
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(batch_size, -1, -1))

        with torch.enable_grad():
            f, B, _ = self._get_f_and_B(x)
        f = f.float().to(device)
        B = B.float().to(device)

        DfDx = jacobian(f, x, create_graph=False)
        DBDx = b_jacobian(B, x, u_dim, create_graph=False)
        f = f.detach(); B = B.detach()

        A = DfDx + sum(
            u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(u_dim)
        )
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = weighted_gradients(M, dot_x, x)

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * cfg.lbd * M
        Cu = Cu + cfg.eps * I

        pd_loss, pd_reg = loss_pos_matrix_eigen(-Cu)
        entropy_loss = info_W.get("entropy", torch.tensor(0.0, device=device))
        if isinstance(entropy_loss, torch.Tensor):
            entropy_loss = entropy_loss.mean() * cfg.W_entropy_scaler
        else:
            entropy_loss = torch.zeros(1, device=device)

        loss = pd_loss + pd_reg - entropy_loss
        return loss, {"pd_loss": pd_loss.item(), "pd_reg": pd_reg.item()}

    # ── TEMP update methods (called by TEMPSkrlTrainer) ──────────────────── #

    def update_con(self, observations: torch.Tensor, *, timestep: int, timesteps: int) -> dict:
        """Inject Mahalanobis rewards into con_memory and run PPO update."""
        maha_r = self._compute_mahalanobis_reward(observations)
        try:
            r_tensor = self._con_ppo.memory.get_tensor_by_name("rewards")
            if r_tensor.shape != maha_r.shape:
                import skrl
                skrl.logger.warning(f"[TEMP] Reward shape mismatch in update_con: {r_tensor.shape} != {maha_r.shape}")
            r_tensor.copy_(maha_r.view_as(r_tensor))
        except RuntimeError as e:
            import skrl
            skrl.logger.warning(f"[TEMP] Failed to inject Mahalanobis reward in update_con: {e}")
        self._con_ppo.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_opt(self, observations: torch.Tensor, *, timestep: int, timesteps: int) -> dict:
        """Inject Mahalanobis rewards into opt_memory and run PPO update."""
        if self._con_only or self._opt_ppo is None:
            return {}
        maha_r = self._compute_mahalanobis_reward(observations)
        try:
            r_tensor = self._opt_ppo.memory.get_tensor_by_name("rewards")
            if r_tensor.shape != maha_r.shape:
                import skrl
                skrl.logger.warning(f"[TEMP] Reward shape mismatch in update_opt: {r_tensor.shape} != {maha_r.shape}")
            r_tensor.copy_(maha_r.view_as(r_tensor))
        except RuntimeError as e:
            import skrl
            skrl.logger.warning(f"[TEMP] Failed to inject Mahalanobis reward in update_opt: {e}")
        self._opt_ppo.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_cmg(self, *, timestep: int, timesteps: int) -> dict:
        """Refresh data buffer and run CMG contraction pd-loss update."""
        self._data = self._get_rollout(self._buffer_size, "c3m")
        self._temp_progress = float(timestep) / max(1, timesteps)
        self._ccm_gen.train()
        for _ in range(self._cfg.cmg_updates_per_iter):
            loss, info = self._compute_cmg_loss()
            self._W_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._ccm_gen.parameters(), 10.0)
            self._W_optimizer.step()
        self._W_lr_scheduler.step()
        self._ccm_gen.eval()

        loss_dict = {
            "TEMP/CMG/pd_loss": info["pd_loss"],
            "TEMP/CMG/pd_reg":  info["pd_reg"],
        }
        for k, v in loss_dict.items():
            self.track_data(f"Loss / {k}", v)
        return loss_dict


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class TEMPSkrlTrainer(Trainer):
    """skrl Trainer for TEMP — alternates con/opt rollouts and CMG updates."""

    def train(self) -> None:
        agent: TEMPAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        num_envs = env.num_envs
        rollout_steps = self.cfg.rollouts if hasattr(self.cfg, "rollouts") else agent._cfg.rollouts
        timesteps = self.cfg.timesteps
        con_only = agent._con_only

        agent.init(trainer_cfg=self.cfg)

        agent._con_memory._memory_size = rollout_steps
        agent._con_memory._num_envs = num_envs
        if not con_only and agent._opt_memory is not None:
            agent._opt_memory._memory_size = rollout_steps
            agent._opt_memory._num_envs = num_envs

        agent._con_ppo.memory = agent._con_memory
        if not con_only and agent._opt_ppo is not None:
            agent._opt_ppo.memory = agent._opt_memory

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0

        import tqdm as _tqdm
        total_iters = timesteps // (rollout_steps * (1 if con_only else 2))
        pbar = _tqdm.tqdm(range(max(1, total_iters)), desc="TEMP training", file=sys.stdout)

        for epoch in pbar:

            # ── Phase 1: contracting rollout ──────────────────────────── #
            agent._con_ppo.enable_training_mode(True)
            agent._con_ppo.memory.reset()
            con_obs_list = []

            for step in range(rollout_steps):
                agent._con_ppo.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = agent._con_ppo.act(
                        observations, states, timestep=global_step, timesteps=timesteps
                    )
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None
                con_obs_list.append(observations.clone())
                agent._con_ppo.record_transition(
                    observations=observations, states=states, actions=actions,
                    rewards=rewards, next_observations=next_obs, next_states=next_states,
                    terminated=terminated, truncated=truncated, infos=infos,
                    timestep=global_step, timesteps=timesteps,
                )
                observations = next_obs
                states = next_states
                global_step += 1

            con_obs_tensor = torch.cat(con_obs_list, dim=0)
            agent.update_con(con_obs_tensor, timestep=global_step, timesteps=timesteps)
            agent._con_ppo.post_interaction(timestep=global_step, timesteps=timesteps)

            # ── Phase 2: optimal rollout ──────────────────────────────── #
            if not con_only and agent._opt_ppo is not None:
                agent._opt_ppo.enable_training_mode(True)
                agent._opt_ppo.memory.reset()
                opt_obs_list = []

                for step in range(rollout_steps):
                    agent._opt_ppo.pre_interaction(timestep=global_step, timesteps=timesteps)
                    with torch.no_grad():
                        actions, _ = agent._opt_ppo.act(
                            observations, states, timestep=global_step, timesteps=timesteps
                        )
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
                    next_states = env.state() if hasattr(env, "state") else None
                    opt_obs_list.append(observations.clone())
                    agent._opt_ppo.record_transition(
                        observations=observations, states=states, actions=actions,
                        rewards=rewards, next_observations=next_obs, next_states=next_states,
                        terminated=terminated, truncated=truncated, infos=infos,
                        timestep=global_step, timesteps=timesteps,
                    )
                    observations = next_obs
                    states = next_states
                    global_step += 1

                opt_obs_tensor = torch.cat(opt_obs_list, dim=0)
                agent.update_opt(opt_obs_tensor, timestep=global_step, timesteps=timesteps)
                agent._opt_ppo.post_interaction(timestep=global_step, timesteps=timesteps)

            # ── CMG update ────────────────────────────────────────────── #
            cmg_dict = agent.update_cmg(timestep=global_step, timesteps=timesteps)
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            pd_loss = cmg_dict.get("TEMP/CMG/pd_loss", float("nan"))
            pbar.set_postfix(epoch=epoch, pd=f"{pd_loss:.3g}")

    def eval(self) -> None:
        agent: TEMPAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        ppo = agent._opt_ppo if (agent._opt_ppo and not agent._con_only) else agent._con_ppo
        ppo.enable_training_mode(False)
        observations, _ = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None
        done = False
        while not done:
            with torch.no_grad():
                actions, _ = ppo.models["policy"].act({"observations": observations}, role="policy")
            observations, rewards, terminated, truncated, _ = self.env.step(actions)
            states = self.env.state() if hasattr(self.env, "state") else None
            done = bool((terminated | truncated).any())
        ppo.enable_training_mode(True)

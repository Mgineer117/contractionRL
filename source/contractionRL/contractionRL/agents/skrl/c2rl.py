"""C2RL — Two-policy contraction-metric synthesis, native skrl Agent.

C2RLAgent alternates between a contracting rollout phase (γ→0) and an optimal
rollout phase (high γ).  Both PPO-backed policies share a CMG trained with the
contraction pd-loss.  The Mahalanobis reward ``-||e||²_M / std`` replaces the
environment reward for both policies.

Training loop (handled by C2RLSkrlTrainer):
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
import tqdm as _tqdm
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from skrl.agents.torch.base import Agent, AgentCfg
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .math_utils import (
    b_jacobian,
    bound_W,
    jacobian,
    loss_pos_matrix_eigen,
    spd_inverse,
    weighted_gradients,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class C2RLCfg(AgentCfg):
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
    # C2RL-specific
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
class C2RLTrainerCfg(TrainerCfg):
    timesteps: int = 300000
    rollouts: int = 300
    con_only: bool = False


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLAgent(Agent):
    """C2RL agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict:
      ``"con_policy"`` — CLActorModel (contracting controller, γ→0)
      ``"con_value"``  — DeterministicMixin Model (value fn for con PPO)
      ``"opt_policy"`` — CLActorModel (optimal controller, high γ)
      ``"opt_value"``  — DeterministicMixin Model (value fn for opt PPO)
      ``"cmg"``        — CMGModel wrapping CCM_Generator

    Extra constructor kwargs:
      ``get_rollout``:  ``(buffer_size, mode) -> dict(x, xref, uref)``
      ``get_f_and_B``:  ``(x) -> (f, B, Bbot)``
      ``num_envs``:     number of parallel environments — con/opt ``RandomMemory``
        buffers are allocated with this shape up front (skrl memory tensors are
        sized at ``init()`` time and cannot be resized afterwards).
    """

    def __init__(
        self,
        *,
        cfg: C2RLCfg | dict,
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
        num_envs: int = 1,
    ) -> None:
        if isinstance(cfg, dict):
            self._raw_cfg = cfg.copy()
            parsed_cfg = C2RLCfg(**{k: v for k, v in cfg.items() if k in C2RLCfg.__dataclass_fields__})
        else:
            self._raw_cfg = cfg.__dict__.copy()
            parsed_cfg = cfg

        super().__init__(
            cfg=parsed_cfg,
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
        self._cfg = parsed_cfg
        self._con_only = parsed_cfg.con_only
        self._base_algorithm = self._raw_cfg.get("base_algorithm", "PPO").upper()

        # ── Build two RL agents (con + opt) ─────────────────────────────── #
        # Memory tensors are physically allocated at agent.init() below with
        # this num_envs — they cannot be resized later, so it must be correct
        # up front (see C2RLSkrlTrainer.train, which used to mutate _num_envs
        # post-hoc; that silently corrupted the con/opt buffers on multi-env
        # Isaac Sim runs since the underlying tensors stayed shape (rollouts, 1, ...)).
        self._num_envs = num_envs
        rollouts = self._raw_cfg.get("rollouts", 300)
        con_memory = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)
        opt_memory = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)

        rl_kwargs = dict(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        def _make_base_cfg(gamma: float) -> dict:
            # Project the raw C2RL config down to *only* the base agent's own
            # fields. Passing C2RL/CMG-specific keys (W_lr, lbd, gamma_*, ...) to
            # PPO_CFG(**cfg) / SAC_CFG(**cfg) would raise TypeError, since those
            # are kw_only dataclasses that reject unknown kwargs. Also rebuild
            # `experiment` as a plain dict (the raw value may be an ExperimentCfg
            # object, which is not subscriptable).
            if self._base_algorithm == "SAC":
                from skrl.agents.torch.sac import SAC_CFG as _BaseCfg
            else:
                from skrl.agents.torch.ppo import PPO_CFG as _BaseCfg
            valid = _BaseCfg.__dataclass_fields__
            d = {k: v for k, v in self._raw_cfg.items() if k in valid and k != "experiment"}
            d["discount_factor"] = gamma
            d["experiment"] = {"write_interval": 0, "checkpoint_interval": 0}
            return d

        if self._base_algorithm == "PPO":
            from skrl.agents.torch.ppo import PPO as BaseRLAgent
            con_models = {"policy": models["con_policy"], "value": models["con_value"]}
            opt_models = {"policy": models["opt_policy"], "value": models["opt_value"]} if "opt_policy" in models else {}
        elif self._base_algorithm == "SAC":
            from skrl.agents.torch.sac import SAC as BaseRLAgent
            con_models = {
                "policy": models["con_policy"], "q1": models["con_q1"], "q2": models["con_q2"],
                "target_q1": models["con_target_q1"], "target_q2": models["con_target_q2"]
            }
            opt_models = {
                "policy": models["opt_policy"], "q1": models["opt_q1"], "q2": models["opt_q2"],
                "target_q1": models["opt_target_q1"], "target_q2": models["opt_target_q2"]
            } if "opt_policy" in models else {}
        else:
            raise ValueError(f"[C2RL] Unsupported base_algorithm: {self._base_algorithm}")

        self._con_agent = BaseRLAgent(
            cfg=_make_base_cfg(parsed_cfg.gamma_contracting),
            models=con_models,
            memory=con_memory,
            **rl_kwargs,
        )
        self._opt_agent = BaseRLAgent(
            cfg=_make_base_cfg(parsed_cfg.gamma_optimal),
            models=opt_models,
            memory=opt_memory,
            **rl_kwargs,
        ) if not parsed_cfg.con_only else None

        self._con_memory = con_memory
        self._opt_memory = opt_memory

        self._con_agent.init()
        if self._opt_agent is not None:
            self._opt_agent.init()

        # ── CMG + C2RL math ───────────────────────────────────────────────── #
        self._ccm_gen = models["cmg"].ccm_gen
        self._con_cl_actor = models["con_policy"].cl_actor
        self._get_f_and_B = get_f_and_B
        self._data = get_rollout(cfg.buffer_size, "c3m")
        self._get_rollout = get_rollout
        self._buffer_size = cfg.buffer_size

        self._W_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=cfg.W_lr)
        self._progress = 0.0
        self._W_lr_scheduler = LambdaLR(
            self._W_optimizer, lr_lambda=lambda _: max(0.0, 1.0 - self._progress)
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
        ppo = self._opt_agent if (self._opt_agent and not self._con_only) else self._con_agent
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
        # C2RLSkrlTrainer calls update_con / update_opt / update_cmg directly.
        pass

    # ── Mahalanobis reward ─────────────────────────────────────────────── #

    def _compute_mahalanobis_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute -||e||²_M / std  (normalised Riemannian tracking energy)."""
        cfg = self._cfg
        x_dim = self._x_dim
        dtype = torch.float32

        x    = observations[:, :x_dim].to(dtype)
        xref = observations[:, x_dim : 2 * x_dim].to(dtype)
        e = (x - xref).unsqueeze(-1)

        with torch.no_grad():
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, cfg.w_lb, x_dim)
            M = spd_inverse(W)
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

    # ── CMG update (inlined from mjrl C2RL.compute_cmg_loss / update_cmg) ─ #

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
        M = spd_inverse(W)

        with torch.enable_grad():
            f, B, _ = self._get_f_and_B(x)
        f = f.float().to(device)
        B = B.float().to(device)

        DfDx = jacobian(f, x, create_graph=False)
        DBDx = b_jacobian(B, x, u_dim, create_graph=False)
        f = f.detach(); B = B.detach()

        A = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, u)
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

    # ── C2RL update methods (called by C2RLSkrlTrainer) ──────────────────── #

    def update_con(self, observations: torch.Tensor, *, timestep: int, timesteps: int) -> dict:
        """Inject Mahalanobis rewards into con_memory and run PPO update."""
        maha_r = self._compute_mahalanobis_reward(observations)
        try:
            r_tensor = self._con_agent.memory.get_tensor_by_name("rewards")
            if r_tensor.shape != maha_r.shape:
                import skrl
                skrl.logger.warning(f"[C2RL] Reward shape mismatch in update_con: {r_tensor.shape} != {maha_r.shape}")
            r_tensor.copy_(maha_r.view_as(r_tensor))
        except RuntimeError as e:
            import skrl
            skrl.logger.warning(f"[C2RL] Failed to inject Mahalanobis reward in update_con: {e}")
        self._con_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_opt(self, observations: torch.Tensor, *, timestep: int, timesteps: int) -> dict:
        """Inject Mahalanobis rewards into opt_memory and run PPO update."""
        if self._con_only or self._opt_agent is None:
            return {}
        maha_r = self._compute_mahalanobis_reward(observations)
        try:
            r_tensor = self._opt_agent.memory.get_tensor_by_name("rewards")
            if r_tensor.shape != maha_r.shape:
                import skrl
                skrl.logger.warning(f"[C2RL] Reward shape mismatch in update_opt: {r_tensor.shape} != {maha_r.shape}")
            r_tensor.copy_(maha_r.view_as(r_tensor))
        except RuntimeError as e:
            import skrl
            skrl.logger.warning(f"[C2RL] Failed to inject Mahalanobis reward in update_opt: {e}")
        self._opt_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_cmg(self, *, timestep: int, timesteps: int) -> dict:
        """Refresh data buffer and run CMG contraction pd-loss update."""
        self._data = self._get_rollout(self._buffer_size, "c3m")
        self._progress = float(timestep) / max(1, timesteps)
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
            "C2RL/CMG/pd_loss": info["pd_loss"],
            "C2RL/CMG/pd_reg":  info["pd_reg"],
        }
        for k, v in loss_dict.items():
            self.track_data(f"Loss / {k}", v)
        return loss_dict


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLSkrlTrainer(Trainer):
    """skrl Trainer for C2RL — alternates con/opt rollouts and CMG updates."""

    def train(self) -> None:
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        num_envs = env.num_envs
        rollout_steps = self.cfg.rollouts if hasattr(self.cfg, "rollouts") else agent._cfg.rollouts
        timesteps = self.cfg.timesteps
        con_only = agent._con_only

        # con/opt RandomMemory buffers are allocated inside C2RLAgent.__init__ (at
        # agent.init() there) using the num_envs passed to its constructor — skrl
        # memory tensors are physically sized at creation and cannot be resized
        # afterwards, so num_envs/rollouts must already match here.
        if agent._num_envs != num_envs:
            raise ValueError(
                f"[C2RL] Agent was constructed with num_envs={agent._num_envs} but the "
                f"environment has num_envs={num_envs}. Pass the correct num_envs to "
                f"C2RLAgent(...) (ContractionRunner does this automatically)."
            )
        if agent._con_memory.memory_size != rollout_steps:
            raise ValueError(
                f"[C2RL] Agent con_memory was sized for rollouts={agent._con_memory.memory_size} "
                f"but the trainer is configured for rollouts={rollout_steps}. Keep "
                f"agent.rollouts and trainer.rollouts in sync in the YAML config."
            )

        agent.init(trainer_cfg=self.cfg)

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0

        total_iters = timesteps // (rollout_steps * (1 if con_only else 2))
        pbar = _tqdm.tqdm(range(max(1, total_iters)), desc="C2RL training", file=sys.stdout)

        for epoch in pbar:

            # ── Phase 1: contracting rollout ──────────────────────────── #
            agent._con_agent.enable_training_mode(True)
            agent._con_agent.memory.reset()
            con_obs_list = []

            for step in range(rollout_steps):
                agent._con_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = agent._con_agent.act(
                        observations, states, timestep=global_step, timesteps=timesteps
                    )
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None
                con_obs_list.append(observations.clone())
                agent._con_agent.record_transition(
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
            agent._con_agent.post_interaction(timestep=global_step, timesteps=timesteps)

            # ── Phase 2: optimal rollout ──────────────────────────────── #
            if not con_only and agent._opt_agent is not None:
                agent._opt_agent.enable_training_mode(True)
                agent._opt_agent.memory.reset()
                opt_obs_list = []

                for step in range(rollout_steps):
                    agent._opt_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                    with torch.no_grad():
                        actions, _ = agent._opt_agent.act(
                            observations, states, timestep=global_step, timesteps=timesteps
                        )
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
                    next_states = env.state() if hasattr(env, "state") else None
                    opt_obs_list.append(observations.clone())
                    agent._opt_agent.record_transition(
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
                agent._opt_agent.post_interaction(timestep=global_step, timesteps=timesteps)

            # ── CMG update ────────────────────────────────────────────── #
            cmg_dict = agent.update_cmg(timestep=global_step, timesteps=timesteps)
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            pd_loss = cmg_dict.get("C2RL/CMG/pd_loss", float("nan"))
            pbar.set_postfix(epoch=epoch, pd=f"{pd_loss:.3g}")

    def eval(self) -> None:
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        ppo = agent._opt_agent if (agent._opt_agent and not agent._con_only) else agent._con_agent
        ppo.enable_training_mode(False)
        observations, _ = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None
        done = False
        while not done:
            with torch.no_grad():
                actions, _ = ppo.models["policy"].act({"observations": observations}, role="policy")
            observations, rewards, terminated, truncated, _ = self.env.step(actions)
            states = self.env.state() if hasattr(self.env, "state") else None
            done = bool((terminated | truncated).all())
        ppo.enable_training_mode(True)

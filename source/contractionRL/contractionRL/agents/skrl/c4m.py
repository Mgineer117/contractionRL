"""C4M — Two-stage contraction-metric synthesis + frozen-CMG RL, native skrl Agent.

C4MAgent ablates C2RL's joint (CMG, con_policy) training into two SEQUENTIAL
stages instead:

  1. **Offline CMG synthesis (Phase A)** — an internal ``C3MAgent`` (see
     ``c3m.py``) trains the CMG (``W(x)``) jointly with a throwaway controller
     for ``c3m_epochs`` epochs, EXACTLY like standalone C3M: same
     ``pd_loss + c1_loss + c2_loss (+ os_loss)`` on a static/periodically
     refreshed offline ``(x, xref, uref)`` buffer, no environment rollouts.
     This is delegated wholesale to ``C3MSkrlTrainer`` (see
     ``C4MSkrlTrainer.train``) — zero duplicated contraction-loss code.
  2. **Freeze the CMG, discard the synthesis controller** — ``freeze_cmg()``
     sets ``requires_grad_(False)`` on every CMG parameter and puts it in
     ``eval()`` permanently. The C3M controller that helped shape it during
     Phase A is never referenced again.
  3. **Train the deployed policy (Phase B)** — a REAL skrl ``PPO`` or ``SAC``
     agent rolls out against the environment and trains on the Mahalanobis
     tracking reward ``-tracking_scaler·||e||²_M - control_scaler·||u-uref||²``,
     where ``M(x) = W(x)⁻¹`` is the now-FROZEN metric from Phase A. Unlike
     C2RL's con_policy/opt_policy (which shape the CMG themselves and need a
     policy Jacobian in the CMG loss), Phase B needs no Jacobian/dynamics at
     all — the CMG is static, so the reward is a pure forward pass.

This is the C2RL vs. C4M contrast: C2RL trains (CMG, con_policy) jointly and
lets opt_policy free-ride on whatever metric con_policy shaped; C4M fully
decouples "synthesize a metric offline" from "learn a policy under it".

Shares two pieces of logic with C2RL verbatim via ``rl_glue.py`` (NOT
duplicated): the Mahalanobis reward computation and the raw-cfg -> real
PPO_CFG/SAC_CFG translation (``make_base_rl_cfg``) — see that module for the
full normalization rationale (uref/angle_idx exclusion, raw-vs-normalized CMG
consistency, etc.), which applies identically here since Phase B's deployed
policy is a bare skrl PPO/SAC agent just like C2RL's con_agent/opt_agent.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import tqdm as _tqdm

from skrl.agents.torch.base import Agent, AgentCfg
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .c3m import C3MAgent, C3MCfg, C3MSkrlTrainer, C3MTrainerCfg
from .rl_glue import compute_mahalanobis_reward, filter_cfg_fields, make_base_rl_cfg


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

# NOTE: base_algorithm is NOT a field on either cfg below — it's an explicit
# C4MAgent constructor kwarg (see ContractionRunner._setup_c4m), same
# convention as C2RLAgent. Three C3M-synthesis fields are prefixed `c3m_`
# (batch_size/learning_rate_scheduler(_kwargs)) because those names collide
# with real PPO_CFG/SAC_CFG fields of the SAME name that also live on this
# flat dataclass — every other C3M-mirroring field (W_lr, lbd, eps, w_ub/
# w_lb, ...) has no such collision and keeps its natural C3M name.

@dataclass
class C4MPPOCfg(AgentCfg):
    """C4M config for base_algorithm="PPO". PPO fields mirror skrl's PPO_CFG."""
    # PPO shared config (see skrl.agents.torch.ppo.PPO_CFG) — the deployed
    # policy's own PPO sub-agent is built from this.
    rollouts: int = 16
    learning_epochs: int = 8
    mini_batches: int = 2
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 2.5
    kl_threshold: float = 0.0
    grad_norm_clip: float = 0.5
    time_limit_bootstrap: bool = False
    use_state_norm: bool = False
    use_value_norm: bool = True
    # Deployed policy's (static) discount factor — Phase B trains against a
    # frozen CMG, so there is no con/opt duality here, just one policy.
    discount_factor: float = 0.99
    # Mahalanobis reward (Phase B)
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    # ── C3M offline CMG synthesis (Phase A) ──────────────────────────────── #
    c3m_epochs: int = 15  # number of C3MAgent.update() epoch-passes over the offline buffer
    c3m_eval_interval: int = 0  # 0 = disabled; C3MSkrlTrainer's periodic bounded-rollout eval
    c3m_batch_size: int = 1024
    c3m_learning_rate_scheduler: str = "ExponentialLR"
    c3m_learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {"gamma": 0.9999})
    W_lr: float = 3e-4
    u_lr: float = 3e-4
    actor_lr: float | None = None
    policy_update_per_cmg_update: int = 1
    lbd: float = 1e-2
    eps: float = 1e-2
    detach_warmup_frac: float = 1.0 / 3.0
    pd_loss_num_samples: int = 128
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Only feeds Phase A
    # (the CMG synthesis); once frozen, Phase B needs no dynamics at all.
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C4MSACCfg(AgentCfg):
    """C4M config for base_algorithm="SAC". SAC fields mirror skrl's SAC_CFG."""
    rollouts: int = 16  # not a real SAC_CFG field — sizes the RandomMemory buffer (see C4MAgent)
    gradient_steps: int = 1
    batch_size: int = 64
    polyak: float = 0.005
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.0
    learn_entropy: bool = True
    initial_entropy_value: float = 0.2
    use_state_norm: bool = False
    discount_factor: float = 0.99
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    # ── C3M offline CMG synthesis (Phase A) ──────────────────────────────── #
    c3m_epochs: int = 15
    c3m_eval_interval: int = 0
    c3m_batch_size: int = 1024
    c3m_learning_rate_scheduler: str = "ExponentialLR"
    c3m_learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {"gamma": 0.9999})
    W_lr: float = 3e-4
    u_lr: float = 3e-4
    actor_lr: float | None = None
    policy_update_per_cmg_update: int = 1
    lbd: float = 1e-2
    eps: float = 1e-2
    detach_warmup_frac: float = 1.0 / 3.0
    pd_loss_num_samples: int = 128
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C4MTrainerCfg(TrainerCfg):
    timesteps: int = 300000  # Phase B (deployed-policy) env steps; Phase A's length is agent.c3m_epochs


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C4MAgent(Agent):
    """C4M agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict:
      ``"c3m_policy"`` — throwaway CLActor/ControllerNetwork controller used
        ONLY to help shape the CMG during Phase A (same role as C3M's
        ``"policy"``); never deployed, discarded once the CMG freezes.
      ``"cmg"``        — MetricNetwork (shared: the SAME instance Phase A
        trains and Phase B reads the frozen metric from).
      ``"policy"``     — the REAL deployed SAC/PPO policy (Phase B).
      ``"value"`` (PPO) or ``"critic_1"``/``"critic_2"``/``"target_critic_1"``/
        ``"target_critic_2"`` (SAC) — Phase B's own critic(s).
      ``"dynamics"``   — optional NeuralDynamics, feeds Phase A only.

    Extra constructor kwargs mirror C2RLAgent's: ``get_rollout``,
    ``get_f_and_B``, ``x_dim``, ``u_dim``, ``num_envs``, ``angle_idx``,
    ``base_algorithm``.
    """

    def __init__(
        self,
        *,
        cfg: C4MPPOCfg | C4MSACCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_rollout: Callable,
        get_f_and_B: Callable,
        base_algorithm: str = "PPO",
        x_dim: int | None = None,
        u_dim: int | None = None,
        num_envs: int = 1,
        angle_idx: list | None = None,
    ) -> None:
        self._angle_idx = list(angle_idx or [])
        CfgCls = C4MSACCfg if base_algorithm.upper() == "SAC" else C4MPPOCfg
        if isinstance(cfg, dict):
            self._raw_cfg = cfg.copy()
            parsed_cfg = CfgCls(**filter_cfg_fields(cfg, CfgCls, context="C4MAgent"))
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
        self._base_algorithm = base_algorithm.upper()
        self._num_envs = num_envs

        # ── Phase A: internal C3MAgent for offline CMG synthesis ─────────── #
        c3m_models = {"policy": models["c3m_policy"], "cmg": models["cmg"]}
        if "dynamics" in models:
            c3m_models["dynamics"] = models["dynamics"]
        c3m_cfg = C3MCfg(
            batch_size=parsed_cfg.c3m_batch_size,
            W_lr=parsed_cfg.W_lr,
            u_lr=parsed_cfg.u_lr,
            actor_lr=parsed_cfg.actor_lr,
            policy_update_per_cmg_update=parsed_cfg.policy_update_per_cmg_update,
            lbd=parsed_cfg.lbd,
            eps=parsed_cfg.eps,
            w_ub=parsed_cfg.w_ub,
            w_lb=parsed_cfg.w_lb,
            detach_warmup_frac=parsed_cfg.detach_warmup_frac,
            pd_loss_num_samples=parsed_cfg.pd_loss_num_samples,
            use_empirical_dynamics=parsed_cfg.use_empirical_dynamics,
            learning_rate_scheduler=parsed_cfg.c3m_learning_rate_scheduler,
            learning_rate_scheduler_kwargs=parsed_cfg.c3m_learning_rate_scheduler_kwargs,
            dynamics_lr=parsed_cfg.dynamics_lr,
            dynamics_lr_scheduler=parsed_cfg.dynamics_lr_scheduler,
            dynamics_lr_scheduler_kwargs=parsed_cfg.dynamics_lr_scheduler_kwargs,
            dynamics_batch_size=parsed_cfg.dynamics_batch_size,
            dynamics_pretrain_epochs=parsed_cfg.dynamics_pretrain_epochs,
            dynamics_pretrain_data_path=parsed_cfg.dynamics_pretrain_data_path,
            experiment={
                "directory": os.path.join(self.experiment_dir, "c3m_synthesis"),
                "write_interval": 1,
                "checkpoint_interval": 0,
            },
        )
        self._c3m_agent = C3MAgent(
            cfg=c3m_cfg,
            models=c3m_models,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        # Shared CMG instance — freezing this (freeze_cmg, below) freezes it
        # for BOTH self._c3m_agent (Phase A) and Phase B's reward computation.
        self._ccm_gen = models["cmg"].ccm_gen

        # ── Phase B: a real skrl PPO/SAC agent for the deployed policy ───── #
        rollouts = self._raw_cfg.get("rollouts", 300)
        if self._base_algorithm == "PPO":
            mem_size = rollouts
        else:
            # See C2RLAgent's identical rationale: a per-parallel-env buffer,
            # so Isaac Sim's 1000+ envs don't try to allocate a billion+
            # transitions with skrl's usual ~1M default.
            mem_size = self._raw_cfg.get("memory_size", 10000)
            if mem_size == -1:
                mem_size = 10000
        memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        self._memory = memory

        # For SAC, monkey-patch sample() to recompute the Mahalanobis reward
        # from the raw stored observations against the (already frozen, by
        # the time Phase B runs) CMG — simpler than C2RL's equivalent patch:
        # no Jacobian/dynamics involved since the metric is static.
        if self._base_algorithm == "SAC":
            def _dynamic_maha_sample(names, batch_size, mini_batches=1, sequence_length=1, _orig=memory.sample):
                batches = _orig(names, batch_size=batch_size, mini_batches=mini_batches, sequence_length=sequence_length)
                try:
                    obs_idx = names.index("observations") if "observations" in names else names.index("states")
                    act_idx = names.index("actions")
                    rew_idx = names.index("rewards")
                    for b in batches:
                        b[rew_idx] = self._compute_mahalanobis_reward(b[obs_idx], b[act_idx])
                except ValueError:
                    pass  # if rewards isn't requested, don't modify
                return batches
            memory.sample = _dynamic_maha_sample

        rl_kwargs = dict(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        base_cfg = make_base_rl_cfg(
            self._raw_cfg,
            base_algorithm=self._base_algorithm,
            gamma=parsed_cfg.discount_factor,
            name="policy",
            experiment_dir=self.experiment_dir,
            device=device,
            observation_space=observation_space,
            angle_idx=self._angle_idx,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        if self._base_algorithm == "PPO":
            from skrl.agents.torch.ppo import PPO as BaseRLAgent
            rl_models = {"policy": models["policy"], "value": models["value"]}
        elif self._base_algorithm == "SAC":
            from skrl.agents.torch.sac import SAC as BaseRLAgent
            rl_models = {
                "policy": models["policy"],
                "critic_1": models["critic_1"], "critic_2": models["critic_2"],
                "target_critic_1": models["target_critic_1"], "target_critic_2": models["target_critic_2"],
            }
        else:
            raise ValueError(f"[C4M] Unsupported base_algorithm: {self._base_algorithm}")

        self._rl_agent = BaseRLAgent(cfg=base_cfg, models=rl_models, memory=memory, **rl_kwargs)

        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        from contractionRL.agents.skrl.models import ControllerNetwork
        patch_kl_logging(self._rl_agent)
        patch_sac_entropy_clamp(self._rl_agent)
        _std_dev_annealing_kwargs = self._raw_cfg.get("std_dev_annealing_kwargs")
        patch_ppo_std_annealing(self._rl_agent, isinstance(models.get("policy"), ControllerNetwork), _std_dev_annealing_kwargs)

        self._rl_agent.init()

        checkpoint_extra = (
            {"value": models["value"]} if self._base_algorithm == "PPO" else
            {"critic_1": models["critic_1"], "critic_2": models["critic_2"]}
        )
        self.checkpoint_modules.update({
            "c3m_policy": models["c3m_policy"],
            "cmg": models["cmg"],
            "policy": models["policy"],
            **checkpoint_extra,
        })
        if "dynamics" in models:
            self.checkpoint_modules["dynamics"] = models["dynamics"]

    # ── skrl Agent interface ────────────────────────────────────────────── #

    def act(self, observations, states, *, timestep: int, timesteps: int):
        with torch.no_grad():
            result = self._rl_agent.models["policy"].act({"observations": observations}, role="policy")
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
        # C4MSkrlTrainer calls update_policy directly (Phase B only — Phase A
        # is driven wholesale by an internal C3MSkrlTrainer, see the module
        # trainer below).
        pass

    # ── Mahalanobis reward (Phase B, frozen CMG) ─────────────────────────── #

    def _compute_mahalanobis_reward(self, observations: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self._cfg
        return compute_mahalanobis_reward(
            self._ccm_gen, observations, actions,
            x_dim=self._x_dim, u_dim=self._u_dim, angle_idx=self._angle_idx,
            w_lb=cfg.w_lb, tracking_scaler=cfg.tracking_scaler, control_scaler=cfg.control_scaler,
        )

    def update_policy(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None,
        *, timestep: int, timesteps: int,
    ) -> dict:
        """Inject the Mahalanobis reward and drive the base agent's update.

        PPO: overwrite the just-collected rollout's rewards, then call
        update() once (PPO's own rollouts-cadence hook never fires here — the
        trainer drives it explicitly, same as C2RL's update_con). SAC: the
        reward is recomputed per mini-batch inside the patched
        memory.sample(); the gradient step is driven by the trainer calling
        ``post_interaction()`` right after, so this is a no-op for SAC.
        """
        if self._base_algorithm == "PPO":
            maha_r = self._compute_mahalanobis_reward(observations, actions)
            try:
                r_tensor = self._rl_agent.memory.get_tensor_by_name("rewards")
                r_tensor.copy_(maha_r.view_as(r_tensor))
            except RuntimeError as e:
                import skrl
                skrl.logger.warning(f"[C4M] Failed to inject Mahalanobis reward in update_policy: {e}")
            self._rl_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def freeze_cmg(self) -> None:
        """Freeze the CMG at the Phase A -> Phase B boundary (see module docstring)."""
        for p in self._ccm_gen.parameters():
            p.requires_grad_(False)
        self._ccm_gen.eval()

    def save_dynamics(self, path: str) -> None:
        self._c3m_agent.save_dynamics(path)


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C4MSkrlTrainer(Trainer):
    """skrl Trainer for C4M — offline CMG synthesis, then frozen-CMG RL."""

    def _env_scalar_attr(self, *names):
        """See the identical helper on C3MSkrlTrainer/C2RLSkrlTrainer."""
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

    @staticmethod
    def _forward_env_log(agent, infos) -> None:
        """See C2RLSkrlTrainer._forward_env_log — same env extras["log"] forwarding."""
        if not isinstance(infos, dict):
            return
        log = infos.get("log")
        if not isinstance(log, dict):
            return
        for k, v in log.items():
            key = k if "/" in k else f"Info / {k}"
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    agent.track_data(key, v.item())
            elif isinstance(v, (int, float)):
                agent.track_data(key, float(v))

    def train(self) -> None:
        agent: C4MAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        timesteps = self.cfg.timesteps

        # Outer agent: resolves "auto" write_interval/checkpoint_interval
        # against `timesteps` (Phase B's length) and creates its own writer/
        # checkpoint dir. checkpoint_modules (c3m_policy/cmg/policy/critics/
        # dynamics) are all saved from HERE, not from the inner rl_agent
        # (whose own checkpoint_interval is hardcoded 0 — see make_base_rl_cfg).
        agent.init(trainer_cfg=self.cfg)
        from .contraction_metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))
        agent.enable_training_mode(True)

        # ── Phase A: reuse C3MSkrlTrainer wholesale for offline CMG synthesis
        # (dynamics pretraining, the alternating CMG/controller epoch loop,
        # periodic eval — zero new code needed here). ──────────────────────
        synth_trainer = C3MSkrlTrainer(
            cfg=C3MTrainerCfg(timesteps=agent._cfg.c3m_epochs, eval_interval=agent._cfg.c3m_eval_interval),
            env=env,
            agents=agent._c3m_agent,
        )
        synth_trainer.train()

        # ── Freeze the CMG; the C3M synthesis controller is discarded ────── #
        agent.freeze_cmg()

        # ── Phase B: rollout + train the deployed policy against the frozen
        # CMG's Mahalanobis reward — a simplified, single-policy version of
        # C2RLSkrlTrainer's rollout loop, minus the CMG update call (the CMG
        # is static here, never re-trained). ──────────────────────────────
        rl_agent = agent._rl_agent
        rl_agent.enable_training_mode(True)
        rollout_steps = agent._cfg.rollouts

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0
        # Coarse flush cadence for the INNER rl_agent — see C2RLSkrlTrainer's
        # identical comment: flushing every step/chunk would collapse the
        # 100-episode reward/timestep deques to a spiky few-episode curve.
        flush_interval = max(1, timesteps // 100)
        next_flush = flush_interval

        pbar = _tqdm.tqdm(total=timesteps, desc="C4M training (Phase B)", file=sys.stdout)
        while global_step < timesteps:
            if agent._base_algorithm == "PPO":
                rl_agent.memory.reset()
            obs_list, act_list = [], []

            steps_to_take = min(rollout_steps, timesteps - global_step)
            for _ in range(steps_to_take):
                rl_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = rl_agent.act(observations, states, timestep=global_step, timesteps=timesteps)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None
                obs_list.append(observations.clone())
                act_list.append(actions.clone())

                # Record with the Mahalanobis reward (mirrors
                # C2RLSkrlTrainer._record_with_maha) — the stored env reward
                # is a placeholder overwritten below (PPO) / at sample time
                # (SAC), so the LOGGED cumulative reward should reflect what
                # the policy actually optimizes, not the raw env reward.
                maha = agent._compute_mahalanobis_reward(observations, actions)
                rl_agent.record_transition(
                    observations=observations, states=states, actions=actions,
                    rewards=maha.view_as(rewards), next_observations=next_obs, next_states=next_states,
                    terminated=terminated, truncated=truncated, infos=infos,
                    timestep=global_step, timesteps=timesteps,
                )
                rl_agent.track_data("Reward / env_reward (mean)", float(rewards.float().mean()))
                self._forward_env_log(agent, infos)
                observations = next_obs
                states = next_states

                if agent._base_algorithm == "SAC":
                    agent.update_policy(None, None, timestep=global_step, timesteps=timesteps)
                    rl_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                    if global_step % flush_interval == 0 and getattr(rl_agent, "writer", None) is not None:
                        rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

                global_step += 1
                pbar.update(1)

            # Chunk-based update for PPO
            if agent._base_algorithm == "PPO":
                obs_tensor = torch.cat(obs_list, dim=0)
                act_tensor = torch.cat(act_list, dim=0)
                agent.update_policy(obs_tensor, act_tensor, timestep=global_step, timesteps=timesteps)
                rl_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                if global_step >= next_flush and getattr(rl_agent, "writer", None) is not None:
                    rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)
                    next_flush = global_step + flush_interval

            # Outer agent: drives ITS OWN checkpoint cadence (checkpoint_modules)
            # and flushes whatever Stability/* logs _forward_env_log collected
            # into tracking_data this chunk — once per chunk, same as C2RL.
            agent.post_interaction(timestep=global_step, timesteps=timesteps)
            if getattr(agent, "writer", None) is not None:
                agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        if getattr(rl_agent, "writer", None) is not None:
            rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        if agent._c3m_agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            os.makedirs(os.path.dirname(dyn_path), exist_ok=True)
            agent.save_dynamics(dyn_path)

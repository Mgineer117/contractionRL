"""C2RL — single-policy contraction-metric RL with a ``use_cmg`` switch.

C2RLAgent trains ONE real skrl ``PPO``/``SAC`` policy (``base_algorithm="PPO"``
→ a skrl ``PPO`` sub-agent, ``base_algorithm="SAC"`` → a skrl ``SAC`` sub-agent)
against a Mahalanobis tracking reward

    ``-tracking_scaler·||e||²_M - control_scaler·||u - uref||²``,   e = x - xref

where the metric ``M(x) = W(x)⁻¹`` comes from one of two interchangeable
sources, selected by the ``use_cmg`` config flag (default ``False``):

  * **use_cmg = False (default) — online CM, no metric network.** The metric is
    solved on-the-fly per state by a convex feasibility SDP (Tsukamoto's Neural
    Contraction Metric, ``cm_synthesis.solve_cm_metric``): at each state a dual
    contraction metric ``W`` is found, inverted to ``M`` and turned into the
    reward. There is no CMG network and no offline synthesis phase — just the
    policy trained under the online CM metric. (This is what C3RL used to be.)
  * **use_cmg = True — regressed + frozen CMG network (Tsukamoto's full NCM).** A
    CMG network ``W(x)`` is fit OFFLINE, before RL: sample ``cmg_memory_size``
    states, solve one convex SDP per state for the target metric ``W*(x)``
    (``cm_synthesis.build_cm_dataset``), then MSE-regress the CMG network onto the
    feasible ``{x -> W*}`` pairs (``regress_cmg``) and FREEZE it (``freeze_cmg``).
    This is the same SDP as the online path — the network just amortizes it into a
    single batched forward pass so Phase B's Mahalanobis reward is O(1) per state
    (``M(x) = W(x)⁻¹``, static). There is NO C3M controller and NO differentiable
    certificate loss — the network is trained purely by regression onto the SDP
    solutions (this is Tsukamoto's actual Neural Contraction Metric recipe).
    The ``cmg_memory_size`` states are drawn uniformly either from the classic
    env's analytic state space (``get_rollout``, unlimited supply) or, when
    ``dynamics_pretrain_data_path`` points to an offline ``dynamics_data.npz``,
    from that same offline data (capped to its size, with a warning if
    ``cmg_memory_size`` asks for more samples than are on disk).

Orthogonal to ``use_cmg``, ``cm_formulation`` selects WHICH pointwise SDP is
solved (both modes, both online and offline) — ``"ccm"`` (default, Manchester-
style, eliminates the control matrix via the annihilator, existence-only) or
``"cvstem"`` (Tsukamoto CV-STEM-style, keeps the control matrix via a Riccati
term instead of eliminating it). See ``cm_synthesis.py``'s module docstring for
the LMIs and the tradeoff between them.

Both modes share the SAME single-policy Phase B rollout loop and the SAME
reward-injection mechanism (see ``C2RLSkrlTrainer``):

  * **SAC** overwrites each replayed transition's reward with the Mahalanobis
    reward inside a patched ``memory.sample()`` and writes it back into the
    replay buffer. With ``use_cmg`` the frozen CMG is re-evaluated per sampled
    mini-batch; without it the per-state CM SDP is solved the FIRST time a
    transition is replayed and CACHED back into the buffer, so later replays of
    the same transition reuse it (off-policy replays each transition many times,
    amortizing the solve — see ``_setup_online_cm_cache``).
  * **PPO** discards its on-policy rollout each update, so there is nothing to
    cache: it computes the Mahalanobis reward over the whole fresh rollout batch
    in ``update_policy`` and overwrites that rollout's rewards right before the
    PPO update.

Normalization: the CMG metric / CM SDP and the Mahalanobis reward always use
RAW observations — ``M(x)`` and the tracking error ``e = x - xref`` are defined
in raw physical coordinates, and per-dimension normalization would scale ``x``
and ``xref`` independently, distorting ``e``. ``uref`` and any ``angle_idx``
columns are likewise excluded from observation normalization (see
``rl_glue.make_base_rl_cfg`` for the full rationale, shared with the deployed
PPO/SAC policy built here). ``use_state_norm`` defaults to False in every
shipped config.

Learned dynamics (Isaac / ``use_empirical_dynamics=True``): a ``NeuralDynamics``
model provides ``f``/``B``/``B_null`` and ``∂f/∂x``. It is pretrained before
training and (use_cmg=True) refined online during Phase A. use_cmg=False needs
it for every online CM SDP; use_cmg=True needs it only for Phase A synthesis.
Classic envs use their analytical ``get_f_and_B`` and skip this.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import tqdm as _tqdm

from skrl.agents.torch.base import Agent, AgentCfg
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .math_utils import build_lr_scheduler
from .rl_glue import compute_mahalanobis_reward, filter_cfg_fields, make_base_rl_cfg


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

# NOTE: base_algorithm is NOT a field on either cfg below — it's an explicit
# C2RLAgent constructor kwarg (see ContractionRunner._setup_c2rl), set by which
# entry point you use (skrl_c2rl_ppo_cfg.yaml / skrl_c2rl_sac_cfg.yaml). Each
# cfg's base-algorithm fields mirror the REAL skrl PPO_CFG/SAC_CFG field names
# 1:1, so a c2rl-sac.yaml actually validates against SAC's own parameter names.
# make_base_rl_cfg() still reads from the raw yaml dict (self._raw_cfg) and
# filters against whichever of PPO_CFG/SAC_CFG applies, so any valid field works
# from yaml even if not declared below.

@dataclass
class C2RLPPOCfg(AgentCfg):
    """C2RL config for base_algorithm="PPO". PPO fields mirror skrl's PPO_CFG."""
    # PPO shared config (see skrl.agents.torch.ppo.PPO_CFG) — the deployed
    # policy's own PPO sub-agent is built from this.
    rollouts: int = 16
    learning_epochs: int = 8
    mini_batches: int = 2
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 2.5
    kl_threshold: float = 0.0
    grad_norm_clip: float = 0.5
    time_limit_bootstrap: bool = False
    use_state_norm: bool = False  # off by default — see module docstring / rl_glue.make_base_rl_cfg
    use_value_norm: bool = True
    rewards_shaper_scale: float = 1.0  # yaml convenience for PPO_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    # Set by ContractionRunner from the yaml `memory:` block's memory_size, NOT
    # read from `agent:` directly; declared purely so filter_cfg_fields()
    # recognizes it instead of warning.
    memory_size: int = -1
    # Deployed policy's discount factor — a single policy trained against the
    # Mahalanobis reward, so there is no con/opt duality here.
    discount_factor: float = 0.99
    # ── Metric source (use_cmg False → online CM SDP; True → frozen CMG net) ─ #
    use_cmg: bool = False
    # Mahalanobis reward (both modes)
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2  # contraction rate λ — shared by BOTH modes' SDP (online per-state / offline dataset)
    # ── SDP contraction metric (shared by both modes) — see cm_synthesis.py ── #
    cm_eps: float = 1e-2   # strict-definiteness margin on the contraction LMI
    cm_solver: str = "SCS"  # cvxpy SDP solver
    # "ccm" (default) — Manchester-style, eliminates B via the annihilator,
    # existence-only certificate (unchanged legacy behavior). "cvstem" —
    # Tsukamoto CV-STEM-style, keeps B via a Riccati BR⁻¹Bᵀ term instead of
    # eliminating it. See cm_synthesis.py module docstring for the LMIs.
    cm_formulation: str = "ccm"
    cvstem_r_scaler: float = 1.0  # R = cvstem_r_scaler·I in the BR⁻¹Bᵀ term (mirrors sdlqr.py's R_scaler); "cvstem" only
    # On SDP infeasibility at a state, retry that state ALONE with λ halved,
    # up to this many times, before giving up on it (0 = old behavior, drop
    # immediately). See cm_synthesis._solve_cm_metric_with_backoff.
    max_lambda_reductions: int = 5
    # Guards build_cm_dataset against silently regressing the CMG onto a small,
    # likely-biased subset of states — raises before regression if the SDP's
    # feasible fraction falls below this (0.0 = old behavior, only guards
    # against 0% feasible; see cm_synthesis.build_cm_dataset). Shared-SDP
    # concern, same as cm_eps/cm_solver, hence grouped with them (yaml `cm:`).
    min_feasibility_rate: float = 0.0
    # Cache path for the synthesized {x, W} CM dataset (build_cm_dataset's
    # expensive per-state SDP solve) — see synthesize_cmg. Loaded instead of
    # re-solving when it exists and matches lbd/w_lb/w_ub/cm_eps/cm_solver/
    # cmg_memory_size exactly; written after a fresh solve otherwise. Defaults
    # to a `cm_data.npz` next to dynamics_pretrain_data_path when unset (Isaac
    # envs); classic envs with no data_path need this set explicitly to get
    # caching at all (there's no offline dynamics file to derive a path from).
    cm_data_path: str = ""
    # ── use_cmg=True: offline SDP-dataset + CMG-network regression (Tsukamoto NCM) ─ #
    # Sample cmg_memory_size states — uniformly from the classic env's analytic
    # state space (get_rollout) or, when dynamics_pretrain_data_path is set,
    # uniformly from that offline dynamics_data.npz (capped + warned if
    # cmg_memory_size exceeds the data on disk; see synthesize_cmg) — solve one
    # SDP per state (solve_cm_metric, reusing lbd/w_lb/w_ub/cm_eps/cm_solver
    # above), then MSE-regress the CMG network onto {x -> W*} for
    # cmg_regress_epochs and freeze it (see build_cm_dataset / regress_cmg). NO
    # C3M controller, NO differentiable certificate loss.
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 200
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    # Held out from cmg_memory_size as a validation split never regressed on;
    # regress_cmg stops once its MSE hasn't improved for cmg_early_stop_patience
    # consecutive epochs, restoring the best-val-epoch CMG weights instead of
    # whatever cmg_regress_epochs happens to land on (see cm_synthesis.regress_cmg
    # / math_utils.EarlyStopper). <=0 disables both (always regress the full budget).
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Feeds every SDP solve
    # (online per state when use_cmg=False, offline dataset when use_cmg=True).
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""
    # Fixed pretraining buffer size — sampled ONCE (offline-data subsample when
    # dynamics_pretrain_data_path is set, else a fresh get_rollout draw), then
    # multi-epoch trained over, mirroring cmg_memory_size's role in CMG
    # synthesis (see dynamics_pretrain.pretrain_dynamics). Classic envs can
    # feasibly use any size (synthetic analytic sampling); Isaac envs with a
    # data_path are capped (+ warned) to the offline data actually on disk.
    emp_dynamics_memory_size: int = 8192
    # Classic envs only: how many distinct control vectors get paired with each
    # sampled state in a "dynamics" rollout (env_base.get_rollout); replaces
    # the old hardcoded 3.
    num_controls_per_state: int = 3
    # Held out from emp_dynamics_memory_size as a validation split never
    # trained on; pretrain_dynamics stops once its MSE hasn't improved for
    # dynamics_early_stop_patience consecutive epochs, restoring the
    # best-val-epoch NeuralDynamics weights instead of whatever
    # dynamics_pretrain_epochs happens to land on (see
    # dynamics_pretrain.pretrain_dynamics / math_utils.EarlyStopper). <=0
    # disables both (always pretrain the full budget).
    dynamics_val_frac: float = 0.1
    dynamics_early_stop_patience: int = 10


@dataclass
class C2RLSACCfg(AgentCfg):
    """C2RL config for base_algorithm="SAC". SAC fields mirror skrl's SAC_CFG."""
    rollouts: int = 16  # not a real SAC_CFG field — sizes the RandomMemory buffer (see C2RLAgent)
    gradient_steps: int = 1
    batch_size: int = 64
    polyak: float = 0.005
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.0
    learn_entropy: bool = True
    initial_entropy_value: float = 0.2
    use_state_norm: bool = False  # off by default — see module docstring / rl_glue.make_base_rl_cfg
    rewards_shaper_scale: float = 1.0  # yaml convenience for SAC_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    memory_size: int = -1
    discount_factor: float = 0.99
    # ── Metric source ─────────────────────────────────────────────────────── #
    use_cmg: bool = False
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2
    # ── SDP contraction metric (shared by both modes) — see cm_synthesis.py ── #
    cm_eps: float = 1e-2
    cm_solver: str = "SCS"
    cm_formulation: str = "ccm"  # "ccm" | "cvstem" — see cm_synthesis.py module docstring
    cvstem_r_scaler: float = 1.0
    max_lambda_reductions: int = 5  # see cm_synthesis._solve_cm_metric_with_backoff
    min_feasibility_rate: float = 0.0
    cm_data_path: str = ""
    # ── use_cmg=True: offline SDP-dataset + CMG-network regression (Tsukamoto NCM) ─ #
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 200
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Dynamics
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""
    emp_dynamics_memory_size: int = 8192
    num_controls_per_state: int = 3
    dynamics_val_frac: float = 0.1
    dynamics_early_stop_patience: int = 10


@dataclass
class C2RLTrainerCfg(TrainerCfg):
    timesteps: int = 300000  # deployed-policy (RL) env steps; use_cmg=True's offline
                             # SDP+regression synthesis runs once before this loop


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLAgent(Agent):
    """C2RL agent — native skrl Agent, single deployed policy.

    Models in ``models`` dict:
      ``"policy"``   — the REAL deployed SAC/PPO policy.
      ``"value"`` (PPO) or ``"critic_1"``/``"critic_2"``/``"target_critic_1"``/
        ``"target_critic_2"`` (SAC) — the deployed policy's own critic(s).
      ``"dynamics"`` — optional NeuralDynamics (use_empirical_dynamics=True).

      Only when ``use_cmg=True``:
      ``"cmg"``        — MetricModel whose network is regressed onto the offline
        SDP solutions and then frozen (see synthesize_cmg); read for the reward.

    Extra constructor kwargs: ``get_rollout``, ``get_f_and_B``, ``x_dim``,
    ``u_dim``, ``num_envs``, ``angle_idx``, ``base_algorithm``.
    """

    def __init__(
        self,
        *,
        cfg: C2RLPPOCfg | C2RLSACCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_rollout: Callable,
        get_f_and_B: Callable | None = None,
        base_algorithm: str = "PPO",
        x_dim: int | None = None,
        u_dim: int | None = None,
        num_envs: int = 1,
        angle_idx: list | None = None,
    ) -> None:
        self._angle_idx = list(angle_idx or [])
        CfgCls = C2RLSACCfg if base_algorithm.upper() == "SAC" else C2RLPPOCfg
        if isinstance(cfg, dict):
            self._raw_cfg = cfg.copy()
            parsed_cfg = CfgCls(**filter_cfg_fields(cfg, CfgCls, context="C2RLAgent"))
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
        self._use_cmg = bool(parsed_cfg.use_cmg)
        self._get_rollout = get_rollout

        # ── Metric source setup ──────────────────────────────────────────── #
        # Both modes solve the SAME convex contraction-metric SDP (cm_synthesis.py);
        # they differ only in WHEN. This agent always owns the (optional) learned
        # dynamics directly — the SDP needs f/B/∂f/∂x either way (online per state
        # for use_cmg=False, over the offline dataset for use_cmg=True).
        #   use_cmg=False: no CMG network — the SDP is solved per rollout state.
        #   use_cmg=True:  a CMG network (models["cmg"]) is fit OFFLINE by solving
        #     the SDP over sampled states and regressing onto {x -> W*}, then frozen
        #     (see C2RLSkrlTrainer.train / build_cm_dataset + regress_cmg).
        self._setup_dynamics(parsed_cfg, models, get_f_and_B, x_dim, u_dim, device)
        self._ccm_gen = models["cmg"].ccm_gen if self._use_cmg else None

        # ── Phase B: a real skrl PPO/SAC agent for the deployed policy ───── #
        if self._base_algorithm == "PPO":
            mem_size = parsed_cfg.rollouts
        else:
            # A per-parallel-env buffer, so Isaac Sim's 1000+ envs don't try to
            # allocate skrl's usual ~1M default × num_envs and OOM.
            mem_size = parsed_cfg.memory_size
            if mem_size == -1:
                mem_size = 10000
        memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        self._memory = memory

        # SAC reward injection: overwrite each replayed transition's reward with
        # the Mahalanobis reward. use_cmg=True re-evaluates the frozen CMG per
        # sampled mini-batch (cheap, static metric); use_cmg=False solves + caches
        # the per-state CM SDP once per transition (see _setup_online_cm_cache).
        if self._base_algorithm == "SAC":
            if self._use_cmg:
                self._setup_frozen_cmg_sample(memory)
            else:
                self._setup_online_cm_cache(memory, num_envs)

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
            raise ValueError(f"[C2RL] Unsupported base_algorithm: {self._base_algorithm}")

        self._rl_agent = BaseRLAgent(cfg=base_cfg, models=rl_models, memory=memory, **rl_kwargs)

        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        from contractionRL.agents.skrl.models import CLActorModel
        patch_kl_logging(self._rl_agent)
        patch_sac_entropy_clamp(self._rl_agent)
        _std_dev_annealing_kwargs = parsed_cfg.std_dev_annealing_kwargs
        patch_ppo_std_annealing(self._rl_agent, isinstance(models.get("policy"), CLActorModel), _std_dev_annealing_kwargs)

        self._rl_agent.init()

        checkpoint_extra = (
            {"value": models["value"]} if self._base_algorithm == "PPO" else
            {"critic_1": models["critic_1"], "critic_2": models["critic_2"]}
        )
        self.checkpoint_modules.update({
            "policy": models["policy"],
            **checkpoint_extra,
        })
        if self._use_cmg:
            self.checkpoint_modules["cmg"] = models["cmg"]
        if self._neural_dynamics is not None:
            self.checkpoint_modules["dynamics"] = self._neural_dynamics

    # ── Setup helpers ───────────────────────────────────────────────────── #

    def _setup_dynamics(self, cfg, models, get_f_and_B, x_dim, u_dim, device) -> None:
        """Own the (optional) learned dynamics directly (both use_cmg modes).

        The contraction-metric SDP needs ``f``/``B``/``B_null`` and ``∂f/∂x`` at
        every state it solves at (online per state when use_cmg=False, over the
        offline dataset when use_cmg=True). Under analytical dynamics that comes
        from the env's exact ``get_f_and_B``; otherwise a NeuralDynamics model
        (pretrained before training by the trainer's ``pretrain_dynamics``)
        provides it. This mirrors C3M's dynamics interface expected by
        ``dynamics_pretrain``.
        """
        if not cfg.use_empirical_dynamics:
            if get_f_and_B is None:
                raise ValueError(
                    "C2RL: analytical dynamics (use_empirical_dynamics=False) requires a "
                    "get_f_and_B callable (classic envs only). Isaac Sim envs have no "
                    "analytical dynamics — set use_empirical_dynamics=True."
                )
            self._get_f_and_B = get_f_and_B
            self._neural_dynamics = None
            self._dynamics_optimizer = None
            self._dynamics_lr_scheduler = None
        else:
            self._neural_dynamics = models.get("dynamics", None)
            if self._neural_dynamics is None:
                raise ValueError(
                    "C2RL requires a 'dynamics' model in the models dict when "
                    "use_empirical_dynamics=True (add a models.dynamics block to the config)."
                )
            self._get_f_and_B = self._neural_dynamics.get_f_and_B
            self._dynamics_optimizer = torch.optim.Adam(
                self._neural_dynamics.parameters(), lr=cfg.dynamics_lr
            )
            self._dynamics_lr_scheduler = build_lr_scheduler(
                self._dynamics_optimizer, cfg.dynamics_lr_scheduler, cfg.dynamics_lr_scheduler_kwargs
            )

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
        # C2RLSkrlTrainer drives update_policy (Phase B) directly.
        pass

    # ── Mahalanobis reward ──────────────────────────────────────────────── #

    def _compute_reward(self, observations: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Mahalanobis tracking reward from the active metric source.

        use_cmg=True: forward pass through the frozen CMG (compute_mahalanobis_reward).
        use_cmg=False: online per-state CM SDP solve (compute_cm_reward_online).
        Both read RAW observations (see module docstring).
        """
        cfg = self._cfg
        if self._use_cmg:
            return compute_mahalanobis_reward(
                self._ccm_gen, observations, actions,
                x_dim=self._x_dim, u_dim=self._u_dim, angle_idx=self._angle_idx,
                w_lb=cfg.w_lb, tracking_scaler=cfg.tracking_scaler, control_scaler=cfg.control_scaler,
            )
        from .cm_synthesis import compute_cm_reward_online
        return compute_cm_reward_online(
            observations, actions, self._get_f_and_B,
            x_dim=self._x_dim, u_dim=self._u_dim, angle_idx=self._angle_idx,
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            tracking_scaler=cfg.tracking_scaler, control_scaler=cfg.control_scaler,
            solver=cfg.cm_solver, formulation=cfg.cm_formulation, r_scaler=cfg.cvstem_r_scaler,
            max_lambda_reductions=cfg.max_lambda_reductions,
        )

    # ── SAC reward-injection patches ────────────────────────────────────── #

    def _setup_frozen_cmg_sample(self, memory) -> None:
        """use_cmg=True SAC: recompute the Mahalanobis reward per sampled
        mini-batch from the (frozen, by Phase B) CMG's raw-obs metric.

        Simpler than the online-CM cache below — the metric is static, so
        there is nothing to invalidate/cache; every sample just re-evaluates the
        pure forward pass. Applying an obs preprocessor here would evaluate M and
        e = x - xref in normalized coordinates and distort the reward, so the raw
        stored observations are used (see module docstring).
        """
        def _dynamic_maha_sample(names, batch_size, mini_batches=1, sequence_length=1, _orig=memory.sample):
            batches = _orig(names, batch_size=batch_size, mini_batches=mini_batches, sequence_length=sequence_length)
            try:
                obs_idx = names.index("observations") if "observations" in names else names.index("states")
                act_idx = names.index("actions")
                rew_idx = names.index("rewards")
                for b in batches:
                    b[rew_idx] = self._compute_reward(b[obs_idx], b[act_idx])
            except ValueError:
                pass  # if rewards isn't requested (e.g. target actions), don't modify
            return batches
        memory.sample = _dynamic_maha_sample

    def _setup_online_cm_cache(self, memory, num_envs: int) -> None:
        """use_cmg=False SAC: online-CM reward with replay-buffer caching.

        No CMG. The FIRST time a transition is replayed we solve the CM
        feasibility SDP at its state to get M = W⁻¹, compute the Mahalanobis
        reward, and write it back into the buffer's ``rewards`` slot — so every
        later replay of the same transition reuses it (off-policy replays each
        transition many times, so one SDP per transition is far cheaper than
        pre-synthesizing a CMG).

        ``_maha_cached`` (one bool per flat buffer slot) tracks which slots hold a
        solved reward; ``add_samples`` is wrapped to invalidate slots as new
        transitions overwrite them (circular buffer); ``sample`` is wrapped to
        lazily fill any uncached sampled slots before returning the minibatches.
        """
        self._maha_cached = torch.zeros(
            memory.memory_size * num_envs, dtype=torch.bool, device=self._device
        )

        # Invalidate cache for buffer rows overwritten by new transitions.
        _orig_add = memory.add_samples

        def _add_samples(**tensors):
            mi0 = memory.memory_index
            _orig_add(**tensors)
            mi1 = memory.memory_index
            ms = memory.memory_size
            if mi1 > mi0:
                rows = range(mi0, mi1)
            elif mi1 < mi0:  # circular wrap
                rows = list(range(mi0, ms)) + list(range(0, mi1))
            else:  # partial-row write (env_index branch): invalidate the whole row
                rows = [mi0]
            for r in rows:
                self._maha_cached[r * num_envs:(r + 1) * num_envs] = False

        memory.add_samples = _add_samples

        # Lazily fill uncached sampled rewards, cache them, and refresh the
        # returned minibatches' rewards from the (now fully cached) buffer.
        _orig_sample = memory.sample

        def _dynamic_maha_sample(names, batch_size, mini_batches=1, sequence_length=1):
            batches = _orig_sample(
                names, batch_size=batch_size, mini_batches=mini_batches, sequence_length=sequence_length
            )
            if "rewards" not in names:
                return batches  # SAC also samples without rewards (e.g. target actions) — leave those
            rew_view = memory.tensors_view["rewards"]
            obs_key = "observations" if "observations" in memory.tensors else "states"
            obs_view = memory.tensors_view[obs_key]
            act_view = memory.tensors_view["actions"]

            idx = memory.sampling_indexes.to(rew_view.device)
            uncached = ~self._maha_cached[idx]
            if uncached.any():
                u_idx = idx[uncached]
                r = self._compute_reward(obs_view[u_idx], act_view[u_idx])
                rew_view[u_idx] = r.view(-1, 1).to(rew_view.dtype)
                self._maha_cached[u_idx] = True

            rew_idx = names.index("rewards")
            chunks = np.array_split(memory.sampling_indexes, mini_batches) if mini_batches > 1 else [memory.sampling_indexes]
            for b, ch in enumerate(chunks):
                batches[b][rew_idx] = rew_view[ch.to(rew_view.device)]
            return batches

        memory.sample = _dynamic_maha_sample

    # ── Phase B update ──────────────────────────────────────────────────── #

    def update_policy(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None,
        *, timestep: int, timesteps: int,
    ) -> dict:
        """Inject the Mahalanobis reward and drive the base agent's update.

        PPO: compute the Mahalanobis reward over the whole fresh rollout batch,
        overwrite the rollout's rewards, then call update() once (PPO's own
        rollouts-cadence hook never fires here — the trainer drives it
        explicitly). SAC: the reward is recomputed/solved inside the patched
        memory.sample() and the gradient step is driven by the trainer calling
        ``post_interaction()`` right after, so this is a no-op for SAC.
        """
        if self._base_algorithm == "PPO":
            maha_r = self._compute_reward(observations, actions)
            try:
                memory = self._rl_agent.memory
                r_tensor = memory.get_tensor_by_name("rewards")
                # The final rollout chunk of training may be shorter than
                # `rollouts` (timesteps not a multiple of rollout_steps), so
                # `observations`/`maha_r` can cover fewer rows than the
                # memory's fixed [rollout_steps, num_envs, 1] reward tensor —
                # only overwrite the rows that were actually filled.
                n_steps = observations.shape[0] // memory.num_envs
                r_tensor[:n_steps].copy_(maha_r.view(n_steps, memory.num_envs, 1))
            except RuntimeError as e:
                import skrl
                skrl.logger.warning(f"[C2RL] Failed to inject Mahalanobis reward in update_policy: {e}")
            self._rl_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def _train_dynamics(self, data: dict) -> float:
        """MSE training of NeuralDynamics on (x, u, x_dot) data (same as C3M).

        Called by the trainer's ``pretrain_dynamics`` (both use_cmg modes) when
        learning dynamics — the SDP needs f/B/∂f/∂x before any solve.
        """
        import torch.nn as nn
        dev = self._neural_dynamics.device
        x     = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        u     = torch.as_tensor(data["u"], dtype=torch.float32, device=dev)
        x_dot = torch.as_tensor(data["x_dot"], dtype=torch.float32, device=dev)

        pred = self._neural_dynamics.predict_x_dot(x, u)
        loss = nn.functional.mse_loss(pred, x_dot)

        self._dynamics_optimizer.zero_grad()
        loss.backward()
        if all(torch.isfinite(p.grad).all() for p in self._neural_dynamics.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._neural_dynamics.parameters(), 1.0)
            self._dynamics_optimizer.step()
        return loss.item()

    def freeze_cmg(self) -> None:
        """Freeze the regressed CMG before Phase B (use_cmg=True only)."""
        if self._ccm_gen is None:
            return
        for p in self._ccm_gen.parameters():
            p.requires_grad_(False)
        self._ccm_gen.eval()

    def _sample_cmg_x(self) -> np.ndarray | None:
        """Draw the ``cmg_memory_size`` states CMG synthesis will solve the SDP over.

        Uniformly subsampled from the offline ``dynamics_data.npz`` when
        ``dynamics_pretrain_data_path`` is set (capped to the data on disk, with
        a warning if ``cmg_memory_size`` asks for more), else ``None`` so
        ``build_cm_dataset`` falls back to freshly sampling the classic env's
        analytic state space via ``get_rollout``.
        """
        cfg = self._cfg
        data_path = getattr(cfg, "dynamics_pretrain_data_path", "") or None
        if not data_path:
            return None
        from .dynamics_pretrain import load_offline_dynamics_data
        x_all = load_offline_dynamics_data(data_path, tag="[C2RL]")["x"]
        n_avail = x_all.shape[0]
        n_samples = cfg.cmg_memory_size
        if n_samples > n_avail:
            print(f"[C2RL] WARNING: cmg_memory_size={n_samples} exceeds the "
                  f"{n_avail} available offline dynamics samples — using {n_avail} instead.")
            n_samples = n_avail
        idx = np.random.choice(n_avail, size=n_samples, replace=False)
        return x_all[idx]

    def synthesize_cmg(self, *, timesteps: int = 0) -> dict:
        """use_cmg=True offline synthesis: sample states, solve one contraction-metric
        SDP per state (build_cm_dataset), MSE-regress the CMG network onto the
        feasible ``{x -> W*}`` targets (regress_cmg), then freeze it. Called once
        by the trainer before Phase B — needs the dynamics already pretrained so
        the SDP has meaningful ``f``/``B``/``∂f/∂x``.

        Logs the SDP-synthesis diagnostics (feasibility rate, contraction-LMI
        residual — see ``cm_synthesis.build_cm_dataset``) as a single point,
        then the full per-epoch CMG-regression loss/LR curve, both at negative
        timesteps so they precede Phase B on the ``global_step`` x-axis — same
        convention ``dynamics_pretrain.py`` uses for the NeuralDynamics fit.

        The per-state SDP solve is the expensive part, so the resulting ``{x, W}``
        dataset is ALWAYS written to disk after a fresh solve — to ``cm_data_path``
        if set, else next to ``dynamics_pretrain_data_path`` (``cm_data_{formulation}.npz``
        — see ``cm_synthesis.cm_dataset_cache_path``) if THAT'S set, else to this
        run's own ``experiment_dir/checkpoints/cm_data_{formulation}.npz`` (same
        convention as the ``dynamics.pt`` checkpoint) so a synthesis is never
        silently thrown away even with no path configured. Every one of these
        paths is suffixed with ``cm_formulation`` (``with_formulation_suffix``)
        so ``"ccm"`` and ``"cvstem"`` runs never share (and can't clobber) the
        same cache file even when ``cm_data_path``/``dynamics_pretrain_data_path``
        is unchanged between them. Loading only happens for the first two
        (explicitly-configured, therefore stable-across-runs) locations — the
        per-run experiment_dir fallback is written but never auto-loaded, since
        its path is unique to this run; set ``cm_data_path`` explicitly to reuse
        a synthesis across runs. See ``load_cached_cm_dataset``/``save_cm_dataset``.
        """
        from .cm_synthesis import (
            build_cm_dataset, cm_dataset_cache_path, load_cached_cm_dataset,
            regress_cmg, save_cm_dataset, with_formulation_suffix,
        )
        cfg = self._cfg
        data_path = getattr(cfg, "dynamics_pretrain_data_path", "") or None
        explicit_cache_path = getattr(cfg, "cm_data_path", "") or None
        # Every cm_data*.npz path (explicit, auto-derived, or the per-run
        # fallback below) is formulation-suffixed — see with_formulation_suffix
        # — so switching cm_formulation between runs never overwrites or
        # silently reuses a differently-solved cache under the same filename.
        if explicit_cache_path:
            cache_path = with_formulation_suffix(Path(explicit_cache_path), cfg.cm_formulation)
        elif data_path:
            cache_path = cm_dataset_cache_path(data_path, cfg.cm_formulation)
        else:
            cache_path = None
        cache_kwargs = dict(
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            solver=cfg.cm_solver, num_samples=cfg.cmg_memory_size, tag="[C2RL]",
            formulation=cfg.cm_formulation, r_scaler=cfg.cvstem_r_scaler,
        )
        dataset = load_cached_cm_dataset(cache_path, **cache_kwargs) if cache_path else None
        if dataset is None:
            dataset = build_cm_dataset(
                self._get_rollout, self._get_f_and_B,
                x_dim=self._x_dim, u_dim=self._u_dim,
                lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
                num_samples=cfg.cmg_memory_size, solver=cfg.cm_solver,
                device=self._device, tag="[C2RL]",
                x_samples=self._sample_cmg_x(),
                min_feasibility_rate=cfg.min_feasibility_rate,
                formulation=cfg.cm_formulation, r_scaler=cfg.cvstem_r_scaler,
                max_lambda_reductions=cfg.max_lambda_reductions,
            )
            save_path = cache_path or with_formulation_suffix(
                Path(self.experiment_dir) / "checkpoints" / "cm_data.npz", cfg.cm_formulation
            )
            save_cm_dataset(save_path, dataset, **cache_kwargs)
        has_writer = getattr(self, "writer", None) is not None
        epochs = cfg.cmg_regress_epochs

        self.track_data("Loss / C2RL/cm_synthesis/feasibility_rate", dataset["feasibility_rate"])
        self.track_data("Loss / C2RL/cm_synthesis/residual_mean", dataset["residual_mean"])
        self.track_data("Loss / C2RL/cm_synthesis/residual_max", dataset["residual_max"])
        self.track_data("Loss / C2RL/cm_synthesis/lambda_reduced_rate", dataset.get("lambda_reduced_rate", 0.0))
        if has_writer:
            self.write_tracking_data(timestep=-epochs - 1, timesteps=timesteps)

        # ~100 wandb points regardless of epochs, final epoch always flushed —
        # same cadence as dynamics_pretrain.pretrain_dynamics.
        log_every = max(1, epochs // 100)

        def _on_epoch(epoch: int, train_mse: float, lr: float, val_mse: float) -> None:
            self.track_data("Loss / C2RL/cmg/regress_mse", train_mse)
            self.track_data("Loss / C2RL/cmg/regress_lr", lr)
            if not np.isnan(val_mse):
                self.track_data("Loss / C2RL/cmg/regress_val_mse", val_mse)
            if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
                self.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        bounded = getattr(self._ccm_gen, "bounded", False)
        info = regress_cmg(
            self._ccm_gen, dataset,
            w_lb=cfg.w_lb, x_dim=self._x_dim, bounded=bounded,
            epochs=epochs, lr=cfg.cmg_regress_lr,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            batch_size=cfg.cmg_regress_batch_size, device=self._device, tag="[C2RL]",
            on_epoch=_on_epoch,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
        )
        self.freeze_cmg()
        # Single post-loop point (distinct key from the per-epoch curves above)
        # for the epoch actually restored into ccm_gen — may differ from the
        # curves' last point when training ran past its best epoch without
        # early-stopping triggering (see regress_cmg's best-epoch restore).
        self.track_data("Loss / C2RL/cmg/regress_mse_best", info["final_loss"])
        if not np.isnan(info["final_val_loss"]):
            self.track_data("Loss / C2RL/cmg/regress_val_mse_best", info["final_val_loss"])
        if has_writer:
            self.write_tracking_data(timestep=-1, timesteps=timesteps)
        return {
            "feasibility_rate": dataset["feasibility_rate"],
            "residual_mean": dataset["residual_mean"],
            "residual_max": dataset["residual_max"],
            "lambda_reduced_rate": dataset.get("lambda_reduced_rate", 0.0),
            "regress_mse": info["final_loss"],
        }

    def save_dynamics(self, path: str) -> None:
        if self._neural_dynamics is not None:
            self._neural_dynamics.save(path)
            print(f"[C2RL] Saved NeuralDynamics → {path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLSkrlTrainer(Trainer):
    """skrl Trainer for C2RL — optional offline CMG synthesis (use_cmg), then
    single-policy RL against the Mahalanobis reward (frozen CMG or online CM)."""

    def _env_scalar_attr(self, *names):
        """See the identical helper on C3M's trainer — cross-backend env attr lookup."""
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
        """Forward the env's per-episode ``extras['log']`` (path_tracking_base's /
        classic env_base's ``Stability/*``) onto the outer agent's tracking_data,
        exactly as skrl's SequentialTrainer does for PPO/SAC. No-op if nothing
        finished this step."""
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
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        timesteps = self.cfg.timesteps

        agent.init(trainer_cfg=self.cfg)
        from .contraction_metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))
        agent.enable_training_mode(True)

        # Pretrain learned dynamics (if any) BEFORE any SDP solve, so the
        # contraction-metric SDP has meaningful f/B/∂f/∂x. No-op for analytical
        # dynamics (classic envs). Needed by BOTH modes — use_cmg=True solves the
        # SDP over the offline dataset next, use_cmg=False solves it online in
        # Phase B.
        from .dynamics_pretrain import pretrain_dynamics
        pretrain_dynamics(
            agent,
            epochs=getattr(agent._cfg, "dynamics_pretrain_epochs", 5),
            data_path=getattr(agent._cfg, "dynamics_pretrain_data_path", "") or None,
            timesteps=timesteps,
            memory_size=getattr(agent._cfg, "emp_dynamics_memory_size", None),
            num_controls_per_state=getattr(agent._cfg, "num_controls_per_state", None),
            tag="[C2RL]",
            val_frac=getattr(agent._cfg, "dynamics_val_frac", 0.1),
            early_stop_patience=getattr(agent._cfg, "dynamics_early_stop_patience", 10),
        )

        if agent._use_cmg:
            # ── use_cmg=True: offline synthesis — sample states, solve one SDP per
            # state (build_cm_dataset), regress the CMG network onto {x -> W*}
            # (regress_cmg), then freeze. Phase B then reads the static metric.
            # synthesize_cmg logs feasibility/residual/loss/LR to the writer itself. ──
            info = agent.synthesize_cmg(timesteps=timesteps)
            print(f"[C2RL] use_cmg=True — CMG regressed from SDP dataset "
                  f"(feasible {info['feasibility_rate']:.1%}, λ-reduced {info['lambda_reduced_rate']:.1%}, "
                  f"MSE {info['regress_mse']:.4g}) and frozen.")
        else:
            print("[C2RL] use_cmg=False — CM metric solved online per state "
                  f"({'per replayed transition, cached' if agent._base_algorithm == 'SAC' else 'per rollout batch'}).")

        # ── Phase B: rollout + train the deployed policy against the Mahalanobis
        # reward (frozen CMG or online CM). ─────────────────────────────────
        rl_agent = agent._rl_agent
        rl_agent.enable_training_mode(True)
        rollout_steps = agent._cfg.rollouts

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0
        # Coarse flush cadence for the INNER rl_agent — flushing every step/chunk
        # would collapse the 100-episode reward/timestep deques to a spiky curve.
        flush_interval = max(1, timesteps // 100)
        next_flush = flush_interval

        pbar = _tqdm.tqdm(total=timesteps, desc="C2RL training (Phase B)", file=sys.stdout)
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

                # Stored/logged reward: use_cmg=True re-uses the frozen CMG's
                # cheap forward pass to log the Mahalanobis reward the policy
                # actually optimizes; use_cmg=False stores a placeholder (the real
                # online-CM reward is solved later — PPO: over the rollout batch;
                # SAC: lazily per replayed transition — so solving it here would be
                # wasted). Either way the raw env reward is logged alongside.
                if agent._use_cmg:
                    stored_reward = agent._compute_reward(observations, actions).view_as(rewards)
                else:
                    stored_reward = torch.zeros_like(rewards)
                rl_agent.record_transition(
                    observations=observations, states=states, actions=actions,
                    rewards=stored_reward, next_observations=next_obs, next_states=next_states,
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
            # and flushes whatever Stability/* logs _forward_env_log collected.
            agent.post_interaction(timestep=global_step, timesteps=timesteps)
            if getattr(agent, "writer", None) is not None:
                agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        if getattr(rl_agent, "writer", None) is not None:
            rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        # Persist the learned dynamics for reuse/inspection (matches C3M).
        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            os.makedirs(os.path.dirname(dyn_path), exist_ok=True)
            agent.save_dynamics(dyn_path)

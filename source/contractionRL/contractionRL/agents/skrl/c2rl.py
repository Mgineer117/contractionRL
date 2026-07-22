"""C2RL — single-policy contraction-metric RL against a Neural Contraction
Metric (NCM) reward.

C2RLAgent trains ONE real skrl ``PPO``/``SAC`` policy (``base_algorithm="PPO"``
→ a skrl ``PPO`` sub-agent, ``base_algorithm="SAC"`` → a skrl ``SAC`` sub-agent)
against a Mahalanobis tracking reward

    ``-tracking_scaler·||e||²_M - control_scaler·||u - uref||²``,   e = x - xref

where the metric ``M(x) = W(x)⁻¹`` comes from one of two sources, selected by
``metric_source`` — this is the knob that decides HOW THE ENV COMPUTES THE
MAHALANOBIS REWARD, since whichever source is chosen is injected into the env
and called from its own ``get_rewards()``:

  * **``"cmg"`` (default)** — a CMG network ``W(x)`` synthesized in Phase A,
    before Phase B, then FROZEN (``freeze_cmg``) for the whole RL run:
    Tsukamoto's Neural Contraction Metric recipe. ``cmg_method`` selects how it
    is trained (below). The CMG network is mandatory (``models["cmg"]``); there
    is no ``use_cmg`` switch.
  * **``"online"``** — no Phase A at all. ``ncm_synthesis.OnlineCVSTEMMetric``
    solves the CV-STEM SDP per env per step, at the states the rollout actually
    visits, so every deployed ``M`` is a VERIFIED feasible metric rather than a
    regression of one. It costs ``num_envs`` SDP solves per env-step (~12 ms
    each with MOSEK on a 4-state system), which makes it a single-run research
    configuration rather than a sweep one, and an infeasible state ABORTS the
    run — a reward has no "open loop" fallback the way CV-STEM-LQR's control
    law does.

``cmg_method="ccm"`` forces ``metric_source="cmg"`` (with a printed note): that
pipeline has no per-state SDP, so a frozen network is the only metric it can
produce. The pair is therefore effectively three configurations, not four.

``cmg_method`` (formerly ``cm_formulation``) selects HOW the CMG network is
trained — the two are the only supported pipelines, both in ``ncm_synthesis.py``:

  * **``"cvstem"`` — CV-STEM regression.** Sample ``cmg_memory_size`` states,
    solve one convex feasibility SDP per state for the target metric ``W*(x)``
    (``ncm_synthesis.build_cm_dataset``, Tsukamoto CV-STEM-style LMI, keeps the
    control matrix via a Riccati term), then MSE-regress the CMG network onto
    the feasible ``{x -> W*}`` pairs (``regress_cmg``). No differentiable
    certificate loss — trained purely by regression onto the SDP solutions.
  * **``"ccm"`` — C1/C2 loss minimization.** Train the CMG network directly
    with Manchester's C1 (contraction) and C2 (killing) differentiable losses
    (``train_cmg_ccm``) over sampled states — no per-state SDP, no MSE
    regression, pure gradient descent on the pointwise certificate.

Both pipelines require the CMG to be a ``BoundedCCM_Generator``
(``constrain_eigenvalues=True`` — hard eigenvalue bounds baked into the
forward pass, not a soft penalty): ``C2RLAgent.__init__`` raises if the
supplied ``models["cmg"].ccm_gen`` isn't ``bounded``. ``ContractionRunner``
enforces this by always passing ``constrain_eigenvalues=True`` when building
C2RL's CMG model, regardless of yaml.

The ``cmg_memory_size`` states (both pipelines) are drawn uniformly either
from the classic env's analytic state space (``get_rollout``, unlimited
supply) or, when ``dynamics_pretrain_data_path`` points to an offline
``dynamics_data.npz``, from that same offline data (capped to its size, with a
warning if ``cmg_memory_size`` asks for more samples than are on disk).

Phase B — the single-policy rollout loop — is the same regardless of
``cmg_method`` OR ``metric_source`` (see ``C2RLSkrlTrainer``): all three hand
off one object with the same ``W, _ = metric(x)`` call contract, either a frozen
``ccm_gen`` or an ``OnlineCVSTEMMetric``. The reward is NOT overwritten anywhere
on the agent side: ``C2RLSkrlTrainer.train`` injects that object into the env
(``_inject_ccm`` → the env's ``set_ccm``), and from then on the env's own
``get_rewards()`` returns the Mahalanobis reward natively. So the reward that
reaches ``record_transition`` — and therefore PPO's GAE and SAC's replayed
critic target alike — is already the right one, with no per-algorithm reward
plumbing to keep in sync. That injection is a hard requirement, not a best
effort: ``_inject_ccm`` raises if no env accepted the CMG, since silently
missing it means training on the plain baseline reward instead.

Normalization: the CMG metric / CM SDP and the Mahalanobis reward always use
RAW observations — ``M(x)`` and the tracking error ``e = x - xref`` are defined
in raw physical coordinates, and per-dimension normalization would scale ``x``
and ``xref`` independently, distorting ``e``. ``uref`` and any ``angle_idx``
columns are likewise excluded from observation normalization (see
``rl_glue.make_base_rl_cfg`` for the full rationale, shared with the deployed
PPO/SAC policy built here). ``use_state_norm`` defaults to False in every
shipped config.

Learned dynamics (Isaac / ``use_empirical_dynamics=True``): a ``NeuralDynamics``
model provides ``f``/``B``/``B_null`` and ``∂f/∂x``. It is pretrained once
before Phase A, which under ``metric_source="cmg"`` is the only consumer —
the SDP dataset (``"cvstem"``) or the C1/C2 gradient computation (``"ccm"``)
both need ``f``/``B``/``∂f/∂x`` at synthesis time; Phase B never touches it
again (the CMG is already frozen). Under ``metric_source="online"`` it is the
opposite: the dynamics are queried EVERY step, since each solve needs
``A = ∂f/∂x`` and ``B`` at the current state. Classic envs use their analytical
``get_f_and_B`` and skip the pretraining.
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
from .rl_glue import filter_cfg_fields, make_base_rl_cfg


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
    use_reward_norm: bool = False  # non-biasing running-std reward normalizer (r/std) — see rl_glue.make_base_rl_cfg
    rewards_shaper_scale: float = 1.0  # yaml convenience for PPO_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    # CAPS action-smoothness regularization on the POLICY LOSS (both 0 = off).
    # Not a reward term: it leaves the MDP, the observation and the dynamics
    # untouched, so the offline CV-STEM synthesis stays valid. See
    # agent_patches.patch_caps_regularizer.
    caps_temporal_scale: float = 0.0  # weight on ||pi(s_t) - pi(s_{t+1})||^2
    caps_spatial_scale: float = 0.0   # weight on ||pi(s) - pi(s_bar)||^2, s_bar ~ N(s, sigma^2)
    caps_spatial_std: float = 0.05    # sigma, in RAW observation units (see patch_caps_regularizer)
    # Set by ContractionRunner from the yaml `memory:` block's memory_size, NOT
    # read from `agent:` directly; declared purely so filter_cfg_fields()
    # recognizes it instead of warning.
    memory_size: int = -1
    # Deployed policy's discount factor — a single policy trained against the
    # Mahalanobis reward, so there is no con/opt duality here.
    discount_factor: float = 0.99
    # ── Metric source: WHERE the env's Mahalanobis reward gets M(x) ──────── #
    # "cmg" (default): a CMG network synthesized in Phase A (cmg_method selects
    # how) and frozen for the whole run. "online": no Phase A at all — solve the
    # CV-STEM SDP per env per step at the visited states
    # (ncm_synthesis.OnlineCVSTEMMetric), so every deployed M is a verified
    # feasible metric instead of a regression of one, at the cost of num_envs
    # SDP solves per step. cmg_method="ccm" FORCES "cmg" (there is no per-state
    # SDP to solve online under the C1/C2 pipeline).
    metric_source: str = "cmg"
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2  # contraction rate λ — used by both cmg_method's synthesis loss
    # ── SDP contraction metric ("cvstem" cmg_method only) — see ncm_synthesis.py ── #
    cm_eps: float = 1e-2   # strict-definiteness margin on the contraction LMI (both methods)
    cm_solver: str = "SCS"  # cvxpy SDP solver ("cvstem" only)
    # "ccm" — C1/C2 loss minimization (train_cmg_ccm): Manchester-style,
    # eliminates B via the annihilator, existence-only certificate, no SDP, pure
    # gradient descent on the pointwise LMI. "cvstem" (default) — CV-STEM
    # regression (build_cm_dataset + regress_cmg): solves a per-state SDP that
    # keeps B via a Riccati BR⁻¹Bᵀ term, then MSE-regresses the CMG onto the
    # solutions. See ncm_synthesis.py module docstring for the LMIs and module
    # docstring above for the two pipelines.
    cmg_method: str = "cvstem"
    # R = cvstem_r_scaler·I in the BR⁻¹Bᵀ Riccati term (mirrors sdlqr.py's
    # R_scaler); "cvstem" method only. See ncm_synthesis.solve_cm_metric — control
    # enters the LMI only through this penalty, not a bounded control box.
    cvstem_r_scaler: float = 1.0
    # Weights of the CV-STEM objective J = cm_chi_weight·χ + cm_nu_weight·ν, which
    # solve_cm_metric ALWAYS minimizes (Tsukamoto's classncm.cvstem0). χ and ν are
    # the metric's condition number and scale, and they are decision variables:
    # W̄ ⪰ I, W̄ ⪯ χI, deployed W = W̄/ν. "cvstem" method only.
    cm_chi_weight: float | None = None  # None → 1/lbd, mirroring Tsukamoto's chi/alp
    cm_nu_weight: float = 1.0           # his d2_over
    # If > 0, include Tsukamoto's Ẇ ≈ (W̄ - I)/dt proxy for the material derivative
    # (classncm.cvstem0 puts the integration step here). 0 = omit it, which is what
    # the pointwise-per-state design otherwise forces (no neighbouring sample to
    # difference against — see ncm_synthesis.py's module docstring). "cvstem" method
    # only; superseded by cm_wdot_trajectory when that's on.
    cm_wdot_dt: float = 0.0
    # Real Ẇ from OFFLINE REFERENCE TRAJECTORIES ("cvstem" method only): instead of
    # dropping Ẇ or using the static cm_wdot_dt proxy above, sample states as
    # trajectory-ordered chunks from dynamics_pretrain_data_path (which must be
    # set — raises otherwise) and difference each state's solved normalized W̄
    # against the ACTUAL PREVIOUS state's along that same reference trajectory —
    # Ẇ ≈ (W̄_t − W̄_{t−1})/cm_temporal_dt, the real material derivative rather
    # than an approximation (see ncm_synthesis.build_cm_dataset's
    # traj_x/traj_lengths/temporal_dt and dynamics_pretrain.load_offline_trajectories).
    # Incompatible with cmg_random_ratio>0 (mixing in i.i.d. random states would
    # break trajectory continuity — ignored when this is on) and with
    # cmg_method="ccm" (train_cmg_ccm has no per-state SDP to add Ẇ to; raises).
    cm_wdot_trajectory: bool = False
    # Integration step between consecutive states in the offline trajectory data —
    # NOT auto-derived from the env (dynamics_data.npz doesn't record it); set it
    # to the same dt used to generate that file (scripts/skrl/train.py's
    # _generate_ref_trajs: env_cfg.sim.dt * env_cfg.decimation). Only read when
    # cm_wdot_trajectory=True.
    cm_temporal_dt: float = 0.05
    # On SDP infeasibility at a state, retry that state ALONE with λ halved,
    # up to this many times, before giving up on it (0 = old behavior, drop
    # immediately). See ncm_synthesis._solve_cm_metric_with_backoff. "cvstem" method only.
    max_lambda_reductions: int = 5
    # Guards build_cm_dataset against silently regressing the CMG onto a small,
    # likely-biased subset of states — raises before regression if the SDP's
    # feasible fraction falls below this (0.0 = old behavior, only guards
    # against 0% feasible; see ncm_synthesis.build_cm_dataset). "cvstem" method only.
    min_feasibility_rate: float = 0.0
    # Cache path for the synthesized {x, W} CM dataset (build_cm_dataset's
    # expensive per-state SDP solve) — see synthesize_cmg. Loaded instead of
    # re-solving when it exists and matches lbd/w_lb/w_ub/cm_eps/cm_solver/
    # cmg_memory_size exactly; written after a fresh solve otherwise. Defaults
    # to a `cm_data.npz` next to dynamics_pretrain_data_path when unset (Isaac
    # envs); classic envs with no data_path need this set explicitly to get
    # caching at all (there's no offline dynamics file to derive a path from).
    # "cvstem" method only.
    cm_data_path: str = ""
    # ── Offline CMG synthesis (Phase A, always runs before Phase B) ─────────── #
    # Sample cmg_memory_size states — uniformly from the classic env's analytic
    # state space (get_rollout) or, when dynamics_pretrain_data_path is set,
    # uniformly from that offline dynamics_data.npz (capped + warned if
    # cmg_memory_size exceeds the data on disk; see synthesize_cmg). "cvstem":
    # solve one SDP per state (solve_cm_metric, reusing lbd/w_lb/w_ub/cm_eps/
    # cm_solver above), then MSE-regress the CMG network onto {x -> W*} for
    # cmg_regress_epochs (build_cm_dataset / regress_cmg). "ccm": train the CMG
    # directly with C1/C2 losses for cmg_regress_epochs, no SDP (train_cmg_ccm).
    # Either way the CMG is frozen (freeze_cmg) before Phase B.
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 1000
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    # Held out from cmg_memory_size as a validation split never regressed on;
    # regress_cmg stops once its MSE hasn't improved for cmg_early_stop_patience
    # consecutive epochs, restoring the best-val-epoch CMG weights instead of
    # whatever cmg_regress_epochs happens to land on (see ncm_synthesis.regress_cmg
    # / math_utils.EarlyStopper). <=0 disables both (always regress the full budget).
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Fraction (0..1) of the CMG-dataset states drawn from the BROAD/off-reference
    # distribution (states an early chaotic policy actually visits) rather than the
    # reference-trajectory tube — the rest are reference states. 0 = old behavior
    # (all reference, or all of the offline pool). Random states come from the
    # offline dynamics-pretrain pool if configured, else get_rollout("dynamics")
    # (uniform state-space coverage). See ncm_synthesis._sample_cm_states.
    cmg_random_ratio: float = 0.0
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Feeds Phase A's CMG
    # synthesis (SDP dataset for "cvstem", C1/C2 gradient computation for "ccm").
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
    # Not a real SAC_CFG field, and NOT the replay buffer size (that's
    # memory_size): it's how many env steps C2RLSkrlTrainer takes per outer
    # iteration, i.e. the cadence of the outer agent's checkpoint/flush pass.
    # SAC still updates every single step within that chunk.
    rollouts: int = 16
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
    use_reward_norm: bool = False  # non-biasing running-std reward normalizer (r/std) — see rl_glue.make_base_rl_cfg
    rewards_shaper_scale: float = 1.0  # yaml convenience for SAC_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    # CAPS action-smoothness regularization — see C2RLPPOCfg / patch_caps_regularizer.
    caps_temporal_scale: float = 0.0
    caps_spatial_scale: float = 0.0
    caps_spatial_std: float = 0.05
    memory_size: int = -1
    discount_factor: float = 0.99
    # ── Metric source — "cmg" frozen CMG network (cmg_method selects how it's
    # trained) | "online" per-step CV-STEM SDP. See C2RLPPOCfg.metric_source. ── #
    metric_source: str = "cmg"
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2
    # ── SDP contraction metric ("cvstem" method only) — see ncm_synthesis.py ── #
    cm_eps: float = 1e-2
    cm_solver: str = "SCS"
    cmg_method: str = "cvstem"  # "ccm" (C1/C2 minimization) | "cvstem" (SDP regression, default) — see module docstring
    cvstem_r_scaler: float = 1.0
    # Weights of the CV-STEM objective J (always minimized) — see
    # C2RLPPOCfg.cm_chi_weight above and ncm_synthesis.solve_cm_metric.
    cm_chi_weight: float | None = None
    cm_nu_weight: float = 1.0
    cm_wdot_dt: float = 0.0  # superseded by cm_wdot_trajectory when that's on
    # Real Ẇ from offline reference trajectories — see C2RLPPOCfg.cm_wdot_trajectory
    # / cm_temporal_dt above.
    cm_wdot_trajectory: bool = False
    cm_temporal_dt: float = 0.05
    max_lambda_reductions: int = 5  # see ncm_synthesis._solve_cm_metric_with_backoff
    min_feasibility_rate: float = 0.0
    cm_data_path: str = ""
    # ── Offline CMG synthesis (Phase A, always runs before Phase B) ─────────── #
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 1000
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Random/off-reference state fraction for the CMG dataset — see
    # C2RLPPOCfg.cmg_random_ratio above / ncm_synthesis._sample_cm_states.
    cmg_random_ratio: float = 0.0
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
    timesteps: int = 300000  # deployed-policy (RL) env steps; the offline CMG-synthesis
                             # phase (Phase A) runs once before this loop


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
      ``"cmg"``      — REQUIRED. MetricModel (``BoundedCCM_Generator``, i.e.
        ``constrain_eigenvalues=True``) synthesized offline (see synthesize_cmg)
        and frozen before Phase B; read for the Mahalanobis reward.

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
        self._get_rollout = get_rollout

        # ── Metric source setup ──────────────────────────────────────────── #
        # This agent always owns the (optional) learned dynamics directly — Phase
        # A's CMG synthesis needs f/B/∂f/∂x (SDP dataset for "cvstem", C1/C2
        # gradient computation for "ccm"). The CMG network (models["cmg"]) is
        # fit OFFLINE (see C2RLSkrlTrainer.train / synthesize_cmg) and frozen
        # before Phase B — see module docstring.
        self._setup_dynamics(parsed_cfg, models, get_f_and_B, x_dim, u_dim, device)
        if "cmg" not in models:
            raise ValueError(
                "[C2RL] models['cmg'] is required — C2RL always synthesizes a CMG "
                "network offline before Phase B (see module docstring)."
            )
        self._ccm_gen = models["cmg"].ccm_gen
        if not getattr(self._ccm_gen, "bounded", False):
            raise ValueError(
                "[C2RL] models['cmg'] must be a BoundedCCM_Generator "
                "(constrain_eigenvalues=True) — C2RL always hard-bounds the CMG's "
                "eigenvalues, regardless of cmg_method. Set "
                "models.cmg.network.constrain_eigenvalues: true in the yaml, or "
                "let ContractionRunner build it (it forces this)."
            )
        self._metric_source = str(getattr(parsed_cfg, "metric_source", "cmg")).lower()
        if self._metric_source not in ("cmg", "online"):
            raise ValueError(
                f"[C2RL] metric_source must be 'cmg' (frozen network from Phase A) or "
                f"'online' (per-step CV-STEM SDP), got {parsed_cfg.metric_source!r}."
            )
        if self._metric_source == "online" and parsed_cfg.cmg_method == "ccm":
            # Not an error: "ccm" simply has no per-state SDP to solve online (it
            # trains the CMG by C1/C2 gradient descent), so the CMG network IS the
            # only metric that pipeline can produce. Downgrade and say so, rather
            # than making every ccm sweep restate a metric_source it cannot vary.
            print(
                "[C2RL] metric_source='online' is not available under cmg_method='ccm' "
                "(no per-state SDP to solve) — falling back to metric_source='cmg'.",
                flush=True,
            )
            self._metric_source = "cmg"
        if bool(getattr(parsed_cfg, "cm_wdot_trajectory", False)):
            if parsed_cfg.cmg_method != "cvstem":
                raise ValueError(
                    "[C2RL] cm_wdot_trajectory=True needs cmg_method='cvstem' — "
                    "'ccm' (train_cmg_ccm) has no per-state SDP to add a Ẇ term to."
                )
            if not getattr(parsed_cfg, "dynamics_pretrain_data_path", ""):
                raise ValueError(
                    "[C2RL] cm_wdot_trajectory=True needs dynamics_pretrain_data_path "
                    "set to a trajectory-structured dynamics_data.npz (see "
                    "dynamics_pretrain.load_offline_trajectories) — there is no other "
                    "source of REAL trajectory order to difference Ẇ against."
                )

        # ── Phase B: a real skrl PPO/SAC agent for the deployed policy ───── #
        # PPO's memory holds exactly one on-policy rollout chunk. SAC's is a
        # persistent replay buffer, sized per-parallel-env (memory_size rows ×
        # num_envs) so Isaac Sim's 1000+ envs don't multiply skrl's usual ~1M
        # default up to an OOM; memory_size=-1 falls back to a modest default.
        _DEFAULT_SAC_BUFFER_ROWS = 10000
        if self._base_algorithm == "PPO":
            mem_size = parsed_cfg.rollouts
        else:
            mem_size = parsed_cfg.memory_size
            if mem_size == -1:
                mem_size = _DEFAULT_SAC_BUFFER_ROWS
        memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        self._memory = memory

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
            patch_caps_regularizer,
            patch_kl_logging,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        patch_kl_logging(self._rl_agent)
        patch_sac_entropy_clamp(self._rl_agent)
        # Applied to the INNER PPO/SAC sub-agent: C2RL's outer agent has no
        # .policy/.scaler of its own for the patch to hook.
        patch_caps_regularizer(
            self._rl_agent,
            temporal_scale=parsed_cfg.caps_temporal_scale,
            spatial_scale=parsed_cfg.caps_spatial_scale,
            spatial_std=parsed_cfg.caps_spatial_std,
        )
        _std_dev_annealing_kwargs = parsed_cfg.std_dev_annealing_kwargs
        # Always anneal for PPO, regardless of the policy's backbone — SAC keeps
        # this off since it learns log_std via its own automatic entropy tuning
        # (see SquashedCLActorModel's docstring in models.py).
        patch_ppo_std_annealing(self._rl_agent, self._base_algorithm == "PPO", _std_dev_annealing_kwargs)

        self._rl_agent.init()

        checkpoint_extra = (
            {"value": models["value"]} if self._base_algorithm == "PPO" else
            {"critic_1": models["critic_1"], "critic_2": models["critic_2"]}
        )
        self.checkpoint_modules.update({
            "policy": models["policy"],
            **checkpoint_extra,
        })
        self.checkpoint_modules["cmg"] = models["cmg"]
        if self._neural_dynamics is not None:
            self.checkpoint_modules["dynamics"] = self._neural_dynamics

    # ── Setup helpers ───────────────────────────────────────────────────── #

    def _setup_dynamics(self, cfg, models, get_f_and_B, x_dim, u_dim, device) -> None:
        """Own the (optional) learned dynamics directly (feeds Phase A's CMG synthesis).

        The CMG synthesis needs ``f``/``B``/``B_null`` and ``∂f/∂x`` at every
        state it uses (SDP dataset for "cvstem", C1/C2 gradient computation for
        "ccm"). Under analytical dynamics that comes from the env's exact
        ``get_f_and_B``; otherwise a NeuralDynamics model (pretrained before
        training by the trainer's ``pretrain_dynamics``) provides it. This
        mirrors C3M's dynamics interface expected by ``dynamics_pretrain``.
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
        """Evaluation entry point (play.py / train_utils' rollouts call this).

        Goes to the policy MODEL rather than delegating to ``self._rl_agent.act``
        so a nonzero ``random_timesteps`` can't turn an eval rollout — which
        always passes ``timestep=0`` — into uniform random actions. The
        observation preprocessor is applied by hand for the same reason it is
        applied inside the sub-agent's own act(): skipping it would evaluate the
        policy on a different input scale than it was trained on whenever
        ``use_state_norm`` is enabled (it is off in every shipped config, so
        this is identity today).
        """
        with torch.no_grad():
            inputs = {"observations": self._rl_agent._observation_preprocessor(observations)}
            actions, outputs = self._rl_agent.models["policy"].act(inputs, role="policy")
        return actions, outputs

    # act/pre_interaction/post_interaction/update are all abstract on skrl's
    # Agent, so they must be defined even where the base behavior is all we
    # want. record_transition is NOT abstract and we add nothing to it, so it's
    # left inherited. C2RLSkrlTrainer drives the actual Phase B update itself
    # (see update_policy), so update() here stays a no-op.
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass

    # ── Phase B update ──────────────────────────────────────────────────── #

    def update_policy(self, *, timestep: int, timesteps: int) -> None:
        """Drive the inner RL agent's update, once per trainer rollout chunk.

        PPO consumes a whole on-policy chunk at a time, so its update belongs
        here. SAC updates every step, driven by its own ``post_interaction`` —
        nothing to do. Neither branch touches rewards: the env already returns
        the Mahalanobis reward (see the module docstring).
        """
        if self._base_algorithm == "PPO":
            self._rl_agent.update(timestep=timestep, timesteps=timesteps)

    def _train_dynamics(self, data: dict) -> float:
        """MSE training of NeuralDynamics on (x, u, x_dot) data (same as C3M).

        Called by the trainer's ``pretrain_dynamics`` when learning dynamics —
        Phase A's CMG synthesis needs f/B/∂f/∂x before it runs.
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
        """Freeze the synthesized CMG before Phase B."""
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

    def _log_cmg_condition_numbers(self, x_np, *, max_states: int = 1000) -> None:
        """Post-pretraining diagnostic: condition number κ(x) = λmax/λmin of the
        frozen CMG's bounded ``W(x)`` over (a ≤``max_states`` subsample of) the
        states it was trained on. κ(W) == κ(M=W⁻¹), so this is also the
        anisotropy of the Mahalanobis reward the policy trains against; the
        w_lb/w_ub box bounds it by w_ub/w_lb. Tracks mean and 95% quantile
        (flushed by the caller's timestep=-1 write)."""
        from .math_utils import bound_W
        x_np = np.asarray(x_np, dtype=np.float32)[:, :self._x_dim]
        if x_np.shape[0] > max_states:
            idx = np.random.choice(x_np.shape[0], size=max_states, replace=False)
            x_np = x_np[idx]
        x = torch.as_tensor(x_np, device=self._device)
        with torch.no_grad():
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, self._cfg.w_lb, self._x_dim,
                        getattr(self._ccm_gen, "bounded", False))
            eig = torch.linalg.eigvalsh(W)  # ascending, (n, x_dim)
            cond = eig[:, -1] / eig[:, 0].clamp(min=1e-12)
        cond_mean = cond.mean().item()
        cond_q95 = torch.quantile(cond, 0.95).item()
        print(f"[C2RL] CMG condition number over {x.shape[0]} training states: "
              f"mean={cond_mean:.4g}, 95%={cond_q95:.4g} "
              f"(bound w_ub/w_lb={self._cfg.w_ub / self._cfg.w_lb:.4g})")
        self.track_data("Loss / C2RL/cmg/cond_mean", cond_mean)
        self.track_data("Loss / C2RL/cmg/cond_q95", cond_q95)

    def build_online_metric(self):
        """The ``metric_source="online"`` stand-in for the frozen CMG network.

        Replaces Phase A entirely: nothing is sampled, solved offline, cached or
        regressed — the SDP is solved at the states the rollout actually visits,
        inside the env's own reward (see ``OnlineCVSTEMMetric``). Returns the
        object the trainer injects via ``set_ccm`` in place of the CMG.
        """
        from .ncm_synthesis import OnlineCVSTEMMetric
        cfg = self._cfg
        return OnlineCVSTEMMetric(
            self._get_f_and_B,
            x_dim=self._x_dim,
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            solver=cfg.cm_solver, r_scaler=cfg.cvstem_r_scaler,
            max_lambda_reductions=cfg.max_lambda_reductions,
            chi_weight=cfg.cm_chi_weight, nu_weight=cfg.cm_nu_weight,
        )

    def synthesize_cmg(self, *, timesteps: int = 0) -> dict:
        """Offline CMG synthesis (Phase A, always runs before Phase B) —
        dispatches to one of two pipelines depending on ``cmg_method``:

        * **``"cvstem"``** (CV-STEM): convex optimization.  Sample states, solve
          one pointwise SDP per state (``build_cm_dataset``), MSE-regress the CMG
          network onto the feasible ``{x → W*}`` targets (``regress_cmg``), then
          freeze.  The SDP results are cached to disk.

        * **``"ccm"``** (default — C1/C2 loss minimization): neural-network
          training.  Train the CMG network end-to-end with C1 (contraction) and
          C2 (killing) losses (``train_cmg_ccm``) over uniformly sampled states
          — no per-state SDP, no MSE regression.  C2 makes the metric
          ``u``-independent by construction, so no u-box vertex enumeration is
          needed.

        Called once by the trainer before Phase B — needs the dynamics already
        pretrained so the SDP / gradient computation has meaningful
        ``f``/``B``/``∂f/∂x``.

        Logs per-epoch loss/LR curves at negative timesteps so they precede
        Phase B on the ``global_step`` x-axis — same convention
        ``dynamics_pretrain.py`` uses for the NeuralDynamics fit.
        """
        cfg = self._cfg
        if cfg.cmg_method == "ccm":
            return self._synthesize_cmg_ccm(timesteps=timesteps)
        else:
            return self._synthesize_cmg_cvstem(timesteps=timesteps)

    def _synthesize_cmg_ccm(self, *, timesteps: int = 0) -> dict:
        """CCM path: train the CMG network directly with C1+C2 losses."""
        from .ncm_synthesis import train_cmg_ccm
        cfg = self._cfg
        has_writer = getattr(self, "writer", None) is not None
        epochs = cfg.cmg_regress_epochs

        # ~100 wandb points regardless of epochs, final epoch always flushed.
        log_every = max(1, epochs // 100)

        def _on_epoch(epoch: int, train_loss: float, lr: float, val_loss: float) -> None:
            self.track_data("Loss / C2RL/cmg/c1c2_loss", train_loss)
            self.track_data("Loss / C2RL/cmg/regress_lr", lr)
            if not np.isnan(val_loss):
                self.track_data("Loss / C2RL/cmg/c1c2_val_loss", val_loss)
            if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
                self.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        info = train_cmg_ccm(
            self._ccm_gen, self._get_f_and_B, self._get_rollout,
            x_dim=self._x_dim, u_dim=self._u_dim,
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            epochs=epochs, lr=cfg.cmg_regress_lr, batch_size=cfg.cmg_regress_batch_size,
            num_samples=cfg.cmg_memory_size,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            device=self._device, tag="[C2RL]",
            on_epoch=_on_epoch,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
            x_samples=self._sample_cmg_x(),
            random_ratio=getattr(cfg, "cmg_random_ratio", 0.0),
        )
        self.freeze_cmg()
        self.track_data("Loss / C2RL/cmg/c1c2_loss_best", info["final_loss"])
        if not np.isnan(info["final_val_loss"]):
            self.track_data("Loss / C2RL/cmg/c1c2_val_loss_best", info["final_val_loss"])
        self._log_cmg_condition_numbers(info["x"])
        if has_writer:
            self.write_tracking_data(timestep=-1, timesteps=timesteps)
        return {
            "feasibility_rate": 1.0,  # no SDP, no infeasibility concept
            "residual_mean": float("nan"),
            "residual_max": float("nan"),
            "lambda_reduced_rate": 0.0,
            "regress_mse": info["final_loss"],
        }

    def _synthesize_cmg_cvstem(self, *, timesteps: int = 0) -> dict:
        """CV-STEM path: convex SDP per state + MSE regression (original pipeline)."""
        from .ncm_synthesis import (
            build_cm_dataset, cm_dataset_cache_path, cm_dataset_filename,
            load_cached_cm_dataset, regress_cmg, save_cm_dataset,
        )
        cfg = self._cfg
        data_path = getattr(cfg, "dynamics_pretrain_data_path", "") or None
        explicit_cache_path = getattr(cfg, "cm_data_path", "") or None
        # cm_wdot_trajectory (see C2RLPPOCfg's docstring): real Ẇ from OFFLINE
        # REFERENCE TRAJECTORIES instead of dropping it or Tsukamoto's static
        # cm_wdot_dt proxy — validated at __init__ time (requires
        # dynamics_pretrain_data_path); random_ratio is meaningless here (the
        # whole point is NOT mixing in i.i.d. states) so it's forced to 0 for
        # both the cache key and the (unused, in this mode) x_samples pool.
        wdot_trajectory = bool(getattr(cfg, "cm_wdot_trajectory", False))
        temporal_dt = cfg.cm_temporal_dt if wdot_trajectory else 0.0
        random_ratio = 0.0 if wdot_trajectory else getattr(cfg, "cmg_random_ratio", 0.0)
        traj_x = traj_lengths = None
        if wdot_trajectory:
            from .dynamics_pretrain import load_offline_trajectories
            traj_data = load_offline_trajectories(data_path, tag="[C2RL]")
            traj_x, traj_lengths = traj_data["x"], traj_data["lengths"]
        if explicit_cache_path:
            # An explicit cm_data_path is treated as a BASE name: the swept SDP
            # knobs are appended (cm_dataset_filename) so different lbd/w_lb/w_ub
            # runs never clobber or wrongly reuse each other's cache.
            base = Path(explicit_cache_path)
            cache_path = base.with_name(
                cm_dataset_filename(cfg.lbd, cfg.w_lb, cfg.w_ub, cfg.cvstem_r_scaler, stem=base.stem))
        elif data_path:
            cache_path = cm_dataset_cache_path(
                data_path, lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, r_scaler=cfg.cvstem_r_scaler)
        else:
            cache_path = None
        cache_kwargs = dict(
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            solver=cfg.cm_solver, num_samples=cfg.cmg_memory_size, tag="[C2RL]",
            r_scaler=cfg.cvstem_r_scaler,
            chi_weight=cfg.cm_chi_weight,
            nu_weight=cfg.cm_nu_weight, wdot_dt=cfg.cm_wdot_dt,
            random_ratio=random_ratio,
            wdot_trajectory=wdot_trajectory, temporal_dt=temporal_dt,
        )
        dataset = load_cached_cm_dataset(cache_path, **cache_kwargs) if cache_path else None
        if dataset is None:
            dataset = build_cm_dataset(
                self._get_rollout, self._get_f_and_B,
                x_dim=self._x_dim,
                lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
                num_samples=cfg.cmg_memory_size, solver=cfg.cm_solver,
                device=self._device, tag="[C2RL]",
                x_samples=self._sample_cmg_x() if not wdot_trajectory else None,
                random_ratio=random_ratio,
                min_feasibility_rate=cfg.min_feasibility_rate,
                r_scaler=cfg.cvstem_r_scaler,
                max_lambda_reductions=cfg.max_lambda_reductions,
                chi_weight=cfg.cm_chi_weight,
                nu_weight=cfg.cm_nu_weight, wdot_dt=cfg.cm_wdot_dt,
                traj_x=traj_x, traj_lengths=traj_lengths, temporal_dt=temporal_dt,
            )
            save_path = cache_path or (
                Path(self.experiment_dir) / "checkpoints"
                / cm_dataset_filename(cfg.lbd, cfg.w_lb, cfg.w_ub, cfg.cvstem_r_scaler))
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
        self._log_cmg_condition_numbers(dataset["x"])
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
    """skrl Trainer for C2RL — offline CMG synthesis (Phase A), then
    single-policy RL against the frozen CMG's Mahalanobis reward (Phase B)."""

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

    @staticmethod
    def _inject_ccm(env, agent, metric=None) -> None:
        """Hand a frozen copy of the synthesized CMG to the env(s) so their own
        ``get_rewards()`` returns the Mahalanobis reward natively (see the
        module docstring — this is the ONLY reward path).

        Raises if no env accepted it: a missing ``set_ccm`` means the run would
        silently train on the plain baseline reward, which is exactly the
        classic-vs-isaaclab parity bug this guard exists to prevent. A DEEPCOPY,
        not the agent's own module, so the env holds a stable frozen metric even
        if the agent's ``_ccm_gen`` were ever touched again.

        ``metric`` overrides what is injected (``metric_source="online"`` passes
        an ``OnlineCVSTEMMetric``). It is injected AS-IS: it is stateless apart
        from counters, and deep-copying it would drag along whatever
        ``get_f_and_B`` is bound to — the env itself, under analytical dynamics.
        """
        import copy

        # Peel wrappers down to the concrete env: first the ._env chain
        # (BatchedGymnasiumWrapper etc.), then gymnasium's .unwrapped.
        inner = env
        while hasattr(inner, "_env") and getattr(inner, "_env") is not inner:
            inner = inner._env
        while hasattr(inner, "unwrapped") and getattr(inner, "unwrapped") is not inner:
            inner = inner.unwrapped

        device = agent.device
        ccm = metric if metric is not None else copy.deepcopy(agent._ccm_gen).to(device)
        # A classic SyncVectorEnv exposes per-env instances via .envs; an Isaac
        # env is a single batched env that takes set_ccm directly.
        targets = list(getattr(inner, "envs", [])) or [inner]
        injected = 0
        for e in targets:
            e = e.unwrapped if hasattr(e, "unwrapped") else e
            if hasattr(e, "set_ccm"):
                e.set_ccm(ccm, agent._cfg.w_lb, device)
                injected += 1
        if injected == 0:
            raise RuntimeError(
                "[C2RL] no env accepted set_ccm — Phase B would train on the plain "
                f"baseline reward instead of the Mahalanobis one. Env type: {type(inner).__name__}."
            )

    def train(self) -> None:
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        timesteps = self.cfg.timesteps

        agent.init(trainer_cfg=self.cfg)
        from .contraction_metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))
        agent.enable_training_mode(True)

        # Pretrain learned dynamics (if any) BEFORE Phase A's CMG synthesis, so it
        # has meaningful f/B/∂f/∂x (SDP dataset for "cvstem", C1/C2 gradient
        # computation for "ccm"). No-op for analytical dynamics (classic envs).
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

        # ── Phase A: offline CMG synthesis — "cvstem" solves one SDP per sampled
        # state (build_cm_dataset) and MSE-regresses the CMG onto {x -> W*}
        # (regress_cmg); "ccm" trains the CMG directly with C1/C2 losses
        # (train_cmg_ccm). Either way the CMG is frozen before Phase B reads its
        # static metric. synthesize_cmg logs feasibility/residual/loss/LR itself. ──
        # metric_source="online" skips Phase A outright — there is nothing to
        # synthesize or freeze, the SDP is solved per step inside the reward.
        online_metric = None
        if agent._metric_source == "online":
            online_metric = agent.build_online_metric()
            print("[C2RL] Phase A SKIPPED (metric_source=online) — the CV-STEM SDP is "
                  f"solved per env per step at lbd={agent._cfg.lbd}, "
                  f"solver={agent._cfg.cm_solver}; an infeasible state aborts the run.")
        else:
            info = agent.synthesize_cmg(timesteps=timesteps)
            print(f"[C2RL] Phase A ({agent._cfg.cmg_method}) — CMG synthesized "
                  f"(feasible {info['feasibility_rate']:.1%}, λ-reduced {info['lambda_reduced_rate']:.1%}, "
                  f"loss {info['regress_mse']:.4g}) and frozen.")

        # ── Phase B: rollout + train the deployed policy against the Mahalanobis
        # reward computed from the frozen CMG. ─────────────────────────────
        rl_agent = agent._rl_agent
        rl_agent.enable_training_mode(True)
        rollout_steps = agent._cfg.rollouts

        self._inject_ccm(env, agent, metric=online_metric)

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
            steps_to_take = min(rollout_steps, timesteps - global_step)
            for _ in range(steps_to_take):
                rl_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = rl_agent.act(observations, states, timestep=global_step, timesteps=timesteps)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None

                # The env's get_rewards() already computes the Mahalanobis reward
                # via the injected frozen CCM — use it directly.
                rl_agent.record_transition(
                    observations=observations, states=states, actions=actions,
                    rewards=rewards, next_observations=next_obs, next_states=next_states,
                    terminated=terminated, truncated=truncated, infos=infos,
                    timestep=global_step, timesteps=timesteps,
                )

                self._forward_env_log(agent, infos)
                observations = next_obs
                states = next_states

                if agent._base_algorithm == "SAC":
                    agent.update_policy(timestep=global_step, timesteps=timesteps)
                    rl_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                    if global_step % flush_interval == 0 and getattr(rl_agent, "writer", None) is not None:
                        rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

                global_step += 1
                pbar.update(1)

            # Chunk-based update for PPO
            if agent._base_algorithm == "PPO":
                agent.update_policy(timestep=global_step, timesteps=timesteps)
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

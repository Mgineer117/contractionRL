"""C2RL — Two-phase contraction-metric synthesis, native skrl Agent.

C2RLAgent trains TWO independent policies against the SAME environment, on
top of a chosen base algorithm (``base_algorithm="PPO"`` → two skrl ``PPO``
sub-agents, ``base_algorithm="SAC"`` → two skrl ``SAC`` sub-agents):

  * ``con_policy`` ("contracting", ``gamma_con`` ≈ 0, a near-per-step
    objective) — optimises the Mahalanobis tracking reward
    ``-tracking_scaler·||e||²_M/std - control_scaler·||u-uref||²/std``,
    where ``M(x) = W(x)⁻¹`` is the CMG's metric. This is what shapes the CMG
    (see ``update_cmg``/``_compute_cmg_loss``): con_policy's mean control and
    its state-Jacobian ``K = du/dx`` feed the contraction condition
    ``Cu ≺ 0`` directly, and its state ``x`` alone (no policy involvement)
    feeds C3M-style ``C1``/``C2`` conditions on the CMG.
  * ``opt_policy`` ("optimal", ``gamma_opt``, e.g. 0.99) — optimises the SAME
    Mahalanobis reward, just under a standard RL discount instead of a
    near-zero one, against the metric con_policy already shaped. This is the
    policy actually DEPLOYED at inference (``act()``) unless ``con_only=True``.

Rather than a three-way (cmg, con_policy, opt_policy) joint optimization,
training is split into two sequential phases by wall-clock timesteps
(``con_phase_fraction`` of ``timesteps``, default 0.5 — see
``C2RLSkrlTrainer.train``):

  1. **Phase 1 (con + CMG, joint)** — for the first ``con_phase_fraction`` of
     ``timesteps``, con_policy rolls out and trains against the Mahalanobis
     reward while the CMG trains alongside it (``update_cmg``, evaluated using
     con_policy's mean control/Jacobian). opt_policy does not exist yet as far
     as training is concerned — it neither acts nor updates.
  2. **Evaluate con_policy, then freeze CMG** — at the phase boundary,
     ``C2RLSkrlTrainer.eval_con_policy`` runs one bounded-episode rollout with
     con_policy's FINAL Phase 1 weights (before it stops training) and logs
     the result under the ``Stability|Reward/con_final`` tabs — the only
     "quality gate" con_policy gets, since it is never the deployed policy
     post-Phase-1. Then ``agent.freeze_cmg()`` sets ``requires_grad_(False)``
     on every CMG parameter and puts it in ``eval()`` permanently. The metric
     ``M(x)`` — and hence the Mahalanobis reward — is fixed for the remainder
     of training.
  3. **Phase 2 (opt alone, frozen CMG)** — for the remaining timesteps,
     opt_policy rolls out and trains against the Mahalanobis reward computed
     from the now-frozen ``M(x)``. con_policy and the CMG no longer update
     (``update_cmg``/``update_con`` are not called); the (now-static) dynamics
     model also stops refining, since it only ever fed ``update_cmg``.
     ``con_only=True`` skips phase 2 and the freeze entirely — con_policy (and
     the CMG) simply train for the FULL ``timesteps``, matching the old
     con_only behavior.

Buffer layout depends on ``base_algorithm``: PPO gives con/opt each its OWN
per-epoch rollout buffer (con_memory/opt_memory, reset & refilled every epoch);
SAC uses ONE shared persistent replay buffer for both (con_memory IS opt_memory
— see ``__init__``). Because the phases no longer interleave, each buffer is
only ever written by its own phase's policy.

Cadence within a phase still differs by base algorithm: PPO is epoch-based (a
full rollout + one PPO update, then CMG in phase 1), SAC is step-based (see
``C2RLSkrlTrainer.train``: after every env step it runs one CMG update — phase
1 only — and ``policy_update_per_cmg_update``/-equivalent con or opt gradient
steps).

Normalization (see ``_make_base_cfg``, ``_compute_mahalanobis_reward``,
``_compute_cmg_loss``):

  * **Observation normalization** — controlled by ``use_state_norm``, which
    defaults to **False** in every shipped config (see the yaml comments).
    When enabled, con_agent and opt_agent EACH get their own independent
    ``PathTrackingObservationScaler`` (see ``preprocessors.py``) — a
    ``RunningStandardScaler`` restricted to the ``x``/``xref`` blocks of the
    ``[x, xref, uref]`` observation, with ``uref`` and any ``angle_idx``
    columns of ``x``/``xref`` passed through UNNORMALIZED. Being independent
    instances, con_agent's scaler is fit to the state distribution
    *con_policy* visits (γ→0 rollouts) and opt_agent's to what *opt_policy*
    visits — these can diverge over training. The scaler is applied (not
    updated) inside ``act()`` and updated (``train=True``) inside each
    sub-agent's own ``update()``/gradient step — standard skrl behavior,
    inherited for free since con_agent/opt_agent are real ``PPO``/``SAC``
    instances.

    Why ``uref`` and ``angle_idx`` are excluded (both are correctness bugs,
    not just style, if normalized):
      - ``uref``: the ``control``/``mlp`` backbones (squashed or not) compute
        ``u = uref + feedback`` by slicing ``uref`` straight out of the
        observation. A normalized ``uref`` would make the applied law
        ``uref_norm + feedback`` instead of ``uref + feedback``, distorting
        the reference-tracking residual.
      - ``angle_idx``: ``models.py`` replaces each such column with
        ``(cos(theta), sin(theta))`` via ``embed_angles`` so the network sees
        a continuous, periodic input. Standardizing the raw angle first
        (``(theta - mean) / std``) breaks that periodicity — the network
        would see ``cos``/``sin`` of a shifted-and-rescaled quantity that no
        longer wraps at ``+-pi`` the way the physical angle does.
  * **Value normalization** (PPO only) — when ``use_value_norm`` (default
    True), a separate ``RunningStandardScaler(size=1)`` normalizes the value
    function's target/output, per sub-agent. SAC has no state-value network,
    so this is a no-op there.
  * **The Mahalanobis reward is UNNORMALIZED** — ``-tracking_scaler·||e||²_M -
    control_scaler·||u-uref||²``, scaled only by the yaml-set ``tracking_scaler``/
    ``control_scaler`` constants, with no running-variance rescaling. The yaml
    ``rewards_shaper_scale`` (skrl's own, a flat multiplicative constant) can
    still be layered on top via the base PPO/SAC cfg.
  * **The CMG metric M(x) and the Mahalanobis reward always use RAW
    observations** — ``_compute_mahalanobis_reward`` and the ``M(x)``/``e``
    parts of ``_compute_cmg_loss`` read straight from ``get_rollout``/the
    just-collected rollout tensors, NEVER through any preprocessor. This is
    required: ``M(x)`` and the tracking error ``e = x - xref`` are defined in
    raw physical coordinates, and per-dimension normalization would scale ``x``
    and ``xref`` independently (they occupy different observation indices with
    their own running stats), distorting ``e`` into the wrong quantity.
  * **The certified POLICY Jacobian tracks whatever the policy is trained on**
    — the one part of ``_compute_cmg_loss`` that touches the policy network
    (``K = du/dx``, con_policy's control sensitivity) routes its input through
    ``con_agent``'s OWN observation preprocessor (``self._con_obs_preprocessor``)
    before calling ``con_policy_model.compute()``:
      - ``use_state_norm`` off (the default): that preprocessor is skrl's
        identity ``_empty_preprocessor`` → the policy sees raw obs and ``K`` is
        the raw-coordinate Jacobian, same as always.
      - ``use_state_norm`` on: the policy sees the SAME ``PathTrackingObservationScaler``
        -normalized obs it was actually optimized/deployed on, and because the
        normalization is kept in the autograd graph (``no_grad=False``) while
        ``jacobian(u, x)`` still differentiates w.r.t. raw ``x``, ``K`` is the
        true sensitivity of the DEPLOYED control to the raw state (it absorbs
        the ``1/std`` factor). So the contraction certificate is always
        consistent with the network's actual input distribution — there is no
        longer a raw-vs-normalized gap between the CMG loss and how con_policy
        is trained. (This is the "consistent state norm across all networks"
        guarantee; C3M never had the gap because its policy is never wrapped in
        an observation-normalizing base agent — see c3m.py.)
"""

from __future__ import annotations

import os
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
    loss_pos_matrix_random_sampling,
    spd_inverse,
    weighted_gradients,
)
from .rl_glue import compute_mahalanobis_reward, filter_cfg_fields, make_base_rl_cfg


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

# NOTE: base_algorithm is NOT a field on either cfg below — it's an explicit
# C2RLAgent constructor kwarg (see ContractionRunner._setup_c2rl), set by
# which entry point you use (skrl_c2rl_ppo_cfg.yaml / skrl_c2rl_sac_cfg.yaml).
# Each cfg's base-algorithm fields mirror the REAL skrl PPO_CFG/SAC_CFG field
# names and defaults 1:1 (not a generic PPO-shaped stand-in used for both),
# so a c2rl-sac.yaml actually validates against SAC's own parameter names.
# _make_base_cfg() still reads from the raw yaml dict (self._raw_cfg) and
# filters against whichever of PPO_CFG/SAC_CFG actually applies, so any valid
# PPO_CFG/SAC_CFG field works from yaml even if not declared below — these
# dataclasses exist for typed access to the commonly-tuned fields (via
# self._cfg.<field>) and self-documentation, not as an exhaustive passthrough.

@dataclass
class C2RLPPOCfg(AgentCfg):
    """C2RL config for base_algorithm="PPO". PPO fields mirror skrl's PPO_CFG."""
    # PPO shared config (see skrl.agents.torch.ppo.PPO_CFG) — con_policy/
    # opt_policy each get their own PPO sub-agent built from this.
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
    use_state_norm: bool = False  # off by default — see module docstring (residual-uref distortion
                                  # + CMG-certificate consistency); every shipped config sets it too
    use_value_norm: bool = True
    # C2RL-specific
    gamma_con: float = 0.0  # con_policy's discount factor (near-zero: approximates a hard per-step constraint)
    gamma_opt: float = 0.99  # opt_policy's discount factor — also what the PathTracking figure's theoretical/
                             # empirical bounds inflate by 1/(1-gamma) with, since opt_policy is deployed unless con_only
    con_only: bool = False
    # CMG
    W_lr: float = 3e-4
    w_ub: float = 10.0
    w_lb: float = 0.1
    lbd: float = 1e-2
    eps: float = 1e-2
    W_entropy_scaler: float = 1e-3
    # Random directions sampled per loss_pos_matrix_random_sampling() call, used
    # by the C1/C2 contraction-condition losses below (mirrors C3MCfg.pd_loss_num_samples).
    pd_loss_num_samples: int = 128
    policy_update_per_cmg_update: int = 1
    batch_size: int = 1024
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    # Fraction of `timesteps` spent in Phase 1 (con_policy + CMG train jointly)
    # before the CMG is frozen and Phase 2 (opt_policy alone, on the frozen
    # metric's Mahalanobis reward) takes over the remainder — see
    # C2RLSkrlTrainer.train and the module docstring. Ignored when con_only.
    con_phase_fraction: float = 0.5
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Same mechanism as C3M:
    # pretrained offline/online before training, then refined online each epoch.
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C2RLSACCfg(AgentCfg):
    """C2RL config for base_algorithm="SAC". SAC fields mirror skrl's SAC_CFG."""
    # SAC shared config (see skrl.agents.torch.sac.SAC_CFG) — con_policy/
    # opt_policy each get their own SAC sub-agent built from this.
    rollouts: int = 16  # not a real SAC_CFG field — sizes the con/opt RandomMemory
                        # buffers C2RL refills every epoch (see C2RLSkrlTrainer)
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
    use_state_norm: bool = False  # off by default — see module docstring (residual-uref distortion
                                  # + CMG-certificate consistency); every shipped config sets it too
    # C2RL-specific
    gamma_con: float = 0.0  # con_policy's discount factor (near-zero: approximates a hard per-step constraint)
    gamma_opt: float = 0.99  # opt_policy's discount factor — also what the PathTracking figure's theoretical/
                             # empirical bounds inflate by 1/(1-gamma) with, since opt_policy is deployed unless con_only
    con_only: bool = False
    # CMG
    W_lr: float = 3e-4
    w_ub: float = 10.0
    w_lb: float = 0.1
    lbd: float = 1e-2
    eps: float = 1e-2
    W_entropy_scaler: float = 1e-3
    # Random directions sampled per loss_pos_matrix_random_sampling() call, used
    # by the C1/C2 contraction-condition losses below (mirrors C3MCfg.pd_loss_num_samples).
    pd_loss_num_samples: int = 128
    policy_update_per_cmg_update: int = 1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    # Fraction of `timesteps` spent in Phase 1 (con_policy + CMG train jointly)
    # before the CMG is frozen and Phase 2 (opt_policy alone, on the frozen
    # metric's Mahalanobis reward) takes over the remainder — see
    # C2RLSkrlTrainer.train and the module docstring. Ignored when con_only.
    con_phase_fraction: float = 0.5
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Same mechanism as C3M:
    # pretrained offline/online before training, then refined online each epoch.
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C2RLTrainerCfg(TrainerCfg):
    timesteps: int = 300000
    # NOTE: no `rollouts` here — the rollout length is read from the agent cfg
    # (agent.rollouts), mirroring ppo/sac. Keeping a trainer-side copy invited a
    # mismatch with agent.rollouts (which sizes the con/opt memory buffers).


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLAgent(Agent):
    """C2RL agent — native skrl Agent, zero mjrl dependency.

    Models in ``models`` dict — con_policy/opt_policy are CLActorModel or
    MLPResidualActorModel (backbone: control | mlp, both u = uref + feedback);
    the rest depend on ``base_algorithm``:

    base_algorithm: PPO (default)
      ``"con_policy"`` — contracting controller (γ→0)
      ``"con_value"``  — DeterministicMixin value fn for con PPO
      ``"opt_policy"`` — optimal controller (high γ)
      ``"opt_value"``  — DeterministicMixin value fn for opt PPO

    base_algorithm: SAC
      ``"con_policy"``, ``"opt_policy"`` — same as above
      ``"con_critic_1"``/``"con_critic_2"``/``"con_target_critic_1"``/``"con_target_critic_2"``
      ``"opt_critic_1"``/``"opt_critic_2"``/``"opt_target_critic_1"``/``"opt_target_critic_2"``

    Both:
      ``"cmg"``        — MetricNetwork wrapping CCM_Generator

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
        cfg: C2RLPPOCfg | C2RLSACCfg | dict,
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
        self._con_only = parsed_cfg.con_only
        self._base_algorithm = base_algorithm.upper()

        # ── Build two RL agents (con + opt) ─────────────────────────────── #
        # Memory tensors are physically allocated at agent.init() below with
        # this num_envs — they cannot be resized later, so it must be correct
        # up front (see C2RLSkrlTrainer.train, which used to mutate _num_envs
        # post-hoc; that silently corrupted the con/opt buffers on multi-env
        # Isaac Sim runs since the underlying tensors stayed shape (rollouts, 1, ...)).
        self._num_envs = num_envs
        rollouts = self._raw_cfg.get("rollouts", 300)
        if self._base_algorithm == "PPO":
            mem_size = rollouts
        else:
            # 10_000 (not skrl's usual 1M-ish default) — this buffer is allocated
            # PER PARALLEL ENV ((memory_size, num_envs, ...), see RandomMemory
            # above), and Isaac Sim runs routinely use 1000+ envs, where a
            # 1,000,000 default would try to allocate a buffer sized for a
            # billion+ transitions and OOM the machine. Configs that need a
            # bigger/smaller buffer should set `memory_size` explicitly.
            mem_size = self._raw_cfg.get("memory_size", 10000)
            if mem_size == -1:
                mem_size = 10000
                
        if self._base_algorithm == "PPO":
            con_memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
            opt_memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        else:
            shared_memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
            con_memory = shared_memory
            opt_memory = shared_memory
            
        self._cmg_buffer_size = self._raw_cfg.get("cmg_memory_size", 131072)

        # For SAC, monkey-patch the sample() method on the persistent replay buffers
        # so that it intercepts the mini-batch and recomputes the Mahalanobis rewards
        # on the fly using the latest CMG contraction metric.
        if self._base_algorithm == "SAC":
            def _patch_sample(mem):
                orig_sample = mem.sample
                def _dynamic_maha_sample(names, batch_size, mini_batches=1, sequence_length=1):
                    batches = orig_sample(names, batch_size=batch_size, mini_batches=mini_batches, sequence_length=sequence_length)
                    # names is a list of strings like ["observations", "actions", "rewards", ...]
                    try:
                        obs_idx = names.index("observations") if "observations" in names else names.index("states")
                        act_idx = names.index("actions")
                        rew_idx = names.index("rewards")
                        for b in batches:
                            obs = b[obs_idx]
                            act = b[act_idx]
                            # Compute the Mahalanobis reward on the RAW stored
                            # observations — NOT the network-preprocessed ones.
                            # The CMG metric M(x) is trained on raw x (see
                            # _compute_cmg_loss / get_rollout), and the PPO reward
                            # path (update_con/update_opt) also uses raw obs.
                            # Applying the obs preprocessor here would evaluate M
                            # and the tracking error e = x - xref in normalized
                            # coordinates — and since RunningStandardScaler scales
                            # every obs dim independently, x and xref would get
                            # different scales, distorting e into a wrong reward.
                            maha_r = self._compute_mahalanobis_reward(obs, act)
                            b[rew_idx] = maha_r
                    except ValueError:
                        pass # if rewards isn't requested, don't modify
                    return batches
                mem.sample = _dynamic_maha_sample
                
            if self._base_algorithm == "SAC":
                _patch_sample(con_memory)
                # Since opt_memory is the exact same instance as con_memory for SAC,
                # we don't need to patch it twice!
            else:
                _patch_sample(con_memory)
                _patch_sample(opt_memory)

        rl_kwargs = dict(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        def _make_base_cfg(gamma: float, name: str) -> dict:
            # Project the raw C2RL config down to *only* the base agent's own
            # fields — see rl_glue.make_base_rl_cfg for the full rationale
            # (shared verbatim with C4M, which needs the identical translation
            # for its own single deployed policy).
            return make_base_rl_cfg(
                self._raw_cfg,
                base_algorithm=self._base_algorithm,
                gamma=gamma,
                name=name,
                experiment_dir=self.experiment_dir,
                device=device,
                observation_space=observation_space,
                angle_idx=self._angle_idx,
                x_dim=self._x_dim,
                u_dim=self._u_dim,
            )

        if self._base_algorithm == "PPO":
            from skrl.agents.torch.ppo import PPO as BaseRLAgent
            con_models = {"policy": models["con_policy"], "value": models["con_value"]}
            opt_models = {"policy": models["opt_policy"], "value": models["opt_value"]} if "opt_policy" in models else {}
        elif self._base_algorithm == "SAC":
            from skrl.agents.torch.sac import SAC as BaseRLAgent
            # skrl's SAC reads self.models["critic_1"/"critic_2"/"target_critic_1"/
            # "target_critic_2"] specifically (see SAC.__init__) — NOT "q1"/"q2".
            con_models = {
                "policy": models["con_policy"], "critic_1": models["con_critic_1"], "critic_2": models["con_critic_2"],
                "target_critic_1": models["con_target_critic_1"], "target_critic_2": models["con_target_critic_2"],
            }
            opt_models = {
                "policy": models["opt_policy"], "critic_1": models["opt_critic_1"], "critic_2": models["opt_critic_2"],
                "target_critic_1": models["opt_target_critic_1"], "target_critic_2": models["opt_target_critic_2"],
            } if "opt_policy" in models else {}
        else:
            raise ValueError(f"[C2RL] Unsupported base_algorithm: {self._base_algorithm}")

        self._con_agent = BaseRLAgent(
            cfg=_make_base_cfg(parsed_cfg.gamma_con, "con"),
            models=con_models,
            memory=con_memory,
            **rl_kwargs,
        )
        self._opt_agent = BaseRLAgent(
            cfg=_make_base_cfg(parsed_cfg.gamma_opt, "opt"),
            models=opt_models,
            memory=opt_memory,
            **rl_kwargs,
        ) if not parsed_cfg.con_only else None

        # Prefix every metric key these inner agents log with "Con /"/"Opt /"
        # so both policies' logs land on distinct wandb panels instead of
        # colliding under the same key. Can't just wrap track_data(): skrl's
        # base Agent.record_transition() writes "Reward / ..."/"Episode / ..."
        # DIRECTLY into self.tracking_data, bypassing track_data() entirely —
        # only their OWN PPO/SAC loss terms go through it. Wrapping
        # write_tracking_data() instead catches both, since it re-keys
        # whatever ended up in tracking_data right before it's flushed.
        def _prefix_on_write(agent, prefix):
            import collections
            orig_write = agent.write_tracking_data
            def _wrapped(*, timestep, timesteps):
                # Re-key into a FRESH defaultdict(list) — tracking_data must stay
                # a defaultdict (skrl's own record_transition relies on that
                # auto-vivifying behavior), not a plain dict, or the NEXT
                # epoch's record_transition raises KeyError on first append.
                prefixed = collections.defaultdict(list)
                for k, v in agent.tracking_data.items():
                    prefixed[k if k.startswith(prefix) else f"{prefix}{k}"] = v
                agent.tracking_data = prefixed
                orig_write(timestep=timestep, timesteps=timesteps)
            agent.write_tracking_data = _wrapped

        _prefix_on_write(self._con_agent, "Con / ")
        if self._opt_agent is not None:
            _prefix_on_write(self._opt_agent, "Opt / ")

        # Standalone PPO/SAC get these applied to runner.agent by train.py — but
        # for a contraction algorithm, runner.agent is this OUTER C2RLAgent,
        # which has no .policy/.scheduler/.entropy_optimizer of its own, so
        # train.py's patching there is a no-op. Apply directly to the real base
        # agents instead. std-dev annealing is auto-enabled per sub-agent from
        # its OWN policy's backbone (con_policy/opt_policy can differ) rather
        # than a single yaml on/off flag — see runner.CONTROL_BACKBONES and
        # ControllerNetwork's frozen log_std (nn_modules.py's anneal_stddev=True).
        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        from contractionRL.agents.skrl.models import ControllerNetwork
        _std_dev_annealing_kwargs = self._raw_cfg.get("std_dev_annealing_kwargs")
        for _agent, _policy_model in (
            (self._con_agent, models.get("con_policy")),
            (self._opt_agent, models.get("opt_policy")),
        ):
            if _agent is None:
                continue
            patch_kl_logging(_agent)
            patch_sac_entropy_clamp(_agent)
            _std_dev_annealing = isinstance(_policy_model, ControllerNetwork)
            patch_ppo_std_annealing(_agent, _std_dev_annealing, _std_dev_annealing_kwargs)

        self._con_memory = con_memory
        self._opt_memory = opt_memory

        self._con_agent.init()
        if self._opt_agent is not None:
            self._opt_agent.init()

        # ── CMG + C2RL math ───────────────────────────────────────────────── #
        self._ccm_gen = models["cmg"].ccm_gen
        # con_policy's mean control, used for the CMG Jacobian below. Read via
        # the common GaussianMixin .compute() interface (not .cl_actor.mean_control)
        # so this works whether con_policy is CLActorModel (backbone: control)
        # or MLPResidualActorModel (backbone: mlp) — both return uref + feedback.
        self._con_policy_model = models["con_policy"]
        # con_agent's OWN observation preprocessor (RunningStandardScaler when
        # use_state_norm, else skrl's identity _empty_preprocessor). The CMG loss
        # routes the policy input through THIS SAME preprocessor (see
        # _compute_cmg_loss) so the certified Jacobian K = du/dx is taken of the
        # ACTUAL deployed controller — con_agent.act()/update() only ever feed
        # con_policy normalized obs, so evaluating it on raw obs in the loss (as
        # this used to) would certify a different function than the one deployed.
        # With use_state_norm off (the default now) this is the identity and the
        # policy sees raw obs everywhere, so there is nothing to reconcile.
        self._con_obs_preprocessor = getattr(self._con_agent, "_observation_preprocessor", None)

        # ── Dynamics: analytical or learned online (same mechanism as C3M) ── #
        # use_empirical_dynamics=True uses the env's exact get_f_and_B (classic
        # only). Otherwise a NeuralDynamics model is pretrained/refined online and
        # its get_f_and_B feeds the CMG contraction loss — Isaac envs always take
        # this path (no closed-form dynamics available).
        if not parsed_cfg.use_empirical_dynamics:
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
                self._neural_dynamics.parameters(), lr=parsed_cfg.dynamics_lr
            )
            if parsed_cfg.dynamics_lr_scheduler:
                sched_cls = getattr(torch.optim.lr_scheduler, parsed_cfg.dynamics_lr_scheduler)
                self._dynamics_lr_scheduler = sched_cls(
                    self._dynamics_optimizer, **parsed_cfg.dynamics_lr_scheduler_kwargs
                )
            else:
                self._dynamics_lr_scheduler = None

        self._data = get_rollout(self._cmg_buffer_size, "c3m")
        self._get_rollout = get_rollout

        self._W_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=parsed_cfg.W_lr)
        # _progress is set by update_cmg as timestep/timesteps of WHATEVER phase
        # it's called with (only Phase 1 — see C2RLSkrlTrainer.train). Linearly
        # anneal W_lr from W_lr to 0 over Phase 1, hitting exactly 0 at the
        # Phase 1->2 boundary where the CMG gets frozen.
        self._progress = 0.0
        self._W_lr_scheduler = LambdaLR(
            self._W_optimizer, lr_lambda=lambda _: max(0.0, 1.0 - self._progress)
        )

        checkpoint_extra = (
            {"con_value": models["con_value"], "opt_value": models.get("opt_value")}
            if self._base_algorithm == "PPO" else
            {
                "con_critic_1": models["con_critic_1"], "con_critic_2": models["con_critic_2"],
                "opt_critic_1": models.get("opt_critic_1"), "opt_critic_2": models.get("opt_critic_2"),
            }
        )
        self.checkpoint_modules.update({k: v for k, v in {
            "con_policy": models["con_policy"],
            "opt_policy": models.get("opt_policy"),
            "cmg":        models["cmg"],
            **checkpoint_extra,
        }.items() if v is not None})
        if self._neural_dynamics is not None:
            self.checkpoint_modules["dynamics"] = self._neural_dynamics

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

    def _compute_mahalanobis_reward(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute -(tracking_scaler * ||e||²_M + control_scaler * ||u - uref||²).

        tracking_scaler/control_scaler play the role of Q/R exactly like
        SD-LQR/LQR's Q_scaler/R_scaler (sdlqr.py): tracking_scaler weights the
        state error under the CURRENT contraction metric M(x), control_scaler
        weights control effort. The control term penalizes the FEEDBACK
        component (action - uref), not the total applied control, matching
        LQR's R term (which weights the closed-loop gain's contribution, not
        uref itself — uref alone isn't "effort", it's just following the
        reference). control_scaler defaults to 0.0 (no control penalty) for
        backward compatibility; pass ``actions`` to enable it.

        Thin wrapper around ``rl_glue.compute_mahalanobis_reward`` (shared
        verbatim with C4M) — see there for the full body.
        """
        cfg = self._cfg
        return compute_mahalanobis_reward(
            self._ccm_gen, observations, actions,
            x_dim=self._x_dim, u_dim=self._u_dim, angle_idx=self._angle_idx,
            w_lb=cfg.w_lb, tracking_scaler=cfg.tracking_scaler, control_scaler=cfg.control_scaler,
        )

    # ── CMG update (inlined from mjrl C2RL.compute_cmg_loss / update_cmg) ─ #

    def _compute_cmg_loss(self):
        cfg = self._cfg
        device = self._device
        x_dim, u_dim = self._x_dim, self._u_dim
        I = torch.eye(x_dim, device=device)

        buf = self._data
        n = buf["x"].shape[0]
        batch_size = min(cfg.batch_size, n)
        idx = np.random.choice(n, size=batch_size, replace=False)

        x    = torch.from_numpy(buf["x"][idx]).float().to(device).requires_grad_()
        xref = torch.from_numpy(buf["xref"][idx]).float().to(device)
        uref = torch.from_numpy(buf["uref"][idx]).float().to(device)

        # NOTE on normalization: x/xref/uref are RAW physical-unit values (that's
        # required — M(x) and e = x - xref below are defined in raw coordinates,
        # and per-dim normalization would scale x and xref independently,
        # distorting e). The POLICY, however, is trained/deployed on whatever
        # con_agent's observation preprocessor produces, so we route the policy
        # input (and only the policy input) through that SAME preprocessor:
        #   - use_state_norm off (default): identity → policy sees raw obs, and
        #     K = du/dx is the raw-coordinate Jacobian — exactly as before.
        #   - use_state_norm on: RunningStandardScaler → the policy sees the same
        #     normalized obs it was optimized on. no_grad=False keeps the (x-mean)/std
        #     map in the autograd graph, so jacobian(u, x) below still differentiates
        #     w.r.t. RAW x and correctly absorbs the 1/std factor — i.e. K is the
        #     true sensitivity of the DEPLOYED control to the raw state, keeping the
        #     certificate consistent with the network's actual input distribution.
        #     train=False so this forward never updates the scaler's running stats
        #     (only con_agent's own act()/update() should move those).
        state = torch.cat([x, xref, uref], dim=1)
        policy_obs = (
            self._con_obs_preprocessor(state, train=False, no_grad=False)
            if self._con_obs_preprocessor is not None else state
        )
        u, _ = self._con_policy_model.compute({"observations": policy_obs}, role="policy")
        K = jacobian(u, x, create_graph=False)
        u = u.detach()
        K = K.detach()

        raw_W, info_W = self._ccm_gen(x)
        # See _compute_mahalanobis_reward: honor the CMG's `bounded` flag so a
        # BoundedCCM_Generator isn't double-bounded (extra +w_lb·I), which would
        # certify a different metric than the one actually used/deployed.
        bounded = getattr(self._ccm_gen, "bounded", False)
        W = bound_W(raw_W, cfg.w_lb, x_dim, bounded)
        M = spd_inverse(W)

        with torch.enable_grad():
            f, B, Bbot = self._get_f_and_B(x)
        f = f.float().to(device)
        B = B.float().to(device)
        Bbot = Bbot.float().to(device)

        DfDx = jacobian(f, x, create_graph=False)
        DBDx = b_jacobian(B, x, u_dim, create_graph=False)
        f = f.detach(); B = B.detach(); Bbot = Bbot.detach()

        A = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, u)
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = weighted_gradients(M, dot_x, x)

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * cfg.lbd * M
        Cu = Cu + cfg.eps * I

        pd_loss, _ = loss_pos_matrix_eigen(-Cu, reg=False)

        # C3M-style C1/C2 contraction conditions on the CMG alone (no policy
        # involvement — mirrors C3MAgent._compute_loss verbatim). pd_loss/Cu
        # above already certifies the closed-loop system under con_policy; C1/C2
        # additionally shape W(x) itself to admit SOME contracting controller
        # (Bbot-projected conditions), same as C3M's offline synthesis.
        n_samp = cfg.pd_loss_num_samples
        DfW = weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * cfg.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)
        C1_reg = C1 + cfg.eps * torch.eye(C1.shape[-1], device=device)
        c1_loss = loss_pos_matrix_random_sampling(-C1_reg, num_samples=n_samp)

        c2_loss = torch.zeros(1, device=device)
        for j in range(u_dim):
            DbW = weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
            C2_inner = DbW - 2 * sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)
            c2_loss = c2_loss + (C2 ** 2).reshape(batch_size, -1).sum(1).mean()

        entropy_loss = info_W.get("entropy", torch.tensor(0.0, device=device))
        if isinstance(entropy_loss, torch.Tensor):
            entropy_loss = entropy_loss.mean() * cfg.W_entropy_scaler
        else:
            entropy_loss = torch.zeros(1, device=device)

        loss = pd_loss - entropy_loss + c1_loss + c2_loss
        return loss, {
            "pd_loss": pd_loss.item(),
            "c1_loss": c1_loss.item(),
            "c2_loss": c2_loss.item(),
        }

    # ── C2RL update methods (called by C2RLSkrlTrainer) ──────────────────── #

    def update_con(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None,
        *, timestep: int, timesteps: int,
    ) -> dict:
        """Inject Mahalanobis rewards into con_memory and drive the base update.

        PPO: overwrite the just-collected rollout's rewards with the Mahalanobis
        reward, then run PPO.update() once on that batch (PPO.post_interaction
        only fires every ``rollouts`` epochs, so the explicit call here is what
        actually drives on-policy learning).

        SAC: the reward is recomputed per mini-batch inside the patched
        memory.sample(), and the single gradient update is driven by
        ``_con_agent.post_interaction()`` (called by the trainer immediately
        after this) — which respects ``learning_starts``. Calling update() here
        as well would DOUBLE SAC's gradient steps per epoch and bypass
        ``learning_starts``, breaking parity with standalone SAC. So for SAC this
        method only refreshes rewards (implicitly, via the sample patch) and
        returns without updating.
        """
        if self._base_algorithm == "PPO":
            maha_r = self._compute_mahalanobis_reward(observations, actions)
            try:
                # con_memory's "rewards" tensor is (rollouts, num_envs, 1) — the
                # step-major/env-minor layout it was filled in during the
                # rollout loop — while maha_r is (rollouts*num_envs, 1), flat,
                # from torch.cat'ing the same step-major/env-minor obs list.
                # .shape therefore never matches even on the happy path (a 3D
                # vs. a 2D tensor); only .numel() need agree for view_as() to
                # be a valid reshape (previously this ALWAYS logged a false
                # "mismatch" warning every epoch, PPO-only — SAC's memory.sample
                # patch never hits this code path).
                r_tensor = self._con_agent.memory.get_tensor_by_name("rewards")
                r_tensor.copy_(maha_r.view_as(r_tensor))
            except RuntimeError as e:
                import skrl
                skrl.logger.warning(f"[C2RL] Failed to inject Mahalanobis reward in update_con: {e}")
            self._con_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_opt(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None,
        *, timestep: int, timesteps: int,
    ) -> dict:
        """Inject Mahalanobis rewards into opt_memory and drive the base update.

        Same PPO-vs-SAC split as ``update_con``: PPO updates explicitly here on
        the fresh rollout; SAC's single update is driven by
        ``_opt_agent.post_interaction()`` (respecting ``learning_starts``), so we
        must NOT also update here or SAC would train twice per epoch.
        """
        if self._con_only or self._opt_agent is None:
            return {}

        if self._base_algorithm == "PPO":
            maha_r = self._compute_mahalanobis_reward(observations, actions)
            try:
                # See the matching comment in update_con — .shape legitimately
                # differs ((rollouts, num_envs, 1) vs. flat (rollouts*num_envs,
                # 1)); only .numel() need agree for view_as() to reshape correctly.
                r_tensor = self._opt_agent.memory.get_tensor_by_name("rewards")
                r_tensor.copy_(maha_r.view_as(r_tensor))
            except RuntimeError as e:
                import skrl
                skrl.logger.warning(f"[C2RL] Failed to inject Mahalanobis reward in update_opt: {e}")
            self._opt_agent.update(timestep=timestep, timesteps=timesteps)
        return {}

    def update_cmg(self, *, timestep: int, timesteps: int) -> dict:
        """Refresh data buffer, refine learned dynamics, run CMG pd-loss update."""
        # Online NeuralDynamics refinement (skipped when analytical) — one MSE
        # step per epoch on fresh (x, u, x_dot) data, mirroring C3M.update().
        if self._neural_dynamics is not None:
            dyn_data = self._get_rollout(self._cfg.dynamics_batch_size, "dynamics")
            dyn_loss = self._train_dynamics(dyn_data)
            self.track_data("Loss / C2RL/dynamics/mse", dyn_loss)

        self._data = self._get_rollout(self._cmg_buffer_size, "c3m")
        self._progress = float(timestep) / max(1, timesteps)
        self._ccm_gen.train()
        
        loss, info = self._compute_cmg_loss()
        self._W_optimizer.zero_grad()
        loss.backward()
        if all(torch.isfinite(p.grad).all() for p in self._ccm_gen.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._ccm_gen.parameters(), 10.0)
            self._W_optimizer.step()
            
        self._W_lr_scheduler.step()
        self._ccm_gen.eval()

        loss_dict = {
            "C2RL/CMG/pd_loss": info["pd_loss"],
            "C2RL/CMG/c1_loss": info["c1_loss"],
            "C2RL/CMG/c2_loss": info["c2_loss"],
        }
        for k, v in loss_dict.items():
            self.track_data(f"Loss / {k}", v)
        return loss_dict

    def _train_dynamics(self, data: dict) -> float:
        """MSE training of NeuralDynamics on (x, u, x_dot) data (same as C3M)."""
        dev = self._neural_dynamics.device
        x     = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        u     = torch.as_tensor(data["u"], dtype=torch.float32, device=dev)
        x_dot = torch.as_tensor(data["x_dot"], dtype=torch.float32, device=dev)

        pred = self._neural_dynamics.predict_x_dot(x, u)
        loss = nn.functional.mse_loss(pred, x_dot)

        self._dynamics_optimizer.zero_grad()
        loss.backward()
        # Guard against NaN/Inf grads from occasional huge MSE spikes.
        if all(torch.isfinite(p.grad).all() for p in self._neural_dynamics.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._neural_dynamics.parameters(), 1.0)
            self._dynamics_optimizer.step()
        return loss.item()

    def save_dynamics(self, path: str) -> None:
        """Save NeuralDynamics checkpoint (for SDLQR/LQR reuse or inspection)."""
        if self._neural_dynamics is not None:
            self._neural_dynamics.save(path)
            print(f"[C2RL] Saved NeuralDynamics → {path}")

    def freeze_cmg(self) -> None:
        """Freeze the CMG at the Phase 1 → Phase 2 boundary (see module docstring).

        Called once by C2RLSkrlTrainer.train right after con_policy's joint
        training with the CMG ends. From here on M(x) = W(x)⁻¹ is fixed, so
        opt_policy's Mahalanobis reward (Phase 2) is computed against a static
        metric instead of one con_policy is still actively shaping.
        """
        for p in self._ccm_gen.parameters():
            p.requires_grad_(False)
        self._ccm_gen.eval()


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLSkrlTrainer(Trainer):
    """skrl Trainer for C2RL — alternates con/opt rollouts and CMG updates."""

    @staticmethod
    def _record_with_maha(
        agent, base_agent, *, observations, states, actions, rewards,
        next_obs, next_states, terminated, truncated, infos, global_step, timesteps,
    ) -> None:
        """Record a transition, logging the CONTRACTION (Mahalanobis) reward.

        The Mahalanobis reward is what con/opt actually OPTIMIZE (the env reward
        is only a placeholder that update_con/opt/SAC-sample overwrite), so it is
        what skrl's per-episode ``Reward / Total reward (max/min/mean)`` and
        ``Episode / Total timesteps`` should reflect for these two policies. The
        raw env reward is kept alongside as ``Reward / env_reward (mean)``. The
        stored memory reward is irrelevant (PPO's update_con and SAC's sample
        patch both recompute it), so only the logged cumulative changes here.
        """
        maha = agent._compute_mahalanobis_reward(observations, actions)
        base_agent.record_transition(
            observations=observations, states=states, actions=actions,
            rewards=maha.view_as(rewards), next_observations=next_obs, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=global_step, timesteps=timesteps,
        )
        base_agent.track_data("Reward / env_reward (mean)", float(rewards.float().mean()))

    @staticmethod
    def _forward_env_log(agent, infos) -> None:
        """Forward the env's per-episode ``extras['log']`` (e.g. path_tracking_base's
        or classic env_base's ``Stability/*``) onto the outer agent's tracking_data —
        exactly as skrl's SequentialTrainer does for PPO/SAC. C2RL steps the real env
        during training (it "collects samples like ppo/sac"), so it gets the SAME
        env-computed metrics/tabs from the live rollout, with NO separate eval
        rollout. Classic envs under SyncVectorEnv only surface "log" nested inside
        "final_info" (SAME_STEP autoreset), which WandbPlotWrapper lifts to a flat
        top-level "log" dict before this runs — see wandb_plot_wrapper.py. No-op if
        that lifting didn't happen (nothing finished this step).
        """
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

    def _env_scalar_attr(self, *names):
        """Fetch a scalar env attribute across both env backends (Isaac vs. classic
        SyncVectorEnv) — see the identical helper on C3MSkrlTrainer for why both
        lookup paths are needed. Duplicated here (not shared) since the two
        trainers don't share a base class."""
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

    def eval_con_policy(self, agent: C2RLAgent, *, timestep: int = 0) -> dict:
        """Bounded-rollout evaluation of con_policy's FINAL Phase 1 weights.

        Called once, right at the Phase 1 → Phase 2 boundary in ``train()`` —
        i.e. right before opt_policy starts training — so con_policy's quality
        is captured (and logged under the ``.../con_final`` tabs) before it
        stops training and the CMG it shaped gets frozen for good. Mirrors
        ``C3MSkrlTrainer.eval()`` (same streaming-metric, bounded-episode
        approach), but acts with ``agent._con_agent`` directly rather than the
        outer ``C2RLAgent.act()``, which dispatches to opt_policy — not yet a
        trained policy at this point.

        Runs its OWN ``env.reset()`` + bounded rollout, independent of the
        training loop's ``observations``/``states`` — callers must re-``reset()``
        the env afterward before resuming training (see ``train()``).
        """
        con_agent = agent._con_agent
        con_agent.enable_training_mode(False)
        observations, infos = self.env.reset()
        states = self.env.state() if hasattr(self.env, "state") else None

        from .contraction_metrics import StreamingErrorStats, log_tracking_plots, reward_summary

        x_dim = agent._x_dim
        num_envs = self.env.num_envs
        stats = StreamingErrorStats(num_envs, self.env.device)
        total_reward = torch.zeros((num_envs, 1), device=self.env.device)

        # Bounded, non-terminating-episode-count loop (same quality-gate
        # pattern as C3MSkrlTrainer.eval() / train.py's _evaluate_best_model):
        # envs desynchronize after their first reset, so track which have
        # finished their FIRST episode and freeze their accumulators.
        max_steps = int(self._env_scalar_attr("max_episode_length", "max_episode_len")) + 1
        finished = torch.zeros(num_envs, dtype=torch.bool, device=self.env.device)

        plot_idx = np.random.choice(num_envs, 1, replace=False)
        traj_x = {i: [] for i in plot_idx}
        traj_xref = {i: [] for i in plot_idx}
        traj_error = {i: [] for i in plot_idx}

        for _ in range(max_steps):
            with torch.no_grad():
                actions, _ = con_agent.act(observations, states, timestep=0, timesteps=1)

            active = (~finished).unsqueeze(-1).float()
            pos_dim_val = self._env_scalar_attr("pos_dimension")
            pos_dim = int(pos_dim_val) if pos_dim_val is not None else (3 if hasattr(self.env, "state") else x_dim)

            x_curr = observations[:, :pos_dim]
            x_ref = observations[:, x_dim:x_dim + pos_dim]
            error = torch.norm(observations[:, :x_dim] - observations[:, x_dim:2 * x_dim], dim=-1, keepdim=True)

            for i in plot_idx:
                if not finished[i]:
                    traj_x[i].append(x_curr[i].cpu().clone().numpy())
                    traj_xref[i].append(x_ref[i].cpu().clone().numpy())
                    traj_error[i].append(error[i].item())
            stats.update(error, active)

            observations, rewards, terminated, truncated, _ = self.env.step(actions)
            states = self.env.state() if hasattr(self.env, "state") else None
            total_reward += rewards.view(num_envs, 1) * active

            finished |= (terminated | truncated).view(num_envs)
            if finished.all():
                break

        con_agent.enable_training_mode(True)

        dt = float(self._env_scalar_attr("step_dt", "dt"))
        f_mask = finished if finished.any() else torch.ones_like(finished)

        log_tracking_plots(traj_x, traj_xref, traj_error, dt=dt, prefix="train/con",
                            step=timestep, title="C2RL con_policy (final, Phase 1)")
        return {
            "stability": stats.summary(dt, f_mask),
            "reward": reward_summary(total_reward, f_mask),
        }

    def train(self) -> None:
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        num_envs = env.num_envs
        # rollouts lives ONLY under `agent:` in the YAML (mirroring ppo/sac); the
        # trainer block must not carry its own — a separate trainer.rollouts that
        # disagreed with agent.rollouts used to trip the con-memory size guard
        # below (agent-sized buffer vs trainer-sized loop).
        rollout_steps = agent._cfg.rollouts
        timesteps = self.cfg.timesteps
        con_only = agent._con_only or agent._opt_agent is None

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
        # PPO's con/opt memory IS the per-epoch rollout buffer (reset & refilled
        # every epoch), so its size must equal rollout_steps. SAC's con/opt
        # memory is a PERSISTENT replay buffer whose size (memory_size, e.g. 10k)
        # is deliberately independent of rollout_steps — so this invariant only
        # applies to PPO. (Applying it to SAC always raised, since a replay
        # buffer never equals the rollout length — C2RL-SAC never got past here.)
        if agent._base_algorithm == "PPO" and agent._con_memory.memory_size != rollout_steps:
            raise ValueError(
                f"[C2RL] Agent con_memory was sized for rollouts={agent._con_memory.memory_size} "
                f"but the rollout loop uses rollouts={rollout_steps}. Both derive from "
                f"agent.rollouts, so this should not happen — check the YAML `agent.rollouts`."
            )

        agent.init(trainer_cfg=self.cfg)
        from .contraction_metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))

        # Pretrain the learned NeuralDynamics before RL (same mechanism as C3M);
        # no-op for analytical dynamics or dynamics_pretrain_epochs<=0. The helper
        # derives its own wandb flush cadence from the epoch count (the training
        # write_interval is ~timesteps//100, far too coarse for pretraining).
        from .dynamics_pretrain import pretrain_dynamics
        pretrain_dynamics(
            agent,
            epochs=getattr(agent._cfg, "dynamics_pretrain_epochs", 0),
            data_path=getattr(agent._cfg, "dynamics_pretrain_data_path", "") or None,
            timesteps=timesteps,
            tag="[C2RL]",
        )

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0

        # Both SAC and PPO flush the currently-active sub-agent's tracking data on
        # this coarse cadence — NOT every step (SAC) or every rollout chunk (PPO).
        # skrl's write_tracking_data clears the 100-episode reward/timestep
        # deques on every call, so flushing too often collapses "Reward / Total
        # reward" and "Episode / Total timesteps" to just the 0–few episodes that
        # happened to finish since the last flush → a spiky, single-episode
        # curve. Flushing every ~timesteps/100 steps (matching the standalone
        # baselines' "auto" write_interval) lets the deques fill across many
        # chunks, giving a smooth running mean. Reset at the Phase 1 → Phase 2
        # handoff since opt_agent starts its own flush cadence from scratch.
        flush_interval = max(1, timesteps // 100)
        next_flush = flush_interval

        # Phase 1 (con_policy + CMG train jointly) runs for con_phase_fraction of
        # `timesteps`; con_only collapses this to the FULL timesteps and Phase 2
        # (below) never runs — matching the old con_only behavior. See the
        # module docstring for the two-phase design this replaced tri-level
        # (cmg/con/opt) alternation with.
        phase1_end = timesteps if con_only else int(timesteps * agent._cfg.con_phase_fraction)

        pbar = _tqdm.tqdm(total=timesteps, desc="C2RL training", file=sys.stdout)

        def _run_chunk(active_agent, update_fn, *, train_cmg: bool, phase_end: int) -> None:
            """Run one rollout chunk (PPO: `rollout_steps` steps + one PPO update;
            SAC: `rollout_steps` steps, each followed by its own gradient step(s))
            for whichever policy is active this phase, optionally driving the CMG
            update alongside it (Phase 1 only — see module docstring)."""
            nonlocal observations, states, global_step, next_flush
            if agent._base_algorithm == "PPO":
                active_agent.memory.reset()
            obs_list, act_list = [], []

            steps_to_take = min(rollout_steps, phase_end - global_step)
            for _ in range(steps_to_take):
                active_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = active_agent.act(observations, states, timestep=global_step, timesteps=timesteps)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None
                obs_list.append(observations.clone())
                act_list.append(actions.clone())
                self._record_with_maha(
                    agent, active_agent, observations=observations, states=states,
                    actions=actions, rewards=rewards, next_obs=next_obs, next_states=next_states,
                    terminated=terminated, truncated=truncated, infos=infos,
                    global_step=global_step, timesteps=timesteps,
                )
                self._forward_env_log(agent, infos)
                observations = next_obs
                states = next_states

                # Step-based update for SAC
                if agent._base_algorithm == "SAC":
                    if train_cmg:
                        # `timesteps=phase_end` (not the full run) so the CMG LR
                        # scheduler's progress (see update_cmg/_W_lr_scheduler)
                        # anneals to 0 exactly at the end of THIS phase (Phase 1,
                        # the only phase update_cmg ever runs in) rather than
                        # stalling partway through a linear decay sized for the
                        # full run.
                        agent.update_cmg(timestep=global_step, timesteps=phase_end)
                    for _ in range(agent._cfg.policy_update_per_cmg_update):
                        update_fn(None, None, timestep=global_step, timesteps=timesteps)
                        active_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                    if global_step % flush_interval == 0 and getattr(active_agent, "writer", None) is not None:
                        active_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

                global_step += 1
                pbar.update(1)

            # Chunk-based update for PPO
            if agent._base_algorithm == "PPO":
                obs_tensor = torch.cat(obs_list, dim=0)
                act_tensor = torch.cat(act_list, dim=0)
                update_fn(obs_tensor, act_tensor, timestep=global_step, timesteps=timesteps)
                active_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                if train_cmg:
                    # See the matching SAC comment above: anneal over Phase 1
                    # (phase_end), not the full run.
                    agent.update_cmg(timestep=global_step, timesteps=phase_end)
                # Flush on the coarse interval, NOT every chunk (see flush_interval
                # note above) — flushing per chunk clears the reward deque ~every
                # rollout and yields the spiky curve.
                if global_step >= next_flush and getattr(active_agent, "writer", None) is not None:
                    active_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)
                    next_flush = global_step + flush_interval

        # ── Phase 1: con_policy + CMG train jointly ─────────────────────── #
        agent._con_agent.enable_training_mode(True)
        while global_step < phase1_end:
            _run_chunk(agent._con_agent, agent.update_con, train_cmg=True, phase_end=phase1_end)
            agent.post_interaction(timestep=global_step, timesteps=timesteps)
            # post_interaction()'s OWN write_interval-modulo check (skrl base
            # Agent) almost never fires here — see the historical note this
            # replaced: for most rollout/write_interval combinations the two
            # never coincide, so the outer agent's tracking_data (Stability/*
            # from _forward_env_log, plus CMG/dynamics losses tracked directly
            # on `agent`) would silently accumulate forever otherwise. Flush it
            # explicitly every chunk instead.
            if getattr(agent, "writer", None) is not None:
                agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        # ── Freeze CMG, hand off to opt_policy alone ─────────────────────── #
        if not con_only and agent._opt_agent is not None:
            # Evaluate con_policy's FINAL (post-Phase-1) weights before it stops
            # training and opt_policy takes over — see eval_con_policy's
            # docstring. Must happen before freeze_cmg()/enable_training_mode so
            # it still measures the policy exactly as Phase 1 left it.
            from .contraction_metrics import track_reward_summary, track_stability_summary
            ev = self.eval_con_policy(agent, timestep=global_step)
            track_stability_summary(agent, ev["stability"], tab="Stability/con_final")
            track_reward_summary(agent, ev["reward"], tab="Reward/con_final")
            if getattr(agent, "writer", None) is not None:
                agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

            agent.freeze_cmg()
            agent._con_agent.enable_training_mode(False)
            agent._opt_agent.enable_training_mode(True)
            next_flush = global_step + flush_interval

            # eval_con_policy ran its own reset + bounded rollout on `env`,
            # independent of the training loop's observations/states — those are
            # now stale (the real env has moved on), so Phase 2 must start from a
            # fresh reset rather than resuming from wherever Phase 1 left off.
            observations, infos = env.reset()
            states = env.state() if hasattr(env, "state") else None

            # ── Phase 2: opt_policy trains alone against the frozen metric ── #
            while global_step < timesteps:
                _run_chunk(agent._opt_agent, agent.update_opt, train_cmg=False, phase_end=timesteps)
                agent.post_interaction(timestep=global_step, timesteps=timesteps)
                if getattr(agent, "writer", None) is not None:
                    agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        # No separate eval rollout DURING each phase: C2RL steps the real env
        # while training, so its Stability/* (path-tracking) and Episode/Reward
        # metrics come from the env's own per-episode extras["log"], forwarded
        # live via _forward_env_log above — exactly like PPO/SAC. (The one
        # exception is the single con_policy quality-gate eval at the Phase 1 →
        # Phase 2 boundary above, under the Stability|Reward/con_final tabs.) The
        # authoritative convergence-rate/overshoot numbers are the post-training
        # _evaluate_best_model / _evaluate_classic_path_tracking (paper
        # fit_exponential_envelope). Live tracking plots come from the
        # env-wrapping WandbPlotWrapper, same as PPO/SAC.

        # Final flush of whatever accumulated since the last coarse-interval
        # flush (the trailing < flush_interval steps), so the tail of every
        # curve reaches wandb/tensorboard instead of being dropped on exit.
        for _sub in (agent._con_agent, agent._opt_agent):
            if _sub is not None and getattr(_sub, "writer", None) is not None:
                _sub.write_tracking_data(timestep=global_step, timesteps=timesteps)

        # Persist the learned dynamics for reuse/inspection (matches C3M).
        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            os.makedirs(os.path.dirname(dyn_path), exist_ok=True)
            agent.save_dynamics(dyn_path)

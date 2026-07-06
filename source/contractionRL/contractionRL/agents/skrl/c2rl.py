"""C2RL — Two-policy contraction-metric synthesis, native skrl Agent.

C2RLAgent trains TWO independent policies against the SAME environment and the
SAME shared CMG (contraction-metric generator), on top of a chosen base
algorithm (``base_algorithm="PPO"`` → two skrl ``PPO`` sub-agents,
``base_algorithm="SAC"`` → two skrl ``SAC`` sub-agents):

  * ``con_policy`` ("contracting") — discount ``gamma_contracting`` (≈0, a
    near-per-step objective) — optimises the Mahalanobis tracking reward
    ``-tracking_scaler·||e||²_M/std - control_scaler·||u-uref||²/std``,
    where ``M(x) = W(x)⁻¹`` is the CMG's CURRENT metric. This is what actually
    shapes the CMG (see ``update_cmg``/``_compute_cmg_loss``): con_policy's
    mean control and its state-Jacobian ``K = du/dx`` feed the contraction
    condition ``Cu ≺ 0`` directly.
  * ``opt_policy`` ("optimal") — discount ``gamma_optimal`` (e.g. 0.99) —
    optimises the SAME Mahalanobis reward, just under a standard RL discount
    instead of a near-zero one. This is the policy actually DEPLOYED at
    inference (``act()``) unless ``con_only=True``.

Both policies share one replay/rollout mechanism per epoch (each gets its OWN
memory buffer — con_memory/opt_memory — but the alternation is what makes them
"share" the environment: opt_policy picks up wherever con_policy's rollout
left the env). Every epoch (``C2RLSkrlTrainer.train``):

  1. **Con rollout** — ``rollout_steps`` env steps with *con_policy* acting,
     recorded into ``con_memory`` with the environment's OWN reward (it gets
     overwritten in step 2).
  2. **update_con** — recompute the Mahalanobis reward from the RAW observations
     just collected and inject it in place of the environment reward, then run
     one PPO/SAC gradient update for con_policy (PPO: explicit ``update()``
     call here, since the base agent's own ``rollouts``-cadence hook never
     fires — one call site drives it manually. SAC: the reward is
     recomputed on the fly by a monkey-patched ``memory.sample()`` — see
     below — and the actual gradient step is deferred to
     ``con_agent.post_interaction()``, called right after, which is what
     respects ``learning_starts``).
  3. **Opt rollout** — same as step 1, with *opt_policy* acting (skipped if
     ``con_only``).
  4. **update_opt** — same as step 2, for opt_policy.
  5. **update_cmg** — refresh the CMG training buffer via ``get_rollout`` and
     run ``cmg_updates_per_iter`` contraction pd-loss gradient steps, evaluated
     using con_policy's mean control/Jacobian (NOT opt_policy's).

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
  * **Reward normalization** — UNRELATED to the two preprocessors above. The
    Mahalanobis tracking term and the (optional) control-effort term are each
    divided by the sqrt of their own EMA'd batch variance
    (``reward_norm_beta``, independent running stats per term, both persisted
    on the OUTER C2RLAgent — not per sub-agent), so their relative scale
    stays roughly stationary even as the CMG (and hence ``M(x)``) changes
    shape during training. This plays the role of skrl's own
    ``rewards_shaper`` but is C2RL-specific and always active; the yaml
    ``rewards_shaper_scale`` (skrl's own, a flat multiplicative constant) can
    still be layered on top.
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

from .angle_utils import wrap_diff
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
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_analytical_dynamics=True (classic envs only). Same mechanism as C3M:
    # pretrained offline/online before training, then refined online each epoch.
    use_analytical_dynamics: bool = False
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
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_analytical_dynamics=True (classic envs only). Same mechanism as C3M:
    # pretrained offline/online before training, then refined online each epoch.
    use_analytical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""


@dataclass
class C2RLTrainerCfg(TrainerCfg):
    timesteps: int = 300000
    rollouts: int = 300


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
            parsed_cfg = CfgCls(**{k: v for k, v in cfg.items() if k in CfgCls.__dataclass_fields__})
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
                
        con_memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        opt_memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)

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

            # Resolve a string "learning_rate_scheduler" (e.g. "KLAdaptiveLR",
            # yaml's usual way of naming it) to the real class — skrl's Runner
            # does this via _process_cfg's eval(), which doesn't run here since
            # con_agent/opt_agent bypass Runner entirely.
            if isinstance(d.get("learning_rate_scheduler"), str):
                from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401 (used by eval below)
                d["learning_rate_scheduler"] = eval(d["learning_rate_scheduler"])
            if d.get("learning_rate_scheduler_kwargs") is None:
                d["learning_rate_scheduler_kwargs"] = {}

            # "rewards_shaper_scale" is a yaml convenience (same as skrl's own
            # Runner._process_cfg) for the real PPO_CFG/SAC_CFG field
            # "rewards_shaper", a Callable — translate it here since con_agent/
            # opt_agent bypass Runner entirely. 1.0 (or unset) is a no-op.
            rewards_shaper_scale = self._raw_cfg.get("rewards_shaper_scale")
            if rewards_shaper_scale is not None and rewards_shaper_scale != 1.0:
                d["rewards_shaper"] = lambda rewards, *a, scale=rewards_shaper_scale, **kw: rewards * scale

            # Standalone PPO/SAC get observation (and, for PPO, value)
            # normalization automatically — train.py's use_state_norm/
            # use_value_norm, applied via skrl's Runner._process_cfg (which
            # resolves the "RunningStandardScaler" string to the real class
            # and, for the legacy "state_preprocessor" yaml key specifically,
            # remaps it to "observation_preprocessor"). None of that runs here
            # — con_agent/opt_agent are built directly, bypassing Runner
            # entirely — so replicate it explicitly: same class, same
            # opt-out flags, and note the field is "observation_preprocessor"
            # (NOT "state_preprocessor", which is a different, unused-in-this-
            # repo field for asymmetric actor-critic "state" input).
            # Default OFF (see the C2RLPPOCfg/C2RLSACCfg field defaults and the
            # module docstring): a config that omits the key gets no observation
            # normalization, matching every shipped config's explicit `false`.
            if self._raw_cfg.get("use_state_norm", False):
                from contractionRL.agents.skrl.preprocessors import PathTrackingObservationScaler
                d["observation_preprocessor"] = PathTrackingObservationScaler
                d["observation_preprocessor_kwargs"] = {
                    "size": observation_space,
                    "x_dim": self._x_dim,
                    "u_dim": self._u_dim,
                    "angle_idx": self._angle_idx,
                    "device": device,
                }
            if self._base_algorithm == "PPO" and self._raw_cfg.get("use_value_norm", True):
                from skrl.resources.preprocessors.torch import RunningStandardScaler
                d["value_preprocessor"] = RunningStandardScaler
                d["value_preprocessor_kwargs"] = {"size": 1, "device": device}
            # write_interval=1 so a SummaryWriter gets created (below) — the
            # actual flush cadence is driven explicitly by C2RLSkrlTrainer
            # (once per rollout epoch), not by skrl's own interval logic, since
            # post_interaction() here is only called once per epoch anyway.
            # checkpoint_interval stays 0: checkpointing is handled by the
            # OUTER C2RLAgent (see self.checkpoint_modules below), so these
            # inner agents don't need their own redundant checkpoint files.
            # experiment.wandb is deliberately omitted (defaults False) so
            # these inner agents never call wandb.init() themselves — the
            # OUTER agent (or train.py, for a sweep) is the sole wandb.init()
            # caller; their own scalars still reach the SAME active run
            # because skrl's SummaryWriter.add_scalar is monkey-patched
            # process-wide by train.py's wandb hookup.
            d["experiment"] = {
                "directory": os.path.join(self.experiment_dir, name),
                "write_interval": 1,
                "checkpoint_interval": 0,
            }
            return d

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
            cfg=_make_base_cfg(parsed_cfg.gamma_contracting, "con"),
            models=con_models,
            memory=con_memory,
            **rl_kwargs,
        )
        self._opt_agent = BaseRLAgent(
            cfg=_make_base_cfg(parsed_cfg.gamma_optimal, "opt"),
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
        # agents instead, reading std_dev_annealing straight from raw_cfg since
        # train.py never gets a chance to pop/forward it for con_agent/opt_agent.
        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        _std_dev_annealing = self._raw_cfg.get("std_dev_annealing", False)
        _std_dev_annealing_kwargs = self._raw_cfg.get("std_dev_annealing_kwargs")
        for _agent in (self._con_agent, self._opt_agent):
            if _agent is None:
                continue
            patch_kl_logging(_agent)
            patch_sac_entropy_clamp(_agent)
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
        # use_analytical_dynamics=True uses the env's exact get_f_and_B (classic
        # only). Otherwise a NeuralDynamics model is pretrained/refined online and
        # its get_f_and_B feeds the CMG contraction loss — Isaac envs always take
        # this path (no closed-form dynamics available).
        if parsed_cfg.use_analytical_dynamics:
            if get_f_and_B is None:
                raise ValueError(
                    "C2RL: use_analytical_dynamics=True requires a get_f_and_B callable "
                    "(classic envs only). Isaac Sim envs have no analytical dynamics."
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
                    "use_analytical_dynamics=False (add a models.dynamics block to the config)."
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

        self._data = get_rollout(parsed_cfg.buffer_size, "c3m")
        self._get_rollout = get_rollout
        self._buffer_size = parsed_cfg.buffer_size

        self._W_optimizer = torch.optim.Adam(self._ccm_gen.parameters(), lr=parsed_cfg.W_lr)
        self._progress = 0.0
        self._W_lr_scheduler = LambdaLR(
            self._W_optimizer, lr_lambda=lambda _: max(0.0, 1.0 - self._progress)
        )

        self._running_reward_var: float | None = None
        self._running_control_var: float | None = None

        checkpoint_extra = (
            {"con_value": models["con_value"], "opt_value": models.get("opt_value")}
            if self._base_algorithm == "PPO" else
            {
                "con_critic_1": models["con_critic_1"], "con_critic_2": models["con_critic_2"],
                "opt_critic_1": models.get("opt_critic_1"), "opt_critic_2": models.get("opt_critic_2"),
            }
        )
        self.checkpoint_modules.update({
            "con_policy": models["con_policy"],
            "opt_policy": models.get("opt_policy"),
            "cmg":        models["cmg"],
            **checkpoint_extra,
        })
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
        """Compute -(tracking_scaler * ||e||²_M + control_scaler * ||u - uref||²),
        each term normalised by its own running variance.

        tracking_scaler/control_scaler play the role of Q/R exactly like
        SD-LQR/LQR's Q_scaler/R_scaler (sdlqr.py): tracking_scaler weights the
        state error under the CURRENT contraction metric M(x), control_scaler
        weights control effort. The control term penalizes the FEEDBACK
        component (action - uref), not the total applied control, matching
        LQR's R term (which weights the closed-loop gain's contribution, not
        uref itself — uref alone isn't "effort", it's just following the
        reference). control_scaler defaults to 0.0 (no control penalty) for
        backward compatibility; pass ``actions`` to enable it.
        """
        cfg = self._cfg
        x_dim = self._x_dim
        dtype = torch.float32

        x    = observations[:, :x_dim].to(dtype)
        xref = observations[:, x_dim : 2 * x_dim].to(dtype)
        e = wrap_diff(x - xref, self._angle_idx).unsqueeze(-1)

        with torch.no_grad():
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, cfg.w_lb, x_dim)
            M = spd_inverse(W)
            quad = (e.transpose(1, 2) @ M @ e).squeeze(-1)

            beta = cfg.reward_norm_beta
            batch_var = quad.var(unbiased=False).item()
            if self._running_reward_var is None:
                self._running_reward_var = batch_var + 1e-8
            else:
                self._running_reward_var = beta * self._running_reward_var + (1 - beta) * batch_var
            std = self._running_reward_var ** 0.5 + 1e-8
            reward = -cfg.tracking_scaler * quad / std

            if actions is not None and cfg.control_scaler > 0:
                u_dim = self._u_dim
                uref = observations[:, 2 * x_dim : 2 * x_dim + u_dim].to(dtype)
                feedback = actions.to(dtype) - uref
                control_cost = (feedback ** 2).sum(dim=-1, keepdim=True)

                control_batch_var = control_cost.var(unbiased=False).item()
                if self._running_control_var is None:
                    self._running_control_var = control_batch_var + 1e-8
                else:
                    self._running_control_var = beta * self._running_control_var + (1 - beta) * control_batch_var
                control_std = self._running_control_var ** 0.5 + 1e-8
                reward = reward - cfg.control_scaler * control_cost / control_std
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
        # PPO's con/opt memory IS the per-epoch rollout buffer (reset & refilled
        # every epoch), so its size must equal rollout_steps. SAC's con/opt
        # memory is a PERSISTENT replay buffer whose size (memory_size, e.g. 10k)
        # is deliberately independent of rollout_steps — so this invariant only
        # applies to PPO. (Applying it to SAC always raised, since a replay
        # buffer never equals the rollout length — C2RL-SAC never got past here.)
        if agent._base_algorithm == "PPO" and agent._con_memory.memory_size != rollout_steps:
            raise ValueError(
                f"[C2RL] Agent con_memory was sized for rollouts={agent._con_memory.memory_size} "
                f"but the trainer is configured for rollouts={rollout_steps}. Keep "
                f"agent.rollouts and trainer.rollouts in sync in the YAML config."
            )

        agent.init(trainer_cfg=self.cfg)

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

        total_iters = timesteps // (rollout_steps * (1 if con_only else 2))
        pbar = _tqdm.tqdm(range(max(1, total_iters)), desc="C2RL training", file=sys.stdout)

        for epoch in pbar:

            # ── Phase 1: contracting rollout ──────────────────────────── #
            agent._con_agent.enable_training_mode(True)
            if agent._base_algorithm == "PPO":
                agent._con_agent.memory.reset()
            con_obs_list, con_act_list = [], []

            for step in range(rollout_steps):
                agent._con_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = agent._con_agent.act(
                        observations, states, timestep=global_step, timesteps=timesteps
                    )
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None
                con_obs_list.append(observations.clone())
                con_act_list.append(actions.clone())
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
            con_act_tensor = torch.cat(con_act_list, dim=0)
            agent.update_con(con_obs_tensor, con_act_tensor, timestep=global_step, timesteps=timesteps)
            agent._con_agent.post_interaction(timestep=global_step, timesteps=timesteps)
            if getattr(agent._con_agent, "writer", None) is not None:
                agent._con_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

            # ── Phase 2: optimal rollout ──────────────────────────────── #
            if not con_only and agent._opt_agent is not None:
                agent._opt_agent.enable_training_mode(True)
                if agent._base_algorithm == "PPO":
                    agent._opt_agent.memory.reset()
                opt_obs_list, opt_act_list = [], []

                for step in range(rollout_steps):
                    agent._opt_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                    with torch.no_grad():
                        actions, _ = agent._opt_agent.act(
                            observations, states, timestep=global_step, timesteps=timesteps
                        )
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
                    next_states = env.state() if hasattr(env, "state") else None
                    opt_obs_list.append(observations.clone())
                    opt_act_list.append(actions.clone())
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
                opt_act_tensor = torch.cat(opt_act_list, dim=0)
                agent.update_opt(opt_obs_tensor, opt_act_tensor, timestep=global_step, timesteps=timesteps)
                agent._opt_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                if getattr(agent._opt_agent, "writer", None) is not None:
                    agent._opt_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

            # ── CMG update ────────────────────────────────────────────── #
            cmg_dict = agent.update_cmg(timestep=global_step, timesteps=timesteps)
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            pd_loss = cmg_dict.get("C2RL/CMG/pd_loss", float("nan"))
            pbar.set_postfix(epoch=epoch, pd=f"{pd_loss:.3g}")

        # Persist the learned dynamics for reuse/inspection (matches C3M).
        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            os.makedirs(os.path.dirname(dyn_path), exist_ok=True)
            agent.save_dynamics(dyn_path)

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

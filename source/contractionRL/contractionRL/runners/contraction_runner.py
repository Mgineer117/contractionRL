"""ContractionRunner — single unified Runner for all algorithms × all envs.

Mirrors the skrl Runner interface::

    runner = ContractionRunner(env, cfg, task_id="Car-Direct-v0",
                               num_envs=4, is_classic=True)
    runner.run()

Algorithm routing
-----------------
PPO / SAC (and any other skrl-native algo):
    → skrl Runner  (skrl.utils.runner.torch.Runner)

C3M / LQR / SDLQR / C2RL-PPO / C2RL-SAC:
    → native skrl Agent subclasses (C3MAgent, LQRAgent, SDLQRAgent, C2RLAgent)
    with custom skrl Trainers (C3MSkrlTrainer, C2RLSkrlTrainer, or
    SequentialTrainer for eval-only analytical agents). C2RL-PPO/C2RL-SAC are
    the SAME C2RLAgent class — the algo string just selects which
    base_algorithm (PPO or SAC) its single deployed policy is built on. C2RL
    always synthesizes an offline, frozen CMG network (Neural Contraction
    Metric); ``cmg_method`` selects how it's trained ("ccm" C1/C2 loss
    minimization, or "cvstem" SDP regression). See _setup_c2rl.

Classic env: internally creates SyncVectorEnv + wrap_env.
Isaac env:   env is already a SkrlVecEnvWrapper — passed through.

Custom YAML flags handled here (stripped before passing to skrl)
-----------------------------------------------------------------
use_state_norm: true/false  →  observation_preprocessor
use_value_norm: true/false  →  value_preprocessor (PPO only)
anneal_log_std: true/false  →  exponential log_std schedule 0 → -20
"""

from __future__ import annotations

import copy
import os
import warnings

import gymnasium as gym

from contractionRL.agents.skrl.rl_glue import filter_cfg_fields as _filter_cfg_fields

_SKRL_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "amp", "ippo", "mappo"})
_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "c2rl-ppo", "c2rl-sac"})


def _make_cmg_bounds_fn(cmg_model, w_lb: float):
    """Build the (x_batch) -> (m_bar, m_underbar) closure a path-tracking env's
    PathTracking figure uses to draw its theoretical exponential bound (see
    PathTrackingBase.set_contraction_certificate) — the CURRENT contraction
    metric's eigenvalue extremes over a batch of states, not the static
    w_lb/w_ub config bounds.
    """
    from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse

    ccm_gen = cmg_model.ccm_gen
    x_dim = cmg_model.x_dim

    def _bounds_fn(x_batch):
        import torch
        with torch.no_grad():
            raw_W, _ = ccm_gen(x_batch)
            # Honor the CMG's `bounded` flag so a BoundedCCM_Generator (already
            # eigenvalue-bounded) isn't double-bounded by an extra +w_lb·I —
            # otherwise the plotted contraction envelope would use a different
            # metric than the deployed one. Mirrors c2rl.py's bound_W calls.
            W = bound_W(raw_W, w_lb, x_dim, getattr(ccm_gen, "bounded", False))
            M = spd_inverse(W)
            eigvals = torch.linalg.eigvalsh(M)
        return float(eigvals.max().item()), float(eigvals.min().clamp(min=1e-12).item())

    return _bounds_fn


def _build_c3m_models(models_cfg: dict, agent_cfg: dict, obs_space, act_space, device,
                       x_dim, u_dim, angle_idx: list, policy_key: str = "policy") -> dict:
    """Build the policy/cmg/(optional) dynamics models for a C3M-style offline
    synthesis phase — used by ``_setup_c3m`` (pure C3M, ``policy_key="policy"``).
    Mutates ``agent_cfg`` (pops ``actor_architecture`` so it doesn't trip the
    C3MCfg field filter). The returned dict uses the keys ``"policy"``/``"cmg"``/
    (optional)``"dynamics"`` — what C3MAgent itself expects.
    """
    angle_idx = list(angle_idx or [])
    hd_policy = models_cfg.get(policy_key, {}).get("network", [{}])[0].get("layers", [128, 128])
    hd_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("layers", [128, 128])
    act_policy = models_cfg.get(policy_key, {}).get("network", [{}])[0].get("activations", "tanh")
    act_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("activations", "tanh")

    # agent.actor_architecture: sweep-friendly override of the *policy* network
    # spec (the CLActor's w1/w2 MLPs share these hidden layers; only their
    # input/output sizes differ). Accepts a bare hidden-layer list
    # (e.g. [256, 256]) or a dict {layers: [...], activations: "tanh"}.
    # Only the policy net is affected — not the CMG.
    actor_arch = agent_cfg.pop("actor_architecture", None)
    if isinstance(actor_arch, dict):
        hd_policy = actor_arch.get("layers", hd_policy)
        act_policy = actor_arch.get("activations", act_policy)
    elif actor_arch is not None:
        hd_policy = list(actor_arch)

    # C3M is a certificate-based controller synthesis (not a stochastic-policy
    # RL method), so the CLActor should be deterministic by default — no
    # Gaussian log_std / sampling machinery. A config may still opt into the
    # GaussianMixin wrapper explicitly.
    #
    # backbone: control-squashed selects the tanh-squashed (bounded) variant —
    # same "backbone name picks the class" convention PPO/SAC's
    # _gaussian_factory uses (see runner.py). C3M certifies whichever control
    # law the selected class's compute() returns (see _compute_loss below), so
    # squashed-vs-unbounded is entirely a function of this choice, not a
    # separate flag.
    policy_class_str = models_cfg.get(policy_key, {}).get("class", "DeterministicMixin")
    policy_backbone = models_cfg.get(policy_key, {}).get("backbone", "control")
    squashed = policy_backbone == "control-squashed"
    if policy_class_str == "GaussianMixin":
        if squashed:
            # SquashedCLActorModel.compute() deliberately returns the PRE-squash
            # feedback (mean of the Normal) — squashing happens inside act(),
            # for SAC's log_prob correction (see _TanhSquashMixin). C3M
            # certifies compute()'s output directly (no sampling), so this
            # combination would silently certify an unbounded, uref-less
            # value instead of the actual bounded control law. Use the
            # DeterministicMixin default instead (policy.class unset, or
            # explicitly "DeterministicMixin") — SquashedCLDeterministicActorModel
            # squashes inside compute() itself.
            raise ValueError(
                "C3M backbone: control-squashed requires class: DeterministicMixin "
                "(the default) — GaussianMixin + control-squashed isn't supported for "
                "C3M's certification, which calls compute() directly rather than act()."
            )
        from contractionRL.agents.skrl.models import CLActorModel
        policy_cls = CLActorModel
    else:
        from contractionRL.agents.skrl.models import CLDeterministicActorModel, SquashedCLDeterministicActorModel
        policy_cls = SquashedCLDeterministicActorModel if squashed else CLDeterministicActorModel

    policy_kwargs = models_cfg.get(policy_key, {}).copy()
    policy_kwargs.pop("class", None)
    policy_kwargs.pop("network", None)
    policy_kwargs.pop("backbone", None)
    policy_kwargs.pop("angle_idx", None)  # angle_idx below is the single source of truth

    constrain_eigenvalues = models_cfg.get("cmg", {}).get("network", [{}])[0].get("constrain_eigenvalues", False)
    w_lb = agent_cfg.get("w_lb", 0.1)
    w_ub = agent_cfg.get("w_ub", 10.0)

    from contractionRL.agents.skrl.models import MetricModel
    models = {
        "policy": policy_cls(obs_space, act_space, device, hidden_dim=hd_policy, activation=act_policy, x_dim=x_dim, angle_idx=angle_idx, **policy_kwargs),
        "cmg": MetricModel(obs_space, act_space, device, hidden_dim=hd_cmg, activation=act_cmg, x_dim=x_dim, angle_idx=angle_idx, constrain_eigenvalues=constrain_eigenvalues, w_lb=w_lb, w_ub=w_ub),
    }

    # Only build a NeuralDynamics when learning dynamics (use_empirical_dynamics
    # =True). Under analytical dynamics the agent ignores it (C3MAgent uses the
    # env's get_f_and_B), so building it would just waste params/memory.
    if "dynamics" in models_cfg and agent_cfg.get("use_empirical_dynamics", False):
        from contractionRL.agents.skrl.nn_modules import NeuralDynamics
        dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
        dyn_hd = dyn_net.get("layers", [256, 256])
        dyn_act = dyn_net.get("activations", "relu")
        models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx)

    return models


def _build_gaussian_policy(models_cfg: dict, block_name: str, obs_space, state_space, act_space,
                           device, x_dim, angle_idx: list, agent_class: str):
    """Build a deployed policy — same backbone dispatch PPO/SAC path-tracking
    configs use (control -> CLActorModel, mlp -> MLPResidualActorModel, both
    u = uref + feedback). Used by ``_setup_c2rl``. ``agent_class`` lets
    ``_gaussian_factory`` reject unbounded backbones for SAC-based agents
    (divergence) and squashed backbones for PPO-based ones.
    """
    from contractionRL.agents.skrl.runner import _gaussian_factory

    spec = models_cfg.get(block_name, {}).copy()
    spec.pop("class", None)
    backbone = spec.pop("backbone", "control")
    spec.setdefault("clip_log_std", True)
    spec.setdefault("min_log_std", -4.605)
    spec.setdefault("max_log_std", 2.0)
    spec.setdefault("angle_idx", angle_idx)
    # x_dim is already known from the env — pass it explicitly so backbones
    # like mlp-squashed, which see both path-tracking and vel-tracking
    # layouts, don't have to guess the [x, xref, uref] split from obs_dim/
    # act_dim parity alone.
    spec.setdefault("x_dim", x_dim)
    return _gaussian_factory(obs_space, state_space, act_space, device,
                             backbone=backbone, agent_class=agent_class, **spec)


def _build_critics(models_cfg: dict, block_name: str, base_algorithm: str, obs_space, act_space,
                   device, x_dim, angle_idx: list, key_prefix: str = "") -> dict:
    """Build the value/critic model(s) for one policy — a single V(obs) model
    for PPO, or the twin-Q + target architecture (critic_1/2 + targets) for
    SAC. Used by ``_setup_c2rl`` (``key_prefix=""``, a single deployed policy).
    """
    from contractionRL.agents.skrl.models import EmbeddedDeterministicModel

    net = models_cfg.get(block_name, {}).get("network", [{}])
    hd = net[0].get("layers", [256, 256]) if net else [256, 256]
    act = net[0].get("activations", "tanh") if net else "tanh"

    if base_algorithm == "PPO":
        return {f"{key_prefix}value": EmbeddedDeterministicModel(
            obs_space, act_space, device, hidden_dim=hd, activation=act,
            x_dim=x_dim, angle_idx=angle_idx, use_actions=False,
        )}
    elif base_algorithm == "SAC":
        return {
            f"{key_prefix}{k}": EmbeddedDeterministicModel(
                obs_space, act_space, device, hidden_dim=hd, activation=act,
                x_dim=x_dim, angle_idx=angle_idx, use_actions=True,
            )
            for k in ("critic_1", "critic_2", "target_critic_1", "target_critic_2")
        }
    else:
        raise ValueError(f"Unsupported base_algorithm: {base_algorithm}")


# ────────────────────────────────────────────────────────────────────────── #
# Helper: translate custom YAML flags to skrl format
# ────────────────────────────────────────────────────────────────────────── #

def _prepare_skrl_cfg(cfg: dict, algo: str) -> tuple[dict, bool]:
    """Pop ContractionRL-specific keys and translate to skrl conventions."""
    cfg = copy.deepcopy(cfg)
    a = cfg.setdefault("agent", {})

    use_state = a.pop("use_state_norm", True)
    use_value = a.pop("use_value_norm", True)
    anneal = a.pop("anneal_log_std", False)

    if use_state:
        a["observation_preprocessor"] = "RunningStandardScaler"
        a["observation_preprocessor_kwargs"] = None
        if algo == "ppo":
            a["state_preprocessor"] = "RunningStandardScaler"
            a["state_preprocessor_kwargs"] = None
    else:
        for k in ("state_preprocessor", "state_preprocessor_kwargs",
                  "observation_preprocessor", "observation_preprocessor_kwargs"):
            a.pop(k, None)

    if use_value and algo == "ppo":
        a["value_preprocessor"] = "RunningStandardScaler"
        a["value_preprocessor_kwargs"] = None
    else:
        a.pop("value_preprocessor", None)
        a.pop("value_preprocessor_kwargs", None)

    return cfg, anneal


# ────────────────────────────────────────────────────────────────────────── #
# Helper: PPO log_std annealing hook
# ────────────────────────────────────────────────────────────────────────── #

def _patch_ppo_annealing(runner) -> None:
    """Exponential schedule: log_std = 0*(1-t^5) + (-20)*t^5, t=progress."""
    import torch
    _orig = runner.agent.post_interaction

    def _hook(timestep: int, timesteps: int) -> None:
        _orig(timestep=timestep, timesteps=timesteps)
        ratio = (timestep / max(1, timesteps)) ** 5.0
        val = max(-20.0, min(2.0, float(-20.0 * ratio)))
        with torch.no_grad():
            runner.agent.policy.log_std_parameter.fill_(val)

    runner.agent.post_interaction = _hook


# ────────────────────────────────────────────────────────────────────────── #
# Helper: build skrl env wrapper for contraction agents
# ────────────────────────────────────────────────────────────────────────── #

def _make_skrl_env(env, task_id: str | None, num_envs: int, is_classic: bool):
    """Return a skrl-wrapped env suitable for contraction agents."""
    if is_classic:
        # Reuse the caller's env if it's already a real, usable one instead of
        # unconditionally rebuilding a bare duplicate. train.py's classic path
        # already constructs SyncVectorEnv + wrap_env + WandbPlotWrapper before
        # calling ContractionRunner — rebuilding from scratch here silently
        # discarded that wrapping, so C3M/LQR/SDLQR/C2RL's actual training loop
        # (which steps whatever this function returns, NOT the caller's `env`)
        # ran against an unwrapped env: no Stability/* forwarding, no
        # normalized_error/path_tracking plots. Building fresh from `task_id`
        # remains the fallback for callers that pass a bare/unvectorized env
        # (see the module docstring's documented usage).
        from ..agents.skrl.contraction_metrics import StatManagerEnvWrapper
        if hasattr(env, "step") and hasattr(env, "observation_space") and hasattr(env, "num_envs"):
            # Auto-collect path-tracking metrics on every reset/step so eval loops
            # don't hand-thread stats.update(...) (see StatManagerEnvWrapper).
            return StatManagerEnvWrapper(env) if not hasattr(env, "stability_summary") else env
        if task_id is None and hasattr(env, "spec") and env.spec:
            task_id = env.spec.id
        if task_id is None:
            raise ValueError("task_id is required for classic envs")
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        vec_env = SyncVectorEnv([lambda: gym.make(task_id)] * num_envs)
        # Do NOT force vec_env.device = "cpu" here. The underlying physics step
        # is numpy/CPU-bound either way, but GymnasiumWrapper.step/reset already
        # bridges numpy <-> torch at the right device via tensorize_space /
        # untensorize_space (actions are .cpu().numpy()'d before being handed to
        # the numpy env; observations are re-tensorized onto `device` after).
        # Leaving vec_env.device unset lets GymnasiumWrapper fall back to the
        # global skrl device (cuda:0 when available) — matching Isaac envs and
        # letting C3M/C2RL's batched Jacobian/backprop math run on GPU instead
        # of silently pinning the whole classic pipeline (including the neural
        # networks) to CPU.
        return StatManagerEnvWrapper(wrap_env(vec_env, wrapper="gymnasium")) if not hasattr(env, "stability_summary") else wrap_env(vec_env, wrapper="gymnasium")
    return env  # Isaac: already wrapped


# ────────────────────────────────────────────────────────────────────────── #
# ContractionRunner
# ────────────────────────────────────────────────────────────────────────── #

class ContractionRunner:
    """Unified runner — same interface as ``skrl.utils.runner.torch.Runner``.

    Args:
        env: gymnasium environment (raw single env or pre-wrapped).
        cfg: agent + trainer config dict (loaded from YAML).
        task_id: gym env id; needed for SyncVectorEnv on classic envs.
        num_envs: number of parallel envs (classic envs only).
        is_classic: True for classic (no-Isaac) envs.
        ml_framework: ``"torch"`` or ``"jax"`` (skrl-native algos only).
        dynamics_model: optional NeuralDynamics for Isaac envs. Injected into
            the env via ``set_dynamics_model()`` so ``get_f_and_B`` works for
            C3M / LQR / SD-LQR / C2RL. Classic envs have analytical dynamics
            and do not need this.
    """

    def __init__(
        self,
        env,
        cfg: dict,
        *,
        task_id: str | None = None,
        num_envs: int = 1,
        is_classic: bool = True,
        ml_framework: str = "torch",
        dynamics_model=None,
    ) -> None:
        algo = cfg.get("agent", {}).get("class", "ppo").lower()

        if algo in _SKRL_ALGOS:
            self._setup_skrl(env, cfg, algo, task_id, num_envs, is_classic, ml_framework)
        elif algo in _CONTRACTION_ALGOS:
            self._setup_contraction(env, cfg, algo, task_id, num_envs, is_classic, dynamics_model)
        else:
            raise ValueError(
                f"Unknown algorithm '{algo}'. "
                f"skrl: {sorted(_SKRL_ALGOS)}, contraction: {sorted(_CONTRACTION_ALGOS)}"
            )

    # ── skrl-native algos (PPO/SAC/etc.) ───────────────────────────────── #

    def _setup_skrl(self, env, cfg, algo, task_id, num_envs, is_classic, ml_framework):
        skrl_cfg, anneal = _prepare_skrl_cfg(cfg, algo)
        skrl_cfg["trainer"]["close_environment_at_exit"] = False

        # Unwrap to the deepest raw env to check for angle_idx
        raw_env = env
        seen = set()
        while True:
            rid = id(raw_env)
            if rid in seen:
                break
            seen.add(rid)
            inner = getattr(raw_env, "unwrapped", None) or getattr(raw_env, "env", None)
            if inner is None or inner is raw_env:
                break
            raw_env = inner

        angle_idx = list(getattr(raw_env, "angle_idx", []) or [])
        if not angle_idx and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            angle_idx = list(getattr(first_env, "angle_idx", []) or [])

        if skrl_cfg.get("agent", {}).get("observation_preprocessor") == "RunningStandardScaler" and angle_idx:
            raise ValueError(
                "use_state_norm=True (RunningStandardScaler) is fundamentally incompatible "
                "with environments that expose angle_idx. Normalizing raw angles linearly "
                "destroys the periodicity required by embed_angles(cos, sin). "
                "Disable use_state_norm or use a custom scaler."
            )

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)

        if ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            from skrl.utils.runner.jax import Runner

        self._runner = Runner(skrl_env, skrl_cfg)
        self._env = skrl_env
        self._mode = "skrl"

        if anneal and algo == "ppo":
            _patch_ppo_annealing(self._runner)

    # ── contraction agents (C3M/LQR/SDLQR/C2RL) ───────────────────────── #

    def _setup_contraction(self, env, cfg, algo, task_id, num_envs, is_classic, dynamics_model=None):
        from contractionRL.agents.skrl.models import CLActorModel, MetricModel

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)
        # Use the wrapped env's device for both classic and Isaac envs. The
        # classic env's own physics step is numpy/CPU-bound regardless (see
        # _make_skrl_env), but the agent's neural networks and gradient math
        # (C3M/C2RL Jacobians, batched contraction losses) benefit hugely from
        # GPU — hardcoding CPU here was pinning that compute off the GPU even
        # when one was available. SD-LQR/LQR are unaffected: they pin their own
        # internal `_compute_device = "cpu"` regardless of this value, since
        # scipy's CARE solver is CPU-only anyway.
        device = skrl_env.device
        obs_space = skrl_env.observation_space
        act_space = skrl_env.action_space
        state_space = getattr(skrl_env, "state_space", None)

        # Unwrap to the deepest raw env that exposes get_rollout / get_f_and_B.
        # For classic envs this is the single gymnasium env; for Isaac envs it
        # is the DirectRLEnv subclass buried under SkrlVecEnvWrapper + gym layers.
        raw_env = env
        seen = set()
        while True:
            rid = id(raw_env)
            if rid in seen:
                break
            seen.add(rid)
            inner = getattr(raw_env, "unwrapped", None) or getattr(raw_env, "env", None)
            if inner is None or inner is raw_env:
                break
            raw_env = inner

        # For Isaac Sim envs: inject the neural dynamics model so the env can
        # provide get_f_and_B. Classic envs have analytical dynamics built-in.
        if not is_classic and dynamics_model is not None:
            if hasattr(raw_env, "set_dynamics_model"):
                raw_env.set_dynamics_model(dynamics_model)

        get_f_and_B = getattr(raw_env, "get_f_and_B", None)
        get_rollout = getattr(raw_env, "get_rollout", None)
        if get_f_and_B is None and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            get_f_and_B = getattr(first_env, "get_f_and_B", None)
            get_rollout = getattr(first_env, "get_rollout", None)

        agent_cfg = copy.deepcopy(cfg.get("agent", {}))
        agent_cfg.pop("class", None)
        trainer_cfg = copy.deepcopy(cfg.get("trainer", {}))
        trainer_cfg.pop("class", None)
        trainer_cfg.setdefault("close_environment_at_exit", False)

        # C2RL keeps its contraction-metric (cm), CMG-synthesis (cmg), and
        # empirical-dynamics knobs in their own top-level yaml categories (not
        # nested under agent:) — see _setup_c2rl, which merges these into
        # agent_cfg before building the C2RLAgent. Other contraction algos
        # (C3M/LQR/SD-LQR) don't have these sections, so this is a no-op {} for
        # them.
        cm_cfg = copy.deepcopy(cfg.get("cm", {}))
        cmg_cfg = copy.deepcopy(cfg.get("cmg", {}))
        empirical_dynamics_cfg = copy.deepcopy(cfg.get("empirical_dynamics", {}))

        # Analytical dynamics (use_empirical_dynamics=False) needs the env's exact
        # get_f_and_B, which only classic envs expose. Isaac envs must learn a
        # NeuralDynamics (use_empirical_dynamics=True) — train.py forces this, so
        # this guard only fires on a hand-rolled/standalone misconfiguration.
        use_empirical_dynamics = agent_cfg.get(
            "use_empirical_dynamics", empirical_dynamics_cfg.get("use_empirical_dynamics", False)
        )
        if not use_empirical_dynamics and not is_classic:
            raise ValueError(
                "Analytical dynamics (use_empirical_dynamics=False) is only valid for "
                "classic envs that expose an analytical get_f_and_B. Isaac Sim envs have "
                "no analytical dynamics — set use_empirical_dynamics=True (pass "
                "--use_empirical_dynamics) and let the agent train NeuralDynamics online."
            )

        x_dim = getattr(raw_env, "x_dim", None)
        u_dim = getattr(raw_env, "u_dim", None)
        # angle_idx: indices within an x-block that hold a raw (wrapping) angle.
        # Every network built below embeds these as (cos, sin) at its input —
        # see angle_utils.py / models.py — while the env/loss/error math keeps
        # using the RAW state. Defaults to [] (no angle dims) for envs that
        # don't declare it.
        angle_idx = list(getattr(raw_env, "angle_idx", []) or [])
        if x_dim is None and hasattr(raw_env, "envs") and len(raw_env.envs) > 0:
            first_env = raw_env.envs[0]
            if hasattr(first_env, "unwrapped"):
                first_env = first_env.unwrapped
            x_dim = getattr(first_env, "x_dim", None)
            u_dim = getattr(first_env, "u_dim", None)
            if not angle_idx:
                angle_idx = list(getattr(first_env, "angle_idx", []) or [])

        models_cfg = copy.deepcopy(cfg.get("models", {}))
        memory_cfg = copy.deepcopy(cfg.get("memory", {}))

        # Snapshot of the raw (pre dataclass-filter) YAML dicts, taken before
        # any _setup_* mutates agent_cfg/trainer_cfg/models_cfg in place — see
        # contraction_metrics.log_raw_config for why this is logged separately
        # from skrl's own dataclass-based wandb config.
        raw_cfg_snapshot = copy.deepcopy({
            "agent": agent_cfg, "trainer": trainer_cfg, "models": models_cfg,
            "cm": cm_cfg, "cmg": cmg_cfg, "empirical_dynamics": empirical_dynamics_cfg,
        })

        if algo in ("c3m",):
            self._setup_c3m(skrl_env, device, obs_space, state_space, act_space,
                            agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                            angle_idx=angle_idx, raw_cfg_snapshot=raw_cfg_snapshot)
        elif algo in ("lqr", "sdlqr"):
            # LQR/SD-LQR build no networks (get_f_and_B is either analytical or
            # an externally-loaded NeuralDynamics whose angle_idx was already
            # baked in when IT was trained) — nothing here to embed.
            self._setup_sdlqr(skrl_env, device, obs_space, state_space, act_space,
                              agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr=(algo == "lqr"), x_dim=x_dim, u_dim=u_dim, angle_idx=angle_idx,
                              raw_cfg_snapshot=raw_cfg_snapshot)
        elif algo in ("c2rl-ppo", "c2rl-sac"):
            # base_algorithm is derived from the algo string itself (i.e. which
            # entry point / yaml you used), not a config toggle — see
            # C2RLAgent's base_algorithm constructor kwarg.
            self._setup_c2rl(skrl_env, device, obs_space, state_space, act_space,
                             agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                             base_algorithm=("PPO" if algo == "c2rl-ppo" else "SAC"), angle_idx=angle_idx,
                             raw_cfg_snapshot=raw_cfg_snapshot,
                             cm_cfg=cm_cfg, cmg_cfg=cmg_cfg, empirical_dynamics_cfg=empirical_dynamics_cfg)

    def _setup_c3m(self, env, device, obs_space, state_space, act_space,
                   agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                   angle_idx=None, raw_cfg_snapshot=None):
        angle_idx = list(angle_idx or [])
        from contractionRL.agents.skrl.c3m import C3MAgent, C3MCfg, C3MSkrlTrainer, C3MTrainerCfg

        models = _build_c3m_models(models_cfg, agent_cfg, obs_space, act_space, device, x_dim, u_dim, angle_idx)
        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)

        agent = C3MAgent(
            cfg=C3MCfg(**_filter_cfg_fields(agent_cfg, C3MCfg, context="ContractionRunner agent")),
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        trainer_cfg.setdefault("timesteps", 30000)
        trainer = C3MSkrlTrainer(
            cfg=C3MTrainerCfg(**_filter_cfg_fields(trainer_cfg, C3MTrainerCfg, context="ContractionRunner trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._trainer._wandb_raw_cfg = raw_cfg_snapshot
        self._env = env
        self._mode = "c3m"

        # C3M has no discounting (no_op discount_factor=None) — its certificate
        # is a hard per-step LMI, not a discounted-objective approximation.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=None,
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(models["cmg"], w_lb),
            )

    def _setup_sdlqr(self, env, device, obs_space, state_space, act_space,
                     agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr: bool = False, x_dim=None, u_dim=None, angle_idx=None,
                     raw_cfg_snapshot=None):
        from contractionRL.agents.skrl.sdlqr import SDLQRAgent, LQRAgent, SDLQRCfg, LQRCfg
        from skrl.trainers.torch import SequentialTrainer
        import dataclasses
        from skrl.trainers.torch.base import TrainerCfg

        cfg_cls = LQRCfg if lqr else SDLQRCfg
        agent_cls = LQRAgent if lqr else SDLQRAgent

        agent = agent_cls(
            cfg=cfg_cls(**_filter_cfg_fields(agent_cfg, cfg_cls, context="ContractionRunner agent")),
            models={},
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
            angle_idx=angle_idx,
        )
        # SequentialTrainer for eval — no gradient updates
        tcfg = dataclasses.asdict(TrainerCfg(**_filter_cfg_fields({
            "timesteps": trainer_cfg.get("timesteps", 10000),
            "close_environment_at_exit": False,
        }, TrainerCfg, context="ContractionRunner trainer")))
        # SequentialTrainer.__init__ already calls agent.init() (and thus
        # wandb.init(), if enabled) synchronously — unlike the other
        # algorithms' project-owned trainers, there's no later train()-time
        # hook to defer to, so push the raw config right here.
        trainer = SequentialTrainer(cfg=tcfg, env=env, agents=agent)
        from contractionRL.agents.skrl.contraction_metrics import log_raw_config
        log_raw_config(raw_cfg_snapshot)
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "lqr" if lqr else "sdlqr"

    def _setup_c2rl(self, env, device, obs_space, state_space, act_space,
                    agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                    base_algorithm: str = "PPO", angle_idx=None, raw_cfg_snapshot=None,
                    cm_cfg=None, cmg_cfg=None, empirical_dynamics_cfg=None):
        """Build a single-policy C2RLAgent. Always builds a CMG network
        (MetricModel), synthesized offline before Phase B and then frozen
        (Tsukamoto NCM) — ``cmg_method`` (yaml `cm:` block) selects how: "ccm"
        (default) trains it directly via C1/C2 loss minimization; "cvstem" solves
        a per-state SDP and MSE-regresses the network onto the solutions. The CMG
        is always built as a ``BoundedCCM_Generator`` (``constrain_eigenvalues``
        forced ``True`` here, regardless of yaml) — see c2rl.py's module
        docstring for why.

        ``cm_cfg``/``cmg_cfg``/``empirical_dynamics_cfg`` are the yaml's
        top-level ``cm:``/``cmg:``/``empirical_dynamics:`` blocks — merged into
        ``agent_cfg`` here so C2RLAgent (which reads its full raw cfg dict as
        one flat namespace, see ``C2RLPPOCfg``/``C2RLSACCfg``) doesn't need to
        know they came from separate yaml sections. Kept out of ``agent:`` in
        the yaml purely so the contraction-metric / CMG-synthesis /
        empirical-dynamics knobs each read as their own config unit.
        """
        angle_idx = list(angle_idx or [])
        from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer, C2RLTrainerCfg

        agent_cfg = {**agent_cfg, **(cm_cfg or {}), **(cmg_cfg or {}), **(empirical_dynamics_cfg or {})}

        base_algorithm = base_algorithm.upper()

        # Phase B's memory: PPO's is exactly one rollout epoch (sized by
        # agent.rollouts); SAC's is a persistent replay buffer, sized via the
        # yaml's memory.memory_size (same convention as C3M — see C2RLAgent).
        mem_size = memory_cfg.get("memory_size", -1)
        rollouts = agent_cfg.get("rollouts", 300)
        if base_algorithm == "PPO" and mem_size not in (-1, rollouts):
            raise ValueError(
                f"[C2RL] memory.memory_size={mem_size} must be -1 (automatically "
                f"determined, same as agent:rollouts) or exactly match agent.rollouts "
                f"({rollouts}) for PPO."
            )
        agent_cfg["memory_size"] = mem_size

        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)
        models = {}

        # Always build a CMG network (MetricModel), synthesized offline (Phase A
        # — see C2RLAgent.synthesize_cmg) and frozen before Phase B. Forced
        # constrain_eigenvalues=True regardless of yaml — see c2rl.py's module
        # docstring for why (C2RLAgent also guards this at construction time).
        from contractionRL.agents.skrl.models import MetricModel
        cmg_net = models_cfg.get("cmg", {}).get("network", [{}])
        cmg_hd = cmg_net[0].get("layers", [256, 256]) if cmg_net else [256, 256]
        cmg_act = cmg_net[0].get("activations", "tanh") if cmg_net else "tanh"
        cmg_model = MetricModel(
            obs_space, act_space, device, hidden_dim=cmg_hd, activation=cmg_act,
            x_dim=x_dim, angle_idx=angle_idx, constrain_eigenvalues=True, w_lb=w_lb, w_ub=w_ub,
        )
        models["cmg"] = cmg_model

        # NeuralDynamics (learned ẋ = f(x) + B(x)·u) — feeds Phase A's CMG
        # synthesis. Built only when learning dynamics (use_empirical_dynamics=
        # True); Isaac envs' sole f/B source, classic envs default to analytical.
        if "dynamics" in models_cfg and agent_cfg.get("use_empirical_dynamics", False):
            from contractionRL.agents.skrl.nn_modules import NeuralDynamics
            dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
            dyn_hd = dyn_net.get("layers", [256, 256])
            dyn_act = dyn_net.get("activations", "relu")
            models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx)

        # Phase B deployed policy + critic — control/mlp backbone dispatch, same
        # as C3M's. Value (PPO) / twin-Q + targets (SAC) see the SAME
        # angle-bearing x/xref blocks as the policy and embed them identically.
        models["policy"] = _build_gaussian_policy(models_cfg, "policy", obs_space, state_space, act_space,
                                                  device, x_dim, angle_idx, agent_class=f"C2RL-{base_algorithm}")
        models.update(_build_critics(models_cfg, "critic", base_algorithm, obs_space, act_space,
                                     device, x_dim, angle_idx, key_prefix=""))

        # Pass the RAW agent_cfg dict (not a pre-parsed C2RLPPOCfg/C2RLSACCfg) —
        # C2RLAgent keeps the full dict in self._raw_cfg, which make_base_rl_cfg()
        # later filters against PPO_CFG/SAC_CFG. Pre-parsing first would silently
        # drop any PPO/SAC-specific field not ALSO declared on the matching C2RL
        # cfg dataclass (this bit us for base_algorithm before it became an
        # explicit constructor kwarg — see C2RLAgent.__init__).
        agent = C2RLAgent(
            cfg=agent_cfg,
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            base_algorithm=base_algorithm,
            x_dim=x_dim,
            u_dim=u_dim,
            num_envs=env.num_envs,
            angle_idx=angle_idx,
        )
        trainer_cfg.setdefault("timesteps", 300000)
        trainer = C2RLSkrlTrainer(
            cfg=C2RLTrainerCfg(**_filter_cfg_fields(trainer_cfg, C2RLTrainerCfg, context="ContractionRunner trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._trainer._wandb_raw_cfg = raw_cfg_snapshot
        self._env = env
        self._mode = "c2rl"

        # Single deployed (Phase B) policy. The PathTracking figure's theoretical
        # bound inflates by 1/(1-discount_factor); the plotted certificate reads
        # the CMG's per-state metric bounds (cmg_bounds_fn), falling back to the
        # static [1/w_lb, 1/w_ub] only if the env doesn't wire cmg_bounds_fn up.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=agent_cfg.get("discount_factor", 0.99),
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(cmg_model, w_lb),
            )

    # ── public interface ────────────────────────────────────────────────── #

    @property
    def agent(self):
        if self._mode == "skrl":
            return self._runner.agent
        return self._agent

    @property
    def trainer(self):
        if self._mode == "skrl":
            return getattr(self._runner, "_trainer", None) or getattr(self._runner, "trainer", None)
        return self._trainer

    def load(self, path: str) -> None:
        if self._mode == "skrl":
            self._runner.agent.load(path)
        else:
            self._agent.load(path)

    def run(self):
        if self._mode == "skrl":
            return self._runner.run()
        return self._trainer.train()

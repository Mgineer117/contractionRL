"""Generic post-construction patches for skrl PPO/SAC agent instances.

Shared between the standalone PPO/SAC path (scripts/skrl/train.py, which
patches ``runner.agent`` directly) and C2RL (c2rl.py, which patches its inner
PPO/SAC sub-agent — C2RL's own outer agent has no ``.policy``/``.scheduler``/
etc. attributes for these to find). Each patch inspects the given agent for the
attributes it needs and no-ops if they're absent, so it's safe to call
unconditionally on any agent.

Every call site patches BEFORE ``agent.init()``, which is where skrl allocates
the memory tensors and the trainer, in turn, only calls at ``train()`` time.
``patch_caps_regularizer`` relies on that ordering.
"""

from __future__ import annotations

import math

import torch


def patch_kl_logging(agent) -> None:
    """Log per-epoch approximate KL divergence to 'Policy / KL divergence'.

    skrl's PPO computes KL every epoch to drive KLAdaptiveLR but never records
    it, so early-stop events (kl_threshold) are invisible in tensorboard/wandb
    even though they silently truncate — and thus deflate — the averaged
    Loss/Policy loss, Loss/Value loss, Loss/Entropy loss for that update
    (skrl divides by the full learning_epochs*mini_batches regardless of how
    many minibatches actually ran before the break). No-ops for agents without
    a KLAdaptiveLR scheduler (SAC, PPO with scheduler=null).
    """
    import skrl.resources.schedulers.torch as _sched

    scheduler = getattr(agent, "scheduler", None)
    if not isinstance(scheduler, _sched.KLAdaptiveLR):
        return

    _orig_step = scheduler.step

    def _step(kl=None, *, epoch=None):
        if kl is not None:
            agent.track_data("Policy / KL divergence", float(kl))
        _orig_step(kl, epoch=epoch)

    scheduler.step = _step


def patch_sac_entropy_clamp(agent, min_log_alpha: float = -5.0, max_log_alpha: float = 2.0) -> None:
    """Clamp log_entropy_coefficient in-place after every entropy optimizer step.

    skrl's SAC applies grad_norm_clip to the policy and critic optimizers but
    NOT to entropy_optimizer, and _entropy_coefficient = exp(log_entropy_coefficient)
    is exponentiated with no bound. A noisy/undertrained critic can push this
    single scalar's gradient large, and exponentiation turns even a moderate
    excursion into a runaway entropy coefficient that then dominates both the
    critic target and the policy loss — a textbook SAC divergence mechanism.
    Bounds exp(log_alpha) to roughly [0.0067, 7.39], mirroring the clip_log_std
    bounds skrl already applies to GaussianMixin policies elsewhere. No-op for
    agents without learn_entropy (PPO, SAC with learn_entropy=False).
    """
    entropy_optimizer = getattr(agent, "entropy_optimizer", None)
    log_alpha = getattr(agent, "log_entropy_coefficient", None)
    if entropy_optimizer is None or log_alpha is None:
        return

    _orig_step = entropy_optimizer.step

    def _step(*args, **kwargs):
        result = _orig_step(*args, **kwargs)
        with torch.no_grad():
            log_alpha.clamp_(min_log_alpha, max_log_alpha)
        return result

    entropy_optimizer.step = _step


def patch_ppo_std_annealing(agent, std_dev_annealing: bool, kwargs: dict | None = None) -> None:
    """Adds manual standard deviation annealing to SKRL's PPO policy.

    If `std_dev_annealing` is True, this disables the entropy loss entirely
    (setting entropy_loss_scale to 0.0) and anneals the policy's
    log_std_parameter from its initial value down to `final_log_std` over the
    total training timesteps, following the chosen schedule.

    `std_dev_annealing` is not a yaml flag: callers auto-derive it. train.py's
    standalone PPO/SAC route still derives it from whether the policy's
    backbone is one of ``runner.CONTROL_BACKBONES`` (``"control"``/
    ``"contraction"``). c2rl.py instead derives it from
    ``self._base_algorithm == "PPO"`` — i.e. always on for PPO regardless of
    backbone, and always off for SAC (which learns log_std via its own
    automatic entropy tuning; see ``SquashedCLActorModel``'s docstring in
    models.py). Either way, this function freezes ``log_std_parameter``
    (``requires_grad=False``) itself and no-ops if the policy has no such
    attribute (e.g. a state-dependent log_std head, as in
    ``mlp-squashed``). Only the schedule itself is yaml-configurable::

        agent:
          std_dev_annealing_kwargs:
            schedule: exponential   # linear | exponential | cosine
            final_log_std: -2.3     # target log_std (std ~= 0.1)
            power: 5.0              # exponential schedule only: progress**power

    Schedules (p = timestep/timesteps in [0, 1]):
      linear:       log_std = init + p * (final - init)
      exponential:  log_std = init + p**power * (final - init)  — slow early,
                    fast late; keeps exploration wide for most of training
      cosine:       log_std = init + (1 - cos(pi*p))/2 * (final - init)
    """
    if not std_dev_annealing:
        return
    kwargs = dict(kwargs or {})
    schedule = str(kwargs.pop("schedule", "linear")).lower()
    final_log_std = float(kwargs.pop("final_log_std", -2.0))
    power = float(kwargs.pop("power", 5.0))
    if kwargs:
        from skrl import logger
        logger.warning(f"std_dev_annealing_kwargs: ignoring unknown keys {sorted(kwargs)}")

    # Ignore entropy: annealing and entropy bonus fight each other
    if hasattr(agent, "_cfg") and isinstance(agent._cfg, dict):
        agent._cfg["entropy_loss_scale"] = 0.0
    if hasattr(agent, "cfg"):
        if isinstance(agent.cfg, dict):
            agent.cfg["entropy_loss_scale"] = 0.0
        else:
            setattr(agent.cfg, "entropy_loss_scale", 0.0)

    if not hasattr(agent, "policy") or not hasattr(agent.policy, "log_std_parameter"):
        return

    # Disable gradients on log_std_parameter because we update it manually
    agent.policy.log_std_parameter.requires_grad_(False)

    initial_log_std = agent.policy.log_std_parameter.mean().item()

    def _ratio(p: float) -> float:
        if schedule == "exponential":
            return p ** power
        if schedule == "cosine":
            return (1.0 - math.cos(math.pi * p)) / 2.0
        return p  # linear

    _orig_post = agent.post_interaction

    def _annealed_post(*, timestep: int, timesteps: int) -> None:
        progress = min(1.0, max(0.0, timestep / max(1, timesteps)))
        current_log_std = initial_log_std + _ratio(progress) * (final_log_std - initial_log_std)
        agent.policy.log_std_parameter.data.fill_(current_log_std)
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _annealed_post


def _resolve_x_dim(policy) -> int | None:
    """Length of the leading ``x`` block of a ``[x, xref, uref]`` observation,
    or ``None`` for a policy over a flat observation with no such split.

    Every path-tracking backbone knows this number, but they spell it
    differently and none of the spellings is universal: the residual/squashed
    MLPs store it as ``_x_dim`` (``None`` when they were built for a
    vel-tracking layout), while ``CLActorModel``/``SquashedCLActorModel`` keep
    it only on the ``cl_actor`` submodule they delegate to. Missing that last
    case silently returned ``None`` for exactly the two backbones whose control
    law makes the x-only restriction load-bearing — see
    ``patch_caps_regularizer``'s note on uref pass-through.
    """
    for owner, attr in ((policy, "x_dim"), (policy, "_x_dim"),
                        (getattr(policy, "cl_actor", None), "x_dim")):
        x_dim = getattr(owner, attr, None) if owner is not None else None
        if x_dim:
            return int(x_dim)
    return None


def patch_caps_regularizer(
    agent,
    *,
    temporal_scale: float = 0.0,
    spatial_scale: float = 0.0,
    spatial_std: float = 0.05,
    batch_size: int = 1024,
) -> None:
    """Add CAPS action-smoothness regularization to the POLICY LOSS.

    CAPS (Mysore et al. 2021, *Regularizing Action Policies for Smooth
    Control*) penalizes two distinct kinds of non-smoothness in the policy MEAN
    (never the sampled action — exploration noise is not what we want to
    suppress, and its gradient would fight std annealing / SAC entropy tuning)::

        L_T = temporal_scale * || pi(s_t)  - pi(s_{t+1}) ||^2   (chatter in time)
        L_S = spatial_scale  * || pi(s)    - pi(s_bar)   ||^2,  s_bar ~ N(s, sigma^2)
                                                               (high state gain)

    Why a LOSS term rather than a reward term. A ``-||u_t - u_{t-1}||^2`` reward
    makes the return depend on u_{t-1}, which is not in the observation — that
    is a different (partially observed) MDP, and the critic can only model the
    extra term as noise. Putting u_{t-1} INTO the observation is also not free
    here: it changes obs_dim and so trips the ``obs_dim == 2*x_dim + u_dim``
    layout assertion the CLActor backbones rely on (see runner.py). And the
    spatial term is not expressible as a reward at all — the environment has no
    handle on pi to evaluate it at a second, perturbed state. As a policy-loss
    term CAPS touches none of that: same MDP, same observation, same dynamics,
    so the offline CV-STEM synthesis and the contraction certificate it produces
    remain valid.

    WHERE THE STATES COME FROM — the agent's OWN memory, so CAPS inherits each
    algorithm's data distribution instead of imposing a third one: for SAC the
    persistent replay buffer its critic is trained on, for PPO the rollout
    memory, whose ``memory_size`` is ``rollouts`` — exactly the on-policy batch
    the policy gradient uses and nothing older.

    The (s_t, s_t+1) pairing needs a ``next_observations`` column, which SAC
    allocates itself (its target computation needs it) but PPO never does. This
    patch allocates it unconditionally: ``create_tensor`` is idempotent for a
    matching name/size/dtype, and since every call site patches BEFORE
    ``agent.init()`` (see the module docstring) the memory is still empty either
    way, so there is no per-algorithm branch to get wrong. It is then filled
    through the SAME ``add_samples`` call the agent already makes — injected via
    a wrapper rather than a second call, so the memory index still advances
    exactly once per transition (and for SAC the injected value is simply the
    one the agent was passing anyway).

    The pairing is read with a separate ``sample_by_index``, never by extending
    ``_tensors_names``: both agents unpack ``memory.sample(...)`` into a
    fixed-arity tuple, and one extra name there would raise.

    AUTORESET. A transition whose ``next_observations`` straddles an episode
    boundary is not a real (s_t, s_{t+1}) pair, and asking the policy to be
    smooth across a reset is asking it to be smooth across a discontinuity it
    cannot control. Both env families here reset IN PLACE at the terminating
    step (classic ``env_base.step`` calls ``reset_idx`` and returns the
    post-reset observation, keeping the true final one in
    ``info["final_observation"]``; Isaac Lab does the same), so the bogus pair
    is the one flagged ``terminated | truncated``. The stored ``caps_valid``
    mask ALSO drops the step immediately after a done, which is what a
    next-step-autoreset env would flag instead — so the mask is correct under
    either convention without having to detect which is in play. The cost is at
    most two transitions per episode (<1% at the ~300-step episodes here);
    splicing ``info["final_observation"]`` back in would recover them, but for
    SAC that would mean rewriting the ``next_observations`` its critic target
    bootstraps from, which is not this patch's business.

    Where the loss is injected. Both skrl PPO and SAC route every policy
    backward through ``self.scaler.scale(<loss>).backward()``, immediately after
    the one grad-enabled ``policy.act(..., role="policy")`` of that update. This
    arms on that act() and consumes on the next scale(), so the CAPS gradient
    lands in the same backward as the policy loss — and therefore INSIDE the
    subsequent ``grad_norm_clip``, which a separate ``.backward()`` before
    ``optimizer.step()`` would have escaped. SAC's critic-loss scale() call
    precedes the policy act() and its entropy-loss call follows the consume, so
    neither is affected. PPO's ``kl_threshold`` early stop breaks out between
    the act() and the scale(), leaving the flag armed; the next scale() to run
    is the next update's policy scale, so the invariant that survives is the one
    that matters — at most one CAPS term per policy backward, never on a critic
    or entropy one.

    Which observation dimensions get perturbed (spatial term). For the
    ``control``/``control-squashed`` backbones the observation is
    ``[x, xref, uref]`` and the control law is ``u = uref + feedback(x - xref)``,
    so perturbing ``uref`` shifts the output by exactly that perturbation. That
    is pure feedforward pass-through, structurally required and NOT reducible by
    the network — penalizing it would put an irreducible floor on L_S and push
    the policy to suppress its own uref term. Only the leading ``x`` block is
    perturbed when the policy exposes an ``x_dim``; otherwise (a flat
    observation with no such split, e.g. velocity tracking) the whole
    observation is. ``_resolve_x_dim`` covers every backbone's spelling of it.

    ``spatial_std`` is in RAW observation units: the perturbation is added to
    the stored observation and the preprocessor is applied afterwards, so
    turning ``use_state_norm`` on would shrink the perturbation the policy
    actually sees by that dimension's running std rather than leaving sigma in
    normalized units. Inert today (state normalization is off in every config
    and train.py force-disables it), but the scale is per-dimension either way —
    a state component with a much wider range than the rest is effectively
    under-regularized at a single scalar sigma.

    No-op unless at least one scale is positive, and no-op for agents without
    ``policy``/``scaler``/``memory`` (C2RL's outer agent — it patches its inner
    PPO/SAC sub-agent directly, see c2rl.py).
    """
    temporal_scale = float(temporal_scale)
    spatial_scale = float(spatial_scale)
    if temporal_scale <= 0.0 and spatial_scale <= 0.0:
        return
    policy = getattr(agent, "policy", None)
    scaler = getattr(agent, "scaler", None)
    memory = getattr(agent, "memory", None)
    if policy is None or scaler is None or memory is None:
        return

    device = getattr(memory, "device", None) or next(policy.parameters()).device
    x_dim = _resolve_x_dim(policy)

    # skrl's own "no preprocessor" fallback is _empty_preprocessor, which already
    # swallows the train= kwarg — so both branches are callable the same way.
    def _identity(t, **_kwargs):
        return t

    obs_pre = getattr(agent, "_observation_preprocessor", None) or _identity
    state_pre = getattr(agent, "_state_preprocessor", None) or _identity
    _orig_act = policy.act

    # ── make the (s_t, s_t+1) pairing readable from the agent's own memory ──
    # create_tensor is idempotent for a matching name/size/dtype, so this is a
    # no-op for whichever columns the agent allocates for itself in init().
    # next_states returns False (nothing allocated) when state_space is None,
    # which is the norm here; sample_by_index then yields None for that name and
    # the policy gets states=None, exactly as it does during a normal update.
    memory.create_tensor(name="next_observations", size=agent.observation_space, dtype=torch.float32)
    memory.create_tensor(name="next_states", size=agent.state_space, dtype=torch.float32)
    memory.create_tensor(name="caps_valid", size=1, dtype=torch.bool)

    _names = ["observations", "next_observations", "caps_valid", "states", "next_states"]

    _prev_done = torch.zeros((memory.num_envs, 1), dtype=torch.bool, device=device)
    _pending: dict = {}

    # Injected into the agent's existing add_samples call rather than added by a
    # second one: add_samples advances memory_index itself, so calling it twice
    # per transition would desynchronize our columns from the agent's.
    _orig_add_samples = memory.add_samples

    def _add_samples(**tensors):
        if _pending:
            tensors.update(_pending)
            _pending.clear()
        return _orig_add_samples(**tensors)

    memory.add_samples = _add_samples

    _orig_record = agent.record_transition

    def _record(*, observations, states, actions, rewards, next_observations, next_states,
                terminated, truncated, infos, timestep, timesteps):
        done = terminated | truncated
        _pending["caps_valid"] = ~done & ~_prev_done   # see AUTORESET in the docstring
        _pending["next_observations"] = next_observations
        _pending["next_states"] = next_states
        _prev_done.copy_(done)
        return _orig_record(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    agent.record_transition = _record

    # ── the CAPS loss itself ────────────────────────────────────────────────
    # _orig_act, not policy.act: the arming wrapper installed below must not see
    # these forwards, and calling the pre-patch method is what keeps it from
    # doing so — there is no re-entrancy to guard against.
    def _policy_mean(obs, states):
        inputs = {"observations": obs_pre(obs, train=False),
                  "states": states if states is None else state_pre(states, train=False)}
        _, outputs = _orig_act(inputs, role="policy")
        return outputs["mean_actions"]

    def _caps_loss():
        size = len(memory)
        if size == 0:
            return None
        # sample_by_index rather than sample(): the latter also overwrites
        # memory.sampling_indexes, which belongs to the agent's own update.
        indexes = torch.randint(0, size, (min(batch_size, size),), device=device)
        obs, next_obs, valid, states, next_states = memory.sample_by_index(
            names=_names, indexes=indexes
        )[0]

        mean = _policy_mean(obs, states)
        loss = None

        if temporal_scale > 0.0 and bool(valid.any()):
            next_mean = _policy_mean(next_obs, next_states)
            # Masked mean over surviving pairs, not sum/N — otherwise the
            # effective coefficient would silently shrink with the episode-
            # boundary fraction (and swing with termination rate as the
            # policy improves).
            sq = ((mean - next_mean) ** 2).sum(dim=-1, keepdim=True)
            l_t = (sq * valid).sum() / valid.sum().clamp(min=1)
            agent.track_data("Loss / CAPS temporal", l_t.item())
            loss = temporal_scale * l_t

        if spatial_scale > 0.0:
            noise = torch.randn_like(obs) * spatial_std
            if x_dim:
                noise[:, x_dim:] = 0.0  # see docstring: never perturb xref/uref
            bar_mean = _policy_mean(obs + noise, states)
            l_s = ((mean - bar_mean) ** 2).sum(dim=-1).mean()
            agent.track_data("Loss / CAPS spatial", l_s.item())
            loss = spatial_scale * l_s if loss is None else loss + spatial_scale * l_s

        return loss

    # ── arm on the update's policy forward, consume on the next scale() ─────
    armed = False

    def _act(inputs, *, role: str = ""):
        nonlocal armed
        out = _orig_act(inputs, role=role)
        if role == "policy" and torch.is_grad_enabled():
            armed = True
        return out

    policy.act = _act

    _orig_scale = scaler.scale

    def _scale(loss):
        nonlocal armed
        if armed:
            armed = False
            caps = _caps_loss()
            if caps is not None:
                loss = loss + caps
        return _orig_scale(loss)

    scaler.scale = _scale


def patch_algo_namespace(agent, algo_name: str) -> None:
    """Namespace this agent's own track_data() keys under ``algo_name``.

    C3M/C2RL already log their internals as "{tab} / {ALGO}/{metric}" (e.g.
    "Loss / C3M/dynamics/mse", see c3m.py/c2rl.py) so runs from different
    algorithms don't collide on the same wandb Loss/Policy/Learning panels.
    Standalone PPO/SAC never got the same treatment — skrl's own track_data
    calls (ppo.py/sac.py: "Loss / Policy loss", "Q-network / Q1 (max)", ...)
    and ours (patch_kl_logging's "Policy / KL divergence") land un-namespaced.
    This rewrites "{category} / {name}" -> "{category} / {algo_name}/{name}"
    for every track_data() call, matching that convention.

    "Reward / Total reward (mean/max/min)" is deliberately left untouched:
    skrl's base Agent writes that EXACT key straight into tracking_data
    (base.py, bypassing track_data()) to pick the best_agent.pt checkpoint,
    and patch_auc_checkpoint below injects into that same key via
    track_data() to redirect the checkpoint metric to contraction_score/AUC —
    renaming it here would silently break both. Stability/*, Episode/*,
    Info/* (also written directly into tracking_data, not through
    track_data()) are unaffected regardless.
    """
    orig_track_data = agent.track_data

    def _wrapped(tag, value):
        if " / " in tag and not tag.startswith("Reward / "):
            category, name = tag.split(" / ", 1)
            if not name.startswith(f"{algo_name}/"):
                tag = f"{category} / {algo_name}/{name}"
        orig_track_data(tag, value)

    agent.track_data = _wrapped


def best_metric_for(algorithm: str) -> str:
    """Which stability metric an algorithm's best_agent.pt should track.

    The two hard per-step contraction controllers that certify a metric
    directly — ``c3m`` and ``cvstem-lqr`` — pick their best checkpoint by
    ``contraction_score`` (λ/overshoot; higher is better). Every learned RL
    policy (``ppo``/``sac``/``c2rl-ppo``/``c2rl-sac``) and the remaining
    analytical baselines (``lqr``/``sdlqr``) pick by AUC (lower is better).

    Accepts either the hyphen (Isaac ``_alg``: ``"cvstem-lqr"``/``"c2rl-ppo"``)
    or underscore (classic ``algorithm``: ``"cvstem_lqr"``/``"c2rl_ppo"``)
    spelling — both normalize to hyphens before the set check.
    """
    norm = str(algorithm).lower().replace("_", "-")
    return "contraction_score" if norm in ("c3m", "cvstem-lqr") else "auc"


def patch_auc_checkpoint(agent, metric: str = "auc") -> None:
    """Override agent.post_interaction to pick best_agent.pt by ``metric``.

    skrl natively saves best_agent.pt by tracking the highest
    'Reward / Total reward (mean)'. This redirects that checkpoint rule to a
    stability metric by injecting into the SAME key (see patch_algo_namespace's
    docstring for why 'Reward / Total reward (mean)' is the injection target —
    base skrl reads exactly that key in post_interaction to choose the best).

    ``metric="contraction_score"`` (c3m/cvstem-lqr): use
    ``Stability/contraction_score_mean`` — higher is better, so it plugs into
    skrl's "maximize reward" rule with no sign flip. ``metric="auc"``
    (everything else): use ``-Stability/auc_mean`` (or ``-Episode/auc`` for
    velocity-tracking envs) — lower AUC = better, hence the flip. If the chosen
    metric isn't logged in a given step, the base reward is left untouched (the
    c3m↔cvstem vs. AUC split is explicit — no cross-fallback). See
    ``best_metric_for`` for the algorithm→metric mapping callers use.
    """
    _orig_post = getattr(agent, "post_interaction", None)
    if _orig_post is None:
        return

    def _metric_post(*, timestep: int, timesteps: int) -> None:
        # Stability/contraction_score and Stability/auc are logged with a
        # "_mean" suffix (see contraction_metrics.py's track_stability_summary);
        # Episode/auc (velocity-tracking envs) has no such suffix — it's a
        # single value from a different, unrelated logging path.
        if metric == "contraction_score":
            score_list = agent.tracking_data.get("Stability/contraction_score_mean")
            if score_list:
                agent.track_data("Reward / Total reward (mean)", score_list[-1])
        else:
            # Prioritize Stability/auc (path tracking) over Episode/auc (velocity tracking)
            score_list = agent.tracking_data.get("Stability/auc_mean") or agent.tracking_data.get("Episode/auc")
            if score_list:
                agent.track_data("Reward / Total reward (mean)", -score_list[-1])
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _metric_post

"""Generic post-construction patches for skrl PPO/SAC agent instances.

Used by train.py (patches ``runner.agent``) and c2rl.py (patches its inner
PPO/SAC sub-agent; C2RL's outer agent has no ``.policy``/``.scheduler``). Each
patch no-ops if the agent lacks what it needs, so calling any of them on any
agent is safe.

Invariant: every call site patches before ``agent.init()`` (where skrl
allocates memory tensors).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


def patch_terminal_observations(agent) -> None:
    """Store/ bootstrap the true endpoint while rollouts receive autoreset states."""
    if getattr(agent, "memory", None) is None or getattr(agent, "_terminal_observations_patched", False):
        return
    original = agent.record_transition

    def record_transition(*, next_observations, terminated, truncated, infos, **kwargs):
        truncated = truncated & ~terminated
        final = infos.get("final_observation") if isinstance(infos, dict) else None
        if final is not None:
            mask = torch.as_tensor(infos.get("_final_observation", terminated | truncated),
                                   device=next_observations.device).reshape(-1).bool()
            if mask.any():
                if isinstance(final, dict) and "policy" in final:
                    final = final["policy"]
                if isinstance(final, dict):
                    final = torch.cat([torch.as_tensor(final[key], device=next_observations.device).reshape(len(final[key]), -1)
                                       for key in sorted(final)], -1)
                final = torch.as_tensor(final, dtype=next_observations.dtype, device=next_observations.device)
                next_observations = next_observations.clone()
                next_observations[mask] = final[mask] if len(final) == len(mask) else final
        return original(next_observations=next_observations, terminated=terminated,
                        truncated=truncated, infos=infos, **kwargs)

    agent.record_transition = record_transition
    agent._terminal_observations_patched = True


def patch_kl_logging(agent) -> None:
    """Log PPO KL and safeguard stock PPO updates against sampled overshoot.

    skrl's PPO computes KL to drive KLAdaptiveLR but never records it, so
    kl_threshold early stops are invisible — even though they deflate that
    update's averaged losses (skrl divides by the full
    learning_epochs*mini_batches regardless of where it broke).

    ``KLAdaptiveLR.step(kl)`` is the only place the KL escapes skrl's
    ``_update`` — it is a local there and nothing else reads it (see skrl
    ppo.py: ``kl_divergences`` is built in the minibatch loop and consumed only
    under ``isinstance(self.scheduler, KLAdaptiveLR)``). So the metric exists
    only if that scheduler is attached.

    Rather than warn and drop the curve, this now attaches a KLAdaptiveLR that
    cannot move the learning rate — ``min_lr = max_lr = the optimizer's current
    lr`` — whenever a PPO-family agent has none. The scheduler clamps its own
    output to [min_lr, max_lr], so training is numerically identical to running
    without a scheduler while the KL hook fires every update.

    This is deliberately done in code rather than per-yaml: the metric is a
    diagnostic every PPO/C2RL-PPO run should have, and making it depend on
    remembering a config key is how it went missing in the first place
    (``learning_rate_scheduler: null`` silently dropped it).

    Stock PPO also gets a full-rollout post-update diagnostic and a rollback
    guard when kl_threshold > 0; custom algorithm subclasses keep their loss.
    """
    import skrl.resources.schedulers.torch as _sched

    if getattr(agent, "_kl_logging_patched", False):
        return
    scheduler = getattr(agent, "scheduler", None)
    if isinstance(scheduler, _sched.KLAdaptiveLR):
        pass
    elif scheduler is None and hasattr(agent, "scaler"):
        # A PPO/SAC-family agent: it runs the update loop that computes the KL.
        # Attach a pinned KLAdaptiveLR so the hook exists. SAC-family agents have
        # no `optimizer` of this shape and never build kl_divergences, so they
        # fall through to the no-op below rather than getting a dead scheduler.
        opt = getattr(agent, "optimizer", None)
        if opt is None or not getattr(opt, "param_groups", None):
            return
        lr = float(opt.param_groups[0]["lr"])
        try:
            scheduler = _sched.KLAdaptiveLR(opt, min_lr=lr, max_lr=lr,
                                            kl_threshold=getattr(agent.cfg, "kl_threshold", 0.008) or 0.008)
        except Exception as exc:  # noqa: BLE001 — never break training for a metric
            print(f"[patch] could not attach a pinned KLAdaptiveLR to "
                  f"{type(agent).__name__} ({exc}); 'Policy / KL divergence' "
                  f"will not be logged.")
            return
        agent.scheduler = scheduler
        print(f"[patch] {type(agent).__name__}: attached a PINNED KLAdaptiveLR "
              f"(min_lr = max_lr = {lr:g}) purely so 'Policy / KL divergence' is "
              f"logged; the learning rate cannot move.")
    else:
        # A wrapper agent (C2RL's outer agent has no .policy/.scaler of its own)
        # — it never computes a KL, so there is nothing to miss. Both train_utils
        # and C2RL call this, the latter on the inner sub-agent that does; warning
        # here would fire on every C2RL run and train everyone to ignore the real one.
        return

    _orig_step = scheduler.step

    def _step(kl=None, *, epoch=None):
        if kl is not None:
            agent.track_data("Policy / KL divergence", float(kl))
        _orig_step(kl, epoch=epoch)

    scheduler.step = _step
    agent._kl_logging_patched = True
    from skrl.agents.torch.ppo import PPO
    if type(agent) is PPO:
        patch_kl_logging_post_update(agent)
        patch_ppo_kl_guard(agent)


def patch_ppo_kl_guard(agent) -> None:
    """Retry stock PPO updates exceeding the sampled KL budget, restoring Adam too.

    skrl checks KL before a minibatch step. The step itself can overshoot, and
    the next epoch's scheduler cannot undo it. Validate both the epoch KLs and
    the final full-rollout estimate against cfg.kl_threshold. This constrains
    the sampled rollout, not every possible observation or the exact KL.
    """
    import copy
    from skrl.agents.torch.ppo import PPO

    # Released ContractionPPO and other subclasses retain their own update rule.
    if type(agent) is not PPO or agent.cfg.kl_threshold <= 0 or getattr(agent, "_kl_guard_patched", False):
        return
    original_update = agent.update

    def guarded_update(*, timestep: int, timesteps: int) -> None:
        modules = {key: copy.deepcopy(module.state_dict()) for key, module in agent.checkpoint_modules.items()}
        scheduler = copy.deepcopy(agent.scheduler.state_dict())
        scaler = copy.deepcopy(agent.scaler.state_dict())
        memory = {key: agent.memory.get_tensor_by_name(key).clone() for key in ("values", "returns", "advantages")}
        tracking = copy.deepcopy(agent.tracking_data)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(agent.device) if agent.device.type == "cuda" else None
        initial_lr = [group["lr"] for group in agent.optimizer.param_groups]
        floor = agent.scheduler.min_lr
        rejected_kl = 0.0

        def restore():
            for key, state in modules.items():
                agent.checkpoint_modules[key].load_state_dict(copy.deepcopy(state))
            agent.scheduler.load_state_dict(copy.deepcopy(scheduler))
            agent.scaler.load_state_dict(scaler)
            for key, value in memory.items():
                agent.memory.set_tensor_by_name(key, value)
            agent.tracking_data.clear()
            agent.tracking_data.update(copy.deepcopy(tracking))
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, agent.device)

        def set_lr(attempt):
            rates = [max(floor, rate * .5 ** attempt) for rate in initial_lr]
            for group, rate in zip(agent.optimizer.param_groups, rates):
                group["lr"] = rate
            agent.scheduler._last_lr = rates
            return rates

        for attempt in range(8):
            if attempt:
                restore()
                set_lr(attempt)
            try:
                original_update(timestep=timestep, timesteps=timesteps)
            except Exception:
                restore()
                raise
            measured = {}
            for key, values in agent.tracking_data.items():
                leaf = key.rsplit("/", 1)[-1].strip()
                if leaf in ("KL divergence", "KL divergence (post-update)"):
                    measured[leaf] = values[len(tracking.get(key, [])):]
            if not all(measured.get(key) for key in ("KL divergence", "KL divergence (post-update)")):
                restore()
                raise RuntimeError("PPO KL guard requires epoch and post-update KL measurements")
            divergences = [float(value) for values in measured.values() for value in values]
            candidate_kl = max(divergences) if all(math.isfinite(value) for value in divergences) else float("inf")
            from skrl import config
            if config.torch.is_distributed:
                shared_kl = torch.tensor(candidate_kl, dtype=torch.float64, device=agent.device)
                torch.distributed.all_reduce(shared_kl, op=torch.distributed.ReduceOp.MAX)
                candidate_kl = shared_kl.item()
            if candidate_kl <= agent.cfg.kl_threshold:
                agent.track_data("Learning / KL guard retries", attempt)
                agent.track_data("Learning / KL guard skipped update", 0.)
                agent.track_data("Learning / KL guard rejected KL", rejected_kl)
                return
            rejected_kl = max(rejected_kl, candidate_kl)
            if all(rate * .5 ** attempt <= floor for rate in initial_lr):
                break

        # No acceptable step: retain the rollout policy and all optimizer state.
        restore()
        set_lr(attempt + 1)
        agent.track_data("Policy / KL divergence", 0.)
        agent.track_data("Policy / KL divergence (post-update)", 0.)
        agent.track_data("Learning / Learning rate", agent.optimizer.param_groups[0]["lr"])
        agent.track_data("Learning / KL guard retries", attempt)
        agent.track_data("Learning / KL guard skipped update", 1.)
        agent.track_data("Learning / KL guard rejected KL", rejected_kl)

    agent.update = guarded_update
    agent._kl_guard_patched = True


def patch_ppo_diagnostics(agent, *, disable_advantage_norm: bool = False) -> None:
    """Phase-0 collapse diagnostics. Swaps skrl's module-level ``compute_gae``
    for an instrumented copy of the same math (loop verbatim from skrl 1.4),
    logging per update::

        Diagnostics / raw advantage mean/std — pre-normalization GAE. Near-zero
            std means skrl's (adv - mean)/(std + 1e-8) manufactures a
            full-scale confident gradient out of noise — the mechanism behind
            "near-optimal policy -> destabilizing update".

    ``disable_advantage_norm=True`` (Ablation C) also skips the normalization.

    Scope: ``compute_gae`` is a bare module-level name with no per-instance
    hook, so this patches it globally — fine while only one PPO agent runs per
    process, true of every entrypoint here. Idempotent; no-op for SAC.
    """
    # PPO.__init__ (already run, see module docstring) sets self.value; SAC has
    # critic_1/critic_2 instead. self._tensors_names would be the better check
    # but isn't set until agent.init(), which runs after this.
    if not hasattr(agent, "value"):
        return
    import skrl.agents.torch.ppo.ppo as _ppo_mod

    if getattr(_ppo_mod, "_c2rl_diagnostics_patched", False):
        return
    _ppo_mod._c2rl_diagnostics_patched = True

    def _instrumented_compute_gae(
        *,
        rewards,
        terminated,
        truncated,
        values,
        last_values,
        discount_factor: float = 0.99,
        lambda_coefficient: float = 0.95,
        time_limit_bootstrap: bool = False,
    ):
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
        memory_size = rewards.shape[0]
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            advantage = (
                rewards[i] - values[i]
                + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
            )
            advantages[i] = advantage
        returns = advantages + values

        agent.track_data("Diagnostics / raw advantage mean", advantages.mean().item())
        agent.track_data("Diagnostics / raw advantage std", advantages.std().item())

        if disable_advantage_norm:
            return returns, advantages
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    _ppo_mod.compute_gae = _instrumented_compute_gae


def patch_sac_entropy_clamp(agent, min_log_alpha: float = -5.0, max_log_alpha: float = 2.0) -> None:
    """Clamp log_entropy_coefficient in-place after every entropy optimizer step.

    skrl's SAC applies grad_norm_clip to the policy/critic optimizers but not to
    entropy_optimizer, and alpha = exp(log_alpha) is unbounded. A noisy critic
    can push this one scalar's gradient large, and exp() turns a moderate
    excursion into a runaway alpha that dominates both the critic target and the
    policy loss — a textbook SAC divergence. Bounds alpha to ~[0.0067, 7.39],
    mirroring skrl's own clip_log_std. No-op without learn_entropy.
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


def patch_algo_namespace(agent, algo_name: str) -> None:
    """Rewrite track_data keys "{cat} / {name}" -> "{cat} / {algo_name}/{name}".

    C3M/C2RL already namespace their internals this way so runs from different
    algorithms don't collide on the same wandb panels; standalone PPO/SAC never
    did, leaving skrl's own keys ("Loss / Policy loss", ...) un-namespaced.

    "Reward / *" is deliberately exempt: skrl's base Agent writes that exact key
    straight into tracking_data (bypassing track_data) to pick best_agent.pt, so
    renaming would silently break selection. Stability/*, Episode/*, Info/* also
    bypass track_data, so they are unaffected either way.
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

    c3m/cvstem-lqr — the two controllers that certify a metric directly — use
    ``contraction_score`` (λ/overshoot, higher better). Everything else (learned
    policies, lqr/sdlqr) uses AUC (lower better). Accepts either the hyphen or
    underscore spelling of the algorithm name.
    """
    norm = str(algorithm).lower().replace("_", "-")
    return "contraction_score" if norm in ("c3m", "cvstem-lqr") else "auc"


BEST_METRIC_KEY = "Checkpoint / best metric"


def patch_auc_checkpoint(agent, metric: str = "auc") -> None:
    """Override agent.post_interaction to pick best_agent.pt by ``metric``.

    skrl selects best_agent.pt by the highest ``Reward / Total reward (mean)``,
    read straight out of ``tracking_data`` in its ``post_interaction``. This
    used to be redirected by overwriting that key with the stability metric —
    which silently turned a reward panel into ``-AUC``: ``(mean)`` no longer
    measured the same quantity as the untouched ``(min)``/``(max)`` (still real
    episode returns), so a diverging run plotted its "mean" below its "min".

    Instead the metric now goes to its own :data:`BEST_METRIC_KEY` panel, and
    the best-module bookkeeping is done here against that value, mirroring
    skrl's own logic. Base skrl's reward-based selection is then neutralized by
    parking ``checkpoint_best_modules["reward"]`` at ``+inf`` so its
    ``reward > stored`` test can never overwrite the modules chosen here.
    ``write_checkpoint`` still runs from the base call and persists them.

    "contraction_score": ``Stability/contraction_score_mean``, higher better, so
    no sign flip. "auc": ``-Stability/auc_mean`` (or ``-Episode/auc`` for
    vel-tracking), flipped since lower is better. If the chosen metric isn't
    logged this step, no selection happens — no cross-fallback between the two.
    See ``best_metric_for`` for the algorithm→metric mapping.
    """
    import copy

    _orig_post = getattr(agent, "post_interaction", None)
    if _orig_post is None:
        return

    best = {"value": -float("inf")}

    def _current_metric():
        # Stability/* carries a "_mean" suffix; Episode/auc (vel-tracking) does
        # not — it comes from a different logging path.
        if metric == "contraction_score":
            score_list = agent.tracking_data.get("Stability/contraction_score_mean")
            return float(score_list[-1]) if score_list else None
        # Prioritize Stability/auc (path tracking) over Episode/auc (velocity tracking)
        score_list = agent.tracking_data.get("Stability/auc_mean") or agent.tracking_data.get("Episode/auc")
        return -float(score_list[-1]) if score_list else None

    def _metric_post(*, timestep: int, timesteps: int) -> None:
        value = _current_metric()
        if value is not None:
            agent.track_data(BEST_METRIC_KEY, value)

        # Capture the policy measured by each report; disk writes retain skrl's cadence.
        ts = timestep + 1
        if agent.checkpoint_interval > 0:
            if value is not None and math.isfinite(value) and value > best["value"]:
                best["value"] = value
                agent.checkpoint_best_modules["timestep"] = ts
                agent.checkpoint_best_modules["saved"] = False
                agent.checkpoint_best_modules["modules"] = {
                    k: copy.deepcopy(agent._get_internal_value(v))
                    for k, v in agent.checkpoint_modules.items()
                }
        # Park the base's comparison value so its own reward-based selection can
        # never win. Must be set every call, not just on checkpoint steps.
        agent.checkpoint_best_modules["reward"] = float("inf")
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _metric_post


def patch_prune_checkpoints(agent) -> None:
    """Keep only the newest ``agent_<timestep>.pt``, alongside ``best_agent.pt``.

    skrl writes one full checkpoint every ``checkpoint_interval`` steps and
    never removes the previous ones — with the configs' ``checkpoint_interval:
    "auto"`` (= timesteps/10) that is 10 full snapshots per run, none of which
    are useful once the next one lands. Only the latest snapshot (to resume
    from) and ``best_agent.pt`` (selected by ``patch_auc_checkpoint``'s metric)
    are ever actually loaded, so the older snapshots are pure disk/inode cost.

    ``best_agent.pt`` is untouched: it does not match the ``agent_*.pt`` glob.
    """
    _orig_write = getattr(agent, "write_checkpoint", None)
    if _orig_write is None:
        return

    def _pruning_write(*, timestep: int, timesteps: int) -> None:
        _orig_write(timestep=timestep, timesteps=timesteps)
        # ponytail: assumes skrl's default whole-agent layout (store_separately
        # is unset repo-wide). If store_separately is ever turned on, the
        # per-module "<name>_<tag>.pt" files need the same prune-by-prefix.
        ckpt_dir = Path(getattr(agent, "experiment_dir", "")) / "checkpoints"
        snapshots = sorted(ckpt_dir.glob("agent_*.pt"), key=lambda p: p.stat().st_mtime)
        for stale in snapshots[:-1]:
            stale.unlink(missing_ok=True)

    agent.write_checkpoint = _pruning_write


def patch_kl_logging_post_update(agent) -> None:
    """Log the KL between the policy the update ENDED on and the rollout policy.

    ``Policy / KL divergence`` is skrl's own number and stays as it is: the mean
    of the per-mini-batch KLs collected during one epoch, taken from the value
    handed to ``KLAdaptiveLR.step``. That mean is not the size of the step the
    update actually took. It mixes early mini-batches, evaluated when the policy
    had barely moved, with the one that tripped ``kl_threshold`` -- skrl appends
    the KL and only then breaks (ppo.py:398 then :401), so the tripping value is
    inside the average that gets logged. A logged 5.5 against a threshold of
    0.032 is therefore the signature of a correct trip, not a failed one, and it
    says nothing directly about where the policy came to rest.

    ``Policy / KL divergence (post-update)`` is that missing quantity: after the
    update returns, recompute ``log pi(a|s)`` for the whole rollout under the
    current parameters and compare it to the ``log_prob`` stored when those
    actions were taken. The tripping mini-batch's gradient is discarded
    (ppo.py's break precedes ``optimizer.step``), so "current parameters" is
    exactly the last applied iterate -- the policy the update ended on -- and
    the stored log_prob is the policy the rollout came from.

    Uses the same k3 estimator as skrl, ``(exp(r) - 1) - r``, so the two curves
    are directly comparable, and evaluates over the full rollout rather than one
    mini-batch, which removes the sampling noise that makes the per-mini-batch
    value swing by orders of magnitude.

    Costs one extra no-grad forward pass per update over ``rollouts x num_envs``
    transitions.
    """
    import torch

    memory = getattr(agent, "memory", None)
    policy = getattr(agent, "policy", None)
    if memory is None or policy is None or not hasattr(agent, "track_data"):
        return
    if getattr(agent, "_kl_post_update_logged", False):
        return

    _orig_update = agent.update

    def _update_then_log_kl(*, timestep: int, timesteps: int) -> None:
        _orig_update(timestep=timestep, timesteps=timesteps)
        try:
            obs = memory.get_tensor_by_name("observations")
            actions = memory.get_tensor_by_name("actions")
            old_log_prob = memory.get_tensor_by_name("log_prob")
        except (KeyError, AttributeError):
            return  # memory not laid out as PPO's; nothing to compare against
        # CHUNKED, not one forward over the whole rollout. The window encoder
        # runs a GRU over ref_length/stride points per sample, so a single call
        # on rollouts x num_envs (24 x 1024 = 24576 here) asked cuDNN for
        # 17.93 GiB and OOM-killed every run on a 22 GiB A10. The metric is a
        # mean over transitions, so summing over chunks is exactly equal to the
        # whole-batch value -- there is nothing to trade off.
        obs_f = obs.view(-1, obs.shape[-1])
        act_f = actions.view(-1, actions.shape[-1])
        old_f = old_log_prob.view(-1)
        try:
            states_f = memory.get_tensor_by_name("states")
            states_f = states_f.view(-1, states_f.shape[-1])
        except (KeyError, AttributeError):
            states_f = None
        total, count = 0.0, 0
        chunk = 1024
        # GaussianMixin.act samples even when scoring taken_actions; diagnostics
        # must not consume the policy's exploration or minibatch RNG stream.
        devices = [agent.device.index] if agent.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for i in range(0, obs_f.shape[0], chunk):
                sl = slice(i, i + chunk)
                flat = {
                    "observations": agent._observation_preprocessor(obs_f[sl]),
                    "taken_actions": act_f[sl],
                }
                if states_f is not None and hasattr(agent, "_state_preprocessor"):
                    flat["states"] = agent._state_preprocessor(states_f[sl])
                _, outputs = policy.act(flat, role="policy")
                r = outputs["log_prob"].view(-1) - old_f[sl]
                total += float(((torch.exp(r) - 1) - r).sum())
                count += r.numel()
        if not count:
            return
        kl = total / count
        agent.track_data("Policy / KL divergence (post-update)", kl)

    agent.update = _update_then_log_kl
    agent._kl_post_update_logged = True

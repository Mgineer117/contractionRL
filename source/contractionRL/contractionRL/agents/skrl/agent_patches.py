"""Generic post-construction patches for skrl PPO/SAC agent instances.

Shared between the standalone PPO/SAC path (scripts/skrl/train.py, which
patches ``runner.agent`` directly) and C2RL (c2rl.py, which patches its two
inner ``con_agent``/``opt_agent`` PPO or SAC sub-agents individually — C2RL's
own outer agent has no ``.policy``/``.scheduler``/etc. attributes for these to
find). Each patch inspects the given agent for the attributes it needs and
no-ops if they're absent, so it's safe to call unconditionally on any agent.
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

    YAML usage::

        agent:
          std_dev_annealing: True
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

def patch_auc_checkpoint(agent) -> None:
    """Override agent.post_interaction to save the best checkpoint using AUC.
    
    skrl natively saves best_agent.pt by tracking the highest 'Reward / Total reward (mean)'.
    This patch replaces that reward score with the negative AUC (so lower AUC is treated as
    better reward) and logs it so SKRL triggers the best checkpoint saving correctly.
    """
    _orig_post = getattr(agent, "post_interaction", None)
    if _orig_post is None:
        return

    def _auc_post(*, timestep: int, timesteps: int) -> None:
        # Prioritize Stability/auc (path tracking) over Episode/auc (velocity tracking)
        # Note: We negate it because SKRL saves checkpoints for the MAXIMIZED reward, and we want to MINIMIZE AUC.
        score_list = agent.tracking_data.get("Stability/auc") or agent.tracking_data.get("Episode/auc")
        if score_list:
            agent.track_data("Reward / Total reward (mean)", -score_list[-1])
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _auc_post

"""Common W&B sections for every algorithm and environment."""

from contextlib import contextmanager
from contextvars import ContextVar
import re
import sys

_defined_run = None
_epoch_metrics = set()
_pending_metrics = ContextVar("contractionrl_metrics", default=None)

STABILITY_SCALARS = frozenset({'auc_mean', 'auc_stddev', 'lambda', 'lambda_stddev',
                             'completed_episodes', 'failed_episodes', 'total_episodes', 'episode_completion'})
_STABILITY_ALIASES = {'contraction_rate': 'lambda', 'contraction_rate_mean': 'lambda',
                      'lambda_mean': 'lambda', 'contraction_rate_stddev': 'lambda_stddev'}
_STABILITY_MEDIA = frozenset({'path_tracking', 'normalized_error', 'normalized_maha_error',
                             'tracking_trajectory', 'tracking_error', 'metric_distributions'})


_ALGORITHM_PARTS = frozenset(name.replace("-", "") for name in (
    "ppo", "sac", "td3", "ddpg", "amp", "ippo", "mappo", "mbpo", "contractionppo",
    "c2rl", "c2rl-ppo", "c2rl-sac", "c3m", "lqr", "sdlqr", "sd-lqr", "cvstem-lqr"))
_SHARED_LOSSES = frozenset({"Policy loss", "Value loss", "Critic loss", "Entropy loss"})


def metric_name(name):
    """Map legacy metrics to their training/evaluation sections; omit extra statistics."""
    parts = [part.strip() for part in name.split("/")]
    if any(part.lower() == "stability_maha" for part in parts):
        return None
    group_index = int(len(parts) > 1 and parts[0] in ("Training", "Evaluation"))
    group = parts[group_index]
    tail = parts[group_index + 1:]
    # Run names identify the algorithm; shared panels must share a metric key.
    shared = group not in ("Loss", "CMG Training", "Dynamics Training")
    if group == "Loss" and 1 <= len(tail) <= 2 and tail[-1] in _SHARED_LOSSES:
        shared = True
    if shared:
        tail = [part for part in tail
                if part.lower().replace("-", "").replace("_", "") not in _ALGORITHM_PARTS]
        if group == "Learning" and tail == ["Policy learning rate"]:
            tail = ["Learning rate"]  # PPO's shared optimizer is also its policy LR.
        parts = parts[:group_index + 1] + tail
    leaf = parts[-1]
    if leaf == "KL divergence (post-update)":
        return None
    if leaf.lower().replace("_", " ") == "critic explained variance":
        return None
    if re.fullmatch(r"(?:total[ _]reward|discounted_return)(?:_(?:mean|stddev|ci95|median|min|max)|\s*\((?:mean|stddev|min|max)\))?", leaf, re.IGNORECASE):
        return None
    if re.fullmatch(r"discounted_return_n(?:_(?:mean|stddev)|\s*\((?:mean|stddev)\))?", leaf):
        return None
    if ("Clamp" in parts or "RefClamp" in parts or leaf.startswith(("frac_", "ref_frac_"))
            or leaf.endswith("_frac") or leaf == "failure_rate"):
        return None
    if leaf == "episode_completion":
        if group.startswith("Stability"):
            return "/".join(["Evaluation", group, leaf])
        return "Training/Learning/episode_completion"
    if re.search(r"(?:_(?:ci95|median|min|max|[pq]\d+)(?:_(?:mean|stddev))?|\s*\((?:min|max)\))$", leaf):
        return None
    if parts[0] == "global_step":
        return name
    if group in ("CMG Training", "Dynamics Training"):
        return "/".join(parts)
    if group == "Loss" and any(part in ("ccm", "cmg", "cm_synthesis") for part in tail):
        return "/".join(["CMG Training", *tail])
    if group in ("Loss", "Learning", "Pretrain") and any(
            part == "dynamics" or part.startswith(("dynamics_", "Dynamics ")) for part in tail):
        return "/".join(["Dynamics Training", *(part for part in tail if part != "Pretrain")])
    if parts[0] == "Evaluation" and len(parts) > 1 and parts[1].startswith("Stability"):
        parts.pop(0)  # Canonical inputs must obey the same allowlist.
    elif parts[0] in ("Evaluation", "Training"):
        return "/".join(parts)
    group = parts.pop(0)
    if group.startswith("Stability"):
        leaf = _STABILITY_ALIASES.get(leaf, leaf)
        if leaf not in STABILITY_SCALARS and leaf not in _STABILITY_MEDIA:
            return None
        return "/".join(["Evaluation", group, *parts[:-1], leaf])
    if group.startswith("Reward"):
        tail = "/".join(parts)
        tail = re.sub(r"\s*\((mean|stddev)\)$", r"_\1", tail).lower().replace(" ", "_")
        if tail.startswith("instantaneous_reward"):
            return None  # skrl retains only per-step means, not the reward population
        return "/".join(["Evaluation", group, tail])
    if group in ("Loss", "Policy", "Learning"):
        return "/".join(["Training", group, *parts])
    if group in ("train", "eval", "final_eval"):
        if leaf in ("video", "video_step"):
            return "Evaluation/" + leaf
        category = "Stability_maha" if "maha" in leaf else "Stability"
        return metric_name("/".join([category, *parts]))
    if group == "Episode":
        return "/".join(["Training", "Learning", "Diagnostics", group, *parts])
    return "/".join(["Training", "Learning", group, *parts])


def pretraining_logging_enabled(agent):
    """skrl creates its scalar writer during agent initialization."""
    return getattr(agent, "writer", None) is not None


@contextmanager
def metric_batch(*, step, axis="global_step"):
    """Combine an environment step's agents, or select a pretraining epoch axis."""
    batch = {}
    token = _pending_metrics.set(batch)
    try:
        yield
    finally:
        _pending_metrics.reset(token)
    log_metrics(batch, step=step, axis=axis)


def log_metrics(metrics, *, step=None, axis="global_step"):
    """Log one batch, using explicit axes rather than W&B's internal row count."""
    global _defined_run
    wandb = sys.modules.get("wandb")
    if wandb is None or wandb.run is None:
        return
    payload = {key: value for name, value in metrics.items()
               if (key := metric_name(name)) is not None}
    if not payload:
        return
    pending = _pending_metrics.get()
    if pending is not None:
        pending.update(payload)
        return
    if _defined_run is not wandb.run:
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("Evaluation/video_step", hidden=True)
        wandb.define_metric("*", step_metric="global_step")
        wandb.define_metric("Evaluation/*", step_metric="global_step")
        wandb.define_metric("Training/*", step_metric="global_step")
        wandb.define_metric("CMG Training/*", step_metric="global_step")
        wandb.define_metric("Dynamics Training/*", step_metric="global_step")
        wandb.define_metric("Evaluation/video", step_metric="Evaluation/video_step")
        _epoch_metrics.clear()
        _defined_run = wandb.run
    if axis != "global_step":
        wandb.define_metric(axis, hidden=True)
        for key in payload:
            if (key, axis) not in _epoch_metrics:
                wandb.define_metric(key, step_metric=axis)
                _epoch_metrics.add((key, axis))
    if step is not None:
        payload[axis] = step
    wandb.log(payload)

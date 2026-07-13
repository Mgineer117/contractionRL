"""Shared glue used across contraction agents (C3M/SDLQR/LQR/C2RL) and the
ContractionRunner: raw-dict -> dataclass config filtering, plus the
CMG-derived Mahalanobis reward machinery used by C2RL's use_cmg path.

Extracted verbatim from C2RLAgent (see c2rl.py's module docstring for the full
normalization rationale) so algorithms call the SAME code instead of copies
that can silently drift apart.
"""
from __future__ import annotations

import os
import warnings
from typing import Sequence

import torch

from .angle_utils import wrap_diff
from .math_utils import bound_W, spd_inverse


def filter_cfg_fields(cfg_dict: dict, dataclass_type, *, context: str) -> dict:
    """Keep only keys that are declared fields of ``dataclass_type``.

    Any other key is *not applied* to the agent/trainer — so instead of
    dropping it silently (which is how config typos and stale sweep parameter
    names went unnoticed), warn loudly with the ignored keys. ``class`` is
    expected to be stripped by the caller and is never reported.
    """
    fields = dataclass_type.__dataclass_fields__
    ignored = sorted(k for k in cfg_dict if k not in fields and k != "class")
    if ignored:
        warnings.warn(
            f"[{context}] ignoring config key(s) not in "
            f"{dataclass_type.__name__} (NOT applied to the algorithm): {ignored}",
            stacklevel=2,
        )
    return {k: v for k, v in cfg_dict.items() if k in fields}


def compute_mahalanobis_reward(
    ccm_gen,
    observations: torch.Tensor,
    actions: torch.Tensor | None = None,
    *,
    x_dim: int,
    u_dim: int,
    angle_idx: Sequence[int],
    w_lb: float,
    tracking_scaler: float,
    control_scaler: float,
) -> torch.Tensor:
    """Compute -(tracking_scaler * ||e||^2_M + control_scaler * ||u - uref||^2).

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
    dtype = torch.float32

    x    = observations[:, :x_dim].to(dtype)
    xref = observations[:, x_dim : 2 * x_dim].to(dtype)
    e = wrap_diff(x - xref, angle_idx).unsqueeze(-1)

    with torch.no_grad():
        raw_W, _ = ccm_gen(x)
        # Pass the CMG's `bounded` flag so a BoundedCCM_Generator (whose
        # eigenvalues are ALREADY in [w_lb, w_ub]) isn't shifted by an extra
        # +w_lb·I here — that would move the metric off the [w_lb, w_ub]
        # bounds the contraction certificate (set_contraction_certificate)
        # advertises. Mirrors c3m.py's bound_W call.
        bounded = getattr(ccm_gen, "bounded", False)
        W = bound_W(raw_W, w_lb, x_dim, bounded)
        M = spd_inverse(W)
        quad = (e.transpose(1, 2) @ M @ e).squeeze(-1)
        reward = -tracking_scaler * quad

        if actions is not None and control_scaler > 0:
            uref = observations[:, 2 * x_dim : 2 * x_dim + u_dim].to(dtype)
            feedback = actions.to(dtype) - uref
            control_cost = (feedback ** 2).sum(dim=-1, keepdim=True)
            reward = reward - control_scaler * control_cost
    return reward


def make_base_rl_cfg(
    raw_cfg: dict,
    *,
    base_algorithm: str,
    gamma: float,
    name: str,
    experiment_dir: str,
    device,
    observation_space,
    angle_idx: Sequence[int],
    x_dim: int,
    u_dim: int,
) -> dict:
    """Project a raw C2RL-style config dict down to a real PPO_CFG/SAC_CFG dict.

    Passing C2RL/CMG-specific keys (W_lr, lbd, use_cmg, ...) to
    PPO_CFG(**cfg) / SAC_CFG(**cfg) would raise TypeError, since those are
    kw_only dataclasses that reject unknown kwargs. Also rebuilds `experiment`
    as a plain dict (the raw value may be an ExperimentCfg object, which is not
    subscriptable).
    """
    if base_algorithm.upper() == "SAC":
        from skrl.agents.torch.sac import SAC_CFG as _BaseCfg
    else:
        from skrl.agents.torch.ppo import PPO_CFG as _BaseCfg
    valid = _BaseCfg.__dataclass_fields__
    d = {k: v for k, v in raw_cfg.items() if k in valid and k != "experiment"}
    d["discount_factor"] = gamma

    # YAML 1.1 (PyYAML) parses unquoted scientific notation WITHOUT a
    # decimal point (e.g. `1e-5`) as a str, not a float. skrl's Runner
    # normally rescues this via _process_cfg's eval(), but con_agent/
    # opt_agent bypass Runner entirely, so a `learning_rate: 1e-5` would
    # reach torch.optim.Adam as the string "1e-5" and blow up with
    # "'<=' not supported between instances of 'float' and 'str'".
    # Coerce any numeric-looking string scalar back to float here (the
    # `learning_rate_scheduler` name string is handled just below and is
    # non-numeric, so float() leaves it untouched via the try/except).
    for _k, _v in list(d.items()):
        if isinstance(_v, str):
            try:
                d[_k] = float(_v)
            except ValueError:
                pass

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
    rewards_shaper_scale = raw_cfg.get("rewards_shaper_scale")
    if rewards_shaper_scale is not None and rewards_shaper_scale != 1.0:
        d["rewards_shaper"] = lambda rewards, *a, scale=rewards_shaper_scale, **kw: rewards * scale

    # Standalone PPO/SAC get observation (and, for PPO, value) normalization
    # automatically via train.py's use_state_norm/use_value_norm, applied
    # through skrl's Runner._process_cfg. None of that runs here — these
    # agents are built directly, bypassing Runner entirely — so replicate it
    # explicitly: same class, same opt-out flags. Default OFF (see the
    # C2RLPPOCfg/C2RLSACCfg field defaults): a config that omits the
    # key gets no observation normalization.
    if raw_cfg.get("use_state_norm", False):
        from contractionRL.agents.skrl.preprocessors import PathTrackingObservationScaler
        d["observation_preprocessor"] = PathTrackingObservationScaler
        d["observation_preprocessor_kwargs"] = {
            "size": observation_space,
            "x_dim": x_dim,
            "u_dim": u_dim,
            "angle_idx": list(angle_idx),
            "device": device,
        }
    if base_algorithm.upper() == "PPO" and raw_cfg.get("use_value_norm", True):
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        d["value_preprocessor"] = RunningStandardScaler
        d["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # write_interval=1 so a SummaryWriter gets created — the actual flush
    # cadence is driven explicitly by the outer trainer (once per rollout
    # epoch), not by skrl's own interval logic. checkpoint_interval stays 0:
    # checkpointing is handled by the OUTER agent's own checkpoint_modules, so
    # these inner agents don't need their own redundant checkpoint files.
    # experiment.wandb is deliberately omitted (defaults False) so these inner
    # agents never call wandb.init() themselves — the OUTER agent (or
    # train.py, for a sweep) is the sole wandb.init() caller; their own
    # scalars still reach the SAME active run because skrl's
    # SummaryWriter.add_scalar is monkey-patched process-wide by train.py's
    # wandb hookup.
    d["experiment"] = {
        "directory": os.path.join(experiment_dir, name),
        "write_interval": 1,
        "checkpoint_interval": 0,
    }
    return d

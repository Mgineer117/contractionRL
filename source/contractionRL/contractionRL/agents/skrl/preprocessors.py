"""Project-side observation preprocessors for the ``[x, xref, uref]``
path-tracking layout used throughout this repo.

Never edit vendored skrl code (see ``skrl.resources.preprocessors.torch.
RunningStandardScaler``) — this module wraps it instead, because standardizing
the FULL observation vector is wrong for this layout.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import compute_space_size


class RunningRewardScaler(nn.Module):
    """Non-biasing reward normalizer for PPO's ``rewards_shaper`` hook.

    Divides the reward by a running estimate of its standard deviation (never
    subtracts a mean), so the reward fed to GAE is ~unit-variance regardless of
    the metric-bound-dependent scale of the Mahalanobis reward
    ``tracking_scaler·(eᵀM e − next_eᵀM next_e)`` (``M = W⁻¹`` has eigenvalues in
    ``[1/w_ub, 1/w_lb]``, so its magnitude — and its heavy tail — grows as
    ``1/w_lb``; see ``env_base.get_rewards`` and c2rl.py). This tames both that
    scale and the early-training transient BEFORE the cumulative ``value_norm``
    (``RunningStandardScaler`` on the value) can catch up, which it lags because
    the frozen-CMG reward is non-stationary as the policy improves.

    Why ``/std`` and no centering (the "does not bias the problem" property):
    multiplying every reward by a positive scalar ``c`` scales all returns and
    advantages by ``c``, leaving ``argmax_π`` unchanged. Subtracting a mean
    shifts returns by a state/time-dependent constant and can change the
    advantage structure — so only the positive-scalar divide is used.

    Reuses skrl's ``RunningStandardScaler`` purely for its parallel-variance
    tracking; only ``sqrt(running_variance)`` is consumed (running_mean is
    ignored). By default tracks the variance of the raw per-step reward
    (``gamma=0.0``), which needs no episode-reset/``done`` signal — the
    ``rewards_shaper`` hook does not receive ``done`` (see ppo.py
    ``record_transition``). Setting ``gamma>0`` switches to the SB3
    ``VecNormalize`` variant (std of the discounted return ``R = γ·R + r``); with
    no ``done`` reset this leaks across episode boundaries, so it is only
    appropriate at small ``gamma`` (this repo's ``discount_factor`` is ~0.01,
    where return ≈ reward and the two variants coincide anyway).
    """

    def __init__(
        self,
        *,
        gamma: float = 0.0,
        scale: float = 1.0,
        epsilon: float = 1e-8,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.epsilon = float(epsilon)
        self._scaler = RunningStandardScaler(size=1, epsilon=epsilon, device=device)
        # per-env discounted-return accumulator (lazily sized on first call)
        self._returns: torch.Tensor | None = None

    def forward(self, rewards: torch.Tensor, timestep=None, timesteps=None) -> torch.Tensor:
        # rewards: (num_envs, 1). Match skrl's rewards_shaper(rewards, timestep, timesteps).
        with torch.no_grad():
            if self.gamma > 0.0:
                if self._returns is None or self._returns.shape != rewards.shape:
                    self._returns = torch.zeros_like(rewards)
                self._returns = self.gamma * self._returns + rewards
                tracked = self._returns
            else:
                tracked = rewards
            # update running variance, then divide reward by std (no centering).
            self._scaler(tracked, train=True)
            std = torch.sqrt(self._scaler.running_variance.float()) + self.epsilon
        return self.scale * rewards / std


class PathTrackingObservationScaler(nn.Module):
    """``RunningStandardScaler`` restricted to the ``[x, xref]`` portion of a
    ``[x, xref, uref]`` observation, and further excluding ``angle_idx``
    columns of ``x``/``xref``.

    Two things must stay raw for this repo's residual/embedding math to be
    correct (see ``c2rl.py``'s module docstring and ``angle_utils.py``):

      - ``uref``: the residual backbones (``control``/``mlp``, squashed or
        not) slice ``uref`` straight out of the observation and add it to the
        network's feedback — for the squashed backbones, that add happens
        AFTER tanh-squashing (see ``models.py``'s ``_TanhSquashMixin``).
        Normalizing ``uref`` would make the applied control law
        ``uref_norm + feedback`` instead of ``uref + feedback``, distorting
        the reference-tracking residual.
      - ``angle_idx`` columns of ``x``/``xref``: ``models.py`` replaces each
        with ``(cos, sin)`` via ``embed_angles`` so the network sees a
        continuous, periodic input. Standardizing the raw angle first
        (``(theta - mean) / std``) breaks that periodicity — ``cos``/``sin``
        of a shifted-and-rescaled angle is not ``2*pi``-periodic in ``theta``.

    Everything else in ``x``/``xref`` (non-angle physical states) is
    standardized exactly like the stock ``RunningStandardScaler``, using its
    own running mean/std fit ONLY over that normalized subset.
    """

    def __init__(
        self,
        size,
        *,
        x_dim: int,
        u_dim: int,
        angle_idx: Sequence[int] = (),
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        obs_dim = compute_space_size(size, occupied_size=True)
        if obs_dim != 2 * x_dim + u_dim:
            raise ValueError(
                "PathTrackingObservationScaler requires the [x, xref, uref] layout "
                f"(obs_dim == 2*x_dim + u_dim), got obs_dim={obs_dim}, x_dim={x_dim}, "
                f"u_dim={u_dim}."
            )
        angle_set = {int(i) for i in angle_idx}
        normalize = torch.ones(obs_dim, dtype=torch.bool)
        normalize[2 * x_dim :] = False  # uref block: never normalized
        for i in angle_set:
            normalize[i] = False  # x block angle column
            normalize[x_dim + i] = False  # xref block angle column (mirrored layout)
        norm_idx = normalize.nonzero(as_tuple=True)[0]
        self.register_buffer("_normalize_idx", norm_idx)
        self._scaler = RunningStandardScaler(
            size=int(norm_idx.numel()), epsilon=epsilon, clip_threshold=clip_threshold, device=device
        )

    def forward(
        self, x: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        if x is None:
            return None
        idx = self._normalize_idx
        scaled = self._scaler(x.index_select(-1, idx), train=train, inverse=inverse, no_grad=no_grad)
        return x.index_copy(-1, idx, scaled)

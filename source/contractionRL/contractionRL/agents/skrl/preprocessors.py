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

"""Shared continuous-angle embedding + wrapped-difference math.

Design (see conversation): the RAW state (with plain angle scalars at
``angle_idx``) is what the environment, the contraction loss, the dynamics
integration, and the tracking error are computed in — nothing about the
physics or the certificate math changes. The embedding below is applied ONLY
at the *input* of each neural network, replacing every raw angle ``theta``
with the continuous ``(cos(theta), sin(theta))`` pair so the network sees no
discontinuity at +-pi and is automatically periodic. Network OUTPUTS (e.g.
NeuralDynamics' f, B) stay in raw coordinates.

``wrap_diff`` is the complementary piece: whenever a state DIFFERENCE is taken
(tracking error x - xref, or CLActor's bilinear feedback error), the angle
dims of that difference must be wrapped to the shortest-angle representative
in (-pi, pi] before it is used in a norm/reward/metric/matmul — otherwise a
raw wraparound (e.g. a U-turn) spikes the difference by ~2*pi.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def embedded_dim(dim: int, angle_idx: Sequence[int]) -> int:
    """Width of a length-``dim`` block after embedding its ``angle_idx`` entries."""
    return dim + len(set(int(i) for i in angle_idx))


def embed_angles(block: torch.Tensor, angle_idx: Sequence[int]) -> torch.Tensor:
    """Replace each ``angle_idx`` column of ``block`` with (cos, sin).

    ``block``: (..., dim) raw state block. Non-angle columns pass through
    unchanged; each angle column at index ``i`` becomes two columns
    ``(cos(block[...,i]), sin(block[...,i]))`` at that position — so the
    output width is ``embedded_dim(block.shape[-1], angle_idx)``.
    """
    if not angle_idx:
        return block
    angle_set = set(int(i) for i in angle_idx)
    dim = block.shape[-1]
    pieces = []
    for i in range(dim):
        col = block[..., i : i + 1]
        if i in angle_set:
            pieces.append(torch.cos(col))
            pieces.append(torch.sin(col))
        else:
            pieces.append(col)
    return torch.cat(pieces, dim=-1)


def wrap_diff(diff: torch.Tensor, angle_idx: Sequence[int]) -> torch.Tensor:
    """Wrap the ``angle_idx`` columns of a raw difference into (-pi, pi].

    Non-angle columns pass through unchanged. Uses ``torch.where`` (no
    in-place indexing) so it stays fully differentiable; matches the
    classic envs' numpy ``wrap_angles`` convention exactly:
    ``(d + pi) % (2*pi) - pi``.
    """
    if not angle_idx:
        return diff
    mask = torch.zeros(diff.shape[-1], dtype=torch.bool, device=diff.device)
    mask[list(int(i) for i in angle_idx)] = True
    wrapped = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
    return torch.where(mask, wrapped, diff)

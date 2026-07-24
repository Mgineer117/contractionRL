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

``pos_dim`` and the ``*_features``/``*_feature_dim`` helpers are the third
piece: the TRANSLATION QUOTIENT. Every env in this repo declares a leading
block of ``pos_dimension`` state dims that ``f`` and ``B`` provably do not
depend on (verified numerically for car/turtlebot/segway/quadrotor: perturbing
those dims changes f and B by exactly 0). The tracking problem is therefore
invariant to translating ``x`` and ``xref`` together along that block, and the
canonical (complete, lossless) invariant replaces the two ABSOLUTE position
blocks with the single RELATIVE one. Networks that see only ``x`` drop the
block outright.

Without this, a network reading absolute positions has to relearn the same
control law at every point of the position box, and it only ever sees large
tracking errors wherever the reference happens to start — measured on a
trained car checkpoint, translating a whole (x0, xref) pair by 3 m (an exact
symmetry of the dynamics) moved the eval failure rate from 4.3% to 30%.

``pos_dim=0`` makes every helper below reduce exactly to its plain
``embed_angles``/``embedded_dim`` behaviour, so the quotient is opt-in and any
site that is not passed a ``pos_dim`` keeps the previous semantics.
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


# ─────────────────────────────────────────────────────────────────────────── #
# Translation quotient — see the module docstring.
# ─────────────────────────────────────────────────────────────────────────── #

def _shifted_angle_idx(angle_idx: Sequence[int], pos_dim: int) -> list[int]:
    """Re-index ``angle_idx`` into a block whose leading ``pos_dim`` dims were
    dropped. Angles are never inside the position block (a position is not an
    angle in any env here), which the assertion makes explicit rather than
    letting a mis-declared ``pos_dimension`` silently corrupt the embedding.
    """
    idx = sorted(set(int(i) for i in angle_idx))
    if pos_dim and idx and idx[0] < pos_dim:
        raise ValueError(
            f"angle_idx={idx} overlaps the position block (pos_dim={pos_dim}): a "
            "translation-invariant dim cannot also be an angle. Check the env's "
            "pos_dimension/angle_idx declaration."
        )
    return [i - pos_dim for i in idx]


def state_feature_dim(dim: int, pos_dim: int, angle_idx: Sequence[int]) -> int:
    """Input width of a network that sees ONE state block (e.g. W(x), f/B nets)."""
    return embedded_dim(dim - pos_dim, _shifted_angle_idx(angle_idx, pos_dim))


def state_features(x: torch.Tensor, pos_dim: int, angle_idx: Sequence[int]) -> torch.Tensor:
    """Network input for a single state block: drop the translation directions,
    then angle-embed what is left.

    Dropping (rather than zeroing) the position columns is what makes the
    invariance exact: for W(x) and for f/B the position dims are pure symmetry
    directions, so d/d(pos) is identically 0 by construction instead of being a
    small nonzero number the optimiser has to push down.
    """
    if not pos_dim:
        return embed_angles(x, angle_idx)
    return embed_angles(x[..., pos_dim:], _shifted_angle_idx(angle_idx, pos_dim))


def pair_feature_dim(x_dim: int, pos_dim: int, angle_idx: Sequence[int]) -> int:
    """Input width of a network that sees the ``(x, xref)`` PAIR.

    ``pos_dim`` relative-position dims (shared by both blocks) + the two
    position-stripped, angle-embedded blocks. At ``pos_dim=0`` this is exactly
    the previous ``2 * embedded_dim(x_dim, angle_idx)``.
    """
    return pos_dim + 2 * state_feature_dim(x_dim, pos_dim, angle_idx)


def pair_features(
    x: torch.Tensor, xref: torch.Tensor, pos_dim: int, angle_idx: Sequence[int]
) -> torch.Tensor:
    """Network input for an ``(x, xref)`` pair, quotiented by translation.

    Layout: ``[x_pos - xref_pos, features(x), features(xref)]``. This is a
    COMPLETE invariant of the translation action — nothing is lost, because the
    dynamics and the tracking reward are themselves translation invariant, so
    the map is a bijection onto the reduced state space.
    """
    if not pos_dim:
        return torch.cat([embed_angles(x, angle_idx), embed_angles(xref, angle_idx)], dim=-1)
    return torch.cat(
        [
            x[..., :pos_dim] - xref[..., :pos_dim],
            state_features(x, pos_dim, angle_idx),
            state_features(xref, pos_dim, angle_idx),
        ],
        dim=-1,
    )


def is_translation_invariant(
    get_f_and_B, x_min: torch.Tensor, x_max: torch.Tensor, pos_dim: int,
    *, samples: int = 256, offset: float = 10.0, atol: float = 1e-5,
) -> bool:
    """Ask the ENV whether translating its leading ``pos_dim`` dims leaves f and
    B unchanged, instead of maintaining a per-env whitelist.

    This is what makes the quotient universal and safe: a new env (or one whose
    ``pos_dimension`` is mis-declared, or one with position-dependent terrain /
    drag / obstacle terms) fails the check and transparently falls back to the
    absolute-observation behaviour. Returns False on any error, since "could not
    prove the symmetry" must never be treated as "the symmetry holds".
    """
    if not pos_dim:
        return False
    try:
        x_min = x_min.flatten().float()
        x_max = x_max.flatten().float()
        x = x_min + torch.rand(samples, x_min.numel(), device=x_min.device) * (x_max - x_min)
        shifted = x.clone()
        shifted[:, :pos_dim] += (torch.rand(samples, pos_dim, device=x.device) * 2 - 1) * offset
        f1, B1, _ = get_f_and_B(x)
        f2, B2, _ = get_f_and_B(shifted)
        return bool(
            torch.allclose(f1, f2, atol=atol) and torch.allclose(B1, B2, atol=atol)
        )
    except Exception:
        return False

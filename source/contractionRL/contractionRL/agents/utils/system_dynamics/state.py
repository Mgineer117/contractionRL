"""Angle embeddings, wrapped differences, and named state symmetries.

Design (see conversation): the raw state (with plain angle scalars at
``angle_idx``) is what the environment, the contraction loss, the dynamics
integration, and the tracking error are computed in — nothing about the
physics or the certificate math changes. The embedding below is applied only
at the *input* of each neural network, replacing every raw angle ``theta``
with the continuous ``(cos(theta), sin(theta))`` pair so the network sees no
discontinuity at +-pi and is automatically periodic. Network outputs (e.g.
NeuralDynamics' f, B) stay in raw coordinates.

``wrap_diff`` is the complementary piece: whenever a state difference is taken
(tracking error x - xref, or CLActor's bilinear feedback error), the angle
dims of that difference must be wrapped to the shortest-angle representative
in (-pi, pi] before it is used in a norm/reward/metric/matmul — otherwise a
raw wraparound (e.g. a U-turn) spikes the difference by ~2*pi.

The symmetry quotient is the third piece, and it lives in
``state_symmetry.py`` (which builds on the two functions here). Every env in this repo declares a leading
block of ``pos_dimension`` state dims that ``f`` and ``B`` provably do not
depend on (verified numerically for car/turtlebot/segway/quadrotor: perturbing
those dims changes f and B by exactly 0). The tracking problem is therefore
invariant to translating ``x`` and ``xref`` together along that block, and the
canonical (complete, lossless) invariant replaces the two absolute position
blocks with the single relative one. Networks that see only ``x`` drop the
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

from collections.abc import Sequence
from dataclasses import dataclass
import math

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


def unwrap_state_angles(x, previous, angle_idx=()):
    """Lift measured orientation to the continuous branch nearest the last sample."""
    if previous is None or not angle_idx:
        return x.clone()
    result = x.clone()
    angles = list(angle_idx)
    delta = x[..., angles] - previous[..., angles]
    lifted = previous[..., angles] + torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    result[..., angles] = torch.where(torch.isfinite(previous[..., angles]), lifted, x[..., angles])
    return result


def unwrap_angle_trajectory(states, angle_idx=()):
    """Continuous angular lifts along the penultimate (time) dimension."""
    if not angle_idx:
        return states
    result = states.clone()
    angles = list(angle_idx)
    values = states[..., angles]
    increments = torch.remainder(values[..., 1:, :] - values[..., :-1, :] + math.pi, 2 * math.pi) - math.pi
    result[..., angles] = torch.cat((values[..., :1, :], values[..., :1, :] + increments.cumsum(-2)), -2)
    return result


def clamp_physical_state(x, low, high, angle_idx=()):
    """Clamp bounded coordinates while retaining continuous full-turn angles."""
    low, high = torch.as_tensor(low, device=x.device), torch.as_tensor(high, device=x.device)
    mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
    mask[list(angle_idx)] = True
    full_turn = mask & (high - low >= 2 * math.pi - 1e-5)
    return torch.where(full_turn, x, x.clamp(low, high))


def canonical_bound_state(x, low, high, angle_idx=()):
    """A coordinate copy for box checks; never changes the simulator state."""
    low, high = torch.as_tensor(low, device=x.device), torch.as_tensor(high, device=x.device)
    mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
    mask[list(angle_idx)] = True
    full_turn = mask & (high - low >= 2 * math.pi - 1e-5)
    return torch.where(full_turn, torch.remainder(x - low, 2 * math.pi) + low, x)


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







# Name vocabulary
# World-frame planar position: translates and rotates.
_POS_XY = ("pos_x", "pos_y")
# Translates but does not rotate (the rotation axis).
_POS_Z = ("pos_z",)
# The planar heading. Rotating the scene by phi adds phi to it.
_YAW = "yaw"
# World-frame planar velocity: rotates with position (suffix _w = world frame).
_VEL_XY_W = ("vel_x_w", "vel_y_w")
# Dims that hold a raw wrapping angle and so need (cos, sin) embedding.
_ANGULAR = frozenset({"roll", "pitch", "yaw"})
# Everything below is invariant to both translation and yaw rotation. Body-frame
# quantities (suffix _b), joint states, and projected gravity are yaw-invariant
# by construction; scalar speeds and forces have no planar direction at all.
_INVARIANT_EXACT = frozenset({
    "vel", "vel_z_w", "vel_x_b", "vel_y_b", "vel_z_b",
    "roll", "pitch", "roll_rate", "pitch_rate", "yaw_rate",
    "ang_vel_x_b", "ang_vel_y_b", "ang_vel_z_b",
    "thrust", "force", "height",
})
# Prefixes for repeated/indexed dims (joint_pos_0, proj_grav_z, ...).
_INVARIANT_PREFIXES = ("joint_pos", "joint_vel", "joint_torque", "proj_grav",
                       "gravity_b", "ee_vel_b", "contact", "phase")


def _is_invariant(name: str) -> bool:
    return name in _INVARIANT_EXACT or name.startswith(_INVARIANT_PREFIXES)


@dataclass(frozen=True)
class StateSymmetry:
    """The symmetry the networks may quotient out, plus the feature maps.

    ``mode`` is the only thing callers branch on: "none" reproduces the historic
    absolute-observation behaviour exactly, so an env whose names are not fully
    understood, or whose dynamics fail the numerical equivariance check, keeps
    the previous semantics rather than being mis-modelled.
    """

    x_dim: int
    names: tuple[str, ...]
    mode: str                                   # "none" | "translation" | "se2"
    pos_idx: tuple[int, ...] = ()               # translation directions
    yaw_idx: int | None = None
    rot_pairs: tuple[tuple[int, int], ...] = () # (x, y) index pairs that co-rotate
    angle_idx: tuple[int, ...] = ()

    # construction
    @classmethod
    def from_names(cls, names: Sequence[str], *, allow_se2: bool = True) -> StateSymmetry:
        names = tuple(str(n) for n in names)
        x_dim = len(names)
        angle_idx = tuple(i for i, n in enumerate(names) if n in _ANGULAR)
        pos_idx = tuple(i for i, n in enumerate(names) if n in _POS_XY + _POS_Z)
        if not pos_idx:
            return cls(x_dim, names, "none", angle_idx=angle_idx)

        # Rotation additionally needs a yaw, a full pos_x/pos_y pair, and every
        # remaining dim to be a recognised yaw-invariant name. An unknown name
        # fails closed to translation.
        idx = {n: i for i, n in enumerate(names)}
        has_planar = _YAW in idx and all(n in idx for n in _POS_XY)
        rot_pairs: tuple[tuple[int, int], ...] = ()
        if has_planar:
            rot_pairs = ((idx["pos_x"], idx["pos_y"]),)
            if all(n in idx for n in _VEL_XY_W):
                rot_pairs += ((idx["vel_x_w"], idx["vel_y_w"]),)
        rotating = {i for pair in rot_pairs for i in pair} | ({idx[_YAW]} if has_planar else set())
        unexplained = [names[i] for i in range(x_dim)
                       if i not in rotating and not _is_invariant(names[i])
                       and names[i] not in _POS_Z]
        se2_ok = allow_se2 and has_planar and not unexplained
        return cls(
            x_dim=x_dim, names=names,
            mode="se2" if se2_ok else "translation",
            pos_idx=pos_idx,
            yaw_idx=idx[_YAW] if se2_ok else None,
            rot_pairs=rot_pairs if se2_ok else (),
            angle_idx=angle_idx,
        )

    def demote_to_translation(self) -> StateSymmetry:
        """Drop back to the translation quotient (e.g. the numerical rotation
        equivariance check failed for this env's dynamics)."""
        if self.mode != "se2":
            return self
        return StateSymmetry(self.x_dim, self.names, "translation",
                             pos_idx=self.pos_idx, angle_idx=self.angle_idx)

    def disabled(self) -> StateSymmetry:
        return StateSymmetry(self.x_dim, self.names, "none", angle_idx=self.angle_idx)

    # derived legacy views
    @property
    def pos_dimension(self) -> int:
        """Back-compat: the historic leading-block position count. Only equal to
        ``len(pos_idx)`` when the position dims really are the leading block —
        which they are in every env here, but the feature maps below use
        ``pos_idx`` directly so they do not depend on it."""
        return len(self.pos_idx)

    # index bookkeeping
    def _kept_idx(self) -> tuple[int, ...]:
        """State dims a network sees per block: everything except the position
        dims (pure symmetry directions) and, under SE(2), the absolute yaw
        (which the canonicalization fixes to zero)."""
        drop = set(self.pos_idx)
        if self.mode == "se2":
            drop.add(self.yaw_idx)
        return tuple(i for i in range(self.x_dim) if i not in drop)

    def _kept_angle_idx(self) -> list[int]:
        kept = self._kept_idx()
        return [kept.index(i) for i in self.angle_idx if i in kept]

    # single-block features: W(x), NeuralDynamics f/B
    def single_dim(self) -> int:
        if self.mode == "none":
            return embedded_dim(self.x_dim, self.angle_idx)
        return embedded_dim(len(self._kept_idx()), self._kept_angle_idx())

    def single_features(self, x: torch.Tensor) -> torch.Tensor:
        """One state block, symmetry directions removed.

        Dropping (not zeroing) the columns is what makes the invariance exact:
        d/d(dropped dim) is identically 0 by construction rather than a small
        number the optimiser has to push down.

        NOTE for SE(2): this drops the absolute yaw without rotating anything,
        so it is a valid map only for quantities that are themselves yaw
        invariant. W(x) is not — a metric transforms as a tensor, W(g.x) =
        dg W(x) dg^T — so the CMG deliberately stays on the translation quotient
        (see contraction_runner._resolve_symmetry).
        """
        if self.mode == "none":
            return embed_angles(x, self.angle_idx)
        kept = list(self._kept_idx())
        return embed_angles(x[..., kept], self._kept_angle_idx())

    # pair features: the actor's gain nets and the critics
    def pair_dim(self) -> int:
        if self.mode == "none":
            return 2 * embedded_dim(self.x_dim, self.angle_idx)
        n_rel = len(self.pos_idx) + (1 if self.mode == "se2" else 0)
        return n_rel + 2 * self.single_dim()

    def pair_features(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """Complete invariant of the ``(x, xref)`` pair under the symmetry.

        translation: ``[x_pos - xref_pos, feats(x), feats(xref)]``
        se2:         ``[R(-yaw_ref) (x_pos - xref_pos), wrap(yaw - yaw_ref),
                        feats(R(-yaw_ref) x), feats(R(-yaw_ref) xref)]``

        Lossless in both cases: the dynamics and the tracking reward are
        themselves invariant, so this is a bijection onto the reduced state
        space, not an approximation.
        """
        if self.mode == "none":
            return torch.cat([embed_angles(x, self.angle_idx),
                              embed_angles(xref, self.angle_idx)], dim=-1)
        if self.mode == "translation":
            pos = list(self.pos_idx)
            return torch.cat([x[..., pos] - xref[..., pos],
                              self.single_features(x),
                              self.single_features(xref)], dim=-1)

        # SE(2): rotate the whole scene so the reference yaw is zero
        yaw_ref = xref[..., self.yaw_idx]
        cos, sin = torch.cos(-yaw_ref), torch.sin(-yaw_ref)
        x_r, xref_r = self._rotate(x, cos, sin), self._rotate(xref, cos, sin)
        pos = list(self.pos_idx)
        d_pos = x_r[..., pos] - xref_r[..., pos]
        d_yaw = torch.remainder(
            x[..., self.yaw_idx] - yaw_ref + math.pi, 2 * math.pi) - math.pi
        return torch.cat([d_pos, d_yaw.unsqueeze(-1),
                          self.single_features(x_r),
                          self.single_features(xref_r)], dim=-1)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the planar rotation to every co-rotating (x, y) pair and to yaw.

        ``cos``/``sin`` are per-sample (shape ``x.shape[:-1]``), so each sample is
        rotated into its own reference frame.
        """
        out = x.clone()
        for ix, iy in self.rot_pairs:
            vx, vy = x[..., ix], x[..., iy]
            out[..., ix] = cos * vx - sin * vy
            out[..., iy] = sin * vx + cos * vy
        # yaw is only read through single_features (which drops it) and d_yaw
        # above, but keep it consistent so _rotate is a true group action.
        ang = torch.atan2(sin, cos)
        out[..., self.yaw_idx] = torch.remainder(
            x[..., self.yaw_idx] + ang + math.pi, 2 * math.pi) - math.pi
        return out

    # the feedback error vector
    def error_features(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """The tracking error CLActor's bilinear feedback multiplies, expressed in
        the canonical frame. Full ``x_dim`` width — every state dim still gets
        feedback; only the frame changes.

        Canonicalizing the gain-net inputs is not enough on its own: the control
        law is ``u = W2(feats) tanh(W1(feats) e)``, so if ``e`` stays in world
        coordinates its position components rotate with the scene and ``u`` moves
        even when the features do not. Under SE(2) the position pairs are rotated
        by ``R(-yaw_ref)``, which makes the feedback act on the longitudinal /
        lateral error in the reference frame — the physically meaningful
        parameterisation for a planar vehicle — and the heading term on the
        shortest-angle yaw error.

        ``e == 0`` still implies ``u == uref`` (a rotation is invertible and fixes
        the origin), which is what the contraction certificate needs.
        """
        e = wrap_diff(x - xref, self.angle_idx)
        if self.mode != "se2":
            return e
        yaw_ref = xref[..., self.yaw_idx]
        cos, sin = torch.cos(-yaw_ref), torch.sin(-yaw_ref)
        out = e.clone()
        for ix, iy in self.rot_pairs:
            ex, ey = e[..., ix], e[..., iy]
            out[..., ix] = cos * ex - sin * ey
            out[..., iy] = sin * ex + cos * ey
        return out

    # numerical verification against the env's own dynamics
    def verify_translation(self, get_f_and_B, x_min, x_max, *, samples=256,
                           offset=10.0, atol=1e-5) -> bool:
        """f and B must not depend on the position dims at all."""
        try:
            x = self._sample(x_min, x_max, samples)
            shifted = x.clone()
            for i in self.pos_idx:
                shifted[:, i] += (torch.rand(samples, device=x.device) * 2 - 1) * offset
            f1, B1, _ = get_f_and_B(x)
            f2, B2, _ = get_f_and_B(shifted)
            return bool(torch.allclose(f1, f2, atol=atol) and torch.allclose(B1, B2, atol=atol))
        except Exception:
            return False

    def verify_rotation(self, get_f_and_B, x_min, x_max, *, samples=256,
                        atol=1e-4) -> bool:
        """f and B must be equivariant, not invariant: rotating the state must
        rotate f and B's rows the same way.

            f(R.x) = R f(x)      B(R.x) = R B(x)

        This is what fails for the quadrotor, whose roll/pitch are world
        referenced so a yaw rotation mixes them — caught here rather than
        assumed away.
        """
        if self.mode != "se2":
            return False
        try:
            x = self._sample(x_min, x_max, samples)
            phi = (torch.rand(samples, device=x.device) * 2 - 1) * math.pi
            cos, sin = torch.cos(phi), torch.sin(phi)
            f1, B1, _ = get_f_and_B(x)
            f2, B2, _ = get_f_and_B(self._rotate(x, cos, sin))
            # Rotate f (a state-space vector) and B's rows (state-space columns)
            f1r = self._rotate_tangent(f1, cos, sin)
            B1r = torch.stack([self._rotate_tangent(B1[..., j], cos, sin)
                               for j in range(B1.shape[-1])], dim=-1)
            return bool(torch.allclose(f1r, f2, atol=atol) and torch.allclose(B1r, B2, atol=atol))
        except Exception:
            return False

    def _rotate_tangent(self, v: torch.Tensor, cos, sin) -> torch.Tensor:
        """Rotate a tangent vector (f, or a column of B). Same planar pairs as
        the state, but yaw's tangent component is a rate, which is rotation
        invariant — so unlike ``_rotate`` there is no additive yaw shift."""
        out = v.clone()
        for ix, iy in self.rot_pairs:
            vx, vy = v[..., ix], v[..., iy]
            out[..., ix] = cos * vx - sin * vy
            out[..., iy] = sin * vx + cos * vy
        return out

    def _sample(self, x_min, x_max, n) -> torch.Tensor:
        lo = x_min.flatten().float()
        hi = x_max.flatten().float()
        return lo + torch.rand(n, lo.numel(), device=lo.device) * (hi - lo)

    def describe(self) -> str:
        return (f"mode={self.mode} x_dim={self.x_dim} pos={list(self.pos_idx)} "
                f"yaw={self.yaw_idx} rot_pairs={list(self.rot_pairs)} "
                f"angles={list(self.angle_idx)} -> pair_dim={self.pair_dim()} "
                f"single_dim={self.single_dim()}")

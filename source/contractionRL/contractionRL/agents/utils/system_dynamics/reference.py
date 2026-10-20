"""Compact/legacy reference layouts and their tracking features.

ReferenceState represents ``s = (x, xref, xi)``. RefWindow retains the legacy
``s = {x, xrefs, urefs}`` layout described below for sequence policies.

Replaces the old flat ``[x, xref, uref, preview]`` Box layout and all of its
inference machinery (``_preview_width``, the ``obs_dim/2`` parity guess, the
``preview_includes_xref`` / ``preview_includes_uref`` flags). The layout is now
declared by the observation space and read back from it, so an env and a model
can no longer silently disagree about where a block starts.

Layout
------
``xrefs[k] = xref[t + k*offset]``, ``urefs[k] = uref[t + k*offset]``, for
``k = 0 .. length-1``. ``k=0`` is the current reference point, so ``xrefs[0]``
is the old ``xref`` and ``urefs[0]`` is the old ``uref`` — the control law
``u = urefs[0] + pi`` and the tracking error ``e = x - xrefs[0]`` are unchanged.
Indices past the end of the episode clamp to the last one (the reference "stops
and holds" at the terminal setpoint).

Flat ordering (load-bearing)
----------------------------
skrl stores a ``gymnasium.spaces.Dict`` observation flattened, and
``unflatten_tensorized_space`` walks ``sorted(space.keys())`` — so the flat
tensor a model receives in ``inputs["observations"]`` is ordered

    [ urefs (length*u_dim) | x (x_dim) | xrefs (length*x_dim) ]

alphabetically, not in declaration order. ``RefWindow.split`` is the single
place that knows this; nothing else may slice the observation by hand.

Markov-ness
-----------
See ``RefWindow.check_markov``. The window is a sufficient statistic for the
value function only if it spans the discount's effective horizon; a window
shorter than that makes the problem a POMDP again, which is the exact failure
the old preview was introduced to fix.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
import torch

from contractionRL.agents.utils.system_dynamics.state import StateSymmetry
from contractionRL.agents.utils.system_dynamics.state import embed_angles, embedded_dim, wrap_diff


@dataclass(frozen=True)
class RefWindow:
    """Shape/layout of a ``{x, xrefs, urefs}`` observation.

    ``offset`` is carried for bookkeeping and the Markov check; models never
    need it (they consume the window as an ordered sequence), which is why
    ``from_space`` can rebuild everything a model uses from the space alone.
    """

    x_dim: int
    u_dim: int
    length: int
    offset: int = 1

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"RefWindow: length must be >= 1, got {self.length}")
        if self.offset < 1:
            raise ValueError(f"RefWindow: offset must be >= 1, got {self.offset}")

    # ── layout ──────────────────────────────────────────────────────────── #
    @property
    def flat_dim(self) -> int:
        return self.x_dim + self.length * (self.x_dim + self.u_dim)

    def space(self, x_lo, x_hi, u_lo, u_hi) -> gym.spaces.Dict:
        """Build the Dict observation space. Bounds are the per-dim state /
        control boxes; the sequence blocks repeat them ``length`` times."""
        def _np(v):
            return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

        x_lo, x_hi, u_lo, u_hi = _np(x_lo), _np(x_hi), _np(u_lo), _np(u_hi)
        L = self.length
        return gym.spaces.Dict({
            "x": gym.spaces.Box(low=x_lo, high=x_hi, dtype=np.float32),
            "xrefs": gym.spaces.Box(
                low=np.tile(x_lo, (L, 1)), high=np.tile(x_hi, (L, 1)), dtype=np.float32),
            "urefs": gym.spaces.Box(
                low=np.tile(u_lo, (L, 1)), high=np.tile(u_hi, (L, 1)), dtype=np.float32),
        })

    @classmethod
    def from_space(cls, space, offset: int = 1) -> RefWindow:
        """Recover the layout from a Dict observation space — how every model
        learns its own input shape, instead of guessing from ``obs_dim``."""
        if hasattr(space, "reference_state"):
            return space.reference_state
        if not isinstance(space, gym.spaces.Dict):
            raise TypeError(
                f"RefWindow.from_space expects a gymnasium Dict observation space "
                f"with keys {{x, xrefs, urefs}}, got {type(space).__name__}. The flat "
                f"[x, xref, uref] layout is no longer supported.")
        missing = {"x", "xrefs", "urefs"} - set(space.spaces)
        if missing:
            raise ValueError(f"RefWindow.from_space: observation space missing keys {sorted(missing)}")
        length, x_dim = space["xrefs"].shape
        u_length, u_dim = space["urefs"].shape
        if u_length != length:
            raise ValueError(
                f"RefWindow.from_space: xrefs has {length} points but urefs has {u_length}")
        if space["x"].shape != (x_dim,):
            raise ValueError(
                f"RefWindow.from_space: x has width {space['x'].shape} but xrefs points are {x_dim}-wide")
        return cls(x_dim=int(x_dim), u_dim=int(u_dim), length=int(length), offset=int(offset))

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat observation -> ``(x, xrefs, urefs)`` with shapes
        ``(N, x_dim)``, ``(N, length, x_dim)``, ``(N, length, u_dim)``.

        The slice order follows skrl's ``sorted(space.keys())`` flattening —
        see the module docstring. This is the only place that encodes it."""
        if obs.shape[-1] != self.flat_dim:
            raise ValueError(
                f"RefWindow.split: expected flat width {self.flat_dim} "
                f"(x_dim={self.x_dim}, u_dim={self.u_dim}, length={self.length}), "
                f"got {obs.shape[-1]}")
        n, L, m = self.x_dim, self.length, self.u_dim
        i = 0
        urefs = obs[:, i:i + L * m].reshape(-1, L, m); i += L * m
        x = obs[:, i:i + n]; i += n
        xrefs = obs[:, i:i + L * n].reshape(-1, L, n)
        return x, xrefs, urefs

    def flatten(self, x: torch.Tensor, xrefs: torch.Tensor, urefs: torch.Tensor) -> torch.Tensor:
        """Inverse of ``split`` — what an env's ``construct_state`` emits so the
        flat tensor matches what skrl would have produced from the Dict."""
        n = x.shape[0]
        return torch.cat([urefs.reshape(n, -1), x, xrefs.reshape(n, -1)], dim=-1)

    def synth_obs(self, x, xref, uref, pool_xr=None, pool_ur=None) -> torch.Tensor:
        """A synthetic observation for the offline losses (C3M's contraction
        certificate, C2RL's pretraining/distillation), which sample states from
        ``get_rollout`` rather than stepping the env.

        Slot 0 is the real current reference, so ``e`` and the feedforward are
        exactly what the certificate is about. Slots 1.. are drawn i.i.d. from
        the pools when given, else the current reference is held across the
        window (a locally-constant reference).

        Prefer passing pools: certifying only against a constant/zero window
        leaves the actor uncertified for the window content it actually meets at
        deployment, and that gap is a real observed divergence — a policy
        pretrained with a zero preview diverged as soon as a live nonzero
        preview was fed in (see project memory).
        """
        b, L = x.shape[0], self.length
        xrefs, urefs = xref.unsqueeze(1), uref.unsqueeze(1)
        if L > 1:
            if pool_xr is not None and pool_ur is not None:
                idx = torch.randint(0, pool_xr.shape[0], (b, L - 1), device=x.device)
                xrefs = torch.cat([xrefs, pool_xr[idx]], dim=1)
                urefs = torch.cat([urefs, pool_ur[idx]], dim=1)
            else:
                xrefs = xrefs.expand(-1, L, -1)
                urefs = urefs.expand(-1, L, -1)
        return self.flatten(x, xrefs, urefs)

    def window_indices(self, t: torch.Tensor, horizon: int) -> torch.Tensor:
        """``(N, length)`` reference indices ``t + k*offset``, clamped to
        ``horizon-1`` (hold the terminal reference — see module docstring)."""
        k = torch.arange(self.length, device=t.device, dtype=torch.long)
        return torch.clamp(t.unsqueeze(-1) + k.unsqueeze(0) * self.offset, max=horizon - 1)

    # ── sizing ──────────────────────────────────────────────────────────── #
    @staticmethod
    def effective_horizon(gamma: float, max_episode_len: int) -> int:
        """``H = 1/(1-gamma)`` in steps, clamped to the episode — how far the
        value function actually looks."""
        if not (0.0 < float(gamma) < 1.0):
            return 1
        return min(int(max_episode_len) - 1, max(1, int(round(1.0 / (1.0 - float(gamma))))))

    @classmethod
    def length_for_horizon(cls, gamma: float, max_episode_len: int, offset: int = 1) -> int:
        """The window length whose span covers the effective horizon.

        ``span = (length-1)*offset >= H``, so ``length = ceil(H/offset) + 1``.
        This is the smallest window for which ``check_markov`` reports no
        horizon shortfall — sizing the observation to the discount rather than
        to a hand-picked constant, which is what lets gamma be swept without
        silently invalidating the layout.
        """
        H = cls.effective_horizon(gamma, max_episode_len)
        return int(math.ceil(H / max(1, int(offset)))) + 1

    # ── Markov check ────────────────────────────────────────────────────── #
    def check_markov(self, gamma: float, max_episode_len: int, *,
                     strict: bool = False) -> str | None:
        """Verify ``{x, xrefs, urefs}`` is a sufficient statistic for the value
        function, and report the reason if it is not.

        The reward ``-||x - xrefs[0]||^2_M - ||u - urefs[0]||^2`` depends only on
        the observation and the action, so the reward is Markov by construction.
        The value is not automatically: ``V(s_t)`` integrates the reward over the
        discount's effective horizon ``H = 1/(1-gamma)`` steps, so every
        reference point within ``H`` must be recoverable from the window.

        Two ways that fails:

        ``span < H``    the window ends before the horizon does, so references
                        the value still weights are simply absent — the agent
                        cannot distinguish two futures that differ past the
                        window. This is the POMDP the old preview existed to
                        fix, and it is what ``strict`` refuses to run with.

        ``offset > 1``  the window skips the intermediate points between its
                        samples. Those are never observed, so the window is a
                        subsampling of the horizon, exact only to the extent the
                        reference is smooth over ``offset`` steps. Reported as a
                        warning, never an error: it is the deliberate trade the
                        offset knob exists to make (wider span per point).

        Returns ``None`` when the window spans the horizon densely, otherwise the
        warning text (also raised when ``strict``).
        """
        if not (0.0 < float(gamma) < 1.0):
            return None
        H = min(int(max_episode_len) - 1, max(1, int(round(1.0 / (1.0 - float(gamma))))))
        span = (self.length - 1) * self.offset
        problems = []
        if span < H:
            problems.append(
                f"window spans {span} steps but the discount's effective horizon is "
                f"{H} steps (gamma={gamma}) — the value function depends on references "
                f"the observation does not contain (POMDP). Need length >= "
                f"{H // self.offset + 1} at offset={self.offset}, or a lower gamma.")
        if self.offset > 1:
            problems.append(
                f"offset={self.offset} subsamples the horizon: {self.offset - 1} of every "
                f"{self.offset} reference points inside the window are never observed. "
                f"Exact only if the reference is smooth over {self.offset} steps.")
        if not problems:
            return None
        msg = "[RefWindow] non-Markov observation: " + " ALSO: ".join(problems)
        if strict and span < H:
            raise ValueError(msg)
        return msg


class Feats:
    """The symmetry feature maps, with the ``sym is None`` fallback in one place.

    Every model previously inlined ``sym.pair_features(x, xref) if sym is not
    None else cat([embed_angles(x), embed_angles(xref)])`` — the same three-line
    branch repeated at ten call sites, each with its own matching width
    computation. Getting the two out of step silently mis-shapes a network, so
    they live together here.
    """

    def __init__(self, x_dim: int, angle_idx: Sequence[int] = (),
                 sym: StateSymmetry | None = None):
        self.x_dim = int(x_dim)
        self.angle_idx = list(angle_idx or [])
        self.sym = sym

    # widths -------------------------------------------------------------- #
    @property
    def single_dim(self) -> int:
        return self.sym.single_dim() if self.sym is not None else embedded_dim(self.x_dim, self.angle_idx)

    @property
    def pair_dim(self) -> int:
        return self.sym.pair_dim() if self.sym is not None else 2 * embedded_dim(self.x_dim, self.angle_idx)

    @property
    def error_dim(self) -> int:
        return self.x_dim

    # maps ---------------------------------------------------------------- #
    def single(self, x: torch.Tensor) -> torch.Tensor:
        """Invariant features of one state block (drops the symmetry directions)."""
        return self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx)

    def pair(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """Complete invariant of the ``(x, xref)`` pair. Broadcasts over any
        leading dims, so it maps a whole ``(N, L, x_dim)`` window at once."""
        if self.sym is not None:
            return self.sym.pair_features(x, xref)
        return torch.cat([embed_angles(x, self.angle_idx),
                          embed_angles(xref, self.angle_idx)], dim=-1)

    def error(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """The canonical-frame tracking error the feedback multiplies (full x_dim)."""
        if self.sym is not None:
            return self.sym.error_features(x, xref)
        return wrap_diff(x - xref, self.angle_idx)

    def sequence(self, x: torch.Tensor, xrefs: torch.Tensor) -> torch.Tensor:
        """``(N, L, pair_dim)`` — every window point expressed relative to the
        current ``x``: relative position, wrapped angle difference, and both
        bodies' invariant features. This is the one input the reference-path
        networks see (the actor's ``W2`` and the critic's ``psi``), so neither
        ever reads an absolute position or a raw wrapping angle.
        """
        return self.pair(x.unsqueeze(1).expand_as(xrefs), xrefs)

class ReferenceState:
    length = offset = 1
    compact = True

    def __init__(self, x_dim, u_dim, parameter_dim, control, *, parameter_bounds=None,
                 time_bound=np.inf, state_names=(), initial_state_parameters=False, angle_idx=()):
        self.x_dim, self.u_dim = int(x_dim), int(u_dim)
        self.parameter_dim = int(parameter_dim)
        # Reference parameters, time and a synthetic-control offset only.
        self.xi_dim = self.parameter_dim + 1 + self.u_dim
        self.control = control
        self.parameter_bounds = parameter_bounds
        self.time_bound = time_bound
        self.state_names = tuple(state_names)
        self.initial_state_parameters = initial_state_parameters
        self.angle_idx = tuple(angle_idx)
        self.preview_length = 0
        self.preview_stride = 1
        self.preview_start = 0
        self.history_length = 0
        self.temporal_context = False

    @property
    def base_dim(self):
        return 2 * self.x_dim + self.xi_dim

    @property
    def flat_dim(self):
        return self.base_dim + self.preview_length * self.x_dim + self.history_length * (self.x_dim + self.u_dim + 1)

    def space(self, x_lo, x_hi, u_lo=None, u_hi=None):
        def array(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)
        lo, hi = array(x_lo), array(x_hi)
        obs_lo, obs_hi = lo.copy(), hi.copy()
        for i in self.angle_idx:
            if hi[i] - lo[i] >= 2 * np.pi - 1e-5:
                obs_lo[i], obs_hi[i] = -np.inf, np.inf
        p_lo, p_hi = self.parameter_bounds if self.parameter_bounds is not None else (np.full(self.parameter_dim, -np.inf), np.full(self.parameter_dim, np.inf))
        xi_lo = np.concatenate((array(p_lo), [0.], np.zeros(self.u_dim))).astype(np.float32)
        xi_hi = np.concatenate((array(p_hi), [self.time_bound], np.zeros(self.u_dim))).astype(np.float32)
        result = gym.spaces.Dict({
            "x": gym.spaces.Box(obs_lo, obs_hi, dtype=np.float32),
            "xref": gym.spaces.Box(obs_lo, obs_hi, dtype=np.float32),
            "xi": gym.spaces.Box(xi_lo, xi_hi, dtype=np.float32),
        })
        if self.preview_length:
            result.spaces['xrefs'] = gym.spaces.Box(np.tile(obs_lo, (self.preview_length, 1)),
                                                   np.tile(obs_hi, (self.preview_length, 1)), dtype=np.float32)
        if self.history_length:
            if u_lo is None or u_hi is None:
                raise ValueError("History observations require applied-control bounds")
            self.control_low, self.control_high = array(u_lo), array(u_hi)
            history_lo = np.concatenate((obs_lo, array(u_lo), [0.])).astype(np.float32)
            history_hi = np.concatenate((obs_hi, array(u_hi), [1.])).astype(np.float32)
            result.spaces['xstates'] = gym.spaces.Box(np.tile(history_lo, (self.history_length, 1)),
                                                     np.tile(history_hi, (self.history_length, 1)), dtype=np.float32)
        result.reference_state = self
        self.feature_low = np.concatenate((lo, xi_lo, lo))
        self.feature_high = np.concatenate((hi, xi_hi, hi))
        return result

    def extract(self, obs):
        if obs.shape[-1] != self.flat_dim:
            raise ValueError(f"Compact observation width {obs.shape[-1]} != {self.flat_dim}")
        n, p = self.x_dim, self.xi_dim
        # Gym/skrl flatten dictionaries in sorted-key order: x, xi, xref.
        return obs[:, :n], obs[:, n + p:n + p + n], obs[:, n:n + p]

    def preview(self, obs):
        end = self.base_dim + self.preview_length * self.x_dim
        return obs[:, self.base_dim:end].reshape(len(obs), self.preview_length, self.x_dim)

    def history(self, obs):
        start = self.base_dim + self.preview_length * self.x_dim
        return obs[:, start:].reshape(len(obs), self.history_length, self.x_dim + self.u_dim + 1)

    def current_control(self, xref, xi):
        p = self.parameter_dim
        offset = xi[:, p + 1:p + 1 + self.u_dim]
        return self.control(xref, xi[:, :p], xi[:, p]) + offset

    def split(self, obs):
        x, xref, xi = self.extract(obs)
        return x, xref.unsqueeze(1), self.current_control(xref, xi).unsqueeze(1)

    def flatten(self, x, xrefs, urefs, xi=None, preview=None, history=None):
        xref = xrefs[:, 0] if xrefs.ndim == 3 else xrefs
        synthesis = xi is None
        if xi is None:
            # Certificate synthesis samples arbitrary current (x, xref, uref)
            # triples rather than a future trajectory. Encode that control.
            xi = x.new_zeros(len(x), self.xi_dim)
            uref = urefs[:, 0] if urefs.ndim == 3 else urefs
            p = self.parameter_dim
            xi[:, p + 1:p + 1 + self.u_dim] = uref - self.current_control(xref, xi)
        parts = [x, xi, xref]
        if self.preview_length:
            if preview is None:
                if not synthesis:
                    raise ValueError("Compact preview observations require their reference window")
                preview = xref[:, None].expand(-1, self.preview_length, -1)
            if preview.shape != (len(x), self.preview_length, self.x_dim):
                raise ValueError("Reference preview shape disagrees with the observation space")
            parts.append(preview.reshape(len(x), -1))
        if self.history_length:
            if history is None:
                if not synthesis:
                    raise ValueError("Temporal observations require their raw state/control history")
                history = x.new_zeros(len(x), self.history_length, self.x_dim + self.u_dim + 1)
            if history.shape != (len(x), self.history_length, self.x_dim + self.u_dim + 1):
                raise ValueError("History shape disagrees with the observation space")
            parts.append(history.reshape(len(x), -1))
        return torch.cat(parts, -1)

    def synth_obs(self, x, xref, uref, pool_xr=None, pool_ur=None, **kwargs):
        return self.flatten(x, xref, uref)

    def check_markov(self, *args, **kwargs):
        # Reference parameters/time replace the finite future window. This
        # checks reference completeness, not observability of hidden plant noise.
        return None


def affine_observation_scaling(low, high):
    """Finite centers/scales; fixed coordinates use one and tiny spans are floored."""
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=low.dtype, device=low.device)
    if low.shape != high.shape or torch.isnan(low).any() or torch.isnan(high).any() or (high < low).any():
        raise ValueError("Observation bounds must have matching shapes and be ordered without NaNs")
    bounded = torch.isfinite(low) & torch.isfinite(high)
    lo, hi = torch.where(bounded, low, 0.), torch.where(bounded, high, 0.)
    center = lo / 2 + hi / 2
    scale = torch.where(bounded & (hi > lo), (hi / 2 - lo / 2).clamp_min(1e-6), torch.ones_like(lo))
    return center, scale


class TrackingFeatures(torch.nn.Module):
    """Relative translation and periodic physical/reference configuration."""
    def __init__(self, window, angle_idx=()):
        super().__init__()
        self.window, self.names = window, window.state_names
        named = [i for i, name in enumerate(self.names or ()) if name in {"yaw", "roll", "pitch"}]
        self.angle_idx = sorted(set(angle_idx) | set(getattr(window, "angle_idx", ())) | set(named))
        self.position = tuple(i for i, name in enumerate(self.names or ()) if name in {"pos_x", "pos_y", "pos_z"})
        self.reference_indices = tuple(i for i in range(window.x_dim) if i not in self.position)
        n = window.x_dim
        self.register_buffer("_raw_tracking_features", torch.tensor(True))
        self.register_buffer("_explicit_tracking_input_blocks", torch.tensor(True))
        self.register_buffer("_relative_position_features", torch.tensor(bool(self.position)))
        self.register_buffer("_periodic_tracking_features", torch.tensor(True))
        self.register_buffer("_legacy_position_rotation", torch.tensor(False))
        self.state_feature_dim = n + len(self.angle_idx)
        self.context_feature_dim = window.xi_dim + (len(self.angle_idx) if window.initial_state_parameters else 0)
        self.output_dim = 2 * self.state_feature_dim + self.context_feature_dim

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if bool(state_dict.pop(prefix + "_episode_local_coordinates", False)):
            raise ValueError("Checkpoint uses removed coordinates; load it with its archived source snapshot")
        state_dict.setdefault(prefix + "_legacy_position_rotation", self._legacy_position_rotation.new_tensor(False))
        if not bool(state_dict.get(prefix + "_periodic_tracking_features", False)):
            raise ValueError("Checkpoint predates periodic tracking features; load it with its archived source snapshot")
        if not bool(state_dict.get(prefix + "_raw_tracking_features", False)):
            raise ValueError("Checkpoint uses fixed feature scaling; load it with its archived source snapshot")
        if not bool(state_dict.get(prefix + "_explicit_tracking_input_blocks", False)):
            raise ValueError("Checkpoint uses the previous tracking input layout; load it with its archived source snapshot")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def error(self, x, xref):
        return wrap_diff(x - xref, self.angle_idx)

    def feedback_error(self, x, xref):
        return embed_angles(x, self.angle_idx) - embed_angles(xref, self.angle_idx)

    def forward(self, obs):
        x, xref, xi = self.window.extract(obs)
        n = self.window.x_dim
        state, reference = x.clone(), xref.clone()
        if self.position:
            pos = list(self.position)
            state[:, pos] = x[:, pos] - xref[:, pos]
            reference[:, pos] = 0.
        # Other reference generators may require a persistent target in their context.
        if self.window.initial_state_parameters:
            initial = xi[:, :n].clone()
            if self.position:
                initial[:, list(self.position)] -= xref[:, list(self.position)]
            xi = torch.cat((embed_angles(initial, self.angle_idx), xi[:, n:]), -1)
        return torch.cat((embed_angles(state, self.angle_idx), embed_angles(reference, self.angle_idx), xi), -1)

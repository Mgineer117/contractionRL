"""Car (Dubins-like) tracking environment (ported to batched PyTorch).

State  x = [p_x, p_y, theta, v]
Control u = [omega, a]
Control-affine dynamics:
    f(x) = [v cos(theta), v sin(theta), 0, 0]
    B(x) = [[0,0],[0,0],[1,0],[0,1]]
"""

from __future__ import annotations

import math

import torch

from ..common.analytical_env import BaseEnv

STATE_NAMES = ("pos_x", "pos_y", "yaw", "vel")
# angle_idx / pos_dimension are derived from STATE_NAMES by BaseEnv (see
# agents/utils/system_dynamics/state.py) -- the names are the single source of truth for
# which dims wrap, which translate, and which co-rotate under a yaw rotation.

v_l, v_h = 1.0, 2.0
# The reference start position is randomized over the whole POS_SPREAD box, not
# pinned near the origin. Why: f(x) = [v cos(theta), v sin(theta), 0, 0] and B are
# independent of (p_x, p_y), so the tracking problem is exactly translation
# invariant — but the policy backbone's gain nets (CLActor's w1/w2 read the
# Absolute [x, xref] pair; only the feedback error e = wrap_diff(x - xref) is
# relative) have no such structure and must learn it from data. With a +-2.0
# start box the reference spends the episode drifting out to ~22 m while tracking
# error has already decayed, so every large-error training sample sits within a
# few metres of the origin and the learned gains do not generalize down the path
# (measured: translating a whole (x0, xref) pair by 3 m — a symmetry of the
# dynamics — changed the eval failure rate from 4% to 30%). Spreading the start
# position decorrelates |xref| from "how much error is left".
POS_SPREAD = 20.0
# Position bound must comfortably contain the reference rollout: xref's velocity is
# fixed at 1.5 (see XREF_INIT below — sample_reference_controls never drives the
# acceleration control) and heading only gets steered by omega, so over the full
# time_bound the reference can travel up to ~v*time_bound in a single direction if
# heading holds roughly constant. Without enough margin here, _rollout_reference's
# torch.clamp(next_x_wrapped, X_MIN, X_MAX) silently clips the reference position mid
# -trajectory, which corrupts tracking-error/reward at the boundary. So the bound is
# POS_SPREAD (worst-case start) + 1.5 * 15.0 = 22.5 (worst-case travel) + margin.
POS_BOUND = POS_SPREAD + 1.5 * 15.0 + 2.5
X_MIN = [-POS_BOUND, -POS_BOUND, -math.pi, v_l]
X_MAX = [POS_BOUND, POS_BOUND, math.pi, v_h]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# rule.md Step 3, checked 2026-08-27 against the certified metric: the local
# closed-loop rate lam(x) is 0.4420 at min, median AND max over the whole X box
# -- a 1.00x spread, with |corr| <= 0.021 against every state dim. The rate is
# NOT state-dependent here, so Step 3's low-rate rule does not apply and the
# init boxes are arbitrary within X. No X_INIT: reset perturbs by XE_INIT, which
# is what keeps e(0) at ~1.0 instead of the box width.
XREF_INIT_MIN = [-POS_SPREAD, -POS_SPREAD, -1.0, 1.5]
XREF_INIT_MAX = [POS_SPREAD, POS_SPREAD, 1.0, 1.5]
# Velocity perturbation is +-0.5, not +-1.0, and the reason is the box.
#
# x_0 = xref_0 + xe_0 is clamped to [X_MIN, X_MAX] (env_base.reset), and xref's
# velocity is PINNED at 1.5 against a [1.0, 2.0] velocity box. At +-1.0 the sum
# spanned [0.5, 2.5], so the clamp folded a quarter of the mass onto v = 1.0 and
# another quarter onto v = 2.0: measured 50.6% of episodes started pinned exactly
# on a box face. That is a worse failure than the out-of-box starts the clamp was
# added to stop -- half the runs shared two identical initial velocities, and the
# CMG is least accurate at the box edge where it has data on one side only.
#
# +-0.5 makes the sum exactly [1.0, 2.0]: uniform, interior, nothing to clamp.
# The other three components keep +-1.0 -- position has POS_BOUND ~22 of room and
# yaw wraps, so neither ever reaches its bound.
XE_INIT_MIN = [-1.0, -1.0, -1.0, -0.5]
XE_INIT_MAX = [1.0, 1.0, 1.0, 0.5]
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]
# u = [omega (rad/s), a (m/s^2)]; the applied box is 2x this (analytical_env.py:37).
# omega: an RC-car-scale platform (wheelbase L=0.3 m, max steer 30 deg) at the
# reference speed v=1.5 gives omega_max = v*tan(30)/L = 2.9 rad/s, so +-3 applied.
# The old +-6 applied meant a 0.25 m turn radius at 1.5 m/s -- not a car.
# a: passenger-car forward accel maxes near 3 m/s^2 (braking is larger, but the
# box is symmetric), so +-3 applied.
UREF_MIN = [-1.5, -1.5]
UREF_MAX = [1.5, 1.5]

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 4, "num_dim_control": 2,
    "dt": 0.03, "time_bound": 15.0,
    "q": 1.0, "r": 0.0,
}


class CarEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "car"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = x[:, 3] * torch.cos(x[:, 2])
        f[:, 1] = x[:, 3] * torch.sin(x[:, 2])
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = 1.0
        B[:, 3, 1] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            val = weight[:, 0] * torch.sin(torch.as_tensor(freq * _t / self.time_bound * 2 * math.pi, device=self.device))
            uref[:, 0] += val
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def set_held_out_mode(self, seed: int | None, n_trajectories: int = 64) -> None:
        """Fix a reproducible bank of ``n_trajectories`` reference-trajectory
        weight vectors (the mixing coefficients over the 10 fixed sinusoidal
        steering components — see ``system_reset``), drawn from a dedicated
        ``torch.Generator`` seeded independently of the run's own training
        seed/RNG stream. Bank parameters must be compared with training's
        parameters to establish disjointness; distinct seeds alone do not prove
        that a trajectory was unseen. Each new episode advances a persistent
        cursor through this bank, including sequential and partial resets.
        Rejected initial conditions retry the same bank row. Fixed-reference
        slots keep their assigned row until the mode is reconfigured.
        ``seed=None`` reverts to that historic
        i.i.d.-random behavior exactly (disables held-out mode).

        Only the steering weights are fixed — the initial state
        (``define_initial_state``: start position/heading/velocity, initial
        tracking error) still draws from the ordinary global RNG every reset,
        same as training. The "goal" being held out is the reference PATH
        shape, not the starting condition.
        """
        if isinstance(n_trajectories, bool) or not isinstance(n_trajectories, int) or n_trajectories < 1:
            raise ValueError("n_trajectories must be a positive integer")
        weights = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            weights = torch.randn(n_trajectories, 10, len(UREF_MIN), generator=gen)
            weights = weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        self._held_out_weights = None if weights is None else weights.to(self.device)
        self._held_out_cursor = 0
        self._held_out_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._held_out_reset_active = False
        self._ref_pool = None
        self._ref_fixed.zero_()
        if self.ref_groups:
            self._group_minted.zero_()

    def _safe_system_reset(self, env_ids):
        if getattr(self, "_held_out_weights", None) is None:
            return super()._safe_system_reset(env_ids)
        fresh = ~self._ref_fixed[env_ids] if self.fix_ref_trajectories else torch.ones_like(env_ids, dtype=torch.bool)
        new_ids = env_ids[fresh]
        previous_cursor = self._held_out_cursor
        previous_indices = self._held_out_indices[env_ids].clone()
        self._held_out_indices[new_ids] = (
            self._held_out_cursor + torch.arange(len(new_ids), device=self.device)
        ) % len(self._held_out_weights)
        self._held_out_cursor += len(new_ids)
        self._held_out_reset_active = True
        try:
            return super()._safe_system_reset(env_ids)
        except Exception:
            self._held_out_cursor = previous_cursor
            self._held_out_indices[env_ids] = previous_indices
            raise
        finally:
            self._held_out_reset_active = False

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        bank = getattr(self, "_held_out_weights", None)
        if bank is not None:
            if self._held_out_reset_active:
                idx = self._held_out_indices[env_ids]
            else:
                idx = (self._held_out_cursor + torch.arange(n, device=self.device)) % bank.shape[0]
                self._held_out_cursor += n
            weights = bank[idx]
        else:
            weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
            weights = weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

"""Segway tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..common.env_base import BaseEnv

# Amplitude of the commanded velocity profile the reference tracks (m/s). Keeps
# |pos_x| well inside X_MIN/X_MAX over an episode so _rollout_reference never
# clamps the reference -- a clamped reference violates its own dynamics.
VEL_REF_SCALE = 0.15

# Denote angle indices to handle smooth transition
STATE_NAMES = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
# angle_idx / pos_dimension are DERIVED from STATE_NAMES by BaseEnv (see
# agents/skrl/state_symmetry.py) -- the names are the single source of truth for
# which dims wrap, which translate, and which co-rotate under a yaw rotation.

# X bounds
X_MIN = [-5.0, -math.pi / 3, -1.0, -math.pi]
X_MAX = [5.0, math.pi / 3, 1.0, math.pi]

# Initial reference state bounds
XREF_INIT_MIN = [0.0, 0, 0.0, 0]
XREF_INIT_MAX = [0.0, 0, 0.0, 0]

# Initial perturbation to the reference state, i.e. the reset error every
# algorithm is scored on (define_initial_state: x_0 = xref_0 + xe_0).
#
# pitch/pitch_rate are NOT +-pi/3 and +-pi. That box contained initial
# conditions this plant cannot recover from with its own control box: the
# segway's open-loop unstable pole is 2.82 rad/s (a 0.35 s tip-over time
# constant) and u in [-6, 6] buys 12.1 rad/s^2 of pitch authority, so a reset at
# 60 deg of tilt AND 180 deg/s of pitch rate is already past the point of no
# return. Measured with an LQR (Q=I, R=0.1I) clipped to the env's own box:
# 135/256 resets recovered balance at +-pi/3, +-pi -- i.e. nearly half of every
# algorithm's episodes were unrecoverable at t=0, which is noise in the mean of
# every Stability/* metric, not a difficulty setting.
#
# +-pi/12 (15 deg) and +-pi/12 (15 deg/s) is the largest box measured to give
# 256/256 recovery, and it also drops peak |u| demand from 84.4 to 17.3.
# pos/vel stay at full range: they only enter the position error, which cannot
# tip the plant over.
XE_INIT_MIN = [-1.0, -math.pi / 12, -0.5, -math.pi / 12]
XE_INIT_MAX = [1.0, math.pi / 12, 0.5, math.pi / 12]

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds
UREF_MIN = [-3.0]
UREF_MAX = [3.0]

ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 4,
    "num_dim_control": 1,
    "dt": 0.03,
    "time_bound": 15.0,
    "q": 1.0,
    "r": 0.0,
}

class SegwayEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "segway"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = v
        f[:, 1] = omega
        f[:, 2] = (
            torch.cos(theta) * (9.8 * torch.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (omega**2) * torch.sin(theta)
        ) / (torch.cos(theta) - 24.7)
        f[:, 3] = (
            -58.8 * v * torch.cos(theta)
            - 243.5 * v
            - torch.sin(theta) * (208.3 + (omega**2) * torch.cos(theta))
        ) / (torch.cos(theta) ** 2 - 24.7)
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = (-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) - 24.7)
        B[:, 3, 0] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7)
        return B

    def _B_null_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7)
        Bbot[:, 3, 2] = -(-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) - 24.7)
        return Bbot

    def _balance_gain(self):
        """LQR gain about the upright equilibrium, cached. Position is left FREE.

        A segway has one input for four states and cannot hold position while
        balancing, so the reference controller regulates (pitch, velocity,
        pitch_rate) only and lets pos_x follow from the velocity profile. The
        position column is zeroed out of the error rather than out of ``Q``, so
        the gain itself stays the plain CARE solution.
        """
        if getattr(self, "_K_bal", None) is None:
            from scipy.linalg import solve_continuous_are
            x0 = torch.zeros(1, self.num_dim_x, device=self.device)
            x0.requires_grad_(True)
            f, B, _ = self.get_f_and_B(x0)
            A = torch.zeros(self.num_dim_x, self.num_dim_x, device=self.device)
            for i in range(self.num_dim_x):
                g = torch.autograd.grad(f[0, i], x0, retain_graph=True)[0]
                A[i] = g[0]
            A_np = A.detach().cpu().numpy().astype(float)
            B_np = B[0].detach().cpu().numpy().astype(float)
            Q = np.diag([0.0, 10.0, 1.0, 1.0])          # position unweighted
            R = np.eye(self.num_dim_control) * 0.1
            P = solve_continuous_are(A_np, B_np, Q, R)
            self._K_bal = torch.as_tensor((np.linalg.solve(R, B_np.T @ P)),
                                         dtype=torch.float32, device=self.device)
        return self._K_bal

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        """Reference control that CLOSES A LOOP around the reference state.

        The segway is open-loop unstable (2.82 rad/s pole), so an open-loop uref
        makes the REFERENCE ITSELF tip over -- which is why this env used to pin
        uref to 0 and hand back a constant reference at the origin. Instead the
        reference tracks a smooth velocity profile built from ``freqs`` under a
        fixed LQR balance gain:

            v_des(t) = sum_i w_i sin(2 pi f_i t / T)
            uref     = -K_bal * (xref_t - [pos_free, 0, v_des, 0])

        ``(xref, uref)`` stays DYNAMICALLY FEASIBLE by construction because
        ``_rollout_reference`` integrates the true plant with exactly this uref --
        closing the loop changes which trajectory is generated, never whether it
        satisfies the dynamics.
        """
        n = weights.shape[0]
        xref_t = infos.get("xref_t", infos["xref_0"])
        v_des = torch.zeros(n, device=self.device)
        for i, freq in enumerate(freqs):
            v_des = v_des + weights[:, i, 0] * math.sin(
                freq * _t / self.time_bound * 2 * math.pi)
        x_des = xref_t.clone()
        x_des[:, 1] = 0.0            # upright
        x_des[:, 2] = v_des          # track the velocity profile
        x_des[:, 3] = 0.0            # no pitch rate
        uref = -(xref_t - x_des) @ self._balance_gain().T
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        # A real reference, not the constant origin. Possible only because
        # sample_reference_controls closes a loop (see there); the old freqs=[]
        # plus a 0.0* weight scale existed because an open-loop uref tips the
        # reference over. VEL_REF_SCALE bounds the commanded velocity so the
        # reference stays inside X_MIN/X_MAX without _rollout_reference's clamp
        # engaging -- a clamped reference is NOT dynamically feasible.
        freqs = list(range(1, 6))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        weights = VEL_REF_SCALE * weights / torch.sqrt(
            (weights**2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

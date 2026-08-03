"""Segway tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

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

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        xref_0 = infos["xref_0"]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        uref[:, 0] = 10.2 * xref_0[:, 2] / 47.9
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            term = weight[:, 0] * ((-1) ** int(freq * _t / self.time_bound)) * math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 0] += term
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = []
        n = len(env_ids)
        if len(freqs) > 0:
            weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
            weights = 0.0 * weights / torch.sqrt((weights**2).sum(dim=1, keepdim=True))
        else:
            weights = torch.zeros(n, 0, len(UREF_MIN), device=self.device)
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

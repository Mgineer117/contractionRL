"""Segway tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import torch

from contractionRL.tasks.direct.common.analytical_env import BaseEnv

# Denote angle indices to handle smooth transition
STATE_NAMES = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
# State names determine wrapping angles and position dimensions.

# Restrict tilt to avoid poorly conditioned corners of the metric domain.
PITCH_LIM = 0.90
X_MIN = [-5.0, -PITCH_LIM, -1.0, -math.pi]
X_MAX = [5.0, PITCH_LIM, 1.0, math.pi]

# Use the state box for optional excursion termination.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Initialize references upright; sustained tilt accelerates the base out of its box.
XREF_INIT_MIN = [0.0, 0, 0.0, 0]
XREF_INIT_MAX = [0.0, 0, 0.0, 0]

# Perturbations leave room for recovering tilt and pitch rate.
XE_INIT_MIN = [-1.0, -0.15, -0.15, -0.5]
XE_INIT_MAX = [1.0, 0.15, 0.15, 0.5]

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# Reference control is normalized wheel torque; BaseEnv doubles the applied bound.
# Retain the actuator range used by the shipped metric dataset.
UREF_MIN = [-24.0]
UREF_MAX = [24.0]

# Mirror pitch, velocity and pitch rate together to sample opposite-sign tilt/rate
# lobes, leaving margin for recovery inside the state box.
X_INIT_MIN = [-2.0, 0.65, 0.0, -1.50]
X_INIT_MAX = [2.0, 0.78, 0.05, -0.60]
X_INIT_SIGN_DIMS = [1, 2, 3]

ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "x_init_min": X_INIT_MIN,
    "x_init_max": X_INIT_MAX,
    "x_init_sign_dims": X_INIT_SIGN_DIMS,
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
    # Allow 60 seconds for the slow certified decay while retaining the 0.03 s step.
    "time_bound": 60.0,
    # Run full episodes to measure divergence; excursion truncation is disabled.
    # Failed rollouts can therefore contribute off-domain transitions.
    "terminate_out_of_box": False,
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

    def sample_reference_controls(self, freqs, weights, _t, infos):
        n = weights.shape[0]
        xref_0 = infos["xref_0"]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        uref[:, 0] = 10.2 * xref_0[:, 2] / 47.9
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            term = weight[:, 0] * torch.cos(torch.floor(torch.as_tensor(freq * _t / self.time_bound, device=self.device)) * math.pi) * torch.sin(torch.as_tensor(freq * _t / self.time_bound * 2 * math.pi, device=self.device))
            uref[:, 0] += term
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

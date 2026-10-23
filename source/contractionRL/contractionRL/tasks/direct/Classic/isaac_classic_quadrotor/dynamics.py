"""Quadrotor tracking environment."""

from __future__ import annotations

import math

import torch

from contractionRL.tasks.direct.common.analytical_env import BaseEnv

STATE_NAMES = (
    "pos_x", "pos_y", "pos_z",
    "vel_x_w", "vel_y_w", "vel_z_w",
    "thrust", "roll", "pitch", "yaw",
)
# State names determine wrapping angles and position dimensions.

G = 9.81

_X10_LIM = math.pi / 3
_X9_LIM = math.pi / 3
_X8_LIM = math.pi / 3
_X7_LOW = 0.5 * G
_X7_HIGH = 2 * G
_X4_LIM = 1.5
_X5_LIM = 1.5
_X6_LIM = 1.5

X_MIN = [-30.0, -30.0, 0.15, -_X4_LIM, -_X5_LIM, -_X6_LIM, _X7_LOW, -_X8_LIM, -_X9_LIM, -_X10_LIM]
X_MAX = [30.0, 30.0, 30.0, _X4_LIM, _X5_LIM, _X6_LIM, _X7_HIGH, _X8_LIM, _X9_LIM, _X10_LIM]

# Use the state box for optional excursion termination.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Initialize nominal references near hover.
XREF_INIT_MIN = [-5.0, -5.0, 3.0, -0.2, -0.2, -0.2, G, 0.0, 0.0, 0.0]
XREF_INIT_MAX = [5.0, 5.0, 8.0, 0.2, 0.2, 0.2, G, 0.0, 0.0, 0.0]

# Use tighter horizontal and attitude perturbations for the relative-degree-three chain.
XE_INIT_MIN = [-0.15, -0.15, -0.3, -0.15, -0.2, -0.3, -0.3, -0.15, -0.15, -0.5]
XE_INIT_MAX = [0.15, 0.15, 0.3, 0.15, 0.2, 0.3, 0.3, 0.15, 0.15, 0.5]

_lim = 1.0
XE_MIN = [-_lim] * 10
XE_MAX = [_lim] * 10

# Controls are mass-normalized thrust rate (m/s^3) and Euler-angle rates (rad/s).
# BaseEnv doubles these reference bounds for applied controls.
UREF_MIN = [-30.0, -5.0, -5.0, -5.0]
UREF_MAX = [30.0, 5.0, 5.0, 5.0]

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 10, "num_dim_control": 4,
    "dt": 0.025, "time_bound": 10.0,
    "q": 1.0, "r": 0.0,
}

class QuadrotorEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "quadrotor"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        force, theta_x, theta_y = x[:, 6], x[:, 7], x[:, 8]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = x[:, 3]
        f[:, 1] = x[:, 4]
        f[:, 2] = x[:, 5]
        f[:, 3] = -force * torch.sin(theta_y)
        f[:, 4] = force * torch.cos(theta_y) * torch.sin(theta_x)
        # Isaac world Z is up: thrust supports altitude against gravity.
        f[:, 5] = force * torch.cos(theta_y) * torch.cos(theta_x) - G
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 6, 0] = 1.0
        B[:, 7, 1] = 1.0
        B[:, 8, 2] = 1.0
        B[:, 9, 3] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            s_val = torch.sin(torch.as_tensor(freq * _t / self.time_bound * 2 * math.pi, device=self.device))
            uref[:, 0] += weight[:, 0] * s_val
            uref[:, 1] += weight[:, 1] * s_val
            uref[:, 2] += weight[:, 2] * s_val
            uref[:, 3] += weight[:, 3] * s_val
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        # Keep excitation small enough to avoid clipping nominal reference trajectories.
        weights = 0.005 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

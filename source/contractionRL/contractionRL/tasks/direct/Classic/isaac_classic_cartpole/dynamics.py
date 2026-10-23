"""Cartpole tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import torch

from contractionRL.tasks.direct.common.analytical_env import BaseEnv

# Cartpole parameters
mc = 1.0
mp = 1.0
g = 9.81
l = 1.0

# Denote angle indices to handle smooth transition
STATE_NAMES = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
# State names determine wrapping angles and position dimensions.

# X bounds
X_MIN = [-5.0, -math.pi / 3, -1.0, -1]
X_MAX = [5.0, math.pi / 3, 1.0, 1]

# Use the state box for optional excursion termination.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Initial reference state bounds
XREF_INIT_MIN = [0.0, 0, 0.0, 0]
XREF_INIT_MAX = [0.0, 0, 0.0, 0]

# Initial perturbation to the reference state
lim = 0.3
XE_INIT_MIN = [-lim, -lim, -lim, -lim]
XE_INIT_MAX = [lim, lim, lim, lim]

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# Reference control is cart force (N); BaseEnv doubles it for applied controls.
# The applied 48 N bound exceeds the 34 N needed to hold a pole at pi/3.
UREF_MIN = [-24.0]
UREF_MAX = [24.0]

# Mirror pitch and pitch rate together so starts tilt toward upright.
# Leave margin for the recovery transient inside the state box.
X_INIT_MIN = [-2.0, 0.25, -0.02, -0.15]
X_INIT_MAX = [2.0, 0.40, 0.02, -0.02]
X_INIT_SIGN_DIMS = [1, 3]

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
    "time_bound": 15.0,
    # Run full episodes to measure divergence; excursion truncation is disabled.
    # Failed rollouts can therefore contribute off-domain transitions.
    "terminate_out_of_box": False,
    "q": 1.0,
    "r": 0.0,
}

class CartPoleEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "cartpole"
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
            mp * torch.sin(theta) * (l * (omega**2) - g * torch.cos(theta))
            / (mc + mp * (torch.sin(theta) ** 2))
        )
        f[:, 3] = (
            (mp * l * (omega**2) * torch.cos(theta) * torch.sin(theta) - (mc + mp) * g * torch.sin(theta))
            / l / (mc + mp * (torch.sin(theta) ** 2))
        )
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = 1 / (mc + mp * (torch.sin(theta) ** 2))
        B[:, 3, 0] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
        return B

    def _B_null_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
        Bbot[:, 3, 2] = -1.0 / (mc + mp * (torch.sin(theta) ** 2))
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

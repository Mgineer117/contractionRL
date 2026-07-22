"""Turtlebot tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math
import torch

from ..common.env_base import BaseEnv

# TURTLEBOT PARAMETERS
k1, k2, k3 = 1.0, 1.0, 1.0

# Denote angle indices to handle smooth transition
ANGLE_IDX = [2]

# X bounds
X_MIN = [-10.0, -10.0, 0]
X_MAX = [10.0, 10.0, 2 * math.pi]

# Initial reference state bounds
XREF_INIT_MIN = [-1.0, -1.0, (1 / 2) * math.pi]
XREF_INIT_MAX = [1.0, 1.0, (3 / 2) * math.pi]

# Initial reference state perturbation bounds
XE_INIT_MIN = [-0.5, -0.5, -(1 / 4) * math.pi]
XE_INIT_MAX = [0.5, 0.5, (1 / 4) * math.pi]

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim]
XE_MAX = [lim, lim, lim]

# reference control bounds
UREF_MIN = [0.0, -1.82]
UREF_MAX = [0.22, 1.82]

ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 3,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.05,
    "time_bound": 30.0,
    "q": 1.0,
    "r": 0.0,
}

class TurtlebotEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "turtlebot"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        f = self._zeros((n, self.num_dim_x), x)
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        theta = x[:, 2]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 0, 0] = k1 * torch.cos(theta)
        B[:, 1, 0] = k2 * torch.sin(theta)
        B[:, 2, 1] = k3
        return B

    def _B_null_logic(self, x):
        n = x.shape[0]
        theta = x[:, 2]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = k2 * torch.sin(theta) * k3
        Bbot[:, 1, 0] = -k1 * torch.cos(theta) * k3
        Bbot[:, 2, 0] = 0.0
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        
        # Turtlebot samples linear velocity dynamically per env based on max bounds
        linear_velocity = self.UREF_MAX[0] * (0.2 + 0.6 * torch.rand(n, device=self.device))
        uref[:, 0] = linear_velocity
        
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            term = weight[:, 1] * math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 1] += term
            
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        weights = weights / torch.sqrt((weights**2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

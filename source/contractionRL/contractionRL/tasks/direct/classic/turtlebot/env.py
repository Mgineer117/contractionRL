"""Turtlebot tracking environment (ported from CAC-dev ``envs/xyD/turtlebot.py``)."""

from __future__ import annotations

import numpy as np

from ..common.env_base import BaseEnv

# TURTLEBOT PARAMETERS
k1, k2, k3 = 1.0, 1.0, 1.0

# Denote angle indices to handle smooth transition
ANGLE_IDX = [2]

# X bounds
X_MIN = np.array([-10.0, -10.0, 0]).reshape(-1, 1)
X_MAX = np.array([10.0, 10.0, 2 * np.pi]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-1.0, -1.0, (1 / 2) * np.pi])
XREF_INIT_MAX = np.array([1.0, 1.0, (3 / 2) * np.pi])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.array([-0.5, -0.5, -(1 / 4) * np.pi])
XE_INIT_MAX = np.array([0.5, 0.5, (1 / 4) * np.pi])

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([0.0, -1.82]).reshape(-1, 1)
UREF_MAX = np.array([0.22, 1.82]).reshape(-1, 1)

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
        sample_mode: str = "uniform",
        reward_mode: str = "default",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "turtlebot"
        super().__init__(self._build_cfg(
            ENV_CONFIG, sample_mode=sample_mode, reward_mode=reward_mode, time_bound=time_bound, dt=dt,
        ))

    def _f_logic(self, x, lib):
        n = x.shape[0]
        f = self._zeros((n, self.num_dim_x), x, lib)
        return f

    def _B_logic(self, x, lib):
        n = x.shape[0]
        theta = x[:, 2]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x, lib)
        B[:, 0, 0] = k1 * lib.cos(theta)
        B[:, 1, 0] = k2 * lib.sin(theta)
        B[:, 2, 1] = k3
        return B

    def _B_null_logic(self, x, n, lib):
        theta = x[:, 2]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x, lib)
        Bbot[:, 0, 0] = k2 * lib.sin(theta) * k3
        Bbot[:, 1, 0] = -k1 * lib.cos(theta) * k3
        Bbot[:, 2, 0] = 0.0
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        linear_velocity = UREF_MAX[0] * np.random.uniform(0.2, 0.8)
        uref = np.array([linear_velocity.squeeze(), 0])
        for freq, weight in zip(freqs, weights):
            uref += np.array([
                0.0,
                weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
            ])
        if add_noise:
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)
        return np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())

    def system_reset(self):
        xref_0, xe_0, x_0 = self.define_initial_state()
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        xref_arr, uref_arr, n = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, n

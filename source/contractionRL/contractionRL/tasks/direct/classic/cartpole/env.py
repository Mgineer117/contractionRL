"""Cartpole tracking environment (ported from CAC-dev ``envs/xD/cartpole.py``)."""

from __future__ import annotations

import numpy as np

from ..common.env_base import BaseEnv

# CARTPOLE PARAMETERS
mc = 1.0
mp = 1.0
g = 9.81
l = 1.0

# Denote angle indices to handle smooth transition
ANGLE_IDX = [1]

# X bounds
X_MIN = np.array([-5.0, -np.pi / 3, -1.0, -1]).reshape(-1, 1)
X_MAX = np.array([5.0, np.pi / 3, 1.0, 1]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([0.0, 0, 0.0, 0])
XREF_INIT_MAX = np.array([0.0, 0, 0.0, 0])

# Initial perturbation to the reference state
lim = 0.3
XE_INIT_MIN = np.array([-lim, -lim, -lim, -lim])
XE_INIT_MAX = np.array([lim, lim, lim, lim])

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([-3.0]).reshape(-1, 1)
UREF_MAX = np.array([3.0]).reshape(-1, 1)

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
    "num_dim_x": 4,
    "num_dim_control": 1,
    "pos_dimension": 1,
    "dt": 0.03,
    "time_bound": 15.0,
    "q": 1.0,
    "r": 0.0,
}

class CartPoleEnv(BaseEnv):
    def __init__(
        self,
        sample_mode: str = "uniform",
        reward_mode: str = "default",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "cartpole"
        super().__init__(self._build_cfg(
            ENV_CONFIG, sample_mode=sample_mode, reward_mode=reward_mode, time_bound=time_bound, dt=dt,
        ))

    def _f_logic(self, x, lib):
        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]
        f = self._zeros((n, self.num_dim_x), x, lib)
        f[:, 0] = v
        f[:, 1] = omega
        f[:, 2] = (
            mp * lib.sin(theta) * (l * (omega**2) - g * lib.cos(theta))
            / (mc + mp * (lib.sin(theta) ** 2))
        )
        f[:, 3] = (
            (mp * l * (omega**2) * lib.cos(theta) * lib.sin(theta) - (mc + mp) * g * lib.sin(theta))
            / l / (mc + mp * (lib.sin(theta) ** 2))
        )
        return f

    def _B_logic(self, x, lib):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x, lib)
        B[:, 2, 0] = 1 / (mc + mp * (lib.sin(theta) ** 2))
        B[:, 3, 0] = lib.cos(theta) / l / (mc + mp * (lib.sin(theta) ** 2))
        return B

    def _B_null_logic(self, x, n, lib):
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x, lib)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = lib.cos(theta) / l / (mc + mp * (lib.sin(theta) ** 2))
        Bbot[:, 3, 2] = -1.0 / (mc + mp * (lib.sin(theta) ** 2))
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        xref_0 = infos["xref_0"]
        uref = np.array([10.2 * xref_0[2] / 47.9])
        for freq, weight in zip(freqs, weights):
            uref += np.array([
                weight[0] * (-1) ** (int(freq * _t / self.time_bound))
                * np.sin(freq * _t / self.time_bound * 2 * np.pi),
            ])
        if add_noise:
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)
        return np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())

    def system_reset(self):
        xref_0, xe_0, x_0 = self.define_initial_state()
        freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN)) if freqs else np.zeros((0, len(UREF_MIN)))
        if len(freqs) > 0:
            weights = (0.0 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        xref_arr, uref_arr, n = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, n

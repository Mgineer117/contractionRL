"""Segway tracking environment (ported from CAC-dev ``envs/xD/segway.py``)."""

from __future__ import annotations

import numpy as np
import torch

from ..common.env_base import BaseEnv

# Denote angle indices to handle smooth transition
ANGLE_IDX = [1]

# X bounds
X_MIN = np.array([-5.0, -np.pi / 3, -1.0, -np.pi]).reshape(-1, 1)
X_MAX = np.array([5.0, np.pi / 3, 1.0, np.pi]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([0.0, 0, 0.0, 0])
XREF_INIT_MAX = np.array([0.0, 0, 0.0, 0])

# Initial perturbation to the reference state
XE_INIT_MIN = np.array([-1.0, -np.pi / 3, -0.5, -np.pi])
XE_INIT_MAX = np.array([1.0, np.pi / 3, 0.5, np.pi])

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

class SegwayEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform", reward_mode: str = "default", **kwargs):
        self.task = "segway"
        cfg = dict(ENV_CONFIG)
        cfg["sample_mode"] = sample_mode
        cfg["reward_mode"] = reward_mode
        super().__init__(cfg)

    def _f_logic(self, x, lib):
        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]
        f = self._zeros((n, self.num_dim_x), x, lib)
        f[:, 0] = v
        f[:, 1] = omega
        f[:, 2] = (
            lib.cos(theta) * (9.8 * lib.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (omega**2) * lib.sin(theta)
        ) / (lib.cos(theta) - 24.7)
        f[:, 3] = (
            -58.8 * v * lib.cos(theta)
            - 243.5 * v
            - lib.sin(theta) * (208.3 + (omega**2) * lib.cos(theta))
        ) / (lib.cos(theta) ** 2 - 24.7)
        return f

    def _B_logic(self, x, lib):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x, lib)
        B[:, 2, 0] = (-1.8 * lib.cos(theta) - 10.9) / (lib.cos(theta) - 24.7)
        B[:, 3, 0] = (9.3 * lib.cos(theta) + 38.6) / (lib.cos(theta) ** 2 - 24.7)
        return B

    def _B_null_logic(self, x, n, lib):
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x, lib)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = (9.3 * lib.cos(theta) + 38.6) / (lib.cos(theta) ** 2 - 24.7)
        Bbot[:, 3, 2] = -(-1.8 * lib.cos(theta) - 10.9) / (lib.cos(theta) - 24.7)
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

        xref_list, xref_wrapped_list, uref_list = [xref_0], [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(freqs, weights, _t, {"xref_0": xref_0})
            xref_t, xref_wrapped_t, term, trunc, _ = self.get_transition(xref_list[-1].copy(), uref_t)
            # Carry forward the wrapped+clipped state (same fix as BaseEnv.step's
            # self.x_t) — using raw xref_t here let non-angle/non-position dims
            # drift unbounded across iterations before reset()'s final np.clip.
            xref_wrapped_t = np.clip(xref_wrapped_t, self.X_MIN.flatten(), self.X_MAX.flatten())
            xref_list.append(xref_wrapped_t)
            xref_wrapped_list.append(xref_wrapped_t)
            uref_list.append(uref_t)
            if term or trunc:
                break
        return x_0, np.array(xref_wrapped_list), np.array(uref_list), i + 1

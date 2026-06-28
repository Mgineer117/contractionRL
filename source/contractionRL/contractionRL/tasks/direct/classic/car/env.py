"""Car (Dubins-like) tracking environment (ported from CAC-dev ``envs/xyD/car.py``).

State  x = [p_x, p_y, theta, v]
Control u = [omega, a]   (angular rate, linear accel) added to reference control.
Control-affine dynamics:
    f(x) = [v cos(theta), v sin(theta), 0, 0]
    B(x) = [[0,0],[0,0],[1,0],[0,1]]
"""

from __future__ import annotations

import numpy as np

from ..common.env_base import BaseEnv

ANGLE_IDX = [2]

v_l, v_h = 1.0, 2.0
X_MIN = np.array([-15.0, -15.0, -np.pi, v_l]).reshape(-1, 1)
X_MAX = np.array([15.0, 15.0, np.pi, v_h]).reshape(-1, 1)
XREF_INIT_MIN = np.array([-2.0, -2.0, -1.0, 1.5])
XREF_INIT_MAX = np.array([2.0, 2.0, 1.0, 1.5])
XE_INIT_MIN = np.full((4,), -1.0)
XE_INIT_MAX = np.full((4,), 1.0)
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)
UREF_MIN = np.array([-3.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([3.0, 3.0]).reshape(-1, 1)

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 4, "num_dim_control": 2, "pos_dimension": 2,
    "dt": 0.03, "time_bound": 9.0,
    "q": 1.0, "r": 0.0,
}


class CarEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform", reward_mode: str = "default", **kwargs):
        self.task = "car"
        cfg = dict(ENV_CONFIG)
        cfg["sample_mode"] = sample_mode
        cfg["reward_mode"] = reward_mode
        super().__init__(cfg)

    def _f_logic(self, x, lib):
        n = x.shape[0]
        f = lib.zeros((n, self.num_dim_x))
        f[:, 0] = x[:, 3] * lib.cos(x[:, 2])
        f[:, 1] = x[:, 3] * lib.sin(x[:, 2])
        return f

    def _B_logic(self, x, lib):
        n = x.shape[0]
        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))
        B[:, 2, 0] = 1
        B[:, 3, 1] = 1
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        uref = np.array([0.0, 0.0])
        for freq, weight in zip(freqs, weights):
            uref += np.array([weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi), 0])
        if add_noise:
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)
        return np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())

    def system_reset(self):
        xref_0, xe_0, x_0 = self.define_initial_state()
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (weights / np.sqrt((weights ** 2).sum(axis=0, keepdims=True))).tolist()

        xref_list, xref_wrapped_list, uref_list = [xref_0], [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(freqs, weights, _t, {"xref_0": xref_0})
            xref_t, xref_wrapped_t, term, trunc, _ = self.get_transition(xref_list[-1].copy(), uref_t)
            xref_list.append(xref_t)
            xref_wrapped_list.append(xref_wrapped_t)
            uref_list.append(uref_t)
            if term or trunc:
                break
        return x_0, np.array(xref_wrapped_list), np.array(uref_list), i + 1

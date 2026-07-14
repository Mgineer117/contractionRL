"""Quadrotor tracking environment (ported from CAC-dev ``envs/xyzD/quadrotor.py``).

State  x = [p_x, p_y, p_z, v_x, v_y, v_z, f, phi, theta, psi]
    (position, velocity, collective thrust, roll/pitch/yaw)
Control u = [f_rate, phi_rate, theta_rate, psi_rate]  (rates on thrust + attitude)
Control-affine dynamics:
    f(x) = [v_x, v_y, v_z,
            -f*sin(theta), f*cos(theta)*sin(phi), g - f*cos(theta)*cos(phi),
            0, 0, 0, 0]
    B(x) = rows 6..9 -> identity (u directly drives f/phi/theta/psi rates)

Unlike car/turtlebot/segway/cartpole (1-2 wrapping angles), this is a genuine
3-D attitude: phi (roll), theta (pitch), psi (yaw) all wrap at +-pi, hence
angle_idx = [7, 8, 9] — every network embeds all three as (cos, sin) pairs
(see agents/skrl/angle_utils.py); the env/dynamics/error stay in raw radians.
"""

from __future__ import annotations

import numpy as np

from ..common.env_base import BaseEnv

ANGLE_IDX = [7, 8, 9]

G = 9.81

_X10_LIM = np.pi / 3
_X9_LIM = np.pi / 3
_X8_LIM = np.pi / 3
_X7_LOW = 0.5 * G
_X7_HIGH = 2 * G
_X4_LIM = 1.5
_X5_LIM = 1.5
_X6_LIM = 1.5

X_MIN = np.array(
    [-30.0, -30.0, -30.0, -_X4_LIM, -_X5_LIM, -_X6_LIM, _X7_LOW, -_X8_LIM, -_X9_LIM, -_X10_LIM]
).reshape(-1, 1)
X_MAX = np.array(
    [30.0, 30.0, 30.0, _X4_LIM, _X5_LIM, _X6_LIM, _X7_HIGH, _X8_LIM, _X9_LIM, _X10_LIM]
).reshape(-1, 1)

XREF_INIT_MIN = np.array([-5.0, -5.0, -5.0, -1.0, -1.0, -1.0, G, 0.0, 0.0, 0.0])
XREF_INIT_MAX = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0, G, 0.0, 0.0, 0.0])

XE_INIT_MIN = -0.5 * np.ones(10)
XE_INIT_MAX = 0.5 * np.ones(10)

_lim = 1.0
XE_MIN = -_lim * np.ones(10).reshape(-1, 1)
XE_MAX = _lim * np.ones(10).reshape(-1, 1)

UREF_MIN = np.array([-1.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 10, "num_dim_control": 4, "pos_dimension": 3,
    "dt": 0.025, "time_bound": 10.0,
    "q": 1.0, "r": 0.0,
}


class QuadrotorEnv(BaseEnv):
    def __init__(
        self,
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "quadrotor"
        super().__init__(self._build_cfg(
            ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt,
        ))

    def _f_logic(self, x, lib):
        n = x.shape[0]
        force, theta_x, theta_y = x[:, 6], x[:, 7], x[:, 8]
        f = self._zeros((n, self.num_dim_x), x, lib)
        f[:, 0] = x[:, 3]
        f[:, 1] = x[:, 4]
        f[:, 2] = x[:, 5]
        f[:, 3] = -force * lib.sin(theta_y)
        f[:, 4] = force * lib.cos(theta_y) * lib.sin(theta_x)
        f[:, 5] = G - force * lib.cos(theta_y) * lib.cos(theta_x)
        return f

    def _B_logic(self, x, lib):
        n = x.shape[0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x, lib)
        B[:, 6, 0] = 1
        B[:, 7, 1] = 1
        B[:, 8, 2] = 1
        B[:, 9, 3] = 1
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        uref = np.array([0.0, 0.0, 0.0, 0.0])
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[2] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[3] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)
        return np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())

    def system_reset(self):
        xref_0, xe_0, x_0 = self.define_initial_state()
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (0.1 * weights / np.sqrt((weights ** 2).sum(axis=0, keepdims=True))).tolist()
        xref_arr, uref_arr, n = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, n

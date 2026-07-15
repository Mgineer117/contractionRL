"""Car (Dubins-like) tracking environment (ported to batched PyTorch).

State  x = [p_x, p_y, theta, v]
Control u = [omega, a]
Control-affine dynamics:
    f(x) = [v cos(theta), v sin(theta), 0, 0]
    B(x) = [[0,0],[0,0],[1,0],[0,1]]
"""

from __future__ import annotations

import math
import torch
import numpy as np

from ..common.env_base import BaseEnv

ANGLE_IDX = [2]

v_l, v_h = 1.0, 2.0
# Position bound must comfortably contain the reference rollout: xref's velocity is
# fixed at 1.5 (see XREF_INIT below — sample_reference_controls never drives the
# acceleration control) and heading only gets steered by omega, so over the full
# time_bound the reference can travel up to ~v*time_bound in a single direction if
# heading holds roughly constant. Without enough margin here, _rollout_reference's
# torch.clamp(next_x_wrapped, X_MIN, X_MAX) silently clips the reference position mid
# -trajectory, which corrupts tracking-error/reward at the boundary. 30.0 covers the
# worst case (1.5 * 15.0 = 22.5) plus the +-2.0 initial spread with margin to spare.
X_MIN = [-30.0, -30.0, -math.pi, v_l]
X_MAX = [30.0, 30.0, math.pi, v_h]
XREF_INIT_MIN = [-2.0, -2.0, -1.0, 1.5]
XREF_INIT_MAX = [2.0, 2.0, 1.0, 1.5]
XE_INIT_MIN = [-1.0, -1.0, -1.0, -1.0]
XE_INIT_MAX = [1.0, 1.0, 1.0, 1.0]
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]
UREF_MIN = [-3.0, -3.0]
UREF_MAX = [3.0, 3.0]

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 4, "num_dim_control": 2, "pos_dimension": 2,
    "dt": 0.03, "time_bound": 15.0,
    "q": 1.0, "r": 0.0,
}


class CarEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "car"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = x[:, 3] * torch.cos(x[:, 2])
        f[:, 1] = x[:, 3] * torch.sin(x[:, 2])
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = 1.0
        B[:, 3, 1] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            val = weight[:, 0] * math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 0] += val
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        weights = weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

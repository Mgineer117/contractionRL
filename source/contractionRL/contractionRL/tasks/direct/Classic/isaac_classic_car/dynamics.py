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

from contractionRL.tasks.direct.common.analytical_env import BaseEnv

STATE_NAMES = ("pos_x", "pos_y", "yaw", "vel")
# State names determine wrapping angles and position dimensions.

v_l, v_h = 1.0, 2.0
# Spread reference starts to cover the translation-invariant tracking task.
POS_SPREAD = 20.0
# Include the worst-case reference travel over the 15-second episode plus margin.
POS_BOUND = POS_SPREAD + 1.5 * 15.0 + 2.5
X_MIN = [-POS_BOUND, -POS_BOUND, -math.pi, v_l]
X_MAX = [POS_BOUND, POS_BOUND, math.pi, v_h]

# Use the state box for optional excursion termination.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Reference starts span the position box at a fixed nominal speed.
XREF_INIT_MIN = [-POS_SPREAD, -POS_SPREAD, -1.0, 1.5]
XREF_INIT_MAX = [POS_SPREAD, POS_SPREAD, 1.0, 1.5]
# Keep velocity perturbations inside [1, 2] around the 1.5 m/s reference.
XE_INIT_MIN = [-1.0, -1.0, -1.0, -0.5]
XE_INIT_MAX = [1.0, 1.0, 1.0, 0.5]
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]
# Reference controls are yaw rate (rad/s) and acceleration (m/s^2).
# BaseEnv doubles these bounds for applied controls.
UREF_MIN = [-1.5, -1.5]
UREF_MAX = [1.5, 1.5]

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 4, "num_dim_control": 2,
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
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
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

    def sample_reference_controls(self, freqs, weights, _t, infos):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            val = weight[:, 0] * torch.sin(torch.as_tensor(freq * _t / self.time_bound * 2 * math.pi, device=self.device))
            uref[:, 0] += val
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def set_held_out_mode(self, seed: int | None, n_trajectories: int = 64) -> None:
        """Fix a reproducible bank of ``n_trajectories`` reference-trajectory
        weight vectors (the mixing coefficients over the 10 fixed sinusoidal
        steering components — see ``system_reset``), drawn from a dedicated
        ``torch.Generator`` seeded independently of the run's own training
        seed/RNG stream. Bank parameters must be compared with training's
        parameters to establish disjointness; distinct seeds alone do not prove
        that a trajectory was unseen. Each new episode advances a persistent
        cursor through this bank, including sequential and partial resets.
        Rejected initial conditions retry the same bank row. Fixed-reference
        slots keep their assigned row until the mode is reconfigured.
        ``seed=None`` reverts to that historic
        i.i.d.-random behavior exactly (disables held-out mode).

        Only the steering weights are fixed — the initial state
        (``define_initial_state``: start position/heading/velocity, initial
        tracking error) still draws from the ordinary global RNG every reset,
        same as training. The "goal" being held out is the reference PATH
        shape, not the starting condition.
        """
        if isinstance(n_trajectories, bool) or not isinstance(n_trajectories, int) or n_trajectories < 1:
            raise ValueError("n_trajectories must be a positive integer")
        weights = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            weights = torch.randn(n_trajectories, 10, len(UREF_MIN), generator=gen)
            weights = weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        self._held_out_weights = None if weights is None else weights.to(self.device)
        self._held_out_cursor = 0
        self._held_out_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._held_out_reset_active = False
        self._ref_pool = None
        self._ref_fixed.zero_()
        if self.ref_groups:
            self._group_minted.zero_()

    def _safe_system_reset(self, env_ids):
        if getattr(self, "_held_out_weights", None) is None:
            return super()._safe_system_reset(env_ids)
        fresh = ~self._ref_fixed[env_ids] if self.fix_ref_trajectories else torch.ones_like(env_ids, dtype=torch.bool)
        new_ids = env_ids[fresh]
        previous_cursor = self._held_out_cursor
        previous_indices = self._held_out_indices[env_ids].clone()
        self._held_out_indices[new_ids] = (
            self._held_out_cursor + torch.arange(len(new_ids), device=self.device)
        ) % len(self._held_out_weights)
        self._held_out_cursor += len(new_ids)
        self._held_out_reset_active = True
        try:
            return super()._safe_system_reset(env_ids)
        except Exception:
            self._held_out_cursor = previous_cursor
            self._held_out_indices[env_ids] = previous_indices
            raise
        finally:
            self._held_out_reset_active = False

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        bank = getattr(self, "_held_out_weights", None)
        if bank is not None:
            if self._held_out_reset_active:
                idx = self._held_out_indices[env_ids]
            else:
                idx = (self._held_out_cursor + torch.arange(n, device=self.device)) % bank.shape[0]
                self._held_out_cursor += n
            weights = bank[idx]
        else:
            weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
            weights = weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

"""Duffing double-well oscillator -- polynomial, open-loop UNSTABLE in part of X.

    xdot_1 = x_2
    xdot_2 = -x_1 + x_1^3 / 3 - delta x_2 + u

G. Duffing, 1918; normalised double-well form as in Guckenheimer & Holmes,
"Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields",
Springer 1983, Sec. 2.2.

Complements ``toy.mg``, whose linearisation is Hurwitz everywhere. Here
A[1,0] = -1 + x_1^2 crosses zero at x_1 = +-1 and turns POSITIVE beyond it, so
the certified rate must hold across a genuine change in open-loop stability, not
merely a change in magnitude. Drift equilibria sit at x_1 = 0 and +-sqrt(3) and
the box spans all three. Stabilisable everywhere (rank 2 at every x), so the
unstable region costs feasibility nothing. B = [0;1] is constant, so all
state-dependence sits in A.
"""

from __future__ import annotations

import math

import torch

from ...classic.common.env_base import BaseEnv

STATE_NAMES = ("x1", "x2")

# Nonzero: at delta = 0 the drift is Hamiltonian and the reference oscillates
# forever, making "did the controller contract" unreadable against the plant.
DELTA = 0.3

# The drift's linear stiffness, the alpha in xdot_2 = -alpha x_1 + x_1^3/3 - ...
ALPHA = 1.0

# Reference-generator feedback gains, used ONLY to synthesize the default
# reference (see sample_reference_controls). They are not a controller the
# agent sees, and nothing certified depends on them.
#
# k_1 >= 1 is the load-bearing one and it is not a tuning preference. Under
# uref = -(alpha + k_1) x_1 - k_2 x_2 the reference obeys
#     xdot_2 = -(alpha + k_1) x_1 + x_1^3/3 - (delta + k_2) x_2,
# whose equilibria sit at x_1 = 0 and +-sqrt(3(alpha + k_1)). At k_1 = 0 that is
# +-2.449, INSIDE the +-2.5 box: the reference generator would have its own
# unstable equilibria in the region it is supposed to roam, and a reference that
# reaches one stops being driven by the excitation. k_1 = 1 puts them at +-3.0,
# outside the box, so the whole box is one basin.
REF_K1 = 1.0
# 0.0: the plant's own delta already damps, so the paper's law is the single
# term u = -(alpha + k_1) x_1 = -2 x_1, whose closed loop is -0.15 +- 1.726j.
REF_K2 = 0.0

# Spans both wells (x1 = +-sqrt(3) ~ +-1.732) and the saddle between them, so the
# sign change in A[1,0] = -1 + x1^2 at x1 = +-1 is interior.
X_MIN = [-2.5, -2.5]
X_MAX = [2.5, 2.5]

X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.5, -0.5]
XREF_INIT_MAX = [0.5, 0.5]

# The FAST band. Every (x1, 0) is an equilibrium here (uref = x1 - x1^3/3, well
# inside the box), so unlike mg the choice is not actuator-limited -- it is just
# where lam is largest on that curve: 1.878 at x1 = -1.76, against 1.570-1.620
# over the start band above. The whole field only spans 1.83x, so this is the
# honest size of duff's state-dependence, not a weak choice of band.
XREF_INIT_FAST_MIN = [-1.7625, 0.0]
XREF_INIT_FAST_MAX = [-1.7625, 0.0]

XE_INIT_MIN = [-0.3, -0.3]
XE_INIT_MAX = [0.3, 0.3]

XE_MIN = [-5.0, -5.0]
XE_MAX = [5.0, 5.0]

# The drift reaches |x1^3/3 - x1| = 1.09 at the well bottoms; +-4 leaves the
# feedback headroom above the trim that holds a reference there.
UREF_MIN = [-4.0]
UREF_MAX = [4.0]

ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xref_init_fast_min": XREF_INIT_FAST_MIN,
    "xref_init_fast_max": XREF_INIT_FAST_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 2,
    "num_dim_control": 1,
    # 100 steps, sized so exhaustive value iteration is tractable (see toy.mg).
    "dt": 0.05,
    "time_bound": 5.0,
    # The toy family trains and is evaluated on a FIXED benchmark: ONE reference
    # trajectory and num_envs distinct initial conditions, so num_envs tasks and
    # ref_groups = 1. Fixing the reference is what makes the optimal value a
    # function of x alone -- with xref frozen the augmented state (x, xref)
    # collapses to x, so V*(x) is a plain polynomial in two variables and the
    # optimal control law falls out of argmin_u [c(x,u) + gamma V(F(x,u))].
    # scripts/precompute_global.py solves each (reference, x_0) pair with the
    # sparse moment-SOS hierarchy; a trajectory optimum is a statement about ONE
    # start, so the task set has to be finite and frozen before it can be solved
    # at all. Declared here rather than passed at the call site so every
    # construction agrees, the bare gym.make() the evaluator does included.
    "fix_ref_trajectories": True,
    "ref_groups": 1,
    # 1.5 arrives closest to the band's own lam (1.868 of 1.878); the clamp never
    # bites at any gain here, so this is purely "which gain gets there".
    "migrate_gain": 1.5,
    "q": 1.0,
    "r": 0.0,
}


class DuffingEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "duff"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        f = self._zeros((x.shape[0], self.num_dim_x), x)
        f[:, 0] = x2
        f[:, 1] = -x1 + x1 ** 3 / 3.0 - DELTA * x2
        return f

    def _B_logic(self, x):
        B = self._zeros((x.shape[0], self.num_dim_x, self.num_dim_control), x)
        B[:, 1, 0] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        if self.reference_mode == "contractive":
            # Nothing to add: _migrate_uref returns a COMPLETE law for this mode
            # (a trim that holds the target plus a gain that steers to it), and a
            # second stabilising term on top would fight it. The two modes are
            # exclusive by design -- "sinusoidal" or "climb the rate field".
            return torch.zeros(weights.shape[0], self.num_dim_control,
                               device=self.device)
        uref = torch.zeros(weights.shape[0], self.num_dim_control, device=self.device)
        # A STABILISING controller, u = -(alpha + k_1) x_1 - k_2 x_2, evaluated at
        # the current reference state -- the reference is the trajectory this law
        # produces when driven by the excitation below.
        #
        # What this replaces cancelled the drift's position term outright
        # (uref = x_1 - x_1^3/3), which leaves xdot_2 = -delta x_2: no restoring
        # force at all, so x_1 integrates x_2 to wherever it happens to stop. The
        # reference did not oscillate into a well, it DRIFTED, and where it
        # ended up was an accident of the excitation rather than a property of a
        # controlled system. Feedback-linearising the plant also makes the
        # reference's own dynamics linear, which is precisely the state-dependence
        # the toy family exists to exhibit.
        xr = infos.get("xref_t")
        if xr is not None:
            uref[:, 0] = -(ALPHA + REF_K1) * xr[:, 0] - REF_K2 * xr[:, 1]
        for i, freq in enumerate(freqs):
            uref[:, 0] += weights[:, i, 0] * math.sin(freq * _t / self.time_bound * 2 * math.pi)
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, _xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 6))
        weights = torch.randn(len(env_ids), len(freqs), len(UREF_MIN), device=self.device)
        # 1.0, re-measured against the stabilising uref that replaced the old
        # feedback-linearising one, by the same criterion the previous 0.6 was
        # picked by: the largest amplitude that clamps NOTHING -- neither the
        # state box nor the uref box. That law pulls the reference back toward
        # the origin, so at 0.6 it barely explored (|xref| peaked at 0.38 of a
        # +-2.5 box). Measured over 4 seeds x 64 envs:
        #
        #     amp     max|x1|  max|x2|  uref saturated  state clamped  lam span
        #     0.6       0.38     0.64        0.0%          0.00%       1.563-1.603
        #     1.0       0.63     1.08        0.0%          0.00%       1.545-1.651
        #     1.5       1.00     1.56        0.0%          0.01%       1.509-1.753
        #     2.0       1.46     2.06        1.8%          0.47%       1.461-1.882
        #
        # 1.5 buys visibly more of the rate field for one clamped step in ~6400,
        # but rule.md Step 4 is not a budget: a clamped step is a step where the
        # stored (xref, uref) is not a trajectory of the plant.
        weights = 1.0 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

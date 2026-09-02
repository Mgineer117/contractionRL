"""Moore-Greitzer jet-engine surge -- polynomial, state-dependent rate.

    xdot_1 = -x_2 - 1.5 x_1^2 - 0.5 x_1^3
    xdot_2 =  3 x_1                 + u

F. K. Moore and E. M. Greitzer, J. Eng. Gas Turbines Power 108, 1986; the
two-state truncation used by E. M. Aylward, P. A. Parrilo and J.-J. E. Slotine,
Automatica 44(8), 2008.

The form usually quoted from that paper carries a ``- x_2`` in xdot_2. That term
is not the plant -- it is somebody's stabilising feedback, u = -x_2, already
substituted in, and the paper studies the resulting CLOSED LOOP. Keeping it here
would mean every algorithm in this repo is handed a plant that is already
stabilised and then measured on how much it improves a solved problem: the
drift's linearisation at the origin would be Hurwitz (-0.5 +- 1.658j) before any
controller acts. The open-loop plant is what is written above, and its
linearisation [[0,-1],[3,0]] has eigenvalues +-i sqrt(3) -- marginally stable, no
free contraction. The stabilising term now lives where it belongs: the default
reference generator applies it as the trim ``uref = -x_2``, so the reference
still traces the same well-behaved curve, but through the actuator.

``f`` is polynomial, so the contraction condition is certifiable as a sum of
squares over the WHOLE box (``contractionRL.solvers.sos_cm``) rather than at N
sampled states the way every classic env is. Only the (1,1) entry of A(x)
depends on x, and B = [0; 1] is constant, so all state-dependence sits in A.
"""

from __future__ import annotations

import math

import torch

from ...classic.common.env_base import BaseEnv

STATE_NAMES = ("x1", "x2")

# x2's half-width is load-bearing, not cosmetic. u enters only x2 (B = [0;1]), so
# x1 is steered purely through x2, and on the face x1 = -1.5 the drift is
#     f1 = -x2 - x1^2 (1.5 + 0.5 x1) = -x2 - 1.6875,
# which points OUT of the box for every x2 > -1.6875. At the original +-1.5 the
# whole face was one-way: no control kept the state inside, every episode there
# pinned to the wall, and the plant stopped being f + Bu. +-2.5 makes x2 <=
# -1.6875 reachable, so the face is two-way again.
# Enforced by tests/test_toy_envs.py::test_no_face_is_entirely_escaping.
X_MIN = [-1.5, -2.5]
X_MAX = [1.5, 2.5]

X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# The SLOW corner of the rate field (rule.md Step 3): every number this env
# reports should be the hard case, and reference_mode="contractive" only means
# something if the reference STARTS somewhere slow.
#
# Both bands live on the plant's REACHABLE EQUILIBRIA, which the open-loop drift
# defines differently from the closed-loop form this env used to carry: x1 = c is
# an equilibrium iff x2 = -1.5c^2 - 0.5c^3 (xdot_1 = 0) and uref = -3c
# (xdot_2 = 0), so |uref| <= 3 caps |c| at 1.0. Past that no reference can STAY
# there, only pass through. Measured along that curve under the shipped metric,
# lam runs [0.554 at c=-1, 1.165 at c=+1], a 2.10x spread.
#
# Neither band sits at an endpoint, because holding c = +-1 costs the ENTIRE
# +-3 actuator and leaves the migration nothing to steer with. c = -0.90
# (lam 0.571, uref +2.70) and c = +0.90 (lam 1.140, uref -2.70) keep 0.30 of
# authority in hand and still span 2.0x of the field.
#
# The width is nearly free here: ref_groups = 1, so ONE reference is drawn and
# the band's spread buys no task diversity -- the 64 tasks differ by x_0, not by
# xref_0. So it is kept narrow enough that every point in it is holdable.
XREF_INIT_MIN = [-1.00, -0.95]
XREF_INIT_MAX = [-0.80, -0.75]

# The FAST band: the highest-lam state this plant can HOLD with authority left
# over, per the equilibrium argument above. A point band, so _migrate_uref
# regulates BOTH dims (see "pinned") and routes through _lqr_migrate, which is
# what an underactuated plant needs.
XREF_INIT_FAST_MIN = [0.9000, -1.5795]
XREF_INIT_FAST_MAX = [0.9000, -1.5795]

XE_INIT_MIN = [-0.25, -0.25]
XE_INIT_MAX = [0.25, 0.25]

XE_MIN = [-3.0, -5.0]
XE_MAX = [3.0, 5.0]

UREF_MIN = [-3.0]
UREF_MAX = [3.0]

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
    # 100 steps. Short enough that exhaustive value iteration over the state box
    # is tractable, which is why the toy family exists.
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
    # 10.0, not the 1.0 default: the LQR gain is scaled by this, and below ~5 the
    # reference approaches the fast band slowly enough that 2-36% of the pool's
    # reference steps are rewritten by the X-box clamp. At 10 the clamp never
    # bites and lam still arrives at 1.744 of the band's 1.752.
    "migrate_gain": 10.0,
    "q": 1.0,
    "r": 0.0,
}


class MooreGreitzerEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "mg"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        f = self._zeros((x.shape[0], self.num_dim_x), x)
        f[:, 0] = -x2 - 1.5 * x1 ** 2 - 0.5 * x1 ** 3
        f[:, 1] = 3.0 * x1
        return f

    def _B_logic(self, x):
        B = self._zeros((x.shape[0], self.num_dim_x, self.num_dim_control), x)
        B[:, 1, 0] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        if self.reference_mode == "contractive":
            # No trim and no sinusoid here: _migrate_uref already returns a
            # COMPLETE law for this mode -- _lqr_migrate's u0 holds the target
            # (it solves B u0 = -f there, which is the -x_2 trim's job) plus a
            # full-state gain that steers to it. Adding the trim on top would
            # double-count it. The migration also needs most of the actuator:
            # holding the fast band costs uref = -2.70 of +-3, so a sinusoid on
            # top is clamped away and the reference wanders instead of arriving
            # (measured on the old plant: lam ran 1.67 -> 0.75, the wrong
            # direction). The two modes are exclusive by design -- "sinusoidal"
            # or "climb the rate field".
            return torch.zeros(weights.shape[0], self.num_dim_control,
                               device=self.device)
        # THE trim, not a nicety: the open-loop drift's linearisation
        # [[0,-1],[3,0]] has eigenvalues +-i sqrt(3), so with uref = 0 the
        # reference orbits (and the cubic pushes it out of the box) instead of
        # settling. u = -x_2 is the classical stabilising feedback for this
        # plant -- the very term the textbook form hides inside xdot_2 -- so
        # applying it here reproduces the Hurwitz reference the old, already
        # stabilised drift produced, while leaving the PLANT open-loop.
        # Evaluated at the CURRENT reference state (infos["xref_t"]), which is
        # what makes it a trajectory of the plant rather than a constant.
        xref_t = infos["xref_t"]
        uref = -xref_t[:, 1:2].clone()
        for i, freq in enumerate(freqs):
            uref[:, 0] += weights[:, i, 0] * math.sin(freq * _t / self.time_bound * 2 * math.pi)
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, _xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 6))
        weights = torch.randn(len(env_ids), len(freqs), len(UREF_MIN), device=self.device)
        weights = 0.6 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length

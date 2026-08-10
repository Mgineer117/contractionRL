"""Car with a WEAK-AUTHORITY velocity box — the class-III half of the car pair.

Identical plant to ``classic/car``: same ``f``, same ``B``, same reward, same
reference generator. The ONLY difference is the box.

Why this exists. ``docs/dynamics_taxonomy.md`` §2.2 shows the car's Hautus margin
at ``s = 0`` is ``sigma = min(1, v)``, so the velocity box alone decides the
contraction class, with every line of the dynamics held fixed:

    classic-car-v0        v in [1.0, 2.0]   sigma == 1        -> CLASS II
    classic-car_weak-v0   v in [0.2, 2.0]   sigma == v < 1    -> CLASS III

Measured at lbd=0.3, N=200 (``scripts/feasibility_certificate.py --verify``):

    env         class  hautus     nu      rho spread  lam_C spread  ||K||max
    car          II    1.000e+00    9.536     1.0000       1.0000      2.862
    car_weak     III   2.049e-01  147.5      15.4690       7.4668      8.775

That pair is the one within-plant control available for "does the contraction
class change how RL behaves": any comparison across two DIFFERENT plants
confounds the class with dimension, stiffness, actuation and reward scale.

Both ends of the box move, and both are load-bearing:

* ``x_min[3]``: 1.0 -> 0.2 lets the STATE reach the weak-authority region at all.
* ``xref_init`` v: pinned 1.5 -> [0.3, 1.5] makes the REFERENCE visit it.
  ``sample_reference_controls`` never drives the acceleration channel, so xref
  holds its INITIAL v for the whole episode; without widening the initial draw,
  low-v states would only ever be transient tracking error and the plant would
  spend no meaningful time where sigma < 1.
"""

from __future__ import annotations

from ..car.env import ENV_CONFIG as CAR_ENV_CONFIG
from ..car.env import STATE_NAMES, CarEnv  # noqa: F401  re-exported for parity

V_WEAK_LO = 0.2
VREF_WEAK_LO = 0.3

# Only the two lower bounds move; every other key is the car's. Passed as
# constructor kwargs, which BaseEnv._build_cfg merges over ENV_CONFIG in place.
BOX_OVERRIDES = {
    "x_min": [*CAR_ENV_CONFIG["x_min"][:3], V_WEAK_LO],
    "xref_init_min": [*CAR_ENV_CONFIG["xref_init_min"][:3], VREF_WEAK_LO],
}

ENV_CONFIG = {**CAR_ENV_CONFIG, **BOX_OVERRIDES}


class CarWeakEnv(CarEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**BOX_OVERRIDES, **kwargs})
        # AFTER super().__init__, which sets self.task = "car". Anything keyed on
        # the task name -- notably the offline CM dataset directory -- must not
        # collide with the class-II variant's.
        self.task = "car_weak"

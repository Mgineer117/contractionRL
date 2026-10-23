"""Low-speed car variant with the same plant and reference generator as CarEnv.

The state and initial-reference velocity boxes include the weak-control region.
Reference acceleration stays zero, so nominal speed is set by its initial draw.
"""

from __future__ import annotations

from contractionRL.tasks.direct.Classic.isaac_classic_car.dynamics import ENV_CONFIG as CAR_ENV_CONFIG
from contractionRL.tasks.direct.Classic.isaac_classic_car.dynamics import STATE_NAMES, CarEnv  # noqa: F401  re-exported for parity

V_WEAK_LO = 0.2
# Start references in the low-speed, weak-control region.
VREF_WEAK_LO = 0.3
# With velocity error +/-0.1, initial speeds stay in [0.2, 0.7] without clipping.
VREF_WEAK_HI = 0.6
XE_INIT_V = 0.1

# Override only the velocity bounds; inherit the remaining car task settings.
BOX_OVERRIDES = {
    "x_min": [*CAR_ENV_CONFIG["x_min"][:3], V_WEAK_LO],
    "xref_init_min": [*CAR_ENV_CONFIG["xref_init_min"][:3], VREF_WEAK_LO],
    "xref_init_max": [*CAR_ENV_CONFIG["xref_init_max"][:3], VREF_WEAK_HI],
    # Contractive references migrate toward the higher-speed band.
    "xref_init_fast_min": [*CAR_ENV_CONFIG["xref_init_min"][:3], 1.0],
    "xref_init_fast_max": [*CAR_ENV_CONFIG["xref_init_max"][:3], 1.4],
    # Velocity slice only: the position/yaw perturbations stay the car's, since
    # lambda* is flat in those (measured ptp 0.0000 across yaw).
    "xe_init_min": [*CAR_ENV_CONFIG["xe_init_min"][:3], -XE_INIT_V],
    "xe_init_max": [*CAR_ENV_CONFIG["xe_init_max"][:3], XE_INIT_V],
    # Termination must follow this task's velocity box, not the parent car's.
    "x_termination_min": [*CAR_ENV_CONFIG["x_min"][:3], V_WEAK_LO],
    "x_termination_max": list(CAR_ENV_CONFIG["x_max"]),
}

ENV_CONFIG = {**CAR_ENV_CONFIG, **BOX_OVERRIDES}


class CarV1Env(CarEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**BOX_OVERRIDES, **kwargs})
        # Use a distinct task name for metric caches and configuration lookup.
        self.task = "car_v1"

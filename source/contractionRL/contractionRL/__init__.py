# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tracking environments and contraction-based reinforcement learning.
"""

# Register Gym environments (Isaac Sim only; skipped when Isaac is absent).
try:
    from .tasks import *
except (ImportError, ModuleNotFoundError):
    pass

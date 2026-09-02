"""Solvers: convex programs that answer questions ABOUT an env, not agents.

- ``sos_cm``      exact contraction metric for a 2-state POLYNOMIAL plant, as an
  SOS identity over the whole box. Toy family only; raises on anything else.
- ``moment_sos``  the toy family's OPTIMAL CONTROLS, by the sparse moment-SOS
  (Lasserre) hierarchy: the hierarchy itself, plus the toy tracking task written
  as the polynomial program it relaxes.
- ``vi``          V* by exhaustive value iteration over a discretised state and
  control space, plus a Richardson error estimate. Bounded by MEMORY rather than
  dimension (``check_budget``), so it applies to any env small enough to grid.

The toy optima come from ``moment_sos``, not from VI: VI is exact only for the
DISCRETISED MDP, and measured against the certified optima its grid error ran
0.1-2.5% in both directions -- on one task reporting a value BELOW the true
global optimum, which is impossible and larger than the effects being studied.
"""

from .moment_sos import SparseMomentProgram, build, check_dynamics, metric_value, read_point
from .sos_cm import SOSCfg, certify_ccm
from .vi import (
    GridVI,
    TrackingVI,
    VICfg,
    check_budget,
    grid_value_at,
    richardson,
    solve_tracking,
    solve_vi,
)

__all__ = ["SOSCfg", "certify_ccm",
           "SparseMomentProgram", "build", "check_dynamics", "metric_value", "read_point",
           "VICfg", "GridVI", "TrackingVI", "solve_vi", "solve_tracking",
           "richardson", "check_budget", "grid_value_at"]

"""Toy (non-Isaac) POLYNOMIAL tracking environments.

Same interface as ``tasks/direct/classic`` -- same ``BaseEnv``, same
``obs = {x, xrefs, urefs}``, same reward -- so every algorithm that runs on a
classic env runs here unchanged. What differs is that ``f`` and ``B`` are
POLYNOMIAL in ``x`` and the state is 2-dimensional, which buys two things the
classic family cannot have:

1. **An exact metric.** ``solvers.sos_cm`` certifies the contraction condition
   as a sum of squares over the whole box -- an algebraic identity, not a claim
   interpolated between sampled states.
2. **A global optimum.** The sparse moment-SOS hierarchy
   (``scripts/precompute_global.py``) returns a certified LOWER bound on J*
   together with a feasible UPPER bound, so ``J^pi - J*`` is measurable and the
   bracket says how tight it is. Every trajectory-based alternative (multi-start
   NLP, MPC lookahead) only upper-bounds J*, which can prove a policy bad but
   never near-optimal.

Both plants have a constant ``B``, so all state-dependence lives in ``A(x)``.
"""

from . import duff  # noqa: F401  registers toy-duff-v0
from . import mg  # noqa: F401  registers toy-mg-v0

__all__ = ["mg", "duff"]

TOY_ENVS = ("mg", "duff")

# Long names, for tables and papers. The short key is what the code uses.
TOY_LONG_NAMES = {"mg": "Moore-Greitzer", "duff": "Duffing"}


def env_id(name: str) -> str:
    """``"duff"`` -> ``toy-duff-v0``."""
    return f"toy-{name}-v0"

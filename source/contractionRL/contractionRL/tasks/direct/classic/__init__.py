"""Classic (non-Isaac) analytical tracking environments, trained via mjrl.

Each subpackage registers a gym environment so it is discoverable by
``scripts/list_envs.py`` and runnable with ``scripts/mjrl/train.py``.

Importing this package registers every classic env. This works both as
``contractionRL.tasks.direct.classic`` (under Isaac) and as a standalone
``classic`` package on sys.path (mjrl path, no Isaac) — the registration
f-strings adapt to whichever package name they are imported under.
"""

from . import car  # noqa: F401  registers Car-Direct-v0

__all__ = ["car"]


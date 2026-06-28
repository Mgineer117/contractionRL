"""mjrl — contraction-control algorithm library for tracking environments.

A skrl-style library providing contraction-based controllers/synthesisers
(LQR, SD-LQR, C3M, TEMP) ported from CAC-dev. Mirrors skrl's high-level API:

    from mjrl.utils.runner.torch import Runner
    runner = Runner(env, cfg)
    runner.run()

Unlike skrl (which targets Isaac Sim vectorised envs), mjrl targets the classic
analytical tracking environments registered under
``tasks/direct/classic`` that expose ``get_f_and_B(x)`` and a
``[x, xref, uref]`` observation layout.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

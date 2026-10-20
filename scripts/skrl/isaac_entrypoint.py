"""Preserve Python failures through Isaac Kit's fast shutdown."""

import os
import signal
import sys
import traceback


def _exception_status(error):
    if isinstance(error, SystemExit):
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(error.code, file=sys.stderr, flush=True)
        return 1
    traceback.print_exception(error)
    return 130 if isinstance(error, KeyboardInterrupt) else 1


def _flush_streams():
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is not None:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass  # Kit may have already closed its intercepted streams.


def run_isaac(main, simulation_app, kit_app):
    """Run training, close the simulator, and exit with the captured status."""
    status = 0
    try:
        # SimulationApp's SIGINT handler exits successfully even when interrupted.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        main()
    except BaseException as error:
        status = _exception_status(error)
    finally:
        wandb = sys.modules.get('wandb')
        run = getattr(wandb, 'run', None)
        if run is not None:
            try:
                run.finish(exit_code=status)
            except BaseException as error:
                cleanup_status = _exception_status(error)
                if status == 0:
                    status = cleanup_status
        _flush_streams()
        try:
            try:
                # Direct ``raise SystemExit`` bypasses Kit's patched sys.exit.
                kit_app.post_quit(status)
            finally:
                simulation_app.close()
        except BaseException as error:
            cleanup_status = _exception_status(error)
            if status == 0:
                status = cleanup_status
        finally:
            _flush_streams()
            os._exit(status)

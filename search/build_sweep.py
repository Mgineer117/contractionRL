#!/usr/bin/env python3
"""Emit a W&B sweep yaml from ``search/configs/<algorithm>.yaml`` + an env.

The searched space lives in ``search/configs/`` (one file per algorithm,
applying to every env). This script is the only thing that knows how to turn
one of those into what ``wandb sweep`` actually wants: it copies the config's
``metric:``/``parameters:`` blocks through verbatim and synthesizes the
``program:``/``project:``/``name:``/``command:`` lines around them.

Every sweep goes into the SAME W&B project (``contractionRL-Search``) and is
identified by its ``name:``, ``<env>-<algorithm>`` — so all the runs for a given
env+algorithm accumulate in one place across relaunches, rather than being split
across a project per launch.

Trials normally invoke ``scripts/skrl/train.py`` directly. When the config sets
``runner.wrapper``, they go through ``search/sweep_runner.py`` instead, which
records a poison metric value on an infeasible SDP rather than leaving a
metric-less run the sweep controller would ignore (see that file's docstring).

Usage:
    python search/build_sweep.py --algorithm cvstem-lqr --env classic-car-v0 \
        [--method bayes] [--project NAME] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = Path(__file__).resolve().parent / "configs"

# The single W&B project every sweep lands in. Set explicitly (rather than left
# to wandb) because an unset project does NOT fall back to a shared default —
# wandb auto-derives one from the local path (observed:
# "contractionRL-scripts_skrl"), which silently splits runs by where they were
# launched from. Sweeps are told apart by their `name:` (env-algorithm) instead.
DEFAULT_PROJECT = "contractionRL-Search"

# Classic envs run through train.py's --classic route; everything else is an
# Isaac Lab task and must NOT get that flag.
CLASSIC_ENVS = (
    "classic-car-v0",
    "classic-cartpole-v0",
    "classic-segway-v0",
    "classic-turtlebot-v0",
    "classic-quadrotor-v0",
)
ISAACLAB_ENVS = (
    "Humanoid-PathTracking-v0",
    "Humanoid-VelTracking-v0",
    "Manipulator-PathTracking-v0",
    "Manipulator-VelTracking-v0",
    "Quadruped-PathTracking-v0",
    "Quadruped-VelTracking-v0",
)


def available_algorithms() -> list[str]:
    """Algorithm names discoverable in configs/ — the file stem IS the name."""
    return sorted(p.stem for p in _CONFIGS.glob("*.yaml"))


def load_config(algorithm: str) -> dict:
    path = _CONFIGS / f"{algorithm}.yaml"
    if not path.exists():
        raise SystemExit(
            f"No search config for '{algorithm}'. Available: {', '.join(available_algorithms())}"
        )
    cfg = yaml.safe_load(path.read_text())
    for required in ("algorithm", "num_envs", "metric", "parameters"):
        if required not in cfg:
            raise SystemExit(f"{path} is missing required key '{required}'.")
    for required in ("name", "goal"):
        if required not in (cfg.get("metric") or {}):
            raise SystemExit(f"{path}: metric is missing required key '{required}'.")

    # Validate the runner block HERE rather than letting a missing key surface as
    # a KeyError traceback deep in build(). A sweep is launched detached, so a
    # config mistake that only trips at build time would otherwise show up as a
    # traceback in a nohup log nobody is watching.
    runner = cfg.get("runner") or {}
    if runner.get("wrapper") and "bad_value" not in runner:
        raise SystemExit(
            f"{path}: runner.wrapper is set but runner.bad_value is missing — the wrapper "
            "needs a value to record on infeasibility, and its sign must match "
            f"metric.goal ({cfg['metric']['goal']})."
        )
    if runner.get("bad_value") is not None and not runner.get("wrapper"):
        raise SystemExit(
            f"{path}: runner.bad_value is set but runner.wrapper is not — nothing would "
            "ever write that value, so the sweep would silently ignore infeasible trials."
        )
    if runner.get("wrapper"):
        # A poison value pointing the wrong way is the worst kind of silent
        # failure: bayes would treat infeasible trials as the BEST observations
        # and drive the whole search into the infeasible region.
        bad = float(runner["bad_value"])
        goal = str(cfg["metric"]["goal"]).lower()
        if goal == "minimize" and bad <= 0:
            raise SystemExit(
                f"{path}: metric.goal is minimize, so runner.bad_value must be a LARGE "
                f"positive number (got {bad}) — otherwise infeasible trials rank as the best."
            )
        if goal == "maximize" and bad >= 0:
            raise SystemExit(
                f"{path}: metric.goal is maximize, so runner.bad_value must be a LARGE "
                f"NEGATIVE number (got {bad}) — otherwise infeasible trials rank as the best."
            )
    return cfg


def episode_length(env: str) -> int:
    """Steps in ONE episode of ``env``, read from the env's own definition.

    Classic envs expose ENV_CONFIG with dt/time_bound, so this is derived rather
    than tabulated — a hardcoded table would silently go stale the moment an
    env's dt or time_bound changed, and the sweep would quietly evaluate a
    fraction of an episode (or several).
    """
    if env not in CLASSIC_ENVS:
        raise SystemExit(
            f"one_episode is only supported for classic envs (got '{env}') — Isaac Lab "
            "envs take their episode length from the sim cfg, not ENV_CONFIG."
        )
    sys.path.insert(0, str(_ROOT / "source" / "contractionRL"))
    import contextlib
    import importlib

    name = env.replace("classic-", "").replace("-v0", "")
    # Importing the env pulls in pygame, which greets on STDOUT — and stdout is
    # where the generated yaml goes under `--out -`. Redirect the import's
    # output to stderr so the yaml on stdout is always parseable.
    with contextlib.redirect_stdout(sys.stderr):
        mod = importlib.import_module(f"contractionRL.tasks.direct.classic.{name}.env")
    cfg = mod.ENV_CONFIG
    return round(cfg["time_bound"] / cfg["dt"])


def build(algorithm: str, env: str, *, method: str, project: str) -> dict:
    cfg = load_config(algorithm)
    runner = cfg.get("runner") or {}
    is_classic = env in CLASSIC_ENVS

    if runner.get("wrapper"):
        program = "search/sweep_runner.py"
        command = ["${env}", "python", "${program}"]
    else:
        program = "scripts/skrl/train.py"
        command = ["${env}", "python", "${program}"]

    if is_classic:
        command.append("--classic")
    command += [
        "--task", env,
        "--algorithm", str(cfg["algorithm"]),
        "--num_envs", str(cfg["num_envs"]),
    ]

    if runner.get("wrapper"):
        # The wrapper needs to know WHICH key to poison; reading it off the
        # metric block rather than repeating it means the two can never drift
        # into a sweep that optimizes a metric no bad trial ever writes.
        command += ["--metric-name", cfg["metric"]["name"]]
        # One token, not two: argparse's negative-number matcher does not cover
        # scientific notation, so a bare negative bad-value in the next slot
        # would be read as an option name. Kept in this form regardless so
        # flipping metric.goal (and the sign) never reintroduces that.
        command.append(f"--bad-value={cfg['runner']['bad_value']}")

    if runner.get("one_episode"):
        # Cap the trial at exactly one episode. train.py applies
        # --num_timesteps to trainer.timesteps for both routes.
        #
        # StatManagerEnvWrapper assigns every env a buffer slot on its reset at
        # step 0 and reduces the whole buffer once every slot has run a full
        # episode — so exactly one episode is also exactly what it takes for
        # Stability/auc_mean to be computed once, from all num_envs envs.
        command += ["--num_timesteps", str(episode_length(env))]
        # And drop the post-training rollout: it is sequential over ONE env and
        # feeds eval.json rather than the swept Stability/* metric, so in a
        # one-episode trial it would dominate the cost while contributing
        # nothing the sweep reads.
        command.append("--skip_final_eval")

    command.append("${args}")

    return {
        "program": program,
        # Every sweep lands in ONE project. What distinguishes them is the sweep
        # NAME below, so all runs for a given env+algorithm stay comparable in a
        # single place instead of being scattered across per-launch projects.
        "project": project,
        "name": f"{env}-{algorithm}",
        "method": method,
        "metric": cfg["metric"],
        "parameters": cfg["parameters"],
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", required=True, choices=available_algorithms())
    parser.add_argument("--env", required=True, choices=[*CLASSIC_ENVS, *ISAACLAB_ENVS])
    parser.add_argument("--method", default="bayes", choices=["bayes", "grid", "random"])
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help=f"W&B project for this sweep (default {DEFAULT_PROJECT}). Sweeps are "
                             "distinguished by their NAME (env-algorithm), not by project.")
    parser.add_argument("--out", default="-", help="Output path, or '-' for stdout.")
    args = parser.parse_args()

    sweep = build(args.algorithm, args.env, method=args.method, project=args.project)
    text = yaml.safe_dump(sweep, sort_keys=False, default_flow_style=False, width=100)

    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

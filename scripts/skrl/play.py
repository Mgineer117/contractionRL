# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate trained checkpoints and report their Stability/* metrics.

By default this script is a multi-model stability evaluator: given one
``--task``, it discovers every trained model for that env
(one per algorithm) and rolls each out for a single episode across
``num_envs_for_eval`` parallel envs, reading the batched contraction metrics
(auc / contraction_rate / overshoot / contraction_score) from
``StatManagerEnvWrapper.stability_summary()`` — the same wrapper the training
loop logs from. Results are printed as a per-algorithm table and saved to
``logs/play/<env>_stability_<ts>.json``.

``--viewer`` instead continuously replays the selected Isaac policy at real time.

Model discovery (per algorithm), in order:
  1. ``<models_dir>/<env>/<alg>.pt``          (default ``models_dir="models"``)
  2. newest ``best_agent.pt`` under that algorithm's log tree
     (classic: ``logs/classic/<alg>/*/checkpoints/``;
      Isaac:   ``logs/skrl/*<alg>*/checkpoints/``)

Examples::

    # Evaluate every algorithm trained on the given task
    python scripts/skrl/play.py --task Isaac-Transfer-path-Quadruped-v0 --num_envs 64
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Prefer this checkout over an editable installation from another worktree.
_checkout = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_checkout, "source", "contractionRL"))

# Argument parsing
# Parsed before any Isaac import so the classic route never touches Isaac Sim.


def _checkpoint_agent_cfg(checkpoint, fallback):
    """Restore the recorded architecture before constructing checkpoint models."""
    import yaml
    path = Path(checkpoint).parent.parent / 'params' / 'agent.yaml'
    if path.is_file():
        fallback = yaml.safe_load(path.read_text())
    from train_utils import restore_checkpoint_model_config
    restore_checkpoint_model_config(fallback, checkpoint)
    return fallback

def _reject_removed_flags():
    removed = {'mode', 'encoder_stride', 'reference_preview_steps', 'critic_encoder_stride', 'critic_embed_dim', 'critic_combine'}
    for arg in sys.argv[1:]:
        if arg.startswith('--') and arg.split('=', 1)[0][2:].replace('-', '_') in removed:
            parser.error(f"{arg.split('=', 1)[0]} was removed; use the recorded encoder configuration")


parser = argparse.ArgumentParser(description="Multi-model Stability evaluation of skrl checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Name of the task/env.")
parser.add_argument(
    "--algorithm", type=str, default=None,
    help="Restrict to a single algorithm (e.g. c3m, ppo, c2rl-ppo). Default: every discovered model.",
)
parser.add_argument(
    "--models_dir", type=str, default="models",
    help="Preferred model root: <models_dir>/<env>/<alg>.pt. Falls back to logs/ if absent.",
)
parser.add_argument(
    "--num_envs", type=int, default=None,
    help="Parallel envs for the rollout (caps num_envs_for_eval). Default: algorithm-specific.",
)
parser.add_argument(
    "--num_envs_for_eval", type=int, default=64,
    help="Stability buffer size (envs whose trajectory is scored). Default 64.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment/agent seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (single algorithm).")
parser.add_argument("--viewer", action="store_true", help="Isaac: continuously replay the policy at real time until the viewer closes.")
viewer_offsets = parser.add_mutually_exclusive_group()
viewer_offsets.add_argument("--viewer_initial_offset", type=float, nargs=2, metavar=("DX", "DY"),
                            help="Analytical car viewer: fixed initial position offset from the reference, in meters.")
viewer_offsets.add_argument("--viewer_initial_offset_box", type=float, nargs=4,
                            metavar=("DX_MIN", "DX_MAX", "DY_MIN", "DY_MAX"),
                            help="Analytical car viewer: independently uniform initial position deviations, in meters.")
viewer_offsets.add_argument("--viewer_initial_offset_signed_range", type=float, nargs=2,
                            metavar=("MIN_MAG", "MAX_MAG"),
                            help="Analytical car viewer: uniform magnitude and independent random sign for each position deviation.")
parser.add_argument("--viewer_disturbance_bounds", type=float, nargs=2, metavar=("D_THETA", "D_V"),
                    help="Analytical car viewer: heading-rate (rad/s) and acceleration (m/s^2) gust bounds.")
parser.add_argument(
    "--ref_length", "--ref-length", dest="ref_length", type=int, default=None,
    help="Reference-window length. Default AUTO from the cfg's discount_factor, matching "
         "train.py — override only if the run was trained with an explicit --ref_length.",
)
# Isaac-only extras (ignored on the classic route).
parser.add_argument("--policy_control_mode", choices=["structured", "full", "residual"], default=None)
parser.add_argument("--disable_disturbance", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Isaac: disable fabric.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])

_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"})


def _env_name(task: str) -> str:
    """``Car-v0`` / ``Car-Play-v0`` -> ``Car`` (the <env> segment of models/<env>/)."""
    name = task.split(":")[-1]
    name = name.replace("-Play", "")
    # strip a trailing gymnasium version suffix (-v0, -v12, …)
    import re
    return re.sub(r"-v\d+$", "", name)


# Model discovery

def _candidate_algorithms(classic: bool, algorithm_filter: str | None) -> list[str]:
    if algorithm_filter:
        return [algorithm_filter.lower()]
    if classic:
        root = os.path.join("logs", "classic")
        if os.path.isdir(root):
            return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        return []
    # Isaac: enumerate algorithms that ship a config entry point for this task
    # (best effort — the log-dir names aren't algorithm-clean like classic's).
    return ["ppo", "contraction_ppo", "sac", "mbpo", "c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"]


def _newest_best_ckpt(patterns: list[str], algorithm: str | None = None, task: str | None = None) -> str | None:
    hits: list[str] = []
    for p in patterns:
        hits.extend(glob.glob(p, recursive=True))
    if algorithm is not None:
        import yaml
        from pathlib import Path
        expected = algorithm.lower().replace("-", "").replace("_", "")
        matched = []
        for path in hits:
            config = Path(path).parent.parent / "params" / "agent.yaml"
            if config.exists():
                recorded = yaml.safe_load(config.read_text())
                saved = recorded["agent"]["class"]
                if saved.lower().replace("-", "").replace("_", "") != expected:
                    continue
                saved_task = recorded.get("environment", {}).get("task") or recorded["agent"].get("experiment", {}).get("wandb_kwargs", {}).get("config", {}).get("task")
                if task is not None and saved_task != task:
                    continue
            elif task is not None:
                continue  # An unidentified legacy checkpoint needs --checkpoint.
            matched.append(path)
        hits = matched
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _discover_models(task: str, classic: bool, models_dir: str,
                     algorithm_filter: str | None, checkpoint: str | None) -> list[tuple[str, str]]:
    """Return ``[(algorithm, checkpoint_path), …]`` for every model found for ``task``.

    Preference: ``<models_dir>/<env>/<alg>.pt``, else the newest ``best_agent.pt``
    under the algorithm's log tree. ``--checkpoint`` (with ``--algorithm``) forces
    a single explicit path.
    """
    env = _env_name(task)
    if algorithm_filter is not None: algorithm_filter = algorithm_filter.lower()
    if algorithm_filter in ("contractionppo", "contraction-ppo"):
        algorithm_filter = "contraction_ppo"
    if checkpoint:
        alg = (algorithm_filter or "model").lower()
        if alg in ("contractionppo", "contraction-ppo"): alg = "contraction_ppo"
        return [(alg, os.path.abspath(checkpoint))]

    found: list[tuple[str, str]] = []
    for alg in _candidate_algorithms(classic, algorithm_filter):
        preferred = os.path.join(models_dir, env, f"{alg}.pt")
        if os.path.isfile(preferred):
            found.append((alg, os.path.abspath(preferred)))
            continue
        # skrl nests the checkpoints under an auto-named experiment subdir
        # (logs/classic/<alg>/<ts>/<..._AgentClass>/checkpoints/), so glob
        # recursively rather than assuming a fixed depth.
        if classic:
            patterns = [os.path.join("logs", "classic", alg, "**", "best_agent.pt")]
        else:
            patterns = [
                os.path.join("logs", "skrl", f"*{alg}*", "**", "best_agent.pt"),
                os.path.join("logs", "skrl", f"*{alg.replace('-', '_')}*", "**", "best_agent.pt"),
            ]
        ckpt = _newest_best_ckpt(patterns, alg, task)
        if ckpt:
            found.append((alg, os.path.abspath(ckpt)))
    return found


# Rollout

def _rollout_stability(env, agent, *, max_steps: int) -> dict:
    from contractionRL.agents.utils.RL.evaluation import rollout_stability
    return rollout_stability(env, agent, max_steps=max_steps)


def _set_viewer_initial_offset(raw, offset, upper=None, *, independent_signs=False):
    """Change the car's reset distribution before observations and metrics reset."""
    import torch

    plant = getattr(raw, "plant", None)
    if getattr(plant, "task", None) != "car" or plant.X_INIT_MIN is not None or plant.fix_ref_trajectories:
        raise ValueError("--viewer_initial_offset requires an analytical car with unfixed reference episodes")
    offset = torch.as_tensor(offset, device=plant.device, dtype=plant.XE_INIT_MIN.dtype)
    upper = offset if upper is None else torch.as_tensor(upper, device=plant.device, dtype=offset.dtype)
    if offset.shape != (2,) or upper.shape != (2,) or not torch.isfinite(torch.cat((offset, upper))).all():
        raise ValueError("Initial position offset must contain two finite values")
    if (offset > upper).any():
        raise ValueError("Initial position offset lower bounds must not exceed upper bounds")
    if independent_signs and (offset < 0).any():
        raise ValueError("Signed initial position offset magnitudes must be nonnegative")
    minimum = -upper if independent_signs else offset
    if ((plant.XREF_INIT_MIN[:2] + minimum < plant.X_MIN[:2]).any()
            or (plant.XREF_INIT_MAX[:2] + upper > plant.X_MAX[:2]).any()):
        raise ValueError("Initial position offset would put reference starts outside the state domain")
    plant.XE_INIT_MIN[:2] = offset
    plant.XE_INIT_MAX[:2] = upper
    plant.XE_INIT_SIGN_DIMS = [0, 1] if independent_signs else []
    plant._ref_pool = None  # Discard starts drawn with the previous offset.


def _view_policy(env, agent, simulation_app):
    """Replay across automatic episode resets without performing learning updates."""
    import time
    import torch
    from contractionRL.agents.utils.RL.evaluation import EvaluationControl

    agent.enable_training_mode(False)
    for model in getattr(agent, "models", {}).values():
        if model is not None and hasattr(model, "eval"):
            model.eval()
    obs, _ = env.reset()
    raw = env.unwrapped
    dt = float(raw.step_dt)
    display = None
    if getattr(getattr(raw, "plant", None), "task", None) == "car":
        from car_viewer import CarViewer
        display = CarViewer(raw)

    def focus_camera():
        plant = getattr(raw, "plant", None)
        if plant is not None and tuple(plant.state_names[:2]) == ("pos_x", "pos_y"):
            px, py = plant.x_t[0, :2].tolist()
            raw.sim.set_camera_view((px + 20., py + 20., 25.), (px + 8., py, 0.))

    if display is None:
        focus_camera()
    print(f"[viewer] Continuous deterministic playback; dt={dt:g}s; no training.", flush=True)
    step = 0
    with EvaluationControl(env, agent) as control:
        while simulation_app.is_running():
            started = time.monotonic()
            states = env.state() if hasattr(env, "state") else None
            actions = control.act(obs, states)
            obs, _, terminated, truncated, _ = env.step(actions)
            control.after_step(terminated | truncated)
            if display is not None:
                display.update()
            elif (terminated | truncated).flatten()[0]:
                focus_camera()
            step += 1
            if step == 100 and os.environ.get("ISAAC_VIEWER_CAPTURE"):
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                capture_viewport_to_file(get_active_viewport(), os.environ["ISAAC_VIEWER_CAPTURE"])
            if step % 500 == 0:
                print(f"[viewer] Simulated {step} steps ({step * dt:g}s). {control.summary()}", flush=True)
            if not control.enabled:
                time.sleep(max(0., dt - (time.monotonic() - started)))
        if control.enabled:
            print(f"[viewer] Control timing: {control.summary()}", flush=True)


# Classic route

def build_classic_eval_runner(task, algorithm, agent_cfg, *, device, num_envs,
                              num_envs_for_eval, ref_length=None, disable_disturbance=False):
    """Env + runner for evaluating a trained CLASSIC checkpoint.

    Shared with scripts/policy_gap.py rather than copied into it: the
    ref-window sizing below is subtle enough that this script once got it wrong
    and could not load a single C2RL checkpoint (see the comment inline).
    Returns ``(env, runner)``; the caller loads the weights and closes the env.
    """
    import gymnasium as gym
    from contractionRL.agents.utils.contraction.metrics import StatManagerEnvWrapper
    from contractionRL.agents.utils.RL.runner import CLActorRunner
    from contractionRL.runners import ContractionRunner
    from train_utils import (
        _default_num_envs_classic,
        _disarm_termination_for_eval,
        _inject_angle_idx,
        _resolve_symmetry_for_env,
        normalize_agent_cfg,
        configure_tracking_context,
    )

    # Eval-time config hygiene: no wandb, throwaway experiment dir, headless.
    exp = agent_cfg["agent"].setdefault("experiment", {})
    exp["wandb"] = False
    exp["directory"] = os.path.abspath(os.path.join("logs", "play", "_scratch"))
    exp["checkpoint_interval"] = 0
    exp["write_interval"] = 0
    agent_cfg.setdefault("trainer", {})
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["trainer"]["headless"] = True

    num_envs = num_envs if num_envs is not None else _default_num_envs_classic(algorithm)
    num_envs = max(num_envs, num_envs_for_eval)

    recorded = agent_cfg.get("environment", {})
    env_kwargs = {key: recorded[key] for key in ("compact_reference", "reference_mode", "disturbance_scales", "time_bound", "dt") if key in recorded}
    if algorithm in ("contraction_ppo", "contraction-ppo", "contractionppo"):
        env_kwargs["compact_reference"] = True
    if disable_disturbance:
        import importlib
        from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
        module = importlib.import_module(type(make_analytical_plant(task)).__module__)
        env_kwargs["disturbance_scales"] = [0.] * module.ENV_CONFIG["num_dim_x"]
    from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
    env = make_analytical_plant(task, num_envs=num_envs, device=device, **env_kwargs)
    # Playback is a measurement, same as train.py's evaluator — see
    # _disarm_termination_for_eval for why a truncated horizon inverts AUC.
    _disarm_termination_for_eval(env, "[play]")
    # Rebuild the same reference window the checkpoint was trained with.
    # train.py sizes it AUTO from gamma (RefWindow.length_for_horizon) unless
    # --ref_length was given, and its evaluator mirrors the training env's
    # window explicitly. play.py did neither, so it built a length-1 window
    # and every AUTO-sized checkpoint failed to load with a state_dict shape
    # mismatch (w2 18-wide in the checkpoint vs 9-wide here at gamma=0.01) —
    # i.e. this script could not evaluate any C2RL run at all.
    if hasattr(env.unwrapped, "configure_ref_window"):
        _gamma = float(agent_cfg["agent"].get("discount_factor", 0.99))
        _offset = int(recorded.get("ref_offset", 1))
        _len = ref_length or recorded.get("ref_length") or env.unwrapped.full_ref_length(offset=_offset)
        env.unwrapped.configure_ref_window(length=_len, offset=_offset, gamma=_gamma)
    configure_tracking_context(env.unwrapped, agent_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    env = StatManagerEnvWrapper(env, num_envs_for_eval=num_envs_for_eval)

    if algorithm in _CONTRACTION_ALGOS:
        runner = ContractionRunner(env, agent_cfg, task_id=task,
                                   num_envs=num_envs, is_classic=True)
    else:
        normalize_agent_cfg(agent_cfg, algorithm=algorithm, device=env.device)
        _angle_idx = list(getattr(env.unwrapped, "angle_idx", []) or [])
        _inject_angle_idx(agent_cfg, _angle_idx, _resolve_symmetry_for_env(env))
        runner = CLActorRunner(env, agent_cfg)
    return env, runner



# Isaac route

def run_isaac(args, simulation_app) -> list[dict]:
    import contractionRL.tasks  # noqa: F401 — registers Isaac tasks
    import gymnasium as gym
    from contractionRL.agents.utils.contraction.metrics import StatManagerEnvWrapper
    from contractionRL.agents.utils.RL.runner import CLActorRunner
    from contractionRL.runners import ContractionRunner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    models = _discover_models(args.task, False, args.models_dir, args.algorithm, args.checkpoint)
    if not models:
        print(f"[play] No models found for {args.task}.")
        return []

    device = args.device
    results: list[dict] = []
    for algorithm, ckpt in models:
        print(f"\n[play] === {algorithm} ===  ({ckpt})")
        entry_key = ("skrl_cfg_entry_point" if algorithm == "ppo"
                     else f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point")
        try:
            agent_cfg = _checkpoint_agent_cfg(ckpt, load_cfg_from_registry(args.task, entry_key))
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] no config entry '{entry_key}' for {args.task}: {e}")
            continue
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        from contractionRL.tasks.direct.common.isaac.backend import configure_backend, configure_backend_agent
        backend = configure_backend(env_cfg, task=args.task)
        if args.policy_control_mode is not None:
            agent_cfg["models"]["policy"]["control_mode"] = args.policy_control_mode
        if args.disable_disturbance:
            if not hasattr(env_cfg, "disturbance_enabled"):
                raise ValueError("--disable_disturbance supports classic-derived Isaac tasks")
            env_cfg.disturbance_enabled = False
        if args.viewer_disturbance_bounds is not None:
            if backend != "analytical" or getattr(env_cfg, "classic_task", None) != "classic-car-v0":
                raise ValueError("--viewer_disturbance_bounds requires the analytical car viewer")
            env_cfg.disturbance_scales = [0., 0., *args.viewer_disturbance_bounds]
        configure_backend_agent(agent_cfg, args.task, classic_derived=hasattr(env_cfg, "classic_task"))
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if device is not None:
            env_cfg.sim.device = device
        env_cfg.seed = args.seed if args.seed is not None else agent_cfg.get("seed", 42)
        agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
        agent_cfg["agent"]["experiment"]["wandb"] = False
        agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(os.path.join("logs", "play", "_scratch"))
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["write_interval"] = 0

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        from contractionRL.tasks.direct.common.transfer_dynamics import prepare_transfer_dynamics, is_transfer_path
        prepare_transfer_dynamics(env.unwrapped, agent_cfg, checkpoint=ckpt)
        if args.viewer_initial_offset is not None:
            _set_viewer_initial_offset(env.unwrapped, args.viewer_initial_offset)
        elif args.viewer_initial_offset_box is not None:
            dx_min, dx_max, dy_min, dy_max = args.viewer_initial_offset_box
            _set_viewer_initial_offset(env.unwrapped, [dx_min, dy_min], [dx_max, dy_max])
        elif args.viewer_initial_offset_signed_range is not None:
            lo, hi = args.viewer_initial_offset_signed_range
            _set_viewer_initial_offset(env.unwrapped, [lo, lo], [hi, hi], independent_signs=True)
        from train_utils import configure_tracking_context
        configure_tracking_context(env.unwrapped, agent_cfg)
        env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
        env = StatManagerEnvWrapper(env, num_envs_for_eval=args.num_envs_for_eval)

        try:
            _alg = algorithm.lower()
            from train_utils import normalize_agent_cfg, _inject_angle_idx, _resolve_symmetry_for_env
            _inject_angle_idx(agent_cfg, list(getattr(env.unwrapped, 'angle_idx', []) or []), _resolve_symmetry_for_env(env))
            normalize_agent_cfg(agent_cfg, algorithm=_alg, device=env.device)
            if _alg in _CONTRACTION_ALGOS:
                runner = ContractionRunner(env, agent_cfg, is_classic=False)
            else:
                runner = CLActorRunner(env, agent_cfg)
            print("  loading checkpoint …")
            runner.agent.load(ckpt)
            from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer
            if isinstance(runner.agent, C2RLAgent):
                runner.agent.freeze_cmg()
                C2RLSkrlTrainer._inject_ccm(env, runner.agent)

            if args.viewer:
                if is_transfer_path(env.unwrapped):
                    from contractionRL.tasks.direct.common.transfer_dynamics import evaluation_backend
                    from contractionRL.agents.utils.RL.logging import evaluation_section
                    with evaluation_backend(env.unwrapped, 'physx'), evaluation_section('Sim-to-Real Evaluation'):
                        _view_policy(env, runner.agent, simulation_app)
                else:
                    _view_policy(env, runner.agent, simulation_app)
                return []
            max_steps = int(getattr(env, "max_episode_length", 1000)) + 1
            if is_transfer_path(env.unwrapped):
                from contractionRL.tasks.direct.common.transfer_dynamics import evaluate_backend_pair
                pair = evaluate_backend_pair(env, lambda: _rollout_stability(env, runner.agent, max_steps=max_steps))
                for backend, summary in pair.items():
                    _print_summary(f'{algorithm}/{backend}', summary)
                    results.append({'algorithm': algorithm, 'backend': backend, 'checkpoint': ckpt, **summary})
                continue
            summary = _rollout_stability(env, runner.agent, max_steps=max_steps)
        finally:
            env.close()

        _print_summary(algorithm, summary)
        results.append({"algorithm": algorithm, "checkpoint": ckpt, **summary})
    return results


# Reporting

_METRIC_KEYS = ["auc_mean", "contraction_rate_mean", "running_lambda_mean",
                "overshoot_mean", "contraction_score_mean"]
# Short column labels so the table header lines up with the 16-wide value cells.
_METRIC_LABELS = {
    "auc_mean": "auc",
    "contraction_rate_mean": "lambda",
    "running_lambda_mean": "lambda_run",
    "overshoot_mean": "overshoot",
    "contraction_score_mean": "score",
}


def _print_summary(algorithm: str, summary: dict) -> None:
    if not summary:
        print(f"  {algorithm}: no stability metrics (buffer never completed).")
        return
    parts = [f"{_METRIC_LABELS[k]}={summary.get(k, float('nan')):.4g}" for k in _METRIC_KEYS]
    if "control_delay_steps" in summary:
        parts.append(f"delayed={summary['control_delay_steps']}/{summary['control_steps']}")
        parts.append(f"held={summary['control_hold_steps']}")
    print(f"  {algorithm}: " + "  ".join(parts))


def _print_table(task: str, results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"Stability summary — {task}   (lower auc = better, higher score = better)")
    print("=" * 78)
    header = f"{'algorithm':<14}" + "".join(f"{_METRIC_LABELS[k]:>16}" for k in _METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['algorithm']:<14}"
        for k in _METRIC_KEYS:
            v = r.get(k)
            row += f"{v:>16.4g}" if isinstance(v, (int, float)) else f"{'—':>16}"
        print(row)
    print("=" * 78)


def _save_results(task: str, results: list[dict]) -> None:
    import json
    from datetime import datetime

    out_dir = os.path.join("logs", "play")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_env_name(task)}_stability_{datetime.now():%Y-%m-%d_%H-%M-%S}.json")
    with open(out_path, "w") as f:
        json.dump({"task": task, "results": results}, f, indent=2)
    print(f"[play] Wrote {out_path}")


# Entry point

def main() -> None:
    _reject_removed_flags()
    args_cli, _ = parser.parse_known_args()
    if any(value is not None for value in (args_cli.viewer_initial_offset, args_cli.viewer_initial_offset_box,
                                           args_cli.viewer_initial_offset_signed_range,
                                           args_cli.viewer_disturbance_bounds)) and not args_cli.viewer:
        parser.error("Viewer initial offsets and disturbance bounds require --viewer")
    if args_cli.viewer_disturbance_bounds is not None and args_cli.disable_disturbance:
        parser.error("--viewer_disturbance_bounds cannot be combined with --disable_disturbance")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _hydra = parser.parse_known_args()
    args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/hangDetector/enabled=false"
    sys.argv = [sys.argv[0]] + _hydra
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        results = run_isaac(args_cli, simulation_app)
        if results:
            _print_table(args_cli.task, results)
            _save_results(args_cli.task, results)
    except Exception:
        # Isaac's fast shutdown may exit the process; print before cleanup.
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    return


if __name__ == "__main__":
    _reject_removed_flags()
    main()

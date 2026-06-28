"""Train / evaluate an mjrl contraction agent on a classic tracking environment.

mjrl mirrors skrl's high-level API: load a YAML config, build a Runner, run.
Unlike scripts/skrl/train.py this does NOT require Isaac Sim — the classic envs
are pure gymnasium + numpy/torch.

Usage:
    python scripts/mjrl/train.py --task Car-Direct-v0 --algorithm c3m
    python scripts/mjrl/train.py --task Car-Direct-v0 --algorithm lqr --num_agent 8
    python scripts/mjrl/train.py --task Car-Direct-v0 --algorithm c3m --epochs 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import yaml


def _bootstrap_paths() -> Path:
    """Put the repo's mjrl package and the classic-env root on sys.path.

    Returns the directory that contains the standalone ``classic`` package.
    """
    repo_root = Path(__file__).resolve().parents[2]
    # mjrl package (top-level in repo)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # classic envs: import as standalone ``classic`` package (no Isaac deps)
    classic_root = repo_root / "source" / "contractionRL" / "contractionRL" / "tasks" / "direct"
    if str(classic_root) not in sys.path:
        sys.path.insert(0, str(classic_root))
    return classic_root


def _resolve_cfg_path(spec: str) -> str:
    """Resolve a gym kwarg entry point of the form 'pkg.subpkg:file.yaml'."""
    module, _, fname = spec.partition(":")
    mod = __import__(module, fromlist=["__file__"])
    return os.path.join(os.path.dirname(mod.__file__), fname)


def main():
    parser = argparse.ArgumentParser(description="Train/eval an mjrl agent on a classic env.")
    parser.add_argument("--task", type=str, default="Car-Direct-v0", help="Registered classic env id.")
    parser.add_argument("--algorithm", type=str, default="c3m", help="Algorithm: lqr | c3m | sd_lqr | temp.")
    parser.add_argument("--num_agent", "--num-agent", type=int, default=None,
                        help="Number of sampler workers (num_agent == num_worker).")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs (C3M/TEMP).")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda override.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--dynamics_checkpoint", "--dynamics-checkpoint",
        type=str, default=None,
        help="Path to a NeuralDynamics .pt checkpoint.  When given the agent uses "
             "learned dynamics instead of the env's analytical get_f_and_B.  "
             "Required for envs without closed-form dynamics (e.g. Isaac Sim).",
    )
    parser.add_argument(
        "--use_analytical_dynamics", "--use-analytical-dynamics",
        action="store_true", default=False,
        help="Force use of the env's analytical get_f_and_B even when "
             "--dynamics_checkpoint is provided.  For classic direct envs only.",
    )
    args = parser.parse_args()

    _bootstrap_paths()

    # register classic envs (standalone import, no Isaac)
    import classic  # noqa: F401

    from mjrl.utils.runner.torch import Runner
    from mjrl.models.dynamics import NeuralDynamics

    # locate the agent cfg entry point registered for this task
    algo = args.algorithm.lower()
    spec = gym.spec(args.task)
    cfg_key = f"mjrl_{algo}_cfg_entry_point"
    if cfg_key not in spec.kwargs:
        available = [k for k in spec.kwargs if k.startswith("mjrl_")]
        raise SystemExit(f"No config '{cfg_key}' for task {args.task}. Available: {available}")

    cfg_path = _resolve_cfg_path(spec.kwargs[cfg_key])
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_agent is not None:
        cfg.setdefault("trainer", {})["num_agent"] = args.num_agent
    if args.epochs is not None:
        cfg.setdefault("trainer", {})["epochs"] = args.epochs
    if args.device is not None:
        cfg.setdefault("agent", {})["device"] = args.device

    # build the env (pass through env kwargs that are not config entry points)
    env = gym.make(args.task).unwrapped

    # resolve dynamics model
    dynamics_model = None
    if not args.use_analytical_dynamics and args.dynamics_checkpoint is not None:
        dynamics_model = NeuralDynamics.load(
            args.dynamics_checkpoint,
            device=cfg.get("agent", {}).get("device", None),
        )
    dynamics_mode = (
        "analytical" if args.use_analytical_dynamics or dynamics_model is None
        else f"learned ({args.dynamics_checkpoint})"
    )

    print(f"[mjrl] task={args.task}  algorithm={algo}  "
          f"num_agent(num_worker)={cfg.get('trainer', {}).get('num_agent')}  "
          f"dynamics={dynamics_mode}")
    print(f"[mjrl] config: {cfg_path}")

    runner = Runner(env, cfg, dynamics_model=dynamics_model)
    runner.run()


if __name__ == "__main__":
    main()

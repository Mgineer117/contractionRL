"""Train an mjrl contraction agent on an Isaac Sim environment.

Unlike scripts/mjrl/train.py (classic envs, no Isaac), this script boots
Isaac Sim and wraps the live env with IsaacMjrlWrapper so that LQR and C3M
can consume a learned NeuralDynamics model.

Requires a pre-trained dynamics checkpoint (--dynamics_checkpoint):
    python scripts/mjrl/collect_isaac_data.py --task <TASK> --save data/x.npz
    python scripts/mjrl/pretrain_dynamics.py  --source npz --data data/x.npz \\
        --x_dim <OBS_DIM> --u_dim <ACT_DIM> --save checkpoints/dyn.pt
    python scripts/mjrl/train_isaac.py \\
        --task <TASK> --algorithm lqr|c3m \\
        --dynamics_checkpoint checkpoints/dyn.pt

Usage examples:
    python scripts/mjrl/train_isaac.py \\
        --task Quadruped-Vel-Tracking-Direct-v0 --algorithm lqr \\
        --dynamics_checkpoint checkpoints/quadruped_dynamics.pt \\
        --num_envs 4 --headless

    python scripts/mjrl/train_isaac.py \\
        --task Quadruped-Vel-Tracking-Direct-v0 --algorithm c3m \\
        --dynamics_checkpoint checkpoints/quadruped_dynamics.pt \\
        --num_envs 4 --headless --epochs 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── argument parsing before Isaac Sim import ──────────────────────────────
parser = argparse.ArgumentParser(description="Train mjrl agent on Isaac Sim env.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--algorithm", type=str, default="lqr", help="lqr | c3m")
parser.add_argument("--dynamics_checkpoint", type=str, required=True,
                    help="Path to a NeuralDynamics .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_agent", type=int, default=4,
                    help="Number of sampler workers (num_agent == num_worker).")
parser.add_argument("--epochs", type=int, default=None, help="Training epochs (C3M).")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--Q_scaler", type=float, default=1.0)
parser.add_argument("--R_scaler", type=float, default=1.0)
args_cli, _ = parser.parse_known_args()

_kit_extra = " --/app/hangDetector/enabled=false"

# ── repo root on sys.path (for mjrl package) ──────────────────────────────
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── IsaacLab bootstrap ────────────────────────────────────────────────────
from isaaclab.app import AppLauncher  # noqa: E402

_app_args = argparse.Namespace(
    task=args_cli.task,
    num_envs=args_cli.num_envs,
    headless=args_cli.headless,
    seed=args_cli.seed,
    kit_args=_kit_extra,
    video=False,
    video_length=0,
    video_interval=0,
    disable_fabric=False,
)
AppLauncher.add_app_launcher_args(parser)
app_launcher = AppLauncher(args=_app_args)
simulation_app = app_launcher.app

# ── imports after app start ───────────────────────────────────────────────
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import contractionRL.tasks  # noqa: F401, E402

from mjrl.models.dynamics import NeuralDynamics  # noqa: E402
from mjrl.utils.isaac_wrapper import IsaacMjrlWrapper  # noqa: E402
from mjrl.utils.runner.torch import Runner  # noqa: E402


def _build_cfg(algo: str, dyn: NeuralDynamics, args) -> dict:
    """Build a minimal mjrl cfg dict for the given algorithm."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if algo == "lqr":
        return {
            "seed": args.seed,
            "agent": {
                "class": "LQR",
                "device": device,
                "Q_scaler": args.Q_scaler,
                "R_scaler": args.R_scaler,
            },
            "trainer": {
                "class": "EvalTrainer",
                "num_agent": args.num_agent,
                "num_eval_rounds": 1,
            },
        }
    elif algo == "c3m":
        return {
            "seed": args.seed,
            "agent": {
                "class": "C3M",
                "device": device,
                "W_lr": 3e-4,
                "u_lr": 3e-4,
                "lbd": 0.5,
                "w_ub": 10.0,
                "w_lb": 0.01,
                "buffer_size": 65536,
                "cmg": {"hidden_dim": [128, 128], "activation": "tanh", "mode": "deterministic"},
                "actor": {"hidden_dim": [128, 128], "activation": "tanh"},
            },
            "trainer": {
                "class": "C3MTrainer",
                "num_agent": args.num_agent,
                "epochs": args.epochs or 30000,
                "eval_interval": 2000,
            },
        }
    else:
        raise SystemExit(f"Unknown algorithm '{algo}'. Choose from: lqr, c3m")


def main():
    device = args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load dynamics checkpoint
    dyn = NeuralDynamics.load(args_cli.dynamics_checkpoint, device=device)

    # build Isaac env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
    )
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)

    # wrap with mjrl interface
    wrapper = IsaacMjrlWrapper(isaac_env, dyn, device=device)

    # build cfg and run
    cfg = _build_cfg(args_cli.algorithm, dyn, args_cli)
    print(
        f"[train_isaac] task={args_cli.task}  algo={args_cli.algorithm}  "
        f"x_dim={dyn.x_dim}  u_dim={dyn.u_dim}  device={device}"
    )

    runner = Runner(wrapper, cfg, dynamics_model=dyn)
    runner.run()

    isaac_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

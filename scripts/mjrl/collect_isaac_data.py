"""Collect (x, u, x_dot) dynamics data from an Isaac Sim environment.

Runs a uniform-random policy for ``--n_steps`` environment steps, records
(obs, action, next_obs) tuples, then computes  x_dot ≈ (next_obs − obs) / dt
and saves a .npz file that ``pretrain_dynamics.py --source npz`` can consume.

Usage:
    python scripts/mjrl/collect_isaac_data.py \\
        --task Quadruped-Vel-Tracking-Direct-v0 \\
        --n_steps 200000 \\
        --save data/quadruped_dynamics_data.npz \\
        --num_envs 64 \\
        --headless

The .npz contains:
    x      (N, obs_dim)  float32  — observations (treated as state)
    u      (N, act_dim)  float32  — random actions applied
    x_dot  (N, obs_dim)  float32  — finite-difference obs derivative
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── argument parsing must happen before Isaac Sim is imported ──────────────
parser = argparse.ArgumentParser(description="Collect dynamics data from Isaac Sim.")
parser.add_argument("--task", type=str, required=True,
                    help="Registered Isaac env id, e.g. Quadruped-Vel-Tracking-Direct-v0.")
parser.add_argument("--n_steps", type=int, default=200_000,
                    help="Total env steps to collect (across all parallel envs).")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of parallel Isaac envs.")
parser.add_argument("--save", type=str, default="data/dynamics_data.npz",
                    help="Output path for the .npz dataset.")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=42)
args_cli, _remainder = parser.parse_known_args()

# Inject kit args before AppLauncher so hang-detector never fires
import sys as _sys  # noqa: E402  (already imported above, but needed for kit_args)

_kit_extra = " --/app/hangDetector/enabled=false"

# ── IsaacLab bootstrap ─────────────────────────────────────────────────────
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

# ── imports after app is alive ─────────────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import contractionRL.tasks  # noqa: F401, E402  — register custom envs


def main():
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_envs=args_cli.num_envs,
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = env.observation_space.shape[-1] if hasattr(env.observation_space, "shape") else int(env.observation_space)
    act_dim = env.action_space.shape[-1]
    num_envs = env.num_envs
    dt = getattr(env, "step_dt", getattr(env.cfg.sim, "dt", 0.02))

    # estimate steps per env
    steps_per_env = max(1, args_cli.n_steps // num_envs)
    n_alloc = steps_per_env * num_envs

    print(f"[collect] task={args_cli.task}  envs={num_envs}  obs_dim={obs_dim}  act_dim={act_dim}  dt={dt:.4f}")
    print(f"[collect] collecting {steps_per_env} steps/env → ~{n_alloc} total transitions")

    xs, us, x_dots = [], [], []

    obs_dict, _ = env.reset()
    obs = _extract_obs(obs_dict, obs_dim)  # (num_envs, obs_dim)

    for step in range(steps_per_env):
        # uniform random action within action space bounds
        act_low = torch.as_tensor(env.action_space.low, device=env.device)
        act_high = torch.as_tensor(env.action_space.high, device=env.device)
        action = act_low + (act_high - act_low) * torch.rand(num_envs, act_dim, device=env.device)

        next_obs_dict, _, terminated, truncated, _ = env.step(action)
        next_obs = _extract_obs(next_obs_dict, obs_dim)  # (num_envs, obs_dim)

        x_dot = (next_obs - obs) / dt  # finite-difference state derivative

        # mask out resets (terminated/truncated envs have discontinuous x_dot)
        done = (terminated | truncated).cpu().numpy()
        mask = ~done  # shape (num_envs,)

        if mask.any():
            xs.append(obs[mask].cpu().numpy())
            us.append(action[mask].cpu().numpy())
            x_dots.append(x_dot[mask].cpu().numpy())

        obs = next_obs

        if (step + 1) % max(1, steps_per_env // 10) == 0:
            n_so_far = sum(a.shape[0] for a in xs)
            print(f"  step {step+1}/{steps_per_env}  transitions collected: {n_so_far}")

    env.close()

    xs = np.concatenate(xs, axis=0).astype(np.float32)
    us = np.concatenate(us, axis=0).astype(np.float32)
    x_dots = np.concatenate(x_dots, axis=0).astype(np.float32)

    out_path = Path(args_cli.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), x=xs, u=us, x_dot=x_dots)
    print(f"[collect] saved {xs.shape[0]} transitions → {out_path}")
    print(f"  x: {xs.shape}  u: {us.shape}  x_dot: {x_dots.shape}")


def _extract_obs(obs_dict, obs_dim: int) -> torch.Tensor:
    """Pull a flat obs tensor from whatever dict the IsaacLab env returns."""
    if isinstance(obs_dict, dict):
        # ManagerBasedRLEnv returns {"policy": tensor}
        if "policy" in obs_dict:
            return obs_dict["policy"].float()
        # fallback: concatenate all values
        tensors = [v.float().reshape(v.shape[0], -1) for v in obs_dict.values()
                   if isinstance(v, torch.Tensor)]
        return torch.cat(tensors, dim=-1)
    if isinstance(obs_dict, torch.Tensor):
        return obs_dict.float()
    raise ValueError(f"Unexpected obs type: {type(obs_dict)}")


if __name__ == "__main__":
    main()
    simulation_app.close()

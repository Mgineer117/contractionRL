"""Pre-train a NeuralDynamics model before running C3M or SD-LQR.

Two data sources:
  analytical  — classic env computes exact x_dot = f(x) + B(x)u analytically.
                Use this for Car-Direct-v0 and any other classic env.
  npz         — load a pre-collected (x, u, x_dot) dataset from a .npz file.
                Use this for Isaac Sim envs after running collect_isaac_data.py.

Usage:
    # classic env, analytical ground truth
    python scripts/mjrl/pretrain_dynamics.py \\
        --task Car-Direct-v0 --source analytical \\
        --n_samples 100000 --epochs 3000 \\
        --save checkpoints/car_dynamics.pt

    # Isaac env data collected by collect_isaac_data.py
    python scripts/mjrl/pretrain_dynamics.py \\
        --source npz --data data/quadruped_data.npz \\
        --x_dim 48 --u_dim 12 \\
        --n_samples 100000 --epochs 3000 \\
        --save checkpoints/quadruped_dynamics.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _bootstrap_paths():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    classic_root = repo_root / "source" / "contractionRL" / "contractionRL" / "tasks" / "direct"
    if str(classic_root) not in sys.path:
        sys.path.insert(0, str(classic_root))


def _collect_classic(env, n_samples: int) -> dict:
    """Use env.get_rollout(..., mode='dynamics') for analytical (x, u, x_dot)."""
    print(f"[pretrain] collecting {n_samples} analytical samples from {env.task}...")
    data = env.get_rollout(n_samples, mode="dynamics")
    print(f"  x: {data['x'].shape}  u: {data['u'].shape}  x_dot: {data['x_dot'].shape}")
    return data


def _load_npz(path: str) -> dict:
    print(f"[pretrain] loading dataset from {path}")
    npz = np.load(path)
    data = {k: npz[k].astype(np.float32) for k in ("x", "u", "x_dot")}
    print(f"  x: {data['x'].shape}  u: {data['u'].shape}  x_dot: {data['x_dot'].shape}")
    return data


def train(
    model,
    data: dict,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    save_path: str,
    val_frac: float = 0.1,
):
    from mjrl.models.dynamics import NeuralDynamics  # noqa: F401

    model = model.to(torch.device(device))
    model.device = torch.device(device)
    model.train()

    x = torch.from_numpy(data["x"]).to(device)
    u = torch.from_numpy(data["u"]).to(device)
    x_dot = torch.from_numpy(data["x_dot"]).to(device)

    n = x.shape[0]
    n_val = max(1, int(n * val_frac))
    idx = torch.randperm(n)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    x_v, u_v, xd_v = x[val_idx], u[val_idx], x_dot[val_idx]
    x_t, u_t, xd_t = x[trn_idx], u[trn_idx], x_dot[trn_idx]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    best_val, best_state = float("inf"), None
    n_trn = x_t.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_trn, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_trn, batch_size):
            bi = perm[start : start + batch_size]
            xb, ub, xdb = x_t[bi], u_t[bi], xd_t[bi]
            xd_pred = model.predict_x_dot(xb, ub)
            loss = mse(xd_pred, xdb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_loss = mse(model.predict_x_dot(x_v, u_v), xd_v).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  epoch {epoch:5d}/{epochs}  "
                f"train={total_loss/n_batches:.4e}  val={val_loss:.4e}  "
                f"best_val={best_val:.4e}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.save(save_path)
    print(f"[pretrain] done — best val loss={best_val:.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-train NeuralDynamics model.")
    parser.add_argument("--source", choices=["analytical", "npz"], default="analytical",
                        help="'analytical': use classic env; 'npz': load pre-collected data file.")
    parser.add_argument("--task", type=str, default="Car-Direct-v0",
                        help="Classic env id (for --source analytical).")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz data file (for --source npz).")
    parser.add_argument("--x_dim", type=int, default=None,
                        help="State dim (required for --source npz).")
    parser.add_argument("--u_dim", type=int, default=None,
                        help="Control dim (required for --source npz).")
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="checkpoints/dynamics.pt",
                        help="Output checkpoint path.")
    args = parser.parse_args()

    _bootstrap_paths()

    from mjrl.models.dynamics import NeuralDynamics

    if args.source == "analytical":
        import gymnasium as gym
        import classic  # noqa: F401 — registers classic envs

        env = gym.make(args.task).unwrapped
        data = _collect_classic(env, args.n_samples)
        x_dim = int(env.num_dim_x)
        u_dim = int(env.num_dim_control)

    else:  # npz
        if args.data is None:
            raise SystemExit("--data PATH required for --source npz")
        if args.x_dim is None or args.u_dim is None:
            raise SystemExit("--x_dim and --u_dim required for --source npz")
        data = _load_npz(args.data)
        x_dim, u_dim = args.x_dim, args.u_dim
        if args.n_samples < data["x"].shape[0]:
            idx = np.random.choice(data["x"].shape[0], args.n_samples, replace=False)
            data = {k: v[idx] for k, v in data.items()}

    model = NeuralDynamics(
        x_dim=x_dim,
        u_dim=u_dim,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
    )
    print(
        f"[pretrain] NeuralDynamics  x_dim={x_dim}  u_dim={u_dim}  "
        f"hidden={args.hidden_dim}  act={args.activation}  device={args.device}"
    )

    train(
        model=model,
        data=data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

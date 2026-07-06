"""Shared NeuralDynamics pretraining loop for contraction agents (C3M, C2RL).

Both C3M and C2RL learn ``ẋ = f(x) + B(x)·u`` with a NeuralDynamics network that
is pretrained *before* the main training loop and then refined online each
update/epoch. This module holds the single pretraining implementation so the two
agents cannot drift apart.

The agent passed in must expose the (identical across C3M/C2RL) interface:
    ._neural_dynamics, ._train_dynamics(data), ._get_rollout(n, "dynamics"),
    ._cfg.dynamics_batch_size, ._dynamics_optimizer, ._dynamics_lr_scheduler,
    .track_data(), .writer, .write_tracking_data()  (skrl Agent methods)
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import tqdm as _tqdm


def pretrain_dynamics(agent, *, epochs: int, data_path: str | None,
                      timesteps: int, log_interval: int | None = None,
                      tag: str = "") -> None:
    """Pretrain ``agent._neural_dynamics`` for ``epochs`` epochs.

    When ``data_path`` points to an offline ``dynamics_data.npz`` (keys ``x``,
    ``u``, ``x_dot``, optional ``lengths``), each epoch iterates minibatches over
    that data; otherwise it falls back to online rollouts (1 rollout = 1 epoch).
    No-op when the agent has no NeuralDynamics (analytical dynamics) or epochs<=0.

    Pretraining metrics (dynamics MSE + LR) are flushed to the agent's writer at
    negative timesteps (so they precede training on the ``global_step`` x-axis).
    The flush cadence is derived from ``epochs`` — NOT the agent's training
    ``write_interval``, which is typically ``timesteps//100`` and would never
    fire within a short pretraining loop — and the final epoch always flushes.
    ``log_interval`` optionally caps that cadence; ``tag`` labels console prints.
    """
    if agent._neural_dynamics is None or epochs <= 0:
        return

    # ~100 wandb points regardless of the (huge) training write_interval.
    log_every = max(1, epochs // 100)
    if log_interval and log_interval > 0:
        log_every = min(log_every, log_interval)
    has_writer = getattr(agent, "writer", None) is not None

    dev = agent._neural_dynamics.device
    x = u = x_dot = n = None
    batches_per_epoch = 1

    if data_path:
        print(f"{tag} Loading dynamics pretrain data from {data_path}")
        npz = np.load(data_path)

        # Filter whole episodes that contain any NaN in x/u/x_dot.
        nan_mask = (
            np.isnan(npz["x"]).any(axis=(1, 2))
            | np.isnan(npz["u"]).any(axis=(1, 2))
            | np.isnan(npz["x_dot"]).any(axis=(1, 2))
        )
        lengths_arr = npz["lengths"] if "lengths" in npz else None
        if nan_mask.any():
            print(f"{tag} WARNING: Found NaNs in {nan_mask.sum()} offline episodes! Filtering them out...")
            valid_mask = ~nan_mask
            x_arr = npz["x"][valid_mask]
            u_arr = npz["u"][valid_mask]
            x_dot_arr = npz["x_dot"][valid_mask]
            if lengths_arr is not None:
                lengths_arr = lengths_arr[valid_mask]
        else:
            x_arr, u_arr, x_dot_arr = npz["x"], npz["u"], npz["x_dot"]

        # Unpack (N, T, S) trajectories into flat (n, S) samples. When a
        # `lengths` array is present, steps >= lengths[n] are padding (the last
        # valid state repeated) — mask them out so the fit isn't biased toward
        # artificial x_dot ~ 0 samples.
        if lengths_arr is not None:
            T_len = x_arr.shape[1]
            step_mask = np.arange(T_len)[None, :] < lengths_arr[:, None]  # (N, T)
            x_arr = x_arr[step_mask]
            u_arr = u_arr[step_mask]
            x_dot_arr = x_dot_arr[step_mask]
            print(f"{tag} Unpacked {step_mask.sum()} valid samples from {step_mask.shape[0]} trajectories (padding masked)")

        x = torch.from_numpy(x_arr).reshape(-1, x_arr.shape[-1]).to(torch.float32).to(dev)
        u = torch.from_numpy(u_arr).reshape(-1, u_arr.shape[-1]).to(torch.float32).to(dev)
        x_dot = torch.from_numpy(x_dot_arr).reshape(-1, x_dot_arr.shape[-1]).to(torch.float32).to(dev)
        n = x.shape[0]
        batches_per_epoch = max(1, n // agent._cfg.dynamics_batch_size)

    dyn_pbar = _tqdm.tqdm(range(epochs), desc="Pretraining dynamics", file=sys.stdout)
    for epoch in dyn_pbar:
        if x is not None:
            for _ in range(batches_per_epoch):
                dbz = min(agent._cfg.dynamics_batch_size, n)
                idx = torch.randint(0, n, (dbz,), device=dev)
                loss_val = agent._train_dynamics({"x": x[idx], "u": u[idx], "x_dot": x_dot[idx]})
        else:
            # Online pretraining using rolling data (1 rollout = 1 epoch).
            dyn_data = agent._get_rollout(agent._cfg.dynamics_batch_size, "dynamics")
            loss_val = agent._train_dynamics(dyn_data)

        dyn_pbar.set_postfix(loss=f"{loss_val:.3g}")

        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent._dynamics_lr_scheduler.step()

        agent.track_data("Loss / Pretrain/dynamics_mse", loss_val)
        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_lr_scheduler.get_last_lr()[0])
        else:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_optimizer.param_groups[0]["lr"])

        # Flush at negative timesteps so pretraining precedes training on the
        # global_step x-axis; always flush the final epoch so short runs log too.
        if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
            agent.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

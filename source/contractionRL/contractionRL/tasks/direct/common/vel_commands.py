"""Velocity command generator: constant (vx, vy, vz) + sinusoidal yaw rate.

Each env gets independent commands sampled at reset.
Sinusoidal yaw makes the robot carve curved trajectories rather than
straight lines, giving richer coverage of the state space.

    yaw_rate(t) = A * sin(omega * t + phi)

where A, omega, phi are sampled uniformly at reset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class VelCmdCfg:
    # linear velocity ranges [m/s], constant per episode
    vx_range: tuple[float, float] = (-1.0, 1.0)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    vz_range: tuple[float, float] = (0.0, 0.0)   # zero for flat-ground locomotion

    # sinusoidal yaw parameters
    yaw_A_range: tuple[float, float] = (0.3, 1.0)    # amplitude  [rad/s]
    yaw_omega_range: tuple[float, float] = (0.5, 2.0) # frequency  [rad/s]
    # phi sampled from [0, 2π] uniformly


class VelCommands:
    """
    Manages per-environment velocity commands.

    Usage in env:
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)

    In _reset_idx:
        self._cmd.reset(env_ids)

    In _get_observations (or _get_rewards):
        t = self.episode_length_buf.float() * self.step_dt   # seconds
        cmds = self._cmd.get(t)   # (N, 4): [vx, vy, vz, yaw_rate]
    """

    def __init__(self, num_envs: int, device: str, cfg: VelCmdCfg):
        self.cfg = cfg
        self.device = device

        self.vx = torch.zeros(num_envs, device=device)
        self.vy = torch.zeros(num_envs, device=device)
        self.vz = torch.zeros(num_envs, device=device)

        self.yaw_A = torch.zeros(num_envs, device=device)
        self.yaw_omega = torch.zeros(num_envs, device=device)
        self.yaw_phi = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        def _u(lo: float, hi: float) -> torch.Tensor:
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        lo, hi = self.cfg.vx_range
        self.vx[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.vy_range
        self.vy[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.vz_range
        self.vz[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.yaw_A_range
        self.yaw_A[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.yaw_omega_range
        self.yaw_omega[env_ids] = _u(lo, hi)
        self.yaw_phi[env_ids] = _u(0.0, 2.0 * math.pi)

    def get(self, episode_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            episode_time: (N,) seconds elapsed since last reset per env.
        Returns:
            (N, 4) tensor: [vx, vy, vz, yaw_rate]
        """
        yaw_rate = self.yaw_A * torch.sin(self.yaw_omega * episode_time + self.yaw_phi)
        return torch.stack([self.vx, self.vy, self.vz, yaw_rate], dim=-1)

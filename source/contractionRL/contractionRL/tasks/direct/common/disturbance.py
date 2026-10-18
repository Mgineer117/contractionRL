"""Bounded sample-and-hold gusts, following Sun et al. (CoRL 2020), App. E.5.

Duration is uniform up to one second (rounded to at least one control step).
Magnitude is uniform on [0, sigma], direction isotropic. No Gaussian white
noise is added to position coordinates. Value and remaining dwell are internal
simulator state, hidden from both actor and critic.
"""
import torch


class BoundedGust:
    def __init__(self, num_envs, scales, dt, device, max_hold=1.0):
        self.scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        if not torch.isfinite(self.scales).all() or (self.scales < 0).any():
            raise ValueError("Disturbance scales must be finite and nonnegative")
        if not 0 < dt <= max_hold:
            raise ValueError("Require 0 < dt <= disturbance max_hold")
        self.enabled = bool((self.scales > 0).any())
        self.dt, self.max_hold = float(dt), float(max_hold)
        self.value = torch.zeros(num_envs, len(self.scales), device=device)
        self.remaining = torch.zeros(num_envs, device=device)

    def reset(self, ids):
        if not self.enabled:
            self.value[ids] = 0
            self.remaining[ids] = 0
            return
        n = len(ids)
        direction = torch.randn(n, len(self.scales), device=self.value.device)
        direction *= (self.scales > 0)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.value[ids] = direction * torch.rand(n, 1, device=direction.device) * self.scales
        duration = torch.rand(n, device=direction.device) * self.max_hold
        self.remaining[ids] = (duration / self.dt).ceil().clamp(1, int(self.max_hold / self.dt)) * self.dt

    def advance(self):
        if not self.enabled:
            return self.value.clone()
        ids = (self.remaining <= 1e-7).nonzero(as_tuple=True)[0]
        if len(ids):
            self.reset(ids)
        applied = self.value.clone()
        self.remaining.sub_(self.dt).clamp_min_(0)
        return applied


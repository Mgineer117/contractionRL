"""Uniform initial-state boxes with independent sign choices."""
import torch


def sample_uniform_box(low, high, count, sign_dims=()):
    values = low + torch.rand(count, low.numel(), device=low.device, dtype=low.dtype) * (high - low)
    if sign_dims:
        signs = 2 * torch.randint(2, (count, len(sign_dims)), device=low.device) - 1
        values[:, sign_dims] *= signs
    return values

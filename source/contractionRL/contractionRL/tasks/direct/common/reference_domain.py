"""Finite robot tracking domains and validation of nominal PhysX recordings."""
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np


def robot_domain(action_dim, base_height_min=None):
    if action_dim == 7:
        low = np.array([-4.] * 7 + [-3.] * 7 + [-1.5, -1.5, .05] + [-2.] * 3 + [-4.], np.float32)
        high = np.array([4.] * 7 + [3.] * 7 + [1.5, 1.5, 1.5] + [2.] * 3 + [4.], np.float32)
    elif action_dim in (12, 19):
        m = action_dim
        high = np.array([80., 80., np.pi] + [1.] * 3 + [3.2] * m + [3., 3., 2.] + [4.] * 3 + [20.] * m + [3.], np.float32)
        low = -high
        low[-1] = (.2 if m == 12 else .5) if base_height_min is None else base_height_min
    else:
        raise ValueError(f"Unsupported robot action width {action_dim}")
    margin = .05 * (high - low)
    if action_dim != 7:
        margin[2:6] = 0.
        margin[-1] = .03
    return low, high, margin


def save_nominal_references(path, *, x, u, x_dot, dt, lengths=None):
    """Save only full, finite, upright, in-domain reference trajectories.

    Old archives are preserved before replacing the canonical file. A failure
    to produce any safe candidate leaves the existing file untouched.
    """
    x, u, x_dot = [np.asarray(v, dtype=np.float32) for v in (x, u, x_dot)]
    low, high, margin = robot_domain(u.shape[-1])
    if x.shape[-1] != len(low) or x_dot.shape != x.shape or x.shape[:2] != u.shape[:2]:
        raise ValueError("Reference recording has an incompatible robot state/action layout")
    if u.shape[-1] != 7:
        x = x.copy()
        x[:, :, 2] = (x[:, :, 2] + np.pi) % (2 * np.pi) - np.pi
    lengths = np.full(len(x), x.shape[1], dtype=np.int64) if lengths is None else np.asarray(lengths, np.int64)
    keep = (lengths == x.shape[1])
    keep &= np.isfinite(x).all(axis=(1, 2)) & np.isfinite(x_dot).all(axis=(1, 2)) & np.isfinite(u).all(axis=(1, 2))
    keep &= ((x >= low + margin) & (x <= high - margin)).all(axis=(1, 2))
    keep &= (np.abs(u) <= 1. + 1e-6).all(axis=(1, 2))
    if u.shape[-1] != 7:
        keep &= (x[:, :, 5] < -.85).all(axis=1)
    if not keep.any():
        raise ValueError("No safe full-length nominal reference; train the nominal controller further or revise its commands/domain")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
        shutil.copy2(target, target.with_name(target.stem + '.before-' + stamp + '.npz'))
    np.savez_compressed(target, x=x[keep], u=u[keep], x_dot=x_dot[keep], lengths=lengths[keep],
                        dt=float(dt), reference_source='physx_nominal_rollout', action_coordinates='pd_offset_v1')
    return int(keep.sum())

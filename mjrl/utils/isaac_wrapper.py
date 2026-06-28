"""IsaacMjrlWrapper — adapts an Isaac Sim env to the mjrl classic-env interface.

Provides the attributes/methods that Runner and mjrl agents expect:
  * num_dim_x        — state/obs dimension
  * num_dim_control  — action dimension
  * get_f_and_B(x)   — delegates to NeuralDynamics (learned dynamics required)
  * get_rollout(N, 'c3m') — samples (x, xref, uref) uniformly from bounds

The obs is treated as the full 'state' x. xref and uref are sampled uniformly
within the observation and action space bounds respectively.

This wrapper does NOT run any Isaac Sim steps itself — it is a thin adapter
around the already-running Isaac env object for use during agent construction.
"""

from __future__ import annotations

import numpy as np
import torch


class IsaacMjrlWrapper:
    """Thin adapter that gives an Isaac Sim env the mjrl classic-env interface.

    Args:
        isaac_env:       The live Isaac Sim env (ManagerBasedRLEnv or DirectEnv).
        dynamics_model:  A NeuralDynamics instance (required — Isaac envs have
                         no analytical dynamics).
        obs_low/high:    Optional obs bounds override.  If None, use
                         env.observation_space.low/high (or ±10 if not set).
        device:          Torch device for tensor ops.
    """

    def __init__(
        self,
        isaac_env,
        dynamics_model,
        obs_low: np.ndarray | None = None,
        obs_high: np.ndarray | None = None,
        device: str = "cpu",
    ):
        self._env = isaac_env
        self._dynamics = dynamics_model
        self.device = device

        self.num_dim_x = dynamics_model.x_dim
        self.num_dim_control = dynamics_model.u_dim

        # state bounds for C3M sampling
        act = isaac_env.action_space
        self._act_low = np.asarray(act.low, dtype=np.float32).flatten()
        self._act_high = np.asarray(act.high, dtype=np.float32).flatten()

        if obs_low is not None:
            self._obs_low = np.asarray(obs_low, dtype=np.float32).flatten()
            self._obs_high = np.asarray(obs_high, dtype=np.float32).flatten()
        else:
            obs_space = isaac_env.observation_space
            if hasattr(obs_space, "low"):
                self._obs_low = np.asarray(obs_space.low, dtype=np.float32).flatten()
                self._obs_high = np.asarray(obs_space.high, dtype=np.float32).flatten()
            else:
                # fallback: ±10 everywhere (common for proprioceptive obs)
                self._obs_low = np.full(self.num_dim_x, -10.0, dtype=np.float32)
                self._obs_high = np.full(self.num_dim_x, 10.0, dtype=np.float32)

        # clip extreme values (inf / very large) from obs bounds
        self._obs_low = np.clip(self._obs_low, -1e4, 0.0)
        self._obs_high = np.clip(self._obs_high, 0.0, 1e4)

        # hook for env simulation path (sim rollouts go through the real Isaac env,
        # not through this wrapper — this is only for agent training)
        self.use_learned_dynamics = True
        self.learned_dynamics_model = dynamics_model

        # expose policy slot (populated by Runner after agent construction)
        self.policy = None

    # ------------------------------------------------------------------ #
    def get_f_and_B(self, x):
        """Delegate to NeuralDynamics — supports both tensor and numpy x."""
        return self._dynamics.get_f_and_B(x)

    def get_rollout(self, buffer_size: int, mode: str) -> dict:
        """Sample training data uniformly from the obs/action space bounds."""
        if mode == "c3m":
            xref = np.random.uniform(
                self._obs_low, self._obs_high, size=(buffer_size, self.num_dim_x)
            ).astype(np.float32)
            uref = np.random.uniform(
                self._act_low, self._act_high, size=(buffer_size, self.num_dim_control)
            ).astype(np.float32)
            # x = xref + small noise (±5 % of range, clipped to bounds)
            noise_scale = 0.05 * (self._obs_high - self._obs_low)
            noise = np.random.uniform(-noise_scale, noise_scale, size=xref.shape)
            x = np.clip(xref + noise, self._obs_low, self._obs_high).astype(np.float32)
            return {"x": x, "xref": xref, "uref": uref}

        raise ValueError(f"IsaacMjrlWrapper.get_rollout: unsupported mode '{mode}'")

    # ------------------------------------------------------------------ #
    # Passthrough to the underlying Isaac env for anything else
    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        return getattr(self._env, name)

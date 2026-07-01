"""BaseEnv for classic analytical tracking environments (ported from CAC-dev).

These environments are pure gymnasium + numpy/torch (no Isaac Sim). They expose
the interface the mjrl contraction algorithms expect:

  * observation = ``[x, xref, uref]``  (current state, reference state, ref control)
  * ``get_f_and_B(x) -> (f, B, B_null)`` analytical control-affine dynamics
  * ``get_rollout(buffer_size, mode="c3m")`` random (x, xref, uref) triples
  * standard ``reset`` / ``step`` driving a reference trajectory

The elaborate contraction-bound matplotlib overlay from CAC-dev is reduced here
to a lightweight trajectory/error render; the dynamics and sampling logic are
preserved verbatim so algorithm behaviour matches CAC-dev.
"""

from __future__ import annotations

from abc import abstractmethod
from math import ceil

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config: dict):
        super().__init__()

        self.X_MIN = env_config["x_min"]
        self.X_MAX = env_config["x_max"]
        self.XREF_INIT_MIN = env_config["xref_init_min"]
        self.XREF_INIT_MAX = env_config["xref_init_max"]
        self.XE_INIT_MIN = env_config["xe_init_min"]
        self.XE_INIT_MAX = env_config["xe_init_max"]
        self.XE_MIN = env_config["xe_min"]
        self.XE_MAX = env_config["xe_max"]
        self.UREF_MIN = env_config["uref_min"]
        self.UREF_MAX = env_config["uref_max"]

        self.num_dim_x = env_config["num_dim_x"]
        self.num_dim_control = env_config["num_dim_control"]
        self.pos_dimension = env_config["pos_dimension"]
        self.angle_idx = env_config.get("angle_idx", [])

        self.time_bound = env_config["time_bound"]
        self.dt = env_config["dt"]
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.tracking_scaler = env_config["q"]
        self.control_scaler = env_config["r"]

        self.use_learned_dynamics = False
        self.sample_mode = env_config.get("sample_mode", "uniform")
        self.reward_mode = env_config.get("reward_mode", "default")

        ref_unit_min = np.concatenate((self.X_MIN.flatten(), self.UREF_MIN.flatten()))
        ref_unit_max = np.concatenate((self.X_MAX.flatten(), self.UREF_MAX.flatten()))
        self.STATE_MIN = np.concatenate((self.X_MIN.flatten(), ref_unit_min))
        self.STATE_MAX = np.concatenate((self.X_MAX.flatten(), ref_unit_max))

        self.observation_space = spaces.Box(
            low=self.STATE_MIN.flatten().astype(np.float32),
            high=self.STATE_MAX.flatten().astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self.UREF_MIN.flatten().astype(np.float32),
            high=self.UREF_MAX.flatten().astype(np.float32),
            dtype=np.float32,
        )

        self.reset()

    # ------------------------------------------------------------------ #
    def get_horizon_matched_gamma(self, scale: float = 1.0):
        scale = max(1e-3, min(scale, 1.0))
        return round(1 - (1 / (scale * self.max_episode_len)), 3)

    def reset(self, seed=None, options: dict | None = None):
        super().reset(seed=seed)
        self.time_steps = 0
        if options is None:
            self.x_t, self.xref, self.uref, self.episode_len = self.system_reset()
            self.xref = np.clip(self.xref, self.X_MIN.flatten(), self.X_MAX.flatten())
        else:
            assert hasattr(self, "xref") and hasattr(self, "uref")
            if options.get("replace_x_0", True):
                _, xe_0, _ = self.define_initial_state()
                self.x_t = self.xref[0] + xe_0
            else:
                raise NotImplementedError("Only replace_x_0 is implemented.")

        self.x_0 = self.x_t.copy()
        state = self.construct_state(self.x_t)
        self.init_tracking_error = np.linalg.norm(self.x_t - self.xref[0], ord=2) ** 2
        self.traj_x, self.traj_y, self.err_history = [], [], []
        return state, {"x": self.x_t, "tracking_error": self.init_tracking_error}

    def step(self, u):
        self.time_steps += 1
        u = self.uref[self.time_steps] + u
        self.current_u = u.copy()
        reward, infos = self.get_rewards(u)
        next_x, next_x_wrapped, termination, truncation, _ = self.get_transition(self.x_t, u)
        next_x_wrapped = np.clip(next_x_wrapped, self.X_MIN.flatten(), self.X_MAX.flatten())
        state = self.construct_state(next_x_wrapped)
        self.x_t = next_x
        return (
            state,
            reward,
            termination,
            truncation,
            {
                "x": next_x_wrapped,
                "tracking_error": infos["tracking_error"],
                "control_effort": infos["control_effort"],
                "relative_tracking_error": infos["tracking_error"] / self.init_tracking_error,
            },
        )

    def get_transition(self, x: np.ndarray, u: np.ndarray):
        x_dot = self.get_dynamics(x, u)
        next_x = x + self.dt * x_dot
        pos_min = self.X_MIN.flatten()[: self.pos_dimension]
        pos_max = self.X_MAX.flatten()[: self.pos_dimension]
        next_pos = next_x[: self.pos_dimension]
        if np.any(next_pos < pos_min) or np.any(next_pos > pos_max):
            next_x[: self.pos_dimension] = x[: self.pos_dimension]
        next_x_wrapped = self.wrap_angles(next_x)
        termination = False
        truncation = self.time_steps == self.episode_len - 1
        return next_x, next_x_wrapped, termination, truncation, x_dot

    def get_dynamics(self, x: np.ndarray, u: np.ndarray):
        f_x, B_x, _ = self.get_f_and_B(x)
        if np.any(np.isnan(u)):
            print("[Warning]: NaN values found in control input u.")
            u = np.nan_to_num(u)
        return f_x + np.matmul(B_x, u[..., np.newaxis]).squeeze()

    def get_f_and_B(self, x: torch.Tensor | np.ndarray):
        if self.use_learned_dynamics:
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(self.wrap_angles(x))
            return (
                f_x.cpu().squeeze(0).numpy(),
                B_x.cpu().squeeze(0).numpy(),
                Bbot_x.cpu().squeeze(0).numpy(),
            )
        return self.f_func(x), self.B_func(x), self.B_null(x)

    def wrap_angles(self, x: np.ndarray):
        x_copy = x.copy()
        for idx in getattr(self, "angle_idx", []):
            x_copy[idx] = (x_copy[idx] + np.pi) % (2 * np.pi) - np.pi
        return x_copy

    def construct_state(self, x: np.ndarray):
        idx = min(self.time_steps, self.episode_len - 1)
        xref = self.xref[idx]
        uref = self.uref[idx]
        return np.concatenate([x, xref.flatten(), uref.flatten()]).astype(np.float32)

    @property
    def x_dim(self) -> int:
        return self.num_dim_x

    @property
    def u_dim(self) -> int:
        return self.num_dim_control

    # ------------------------------------------------------------------ #
    # control-affine dynamics: f(x), B(x), B_null(x) (torch/numpy dual)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _f_logic(self, x, lib):
        ...

    @abstractmethod
    def _B_logic(self, x, lib):
        ...

    def _B_null_logic(self, x, n, lib):
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)
        if lib == torch:
            Bbot = torch.cat((torch.eye(eye_dims), torch.zeros(zero_dims)), dim=0)
            return Bbot.repeat(n, 1, 1)
        Bbot = np.concatenate((np.eye(eye_dims), np.zeros(zero_dims)), axis=0)
        return np.repeat(Bbot[np.newaxis, :, :], n, axis=0)

    def f_func(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            lib = np
            if x.ndim == 1:
                x = x[np.newaxis, :]
        result = self._f_logic(x, lib)
        try:
            return result.squeeze(0)
        except Exception:
            return result

    def B_func(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            lib = np
            if x.ndim == 1:
                x = x[np.newaxis, :]
        result = self._B_logic(x, lib)
        try:
            return result.squeeze(0)
        except Exception:
            return result

    def B_null(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            n = 1 if x.dim() == 1 else x.shape[0]
        else:
            lib = np
            n = 1 if x.ndim == 1 else x.shape[0]
        result = self._B_null_logic(x, n, lib)
        try:
            return result.squeeze(0)
        except Exception:
            return result

    def define_initial_state(self):
        xref_0 = self.XREF_INIT_MIN + np.random.rand(len(self.XREF_INIT_MIN)) * (
            self.XREF_INIT_MAX - self.XREF_INIT_MIN
        )
        xe_0 = self.XE_INIT_MIN + np.random.rand(len(self.XE_INIT_MIN)) * (
            self.XE_INIT_MAX - self.XE_INIT_MIN
        )
        return xref_0, xe_0, xref_0 + xe_0

    @abstractmethod
    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise):
        ...

    @abstractmethod
    def system_reset(self):
        ...

    # ------------------------------------------------------------------ #
    def get_rewards(self, u):
        error = self.wrap_angles(self.x_t - self.xref[self.time_steps])
        tracking_error = np.linalg.norm(error, ord=2) ** 2
        control_effort = np.linalg.norm(u, ord=2) ** 2

        tracking_reward = -self.tracking_scaler * tracking_error
        control_reward = -self.control_scaler * control_effort
        if self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + abs(tracking_reward))
            control_reward = 1 / (1 + abs(control_reward))
        reward = (0.5 * tracking_reward) + (0.5 * control_reward)
        return reward, {"tracking_error": tracking_error, "control_effort": control_effort}

    def get_rollout(self, buffer_size: int, mode: str):
        """Sample random (x, xref, uref) triples for C3M-style synthesis."""
        if mode == "c3m":
            xref = (self.X_MAX - self.X_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.X_MIN.flatten()
            uref = (self.UREF_MAX - self.UREF_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_control
            ) + self.UREF_MIN.flatten()
            xe = (self.XE_MAX - self.XE_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.XE_MIN.flatten()
            x = np.clip(xe + xref, self.X_MIN.flatten(), self.X_MAX.flatten())
            return {
                "x": x.astype(np.float32),
                "xref": xref.astype(np.float32),
                "uref": uref.astype(np.float32),
            }

        # dynamics-learning rollout (uniform/gaussian random control-affine samples)
        n_control_per_x = 3
        batch_size = ceil(buffer_size / n_control_per_x)
        if self.sample_mode == "Gaussian":
            x_mean = (self.X_MAX.flatten() + self.X_MIN.flatten()) / 2.0
            x_std = (self.X_MAX.flatten() - self.X_MIN.flatten()) / 6.0
            u_mean = (self.UREF_MAX.flatten() + self.UREF_MIN.flatten()) / 2.0
            u_std = (self.UREF_MAX.flatten() - self.UREF_MIN.flatten()) / 6.0
            x = np.random.normal(x_mean, x_std, size=(batch_size, len(x_mean)))
            u = np.random.normal(u_mean, u_std, size=(batch_size, len(u_mean)))
        else:
            x = np.random.uniform(self.X_MIN.flatten(), self.X_MAX.flatten(), size=(batch_size, self.num_dim_x))
            u = np.random.uniform(self.UREF_MIN.flatten(), self.UREF_MAX.flatten(), size=(batch_size, self.num_dim_control))
        x = np.concatenate([x] * n_control_per_x, axis=0)
        u = np.concatenate([u[np.random.permutation(len(u))] for _ in range(n_control_per_x)], axis=0)
        x_dot = self.get_dynamics(x, u)
        return {
            "x": x[:buffer_size].astype(np.float32),
            "u": u[:buffer_size].astype(np.float32),
            "x_dot": x_dot[:buffer_size].astype(np.float32),
        }

    # ------------------------------------------------------------------ #
    def render(self, mode="rgb_array"):
        """Lightweight trajectory + tracking-error render."""
        import matplotlib.pyplot as plt

        if not hasattr(self, "fig"):
            if mode == "rgb_array":
                plt.switch_backend("Agg")
            elif mode == "human":
                plt.ion()
            self.fig, (self.ax, self.ax_err) = plt.subplots(1, 2, figsize=(12, 5))

        self.ax.clear(); self.ax_err.clear()
        pos = self.x_t[: self.pos_dimension]
        p_x = pos[0]
        p_y = pos[1] if self.pos_dimension > 1 else 0.0
        self.traj_x.append(p_x); self.traj_y.append(p_y)

        if hasattr(self, "xref"):
            ref_x = self.xref[:, 0]
            ref_y = self.xref[:, 1] if self.pos_dimension > 1 else np.zeros_like(ref_x)
            self.ax.plot(ref_x, ref_y, "k:", label="Reference")
            e_t = self.wrap_angles(self.x_t - self.xref[self.time_steps])
            self.err_history.append(np.linalg.norm(e_t))

        self.ax.plot(self.traj_x, self.traj_y, "b-", label="Trajectory")
        self.ax.scatter(p_x, p_y, color="r", s=40)
        self.ax.legend(loc="upper right")
        self.ax.set_title(f"Time Step: {self.time_steps}")

        if self.err_history:
            times = np.arange(len(self.err_history)) * self.dt
            self.ax_err.plot(times, self.err_history, "b-", label="Tracking Error")
            self.ax_err.set_xlabel("Time (s)"); self.ax_err.set_ylabel("Error Norm")
            self.ax_err.set_yscale("log"); self.ax_err.grid(True, alpha=0.3)
            self.ax_err.legend(loc="upper right")

        if mode == "human":
            plt.draw(); plt.pause(0.001)
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.asarray(self.fig.canvas.buffer_rgba())
            return img[:, :, :3].copy()

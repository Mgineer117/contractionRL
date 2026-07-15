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

from contractionRL.tasks.direct.common.state_guard import carry_forward_nonfinite


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
    @staticmethod
    def _build_cfg(
        env_config: dict,
        *,
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
    ) -> dict:
        """Merge each subclass's constructor overrides into its ``ENV_CONFIG``.

        Shared by every classic env's ``__init__`` (car/cartpole/quadrotor/
        segway/turtlebot) so the sample_mode/time_bound/dt
        override logic lives in exactly one place.
        """
        cfg = dict(env_config)
        cfg["sample_mode"] = sample_mode
        if time_bound is not None:
            cfg["time_bound"] = time_bound
        if dt is not None:
            cfg["dt"] = dt
        return cfg

    def _rollout_reference(self, xref_0: np.ndarray, freqs, weights) -> tuple[np.ndarray, np.ndarray, int]:
        """Drive the reference trajectory forward via ``sample_reference_controls``.

        Shared tail of every classic env's ``system_reset``: step the
        reference state with ``get_transition``, carry forward the
        wrapped+clipped state (same fix as ``step``'s ``self.x_t`` — using the
        raw state would let non-angle/non-position dims drift unbounded
        across iterations before ``reset()``'s final ``np.clip``), and stop
        early on termination/truncation.

        Returns ``(xref_wrapped_array, uref_array, num_steps)`` — subclasses
        combine this with their own ``define_initial_state()``-provided
        ``x_0`` to build ``system_reset``'s ``(x_0, xref, uref, episode_len)``.
        """
        xref_list, xref_wrapped_list, uref_list = [xref_0], [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(freqs, weights, _t, {"xref_0": xref_0})
            xref_t, xref_wrapped_t, term, trunc, _ = self.get_transition(xref_list[-1].copy(), uref_t)
            xref_wrapped_t = np.clip(xref_wrapped_t, self.X_MIN.flatten(), self.X_MAX.flatten())
            xref_list.append(xref_wrapped_t)
            xref_wrapped_list.append(xref_wrapped_t)
            uref_list.append(uref_t)
            if term or trunc:
                break
        return np.array(xref_wrapped_list), np.array(uref_list), i + 1

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
        self.episode_reward = 0.0
        
        # Initialize M for the starting state (if CMG is injected via set_ccm)
        if getattr(self, "ccm_gen", None) is not None:
            import torch
            from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse
            with torch.no_grad():
                x_t_tensor = torch.tensor(self.x_0, dtype=torch.float32, device=self.ccm_device).unsqueeze(0)
                W_raw, _ = self.ccm_gen(x_t_tensor)
                W = bound_W(W_raw, self.w_lb, self.num_dim_x, getattr(self.ccm_gen, "bounded", False))
                self.M = spd_inverse(W)
                
        return state, {"x": self.x_t, "tracking_error": self.init_tracking_error}

    def step(self, u):
        self.time_steps += 1
        # The agent emits the *full* control: CLActor returns uref + feedback,
        # LQR/SD-LQR return uref - K·e, and an MLP policy learns the full u
        # (uref is part of the observation). Apply it directly — do NOT re-add
        # uref here. This (a) keeps the executed control identical to the action
        # the policy sampled, so PPO's stored log π(a|s) is the log-prob of what
        # actually runs, and (b) matches the Isaac envs, which map actions to
        # actuator targets without re-adding uref. Re-adding it would double the
        # feedforward term (u = uref + (uref + fb)) and corrupt both tracking and
        # the contraction certificate.
        self.current_u = u.copy()

        next_x, next_x_wrapped, termination, truncation, _ = self.get_transition(self.x_t, u)
        next_x_wrapped = np.clip(next_x_wrapped, self.X_MIN.flatten(), self.X_MAX.flatten())

        # Post-transition reward: pair the state the action actually produced
        # (next_x_wrapped) with self.xref[self.time_steps] (already the
        # post-transition reference, since time_steps was incremented above).
        # Using the PRE-transition self.x_t here (the old behavior) paired a
        # stale state against the NEW reference — an inconsistent, off-by-one
        # tracking error that also gave PPO/SAC's native reward the same
        # broken actor->reward gradient the Mahalanobis reward had at low
        # gamma (see contractionRL "low-gamma CMG" / next_obs discussion).
        reward, infos = self.get_rewards(self.x_t, u, next_x_wrapped)
        self.episode_reward += float(reward)

        # Track raw distance error for AUC/Contraction envelope
        err_dist = np.sqrt(max(infos["tracking_error"], 0.0))
        self.err_history.append(err_dist)

        state = self.construct_state(next_x_wrapped)
        # Carry forward the WRAPPED+CLIPPED state, not the raw `next_x` — the
        # observation the agent sees is built from next_x_wrapped, so self.x_t
        # (which drives the NEXT get_dynamics()/get_rewards() call) must match
        # it exactly. Using raw next_x here let non-angle, non-position state
        # dims (e.g. velocity) drift silently outside [X_MIN, X_MAX] — bounded
        # in what the agent observes, unbounded in what actually evolves.
        self.x_t = next_x_wrapped
        
        info_dict = {
            "x": next_x_wrapped,
            "tracking_error": infos["tracking_error"],
            "control_effort": infos["control_effort"],
            "relative_tracking_error": infos["tracking_error"] / self.init_tracking_error,
        }
        
        if termination or truncation:
            # Compute stability metrics for the completed episode
            e0 = max(self.err_history[0], 1e-8) if self.err_history else 1.0
            norm_traj = np.asarray(self.err_history) / e0
            
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            auc = float(_trapz(norm_traj, dx=self.dt))
            
            # Populate SKRL's expected info["log"] dictionary. Bare
            # "Reward/total_reward" (no _mean/_ci95 suffix, same convention as
            # Stability/auc) — WandbPlotWrapper aggregates it across envs into
            # "Reward/total_reward_mean"/"_ci95" at the SAME per-episode-reset
            # cadence as Stability/*, instead of that key only ever being
            # written once by the post-training evaluator.
            info_dict["log"] = {
                "Stability/auc": auc,
                "Reward/total_reward": self.episode_reward,
            }
            
            try:
                import torch
                from contractionRL.agents.skrl.contraction_metrics import per_env_metrics
                
                err_arr = np.asarray(self.err_history)
                # Create batched (size 1) tensors for the unified metric function
                t_e0 = torch.tensor([err_arr[0] if len(err_arr) > 0 else 1.0], dtype=torch.float32)
                t_e_last = torch.tensor([err_arr[-1] if len(err_arr) > 0 else 1.0], dtype=torch.float32)
                t_e_max = torch.tensor([np.max(err_arr) if len(err_arr) > 0 else 1.0], dtype=torch.float32)
                t_err_sum = torch.tensor([np.sum(err_arr)], dtype=torch.float32)
                t_steps = torch.tensor([len(err_arr)], dtype=torch.float32)
                
                metrics = per_env_metrics(
                    e0=t_e0, e_last=t_e_last, e_max=t_e_max,
                    err_sum=t_err_sum, steps=t_steps, dt=self.dt
                )
                
                info_dict["log"]["Stability/overshoot"] = metrics["overshoot"][0].item()
                info_dict["log"]["Stability/contraction_rate"] = metrics["contraction_rate"][0].item()
                info_dict["log"]["Stability/contraction_score"] = metrics["contraction_score"][0].item()
            except Exception:
                pass  # Fallback to just AUC if computation fails
                
        return (
            state,
            reward,
            termination,
            truncation,
            info_dict,
        )

    def get_transition(self, x: np.ndarray, u: np.ndarray):
        x_dot = self.get_dynamics(x, u)
        next_x = x + self.dt * x_dot
        # Divergence guard: a poor policy can drive x_dot to NaN/Inf (e.g. a
        # control-affine term blowing up). Carry the previous state forward
        # element-wise rather than terminating — episodes never terminate here
        # (see `termination = False` below), and np.clip(nan, ...) would leave
        # NaN untouched, silently poisoning every downstream reward/obs.
        next_x = carry_forward_nonfinite(next_x, x)
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
    @staticmethod
    def _zeros(shape, x, lib):
        """``lib.zeros`` allocated on ``x``'s device when ``lib`` is torch.

        ``torch.zeros(shape)`` defaults to CPU regardless of where ``x`` lives,
        so a subsequent ``f[:, i] = x[:, j] * ...`` silently mixes devices and
        raises once ``x`` is a CUDA tensor (e.g. C3M/C2RL running their
        contraction math on GPU). The numpy path (env.step's physics rollout)
        is unaffected since ``np.zeros`` has no device concept.
        """
        if lib is torch:
            return lib.zeros(shape, device=x.device, dtype=x.dtype)
        return lib.zeros(shape)

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
            Bbot = torch.cat(
                (torch.eye(eye_dims, device=x.device, dtype=x.dtype),
                 torch.zeros(zero_dims, device=x.device, dtype=x.dtype)),
                dim=0,
            )
            return Bbot.repeat(n, 1, 1)
        Bbot = np.concatenate((np.eye(eye_dims), np.zeros(zero_dims)), axis=0)
        return np.repeat(Bbot[np.newaxis, :, :], n, axis=0)

    # NOTE: each of f_func/B_func/B_null squeezes the batch dim back out ONLY
    # when it added one for a 1-D input (``added_batch``). Squeezing
    # unconditionally (the old behavior) also collapsed a genuine batch of size
    # 1: a (1, x_dim) input returned a (x_dim,) result, so a downstream
    # ``jacobian(f, x)`` — which indexes ``f[:, i]`` — crashed with "too many
    # indices for tensor of dimension 1". That's exactly the single-env eval
    # path for SD-LQR/LQR (batch=1), which never showed up in training (num_envs
    # > 1). The 1-D physics path in get_dynamics/env.step still gets its 1-D
    # result back, so both callers are satisfied.
    def f_func(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            added_batch = x.dim() == 1
            if added_batch:
                x = x.unsqueeze(0)
        else:
            lib = np
            added_batch = x.ndim == 1
            if added_batch:
                x = x[np.newaxis, :]
        result = self._f_logic(x, lib)
        if added_batch:
            try:
                return result.squeeze(0)
            except Exception:
                return result
        return result

    def B_func(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            added_batch = x.dim() == 1
            if added_batch:
                x = x.unsqueeze(0)
        else:
            lib = np
            added_batch = x.ndim == 1
            if added_batch:
                x = x[np.newaxis, :]
        result = self._B_logic(x, lib)
        if added_batch:
            try:
                return result.squeeze(0)
            except Exception:
                return result
        return result

    def B_null(self, x):
        if isinstance(x, torch.Tensor):
            lib = torch
            added_batch = x.dim() == 1
            if added_batch:
                x = x.unsqueeze(0)
        else:
            lib = np
            added_batch = x.ndim == 1
            if added_batch:
                x = x[np.newaxis, :]
        n = 1 if added_batch else x.shape[0]
        result = self._B_null_logic(x, n, lib)
        if added_batch:
            try:
                return result.squeeze(0)
            except Exception:
                return result
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

    def set_ccm(self, ccm_gen, w_lb, device):
        """Inject the frozen CCM for Mahalanobis reward computation."""
        self.ccm_gen = ccm_gen
        self.w_lb = w_lb
        self.ccm_device = device

    # ------------------------------------------------------------------ #
    def get_rewards(self, x, u, next_x):
        """Tracking + control-effort reward.
        Returns the telescoping difference: V(x) - V(next_x).
        """
        error = self.wrap_angles(x - self.xref[self.time_steps - 1])
        next_error = self.wrap_angles(next_x - self.xref[self.time_steps])
        
        tracking_error = np.linalg.norm(next_error, ord=2) ** 2
        control_effort = np.linalg.norm(u, ord=2) ** 2

        if getattr(self, "ccm_gen", None) is not None:
            import torch
            from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse
            with torch.no_grad():
                to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=self.ccm_device).unsqueeze(0)
                
                # Simply read the cached M from reset() or the previous step
                M = self.M
                
                # Evaluate metric at next_x
                next_W_raw, _ = self.ccm_gen(to_t(next_x))
                next_W = bound_W(next_W_raw, self.w_lb, self.num_dim_x, getattr(self.ccm_gen, "bounded", False))
                next_M = spd_inverse(next_W)
                
                # Cache next_M for the next step unconditionally
                self.M = next_M
                
                # Convert errors to tensor
                err_t = to_t(error).unsqueeze(-1)
                next_err_t = to_t(next_error).unsqueeze(-1)
                
                # Calculate quad forms: e^T M e
                quad = (err_t.transpose(1, 2) @ M @ err_t).squeeze()
                next_quad = (next_err_t.transpose(1, 2) @ next_M @ next_err_t).squeeze()
                
                tracking_reward = self.tracking_scaler * (quad - next_quad).item()
                control_reward = -self.control_scaler * control_effort
                
                # Reward is V(x) - V(next_x) (positive if error decreases) + control penalty
                reward = tracking_reward + control_reward
        else:
            # Reward is V(x) - V(next_x)
            reward = np.dot(error, error) - np.dot(next_error, next_error)
            
        return reward, {"tracking_error": tracking_error, "control_effort": control_effort}

    def get_rollout(self, buffer_size: int, mode: str, num_control_per_state: int | None = None):
        """Sample random (x, xref, uref) triples for C3M-style synthesis.

        ``num_control_per_state`` (``mode="dynamics"`` only) — how many distinct
        control vectors get paired with each sampled state (C3M/C2RL's
        ``num_controls_per_state``); defaults to 3 (the old hardcoded value) when
        not given, e.g. by a caller unaware of the config knob.
        """
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
        n_control_per_x = num_control_per_state if num_control_per_state is not None else 3
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
    # SyncVectorEnv broadcasts render() to every parallel sub-env instance
    # (skrl's GymnasiumWrapper.render() -> VectorEnv.call("render", ...)), but
    # we only ever want to plot one representative episode. The first instance
    # to have render() invoked claims ownership (a class attr, so it's shared
    # across all sub-envs of this task); every other instance no-ops instead
    # of allocating its own pyplot figure. Without this, num_envs figures pile
    # up unclosed and trip matplotlib's >20-open-figures warning/leak.
    _render_owner_id: int | None = None

    def render(self, mode="rgb_array"):
        """Lightweight trajectory + tracking-error render (first env only)."""
        import matplotlib.pyplot as plt

        cls = type(self)
        if cls._render_owner_id is None:
            cls._render_owner_id = id(self)
        if id(self) != cls._render_owner_id:
            return None

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

    def close(self):
        # Only the render-owning instance ever allocates self.fig; the other
        # (num_envs - 1) sub-envs no-op here. Runs for every algorithm since
        # SyncVectorEnv.close() -> each sub-env's close() at the end of train/play.
        if hasattr(self, "fig"):
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            type(self)._render_owner_id = None
        super().close()

"""Batched Torch BaseEnv for classic analytical tracking environments."""

from __future__ import annotations

from abc import abstractmethod
import math
import numpy as np
import torch
import gymnasium as gym

from contractionRL.tasks.direct.common.state_guard import carry_forward_nonfinite

class BaseEnv(gym.Env):
    def __init__(self, env_config: dict, num_envs: int = 1, device: str = "cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.X_MIN = torch.tensor(env_config["x_min"], device=self.device, dtype=torch.float32).flatten()
        self.X_MAX = torch.tensor(env_config["x_max"], device=self.device, dtype=torch.float32).flatten()
        self.XREF_INIT_MIN = torch.tensor(env_config["xref_init_min"], device=self.device, dtype=torch.float32).flatten()
        self.XREF_INIT_MAX = torch.tensor(env_config["xref_init_max"], device=self.device, dtype=torch.float32).flatten()
        self.XE_INIT_MIN = torch.tensor(env_config["xe_init_min"], device=self.device, dtype=torch.float32).flatten()
        self.XE_INIT_MAX = torch.tensor(env_config["xe_init_max"], device=self.device, dtype=torch.float32).flatten()
        self.XE_MIN = torch.tensor(env_config["xe_min"], device=self.device, dtype=torch.float32).flatten()
        self.XE_MAX = torch.tensor(env_config["xe_max"], device=self.device, dtype=torch.float32).flatten()
        self.UREF_MIN = torch.tensor(env_config["uref_min"], device=self.device, dtype=torch.float32).flatten()
        self.UREF_MAX = torch.tensor(env_config["uref_max"], device=self.device, dtype=torch.float32).flatten()

        self.num_dim_x = env_config["num_dim_x"]
        self.num_dim_control = env_config["num_dim_control"]
        self.pos_dimension = env_config["pos_dimension"]
        self.angle_idx = env_config.get("angle_idx", [])

        self.time_bound = env_config["time_bound"]
        self.dt = env_config["dt"]
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = self.max_episode_len
        self.t = torch.arange(0, self.time_bound, self.dt, device=self.device, dtype=torch.float32)

        self.tracking_scaler = env_config["q"]
        self.control_scaler = env_config["r"]
        self.use_learned_dynamics = False

        ref_unit_min = torch.cat([self.X_MIN, self.UREF_MIN])
        ref_unit_max = torch.cat([self.X_MAX, self.UREF_MAX])
        self.STATE_MIN = torch.cat([self.X_MIN, ref_unit_min])
        self.STATE_MAX = torch.cat([self.X_MAX, ref_unit_max])

        # Mimic standard spaces for compatibility
        self.observation_space = gym.spaces.Box(
            low=self.STATE_MIN.cpu().numpy(),
            high=self.STATE_MAX.cpu().numpy(),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=self.UREF_MIN.cpu().numpy(),
            high=self.UREF_MAX.cpu().numpy(),
            dtype=np.float32,
        )

        # Buffers
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.x_t = torch.zeros(self.num_envs, self.num_dim_x, dtype=torch.float32, device=self.device)
        self.xref = torch.zeros(self.num_envs, self.max_episode_len, self.num_dim_x, dtype=torch.float32, device=self.device)
        self.uref = torch.zeros(self.num_envs, self.max_episode_len, self.num_dim_control, dtype=torch.float32, device=self.device)
        self.init_tracking_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.reset()

    @staticmethod
    def _build_cfg(env_config: dict, *, sample_mode: str = "uniform", time_bound: float | None = None, dt: float | None = None) -> dict:
        cfg = dict(env_config)
        cfg["sample_mode"] = sample_mode
        if time_bound is not None: cfg["time_bound"] = time_bound
        if dt is not None: cfg["dt"] = dt
        return cfg

    def get_horizon_matched_gamma(self, scale: float = 1.0):
        scale = max(1e-3, min(scale, 1.0))
        return round(1 - (1 / (scale * self.max_episode_len)), 3)

    def define_initial_state(self, env_ids: torch.Tensor):
        n = len(env_ids)
        rand_xref = torch.rand(n, self.num_dim_x, device=self.device, dtype=torch.float32)
        xref_0 = self.XREF_INIT_MIN + rand_xref * (self.XREF_INIT_MAX - self.XREF_INIT_MIN)
        
        rand_xe = torch.rand(n, self.num_dim_x, device=self.device, dtype=torch.float32)
        xe_0 = self.XE_INIT_MIN + rand_xe * (self.XE_INIT_MAX - self.XE_INIT_MIN)
        
        return xref_0, xe_0, xref_0 + xe_0

    @abstractmethod
    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        ...

    def _rollout_reference(self, xref_0: torch.Tensor, freqs, weights) -> tuple[torch.Tensor, torch.Tensor, int]:
        n = xref_0.shape[0]
        xref_list = [xref_0]
        xref_wrapped_list = [xref_0]
        uref_list = []
        
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(freqs, weights, _t, {"xref_0": xref_0})
            xref_prev = xref_list[-1]
            f_x, B_x, _ = self.get_f_and_B(xref_prev)
            x_dot = f_x + torch.bmm(B_x, uref_t.unsqueeze(-1)).squeeze(-1)
            next_x = xref_prev + self.dt * x_dot
            
            next_x_wrapped = self.wrap_angles(next_x)
            next_x_wrapped = torch.clamp(next_x_wrapped, self.X_MIN, self.X_MAX)
            
            xref_list.append(next_x_wrapped)
            xref_wrapped_list.append(next_x_wrapped)
            uref_list.append(uref_t)
            
        return torch.stack(xref_wrapped_list[:-1], dim=1), torch.stack(uref_list, dim=1), i + 1

    @abstractmethod
    def system_reset(self, env_ids: torch.Tensor):
        ...

    def set_ccm(self, ccm_gen, w_lb, device):
        self.ccm_gen = ccm_gen
        self.w_lb = w_lb
        self.ccm_device = device

    def get_rewards(self, x, u, next_x, env_ids):
        t_idx = self.time_steps[env_ids]
        xref_prev = self.xref[env_ids, torch.clamp(t_idx - 1, min=0)]
        xref_curr = self.xref[env_ids, torch.clamp(t_idx, max=self.max_episode_len - 1)]
        
        error = self.wrap_angles(x - xref_prev)
        next_error = self.wrap_angles(next_x - xref_curr)
        
        tracking_error = torch.norm(next_error, p=2, dim=-1) ** 2
        control_effort = torch.norm(u, p=2, dim=-1) ** 2

        if getattr(self, "ccm_gen", None) is not None:
            from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse
            with torch.no_grad():
                if not hasattr(self, "M"):
                    self.M = torch.zeros(self.num_envs, self.num_dim_x, self.num_dim_x, device=self.device)
                M = self.M[env_ids]
                next_W_raw, _ = self.ccm_gen(next_x)
                next_W = bound_W(next_W_raw, self.w_lb, self.num_dim_x, getattr(self.ccm_gen, "bounded", False))
                next_M = spd_inverse(next_W)
                self.M[env_ids] = next_M
                
                err_t = error.unsqueeze(-1)
                next_err_t = next_error.unsqueeze(-1)
                
                V = torch.bmm(torch.bmm(err_t.transpose(1, 2), M), err_t).squeeze(-1).squeeze(-1)
                next_V = torch.bmm(torch.bmm(next_err_t.transpose(1, 2), next_M), next_err_t).squeeze(-1).squeeze(-1)
                
                reward = self.tracking_scaler * (V - next_V) - self.control_scaler * control_effort
        else:
            reward = -self.tracking_scaler * tracking_error - self.control_scaler * control_effort

        infos = {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }
        return reward, infos

    def reset(self, seed=None, options=None):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.construct_state(self.x_t), {"x": self.x_t.clone(), "tracking_error": self.init_tracking_error.clone()}

    def reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
            
        self.time_steps[env_ids] = 0
        self.episode_reward[env_ids] = 0.0
        
        x_0, xref_arr, uref_arr, _ = self.system_reset(env_ids)
        self.xref[env_ids] = torch.clamp(xref_arr, self.X_MIN, self.X_MAX)
        self.uref[env_ids] = uref_arr
        self.x_t[env_ids] = x_0
        
        self.init_tracking_error[env_ids] = torch.norm(x_0 - self.xref[env_ids, 0], p=2, dim=-1) ** 2
        
        if getattr(self, "ccm_gen", None) is not None:
            if not hasattr(self, "M"):
                self.M = torch.zeros(self.num_envs, self.num_dim_x, self.num_dim_x, device=self.device)
            from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse
            with torch.no_grad():
                W_raw, _ = self.ccm_gen(x_0)
                W = bound_W(W_raw, self.w_lb, self.num_dim_x, getattr(self.ccm_gen, "bounded", False))
                self.M[env_ids] = spd_inverse(W)

    def step(self, u: torch.Tensor):
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
        u = torch.nan_to_num(u)
        self.time_steps += 1
        
        f_x, B_x, _ = self.get_f_and_B(self.x_t)
        x_dot = f_x + torch.bmm(B_x, u.unsqueeze(-1)).squeeze(-1)
        next_x = self.x_t + self.dt * x_dot
        
        next_x = carry_forward_nonfinite(next_x, self.x_t)
        
        pos_min = self.X_MIN[:self.pos_dimension]
        pos_max = self.X_MAX[:self.pos_dimension]
        out_of_bounds = (next_x[:, :self.pos_dimension] < pos_min) | (next_x[:, :self.pos_dimension] > pos_max)
        invalid_mask = out_of_bounds.any(dim=-1)
        next_x[invalid_mask, :self.pos_dimension] = self.x_t[invalid_mask, :self.pos_dimension]
        
        next_x_wrapped = self.wrap_angles(next_x)
        next_x_wrapped = torch.clamp(next_x_wrapped, self.X_MIN, self.X_MAX)
        
        reward, infos = self.get_rewards(self.x_t, u, next_x_wrapped, torch.arange(self.num_envs, device=self.device))
        self.episode_reward += reward

        self.x_t = next_x_wrapped
        state = self.construct_state(self.x_t)
        
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncation = self.time_steps >= self.episode_len
        dones = termination | truncation
        
        info_dict = {
            "x": self.x_t.clone(),
            "tracking_error": infos["tracking_error"],
            "control_effort": infos["control_effort"],
            "relative_tracking_error": infos["tracking_error"] / torch.clamp(self.init_tracking_error, min=1e-8),
        }
        
        if dones.any():
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)

            # Stability/* is computed centrally by StatManagerEnvWrapper (the
            # paper-style batched C/lambda metrics) — the env only reports the
            # episodic reward. Must be a scalar torch tensor: skrl's trainer
            # (`environment_info: log`) silently drops plain Python floats.
            info_dict["log"] = {
                "Reward/total_reward_mean": self.episode_reward[done_idx].mean().detach().clone(),
            }

            info_dict["final_observation"] = state[done_idx].clone()
            info_dict["_final_observation"] = dones.clone()
            
            self.reset_idx(done_idx)
            # Reconstruct state after reset for done envs (IsaacLab standard)
            state = self.construct_state(self.x_t)
            
        return state, reward, termination, truncation, info_dict

    def get_f_and_B(self, x: torch.Tensor):
        if getattr(self, "use_learned_dynamics", False):
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(self.wrap_angles(x))
            return f_x, B_x, Bbot_x
        
        # fallback for envs that don't implement _B_null_logic
        if hasattr(self, "_B_null_logic"):
            return self._f_logic(x), self._B_logic(x), self._B_null_logic(x)
        return self._f_logic(x), self._B_logic(x)

    def get_rollout(self, buffer_size: int, mode: str, num_control_per_state: int | None = None):
        if mode == "c3m":
            xref = (self.X_MAX - self.X_MIN) * torch.rand(buffer_size, self.num_dim_x, device=self.device) + self.X_MIN
            uref = (self.UREF_MAX - self.UREF_MIN) * torch.rand(buffer_size, self.num_dim_control, device=self.device) + self.UREF_MIN
            xe = (self.XE_MAX - self.XE_MIN) * torch.rand(buffer_size, self.num_dim_x, device=self.device) + self.XE_MIN
            x = torch.clamp(xe + xref, self.X_MIN, self.X_MAX)
            return {
                "x": x,
                "xref": xref,
                "uref": uref,
            }

        n_control_per_x = num_control_per_state if num_control_per_state is not None else 3
        batch_size = math.ceil(buffer_size / n_control_per_x)
        if self.sample_mode == "Gaussian":
            x_mean = (self.X_MAX + self.X_MIN) / 2.0
            x_std = (self.X_MAX - self.X_MIN) / 6.0
            u_mean = (self.UREF_MAX + self.UREF_MIN) / 2.0
            u_std = (self.UREF_MAX - self.UREF_MIN) / 6.0
            x = torch.normal(x_mean.expand(batch_size, -1), x_std.expand(batch_size, -1))
            u = torch.normal(u_mean.expand(batch_size, -1), u_std.expand(batch_size, -1))
        else:
            x = self.X_MIN + torch.rand(batch_size, self.num_dim_x, device=self.device) * (self.X_MAX - self.X_MIN)
            u = self.UREF_MIN + torch.rand(batch_size, self.num_dim_control, device=self.device) * (self.UREF_MAX - self.UREF_MIN)
        
        x = x.repeat(n_control_per_x, 1)
        u_list = [u[torch.randperm(batch_size)] for _ in range(n_control_per_x)]
        u = torch.cat(u_list, dim=0)
        
        f, B = self.get_f_and_B(x)
        x_dot = f + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        return {
            "x": x,
            "u": u,
            "x_next": x + x_dot * self.dt,
        }

    def wrap_angles(self, x: torch.Tensor):
        x_copy = x.clone()
        for idx in self.angle_idx:
            x_copy[:, idx] = (x_copy[:, idx] + math.pi) % (2 * math.pi) - math.pi
        return x_copy

    def construct_state(self, x: torch.Tensor):
        idx = torch.clamp(self.time_steps, max=self.max_episode_len - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        xref = self.xref[env_idx, idx]
        uref = self.uref[env_idx, idx]
        return torch.cat([x, xref, uref], dim=-1)

    @staticmethod
    def _zeros(shape, x):
        return torch.zeros(shape, device=x.device, dtype=x.dtype)

    @abstractmethod
    def _f_logic(self, x):
        ...

    @abstractmethod
    def _B_logic(self, x):
        ...

    def _B_null_logic(self, x):
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)
        n = x.shape[0]
        Bbot = torch.cat(
            (torch.eye(eye_dims, device=x.device, dtype=x.dtype),
             torch.zeros(zero_dims, device=x.device, dtype=x.dtype)),
            dim=0,
        )
        return Bbot.repeat(n, 1, 1)


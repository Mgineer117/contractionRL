import sys
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PIL import Image

class WandbPlotWrapper:
    """
    Gym wrapper that dynamically tracks live stochastic episodes.
    It automatically pushes train/normalized_error, train/position_tracking,
    and train/velocity_tracking trajectory curves to wandb.
    """
    def __init__(self, env, total_timesteps=None, num_plots=10):
        self.env = env
        self._episode_count = 0
        self.num_envs = getattr(env, "num_envs", 1)
        self.plot_idx = np.random.choice(self.num_envs, 1, replace=False)
        self._norm_errs = {i: [] for i in self.plot_idx}
        self._traj_x = {i: [] for i in self.plot_idx}
        self._traj_xref = {i: [] for i in self.plot_idx}
        self._total_steps = 0
        # Plot cadence is measured in `.step()` CALLS (== the trainer's "timesteps"/
        # "global_step" unit, i.e. NOT multiplied by num_envs), so it scales with an
        # env's actual episode length instead of a fixed episode COUNT. The previous
        # "every 50 completed episodes" default silently never fired for any run
        # whose (timesteps / episode_len) never reached a multiple of 50 — e.g. a
        # 30000-timestep run of a 500-step-episode classic env completes only ~60
        # episodes total, so it produced at most a single plot (or none, for longer-
        # episode Isaac path-tracking envs). Spacing plots evenly across the whole
        # run guarantees ~num_plots of them regardless of episode length.
        self._step_calls = 0
        self._plot_freq_steps = max(1, int(total_timesteps) // num_plots) if total_timesteps else 1000
        # Negative so the very first completed episode always triggers a plot
        # (early feedback), even before a full `_plot_freq_steps` has elapsed.
        self._last_plot_call = -self._plot_freq_steps
        
    def __getattr__(self, name):
        return getattr(self.env, name)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._norm_errs = {i: [] for i in self.plot_idx}
        self._traj_x = {i: [] for i in self.plot_idx}
        self._traj_xref = {i: [] for i in self.plot_idx}
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += self.num_envs
        self._step_calls += 1

        # Handle batched (Isaac Lab / VectorEnv) vs unbatched
        if isinstance(terminated, torch.Tensor):
            done_0 = bool((terminated | truncated)[0].item())
        elif isinstance(terminated, np.ndarray):
            done_0 = bool((terminated | truncated)[0])
        else:
            done_0 = bool(terminated or truncated)
            
        if hasattr(self.env.unwrapped, "get_tracking_error"):
            # Isaac Lab or wrapped env with get_tracking_error
            try:
                err = self.env.unwrapped.get_tracking_error()
                err = err.cpu().numpy() if isinstance(err, torch.Tensor) else np.array(err)
                for i in self.plot_idx:
                    self._norm_errs[i].append(err[i].item() if err.ndim > 0 else err.item())
            except Exception:
                pass
        else:
            # Check for final_info first so we don't grab the new episode's initial error
            for i in self.plot_idx:
                err = None
                if "final_info" in info and isinstance(info["final_info"], (tuple, list, np.ndarray)) and info["final_info"][i] is not None:
                    if "tracking_error" in info["final_info"][i]:
                        err = info["final_info"][i]["tracking_error"]
                elif "tracking_error" in info:
                    err_array = info["tracking_error"]
                    err_array = err_array.cpu().numpy() if isinstance(err_array, torch.Tensor) else np.array(err_array)
                    err = err_array[i].item() if err_array.ndim > 0 else err_array.item()
                
                if err is not None:
                    self._norm_errs[i].append(err)
                
        for i in self.plot_idx:
            x_i = None
            xref_i = None
            try:
                if hasattr(self.env.unwrapped, "envs"):
                    env_i = self.env.unwrapped.envs[i]
                    if hasattr(env_i, "x_t"):
                        pos_dim = getattr(env_i, "pos_dimension", len(env_i.x_t))
                        x_i = env_i.x_t[:pos_dim]
                        if hasattr(env_i, "xref"):
                            t_idx = min(env_i.time_steps, len(env_i.xref) - 1)
                            xref_i = env_i.xref[t_idx][:pos_dim]
                else:
                    # Isaac envs: state()[..., :3] is position — but path/vel-
                    # tracking envs expose no separate "critic" observation, so
                    # state() returns None. Fall back to the flat policy obs
                    # ([x, xref, uref]), whose first dims are x itself.
                    x_val = self.env.state() if hasattr(self.env, "state") else None
                    if isinstance(x_val, tuple): x_val = x_val[0]
                    if x_val is None:
                        x_val = obs
                    x_i = x_val[i, :3].detach().cpu().numpy() if isinstance(x_val, torch.Tensor) else np.array(x_val)[i, :3]

                    xref_val = getattr(self.env.unwrapped, "_x_ref", None)
                    if xref_val is None and hasattr(self.env.unwrapped, "get_reference_state"):
                        xref_val = self.env.unwrapped.get_reference_state()
                    if xref_val is not None:
                        xref_i = xref_val[i, :3].detach().cpu().numpy() if isinstance(xref_val, torch.Tensor) else np.array(xref_val)[i, :3]
            except Exception:
                # Best-effort trajectory plotting must never crash training.
                pass

            if x_i is not None:
                self._traj_x[i].append(x_i)
            if xref_i is not None:
                self._traj_xref[i].append(xref_i)
            
        if done_0:
            self._episode_count += 1
            if (self._step_calls - self._last_plot_call) >= self._plot_freq_steps \
                    and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
                self._plot_and_push()
                self._last_plot_call = self._step_calls
            self._norm_errs = {i: [] for i in self.plot_idx}
            self._traj_x = {i: [] for i in self.plot_idx}
            self._traj_xref = {i: [] for i in self.plot_idx}
            
        # Lift "log" dicts so SKRL's trainer can log them natively.
        # SyncVectorEnv structures info differently depending on autoreset mode:
        #   NEXT_STEP (default): no "final_info"; per-env "log" dicts are flattened
        #       by _add_info into info["log"] = {metric_key: np.array(num_envs)}
        #       with a boolean mask info["_log"] indicating which envs have data.
        #   SAME_STEP: info["final_info"] is a tuple of per-env dicts (or None).
        device = getattr(self.env, "device", "cpu")
        lifted = False

        # --- Format 1: NEXT_STEP autoreset (default SyncVectorEnv) ---
        # _add_info recursively flattens env_info["log"] into info["log"] as
        # a dict of numpy arrays, with info["_log"] as a boolean mask.
        if "log" in info and "_log" in info and not lifted:
            mask = info["_log"]  # bool array (num_envs,)
            if isinstance(mask, np.ndarray) and mask.any():
                from contractionRL.agents.skrl.eval_metrics import mean_confidence_interval
                raw_log = info["log"]
                # raw_log is a dict whose leaf values are numpy arrays and whose
                # sub-keys starting with "_" are masks — skip those.
                new_log = {}
                for k, v in raw_log.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, np.ndarray):
                        valid = v[mask]
                        if valid.size > 0:
                            v_m, v_ci = mean_confidence_interval(valid)
                            new_log[f"{k}_mean"] = torch.tensor(v_m, dtype=torch.float32, device=device)
                            new_log[f"{k}_ci95"] = torch.tensor(v_ci, dtype=torch.float32, device=device)
                    elif isinstance(v, (int, float)):
                        new_log[f"{k}_mean"] = torch.tensor(float(v), dtype=torch.float32, device=device)
                        new_log[f"{k}_ci95"] = torch.tensor(0.0, dtype=torch.float32, device=device)
                if new_log:
                    info["log"] = new_log
                    lifted = True

        # --- Format 2: SAME_STEP autoreset (final_info is a tuple/list/ndarray) ---
        # gymnasium's SyncVectorEnv._add_info funnels "final_info" through the
        # SAME generic per-key aggregation as every other info key: since each
        # env's `info["final_info"]` value is a `dict`, `_init_info_arrays`
        # allocates an OBJECT-DTYPE NUMPY ARRAY (`np.zeros(num_envs, dtype=object)`),
        # not a tuple/list — checking only (tuple, list) silently skipped this
        # branch on gymnasium>=0.29's SyncVectorEnv, so no Stability/* from
        # classic envs' terminal info dict ever reached wandb.
        if not lifted and "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, (tuple, list, np.ndarray)):
                logs_list = {}
                for fin_info in final_info:
                    if fin_info is not None and isinstance(fin_info, dict) and "log" in fin_info:
                        for k, v in fin_info["log"].items():
                            logs_list.setdefault(k, []).append(float(v))
                
                if logs_list:
                    from contractionRL.agents.skrl.eval_metrics import mean_confidence_interval
                    info["log"] = {}
                    for k, v_list in logs_list.items():
                        v_m, v_ci = mean_confidence_interval(np.array(v_list))
                        info["log"][f"{k}_mean"] = torch.tensor(v_m, dtype=torch.float32, device=device)
                        info["log"][f"{k}_ci95"] = torch.tensor(v_ci, dtype=torch.float32, device=device)

        return obs, reward, terminated, truncated, info

    def close(self, **kwargs):
        # Safety net for envs whose episodes never complete within the run
        # (episode length > total timesteps) — `done_0` then never fires, so
        # the step()-based trigger above never runs either. Push whatever
        # partial trajectory has accumulated so far rather than emitting
        # nothing for the whole run.
        if self._episode_count == 0 and any(len(v) > 0 for v in self._traj_x.values()) \
                and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
            self._plot_and_push()
        return self.env.close(**kwargs)

    def _plot_and_push(self):
        # Delegate to the shared plotter so the standalone PPO/SAC/LQR/SD-LQR
        # path emits the SAME "train/normalized_error" and "train/path_tracking"
        # figures (same style/keys) that C3M/C2RL emit from their eval() — a
        # single source of truth for all tracking plots. self._norm_errs holds
        # RAW per-step error norms (the plotter normalizes by e(0) itself).
        from contractionRL.agents.skrl.contraction_metrics import log_tracking_plots

        # For classic envs, self.env.unwrapped is the SyncVectorEnv itself —
        # `dt` lives on its per-env `.envs[i]`, not on the vector env — so the
        # generic getattr chain below silently fell back to 1.0 (wrong AUC/
        # legend scale). Isaac envs expose `step_dt` directly on `.unwrapped`.
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "envs"):
            dt = getattr(unwrapped.envs[0], "dt", 1.0)
        else:
            dt = getattr(unwrapped, "step_dt", None) or getattr(unwrapped, "dt", 1.0)
        log_tracking_plots(
            self._traj_x, self._traj_xref, self._norm_errs,
            dt=float(dt), prefix="train", step=getattr(self, "_total_steps", 0), title="Train",
        )

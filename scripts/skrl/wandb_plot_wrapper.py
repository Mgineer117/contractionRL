import sys

import numpy as np


class WandbPlotWrapper:
    """Publish tracking curves from the validated StatManagerEnvWrapper buffers."""
    def __init__(self, env, total_timesteps=None, num_plots=10):
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        # Space plots by vector step calls, independently of episode length.
        self._step_calls = 0
        self._plot_freq_steps = max(1, int(total_timesteps) // num_plots) if total_timesteps else 1000
        self._last_plot_call = -self._plot_freq_steps
        # StatManagerEnvWrapper's buffer-completion counter last seen — a plot
        # is only pushed once this has actually advanced, so a slow env (whose
        # eval buffer hasn't completed a fresh round yet) never gets a stale
        # re-push of the previous round's curves.
        self._last_compute_count = getattr(self.env, "_compute_count", 0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_calls += 1

        compute_count = getattr(self.env, "_compute_count", None)
        if compute_count is not None and compute_count != self._last_compute_count:
            if (self._step_calls - self._last_plot_call) >= self._plot_freq_steps \
                    and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
                self._plot_and_push()
                self._last_plot_call = self._step_calls
            self._last_compute_count = compute_count

        # Lift SyncVectorEnv logs for both NEXT_STEP and SAME_STEP autoreset layouts.
        device = getattr(self.env, "device", "cpu")
        lifted = False

        # --- Format 1: NEXT_STEP autoreset (default SyncVectorEnv) ---
        # _add_info recursively flattens env_info["log"] into info["log"] as
        # a dict of numpy arrays, with info["_log"] as a boolean mask.
        if "log" in info and "_log" in info and not lifted:
            mask = info["_log"]  # bool array (num_envs,)
            if isinstance(mask, np.ndarray) and mask.any():
                import torch
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
                            if k.endswith(("_mean", "_stddev", "_ci95")):
                                new_log[k] = torch.tensor(float(valid.mean()), dtype=torch.float32, device=device)
                            else:
                                new_log[f"{k}_mean"] = torch.tensor(float(valid.mean()), dtype=torch.float32, device=device)
                                new_log[f"{k}_stddev"] = torch.tensor(float(valid.std()), dtype=torch.float32, device=device)
                    elif isinstance(v, (int, float)):
                        new_log[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
                if new_log:
                    info["log"] = new_log
                    lifted = True

        # SAME_STEP final_info may be an object array as well as a tuple/list.
        if not lifted and "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, (tuple, list, np.ndarray)):
                logs_list = {}
                for fin_info in final_info:
                    if fin_info is not None and isinstance(fin_info, dict) and "log" in fin_info:
                        for k, v in fin_info["log"].items():
                            logs_list.setdefault(k, []).append(float(v))

                if logs_list:
                    import torch
                    info["log"] = {}
                    for k, v_list in logs_list.items():
                        values = np.asarray(v_list)
                        if k.endswith(("_mean", "_stddev", "_ci95")):
                            info["log"][k] = torch.tensor(float(values.mean()), dtype=torch.float32, device=device)
                        else:
                            info["log"][f"{k}_mean"] = torch.tensor(float(values.mean()), dtype=torch.float32, device=device)
                            info["log"][f"{k}_stddev"] = torch.tensor(float(values.std()), dtype=torch.float32, device=device)

        return obs, reward, terminated, truncated, info

    def close(self, **kwargs):
        # Safety net for envs whose episodes never complete within the run
        # (episode length > total timesteps) — the buffer-completion trigger
        # above then never fires. Push whatever the StatManager buffer holds
        # so far rather than emitting nothing for the whole run.
        if self._last_compute_count == 0 and getattr(self.env, "_compute_count", 0) > 0 \
                and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
            self._plot_and_push()
        return self.env.close(**kwargs)

    def _plot_and_push(self):
        # Delegate to the shared plotter so the standalone PPO/SAC/LQR/SD-LQR
        # path emits the same "train/normalized_error" and "train/path_tracking"
        # figures (same style/keys) that C3M/C2RL emit from their eval() — a
        # single source of truth for all tracking plots, backed by
        # StatManagerEnvWrapper's buffer (already normalized by e(0)).
        from contractionRL.agents.utils.contraction.metrics import log_tracking_plots

        traj_x, traj_xref, traj_err = self.env.trajectories()
        if not traj_err:
            return
        # Filter unused buffer slots; arrays require len() rather than truth testing.
        plot_keys = sorted(k for k in traj_err
                           if traj_x.get(k) is not None and len(traj_x[k]) > 0)
        if not plot_keys:
            return
        traj_x = {k: traj_x.get(k, []) for k in plot_keys}
        traj_xref = {k: traj_xref.get(k, []) for k in plot_keys}
        traj_err = {k: traj_err[k] for k in plot_keys}

        # C2RL only: the metric-weighted analog of normalized_error. Empty (→ no
        # extra figure) for envs that emit no Mahalanobis error (see
        # StatManagerEnvWrapper.maha_trajectories).
        maha_fn = getattr(self.env, "maha_trajectories", None)
        traj_maha = maha_fn() if callable(maha_fn) else {}
        traj_maha = {k: traj_maha[k] for k in plot_keys if k in traj_maha}

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
            traj_x, traj_xref, traj_err,
            dt=float(dt), prefix="train", step=self._step_calls, title="Train",
            traj_maha_error=traj_maha,
        )

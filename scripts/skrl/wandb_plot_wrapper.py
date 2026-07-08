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
    def __init__(self, env, plot_freq_episodes=50):
        self.env = env
        self._plot_freq = plot_freq_episodes
        self._episode_count = 0
        self.num_envs = getattr(env, "num_envs", 1)
        self.plot_idx = np.random.choice(self.num_envs, 1, replace=False)
        self._norm_errs = {i: [] for i in self.plot_idx}
        self._traj_x = {i: [] for i in self.plot_idx}
        self._traj_xref = {i: [] for i in self.plot_idx}
        
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
                if "final_info" in info and isinstance(info["final_info"], tuple) and info["final_info"][i] is not None:
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
            if hasattr(self.env.unwrapped, "envs"):
                env_i = self.env.unwrapped.envs[i]
                if hasattr(env_i, "x_t"):
                    pos_dim = getattr(env_i, "pos_dimension", len(env_i.x_t))
                    x_i = env_i.x_t[:pos_dim]
                    if hasattr(env_i, "xref"):
                        t_idx = min(env_i.time_steps, len(env_i.xref) - 1)
                        xref_i = env_i.xref[t_idx][:pos_dim]
            else:
                if hasattr(self.env, "state"):
                    x_val = self.env.state()
                    if isinstance(x_val, tuple): x_val = x_val[0]
                    # Isaac envs: state()[..., :3] is position
                    x_i = x_val[i, :3].detach().cpu().numpy() if isinstance(x_val, torch.Tensor) else np.array(x_val)[i, :3]
                else:
                    x_i = obs[i, :3].detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.array(obs)[i, :3]
                
                xref_val = getattr(self.env.unwrapped, "_x_ref", None)
                if xref_val is None and hasattr(self.env.unwrapped, "get_reference_state"):
                    xref_val = self.env.unwrapped.get_reference_state()
                if xref_val is not None:
                    xref_i = xref_val[i, :3].detach().cpu().numpy() if isinstance(xref_val, torch.Tensor) else np.array(xref_val)[i, :3]

            if x_i is not None:
                self._traj_x[i].append(x_i)
            if xref_i is not None:
                self._traj_xref[i].append(xref_i)
            
        if done_0:
            self._episode_count += 1
            if self._episode_count % self._plot_freq == 0 and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
                self._plot_and_push()
            self._norm_errs = {i: [] for i in self.plot_idx}
            self._traj_x = {i: [] for i in self.plot_idx}
            self._traj_xref = {i: [] for i in self.plot_idx}
            
        return obs, reward, terminated, truncated, info
        
    def _plot_and_push(self):
        import wandb
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for i, errs in self._norm_errs.items():
            if not errs:
                continue
            errs_arr = np.array(errs)
            e0 = max(errs_arr[0], 1e-8)
            norm_errs = errs_arr / e0
            
            dt = getattr(self.env.unwrapped, "step_dt", None) or getattr(self.env.unwrapped, "dt", 1.0)
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            auc = float(_trapz(norm_errs, dx=float(dt)))
            
            ax.plot(norm_errs, label=f'Env {i} (AUC: {auc:.2f})')
            plotted = True
            
        if not plotted:
            plt.close(fig)
            return
            
        ax.set_title("Live Stochastic Tracking Error")
        ax.set_xlabel("Step")
        ax.set_ylabel("Normalized Error")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # We only log if wandb run is active
        try:
            img = Image.open(buf)
            wandb.log({"train/normalized_error": wandb.Image(img)})
        except Exception:
            pass
            
        # Plot Trajectory Positions (if reference available)
        plotted_pos = False
        fig2 = plt.figure(figsize=(12, 4))
        for i in self.plot_idx:
            if len(self._traj_x[i]) > 0 and len(self._traj_xref[i]) > 0:
                tx = np.array(self._traj_x[i])
                txref = np.array(self._traj_xref[i])
                if tx.shape[-1] >= 1 and txref.shape[-1] >= 1:
                    plotted_pos = True
                    if tx.shape[-1] == 1:
                        ax2 = fig2.add_subplot(111) if not plotted_pos or 'ax2' not in locals() else ax2
                        time_steps = np.arange(len(tx))
                        ax2.scatter(time_steps, tx[:, 0], c=time_steps, cmap='viridis', s=10, label=f'x (env {i})')
                        ax2.plot(time_steps, txref[:, 0], '--', color='red', label=f'x_ref (env {i})')
                        ax2.set_xlabel("Time Step")
                        ax2.set_ylabel("Position")
                    elif tx.shape[-1] == 2:
                        ax2 = fig2.add_subplot(111) if not plotted_pos or 'ax2' not in locals() else ax2
                        time_steps = np.arange(len(tx))
                        ax2.scatter(tx[:, 0], tx[:, 1], c=time_steps, cmap='viridis', s=10, label=f'x (env {i})')
                        ax2.plot(txref[:, 0], txref[:, 1], '--', color='red', label=f'x_ref (env {i})')
                        ax2.set_xlabel("X Position")
                        ax2.set_ylabel("Y Position")
                    else:
                        ax2 = fig2.add_subplot(111, projection='3d') if not plotted_pos or 'ax2' not in locals() else ax2
                        time_steps = np.arange(len(tx))
                        ax2.scatter(tx[:, 0], tx[:, 1], tx[:, 2], c=time_steps, cmap='viridis', s=10, label=f'x (env {i})')
                        ax2.plot(txref[:, 0], txref[:, 1], txref[:, 2], '--', color='red', label=f'x_ref (env {i})')
                        ax2.set_xlabel("X Position")
                        ax2.set_ylabel("Y Position")
                        ax2.set_zlabel("Z Position")
                        
        if plotted_pos:
            ax2.set_title("Path-Tracking Positions")
            ax2.legend()
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', bbox_inches='tight')
            buf2.seek(0)
            try:
                img2 = Image.open(buf2)
                wandb.log({"train/trajectory_positions": wandb.Image(img2)})
            except Exception:
                pass
        plt.close(fig2)

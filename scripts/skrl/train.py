"""Train RL agents with skrl — Isaac Sim and classic gymnasium environments.

Isaac Sim (default):
    python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo
    python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m

Classic gymnasium (--classic flag, no Isaac Sim required):
    python scripts/skrl/train.py --classic --task Car-v0 --algorithm ppo
    python scripts/skrl/train.py --classic --task Car-v0 --algorithm c3m
"""

import argparse
import sys

# ─── Pre-parse: must know --classic BEFORE any Isaac Sim imports ──────────── #
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--classic", action="store_true", default=False)
_pre_args, _ = _pre.parse_known_args()
_is_classic = _pre_args.classic

if not _is_classic:
    from isaaclab.app import AppLauncher

# ─── Full argument parser ─────────────────────────────────────────────────── #
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--classic", action="store_true", default=False,
                    help="Use classic gymnasium environment (no Isaac Sim).")
parser.add_argument("--task", type=str, default=None, help="Environment ID.")
parser.add_argument(
    "--algorithm", "--algo", type=str, default="PPO",
    help="Algorithm: ppo | sac | c3m | lqr | sdlqr | c2rl-ppo | c2rl-sac | AMP | DDPG | TD3 | …"
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None,
                    help="Total training timesteps.")
parser.add_argument("--analytical", type=str, default="",
                    help="Pass 'dynamics' to use analytical dynamics (C3M/LQR).")
parser.add_argument("--use_analytical_dynamics", "--use-analytical-dynamics",
                    action="store_true", default=False,
                    help="Use the env's exact analytical get_f_and_B instead of a learned "
                         "NeuralDynamics (classic envs only). When NOT passed, C3M/C2RL "
                         "pretrain + refine a NeuralDynamics model (the default for all envs).")

# W&B
parser.add_argument("--no_wandb", "--no-wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging.")
parser.add_argument("--wandb_project", "--wandb-project", type=str, default="contractionRL",
                    help="W&B project name.")
parser.add_argument("--wandb_run_name", "--wandb-run-name", type=str, default=None,
                    help="W&B run name.")

# Isaac Sim-specific
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training (Isaac only).")
parser.add_argument("--video_length", type=int, default=0,
                    help="Length of video in steps (0 = auto-calculate to 1 episode length)")
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--agent", type=str, default=None,
                    help="Explicit skrl cfg entry-point key (Isaac only).")
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
parser.add_argument("--debug_vis", action="store_true", default=False)
# SAC HP overrides
parser.add_argument("--sac_lr", "--sac-lr", type=float, default=None)
parser.add_argument("--sac_batch_size", "--sac-batch-size", type=int, default=None)
parser.add_argument("--sac_discount", "--sac-discount", type=float, default=None)
parser.add_argument("--sac_polyak", "--sac-polyak", type=float, default=None)
parser.add_argument("--sac_gradient_steps", "--sac-gradient-steps", type=int, default=None)
parser.add_argument("--sac_entropy", "--sac-entropy", type=float, default=None)
parser.add_argument("--sac_memory_size", "--sac-memory-size", type=int, default=None)
# PPO HP overrides
parser.add_argument("--ppo_lr", "--ppo-lr", type=float, default=None)
parser.add_argument("--ppo_rollouts", "--ppo-rollouts", type=int, default=None)
parser.add_argument("--ppo_learning_epochs", "--ppo-learning-epochs", type=int, default=None)
parser.add_argument("--ppo_mini_batches", "--ppo-mini-batches", type=int, default=None)
parser.add_argument("--ppo_discount", "--ppo-discount", type=float, default=None)
parser.add_argument("--ppo_lambda", "--ppo-lambda", type=float, default=None)
parser.add_argument("--ppo_ratio_clip", "--ppo-ratio-clip", type=float, default=None)
parser.add_argument("--ppo_entropy_scale", "--ppo-entropy-scale", type=float, default=None)
parser.add_argument("--ppo_kl_threshold", "--ppo-kl-threshold", type=float, default=None)
parser.add_argument("--ppo_use_state_norm", "--ppo-use-state-norm", type=str, default=None)
parser.add_argument("--ppo_use_value_norm", "--ppo-use-value-norm", type=str, default=None)
parser.add_argument("--ppo_activations", "--ppo-activations", type=str, default=None)
parser.add_argument("--ppo_network_arch", "--ppo-network-arch", type=str, default=None)

# Classic-specific
parser.add_argument("--cfg", type=str, default=None,
                    help="Path to a custom YAML config (classic only).")
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate override (classic only).")
parser.add_argument("--epochs", type=int, default=None,
                    help="Training epochs override (classic only).")

# Reference trajectory generation (auto-triggered after vel-tracking training)
parser.add_argument("--ref_num_trajs", type=int, default=1000,
                    help="Number of reference trajectories to collect after vel-tracking training.")
parser.add_argument("--min_ref_quality", type=float, default=None,
                    help="Minimum mean episode reward before generating ref trajs. 0 to skip check.")
parser.add_argument("--min_ref_traj_length_frac", type=float, default=0.5,
                    help="Minimum fraction of the max episode length (T) a trajectory must survive "
                         "to be accepted into dynamics_data.npz. Trajectories shorter than "
                         "min_ref_traj_length_frac * T are discarded (default 0.5, i.e. half of T).")
parser.add_argument("--ref_oversample_factor", type=float, default=2.0,
                    help="Collect this many times ref_num_trajs candidate trajectories (that clear "
                         "min_ref_traj_length_frac) before keeping only the longest ref_num_trajs of "
                         "them — gives the selection room to prefer complete rollouts over ones that "
                         "survived just past the minimum length. 1.0 disables oversampling.")

if not _is_classic:
    AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if not _is_classic and args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

if not _is_classic:
    args_cli.kit_args = (args_cli.kit_args or "") + " --/app/hangDetector/enabled=false"
    hydra_args = [arg for arg in hydra_args if not (arg.startswith("--") and ("=" in arg or "." in arg))]
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    # Suppress noisy mesh/hydra warnings from Isaac Sim assets
    import carb
    carb.settings.get_settings().set("/log/logger/channelFilter", "-omni.hydra")

# ─── Shared imports ───────────────────────────────────────────────────────── #
import logging
import os
import random
from datetime import datetime

import gymnasium as gym
import yaml

algorithm = args_cli.algorithm.lower()
_CONTRACTION_ALGOS = {"c3m", "lqr", "sdlqr", "c2rl-ppo", "c2rl-sac"}

seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}


def _inject_angle_idx(agent_cfg: dict, angle_idx: list) -> None:
    """Inject ``angle_idx`` into every model sub-block of agent_cfg["models"].

    Only the STANDALONE PPO/SAC path needs this: those models are built by
    _gaussian_factory/_deterministic_factory (runner.py) purely from each
    yaml/cfg block's own keys, with no access to the env object. The
    ContractionRunner path (C3M/LQR/SDLQR/C2RL) is self-sufficient — it reads
    angle_idx directly off the env in _setup_contraction — so this is a no-op
    for that path. A no-op (angle_idx=[]) here is also harmless: every
    consumer treats an empty angle_idx as "nothing to embed".
    """
    if not angle_idx:
        return
    for block in agent_cfg.get("models", {}).values():
        if isinstance(block, dict):
            block.setdefault("angle_idx", angle_idx)


def _max_step_reward(robot: str, env_cfg) -> float:
    """Best-case per-step reward for a vel-tracking env's reward function.

    quadruped/humanoid: alive bonus + exp-tracking terms, each of which saturates
    at its scale when the tracking error is 0; every other term is `nonneg * a
    non-positive scale`, so its best case is 0. manipulator has no alive bonus
    and every term is `error * negative_scale`, so its best case is exactly 0.
    """
    if robot in ("quadruped", "humanoid"):
        return env_cfg.rew_alive + env_cfg.rew_lin_vel + env_cfg.rew_yaw_rate
    if robot == "manipulator":
        return 0.0
    raise ValueError(f"no max-reward formula for robot '{robot}'")


def _generate_ref_trajs(*, task, runner, isaac_env, skrl_env, env_cfg):
    import numpy as np
    import torch

    robot = next((name for prefix, name in _VEL_TASK_TO_ROBOT.items() if task.startswith(prefix)), None)
    if robot is None:
        print(f"[RefTraj] No robot mapping for task '{task}'; skipping.")
        return

    out_dir = os.path.join(_ROOT, "logs", robot)
    out_path = os.path.join(out_dir, "dynamics_data.npz")
    T = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    # Quality threshold = half of the theoretical best-case total episode reward
    # (best-case per-step reward × T), rather than a hand-picked constant — this
    # tracks whatever the reward scales in env_cfg actually are per task.
    min_reward = args_cli.min_ref_quality if args_cli.min_ref_quality is not None \
        else 0.5 * _max_step_reward(robot, env_cfg) * T

    # Load best checkpoint
    import logging as _logging
    _skrl_log = _logging.getLogger("skrl")
    _prev_level = _skrl_log.level
    _skrl_log.setLevel(_logging.ERROR)
    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[RefTraj] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print(f"[RefTraj] WARNING: best_agent.pt not found; using final weights.")
    _skrl_log.setLevel(_prev_level)
    for model in agent.models.values():
        if model is not None:
            model.eval()

    def _get_obs(o):
        return o["policy"] if isinstance(o, dict) else o

    # Quality gate: 1 full episode across all parallel environments
    if min_reward > 0:
        print(f"\n[RefTraj] Evaluating quality (threshold: mean total reward >= {min_reward}) …")

        # SkrlVecEnvWrapper auto-resets each env individually on done, so we
        # must NOT mask out rewards after done — the env is already running its
        # next episode. We evaluate exactly 1 full episode for all parallel envs
        # to prevent aggressively selecting early-terminating failure episodes.
        ep_rewards = []
        ep_r = torch.zeros(skrl_env.num_envs, device=skrl_env.device)
        finished = torch.zeros(skrl_env.num_envs, dtype=torch.bool, device=skrl_env.device)
        
        if hasattr(skrl_env, "_reset_once"):
            skrl_env._reset_once = True
        obs_dict, _ = skrl_env.reset()
        obs = _get_obs(obs_dict)
        
        # We run for slightly more than T steps to ensure all envs finish their first episode
        for _ in range(T + 1):
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
            obs_dict, rewards, terminated, truncated, _ = skrl_env.step(actions)
            obs = _get_obs(obs_dict)
            
            # accumulate reward only for envs that haven't finished their first episode
            ep_r += rewards.squeeze(-1) * (~finished).float()
            done = (terminated | truncated).squeeze(-1)
            
            # only record the reward when an env finishes its first episode
            just_finished = done & (~finished)
            for i in just_finished.nonzero(as_tuple=True)[0]:
                ep_rewards.append(ep_r[i.item()].item())
            
            finished |= done
            if finished.all():
                break

        # If any envs somehow didn't finish, we record their accumulated rewards
        not_finished = ~finished
        for i in not_finished.nonzero(as_tuple=True)[0]:
            ep_rewards.append(ep_r[i.item()].item())

        if not ep_rewards:
            print("[RefTraj] WARNING: no complete episodes; skipping.")
            return
        mean_r = sum(ep_rewards) / len(ep_rewards)
        print(f"[RefTraj] Mean total reward: {mean_r:.1f}")
        if mean_r < min_reward:
            print(
                f"[RefTraj] SKIPPED — policy quality too low "
                f"({mean_r:.1f} < {min_reward}). Train longer or pass --min_ref_quality 0."
            )
            return

    # Collect trajectories. We over-collect a candidate pool larger than
    # num_trajs (oversample_factor x) and then keep the LONGEST num_trajs of
    # them — early termination is exactly what a poor/failing rollout looks
    # like, so ranking by survival length is a simple, direct proxy for
    # "better trajectory". Recording every one of num_envs (rather than just
    # the first min(num_trajs, num_envs)) maximizes that pool for free: Isaac
    # can't shrink the batch, so the extra envs are being simulated regardless.
    # This also means num_envs < num_trajs naturally loops through as many
    # per-env episode rounds as it takes to fill the pool — no special-casing
    # needed for that direction.
    import math
    import tqdm

    num_trajs = args_cli.ref_num_trajs
    pool_target = max(num_trajs, int(math.ceil(num_trajs * max(1.0, args_cli.ref_oversample_factor))))
    print(f"[RefTraj] Collecting a candidate pool of {pool_target} trajectories "
          f"(oversample x{args_cli.ref_oversample_factor:g}), keeping the longest {num_trajs} → {out_path}")
    unwrapped = isaac_env.unwrapped
    num_envs = skrl_env.num_envs
    all_states, all_actions, all_pos, all_lengths = [], [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)

    # Action-space bounds — used ONLY when writing into the saved `u` array
    # below, never to modify what's stepped through the env. The policy
    # samples with clip_actions=False (clipping inside the actor corrupts the
    # log-prob), and the env already enforces action bounds on its own (its
    # actuator/physics pipeline), so re-clipping before `skrl_env.step()` would
    # be redundant. But the *saved* dynamics_data.npz must record actions
    # within the declared action space — an unclipped, possibly out-of-range
    # sample is not a valid "u" for fitting f(x) + B(x)u.
    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]

    ep_states = torch.zeros((num_envs, T, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T, u_dim), dtype=torch.float32, device=skrl_env.device)
    ep_pos = torch.zeros((num_envs, T, 3), dtype=torch.float32, device=skrl_env.device)
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=skrl_env.device)

    pbar = tqdm.tqdm(total=pool_target, desc="[RefTraj] Collecting candidates")

    while len(all_states) < pool_target:
        with torch.no_grad():
            actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
        state_tensor = unwrapped.get_physical_state()

        # Record state and action for envs that are still within T steps
        valid_mask = step_counts < T
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        # Clip only for the SAVED record, not for stepping (see note above).
        ep_actions[valid_indices, step_counts[valid_indices]] = \
            torch.clamp(actions[valid_indices], _act_low, _act_high).float()
        if hasattr(unwrapped, "_robot"):
            ep_pos[valid_indices, step_counts[valid_indices]] = unwrapped._robot.data.root_pos_w[valid_indices].float()
        
        step_counts[valid_indices] += 1
        
        obs_dict, _, terminated, truncated, _ = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)

        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # Accept trajectories that survived at least min_ref_traj_length_frac
            # of the max length (default 0.5 = half of T). This handles policies
            # that fall slightly early but pass the quality gate, as well as
            # off-by-one errors with Isaac Gym's max_episode_length.
            min_len = int(args_cli.min_ref_traj_length_frac * T)
            success_mask = step_counts[done_indices] >= min_len
            success_indices = done_indices[success_mask]
            
            if len(success_indices) > 0:
                # Pad any missing steps with the final valid state to ensure x_dot is stable
                for i in success_indices:
                    length = step_counts[i].item()
                    if length < T and length > 0:
                        ep_states[i, length:] = ep_states[i, length - 1].clone()
                        ep_actions[i, length:] = ep_actions[i, length - 1].clone()
                        ep_pos[i, length:] = ep_pos[i, length - 1].clone()

                # Move to CPU in bounded chunks. Episodes are length-synchronized
                # (fixed T), so on the first `done` event success_indices can be
                # ~num_envs at once — gathering all of them in one fancy-index
                # would allocate a full (len(success_indices), T, dim) CUDA
                # temporary, so keep it chunked regardless of pool size.
                #
                # Deliberately don't early-break once len(all_states) hits
                # pool_target here: this round's successes are already sitting
                # in GPU memory finished at the same time, so cutting the chunk
                # loop short would arbitrarily favor low env-index trajectories
                # over otherwise-equal ones later in `success_indices`. Letting
                # the whole round in (pool may overshoot pool_target a bit)
                # keeps every env that finished this round in the running for
                # the final longest-num_trajs selection.
                _CHUNK = 256
                for start in range(0, len(success_indices), _CHUNK):
                    idx = success_indices[start:start + _CHUNK]
                    s_np = ep_states[idx].cpu().numpy()
                    a_np = ep_actions[idx].cpu().numpy()
                    p_np = ep_pos[idx].cpu().numpy()
                    l_np = step_counts[idx].cpu().numpy()
                    for i in range(len(idx)):
                        all_states.append(s_np[i])
                        all_actions.append(a_np[i])
                        all_pos.append(p_np[i])
                        all_lengths.append(int(l_np[i]))
                        pbar.update(1)
            
            # Reset the step counts for all finished environments
            step_counts[done_indices] = 0

    pbar.close()

    # Keep the num_trajs LONGEST candidates out of the oversampled pool.
    all_lengths_np = np.asarray(all_lengths, dtype=np.int64)
    keep = np.argsort(all_lengths_np)[::-1][:num_trajs]
    print(f"[RefTraj] Pool lengths: min={all_lengths_np.min()}, max={all_lengths_np.max()}, "
          f"median={int(np.median(all_lengths_np))} (T={T}) — keeping top {num_trajs} by length")
    states_arr = np.stack([all_states[i] for i in keep]).astype(np.float32)
    actions_arr = np.stack([all_actions[i] for i in keep]).astype(np.float32)
    pos_arr = np.stack([all_pos[i] for i in keep]).astype(np.float32)
    lengths_arr = all_lengths_np[keep]

    os.makedirs(out_dir, exist_ok=True)
    
    # Plot absolute position of 10 sampled trajectories
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for i in range(min(10, num_trajs)):
            plt.plot(pos_arr[i, :, 0], pos_arr[i, :, 1], label=f"Traj {i+1}")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Sampled Reference Trajectories (Absolute Position)")
        plt.legend()
        plot_path = os.path.join(out_dir, "position_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[RefTraj] Saved position plot → {plot_path}")
    except Exception as e:
        print(f"[RefTraj] Failed to generate position plot: {e}")
    
    # Generate dynamics data via finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"[RefTraj] Computing dynamics (x_dot) via 4th-order central difference (dt={dt:.3f})...")

    # angle_idx columns (e.g. yaw) wrap at +-pi in the SAVED states_arr — a raw
    # finite difference across that wrap would spike x_dot by ~2*pi/dt for one
    # sample. Difference an UNWRAPPED copy instead (np.unwrap makes each angle
    # column continuous by adding +-2*pi at jumps); states_arr itself (saved as
    # `x` below) is left untouched — NeuralDynamics only ever consumes x through
    # its (cos, sin) embedding, which is identical for theta and theta + 2*pi*k,
    # so this is purely a finite-difference cleanup, not a semantic change to x.
    angle_idx = list(getattr(isaac_env.unwrapped, "angle_idx", []) or [])
    diff_states = states_arr
    if angle_idx:
        diff_states = states_arr.copy()
        for idx in angle_idx:
            diff_states[:, :, idx] = np.unwrap(diff_states[:, :, idx], axis=1)

    x_dot_arr = np.zeros_like(states_arr)
    for i in range(2, diff_states.shape[1] - 2):
        x_dot_arr[:, i] = (-diff_states[:, i + 2] + 8 * diff_states[:, i + 1] - 8 * diff_states[:, i - 1] + diff_states[:, i - 2]) / (12 * dt)
    # Forward/backward differences for boundaries
    x_dot_arr[:, 0] = (-3 * diff_states[:, 0] + 4 * diff_states[:, 1] - diff_states[:, 2]) / (2 * dt)
    x_dot_arr[:, 1] = (-3 * diff_states[:, 1] + 4 * diff_states[:, 2] - diff_states[:, 3]) / (2 * dt)
    x_dot_arr[:, -2] = (3 * diff_states[:, -2] - 4 * diff_states[:, -3] + diff_states[:, -4]) / (2 * dt)
    x_dot_arr[:, -1] = (3 * diff_states[:, -1] - 4 * diff_states[:, -2] + diff_states[:, -3]) / (2 * dt)
    
    # Filter out any episodes that contain NaNs
    nan_mask = np.isnan(states_arr).any(axis=(1, 2)) | np.isnan(actions_arr).any(axis=(1, 2)) | np.isnan(x_dot_arr).any(axis=(1, 2))
    if nan_mask.any():
        num_nans = nan_mask.sum()
        print(f"[RefTraj] WARNING: Found NaNs in {num_nans} episodes! Filtering them out before saving...")
        valid_mask = ~nan_mask
        states_arr = states_arr[valid_mask]
        actions_arr = actions_arr[valid_mask]
        x_dot_arr = x_dot_arr[valid_mask]
        pos_arr = pos_arr[valid_mask]
        lengths_arr = lengths_arr[valid_mask]

    # Single unified file: reference trajectories ARE the (x, u) part of the
    # dynamics data, so there is no separate ref_trajs.npz anymore.
    #   x       (N, T, x_dim)  physical states; steps >= lengths[n] are padding
    #                          (the last valid state repeated, keeping x_dot ~ 0)
    #   u       (N, T, u_dim)  executed (clipped) actions, same padding rule
    #   x_dot   (N, T, x_dim)  4th-order central differences of x
    #   pos     (N, T, 3)      world-frame [x, y, z] root position — VISUALIZATION ONLY, not
    #                          part of the dynamics; consumers that fit f(x)/B(x)
    #                          (e.g. C3M's NeuralDynamics pretraining) must read
    #                          only x/u/x_dot/lengths from this file and ignore
    #                          this key, exactly like they already do today.
    #   lengths (N,)           number of VALID steps per trajectory — consumers
    #                          mask with arange(T) < lengths[:, None]
    dyn_path = os.path.join(out_dir, "dynamics_data.npz")
    np.savez_compressed(dyn_path, x=states_arr, u=actions_arr, x_dot=x_dot_arr, pos=pos_arr, lengths=lengths_arr)
    print(f"[RefTraj] Saved dynamics  → {dyn_path}")
    print(f"       x       shape: {states_arr.shape}")
    print(f"       u       shape: {actions_arr.shape}   (clipped to action-space bounds)")
    print(f"       x_dot   shape: {x_dot_arr.shape}")
    print(f"       pos     shape: {pos_arr.shape}   (visualization only)")
    print(f"       lengths shape: {lengths_arr.shape}  (min {lengths_arr.min()}, max {lengths_arr.max()})")


def _evaluate_classic_path_tracking(*, task, runner, num_groups: int = 10, episodes_per_group: int = 5):
    """Post-training evaluation for CLASSIC path-tracking envs (CAC-dev style).

    Classic envs (car/cartpole/segway/turtlebot under tasks/direct/classic/)
    are plain (non-vectorized) gymnasium Envs, ported directly from CAC-dev's
    envs/xyD/*.py, with variable-length episodes (BaseEnv.system_reset() can
    end early) and no early termination (`termination` is always False —
    episodes only truncate at their sampled length). That means, unlike the
    Isaac path-tracking rollout, there is no "terminate_on_fall" concept and
    no vectorized-episode-boundary bookkeeping needed — this mirrors CAC-dev's
    trainer/evaluator.py directly: a plain python loop over ONE env instance,
    one episode at a time, using its native `tracking_error`/`dt` step info.

    Reports mean +/- 95% CI of: total reward, error AUC (normalized error,
    trapezoid), and overshoot C / contraction rate lambda from the minimal-AUC
    exponential envelope C * exp(-lambda * k * dt).
    """
    import json

    import numpy as np
    import torch

    from contractionRL.agents.skrl.eval_metrics import (
        fit_exponential_envelope,
        mean_confidence_interval,
    )

    probe = gym.make(task)
    if not hasattr(probe.unwrapped, "xref"):
        print(f"[Eval] SKIPPED — env {type(probe.unwrapped).__name__} has no reference trajectory (xref).")
        probe.close()
        return
    probe.close()

    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[Eval] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print("[Eval] WARNING: best_agent.pt not found; evaluating final weights.")
    for model in agent.models.values():
        if model is not None:
            model.eval()

    device = agent.device
    env = gym.make(task)

    reward_list, auc_list, C_list, lbd_list = [], [], [], []
    print(f"[Eval] Rolling out {num_groups * episodes_per_group} episodes on {task} …")
    for _g in range(num_groups):
        error_trajs = []
        for _e in range(episodes_per_group):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            error_traj = []
            dt = env.unwrapped.dt
            while not done:
                obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    # see _evaluate_best_model for why agent.act() (not
                    # agent.policy.act()) is the algorithm-agnostic interface
                    actions, outputs = agent.act(obs_t, None, timestep=0, timesteps=0)
                    action = outputs.get("mean_actions", actions)[0].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += float(reward)
                # tracking_error is a squared norm (see BaseEnv.get_rewards);
                # sqrt to work in the same plain-norm convention as the Isaac
                # evaluator's get_tracking_error().
                error_traj.append(float(np.sqrt(max(info["tracking_error"], 0.0))))
            e0 = max(error_traj[0], 1e-8) if error_traj else 1.0
            norm_traj = np.asarray(error_traj) / e0
            error_trajs.append(norm_traj)
            reward_list.append(ep_reward)
            auc_list.append(float(np.trapezoid(norm_traj, dx=dt)) if hasattr(np, "trapezoid")
                             else float(np.trapz(norm_traj, dx=dt)))
        # paper fit: one overshoot C* per group, one convergence rate per curve
        C, lbds = fit_exponential_envelope(error_trajs, dt)
        C_list.append(C)
        lbd_list.extend(float(x) for x in lbds)
    env.close()

    rew_mean, rew_ci = mean_confidence_interval(reward_list)
    auc_mean, auc_ci = mean_confidence_interval(auc_list)
    C_mean, C_ci = mean_confidence_interval(C_list)
    lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
    results = {
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": num_groups * episodes_per_group,
        "total_reward_mean": rew_mean, "total_reward_ci95": rew_ci,
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
        "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
        "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
        "num_fit_groups": num_groups,
    }

    print("[Eval] ── Best-model evaluation (classic path-tracking) ──")
    print(f"[Eval] total reward     : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={len(reward_list)})")
    print(f"[Eval] error AUC        : {auc_mean:.4f} ± {auc_ci:.4f}")
    print(f"[Eval] overshoot C      : {C_mean:.3f} ± {C_ci:.3f}")
    print(f"[Eval] contraction rate : {lbd_mean:.4f} ± {lbd_ci:.4f}  (C·e^(−λkΔt), min AUC)")

    out_json = os.path.join(agent.experiment_dir, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved → {out_json}")

    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        sys.modules["wandb"].log({f"final_eval/{k}": v for k, v in results.items()
                                  if isinstance(v, (int, float))})


def _evaluate_best_model(*, task, runner, isaac_env, skrl_env, env_cfg, num_groups: int = 10):
    """Post-training evaluation of the BEST checkpoint (CAC-dev style).

    Loads best_agent.pt, disables fall termination (episodes always run the
    full length so metrics are comparable across policies), rolls out one full
    episode in every parallel env with deterministic (mean) actions clipped to
    the action space, and reports mean +/- 95% CI of:

      * total reward
      * AUC of the velocity-tracking error (trapezoid, dt-weighted)
      * contraction rate lambda and overshoot C — the exponential envelope
        C * exp(-lambda * k * dt) bounding the normalized error curves with
        minimal envelope AUC (= C/lambda), fitted per env-group (CAC-dev
        trainer/evaluator.py compute_contraction_rate).

    Results are printed, logged to wandb (if active), and saved as
    eval_results.json next to the checkpoints.
    """
    import json

    import numpy as np
    import torch

    from contractionRL.agents.skrl.eval_metrics import (
        fit_exponential_envelope,
        mean_confidence_interval,
    )

    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[Eval] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print("[Eval] WARNING: best_agent.pt not found; evaluating final weights.")
    for model in agent.models.values():
        if model is not None:
            model.eval()

    unwrapped = isaac_env.unwrapped
    dt = env_cfg.sim.dt * env_cfg.decimation
    T = int(env_cfg.episode_length_s / dt)
    num_envs = skrl_env.num_envs

    if not hasattr(unwrapped, "get_tracking_error"):
        print(f"[Eval] SKIPPED — env {type(unwrapped).__name__} has no get_tracking_error().")
        return

    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Non-terminating evaluation: flip the cfg flag (read every step by
    # _get_dones) and restore afterwards.
    prev_flag = getattr(unwrapped.cfg, "terminate_on_fall", True)
    unwrapped.cfg.terminate_on_fall = False
    try:
        if hasattr(skrl_env, "_reset_once"):
            skrl_env._reset_once = True
        obs_dict, _ = skrl_env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        total_reward = torch.zeros(num_envs, device=skrl_env.device)
        errors = torch.zeros(num_envs, T + 1, device=skrl_env.device)
        errors[:, 0] = unwrapped.get_tracking_error()

        print(f"[Eval] Rolling out {num_envs} non-terminating episodes of {T} steps …")
        for k in range(T):
            with torch.no_grad():
                # agent.act() is the uniform interface across every skrl Agent
                # (PPO/SAC/C3M/C2RL/SDLQR/LQR) — unlike agent.policy.act(...),
                # which assumes PPO/SAC's internal attribute names and breaks
                # on contraction agents. "mean_actions" (present for Gaussian
                # policies) gives the deterministic action; deterministic
                # policies (e.g. C3M's CLDeterministicActorModel) have no
                # separate mean, so their raw action IS already deterministic.
                actions, outputs = agent.act(obs, None, timestep=0, timesteps=0)
                actions = torch.clamp(outputs.get("mean_actions", actions), _act_low, _act_high)
            obs_dict, rewards, terminated, truncated, _ = skrl_env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            total_reward += rewards.squeeze(-1)
            errors[:, k + 1] = unwrapped.get_tracking_error()
    finally:
        unwrapped.cfg.terminate_on_fall = prev_flag

    err_np = errors.cpu().numpy()  # (N, T+1)
    rew_np = total_reward.cpu().numpy()

    # AUC over the raw error curve (dt-weighted trapezoid), per episode.
    # np.trapezoid is numpy>=2 only; env_isaaclab ships numpy 1.26 (trapz).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_np = _trapz(err_np, dx=dt, axis=1)

    # Contraction envelope on NORMALIZED error e(t)/e(0) — CAC-dev convention.
    # Envs whose initial error is ~0 (near-zero commanded velocity) carry no
    # contraction information and are excluded from the fit.
    e0 = err_np[:, 0]
    fit_mask = e0 > 0.05
    C_list, lbd_list = [], []
    fit_ids = np.nonzero(fit_mask)[0]
    if len(fit_ids) >= num_groups:
        groups = np.array_split(fit_ids, num_groups)
        for g in groups:
            # raw error curves; fit_exponential_envelope normalizes by e(0) itself
            raw_trajs = [err_np[i] for i in g]
            C, lbds = fit_exponential_envelope(raw_trajs, dt)
            C_list.append(C)
            lbd_list.extend(float(x) for x in lbds)
    else:
        print(f"[Eval] WARNING: only {len(fit_ids)} envs with e(0) > 0.05; skipping contraction fit.")

    rew_mean, rew_ci = mean_confidence_interval(rew_np)
    auc_mean, auc_ci = mean_confidence_interval(auc_np)
    results = {
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": int(num_envs),
        "episode_steps": int(T),
        "total_reward_mean": rew_mean, "total_reward_ci95": rew_ci,
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
    }
    if C_list:
        C_mean, C_ci = mean_confidence_interval(C_list)
        lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
        results.update({
            "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
            "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
            "num_fit_groups": len(C_list),
        })

    print("[Eval] ── Best-model evaluation (non-terminating) ──")
    print(f"[Eval] total reward     : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={num_envs})")
    print(f"[Eval] error AUC        : {auc_mean:.4f} ± {auc_ci:.4f}")
    if C_list:
        print(f"[Eval] overshoot C      : {C_mean:.3f} ± {C_ci:.3f}")
        print(f"[Eval] contraction rate : {lbd_mean:.4f} ± {lbd_ci:.4f}  (C·e^(−λkΔt), min AUC)")

    out_json = os.path.join(agent.experiment_dir, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved → {out_json}")

    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        sys.modules["wandb"].log({f"final_eval/{k}": v for k, v in results.items()
                                  if isinstance(v, (int, float))})


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIC ROUTE  (--classic flag)
# ══════════════════════════════════════════════════════════════════════════════
if _is_classic:
    import os as _os
    import sys as _sys
    import random as _random
    import numpy as np
    import torch

    # Register classic envs by importing the classic package
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _classic_dir = os.path.join(
        _root, "source", "contractionRL", "contractionRL", "tasks", "direct",
    )
    if _classic_dir not in sys.path:
        sys.path.insert(0, _classic_dir)
    import contractionRL.tasks.direct.classic  # noqa: F401 — registers gymnasium envs (e.g. Car-v0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Config loading ────────────────────────────────────────────────────── #
    def _load_cfg(entry_point_key: str, custom_path: str | None = None) -> dict:
        if custom_path:
            with open(custom_path) as f:
                return yaml.safe_load(f)
        spec = gym.spec(args_cli.task)
        kwargs = spec.kwargs or {}
        entry = kwargs.get(entry_point_key)
        if entry is None:
            raise ValueError(
                f"No '{entry_point_key}' registered for {args_cli.task}. "
                f"Available: {list(kwargs.keys())}"
            )
        pkg, fname = entry.split(":")
        import importlib
        pkg_obj = importlib.import_module(pkg)
        cfg_path = os.path.join(os.path.dirname(pkg_obj.__file__), fname)
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    entry_key = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
    agent_cfg = _load_cfg(entry_key, args_cli.cfg)
    agent_cfg["seed"] = seed

    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    if args_cli.lr is not None:
        agent_cfg["agent"]["learning_rate"] = args_cli.lr
    # Analytical dynamics is OFF by default (every contraction config learns a
    # NeuralDynamics); --use_analytical_dynamics (or legacy --analytical dynamics)
    # switches to the env's exact get_f_and_B. Classic envs only.
    if args_cli.analytical == "dynamics" or args_cli.use_analytical_dynamics:
        agent_cfg["agent"]["use_analytical_dynamics"] = True

    _run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "classic", algorithm, _run_ts)
    os.makedirs(log_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(log_dir)
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # W&B
    if not args_cli.no_wandb:
        import skrl.utils.tensorboard as _skrl_tb
        import wandb as _wandb

        agent_cfg["agent"]["experiment"]["wandb"] = ("WANDB_SWEEP_ID" not in os.environ)
        _wkw = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
        _wkw["project"] = args_cli.wandb_project
        _wkw["sync_tensorboard"] = False
        # Force console capture even when stdout isn't a tty — sweep agents are
        # launched backgrounded with stdout/stderr redirected to a logfile
        # (search/run_c3m_sweeps.sh: `wandb agent ... > logfile 2>&1 &`), which
        # is exactly the case where wandb's tty auto-detection for the Logs tab
        # can silently fail to capture anything. "wrap" forces it regardless.
        _wkw["settings"] = _wandb.Settings(console="wrap")
        # Consistent run name: CLI override > YAML-provided name > deterministic
        # default that matches the local log directory (logs/classic/<algo>/<ts>).
        _wkw["name"] = args_cli.wandb_run_name or _wkw.get("name") or f"classic_{algorithm}_{_run_ts}"

        # If a sweep is running, init early to retrieve hyperparams and inject into agent_cfg
        if "WANDB_SWEEP_ID" in os.environ:
            if _wandb.run is None:
                _wandb.init(
                    project=_wkw["project"], name=_wkw.get("name"), sync_tensorboard=False,
                    settings=_wkw["settings"],
                )
            for k, v in _wandb.config.items():
                keys = k.split('.')
                curr = agent_cfg
                for key in keys[:-1]:
                    if key not in curr:
                        curr[key] = {}
                    curr = curr[key]
                curr[keys[-1]] = v

        _orig_add_scalar = _skrl_tb.SummaryWriter.add_scalar
        _scalar_metric_defined = [False]

        def _wandb_add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
            _orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
            if _wandb.run is not None:
                # Custom step metric instead of wandb's internal step counter —
                # see the Isaac-route patch below for why (step-less media logs
                # advance the counter and get later explicit-step scalars dropped).
                if not _scalar_metric_defined[0]:
                    _wandb.define_metric("global_step")
                    _wandb.define_metric("*", step_metric="global_step")
                    _scalar_metric_defined[0] = True
                _wandb.log({tag: value, "global_step": timestep})

        _skrl_tb.SummaryWriter.add_scalar = _wandb_add_scalar
    else:
        # --no_wandb: override the YAML default (wandb: true) so the skrl agent
        # does not call wandb.init() during agent.init().
        agent_cfg["agent"].setdefault("experiment", {})["wandb"] = False

    if algorithm in _CONTRACTION_ALGOS:
        # Contraction algorithms use ContractionRunner
        _src_dir = os.path.join(_root, "source", "contractionRL")
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)

        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        from contractionRL.runners import ContractionRunner

        num_envs = args_cli.num_envs or 4
        vec_env = SyncVectorEnv([lambda: gym.make(args_cli.task)] * num_envs)
        # vec_env.device = "cpu"  # REMOVED: This was causing C3M to run its heavy batch gradients on the CPU!
        env = wrap_env(vec_env, wrapper="gymnasium")

        runner = ContractionRunner(env, agent_cfg, task_id=args_cli.task, num_envs=num_envs, is_classic=True)
        if args_cli.checkpoint:
            runner.load(args_cli.checkpoint)
        runner.run()
        env.close()

        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner)

        if not args_cli.no_wandb and 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()

    else:
        # PPO / SAC use the built-in skrl Runner
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        from contractionRL.agents.skrl.runner import CLActorRunner as Runner

        _a = agent_cfg["agent"]
        _use_state = _a.pop("use_state_norm", False)  # OFF by default (see agent configs / c2rl.py docstring)
        _use_value = _a.pop("use_value_norm", True)

        num_envs = args_cli.num_envs or 4
        vec_env = SyncVectorEnv([lambda: gym.make(args_cli.task)] * num_envs)
        env = wrap_env(vec_env, wrapper="gymnasium")

        # Standalone PPO/SAC build models purely from agent_cfg (no env access) —
        # inject angle_idx (e.g. yaw at [2]) from the underlying classic env so
        # every network embeds it continuously (see runner.py's _gaussian_factory
        # / _deterministic_factory). No-op for envs with no wrapping angle.
        _first_env = vec_env.envs[0]
        _angle_idx = list(getattr(getattr(_first_env, "unwrapped", _first_env), "angle_idx", []) or [])
        _inject_angle_idx(agent_cfg, _angle_idx)

        if _use_state:
            # [x, xref, uref] path-tracking layout (obs_dim = 2*x_dim + u_dim,
            # same detection as runner.py's "mlp"/"mlp-squashed" backbones) needs
            # the masked scaler — see c2rl.py's module docstring / preprocessors.py
            # for why normalizing uref or angle_idx columns there is a
            # correctness bug, not just a style choice. Flat (e.g. vel-tracking)
            # layouts have no residual/angle-embedding structure, so the stock
            # full-vector scaler is fine there.
            _u_dim = int(env.action_space.shape[0])
            _remainder = int(env.observation_space.shape[0]) - _u_dim
            if _remainder > 0 and _remainder % 2 == 0:
                from contractionRL.agents.skrl.preprocessors import PathTrackingObservationScaler
                _a["state_preprocessor"] = PathTrackingObservationScaler
                _a["state_preprocessor_kwargs"] = {
                    "x_dim": _remainder // 2, "u_dim": _u_dim, "angle_idx": _angle_idx,
                }
            else:
                _a["state_preprocessor"] = "RunningStandardScaler"
                _a["state_preprocessor_kwargs"] = None
        if _use_value and algorithm == "ppo":
            _a["value_preprocessor"] = "RunningStandardScaler"
            _a["value_preprocessor_kwargs"] = None
        _a.pop("anneal_stddev", None)   # no longer a PPO_CFG field; handled below
        _a.pop("anneal_log_std", None)  # legacy alias, superseded by std_dev_annealing
        _std_dev_annealing = _a.pop("std_dev_annealing", False)
        _std_dev_annealing_kwargs = _a.pop("std_dev_annealing_kwargs", None)

        runner = Runner(env, agent_cfg)
        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging as _patch_kl_logging,
            patch_ppo_std_annealing as _patch_ppo_std_annealing,
            patch_sac_entropy_clamp as _patch_sac_entropy_clamp,
        )
        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        _patch_ppo_std_annealing(runner.agent, _std_dev_annealing, _std_dev_annealing_kwargs)
        if args_cli.checkpoint:
            runner.agent.load(args_cli.checkpoint)
        runner.run()
        env.close()

        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner)

        if not args_cli.no_wandb and 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()


# ══════════════════════════════════════════════════════════════════════════════
# ISAAC SIM ROUTE  (default)
# ══════════════════════════════════════════════════════════════════════════════
else:
    import time

    import skrl
    from packaging import version

    SKRL_VERSION = "2.0.0"
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        sys.exit(1)

    if args_cli.ml_framework.startswith("torch"):
        from contractionRL.agents.skrl.runner import CLActorRunner as Runner
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import contractionRL.tasks  # noqa: F401

    if args_cli.agent is None:
        agent_cfg_entry_point = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
        # Algorithm-aware num_envs defaults: off-policy SAC-based algorithms
        # need far fewer parallel envs (large replay buffer >> many envs),
        # while on-policy PPO-based algorithms benefit from massively parallel
        # envs.  The user can always override with --num_envs.
        _SAC_ALGOS = {"sac", "c2rl-sac", "c2rl_sac"}
        _DEFAULT_NUM_ENVS_SAC = 64
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        elif algorithm.lower() in _SAC_ALGOS:
            env_cfg.scene.num_envs = _DEFAULT_NUM_ENVS_SAC
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training is not supported on CPU.")
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        if args_cli.num_timesteps is not None:
            agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
        agent_cfg["trainer"]["close_environment_at_exit"] = False

        # HP overrides
        a = agent_cfg["agent"]
        if args_cli.sac_lr is not None:             a["learning_rate"]          = args_cli.sac_lr
        if args_cli.sac_batch_size is not None:      a["batch_size"]             = args_cli.sac_batch_size
        if args_cli.sac_discount is not None:        a["discount_factor"]        = args_cli.sac_discount
        if args_cli.sac_polyak is not None:          a["polyak"]                 = args_cli.sac_polyak
        if args_cli.sac_gradient_steps is not None:  a["gradient_steps"]         = args_cli.sac_gradient_steps
        if args_cli.sac_entropy is not None:         a["initial_entropy_value"]  = args_cli.sac_entropy
        if args_cli.sac_memory_size is not None:     agent_cfg["memory"]["memory_size"] = args_cli.sac_memory_size
        if args_cli.ppo_lr is not None:              a["learning_rate"]    = args_cli.ppo_lr
        if args_cli.ppo_rollouts is not None:        a["rollouts"]         = args_cli.ppo_rollouts
        if args_cli.ppo_learning_epochs is not None: a["learning_epochs"]  = args_cli.ppo_learning_epochs
        if args_cli.ppo_mini_batches is not None:    a["mini_batches"]     = args_cli.ppo_mini_batches
        if args_cli.ppo_discount is not None:        a["discount_factor"]  = args_cli.ppo_discount
        if args_cli.ppo_lambda is not None:          
            a["lambda"] = args_cli.ppo_lambda
            a["gae_lambda"] = args_cli.ppo_lambda
        if args_cli.ppo_ratio_clip is not None:      a["ratio_clip"]       = args_cli.ppo_ratio_clip
        if args_cli.ppo_entropy_scale is not None:   a["entropy_loss_scale"] = args_cli.ppo_entropy_scale
        if args_cli.ppo_kl_threshold is not None:    a["kl_threshold"]     = args_cli.ppo_kl_threshold
        if args_cli.ppo_use_state_norm is not None:  a["use_state_norm"]   = (args_cli.ppo_use_state_norm.lower() == 'true')
        if args_cli.ppo_use_value_norm is not None:  a["use_value_norm"]   = (args_cli.ppo_use_value_norm.lower() == 'true')

        if args_cli.ppo_activations is not None:
            models_cfg = agent_cfg.get("models", {})
            for model_type in ["policy", "value"]:
                if model_type in models_cfg:
                    for layer in models_cfg[model_type].get("network", []):
                        if "activations" in layer:
                            layer["activations"] = args_cli.ppo_activations
                            
        if args_cli.ppo_network_arch is not None:
            arch_str = args_cli.ppo_network_arch.replace("[", "").replace("]", "")
            layers = [int(x.strip()) for x in arch_str.split(",")]
            models_cfg = agent_cfg.get("models", {})
            for model_type in ["policy", "value"]:
                if model_type in models_cfg:
                    for layer in models_cfg[model_type].get("network", []):
                        if "layers" in layer:
                            layer["layers"] = layers

        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]

        log_root_path = os.path.abspath(
            os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        )
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # W&B
        _wandb_video_thread = None
        _wandb_stop_event = None
        if not args_cli.no_wandb:
            import glob as _glob
            import re as _re
            import threading as _threading
            import skrl.utils.tensorboard as _skrl_tb
            import wandb as _wandb

            agent_cfg["agent"]["experiment"]["wandb"] = ("WANDB_SWEEP_ID" not in os.environ)
            agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})["sync_tensorboard"] = False
            _orig_add_scalar = _skrl_tb.SummaryWriter.add_scalar
            _scalar_metric_defined = [False]

            def _wandb_add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
                _orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
                if _wandb.run is not None:
                    # Log against a custom step metric instead of wandb's internal
                    # step counter: any wandb.log() without step= (e.g. video/media
                    # uploads) advances the internal counter, after which scalars
                    # logged with an explicit smaller step= are silently dropped
                    # ("steps must be monotonically increasing" warning). A step
                    # *metric* has no monotonicity requirement.
                    if not _scalar_metric_defined[0]:
                        _wandb.define_metric("global_step")
                        _wandb.define_metric("*", step_metric="global_step")
                        _scalar_metric_defined[0] = True
                    _wandb.log({tag: value, "global_step": timestep})

            _skrl_tb.SummaryWriter.add_scalar = _wandb_add_scalar

            if args_cli.video:
                _video_dir = os.path.join(log_dir, "videos", "train")
                _uploaded_videos: set = set()
                _wandb_stop_event = _threading.Event()
                _video_metric_defined = [False]

                def _upload_pending_videos(step: int | None = None) -> None:
                    for mp4 in sorted(_glob.glob(os.path.join(_video_dir, "*.mp4"))):
                        if mp4 not in _uploaded_videos and _wandb.run is not None:
                            if not _video_metric_defined[0]:
                                # The video-watcher thread uploads asynchronously (polling every
                                # 30s) and can land after the main loop has already logged
                                # scalars at a later step — wandb rejects any step= that isn't
                                # monotonically increasing across ALL calls in the run. Give the
                                # video its own x-axis instead of the shared step counter, so an
                                # out-of-order upload is never rejected (https://wandb.me/define-metric).
                                _wandb.define_metric("train/video_step")
                                _wandb.define_metric("train/video", step_metric="train/video_step")
                                _video_metric_defined[0] = True
                            m = _re.search(r"step-(\d+)", os.path.basename(mp4))
                            log_step = int(m.group(1)) if m else step
                            try:
                                _wandb.log({"train/video": _wandb.Video(mp4, format="mp4"), "train/video_step": log_step})
                                _uploaded_videos.add(mp4)
                            except Exception as _e:
                                logger.warning(f"wandb video upload failed: {_e}")

                def _video_watcher() -> None:
                    while not _wandb_stop_event.is_set():
                        _upload_pending_videos()
                        _wandb_stop_event.wait(timeout=30)
                    _upload_pending_videos()

                _wandb_video_thread = _threading.Thread(target=_video_watcher, daemon=True)
                _wandb_video_thread.start()

        _wkw = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
        if args_cli.wandb_project is not None:
            _wkw["project"] = args_cli.wandb_project
        # Consistent run name: CLI override > YAML-provided name > deterministic
        # default that matches the local experiment_name (tensorboard dir), so
        # every W&B run can be correlated with its local log directory.
        _wkw["name"] = args_cli.wandb_run_name or _wkw.get("name") or agent_cfg["agent"]["experiment"]["experiment_name"]

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.log_dir = log_dir

        if hasattr(env_cfg, "vel_cmd"):
            vc = env_cfg.vel_cmd
            print(
                f"[INFO] Velocity command distribution:\n"
                f"         vx       ~ U{vc.vx_range}\n"
                f"         vy       ~ U{vc.vy_range}\n"
                f"         yaw amp  ~ U{vc.yaw_A_range} rad/s\n"
                f"         yaw freq ~ U{vc.yaw_omega_range} rad/s\n"
                f"         yaw phi  ~ U[0, 2π]"
            )

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        if args_cli.debug_vis or args_cli.video:
            # Recording a video implies you want to see the tracked-vs-target velocity
            # arrows in it, not just the raw robot — enable debug markers automatically
            # instead of requiring a separate --debug_vis flag on top of --video.
            env.unwrapped.set_debug_vis(True)

        if args_cli.video:
            video_len = args_cli.video_length if args_cli.video_length > 0 else getattr(env.unwrapped, "max_episode_length", 200)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": video_len,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        _isaac_env = env  # save reference for get_physical_state() during ref-traj generation
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # angle_idx (e.g. yaw at [2] on quadruped/humanoid path-tracking) is an
        # ENV attribute, not a cfg field — inject it into agent_cfg["models"] so
        # standalone PPO/SAC's _gaussian_factory/_deterministic_factory (which
        # only see per-block yaml kwargs) embed it too. ContractionRunner
        # (C3M/LQR/SDLQR/C2RL) reads it directly off the env and needs no
        # injection. _isaac_env is the pre-SkrlVecEnvWrapper raw env saved above.
        _angle_idx = list(getattr(_isaac_env.unwrapped, "angle_idx", []) or [])
        _inject_angle_idx(agent_cfg, _angle_idx)

        _a = agent_cfg["agent"]
        _alg = _a.get("class", "").lower()
        _use_state = _a.pop("use_state_norm", False)  # OFF by default (see agent configs / c2rl.py docstring)
        _use_value = _a.pop("use_value_norm", True)
        if _use_state:
            # [x, xref, uref] path-tracking layout needs the masked scaler (see
            # c2rl.py's module docstring / preprocessors.py) — normalizing uref
            # or angle_idx columns there is a correctness bug, not just a style
            # choice. Flat (e.g. vel-tracking) layouts have no residual/angle-
            # embedding structure, so the stock full-vector scaler is fine there.
            _u_dim = int(env.action_space.shape[0])
            _remainder = int(env.observation_space.shape[0]) - _u_dim
            if _remainder > 0 and _remainder % 2 == 0:
                from contractionRL.agents.skrl.preprocessors import PathTrackingObservationScaler
                _obs_preproc_cls = PathTrackingObservationScaler
                _obs_preproc_kwargs = {
                    "x_dim": _remainder // 2, "u_dim": _u_dim, "angle_idx": _angle_idx,
                }
            else:
                _obs_preproc_cls = "RunningStandardScaler"
                _obs_preproc_kwargs = None
            _a["observation_preprocessor"] = _obs_preproc_cls
            _a["observation_preprocessor_kwargs"] = (
                dict(_obs_preproc_kwargs) if _obs_preproc_kwargs is not None else None
            )
            if _alg == "ppo":
                _a["state_preprocessor"] = _obs_preproc_cls
                _a["state_preprocessor_kwargs"] = (
                    dict(_obs_preproc_kwargs) if _obs_preproc_kwargs is not None else None
                )
        else:
            for _k in ("state_preprocessor", "state_preprocessor_kwargs",
                       "observation_preprocessor", "observation_preprocessor_kwargs"):
                _a.pop(_k, None)
        if _use_value and _alg == "ppo":
            _a["value_preprocessor"] = "RunningStandardScaler"
            _a["value_preprocessor_kwargs"] = None
        else:
            _a.pop("value_preprocessor", None)
            _a.pop("value_preprocessor_kwargs", None)

        _a.pop("anneal_stddev", None)   # no longer a PPO_CFG field; handled below
        _a.pop("anneal_log_std", None)
        _std_dev_annealing = _a.pop("std_dev_annealing", False)
        _std_dev_annealing_kwargs = _a.pop("std_dev_annealing_kwargs", None)

        # Auto-enable stddev annealing when policy uses the CLActor backbone
        # ("control", with "contraction" kept as a backward-compatible alias)
        _is_contraction = (
            agent_cfg.get("models", {}).get("policy", {}).get("backbone") in ("control", "contraction")
        )

        if _alg in _CONTRACTION_ALGOS:
            from contractionRL.runners import ContractionRunner
            # Default: learn a NeuralDynamics (pretrain + online). Passing
            # --use_analytical_dynamics for an Isaac task deliberately trips the
            # runner's guard below (no analytical dynamics exist for Isaac envs).
            if args_cli.use_analytical_dynamics:
                agent_cfg["agent"]["use_analytical_dynamics"] = True
            runner = ContractionRunner(env, agent_cfg, is_classic=False)
        else:
            runner = Runner(env, agent_cfg)

        from contractionRL.agents.skrl.agent_patches import (
            patch_kl_logging as _patch_kl_logging,
            patch_ppo_std_annealing as _patch_ppo_std_annealing,
            patch_sac_entropy_clamp as _patch_sac_entropy_clamp,
        )
        # No-ops for C2RL's outer agent (it has no .policy/.scheduler/.entropy_optimizer
        # of its own) — C2RLAgent applies these directly to its con_agent/opt_agent
        # sub-agents internally (see c2rl.py), which is where they actually matter.
        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        _patch_ppo_std_annealing(runner.agent, _std_dev_annealing, _std_dev_annealing_kwargs)

        if _is_contraction and hasattr(runner.agent, "policy") and hasattr(runner.agent.policy, "cl_actor"):
            _orig_post = runner.agent.post_interaction

            def _annealed_post(*, timestep: int, timesteps: int) -> None:
                runner.agent.policy.cl_actor.anneal_stddev(timestep / max(1, timesteps))
                _orig_post(timestep=timestep, timesteps=timesteps)

            runner.agent.post_interaction = _annealed_post

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        runner.run()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

        if _wandb_stop_event is not None:
            _wandb_stop_event.set()
        if _wandb_video_thread is not None:
            _wandb_video_thread.join(timeout=120)

        # Best-model evaluation (CAC-dev-style: reward/AUC/contraction-rate/
        # overshoot with 95% CI) applies to PATH-TRACKING envs — that's where
        # a genuine reference-trajectory tracking error is defined and where
        # C3M/LQR/SD-LQR/C2RL's contraction analysis is meaningful. It also
        # runs for VelTracking (which exposes the same get_tracking_error()
        # duck-type against a velocity command instead of a trajectory) since
        # that's used as this quality gate before ref-traj generation.
        # _evaluate_best_model no-ops with a SKIPPED message for any env that
        # doesn't implement get_tracking_error(), so it's safe to always call.
        _evaluate_best_model(
            task=args_cli.task,
            runner=runner,
            isaac_env=_isaac_env,
            skrl_env=env,
            env_cfg=env_cfg,
        )

        if "VelTracking" in (args_cli.task or ""):
            _generate_ref_trajs(
                task=args_cli.task,
                runner=runner,
                isaac_env=_isaac_env,
                skrl_env=env,
                env_cfg=env_cfg,
            )

        env.close()
        
        if not args_cli.no_wandb and 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()

    if __name__ == "__main__":
        main()
        if not _is_classic:
            simulation_app.close()
            import os
            os._exit(0)

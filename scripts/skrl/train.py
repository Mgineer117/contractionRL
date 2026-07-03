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
    help="Algorithm: ppo | sac | c3m | lqr | sdlqr | temp | AMP | DDPG | TD3 | …"
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None,
                    help="Total training timesteps.")
parser.add_argument("--analytical", type=str, default="",
                    help="Pass 'dynamics' to use analytical dynamics (C3M/LQR).")

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
parser.add_argument("--ref_num_trajs", type=int, default=2000,
                    help="Number of reference trajectories to collect after vel-tracking training.")
parser.add_argument("--min_ref_quality", type=float, default=None,
                    help="Minimum mean episode reward before generating ref trajs. 0 to skip check.")

if not _is_classic:
    AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if not _is_classic and args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

if not _is_classic:
    args_cli.kit_args = (args_cli.kit_args or "") + " --/app/hangDetector/enabled=false"
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
_CONTRACTION_ALGOS = {"c3m", "lqr", "sdlqr", "temp"}

seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}
_MIN_QUALITY_REWARD = {"quadruped": 750.0, "humanoid": 875.0, "manipulator": 500.0}


def _generate_ref_trajs(*, task, runner, isaac_env, skrl_env, env_cfg):
    import numpy as np
    import torch

    robot = next((name for prefix, name in _VEL_TASK_TO_ROBOT.items() if task.startswith(prefix)), None)
    if robot is None:
        print(f"[RefTraj] No robot mapping for task '{task}'; skipping.")
        return

    out_dir = os.path.join(_ROOT, "logs", robot)
    out_path = os.path.join(out_dir, "ref_trajs.npz")
    min_reward = args_cli.min_ref_quality if args_cli.min_ref_quality is not None \
        else _MIN_QUALITY_REWARD.get(robot, 750.0)

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

    T = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    # Quality gate: 1 full episode across all parallel environments
    if min_reward > 0:
        print(f"\n[RefTraj] Evaluating quality (threshold: mean total reward >= {min_reward}) …")
        def _get_obs(o):
            return o["policy"] if isinstance(o, dict) else o

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

    # Collect trajectories
    num_trajs = args_cli.ref_num_trajs
    print(f"[RefTraj] Collecting {num_trajs} trajectories → {out_path}")
    unwrapped = isaac_env.unwrapped
    num_envs = skrl_env.num_envs
    import tqdm
    all_states, all_actions, all_pos = [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)
    
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
    
    pbar = tqdm.tqdm(total=num_trajs, desc="[RefTraj] Collecting")

    while len(all_states) < num_trajs:
        with torch.no_grad():
            actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
        state_tensor = unwrapped.get_physical_state()
        
        # Record state and action for envs that are still within T steps
        valid_mask = step_counts < T
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        
        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        ep_actions[valid_indices, step_counts[valid_indices]] = actions[valid_indices].float()
        if hasattr(unwrapped, "_robot"):
            ep_pos[valid_indices, step_counts[valid_indices]] = unwrapped._robot.data.root_pos_w[valid_indices].float()
        
        step_counts[valid_indices] += 1
        
        obs_dict, _, terminated, truncated, _ = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)
        
        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # Accept trajectories that survived at least half the max length.
            # This handles policies that fall slightly early but pass the quality gate,
            # as well as off-by-one errors with Isaac Gym's max_episode_length.
            success_mask = step_counts[done_indices] >= (T // 2)
            success_indices = done_indices[success_mask]
            
            if len(success_indices) > 0:
                # Pad any missing steps with the final valid state to ensure x_dot is stable
                for i in success_indices:
                    length = step_counts[i].item()
                    if length < T and length > 0:
                        ep_states[i, length:] = ep_states[i, length - 1].clone()
                        ep_actions[i, length:] = ep_actions[i, length - 1].clone()
                        ep_pos[i, length:] = ep_pos[i, length - 1].clone()
                        
                s_np = ep_states[success_indices].cpu().numpy()
                a_np = ep_actions[success_indices].cpu().numpy()
                p_np = ep_pos[success_indices].cpu().numpy()
                for i in range(len(success_indices)):
                    if len(all_states) >= num_trajs:
                        break
                    all_states.append(s_np[i])
                    all_actions.append(a_np[i])
                    all_pos.append(p_np[i])
                    pbar.update(1)
            
            # Reset the step counts for all finished environments
            step_counts[done_indices] = 0

    pbar.close()
    states_arr = np.stack(all_states[:num_trajs]).astype(np.float32)
    actions_arr = np.stack(all_actions[:num_trajs]).astype(np.float32)
    pos_arr = np.stack(all_pos[:num_trajs]).astype(np.float32)
    
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print(f"\n[RefTraj] Saved → {out_path}  states{states_arr.shape}  actions{actions_arr.shape}")
    
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
    x_dot_arr = np.zeros_like(states_arr)
    for i in range(2, states_arr.shape[1] - 2):
        x_dot_arr[:, i] = (-states_arr[:, i + 2] + 8 * states_arr[:, i + 1] - 8 * states_arr[:, i - 1] + states_arr[:, i - 2]) / (12 * dt)
    # Forward/backward differences for boundaries
    x_dot_arr[:, 0] = (-3 * states_arr[:, 0] + 4 * states_arr[:, 1] - states_arr[:, 2]) / (2 * dt)
    x_dot_arr[:, 1] = (-3 * states_arr[:, 1] + 4 * states_arr[:, 2] - states_arr[:, 3]) / (2 * dt)
    x_dot_arr[:, -2] = (3 * states_arr[:, -2] - 4 * states_arr[:, -3] + states_arr[:, -4]) / (2 * dt)
    x_dot_arr[:, -1] = (3 * states_arr[:, -1] - 4 * states_arr[:, -2] + states_arr[:, -3]) / (2 * dt)
    
    # Filter out any episodes that contain NaNs
    nan_mask = np.isnan(states_arr).any(axis=(1, 2)) | np.isnan(actions_arr).any(axis=(1, 2)) | np.isnan(x_dot_arr).any(axis=(1, 2))
    if nan_mask.any():
        num_nans = nan_mask.sum()
        print(f"[RefTraj] WARNING: Found NaNs in {num_nans} episodes! Filtering them out before saving...")
        valid_mask = ~nan_mask
        states_arr = states_arr[valid_mask]
        actions_arr = actions_arr[valid_mask]
        x_dot_arr = x_dot_arr[valid_mask]
    
    dyn_path = os.path.join(out_dir, "dynamics_data.npz")
    np.savez_compressed(dyn_path, x=states_arr, u=actions_arr, x_dot=x_dot_arr)
    print(f"[RefTraj] Saved dynamics  → {dyn_path}")
    print(f"       x      shape: {states_arr.shape}")
    print(f"       u      shape: {actions_arr.shape}")
    print(f"       x_dot  shape: {x_dot_arr.shape}")


def _patch_kl_logging(agent) -> None:
    """Log per-epoch approximate KL divergence to 'Policy / KL divergence'.

    skrl's PPO computes KL every epoch to drive KLAdaptiveLR but never records
    it, so early-stop events (kl_threshold) are invisible in tensorboard/wandb
    even though they silently truncate — and thus deflate — the averaged
    Loss/Policy loss, Loss/Value loss, Loss/Entropy loss for that update
    (skrl divides by the full learning_epochs*mini_batches regardless of how
    many minibatches actually ran before the break). No-ops for agents without
    a KLAdaptiveLR scheduler (SAC, contraction agents, PPO with scheduler=null).
    """
    import skrl.resources.schedulers.torch as _sched

    scheduler = getattr(agent, "scheduler", None)
    if not isinstance(scheduler, _sched.KLAdaptiveLR):
        return

    _orig_step = scheduler.step

    def _step(kl=None, *, epoch=None):
        if kl is not None:
            agent.track_data("Policy / KL divergence", float(kl))
        _orig_step(kl, epoch=epoch)

    scheduler.step = _step


def _patch_sac_entropy_clamp(agent, min_log_alpha: float = -5.0, max_log_alpha: float = 2.0) -> None:
    """Clamp log_entropy_coefficient in-place after every entropy optimizer step.

    skrl's SAC applies grad_norm_clip to the policy and critic optimizers but
    NOT to entropy_optimizer, and _entropy_coefficient = exp(log_entropy_coefficient)
    is exponentiated with no bound. A noisy/undertrained critic can push this
    single scalar's gradient large, and exponentiation turns even a moderate
    excursion into a runaway entropy coefficient that then dominates both the
    critic target and the policy loss — a textbook SAC divergence mechanism.
    Bounds exp(log_alpha) to roughly [0.0067, 7.39], mirroring the clip_log_std
    bounds skrl already applies to GaussianMixin policies elsewhere. No-op for
    agents without learn_entropy (PPO, contraction agents, SAC with learn_entropy=False).
    """
    import torch

    entropy_optimizer = getattr(agent, "entropy_optimizer", None)
    log_alpha = getattr(agent, "log_entropy_coefficient", None)
    if entropy_optimizer is None or log_alpha is None:
        return

    _orig_step = entropy_optimizer.step

    def _step(*args, **kwargs):
        result = _orig_step(*args, **kwargs)
        with torch.no_grad():
            log_alpha.clamp_(min_log_alpha, max_log_alpha)
        return result

    entropy_optimizer.step = _step

def _patch_ppo_std_annealing(agent, std_dev_annealing: bool) -> None:
    """Adds manual standard deviation annealing to SKRL's PPO policy.
    
    If `std_dev_annealing` is True, this disables the entropy loss entirely
    (setting entropy_loss_scale to 0.0) and linearly anneals the policy's
    log_std_parameter from its initial value down to its configured minimum
    (min_log_std) over the total training timesteps.
    """
    if not std_dev_annealing:
        return

    # Ignore entropy
    if hasattr(agent, "_cfg") and isinstance(agent._cfg, dict):
        agent._cfg["entropy_loss_scale"] = 0.0
    if hasattr(agent, "cfg"):
        if isinstance(agent.cfg, dict):
            agent.cfg["entropy_loss_scale"] = 0.0
        else:
            setattr(agent.cfg, "entropy_loss_scale", 0.0)

    if not hasattr(agent, "policy") or not hasattr(agent.policy, "log_std_parameter"):
        return

    # Disable gradients on log_std_parameter because we update it manually
    agent.policy.log_std_parameter.requires_grad_(False)

    initial_log_std = agent.policy.log_std_parameter.mean().item()
    final_log_std = getattr(agent.policy, "_g_min_log_std", -2.0)

    _orig_post = agent.post_interaction

    def _annealed_post(*, timestep: int, timesteps: int) -> None:
        progress = min(1.0, max(0.0, timestep / max(1, timesteps)))
        current_log_std = initial_log_std + progress * (final_log_std - initial_log_std)
        agent.policy.log_std_parameter.data.fill_(current_log_std)
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _annealed_post

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

    entry_key = f"skrl_{algorithm}_cfg_entry_point"
    agent_cfg = _load_cfg(entry_key, args_cli.cfg)
    agent_cfg["seed"] = seed

    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    if args_cli.lr is not None:
        agent_cfg["agent"]["learning_rate"] = args_cli.lr
    if args_cli.analytical == "dynamics":
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
        # Consistent run name: CLI override > YAML-provided name > deterministic
        # default that matches the local log directory (logs/classic/<algo>/<ts>).
        _wkw["name"] = args_cli.wandb_run_name or _wkw.get("name") or f"classic_{algorithm}_{_run_ts}"

        # If a sweep is running, init early to retrieve hyperparams and inject into agent_cfg
        if "WANDB_SWEEP_ID" in os.environ:
            if _wandb.run is None:
                _wandb.init(project=_wkw["project"], name=_wkw["name"])
            for k, v in _wandb.config.items():
                keys = k.split('.')
                curr = agent_cfg
                for key in keys[:-1]:
                    if key not in curr:
                        curr[key] = {}
                    curr = curr[key]
                curr[keys[-1]] = v

        _orig_add_scalar = _skrl_tb.SummaryWriter.add_scalar

        def _wandb_add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
            _orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
            if _wandb.run is not None:
                _wandb.log({tag: value}, step=timestep)

        _skrl_tb.SummaryWriter.add_scalar = _wandb_add_scalar

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
        
        if not args_cli.no_wandb and 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()

    else:
        # PPO / SAC use the built-in skrl Runner
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        from contractionRL.agents.skrl.runner import CLActorRunner as Runner

        _a = agent_cfg["agent"]
        _use_state = _a.pop("use_state_norm", True)
        _use_value = _a.pop("use_value_norm", True)
        if _use_state:
            _a["state_preprocessor"] = "RunningStandardScaler"
            _a["state_preprocessor_kwargs"] = None
        if _use_value and algorithm == "ppo":
            _a["value_preprocessor"] = "RunningStandardScaler"
            _a["value_preprocessor_kwargs"] = None

        num_envs = args_cli.num_envs or 4
        vec_env = SyncVectorEnv([lambda: gym.make(args_cli.task)] * num_envs)
        env = wrap_env(vec_env, wrapper="gymnasium")

        runner = Runner(env, agent_cfg)
        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        if args_cli.checkpoint:
            runner.agent.load(args_cli.checkpoint)
        runner.run()
        env.close()
        
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
        agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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

            def _wandb_add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
                _orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
                if _wandb.run is not None:
                    _wandb.log({tag: value}, step=timestep)

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

        _a = agent_cfg["agent"]
        _alg = _a.get("class", "").lower()
        _use_state = _a.pop("use_state_norm", True)
        _use_value = _a.pop("use_value_norm", True)
        if _use_state:
            _a["observation_preprocessor"] = "RunningStandardScaler"
            _a["observation_preprocessor_kwargs"] = None
            if _alg == "ppo":
                _a["state_preprocessor"] = "RunningStandardScaler"
                _a["state_preprocessor_kwargs"] = None
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

        # Auto-enable stddev annealing when policy uses the contraction backbone
        _is_contraction = (
            agent_cfg.get("models", {}).get("policy", {}).get("backbone") == "contraction"
        )

        if _alg in _CONTRACTION_ALGOS:
            from contractionRL.runners import ContractionRunner
            runner = ContractionRunner(env, agent_cfg, is_classic=False)
        else:
            runner = Runner(env, agent_cfg)

        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        _patch_ppo_std_annealing(runner.agent, _std_dev_annealing)

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
        simulation_app.close()

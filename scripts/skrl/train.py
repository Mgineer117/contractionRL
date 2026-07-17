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
_pre.add_argument("--task", type=str, default="")
_pre_args, _ = _pre.parse_known_args()
_is_classic = _pre_args.classic or _pre_args.task.startswith("classic")

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
parser.add_argument("--use_empirical_dynamics", "--use-empirical-dynamics",
                    action="store_true", default=False,
                    help="Use a learned NeuralDynamics model instead of the env's exact analytical get_f_and_B "
                         "(classic envs only). When NOT passed, C3M/C2RL use analytical dynamics.")
parser.add_argument("--eig_reshape", "--eig-reshape", type=float, default=None,
                    help="ABLATION (c2rl_ppo classic only): reshape the Mahalanobis reward's M "
                         "eigenvalue SPREAD to this target cond(M), keeping eigenvectors and "
                         "geometric-mean scale fixed — isolates conditioning from what the C1/C2 "
                         "fit converged to. See env_base.py's set_eig_reshape.")

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
from train_utils import _default_num_envs_classic, _inject_angle_idx, _max_step_reward, _generate_ref_trajs, _evaluate_classic_path_tracking, _evaluate_best_model

algorithm = args_cli.algorithm.lower()
# Bare "c2rl" (no -ppo/-sac suffix) defaults to the PPO variant, since it has
# no standalone (non-suffixed) config entry point registered.
if algorithm in ("c2rl",):
    algorithm = f"{algorithm}_ppo"
_CONTRACTION_ALGOS = {
    "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr",
    "c2rl-ppo", "c2rl-sac", "c2rl_ppo", "c2rl_sac",
}

# Algorithm-aware num_envs defaults (used when --num_envs is not given).
# SAC-based algorithms (and c3m/lqr/sdlqr, which sample from a large buffer the
# same way SAC does) need far fewer parallel envs; PPO-based algorithms are
# on-policy and benefit from massively parallel envs. Applies to both the
# classic gymnasium route and the Isaac Sim route.
_SAC_LIKE_ALGOS = {"sac", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024



seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}







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
    # --seed CLI arg wins; otherwise fall back to the yaml's own seed (NOT the
    # random.randint(...) module-level `seed` computed at line 164 before the
    # yaml was even loaded — using that unconditionally silently discarded
    # every config's `seed:` field and made "the same command" produce a
    # different random init/data-sampling trajectory on every invocation).
    # Mirrors the Isaac-env branch below (search "env_cfg.seed"), which
    # already got this right.
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent_cfg["seed"] = seed

    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    if args_cli.lr is not None:
        agent_cfg["agent"]["learning_rate"] = args_cli.lr
    # Classic contraction envs use the env's exact analytical get_f_and_B by
    # default (use_empirical_dynamics=False); pass --use_empirical_dynamics to
    # learn a NeuralDynamics instead. Classic envs only (Isaac forces empirical).
    if algorithm in ["c3m", "c2rl_ppo", "c2rl_sac", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"]:
        agent_cfg["agent"]["use_empirical_dynamics"] = args_cli.use_empirical_dynamics

    _run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "classic", algorithm, _run_ts)
    os.makedirs(log_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(log_dir)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["trainer"]["headless"] = True

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
        try:
            _wkw["settings"] = _wandb.Settings(console="wrap")
        except Exception:
            pass
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

        from contractionRL.runners import ContractionRunner
        
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else _default_num_envs_classic(algorithm)
        device = getattr(args_cli, "device", "cuda:0")
        
        env = gym.make(args_cli.task, num_envs=num_envs, device=device)
        if args_cli.eig_reshape is not None:
            if not hasattr(env.unwrapped, "set_eig_reshape"):
                raise SystemExit("--eig_reshape requires a classic env_base env (got "
                                 f"{type(env.unwrapped).__name__})")
            env.unwrapped.set_eig_reshape(args_cli.eig_reshape)
            print(f"[train] eig_reshape ACTIVE: Mahalanobis reward's M reshaped to "
                  f"cond(M) = {args_cli.eig_reshape:g} every step")
        from train_utils import BatchedGymnasiumWrapper
        env = BatchedGymnasiumWrapper(env)

        from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper
        env = StatManagerEnvWrapper(env)

        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(__file__))
        from wandb_plot_wrapper import WandbPlotWrapper
        env = WandbPlotWrapper(env, total_timesteps=agent_cfg["trainer"]["timesteps"])

        runner = ContractionRunner(env, agent_cfg, task_id=args_cli.task, num_envs=num_envs, is_classic=True)
        if args_cli.checkpoint:
            runner.load(args_cli.checkpoint)
        runner.run()
        env.close()

        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner, args_cli=args_cli, _is_classic=_is_classic)

        if not args_cli.no_wandb and 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()

    else:
        # PPO / SAC use the built-in skrl Runner
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        from contractionRL.agents.skrl.runner import CLActorRunner as Runner, CONTROL_BACKBONES

        _a = agent_cfg["agent"]
        _use_state = _a.pop("use_state_norm", False)  # OFF by default (see agent configs / c2rl.py docstring)
        _use_value = _a.pop("use_value_norm", True)

        num_envs = args_cli.num_envs if args_cli.num_envs is not None else _default_num_envs_classic(algorithm)
        device = getattr(args_cli, "device", "cuda:0")
        
        env = gym.make(args_cli.task, num_envs=num_envs, device=device)
        
        _isaac_env = env
        from train_utils import BatchedGymnasiumWrapper
        env = BatchedGymnasiumWrapper(env)

        from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper
        env = StatManagerEnvWrapper(env)
        
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(__file__))
        from wandb_plot_wrapper import WandbPlotWrapper
        env = WandbPlotWrapper(env, total_timesteps=agent_cfg["trainer"]["timesteps"])

        # Standalone PPO/SAC build models purely from agent_cfg (no env access) —
        # inject angle_idx (e.g. yaw at [2]) from the underlying classic env so
        # every network embeds it continuously (see runner.py's _gaussian_factory
        # / _deterministic_factory). No-op for envs with no wrapping angle.
        _angle_idx = list(getattr(_isaac_env.unwrapped, "angle_idx", []) or [])
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
        _a.pop("anneal_log_std", None)  # legacy alias, superseded by backbone-driven annealing
        _a.pop("std_dev_annealing", None)  # legacy on/off flag; now auto-derived from backbone below
        _std_dev_annealing_kwargs = _a.pop("std_dev_annealing_kwargs", None)
        _std_dev_annealing = (
            agent_cfg.get("models", {}).get("policy", {}).get("backbone") in CONTROL_BACKBONES
        )

        runner = Runner(env, agent_cfg)
        from contractionRL.agents.skrl.agent_patches import (
            patch_algo_namespace as _patch_algo_namespace,
            patch_kl_logging as _patch_kl_logging,
            patch_ppo_std_annealing as _patch_ppo_std_annealing,
            patch_sac_entropy_clamp as _patch_sac_entropy_clamp,
        )
        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        _patch_ppo_std_annealing(runner.agent, _std_dev_annealing, _std_dev_annealing_kwargs)
        _patch_algo_namespace(runner.agent, algorithm.upper())
        if args_cli.checkpoint:
            runner.agent.load(args_cli.checkpoint)
        runner.run()
        env.close()

        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner, args_cli=args_cli, _is_classic=_is_classic)

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

    from contractionRL.agents.skrl.runner import CONTROL_BACKBONES

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
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        elif algorithm.lower() in _SAC_LIKE_ALGOS:
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

        from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper
        env = StatManagerEnvWrapper(env)
        # Raw (un-plot-wrapped) reference for _evaluate_best_model/_generate_ref_trajs:
        # both toggle skrl_env._reset_once directly to force a real sim reset, which
        # is an attribute SET — WandbPlotWrapper.__getattr__ only intercepts GETs, so
        # setting it through the plot wrapper would shadow it on the wrapper instance
        # instead of the real IsaacLabWrapper, silently breaking that reset.
        _skrl_env = env
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(__file__))
        from wandb_plot_wrapper import WandbPlotWrapper
        # WandbPlotWrapper must wrap the SKRL-wrapped env (flat tensor obs +
        # .state()), not the raw Isaac env — the raw env's step() returns a dict
        # obs ({"policy": ...}), which crashes WandbPlotWrapper's trajectory
        # extraction (obs[i, :3] on a dict). Order matches the classic branch above.
        env = WandbPlotWrapper(env, total_timesteps=agent_cfg["trainer"]["timesteps"])

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
        _a.pop("std_dev_annealing", None)  # legacy on/off flag; now auto-derived from backbone below
        _std_dev_annealing_kwargs = _a.pop("std_dev_annealing_kwargs", None)

        # Auto-enable stddev annealing when policy uses the CLActor backbone
        # ("control", with "contraction" kept as a backward-compatible alias)
        _std_dev_annealing = (
            agent_cfg.get("models", {}).get("policy", {}).get("backbone") in CONTROL_BACKBONES
        )

        if _alg in _CONTRACTION_ALGOS:
            from contractionRL.runners import ContractionRunner
            # Isaac envs have no closed-form dynamics, so they ALWAYS learn a
            # NeuralDynamics (pretrain + online). Forcing use_empirical_dynamics=True
            # here also makes the runner's guard reject any config that tries to
            # request analytical dynamics (use_empirical_dynamics=False) for Isaac.
            agent_cfg["agent"]["use_empirical_dynamics"] = True
            runner = ContractionRunner(env, agent_cfg, is_classic=False)
        else:
            runner = Runner(env, agent_cfg)

        from contractionRL.agents.skrl.agent_patches import (
            patch_algo_namespace as _patch_algo_namespace,
            patch_kl_logging as _patch_kl_logging,
            patch_ppo_std_annealing as _patch_ppo_std_annealing,
            patch_sac_entropy_clamp as _patch_sac_entropy_clamp,
            patch_auc_checkpoint as _patch_auc_checkpoint,
        )
        # No-ops for C2RL's outer agent (it has no .policy/.scheduler/.entropy_optimizer
        # of its own) — C2RLAgent applies these directly to its con_agent/opt_agent
        # sub-agents internally (see c2rl.py), which is where they actually matter.
        _patch_kl_logging(runner.agent)
        _patch_sac_entropy_clamp(runner.agent)
        _patch_ppo_std_annealing(runner.agent, _std_dev_annealing, _std_dev_annealing_kwargs)
        if _alg not in _CONTRACTION_ALGOS:
            # C3M/C2RL/LQR/SD-LQR already namespace their own track_data() keys
            # ("Loss / C3M/...", "Loss / C2RL/...", "Con / "/"Opt / " wrapping);
            # this is standalone PPO/SAC's equivalent (see patch_algo_namespace).
            _patch_algo_namespace(runner.agent, _alg.upper())

        # All agents should use AUC if available.
        _patch_auc_checkpoint(runner.agent)

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
            skrl_env=_skrl_env,
            env_cfg=env_cfg,
        )

        if "VelTracking" in (args_cli.task or ""):
            _generate_ref_trajs(
                task=args_cli.task,
                runner=runner,
                isaac_env=_isaac_env,
                skrl_env=_skrl_env,
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

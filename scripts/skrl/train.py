# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help=(
        "Name of the RL algorithm to use (e.g. AMP, DDPG, IPPO, MAPPO, PPO, SAC, TD3, etc.) "
        "when several algorithms exist for the same task. For a more specific selection, use the argument --agent."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
parser.add_argument("--wandb_project", "--wandb-project", type=str, default=None, help="W&B project name (overrides YAML).")
parser.add_argument("--wandb_run_name", "--wandb-run-name", type=str, default=None, help="W&B run name (overrides YAML).")
# Hyperparameter overrides injected by search scripts via wandb sweep ${args}.
# Each flag has both underscore (wandb) and hyphen (human-friendly) forms.
# SAC
parser.add_argument("--sac_lr", "--sac-lr", type=float, default=None, help="SAC learning rate.")
parser.add_argument("--sac_batch_size", "--sac-batch-size", type=int, default=None, help="SAC batch size.")
parser.add_argument("--sac_discount", "--sac-discount", type=float, default=None, help="SAC discount factor.")
parser.add_argument("--sac_polyak", "--sac-polyak", type=float, default=None, help="SAC polyak coefficient.")
parser.add_argument("--sac_gradient_steps", "--sac-gradient-steps", type=int, default=None, help="SAC gradient steps per env step.")
parser.add_argument("--sac_entropy", "--sac-entropy", type=float, default=None, help="SAC initial entropy value.")
parser.add_argument("--sac_memory_size", "--sac-memory-size", type=int, default=None, help="SAC replay buffer size per env.")
# PPO
parser.add_argument("--ppo_lr", "--ppo-lr", type=float, default=None, help="PPO learning rate.")
parser.add_argument("--ppo_rollouts", "--ppo-rollouts", type=int, default=None, help="PPO rollout steps per update.")
parser.add_argument("--ppo_learning_epochs", "--ppo-learning-epochs", type=int, default=None, help="PPO gradient epochs per update.")
parser.add_argument("--ppo_mini_batches", "--ppo-mini-batches", type=int, default=None, help="PPO mini-batches per epoch.")
parser.add_argument("--ppo_discount", "--ppo-discount", type=float, default=None, help="PPO discount factor.")
parser.add_argument("--ppo_lambda", "--ppo-lambda", type=float, default=None, help="PPO GAE lambda.")
parser.add_argument("--ppo_ratio_clip", "--ppo-ratio-clip", type=float, default=None, help="PPO clip ratio epsilon.")
parser.add_argument("--ppo_entropy_scale", "--ppo-entropy-scale", type=float, default=None, help="PPO entropy loss scale.")
# shared
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None, help="Total training timesteps.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Disable Kit's hang-detector watchdog at launch. Long training/sim steps block Kit's
# main thread; after 120s the watchdog wrongly assumes Kit froze and tries to pop a GTK
# "send crash report" dialog via zenity (which spams "Failed to open display" in
# headless/no-DISPLAY sessions). The sim is not hung. Passed as a kit arg so it is set
# before the watchdog plugin initialises.
args_cli.kit_args = (args_cli.kit_args or "") + " --/app/hangDetector/enabled=false"

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
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

# import logger
logger = logging.getLogger(__name__)

import contractionRL.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Hyperparameter overrides from search scripts (injected via wandb sweep ${args})
    a = agent_cfg["agent"]
    # SAC
    if args_cli.sac_lr is not None:               a["learning_rate"]          = args_cli.sac_lr
    if args_cli.sac_batch_size is not None:        a["batch_size"]             = args_cli.sac_batch_size
    if args_cli.sac_discount is not None:          a["discount_factor"]        = args_cli.sac_discount
    if args_cli.sac_polyak is not None:            a["polyak"]                 = args_cli.sac_polyak
    if args_cli.sac_gradient_steps is not None:    a["gradient_steps"]         = args_cli.sac_gradient_steps
    if args_cli.sac_entropy is not None:           a["initial_entropy_value"]  = args_cli.sac_entropy
    if args_cli.sac_memory_size is not None:       agent_cfg["memory"]["memory_size"] = args_cli.sac_memory_size
    # PPO
    if args_cli.ppo_lr is not None:               a["learning_rate"]    = args_cli.ppo_lr
    if args_cli.ppo_rollouts is not None:          a["rollouts"]         = args_cli.ppo_rollouts
    if args_cli.ppo_learning_epochs is not None:   a["learning_epochs"]  = args_cli.ppo_learning_epochs
    if args_cli.ppo_mini_batches is not None:      a["mini_batches"]     = args_cli.ppo_mini_batches
    if args_cli.ppo_discount is not None:          a["discount_factor"]  = args_cli.ppo_discount
    if args_cli.ppo_lambda is not None:            a["lambda"]           = args_cli.ppo_lambda
    if args_cli.ppo_ratio_clip is not None:        a["ratio_clip"]       = args_cli.ppo_ratio_clip
    if args_cli.ppo_entropy_scale is not None:     a["entropy_loss_scale"] = args_cli.ppo_entropy_scale
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # configure wandb from CLI flags (bypasses Hydra struct-mode restrictions)
    _wandb_video_thread = None
    _wandb_stop_event = None
    if args_cli.wandb:
        agent_cfg["agent"]["experiment"]["wandb"] = True
        # skrl uses its own custom SummaryWriter that bypasses PyTorch's writer,
        # so wandb's sync_tensorboard=True never intercepts any writes.
        # Patch add_scalar to also call wandb.log() directly.
        import glob as _glob
        import re as _re
        import threading as _threading
        import skrl.utils.tensorboard as _skrl_tb
        import wandb as _wandb
        _orig_add_scalar = _skrl_tb.SummaryWriter.add_scalar

        def _wandb_add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
            _orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
            if _wandb.run is not None:
                _wandb.log({tag: value}, step=timestep)

        _skrl_tb.SummaryWriter.add_scalar = _wandb_add_scalar

        # background thread: upload RecordVideo MP4s to wandb as they appear
        if args_cli.video:
            _video_dir = os.path.join(log_dir, "videos", "train")
            _uploaded_videos: set = set()
            _wandb_stop_event = _threading.Event()

            def _upload_pending_videos(step: int | None = None) -> None:
                for mp4 in sorted(_glob.glob(os.path.join(_video_dir, "*.mp4"))):
                    if mp4 not in _uploaded_videos and _wandb.run is not None:
                        m = _re.search(r"step-(\d+)", os.path.basename(mp4))
                        log_step = int(m.group(1)) if m else step
                        try:
                            _wandb.log({"train/video": _wandb.Video(mp4, format="mp4")}, step=log_step)
                            _uploaded_videos.add(mp4)
                        except Exception as _e:
                            logger.warning(f"wandb video upload failed: {_e}")

            def _video_watcher() -> None:
                while not _wandb_stop_event.is_set():
                    _upload_pending_videos()
                    _wandb_stop_event.wait(timeout=30)
                _upload_pending_videos()  # final flush

            _wandb_video_thread = _threading.Thread(target=_video_watcher, daemon=True)
            _wandb_video_thread.start()

    if args_cli.wandb_project is not None:
        agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})["project"] = args_cli.wandb_project
    if args_cli.wandb_run_name is not None:
        agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})["name"] = args_cli.wandb_run_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # stop video watcher thread and wait for final upload
    if _wandb_stop_event is not None:
        _wandb_stop_event.set()
    if _wandb_video_thread is not None:
        _wandb_video_thread.join(timeout=120)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

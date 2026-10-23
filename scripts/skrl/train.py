"""Train skrl agents in Isaac Sim."""

import argparse
import os
import sys

# Prefer this checkout over an editable installation from another worktree.
_checkout = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_checkout, "source", "contractionRL"))

# Local wandb run files (config/history/media, not the same as tensorboard
# events) rack up one small file per run and are the #1 inode-quota killer on
# the cluster's home filesystem (~72k files observed, see search sweeps).
# ~/scratch has no file-count limit, so park them there instead — cloud
# syncing is unaffected. Must be set before wandb is ever imported.
os.environ.setdefault("WANDB_DIR", os.path.expanduser("~/scratch/wandb"))

from isaaclab.app import AppLauncher

# Full argument parser
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--reference_mode", choices=["stabilizing", "contractive"], default=None,
                    help="WHERE the reference is asked to operate. 'arbitrary' (default) is "
                         "the shipped XREF_INIT, which for a state-dependent plant is the "
                         "LOW-rate region by design (rule.md Step 3). 'contractive' uses the "
                         "env's measured high-rate band instead, so the same certified "
                         "controller is asked an easier question -- car_v1 goes 0.281 -> 0.463 "
                         "achieved rate. Only envs declaring xref_init_fast_* accept it.")
parser.add_argument("--disable_disturbance", action="store_true", default=False,
                    help="Disable disturbances for a classic or classic-derived Isaac analysis.")
parser.add_argument("--policy_control_mode", choices=["structured", "factorized", "full", "residual"], default=None,
                    help="Tracking actor: structured feedback (default), factorized full control W2(s)tanh(W1(s)), or plain full/residual MLP.")
parser.add_argument("--headon", action="store_true", default=False,
                    help="Show the Isaac Sim GUI. Isaac Sim runs headless by default; "
                         "pass this to disable headless mode and render a window.")
parser.add_argument("--task", type=str, default=None, help="Environment ID.")
parser.add_argument(
    "--algorithm", "--algo", type=str, default="PPO",
    help="Algorithm: ppo | contraction_ppo | sac | mbpo | c3m | lqr | sdlqr | c2rl-ppo | c2rl-sac | AMP | DDPG | TD3 | …"
)
parser.add_argument("--cmg_method", "--cmg-method", choices=["cvstem", "ccm"], default=None,
                    help="C2RL metric synthesis: CV-STEM regression or direct CCM losses.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None,
                    help="Total batched training steps. Isaac defaults: car 500k, cartpole/segway 1M, "
                         "PhysX quadrotor 1M (Classic 2M), manipulator 3M, quadruped 4M, humanoid 5M.")
parser.add_argument("--skip_final_eval", "--skip-final-eval", action="store_true", default=False,
                    help="Skip the post-training best-model evaluation rollout. That rollout is "
                         "sequential over a single env and does NOT feed the swept "
                         "Stability/* metrics (those come from the trainer loop), so sweeps "
                         "skip it — see search/build_sweep.py.")
parser.add_argument("--dynamics_checkpoint", "--dynamics-checkpoint", type=str, default=None,
                    help="Path to a dynamics.pt written by a previous C3M/C2RL run "
                         "(<log_dir>/checkpoints/dynamics.pt). REQUIRED for lqr/sdlqr/cvstem-lqr "
                         "on PhysX envs: those algorithms train no dynamics model of their own. "
                         "C3M/C2RL fit their own models; exact environments use their equations.")
parser.add_argument("--eig_reshape", "--eig-reshape", type=float, default=None,
                    help="ABLATION (c2rl_ppo classic only): reshape the Mahalanobis reward's M "
                         "eigenvalue SPREAD to this target cond(M), keeping eigenvectors and "
                         "geometric-mean scale fixed — isolates conditioning from what the C1/C2 "
                         "fit converged to. See analytical_env.py's set_eig_reshape.")

# W&B
parser.add_argument("--no_wandb", "--no-wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging.")
parser.add_argument("--wandb_project", "--wandb-project", type=str, default="contractionRL",
                    help="W&B project name.")
parser.add_argument("--wandb_run_name", "--wandb-run-name", type=str, default=None,
                    help="W&B run name.")
parser.add_argument("--wandb_batch", "--wandb-batch", type=str, default=None,
                    help="Free-text label grouping this launch, recorded to wandb.config as "
                         "`run_batch` so one relaunch is filterable/groupable apart from the "
                         "previous one. Defaults to the run date (YYYY-MM-DD), which already "
                         "separates today's runs from yesterday's with no flag.")

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
# CLI agent overrides use the same names as their YAML keys.
_ov = parser.add_argument_group("agent config overrides (flag == YAML key)")
# Shared by PPO and SAC.
_ov.add_argument("--learning_rate", "--learning-rate", "--lr", "--ppo_lr", "--ppo-lr",
                 "--sac_lr", "--sac-lr", type=float, default=None)
_ov.add_argument("--discount_factor", "--discount-factor", "--ppo_discount", "--ppo-discount",
                 "--sac_discount", "--sac-discount", type=float, default=None)
# PPO.
_ov.add_argument("--rollouts", "--ppo_rollouts", "--ppo-rollouts", type=int, default=None)
_ov.add_argument("--learning_epochs", "--learning-epochs", "--epochs",
                 "--ppo_learning_epochs", "--ppo-learning-epochs", type=int, default=None)
_ov.add_argument("--mini_batches", "--mini-batches", "--ppo_mini_batches", "--ppo-mini-batches",
                 type=int, default=None)
# skrl PPO's GAE key is `lambda`; the yaml also carries `gae_lambda` — set both.
_ov.add_argument("--gae_lambda", "--gae-lambda", "--lambda", "--ppo_lambda", "--ppo-lambda",
                 type=float, default=None)
_ov.add_argument("--ratio_clip", "--ratio-clip", "--ppo_ratio_clip", "--ppo-ratio-clip",
                 type=float, default=None)
_ov.add_argument("--entropy_loss_scale", "--entropy-loss-scale",
                 "--ppo_entropy_scale", "--ppo-entropy-scale", type=float, default=None)
_ov.add_argument("--kl_threshold", "--kl-threshold", "--ppo_kl_threshold", "--ppo-kl-threshold",
                 type=float, default=None)
_ov.add_argument("--value_loss_scale", "--value-loss-scale", type=float, default=None)
# AUC-aligned Euclidean-decrement reward (see c2rl.C2RLPPOCfg.reward_euclidean).
# Works for standalone PPO/SAC too (applied straight to the env — see the
# `not _is_contraction` block right after raw_env is built) as well as C2RL.
_ov.add_argument("--reward_euclidean", "--reward-euclidean",
                 dest="reward_euclidean", action="store_true", default=None)
_ov.add_argument("--reward_level", "--reward-level",
                 dest="reward_level", action="store_true", default=None)
# End the episode when the state leaves the env's X_TERMINATION_* box, instead of
# silently pinning it there for the rest of the horizon (env_base.step's clamp).
# On by default (env_base) — `--no_terminate_out_of_box` restores the old
# never-terminating behaviour, which is what any pre-flip number was measured
# under. None means "not specified", so the env keeps its own default.
parser.add_argument("--terminate_out_of_box", "--terminate-out-of-box",
                    dest="terminate_out_of_box", action="store_true", default=None)
parser.add_argument("--no_terminate_out_of_box", "--no-terminate-out-of-box",
                    dest="terminate_out_of_box", action="store_false")
# Report that excursion on `terminated` rather than `truncated`. Zeroes the GAE
# bootstrap, which on a cost reward is a suicide bonus — see
# BaseEnv.set_terminate_out_of_box.__doc__ before using it.
parser.add_argument("--terminate_as_terminal", "--terminate-as-terminal",
                    dest="terminate_as_terminal", action="store_true", default=False)
# C2RL only: penalise ||u - uref||^2 in the reward (residual trust anchor).
_ov.add_argument("--residual_anchor_scale", "--residual-anchor-scale",
                 dest="residual_anchor_scale", type=float, default=None)
_ov.add_argument("--pretrain_critic_steps", "--pretrain-critic-steps",
                 dest="pretrain_critic_steps", type=int, default=None)
_ov.add_argument("--pretrain_critic_epochs", "--pretrain-critic-epochs",
                 dest="pretrain_critic_epochs", type=int, default=None)
_ov.add_argument("--pretrain_critic_lr", "--pretrain-critic-lr",
                 dest="pretrain_critic_lr", type=float, default=None)
_ov.add_argument("--disable_advantage_norm", "--disable-advantage-norm",
                 dest="disable_advantage_norm", action="store_true", default=None)
_ov.add_argument("--grad_norm_clip", "--grad-norm-clip", type=float, default=None)
_ov.add_argument("--use_value_norm", "--use-value-norm", "--ppo_use_value_norm",
                 "--ppo-use-value-norm", type=str, default=None)
_ov.add_argument("--use_value_expansion", "--use-value-expansion", type=str, default=None)
_ov.add_argument("--value_expansion_horizon", "--value-expansion-horizon", type=int, default=None)
_ov.add_argument("--value_expansion_batch_size", "--value-expansion-batch-size", type=int, default=None)
_ov.add_argument("--action_bound_penalty", "--action-bound-penalty", type=float, default=None)
# Train on a fixed set of num_envs reference trajectories (one minted per env
# slot on its first reset) instead of drawing a fresh one every episode. The
# initial error stays random — this fixes the task, not the initial condition.
_ov.add_argument("--fix_ref_trajectories", "--fix-ref-trajectories",
                 action="store_true", default=False)
# SAC.
_ov.add_argument("--batch_size", "--batch-size", "--sac_batch_size", "--sac-batch-size",
                 type=int, default=None)
_ov.add_argument("--polyak", "--sac_polyak", "--sac-polyak", type=float, default=None)
_ov.add_argument("--gradient_steps", "--gradient-steps", "--sac_gradient_steps",
                 "--sac-gradient-steps", type=int, default=None)
_ov.add_argument("--initial_entropy_value", "--initial-entropy-value", "--sac_entropy",
                 "--sac-entropy", type=float, default=None)
# Under the `memory:` block, not `agent:`.
_ov.add_argument("--memory_size", "--memory-size", "--sac_memory_size", "--sac-memory-size",
                 type=int, default=None)
# Model-structure overrides (manipulate the `models:` block, not `agent:`).
parser.add_argument("--ppo_activations", "--ppo-activations", type=str, default=None)
parser.add_argument("--ppo_network_arch", "--ppo-network-arch", type=str, default=None)

# Reference windows expose current and future reference states and controls.
parser.add_argument("--ref_length", "--ref-length", type=int, default=None,
                    help="Reference points in the observation window. Default: AUTO — "
                         "sized so the window spans the discount's effective horizon "
                         "1/(1-discount_factor). Pass an int to pin it.")
parser.add_argument("--ref_offset", "--ref-offset", type=int, default=1,
                    help="Step stride between window points: xrefs[k] = xref[t + k*OFFSET]. "
                         "Widens the span per point, at the cost of subsampling the "
                         "horizon (>1 is reported as non-Markov — see RefWindow.check_markov).")

# Omitted flags preserve the YAML selection; explicit null disables GRU context.
parser.add_argument("--encoder", "--enc", type=lambda value: None if value == "null" else value,
                    default=argparse.SUPPRESS, choices=[None, "gru"], metavar="{null,gru}",
                    help="Policy context: null uses the base state; gru encodes past and future.")
parser.add_argument("--critic_encoder", "--critic-encoder", type=lambda value: None if value == "null" else value,
                    default=argparse.SUPPRESS, choices=[None, "gru"], metavar="{null,gru}",
                    help="Critic context, independently of policy: null or past/future GRUs.")
parser.add_argument("--feature_dim", "--feature-dim", dest="gru_feature_dim",
                    type=lambda value: None if value.lower() == "null" else int(value), default=argparse.SUPPRESS,
                    help="Output width of each past/future GRU; null defaults to 3 * state dimension.")
parser.add_argument("--encoder_window", "--encoder-window",
                    type=lambda value: None if value.lower() == "null" else int(value), default=argparse.SUPPRESS,
                    help="Past/future window length; null uses ceil(1 / (1 - discount_factor)).")

# O6: parameterize the critic as V(s) = f_theta(s) + ||e||^2_M, with the second
# term computed analytically from the frozen CMG instead of learned. The
# decrement reward's telescoping identity is V_shaped = (1-gamma)V_orig - Phi(s),
# so the O(1) potential the shaping removes from the reward reappears in the
# critic's target; adding it back in closed form leaves f_theta only the O(dt)
# part. Pair with --use_value_norm false (the term is in real value units).
parser.add_argument("--critic_analytic_potential", "--critic-analytic-potential",
                    dest="critic_analytic_potential", action="store_true", default=False,
                    help="Critic V(s) = f_theta(s) + ||e||^2_M using the frozen CMG "
                         "(see models._AnalyticPotentialMixin). Use with use_value_norm=false.")

# UVFA-style generalization test: Also evaluate the trained policy on a fixed
# bank of reference-trajectory shapes drawn from a generator seeded
# independently of training (see env.CarEnv.set_held_out_mode) — guaranteed
# never encountered during training, unlike the normal post-training eval's
# i.i.d.-random trajectories (a different draw from the same distribution).
# None (default) = skip this second eval pass entirely.
parser.add_argument("--eval_held_out_seed", "--eval-held-out-seed", type=int, default=None,
                    help="Generator seed for a FIXED, held-out bank of reference-"
                         "trajectory shapes (must differ from --seed). None = skip "
                         "this extra generalization eval.")
parser.add_argument("--eval_held_out_trajectories", "--eval-held-out-trajectories",
                    type=int, default=64,
                    help="Size of the fixed held-out reference-trajectory bank.")

# Reference trajectory generation (auto-triggered after vel-tracking training)
parser.add_argument("--ref_num_trajs", type=int, default=1000,
                    help="Number of reference trajectories to collect after vel-tracking training.")
parser.add_argument("--ref_out_dir", type=str, default=None, help="Reference archive directory (default: data/<robot>).")
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

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
_removed_flags = {"mode", "encoder_stride", "reference_preview_steps", "critic_encoder_stride", "critic_embed_dim", "critic_combine"}
for _arg in hydra_args:
    if _arg.split("=", 1)[0].lstrip("-").replace("-", "_") in _removed_flags:
        parser.error(f"{_arg.split('=', 1)[0]} was removed; use independent encoders and models.encoder")
_cli_dotted_args = [arg for arg in hydra_args if arg.startswith("--") and "=" in arg and "." in arg.split("=", 1)[0]]
args_cli.headless = not args_cli.headon
args_cli.enable_cameras = True
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args_cli.kit_args = (args_cli.kit_args or "") + " --/app/hangDetector/enabled=false"
hydra_args = [arg for arg in hydra_args if not (arg.startswith("--") and ("=" in arg or "." in arg))]
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Suppress noisy mesh/hydra warnings from Isaac Sim assets
import carb
carb.settings.get_settings().set("/log/logger/channelFilter", "-omni.hydra")

# Shared imports
import logging
import os
import random
from datetime import datetime

import gymnasium as gym
from contractionRL.agents.utils.RL.config import load_agent_config
from train_utils import (
    _evaluate_best_model,
    _generate_ref_trajs,
    _inject_angle_idx,
    _resolve_symmetry_for_env,
    apply_agent_patches,
    apply_cli_dotted_overrides,
    apply_wandb_sweep_overrides,
    disable_tensorboard_files,
    finish_wandb,
    install_wandb_scalar_hook,
    normalize_agent_cfg,
    restore_checkpoint_model_config,
    tracking_state_metadata,
    configure_tracking_context,
    validate_tracking_options,
)

# Metrics go to W&B; avoid unused TensorBoard event files.
disable_tensorboard_files()

algorithm = args_cli.algorithm.lower()
if algorithm in ("contractionppo", "contraction-ppo"):
    algorithm = "contraction_ppo"
if algorithm == "mbpo" and args_cli.ml_framework != "torch":
    raise SystemExit("MBPO is implemented for --ml_framework torch")
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
_SAC_LIKE_ALGOS = {"sac", "mbpo", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024



def _apply_agent_overrides(agent_cfg, args):
    """Write every set agent-config override into agent_cfg, by YAML key name.

    Shared by the classic and Isaac branches so a flag behaves identically on
    both (before, only the Isaac branch applied most of them). Each flag's dest
    is the YAML key it targets; most flags use ``None`` for "not passed".
    Encoder flags are absent when omitted; explicit ``None`` disables GRUs.
    ``lambda``/``gae_lambda`` and ``memory_size`` are the only keys whose
    destination differs from a plain ``agent[<dest>]`` write.
    """
    a = agent_cfg["agent"]
    policy = agent_cfg.get("models", {}).get("policy")
    if policy is not None:
        policy.setdefault("control_mode", "structured")
    if args.policy_control_mode is not None:
        if policy is None:
            raise ValueError("--policy_control_mode requires a learned policy")
        policy["control_mode"] = args.policy_control_mode
    if args.cmg_method is not None:
        a["cmg_method"] = args.cmg_method
    simple = (
        "learning_rate", "discount_factor", "rollouts", "learning_epochs",
        "mini_batches", "ratio_clip", "entropy_loss_scale", "kl_threshold",
        "value_loss_scale", "grad_norm_clip", "batch_size", "polyak",
        "gradient_steps", "initial_entropy_value",
        "value_expansion_horizon", "value_expansion_batch_size",
        "action_bound_penalty",
    )
    for key in simple:
        val = getattr(args, key, None)
        if val is not None:
            a[key] = val
            agent_cfg.setdefault('_tuned_parameters', []).append('agent.'+key)
    if args.gae_lambda is not None:
        a["gae_lambda"] = args.gae_lambda  # yaml's own spelling
    for key in ("use_value_norm", "use_value_expansion"):
        val = getattr(args, key, None)
        if val is not None:
            if str(val).lower() not in ("true", "false"):
                raise ValueError(f"--{key} must be true or false")
            a[key] = (str(val).lower() == "true")
    if args.memory_size is not None:
        agent_cfg["memory"]["memory_size"] = args.memory_size
    if getattr(args, "ppo_network_arch", None) is not None or getattr(args, "ppo_activations", None) is not None:
        layers = None
        if getattr(args, "ppo_network_arch", None) is not None:
            layers = [int(value.strip()) for value in args.ppo_network_arch.strip("[]").split(",")]
            if not layers or min(layers) < 1:
                raise ValueError("--ppo_network_arch requires positive layer widths")
        for name in ("policy", "value", "critic"):
            for layer in agent_cfg.get("models", {}).get(name, {}).get("network", []):
                if layers is not None:
                    layer["layers"] = layers
                if getattr(args, "ppo_activations", None) is not None:
                    layer["activations"] = args.ppo_activations
    if getattr(args, "max_iterations", None) is not None:
        agent_cfg["trainer"]["timesteps"] = args.max_iterations * a["rollouts"]
    if getattr(args, "num_timesteps", None) is not None:
        agent_cfg["trainer"]["timesteps"] = args.num_timesteps
    if isinstance(policy, dict) and hasattr(args, "encoder"):
        policy["encoder"] = args.encoder
    if hasattr(args, "critic_encoder"):
        for name in ("critic", "value", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            block = agent_cfg.get("models", {}).get(name)
            if isinstance(block, dict):
                block["encoder"] = args.critic_encoder
    for source, target in (("gru_feature_dim", "feature_dim"), ("encoder_window", "encoder_window")):
        if hasattr(args, source):
            agent_cfg.setdefault("models", {}).setdefault("encoder", {})[target] = getattr(args, source)
            agent_cfg.setdefault("_tuned_parameters", []).append("models.encoder."+target)
    if getattr(args, "critic_analytic_potential", False):
        if str(a.get("class", "")).lower() != "c2rl-ppo":
            raise ValueError("--critic_analytic_potential requires C2RL-PPO")
        agent_cfg["models"].setdefault("critic", {})["analytic_potential"] = True
    if getattr(args, "reward_euclidean", None):
        a["reward_euclidean"] = True
    if getattr(args, "reward_level", None):
        a["reward_level"] = True
    if getattr(args, "residual_anchor_scale", None) is not None:
        a["residual_anchor_scale"] = args.residual_anchor_scale
    if getattr(args, "pretrain_critic_steps", None) is not None:
        a["pretrain_critic_steps"] = args.pretrain_critic_steps
    if getattr(args, "pretrain_critic_epochs", None) is not None:
        a["pretrain_critic_epochs"] = args.pretrain_critic_epochs
    if getattr(args, "pretrain_critic_lr", None) is not None:
        a["pretrain_critic_lr"] = args.pretrain_critic_lr
    if getattr(args, "disable_advantage_norm", None):
        a["disable_advantage_norm"] = True
    from contractionRL.agents.utils.RL.config import validate_experiment_config
    validate_experiment_config(agent_cfg)


seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))








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
    from contractionRL.agents.utils.RL.runner import CLActorRunner as Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import contractionRL.tasks  # noqa: F401

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

if args_cli.agent is None:
    agent_cfg_entry_point = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    from contractionRL.agents.utils.RL.config import apply_isaac_rl_training_default
    apply_isaac_rl_training_default(agent_cfg, args_cli.task, algorithm, overrides=hydra_args)
    apply_cli_dotted_overrides(agent_cfg, _cli_dotted_args)
    for override in hydra_args:
        key = override.split('=', 1)[0].lstrip('+~')
        if '=' in override and key.startswith('agent.'):
            agent_cfg.setdefault('_tuned_parameters', []).append(key.removeprefix('agent.'))
    from contractionRL.tasks.direct.common.isaac.backend import configure_backend, configure_backend_agent
    if hasattr(env_cfg, "reference_mode") and args_cli.reference_mode is not None:
        env_cfg.reference_mode = args_cli.reference_mode
    backend = configure_backend(env_cfg, task=args_cli.task)
    if args_cli.disable_disturbance:
        if not hasattr(env_cfg, "disturbance_enabled"):
            raise ValueError("--disable_disturbance supports classic-derived Isaac tasks")
        env_cfg.disturbance_enabled = False
    configure_backend_agent(agent_cfg, args_cli.task, classic_derived=hasattr(env_cfg, "classic_task"))
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

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Agent-config overrides (flag name == YAML key); see _apply_agent_overrides.
    _apply_agent_overrides(agent_cfg, args_cli)
    configure_backend_agent(agent_cfg, args_cli.task, classic_derived=hasattr(env_cfg, "classic_task"))

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    args_cli.seed = env_cfg.seed

    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if (args_cli.task or '').split(':')[-1].startswith('Isaac-Transfer-path-'):
        log_dir += '_learned'
    elif backend == "physx":
        log_dir += "_physx"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # Sweep trials are throwaway: don't let hundreds of trials each write
    # ~10 full checkpoints to logs/ (see play.py:239,327 for the same idiom).
    if "WANDB_SWEEP_ID" in os.environ:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # W&B
    _wandb_video_thread = None
    _wandb_stop_event = None
    if not args_cli.no_wandb:
        import glob as _glob
        import re as _re
        import threading as _threading

        import wandb as _wandb
        from contractionRL.agents.utils.RL.logging import log_metrics

        agent_cfg["agent"]["experiment"]["wandb"] = ("WANDB_SWEEP_ID" not in os.environ)
        agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})["sync_tensorboard"] = False
        install_wandb_scalar_hook()

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
                            log_metrics({"train/video": _wandb.Video(mp4, format="mp4"), "train/video_step": log_step})
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
    else:
        agent_cfg["agent"]["experiment"]["wandb"] = False

    _wkw = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
    if args_cli.wandb_project is not None:
        _wkw["project"] = args_cli.wandb_project
    # Consistent run name: CLI override > YAML-provided name > deterministic
    # default that matches the local experiment_name (tensorboard dir), so
    # every W&B run can be correlated with its local log directory.
    _wkw["name"] = args_cli.wandb_run_name or _wkw.get("name") or agent_cfg["agent"]["experiment"]["experiment_name"]
    # run_batch makes this launch filterable in W&B.
    _wkw.setdefault("config", {}).update({
        "run_name": _wkw["name"],
        "run_batch": args_cli.wandb_batch or datetime.now().strftime("%Y-%m-%d"),
        "dynamics_backend": ('learned' if (args_cli.task or '').split(':')[-1].startswith('Isaac-Transfer-path-') else backend),
        "task": args_cli.task, "algorithm": algorithm, "seed": env_cfg.seed,
    })
    if not args_cli.no_wandb and "WANDB_SWEEP_ID" in os.environ:
        if _wandb.run is None:
            _wandb.init(project=_wkw["project"], name=_wkw.get("name"), sync_tensorboard=False)
        apply_wandb_sweep_overrides(agent_cfg)
        env_cfg.seed = args_cli.seed = int(agent_cfg["seed"])

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    restore_checkpoint_model_config(agent_cfg, resume_path)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
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
    validate_tracking_options(env.unwrapped, args_cli)
    if args_cli.fix_ref_trajectories:
        if not hasattr(env.unwrapped, "set_fix_ref_trajectories"):
            raise ValueError("--fix_ref_trajectories requires a tracking environment")
        env.unwrapped.set_fix_ref_trajectories(True)

    if getattr(env.unwrapped, "has_analytical_dynamics", False):
        raw = env.unwrapped
        length = args_cli.ref_length
        if length is None:
            length = raw.full_ref_length(offset=args_cli.ref_offset)
        raw.configure_ref_window(length, args_cli.ref_offset,
                                 agent_cfg["agent"].get("discount_factor", 0.99))
        if args_cli.terminate_out_of_box is not None or args_cli.terminate_as_terminal:
            raw.set_terminate_out_of_box(
                True if args_cli.terminate_out_of_box is None else args_cli.terminate_out_of_box,
                terminal=args_cli.terminate_as_terminal)
        if args_cli.eig_reshape is not None:
            raw.set_eig_reshape(args_cli.eig_reshape)
        if args_cli.reward_euclidean or args_cli.reward_level:
            raw.plant.reward_euclidean = args_cli.reward_euclidean
            raw.plant.reward_level = args_cli.reward_level
        # Persist the actual Dict observation space after window sizing.
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    if backend == "physx":
        dump_yaml(os.path.join(log_dir, 'params', 'env.yaml'), env_cfg)

    from contractionRL.tasks.direct.common.transfer_dynamics import prepare_transfer_dynamics
    prepare_transfer_dynamics(env.unwrapped, agent_cfg, log_dir, checkpoint=resume_path,
                              dynamics_checkpoint=args_cli.dynamics_checkpoint)
    configure_tracking_context(env.unwrapped, agent_cfg)
    from contractionRL.agents.utils.RL.config_usage import config_parameter_usage
    agent_cfg['parameter_usage'] = config_parameter_usage(agent_cfg, env.unwrapped)
    if getattr(env.unwrapped, "ref_window", None) is not None:
        _wkw.setdefault("config", {}).update(tracking_state_metadata(env.unwrapped.ref_window, agent_cfg.get('models')))
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

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

    from contractionRL.agents.utils.contraction.metrics import StatManagerEnvWrapper
    env = StatManagerEnvWrapper(env)
    # Raw (un-plot-wrapped) reference for _evaluate_best_model/_generate_ref_trajs:
    # both toggle skrl_env._reset_once directly to force a real sim reset, which
    # is an attribute set — WandbPlotWrapper.__getattr__ only intercepts GETs, so
    # setting it through the plot wrapper would shadow it on the wrapper instance
    # instead of the real IsaacLabWrapper, silently breaking that reset.
    _skrl_env = env
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(__file__))
    from wandb_plot_wrapper import WandbPlotWrapper
    # WandbPlotWrapper must wrap the SKRL-wrapped env (flat tensor obs +
    # .state()), not the raw Isaac env — the raw env's step() returns a dict
    # obs ({"policy": ...}), which crashes WandbPlotWrapper's trajectory
    # extraction (obs[i, :3] on a dict). Order matches the classic branch above.
    env = WandbPlotWrapper(env, total_timesteps=agent_cfg["trainer"]["timesteps"])

    # angle_idx (e.g. yaw at [2] on quadruped/humanoid path-tracking) is an
    # Env attribute, not a cfg field — inject it into agent_cfg["models"] so
    # standalone PPO/SAC's _gaussian_factory/_deterministic_factory (which
    # only see per-block yaml kwargs) embed it too. ContractionRunner
    # (C3M/LQR/SDLQR/C2RL) reads it directly off the env and needs no
    # injection. _isaac_env is the pre-SkrlVecEnvWrapper raw env saved above.
    _angle_idx = list(getattr(_isaac_env.unwrapped, "angle_idx", []) or [])
    _inject_angle_idx(agent_cfg, _angle_idx, _resolve_symmetry_for_env(_isaac_env))

    _alg = agent_cfg["agent"].get("class", "").lower()
    _is_contraction = _alg in _CONTRACTION_ALGOS
    normalize_agent_cfg(agent_cfg, algorithm=_alg, device=env.device)


    if _is_contraction:
        from contractionRL.runners import ContractionRunner
        # lqr/sdlqr/cvstem-lqr own no dynamics model — they need a pretrained
        # one loaded from disk here (ContractionRunner raises with the flag
        # to pass if it's missing). See --dynamics_checkpoint.
        if args_cli.dynamics_checkpoint:
            agent_cfg["agent"]["dynamics_checkpoint"] = args_cli.dynamics_checkpoint
        runner = ContractionRunner(env, agent_cfg, is_classic=False)
    else:
        runner = Runner(env, agent_cfg)

    # Every patch no-ops on C2RL's outer agent (it owns no .policy/.scheduler/
    # .entropy_optimizer) — C2RLAgent applies them to its inner PPO/SAC
    # sub-agent itself, which is where they matter. Contraction algorithms
    # already namespace their own track_data() keys, hence namespace=False.
    apply_agent_patches(runner.agent, algorithm=_alg, namespace=not _is_contraction)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    from contractionRL.agents.utils.RL.logging import evaluation_section
    from contractionRL.tasks.direct.common.transfer_dynamics import is_transfer_path
    with evaluation_section('Simulation Evaluation' if is_transfer_path(_isaac_env.unwrapped) else None):
        runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    if _wandb_stop_event is not None:
        _wandb_stop_event.set()
    if _wandb_video_thread is not None:
        _wandb_video_thread.join(timeout=120)

    # Evaluate saved best models only for reference-tracking environments.
    _evaluate_best_model(
        task=args_cli.task,
        runner=runner,
        isaac_env=_isaac_env,
        skrl_env=_skrl_env,
        env_cfg=env_cfg,
        args_cli=args_cli,
    )

    if (args_cli.task or "").split(':')[-1].startswith("Isaac-Transfer-vel-"):
        _generate_ref_trajs(
            task=args_cli.task,
            runner=runner,
            isaac_env=_isaac_env,
            skrl_env=_skrl_env,
            env_cfg=env_cfg,
            args_cli=args_cli,
        )

    env.close()

    finish_wandb(args_cli)

if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)

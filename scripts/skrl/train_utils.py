import os
import sys

import torch

from contractionRL.agents.utils.RL.logging import log_metrics
from contractionRL.agents.utils.RL.evaluation import EvaluationControl, evaluation_act

_SAC_LIKE_ALGOS = {"sac", "mbpo", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024
_VEL_TASK_TO_ROBOT = {f"Isaac-Transfer-vel-{name}-v0": name.lower()
                      for name in ("Quadruped", "Humanoid", "Manipulator")}
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generated reusable inputs live in data/; per-run outputs live in logs/.
_DATA_ROOT = os.path.join(_ROOT, "data")

def _default_num_envs_classic(algo: str) -> int:
    return _DEFAULT_NUM_ENVS_SAC if algo.lower() in _SAC_LIKE_ALGOS else _DEFAULT_NUM_ENVS_PPO_CLASSIC


def validate_tracking_options(env, args):
    """Reject legacy switches that the compact MDP cannot apply."""
    if not getattr(getattr(env, "ref_window", None), "compact", False):
        return
    if getattr(args, "ref_length", None) not in (None, 1) or getattr(args, "ref_offset", 1) != 1:
        raise ValueError("Compact tracking uses (x, xref, xi), not a reference window; ref_length/ref_offset must be 1")
    if (getattr(args, "eig_reshape", None) is not None or getattr(args, "reward_level", False)
            or getattr(args, "reward_euclidean", False)):
        raise ValueError("Compact tracking uses its fixed energy-decrease reward; legacy reward/eig_reshape switches do not apply")
    if getattr(args, "terminate_out_of_box", None) is False:
        raise ValueError("Compact tracking keeps mandatory position/failure termination; the legacy disable switch does not apply")


def restore_checkpoint_model_config(agent_cfg, checkpoint):
    """Keep historical C2RL critics loadable without hiding their old layout."""
    if not checkpoint or not os.path.isfile(checkpoint):
        return
    from pathlib import Path
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    critic = saved.get("value", saved.get("critic_1", {}))
    if any(key.startswith("phi.") for key in critic):
        agent_cfg.setdefault("models", {}).setdefault("critic", {})["legacy_reference_critic"] = True


def configure_tracking_context(raw_env, agent_cfg):
    from contractionRL.agents.utils.RL.config import configure_encoder_context
    return configure_encoder_context(raw_env, agent_cfg)


def tracking_state_metadata(window, models_cfg=None):
    if not getattr(window, "compact", False):
        return {}
    from contractionRL.agents.utils.system_dynamics.reference import TrackingFeatures
    models = models_cfg or {}
    shared = models.get('encoder', {})
    features = TrackingFeatures(window)
    policy = models.get('policy', {}).get('encoder')
    critic = models.get('critic', models.get('value', models.get('critic_1', {}))).get('encoder')
    dim = shared.get('feature_dim') or 3 * window.x_dim
    return {"observation_width": window.flat_dim,
            "policy_input_width": features.output_dim + (2 * dim if policy == 'gru' else 0),
            "critic_input_width": features.output_dim + (2 * dim if critic == 'gru' else 0),
            "policy_encoder": policy, "critic_encoder": critic,
            "encoder_window": window.history_length,
            "gru_feature_dim": dim if 'gru' in (policy, critic) else None,
            "tracking_state_representation": "relative_position_periodic_angles",
            "relative_position_axes": [window.state_names[i] for i in features.position],
            "physical_angle_coordinates": "continuous_lift", "tracking_energy": "minimum_over_windings"}


def save_classic_config(agent_cfg, raw_env, directory):
    """Record the resolved model and MDP settings beside their checkpoints."""
    import copy
    from pathlib import Path
    import yaml
    cfg = copy.deepcopy(agent_cfg)
    from contractionRL.agents.utils.RL.config_usage import config_parameter_usage
    cfg.pop('_tuned_parameters', None)
    cfg.get("agent", {}).get("experiment", {}).get("wandb_kwargs", {}).pop("settings", None)
    cfg["environment"] = {
        "task": raw_env.spec.id if raw_env.spec else None,
        "compact_reference": bool(getattr(raw_env.ref_window, "compact", False)),
        "reference_mode": raw_env.reference_mode,
        "time_bound": raw_env.time_bound, "dt": raw_env.dt,
        "ref_length": raw_env.ref_window.length, "ref_offset": raw_env.ref_window.offset,
        "disturbance_scales": raw_env.disturbance.scales.cpu().tolist(),
        **tracking_state_metadata(raw_env.ref_window, cfg.get('models')),
    }
    cfg["parameter_usage"] = config_parameter_usage(cfg, raw_env)
    target = Path(directory)/"params"
    target.mkdir(parents=True, exist_ok=True)
    (target/"agent.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run_metadata(args_cli, task: str) -> dict:
    """Identity of the run that produced an eval_results.json.

    The metrics in eval_results.json are anonymous on their own — the run
    directory is named by timestamp only, so nothing in the file says which
    seed/env/algorithm it came from. Multi-seed aggregation (run_seeds.sh +
    scripts/skrl/utility/aggregate_seeds.py) needs exactly that to group runs, so it is
    written into the json rather than parsed back out of a path.

    CRL_RUN_TAG is set by run_seeds.sh to a per-launch tag, so the aggregator
    can select one batch of seeds instead of every historical run under logs/.
    """
    return {
        "task": task,
        "algorithm": getattr(args_cli, "algorithm", None),
        "seed": getattr(args_cli, "seed", None),
        "run_tag": os.environ.get("CRL_RUN_TAG", ""),
    }



def _resolve_symmetry_for_env(raw_env) -> int:
    """``_resolve_symmetry`` for the standalone PPO/SAC path, which has no
    ContractionRunner to ask. Delegates to the same verified check so both
    routes make the same call for the same env (never one quotiented and one
    not, which would make PPO and C2RL-PPO architecturally incomparable).
    """
    from contractionRL.runners.contraction_runner import _env_attrs, _resolve_symmetry, _unwrap_env
    env = _unwrap_env(raw_env)
    # Read attributes through sub-environments too: SyncVectorEnv does not forward them.
    x_dim, angle_idx = _env_attrs(env, "x_dim", "angle_idx")
    if x_dim is None:
        (x_dim,) = _env_attrs(env, "num_dim_x")
    return _resolve_symmetry(env, x_dim, list(angle_idx or []))


def _inject_angle_idx(agent_cfg: dict, angle_idx: list, sym=None) -> None:
    """Inject ``angle_idx``/``pos_dim`` into every model sub-block of agent_cfg["models"].

    Only the standalone PPO/SAC path needs this: those models are built by
    _gaussian_factory/_deterministic_factory (runner.py) purely from each
    yaml/cfg block's own keys, with no access to the env object. The
    ContractionRunner path (C3M/LQR/SDLQR/C2RL) is self-sufficient — it reads
    angle_idx directly off the env in _setup_contraction — so this is a no-op
    for that path. A no-op (angle_idx=[]) here is also harmless: every
    consumer treats an empty angle_idx as "nothing to embed".

    ``pos_dim`` is the width of the leading translation-invariant state block
    (see angle_utils' translation quotient). It travels with ``angle_idx``
    because every consumer needs both to size and build its input; 0 keeps the
    previous absolute-observation behaviour.
    """
    if not angle_idx and sym is None:
        return
    for role, block in agent_cfg.get("models", {}).items():
        if isinstance(block, dict) and role not in ("dynamics", "cmg", "encoder"):
            if angle_idx:
                block.setdefault("angle_idx", angle_idx)
            if sym is not None:
                block.setdefault("sym", sym)





def disable_tensorboard_files() -> None:
    """Stop skrl's SummaryWriter from touching disk, without breaking wandb logging.

    skrl's ``SummaryWriter.__init__`` (skrl.utils.tensorboard) eagerly
    constructs a real ``EventFileWriter``, which creates the run's tensorboard
    event file on disk the moment the agent is built — regardless of whether
    ``add_scalar`` is ever called. Those event files were the other big
    contributor to the cluster's inode quota (alongside wandb/, see
    ``install_wandb_scalar_hook``). ``install_wandb_scalar_hook`` mirrors
    every ``add_scalar`` call into wandb by wrapping this same class, so
    wandb logging must keep working after this call — only the on-disk
    writer is replaced with a no-op stub.
    """
    import skrl.utils.tensorboard as _skrl_tb

    class _NoopEventFileWriter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_event(self, event) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    _skrl_tb.EventFileWriter = _NoopEventFileWriter


def install_wandb_scalar_hook() -> None:
    """Batch each skrl scalar write into the common W&B metric layout."""
    from contextvars import ContextVar
    import skrl.utils.tensorboard as _skrl_tb
    from skrl.agents.torch import Agent
    from contractionRL.agents.utils.RL.logging import log_metrics, metric_name

    if getattr(_skrl_tb.SummaryWriter.add_scalar, "_contractionrl_logging", False):
        return
    pending = ContextVar("wandb_scalar_batch", default=None)
    original_scalar = _skrl_tb.SummaryWriter.add_scalar
    original_write = Agent.write_tracking_data

    def _add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        tag = metric_name(tag)
        if tag is None:
            return
        batch = pending.get()
        if batch is not None and tag in batch and batch[tag] == value:
            return  # A legacy alias must not write the same scalar twice.
        original_scalar(self, tag=tag, value=value, timestep=timestep)
        if batch is None:
            log_metrics({tag: value}, step=timestep)
        else:
            batch[tag] = value

    def _write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        batch = {}
        token = pending.set(batch)
        try:
            original_write(self, timestep=timestep, timesteps=timesteps)
        finally:
            pending.reset(token)
        log_metrics(batch, step=timestep)

    _add_scalar._contractionrl_logging = True
    _skrl_tb.SummaryWriter.add_scalar = _add_scalar
    Agent.write_tracking_data = _write_tracking_data


def apply_wandb_sweep_overrides(agent_cfg):
    """Apply explicit dotted config paths without coupling policy and critic encoders."""
    import wandb
    tuned = agent_cfg.setdefault('_tuned_parameters', [])
    for dotted, value in wandb.config.items():
        if dotted not in tuned:
            tuned.append(dotted)
        *parents, leaf = dotted.split(".")
        node = agent_cfg
        for key in parents:
            node = node.setdefault(key, {})
        if isinstance(value, dict) and isinstance(node.get(leaf), dict):
            node[leaf].update(value)
        else:
            node[leaf] = value


def _disarm_termination_for_eval(env, tag: str = "[Eval]") -> None:
    """Force the early-termination box off on an evaluation env.

    Evaluation measures AUC = integral of ||e||/||e0|| over the fixed horizon.
    Cutting an episode short does not just shorten that integral, it inverts the
    metric: a policy that falls at step 20 stops accumulating error and scores a
    smaller (better-looking) AUC than one that tracks imperfectly for all 500
    steps. The same truncation makes the number incomparable with every
    LQR/C3M/CV-STEM baseline, all measured over full episodes.

    Early termination is a training-data intervention -- it keeps 500 steps of
    already-fallen transitions out of the rollout batch. It is not a measurement
    change, so it is disarmed here regardless of the training setting.
    """
    unwrapped = getattr(env, "unwrapped", env)
    if getattr(getattr(unwrapped, "ref_window", None), "compact", False):
        return  # Compact tracking MDPs preserve failure termination during evaluation.
    if getattr(unwrapped, "terminate_out_of_box", False):
        unwrapped.set_terminate_out_of_box(False)
        print(f"{tag} early termination disarmed for evaluation — AUC is defined on the "
              f"full horizon, and truncating it rewards a policy for failing sooner.")


def apply_cli_dotted_overrides(agent_cfg: dict, extra_args: list[str]) -> list[str]:
    """Fold CLI ``--a.b.c=value`` args into ``agent_cfg``; return what was applied.

    Same dotted-path semantics as :func:`apply_wandb_sweep_overrides`, so a knob
    can be pinned by hand exactly the way a sweep would set it
    (``--agent.mini_batches=16``). Without this these args land in argparse's
    leftovers and are silently dropped on the classic route — the run then trains
    on the yaml defaults while its command line claims otherwise, which is the
    same class of silent-config failure as an unknown yaml key.

    The leaf must already exist in the yaml. A typo would otherwise create a dead
    key that ``rl_glue.filter_cfg_fields`` drops later without a word — the exact
    failure this function exists to prevent.
    """
    import ast

    applied = []
    for arg in extra_args:
        if not (arg.startswith("--") and "=" in arg and "." in arg.split("=", 1)[0]):
            continue
        dotted, raw = arg[2:].split("=", 1)
        try:
            # Literal first ("16" -> int, "0.3" -> float, "true"/"mlp" fall
            # through to str), so a numeric knob never arrives as a string.
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = {"true": True, "false": False, "null": None}.get(raw.lower(), raw)

        *parents, leaf = dotted.split(".")
        node = agent_cfg
        for key in parents:
            if not isinstance(node.get(key), dict):
                raise SystemExit(
                    f"[train] --{dotted}: no '{key}' section in this env's config — "
                    "the override would be silently dropped.")
            node = node[key]
        if leaf not in node:
            raise SystemExit(
                f"[train] --{dotted}: '{leaf}' is not a key of "
                f"{'.'.join(parents) or 'the config'} (have: "
                f"{', '.join(sorted(node)[:12])}...). Refusing to set a key the "
                "config does not declare — it would be dropped without a word.")
        node[leaf] = value
        agent_cfg.setdefault('_tuned_parameters', []).append(dotted)
        applied.append(f"{dotted}={value!r}")
    if applied:
        print("[train] CLI overrides: " + "  ".join(applied))
    return applied


def finish_wandb(args_cli) -> None:
    """Close the active W&B run, if this process started one."""
    if getattr(args_cli, "no_wandb", False):
        return
    wandb = sys.modules.get("wandb")
    if wandb is not None and wandb.run is not None:
        wandb.finish()


from contractionRL.agents.utils.RL.config import normalize_agent_cfg


def apply_agent_patches(agent, *, algorithm: str,
                        namespace: bool = True) -> None:
    """Apply every post-construction patch an agent needs, in dependency order.

    Each patch no-ops when the agent lacks the attribute it hooks, so this is
    safe to call on any agent — including C2RL's outer agent, which owns no
    ``.policy``/``.scheduler``/``.scaler`` and patches its inner PPO/SAC
    sub-agent itself (see c2rl.py).

    ``namespace=False`` for contraction algorithms, which already namespace
    their own ``track_data`` keys.
    """
    from contractionRL.agents.utils.RL.agent_patches import (
        best_metric_for,
        patch_algo_namespace,
        patch_auc_checkpoint,
        patch_kl_logging,
        patch_prune_checkpoints,
        patch_sac_entropy_clamp,
        patch_terminal_observations,
    )

    patch_kl_logging(agent)
    patch_terminal_observations(agent)
    patch_sac_entropy_clamp(agent)
    if namespace:
        patch_algo_namespace(agent, algorithm.upper())
    patch_prune_checkpoints(agent)
    patch_auc_checkpoint(agent, metric=best_metric_for(algorithm))


def _max_step_reward(robot: str, env_cfg) -> float:
    """Best-case per-step reward for a vel-tracking env's reward function.

    Sum of the maxima of every reward term that can be positive; terms of the
    form `nonneg_quantity * non_positive_scale` (all the tracking/regularization
    penalties) have a best case of 0 and are omitted.

    quadruped/humanoid: alive bonus + the two exp-tracking terms (each saturates
    at its scale when the tracking error is 0). The quadruped also has a gait
    term `(2*gait_score - 1) * rew_gait`, gait_score in [0, 1], whose best case
    is `+rew_gait` (>0) — it must be included or the "theoretical max" it feeds
    (0.5 * max * T for the ref-traj quality gate) is under-counted, so an
    actually-achievable return can exceed it (observed: a real run hit ~6200
    against a mis-computed 5200 ceiling). Humanoid has no gait term. manipulator
    has no alive bonus and every term is `error * negative_scale`, best case 0.
    """
    if robot in ("quadruped", "humanoid"):
        # rew_gait absent on humanoid (getattr default 0.0); its max contribution
        # is +rew_gait (gait_score = 1 → (2*1 - 1)*rew_gait).
        return (env_cfg.rew_alive + env_cfg.rew_lin_vel + env_cfg.rew_yaw_rate
                + max(0.0, getattr(env_cfg, "rew_gait", 0.0)))
    if robot == "manipulator":
        return 0.0
    raise ValueError(f"no max-reward formula for robot '{robot}'")



def _generate_ref_trajs(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli,
                        out_dir=None, checkpoint_path=None):
    import numpy as np
    import torch

    robot = _VEL_TASK_TO_ROBOT.get(task.split(':')[-1])
    if robot is None:
        print(f"[RefTraj] No robot mapping for task '{task}'; skipping.")
        return

    out_dir = out_dir or getattr(args_cli, 'ref_out_dir', None) or os.path.join(_DATA_ROOT, robot)
    out_path = os.path.join(out_dir, "dynamics_data.npz")
    import math
    dt = float(env_cfg.sim.dt * env_cfg.decimation)
    duration = float(env_cfg.episode_length_s)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Reference episode duration and timestep must be finite and positive")
    T = int(duration / dt)
    if T < 4 or args_cli.ref_num_trajs < 1 or skrl_env.num_envs < 1:
        raise ValueError("Reference export requires positive environment/trajectory counts and at least four steps")
    oversample = float(args_cli.ref_oversample_factor)
    if not math.isfinite(oversample) or oversample < 1:
        raise ValueError("Reference oversampling factor must be finite and at least 1")
    # Quality threshold = half of the theoretical best-case total episode reward
    # (best-case per-step reward × T), rather than a hand-picked constant — this
    # tracks whatever the reward scales in env_cfg actually are per task.
    explicit_quality = args_cli.min_ref_quality
    min_reward = float(explicit_quality) if explicit_quality is not None \
        else 0.5 * _max_step_reward(robot, env_cfg) * T
    if not math.isfinite(min_reward):
        raise ValueError("Reference quality threshold must be finite")
    quality_enabled = min_reward != 0 if explicit_quality is not None else min_reward > 0

    # Load best checkpoint
    import logging as _logging
    _skrl_log = _logging.getLogger("skrl")
    _prev_level = _skrl_log.level
    _skrl_log.setLevel(_logging.ERROR)
    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    selected_checkpoint = checkpoint_path or (best_ckpt if os.path.exists(best_ckpt) else None)
    try:
        if selected_checkpoint:
            print(f"[RefTraj] Loading checkpoint: {selected_checkpoint}")
            agent.load(selected_checkpoint)
        else:
            print("[RefTraj] WARNING: best_agent.pt not found; using final weights.")
    finally:
        _skrl_log.setLevel(_prev_level)
    for model in agent.models.values():
        if model is not None:
            model.eval()

    def _get_obs(o):
        return o["policy"] if isinstance(o, dict) else o

    unwrapped = isaac_env.unwrapped
    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Quality gate: 1 full episode across all parallel environments
    if quality_enabled:
        print(f"\n[RefTraj] Evaluating quality (threshold: mean total reward >= {min_reward}) …")

        # Use full episode return, matching the scale of the training reward threshold.
        # Divide by the actual collected count when too few episodes complete.
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
                actions, _ = evaluation_act(agent, obs)
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
        if not math.isfinite(mean_r):
            raise RuntimeError("Reference quality evaluation produced a nonfinite return")
        if mean_r < min_reward:
            print(
                f"[RefTraj] SKIPPED — policy quality too low "
                f"({mean_r:.1f} < {min_reward}). Train longer or pass --min_ref_quality 0."
            )
            return

    # Oversample, then retain the longest trajectories; warn when the quality gate
    # must accept fewer valid candidates than requested.
    import math

    import tqdm

    num_trajs = args_cli.ref_num_trajs
    pool_target = max(num_trajs, int(math.ceil(num_trajs * max(1.0, args_cli.ref_oversample_factor))))
    print(f"[RefTraj] Collecting a candidate pool of {pool_target} trajectories "
          f"(oversample x{args_cli.ref_oversample_factor:g}), keeping the longest {num_trajs} → {out_path}")
    num_envs = skrl_env.num_envs
    all_states, all_actions, all_lengths = [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)

    # Save applied (clipped) controls without changing the policy actions sent to step().
    # The saved dynamics pairs must use the same control the plant applied.

    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = evaluation_act(agent, obs)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]

    ep_states = torch.zeros((num_envs, T + 1, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T + 1, u_dim), dtype=torch.float32, device=skrl_env.device)
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=skrl_env.device)

    pbar = tqdm.tqdm(total=pool_target, desc="[RefTraj] Collecting candidates")

    attempted_episodes = 0
    while len(all_states) < pool_target:
        with torch.no_grad():
            actions, _ = evaluation_act(agent, obs)
        state_tensor = unwrapped.get_physical_state()

        # Record state and action for envs that are still within T steps
        valid_mask = step_counts < T
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        # Clip only for the saved record, not for stepping (see note above).
        ep_actions[valid_indices, step_counts[valid_indices]] = \
            torch.clamp(actions[valid_indices], _act_low, _act_high).float()

        step_counts[valid_indices] += 1

        obs_dict, _, terminated, truncated, transition_info = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)

        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # References need every transition and the true pre-reset endpoint.
            attempted_episodes += len(done_indices)
            success_mask = (step_counts[done_indices] == T) & (~terminated.reshape(-1)[done_indices])
            success_indices = done_indices[success_mask]
            if attempted_episodes >= 20 * pool_target and len(all_states) < pool_target and not len(success_indices):
                raise RuntimeError("Nominal controller failed to produce enough full-length references after the collection budget")
            if len(success_indices):
                terminal_state = transition_info.get("terminal_physical_state")
                if terminal_state is None:
                    raise RuntimeError("Reference generation requires the pre-reset terminal physical state")
                ep_states[success_indices, T] = terminal_state[success_indices]
                ep_actions[success_indices, T] = ep_actions[success_indices, T - 1]
                step_counts[success_indices] = T + 1

            if len(success_indices) > 0:
                # Copy completed trajectories in bounded CPU chunks to avoid large GPU gathers.
                _CHUNK = 256
                for start in range(0, len(success_indices), _CHUNK):
                    idx = success_indices[start:start + _CHUNK]
                    s_np = ep_states[idx].cpu().numpy()
                    a_np = ep_actions[idx].cpu().numpy()
                    l_np = step_counts[idx].cpu().numpy()
                    for i in range(len(idx)):
                        all_states.append(s_np[i])
                        all_actions.append(a_np[i])
                        all_lengths.append(int(l_np[i]))
                        pbar.update(1)

            # Reset the step counts for all finished environments
            step_counts[done_indices] = 0

    pbar.close()

    # Keep the num_trajs longest candidates out of the oversampled pool.
    all_lengths_np = np.asarray(all_lengths, dtype=np.int64)
    keep = np.argsort(all_lengths_np)[::-1][:num_trajs]
    print(f"[RefTraj] Pool lengths: min={all_lengths_np.min()}, max={all_lengths_np.max()}, "
          f"median={int(np.median(all_lengths_np))} (T={T}) — keeping top {num_trajs} by length")
    states_arr = np.stack([all_states[i] for i in keep]).astype(np.float32)
    actions_arr = np.stack([all_actions[i] for i in keep]).astype(np.float32)
    lengths_arr = all_lengths_np[keep]

    os.makedirs(out_dir, exist_ok=True)

    # The diagnostic plot is a log, not data — it documents this generation run
    # rather than feeding a later one, so it goes to logs/ and leaves data/
    # holding only the npz artifacts other algorithms consume.
    plot_dir = os.path.join(_ROOT, "logs", robot)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot absolute position of 10 sampled trajectories
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for i in range(min(10, num_trajs)):
            plt.plot(states_arr[i, :, 0], states_arr[i, :, 1], label=f"Traj {i+1}")
        plt.xlabel("X Position (m, relative)")
        plt.ylabel("Y Position (m, relative)")
        plt.title("Sampled Reference Trajectories (Relative Position)")
        plt.legend()
        plot_path = os.path.join(plot_dir, "position_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[RefTraj] Saved position plot → {plot_path}")
    except Exception as e:
        print(f"[RefTraj] Failed to generate position plot: {e}")

    # Generate dynamics data via finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"[RefTraj] Computing dynamics (x_dot) via 4th-order central difference (dt={dt:.3f})...")

    # Unwrap angular columns before finite differences to avoid 2*pi/dt spikes.
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
        lengths_arr = lengths_arr[valid_mask]

    # Store one archive with x, u, x_dot and valid trajectory lengths; padding is excluded.
    dyn_path = os.path.join(out_dir, "dynamics_data.npz")
    from contractionRL.tasks.direct.common.reference_domain import save_nominal_references
    accepted = save_nominal_references(dyn_path, x=states_arr, u=actions_arr,
                                       x_dot=x_dot_arr, lengths=lengths_arr, dt=dt)
    if accepted < 1:
        raise RuntimeError("Reference export produced no accepted nominal trajectories")
    print(f"[RefTraj] Accepted {accepted} full-length references inside the domain margins")
    print(f"[RefTraj] Saved dynamics  → {dyn_path}")
    print(f"       x       shape: {states_arr.shape}")
    print(f"       u       shape: {actions_arr.shape}   (clipped to action-space bounds)")
    print(f"       x_dot   shape: {x_dot_arr.shape}")
    print(f"       lengths shape: {lengths_arr.shape}  (min {lengths_arr.min()}, max {lengths_arr.max()})")
    return dyn_path



def _evaluate_classic_path_tracking(*, task, runner, args_cli, _is_classic, num_groups: int = 10,
                                    episodes_per_group: int = 5, label: str = "",
                                    held_out_seed: int | None = None, held_out_trajectories: int = 64):
    """Post-training evaluation for classic path-tracking envs (CAC-dev style).

    Classic envs are plain non-vectorized gymnasium Envs with variable-length
    episodes and no early termination (only truncation at the sampled length),
    so unlike the Isaac rollout there is no terminate_on_fall concept and no
    vectorized-boundary bookkeeping: a plain loop over one env instance, one
    episode at a time, using its native ``tracking_error``/``dt`` step info.

    Reports mean +/- 95% CI of total reward, error AUC, and overshoot C /
    contraction rate lambda from the minimal-AUC envelope C*exp(-lambda*k*dt).

    ``held_out_seed`` (UVFA-style generalization test): evaluates on a fixed
    bank of ``held_out_trajectories`` shapes from a generator seeded
    independently of training — guaranteed unseen, unlike this function's
    ordinary i.i.d. eval trajectories (a fresh draw from the same distribution,
    not a held-out slice of it). Pair with a second call using
    ``label="HeldOut"`` so both land in separate ``eval_results.json``.

    Skipped by ``--skip_final_eval``. Sequential over one env instance (50
    episodes), fine as an end-of-training report but dead weight in a sweep: it
    does not feed the sweep metric at all (that ``Stability/auc_mean`` comes
    from StatManagerEnvWrapper during the trainer loop; the ``auc_mean`` here is
    a separate, differently-scoped number). For an SDP-per-step controller like
    online CV-STEM-LQR it is also the most expensive thing in the trial.
    """
    from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
    if getattr(args_cli, "skip_final_eval", False):
        print("[Eval] SKIPPED — --skip_final_eval (does not feed the sweep metric).")
        return

    import json

    import gymnasium as gym
    import numpy as np
    import torch
    from contractionRL.tasks.direct.common.eval_metrics import (
        fit_exponential_envelope,
        mean_confidence_interval,
    )

    probe = make_analytical_plant(task)
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
    from contractionRL.runners.contraction_runner import _unwrap_env
    training_env = _unwrap_env(getattr(runner, "_env", None))
    env_kwargs = {}
    if training_env is not None and hasattr(training_env, "ref_window"):
        env_kwargs = {"compact_reference": bool(getattr(training_env.ref_window, "compact", False)),
                      "reference_mode": training_env.reference_mode,
                      "time_bound": training_env.time_bound, "dt": training_env.dt,
                      "disturbance_scales": training_env.disturbance.scales.cpu().tolist()}
    env = make_analytical_plant(task, device=device, **env_kwargs)
    from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer
    if isinstance(agent, C2RLAgent):
        agent.freeze_cmg()
        C2RLSkrlTrainer._inject_ccm(env, agent)
    _disarm_termination_for_eval(env, "[Eval]")

    if held_out_seed is not None:
        if hasattr(env.unwrapped, "set_held_out_mode"):
            env.unwrapped.set_held_out_mode(held_out_seed, held_out_trajectories)
            print(f"[Eval] held-out mode: {held_out_trajectories} FIXED reference-trajectory "
                  f"shapes (generator seed={held_out_seed}, disjoint from training's RNG stream).")
        else:
            print(f"[Eval] --eval_held_out_seed set but {type(env.unwrapped).__name__} has no "
                  f"set_held_out_mode — falling back to ordinary i.i.d.-random eval trajectories.")

    # Mirror the training env's reference window so the eval observation matches
    # what the models were built for. A mismatch is not a shape error that
    # surfaces loudly — RefWindow.split would raise, but only after the eval env
    # has already been built — so copy the layout explicitly.
    from contractionRL.runners.contraction_runner import _unwrap_env
    _train_env = _unwrap_env(getattr(runner, "_env", None)) if getattr(runner, "_env", None) is not None else None
    _w = getattr(_train_env, "ref_window", None)
    if _w is not None and hasattr(env.unwrapped, "configure_ref_window"):
        env.unwrapped.configure_ref_window(length=_w.length, offset=_w.offset)
        if _w.history_length:
            env.unwrapped.configure_encoder_context(_w.history_length)
        elif getattr(_w, 'preview_length', 0):
            env.unwrapped.configure_ref_preview(_w.preview_length, _w.preview_stride)
        print(f"[Eval] reference window mirrored: length={_w.length} offset={_w.offset}")

    reward_list, auc_list, C_list, lbd_list = [], [], [], []
    print(f"[Eval] Rolling out {num_groups * episodes_per_group} episodes on {task} …")
    for _g in range(num_groups):
        error_trajs = []
        for _e in range(episodes_per_group):
            obs, reset_info = env.reset()
            done = False
            ep_reward = 0.0
            initial_error = float(torch.as_tensor(reset_info["tracking_error"]).reshape(-1)[0].clamp_min(0).sqrt())
            error_traj = [initial_error]
            dt = env.unwrapped.dt
            while not done:
                if isinstance(obs, dict):
                    # {x, xrefs, urefs} — flatten exactly as skrl does so the
                    # models see the same layout they were built for.
                    obs_t = torch.cat([torch.as_tensor(obs[k], dtype=torch.float32, device=device).reshape(env.unwrapped.num_envs, -1)
                                       for k in sorted(obs)], dim=-1)
                elif isinstance(obs, torch.Tensor):
                    obs_t = obs.clone().detach().to(dtype=torch.float32, device=device)
                else:
                    obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                with torch.no_grad():
                    # see _evaluate_best_model for why agent.act() (not
                    # agent.policy.act()) is the algorithm-agnostic interface
                    actions, outputs = evaluation_act(agent, obs_t)
                    action = outputs.get("mean_actions", actions)
                obs, reward, terminated, truncated, info = env.step(action)

                term_val = terminated.item() if isinstance(terminated, torch.Tensor) else bool(terminated)
                trunc_val = truncated.item() if isinstance(truncated, torch.Tensor) else bool(truncated)
                done = term_val or trunc_val
                ep_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)

                err_val = info["tracking_error"].item() if isinstance(info["tracking_error"], torch.Tensor) else info["tracking_error"]
                error_traj.append(float(np.sqrt(max(err_val, 0.0))))
            e0 = max(error_traj[0], 1e-8) if error_traj else 1.0
            norm_traj = np.asarray(error_traj) / e0
            complete = (not term_val and not bool(torch.as_tensor(info.get("episode_ended_early", False)).any())
                        and len(error_traj) == env.unwrapped.max_episode_len + 1)
            if complete:
                error_trajs.append(norm_traj)
                auc_list.append(float(np.trapezoid(norm_traj, dx=dt)) if hasattr(np, "trapezoid")
                                 else float(np.trapz(norm_traj, dx=dt)))
            reward_list.append(ep_reward / max(len(error_traj) - 1, 1))
        # paper fit: one overshoot C* per group, one convergence rate per curve
        if error_trajs:
            C, lbds = fit_exponential_envelope(error_trajs, dt)
            C_list.append(C)
            lbd_list.extend(float(x) for x in lbds)
    # Read before close() — used for the divergence threshold below.
    _time_bound = float(getattr(env.unwrapped, "time_bound", 0.0) or 0.0)
    env.close()

    rew_mean, rew_ci = mean_confidence_interval(reward_list)
    auc_mean, auc_ci = mean_confidence_interval(auc_list)
    C_mean, C_ci = mean_confidence_interval(C_list)
    lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
    results = {
        **_run_metadata(args_cli, task),
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": num_groups * episodes_per_group,
        "completed_episodes": len(auc_list),
        "episode_completion": len(auc_list) / max(len(reward_list), 1),
        "average_reward_mean": rew_mean, "average_reward_ci95": rew_ci,
        "average_reward_stddev": float(np.std(reward_list)),
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
        "auc_stddev": float(np.std(auc_list)) if auc_list else None,
        "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
        "overshoot_stddev": float(np.std(C_list)) if C_list else None,
        "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
        "contraction_rate_stddev": float(np.std(lbd_list)) if lbd_list else None,
        "num_fit_groups": num_groups,
        # Raw per-episode contraction rates (one per evaluated episode, before
        # the running/population mean above collapses them) — kept alongside
        # contraction_rate_mean so a caller can distinguish a single episode's
        # fitted lambda from the running mean lambda over the eval population.
        "contraction_rate_all": [round(float(x), 6) for x in lbd_list],
        "overshoot_all": [round(float(x), 6) for x in C_list],
        # Keep episode-level AUCs to distinguish a population shift from heavy-tail failures.
        "auc_all": [round(float(x), 6) for x in auc_list],
        "auc_median": float(np.median(auc_list)) if auc_list else None,
        "auc_p90": float(np.percentile(auc_list, 90)) if auc_list else None,
        "auc_max": float(np.max(auc_list)) if auc_list else None,
        # Flag AUC above one third of the episode duration as a heavy-tail heuristic.
        # Interpret it alongside the AUC distribution, not as a sharp failure classifier.
        "auc_diverged_frac": (float(np.mean(np.asarray(auc_list) > (1.0 / 3.0) * _time_bound))
                              if _time_bound > 0 and auc_list else None),
    }
    if not auc_list:
        results.update(auc_mean=None, auc_ci95=None)
    if not lbd_list:
        results.update(contraction_rate_mean=None, contraction_rate_ci95=None,
                       overshoot_mean=None, overshoot_ci95=None)

    _tag = f"[Eval{'-' + label if label else ''}]"
    print(f"{_tag} ── Best-model evaluation (classic path-tracking){(' — ' + label) if label else ''} ──")
    print(f"{_tag} total reward     : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={len(reward_list)})")
    if auc_list:
        print(f"{_tag} error AUC        : {auc_mean:.4f} ± {results['auc_stddev']:.4f} (stddev)")
        print(f"{_tag} contraction rate : {lbd_mean:.4f} ± {results['contraction_rate_stddev']:.4f} (stddev)")
    else:
        print(f"{_tag} stability unavailable: no full-length successful episode")

    # wandb first, disk second. Training has already succeeded by this point and
    # the numbers above are the result; a filesystem problem must not be able to
    # lose them. This ordering used to be reversed, and the disk write threw.
    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        log_metrics(wandb_logs)

    # Create lazily initialized run directories before writing evaluation results.
    _json_name = f"eval_results{'_' + label.lower() if label else ''}.json"
    out_json = os.path.join(agent.experiment_dir, _json_name)
    try:
        os.makedirs(agent.experiment_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Saved → {out_json}")
    except OSError as exc:
        # Loud, but not fatal. The metrics are already in wandb above, and the
        # json is a convenience copy: a full quota or a read-only path is not a
        # reason to discard a finished run. Raising here is what turned a
        # successful trial into a crashed one.
        print(f"[Eval] WARNING: could not write {out_json}: {exc}. "
              f"Metrics were logged to wandb; continuing.", flush=True)



def _evaluate_best_model(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli, num_groups: int = 10):
    from contractionRL.tasks.direct.common.transfer_dynamics import is_transfer_path, evaluation_backend
    from contractionRL.agents.utils.RL.logging import evaluation_section
    raw = isaac_env.unwrapped
    kwargs = dict(task=task, runner=runner, isaac_env=isaac_env, skrl_env=skrl_env,
                  env_cfg=env_cfg, args_cli=args_cli, num_groups=num_groups)
    if not is_transfer_path(raw) or not hasattr(raw, 'set_transition_backend'):
        return _evaluate_best_model_single(**kwargs)
    if getattr(args_cli, 'skip_final_eval', False):
        return
    results = {}
    for backend, label, key in (('learned', 'Simulation Evaluation', 'simulation'),
                                 ('physx', 'Sim-to-Real Evaluation', 'sim_to_real')):
        print(f'[Eval] {label}')
        with evaluation_backend(raw, backend), evaluation_section(label):
            results[key] = _evaluate_best_model_single(**kwargs, output_name=f'eval_results_{key}.json')
    import json
    path = os.path.join(runner.agent.experiment_dir, 'eval_results.json')
    with open(path, 'w') as stream:
        json.dump({**_run_metadata(args_cli, task), **results}, stream, indent=2)
    return results


def _evaluate_best_model_single(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli,
                                num_groups: int = 10, output_name='eval_results.json'):
    """Post-training evaluation of the best checkpoint (CAC-dev style).

    Loads best_agent.pt and rolls out one episode per environment, preserving
    failure termination. Reports reward/failure statistics and full-horizon
    stability statistics only for surviving episodes. Actions are deterministic
    policy means clipped to the action space.

      * total reward
      * AUC of the velocity-tracking error (trapezoid, dt-weighted)
      * contraction rate lambda and overshoot C — the exponential envelope
        C * exp(-lambda * k * dt) bounding the normalized error curves with
        minimal envelope AUC (= C/lambda), fitted per env-group (CAC-dev
        trainer/evaluator.py compute_contraction_rate).

    Results are printed, logged to wandb (if active), and saved as
    eval_results.json next to the checkpoints.

    Skipped entirely by ``--skip_final_eval`` — see
    _evaluate_classic_path_tracking for why a sweep does not want this.
    """
    if getattr(args_cli, "skip_final_eval", False):
        print("[Eval] SKIPPED — --skip_final_eval (does not feed the sweep metric).")
        return

    import json

    import numpy as np
    import torch
    from contractionRL.tasks.direct.common.eval_metrics import (
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

    from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer
    if isinstance(agent, C2RLAgent):
        agent.freeze_cmg()
        C2RLSkrlTrainer._inject_ccm(isaac_env, agent)
    unwrapped = isaac_env.unwrapped
    # Freeze episode completion during evaluation so auto-resets cannot splice
    # several episodes into a single fixed-horizon error curve.
    _disarm_termination_for_eval(isaac_env, "[Eval]")
    dt = env_cfg.sim.dt * env_cfg.decimation
    T = int(env_cfg.episode_length_s / dt)
    num_envs = skrl_env.num_envs

    if not hasattr(unwrapped, "get_tracking_error"):
        print(f"[Eval] SKIPPED — env {type(unwrapped).__name__} has no get_tracking_error().")
        return

    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Preserve the configured fall rule during evaluation.
    prev_flag = getattr(unwrapped.cfg, "terminate_on_fall", True)
    # Preserve robot failures; never join a reset episode to the original curve.
    try:
        from contractionRL.tasks.direct.common.transfer_dynamics import force_wrapper_reset
        force_wrapper_reset(skrl_env)
        obs_dict, _ = skrl_env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        total_reward = torch.zeros(num_envs, device=skrl_env.device)
        reward_steps = torch.zeros(num_envs, device=skrl_env.device)
        errors = torch.zeros(num_envs, T + 1, device=skrl_env.device)
        errors[:, 0] = unwrapped.get_tracking_error()

        alive = torch.ones(num_envs, dtype=torch.bool, device=skrl_env.device)
        failed = torch.zeros_like(alive)
        print(f"[Eval] Rolling out {num_envs} episodes of at most {T} steps …")
        with EvaluationControl(skrl_env, agent) as control:
            for k in range(T):
                states = skrl_env.state() if hasattr(skrl_env, "state") else None
                actions = control.act(obs, states, active=alive).clamp(_act_low, _act_high)
                obs_dict, rewards, terminated, truncated, transition_info = skrl_env.step(actions)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                total_reward += rewards.reshape(-1) * alive
                reward_steps += alive
                failed |= alive & terminated.reshape(-1)
                terminal_error = transition_info.get("terminal_tracking_error")
                if terminal_error is None and "tracking_error" in transition_info:
                    terminal_error = transition_info["tracking_error"].clamp(min=0).sqrt()
                endpoint_error = unwrapped.get_tracking_error() if terminal_error is None else terminal_error.reshape(-1)
                errors[:, k + 1] = torch.where(alive & ~terminated.reshape(-1), endpoint_error, torch.nan)
                done = (terminated | truncated).reshape(-1)
                control.after_step(done)
                alive &= ~done
            control_metrics = control.summary(num_envs=min(num_envs, _DEFAULT_NUM_ENVS_SAC))
    finally:
        unwrapped.cfg.terminate_on_fall = prev_flag

    err_np = errors.cpu().numpy()  # (N, T+1)
    rew_np = (total_reward / reward_steps.clamp_min(1)).cpu().numpy()
    failure_rate = float(failed.float().mean())

    # Cap stability samples consistently across algorithms without changing training parallelism.
    if num_envs > _DEFAULT_NUM_ENVS_SAC:
        num_envs = _DEFAULT_NUM_ENVS_SAC
        err_np = err_np[:num_envs]
        rew_np = rew_np[:num_envs]

    # AUC over the normalized error curve (dt-weighted trapezoid), per episode.
    # np.trapezoid is numpy>=2 only; env_isaaclab ships numpy 1.26 (trapz).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    e0_np = err_np[:, 0]
    e0_np_safe = np.maximum(e0_np, 1e-8)
    norm_err_np = err_np / e0_np_safe[:, None]
    auc_np = _trapz(norm_err_np, dx=dt, axis=1)
    auc_np = auc_np[np.isfinite(auc_np)]

    # Contraction envelope on normalized error e(t)/e(0) — CAC-dev convention.
    # Envs whose initial error is ~0 (near-zero commanded velocity) carry no
    # contraction information and are excluded from the fit.
    e0 = err_np[:, 0]
    fit_mask = (e0 > 0.05) & np.isfinite(err_np).all(axis=1)
    C_list, lbd_list, score_list = [], [], []
    fit_ids = np.nonzero(fit_mask)[0]
    if len(fit_ids) >= num_groups:
        groups = np.array_split(fit_ids, num_groups)
        for g in groups:
            # raw error curves; fit_exponential_envelope normalizes by e(0) itself
            raw_trajs = [err_np[i] for i in g]
            C, lbds = fit_exponential_envelope(raw_trajs, dt)
            C_list.append(C)
            lbd_list.extend(float(x) for x in lbds)
            score_list.extend(float(x) / max(C, 1e-6) for x in lbds)
    else:
        print(f"[Eval] WARNING: only {len(fit_ids)} envs with e(0) > 0.05; skipping contraction fit.")

    rew_mean, rew_ci = mean_confidence_interval(rew_np)
    auc_mean, auc_ci = mean_confidence_interval(auc_np)
    results = {
        **_run_metadata(args_cli, task),
        **control_metrics,
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "failure_rate": failure_rate,
        "num_episodes": int(num_envs),
        "episode_steps": int(T),
        "average_reward_mean": rew_mean, "average_reward_ci95": rew_ci,
        "average_reward_stddev": float(np.std(rew_np)),
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
        "auc_stddev": float(np.std(auc_np)),
    }
    if C_list:
        C_mean, C_ci = mean_confidence_interval(C_list)
        lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
        score_mean, score_ci = mean_confidence_interval(score_list)
        results.update({
            "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
            "overshoot_stddev": float(np.std(C_list)),
            "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
            "contraction_rate_stddev": float(np.std(lbd_list)),
            "contraction_score_mean": score_mean, "contraction_score_ci95": score_ci,
            "contraction_score_stddev": float(np.std(score_list)),
            "num_fit_groups": len(C_list),
        })

    print("[Eval] ── Best-model evaluation (failure termination enabled) ──")
    print(f"[Eval] average reward   : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={num_envs})")
    print(f"[Eval] error AUC        : {auc_mean:.4f} ± {auc_ci:.4f}")
    if control_metrics:
        print(f"[Eval] Control timing   : {control_metrics}")
    if C_list:
        print(f"[Eval] overshoot C      : {C_mean:.3f} ± {C_ci:.3f}")
        print(f"[Eval] contraction rate : {lbd_mean:.4f} ± {lbd_ci:.4f}  (C·e^(−λkΔt), min AUC)")

    out_json = os.path.join(agent.experiment_dir, output_name)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved → {out_json}")

    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if k.startswith("control_"):
                    wandb_logs[f"Evaluation/Control/{k}"] = v
                elif "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        log_metrics(wandb_logs)
    return results


# Classic route  (--classic flag)
from contractionRL.agents.utils.RL.env import BatchedGymnasiumWrapper

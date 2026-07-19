import os
import sys
import torch

_SAC_LIKE_ALGOS = {"sac", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024
_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generated DATA lives under data/, not logs/. The distinction is lifetime and
# role, not format: logs/ holds the output of one run (checkpoints, tensorboard
# events, eval json) and is disposable per run, while data/ holds artifacts that
# are INPUTS to later runs of other algorithms — dynamics_data.npz (the
# reference trajectories path-tracking envs track) and the cm_data*.npz metric
# caches synthesized from it. Deleting logs/ must never force a re-synthesis.
_DATA_ROOT = os.path.join(_ROOT, "data")

def _default_num_envs_classic(algo: str) -> int:
    return _DEFAULT_NUM_ENVS_SAC if algo.lower() in _SAC_LIKE_ALGOS else _DEFAULT_NUM_ENVS_PPO_CLASSIC



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

    Sum of the maxima of every reward term that can be positive; terms of the
    form `nonneg_quantity * non_positive_scale` (all the tracking/regularization
    penalties) have a best case of 0 and are omitted.

    quadruped/humanoid: alive bonus + the two exp-tracking terms (each saturates
    at its scale when the tracking error is 0). The quadruped ALSO has a gait
    term `(2*gait_score - 1) * rew_gait`, gait_score in [0, 1], whose best case
    is `+rew_gait` (>0) — it MUST be included or the "theoretical max" it feeds
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



def _generate_ref_trajs(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli):
    import numpy as np
    import torch

    robot = next((name for prefix, name in _VEL_TASK_TO_ROBOT.items() if task.startswith(prefix)), None)
    if robot is None:
        print(f"[RefTraj] No robot mapping for task '{task}'; skipping.")
        return

    out_dir = os.path.join(_DATA_ROOT, robot)
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
        print("[RefTraj] WARNING: best_agent.pt not found; using final weights.")
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
    if min_reward > 0:
        print(f"\n[RefTraj] Evaluating quality (threshold: mean total reward >= {min_reward}) …")

        # This gate measures the SAME quantity as the training-time
        # "Reward / Total reward (mean)" wandb metric, and that is deliberate:
        # min_reward is calibrated as 0.5 * best-case-per-step * T (see
        # _max_step_reward), i.e. on the scale of a full TRAINING episode's
        # return. So the gate rollout must reproduce training's episode
        # structure — fall termination left at its cfg default (True), and the
        # policy's own (stochastic) actions, exactly as the trainer collected
        # the reward the threshold was derived from.
        #
        # It must NOT disable terminate_on_fall the way _evaluate_best_model
        # does: with fall termination OFF, a fallen robot keeps flailing for
        # the full T steps, accumulating the reward function's deliberate
        # lying-down penalties (rew_flat, rew_base_height, zero tracking) at
        # roughly -2.5/step — a large NEGATIVE tail that training never sees
        # (it resets on fall). That non-terminating measurement produced gate
        # values like -545 against a +2600 threshold — structurally
        # unreachable regardless of policy quality — even while the same
        # policy's training "Total reward (mean)" was well positive. Fall
        # termination ON keeps a fallen episode SHORT (small return) instead of
        # a huge negative, so the gate number stays on the threshold's scale.
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
    num_envs = skrl_env.num_envs
    all_states, all_actions, all_lengths = [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)

    # _act_low/_act_high (defined above, alongside `unwrapped`) are used ONLY
    # when writing into the saved `u` array below, never to modify what's
    # stepped through the env. The policy samples with clip_actions=False
    # (clipping inside the actor corrupts the log-prob), and the env already
    # enforces action bounds on its own (its actuator/physics pipeline), so
    # re-clipping before `skrl_env.step()` would be redundant. But the *saved*
    # dynamics_data.npz must record actions within the declared action space —
    # an unclipped, possibly out-of-range sample is not a valid "u" for
    # fitting f(x) + B(x)u.

    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]

    ep_states = torch.zeros((num_envs, T, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T, u_dim), dtype=torch.float32, device=skrl_env.device)
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
                    l_np = step_counts[idx].cpu().numpy()
                    for i in range(len(idx)):
                        all_states.append(s_np[i])
                        all_actions.append(a_np[i])
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
    lengths_arr = all_lengths_np[keep]

    os.makedirs(out_dir, exist_ok=True)

    # The diagnostic plot is a LOG, not data — it documents this generation run
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
        lengths_arr = lengths_arr[valid_mask]

    # Single unified file: reference trajectories ARE the (x, u) part of the
    # dynamics data, so there is no separate ref_trajs.npz anymore.
    #   x       (N, T, x_dim)  physical states; steps >= lengths[n] are padding
    #                          (the last valid state repeated, keeping x_dot ~ 0)
    #   u       (N, T, u_dim)  executed (clipped) actions, same padding rule
    #   x_dot   (N, T, x_dim)  4th-order central differences of x
    #   lengths (N,)           number of VALID steps per trajectory — consumers
    #                          mask with arange(T) < lengths[:, None]
    dyn_path = os.path.join(out_dir, "dynamics_data.npz")
    np.savez_compressed(dyn_path, x=states_arr, u=actions_arr, x_dot=x_dot_arr, lengths=lengths_arr)
    print(f"[RefTraj] Saved dynamics  → {dyn_path}")
    print(f"       x       shape: {states_arr.shape}")
    print(f"       u       shape: {actions_arr.shape}   (clipped to action-space bounds)")
    print(f"       x_dot   shape: {x_dot_arr.shape}")
    print(f"       lengths shape: {lengths_arr.shape}  (min {lengths_arr.min()}, max {lengths_arr.max()})")



def _evaluate_classic_path_tracking(*, task, runner, args_cli, _is_classic, num_groups: int = 10, episodes_per_group: int = 5):
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

    Skipped entirely by ``--skip_final_eval``. This rollout is SEQUENTIAL over a
    single env instance (50 episodes by default), which is fine as a one-off
    end-of-training report but is dead weight inside a sweep: it does not feed
    the sweep metric at all. The swept ``Stability/auc_mean`` comes from
    StatManagerEnvWrapper during the trainer loop; the ``auc_mean`` computed
    here is a separate, differently-scoped number written to eval.json. For an
    analytical controller that solves an SDP per env per step (CV-STEM-LQR
    online) it is also the single most expensive thing in the trial.
    """
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
    env = gym.make(task, device=device)

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
                if isinstance(obs, torch.Tensor):
                    obs_t = obs.clone().detach().to(dtype=torch.float32, device=device)
                else:
                    obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                with torch.no_grad():
                    # see _evaluate_best_model for why agent.act() (not
                    # agent.policy.act()) is the algorithm-agnostic interface
                    actions, outputs = agent.act(obs_t, None, timestep=0, timesteps=0)
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
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        sys.modules["wandb"].log(wandb_logs)



def _evaluate_best_model(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli, num_groups: int = 10):
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

    # Cap the Stability-tab sample size to the SAC-family env count, regardless
    # of how many parallel envs THIS run actually used. PPO-family algorithms
    # train/roll out with far more parallel envs (e.g. 4096) than SAC-family
    # ones (64, see _DEFAULT_NUM_ENVS_SAC) — without this cap, PPO's mean/CI
    # would be computed from a much larger sample than SAC's, so the two
    # wouldn't be comparable on the Stability tab. Truncating (not resampling)
    # keeps this deterministic across reruns.
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

    # Contraction envelope on NORMALIZED error e(t)/e(0) — CAC-dev convention.
    # Envs whose initial error is ~0 (near-zero commanded velocity) carry no
    # contraction information and are excluded from the fit.
    e0 = err_np[:, 0]
    fit_mask = e0 > 0.05
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
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": int(num_envs),
        "episode_steps": int(T),
        "total_reward_mean": rew_mean, "total_reward_ci95": rew_ci,
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
    }
    if C_list:
        C_mean, C_ci = mean_confidence_interval(C_list)
        lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
        score_mean, score_ci = mean_confidence_interval(score_list)
        results.update({
            "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
            "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
            "contraction_score_mean": score_mean, "contraction_score_ci95": score_ci,
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
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        sys.modules["wandb"].log(wandb_logs)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIC ROUTE  (--classic flag)
# ══════════════════════════════════════════════════════════════════════════════
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
import gymnasium

class BatchedGymnasiumWrapper(GymnasiumWrapper):
    """Overrides SKRL's default GymnasiumWrapper to prevent tensor-copy warnings.
    
    Our classic environments natively output PyTorch tensors for speed. SKRL's default
    wrapper forces torch.tensor() on the outputs, which throws a UserWarning in PyTorch
    when the output is already a tensor. This wrapper safely applies torch.as_tensor
    instead, fully implementing PyTorch's recommendation without modifying SKRL's library.
    """
    def step(self, actions: torch.Tensor):
        from skrl.utils.spaces.torch import untensorize_space, unflatten_tensorized_space, flatten_tensorized_space, tensorize_space
        
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to torch using .clone().detach() or as_tensor (implementing the PyTorch recommendation)
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        
        # Here we fix the SKRL warning by checking if it's already a tensor!
        if torch.is_tensor(reward):
            reward = reward.clone().detach().to(self.device).view(self.num_envs, -1)
            terminated = terminated.clone().detach().to(self.device).view(self.num_envs, -1)
            truncated = truncated.clone().detach().to(self.device).view(self.num_envs, -1)
        else:
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
            terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

"""Generate reference trajectories from a trained velocity-tracking policy.

The script rolls out the trained policy for many episodes, records the
physical robot state and actions at each step, and saves the results as
a compressed NumPy archive.  The path-tracking environments load these
archives to provide (x_ref, u_ref) to the contraction-tracking agent.

Usage
-----
    python scripts/generate_ref_traj.py \\
        --task    Template-Quadruped-VelTracking-Direct-v0 \\
        --checkpoint logs/skrl/quadruped_vel_tracking/.../checkpoints/agent_*.pt \\
        --robot   quadruped \\
        --num_envs 128 \\
        --num_trajs 2000 \\
        --headless

Outputs
-------
    logs/ref_trajs/{robot}.npz
        states:  float32 (num_trajs, T, state_dim)
        actions: float32 (num_trajs, T, action_dim)
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--task",       type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl agent checkpoint (.pt)")
parser.add_argument("--robot",      type=str, required=True, choices=["quadruped", "humanoid", "manipulator"])
parser.add_argument("--num_envs",   type=int, default=128)
parser.add_argument("--num_trajs",  type=int, default=2000, help="Target number of episodes to record")
parser.add_argument("--out_dir",    type=str, default="logs/ref_trajs")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ------------------------------------------------------------------ #
# Everything below runs inside the Isaac Sim context
# ------------------------------------------------------------------ #

import numpy as np
import torch
import gymnasium as gym
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import contractionRL.tasks  # noqa: F401 — registers all envs

# Map from robot name to the physical-state extractor method name
_STATE_METHOD = "get_physical_state"


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg["trainer"]["timesteps"] = 0    # we drive the loop manually

    # Build env + wrapper
    env_raw = gym.make(args.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env_raw, ml_framework="torch")
    unwrapped = env_raw.unwrapped

    # Build runner to instantiate the trained agent with correct model/memory
    runner = Runner(env, agent_cfg)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    num_envs = env.num_envs
    T = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    ep_states  = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_done    = [False] * num_envs

    print(f"[INFO] Generating {args.num_trajs} reference trajectories …")

    while len(all_states) < args.num_trajs:
        with torch.no_grad():
            actions, _ = runner.agent.act({"states": obs}, timestep=0, timesteps=0)

        # record before step so state[t] pairs with action[t]
        state = getattr(unwrapped, _STATE_METHOD)()  # (N, state_dim)
        for i in range(num_envs):
            if not ep_done[i]:
                ep_states[i].append(state[i].cpu().numpy())
                ep_actions[i].append(actions[i].cpu().numpy())

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        done = (terminated | truncated).squeeze(-1)  # (N,)

        for i in range(num_envs):
            if done[i] and not ep_done[i]:
                if len(ep_states[i]) == T:     # only keep full-length episodes
                    all_states.append(np.stack(ep_states[i]))   # (T, state_dim)
                    all_actions.append(np.stack(ep_actions[i]))
                    if len(all_states) % 100 == 0:
                        print(f"  {len(all_states)} / {args.num_trajs}")
                ep_states[i]  = []
                ep_actions[i] = []
                if len(all_states) >= args.num_trajs:
                    break

    states_arr  = np.stack(all_states[:args.num_trajs], axis=0).astype(np.float32)
    actions_arr = np.stack(all_actions[:args.num_trajs], axis=0).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.robot}.npz")
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print(f"[INFO] Saved {args.num_trajs} trajectories → {out_path}")
    print(f"       states  shape: {states_arr.shape}")
    print(f"       actions shape: {actions_arr.shape}")

    env.close()


if __name__ == "__main__":
    main()
    sim_app.close()

"""
WandB Bayesian hyperparameter search for contractionRL (SAC / PPO).

Architecture
------------
Uses a *command-based* sweep so that `wandb agent` runs train.py directly as
the subprocess.  train.py is the only process that calls wandb.init(), which
eliminates the double-init conflict that silently drops all metrics.

Usage
-----
# 1. Create a new sweep (prints SWEEP_ID, exits immediately):
python scripts/search/search_algo.py \\
    --task Quadruped-VelTracking-Direct-v0 \\
    --algorithm SAC \\
    --count 0

# 2. Run agents against an existing sweep:
wandb agent UIUC-LIRA/contractionRL-search/<SWEEP_ID> --count 10

# 3. Use the bash wrappers (recommended):
bash scripts/search/run_quadruped_vel_search.bash --algorithm SAC
bash scripts/search/run_quadruped_vel_search.bash --algorithm PPO
"""

import argparse
import sys
import wandb

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PROJECT   = "contractionRL-search"
DEFAULT_TASK      = "Quadruped-VelTracking-Direct-v0"
DEFAULT_ALGORITHM = "SAC"
DEFAULT_NUM_ENVS  = 128
DEFAULT_TIMESTEPS = 50_000

TRAIN_SCRIPT = "scripts/skrl/train.py"

# ── algorithm-specific sweep parameter spaces ─────────────────────────────────

SAC_PARAMETERS = {
    "sac_lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-3,
    },
    "sac_batch_size": {
        "values": [256, 512, 1024],
    },
    "sac_discount": {
        "distribution": "uniform",
        "min": 0.97,
        "max": 0.999,
    },
    "sac_polyak": {
        "distribution": "log_uniform_values",
        "min": 0.001,
        "max": 0.02,
    },
    "sac_gradient_steps": {
        "values": [1, 2, 4],
    },
    "sac_entropy": {
        "distribution": "log_uniform_values",
        "min": 0.05,
        "max": 1.0,
    },
    "sac_memory_size": {
        "values": [10_000, 15_000, 20_000],
    },
}

PPO_PARAMETERS = {
    "ppo_lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-3,
    },
    "ppo_rollouts": {
        "values": [16, 24, 32],
    },
    "ppo_learning_epochs": {
        "values": [4, 5, 8],
    },
    "ppo_mini_batches": {
        "values": [2, 4, 8],
    },
    "ppo_discount": {
        "distribution": "uniform",
        "min": 0.97,
        "max": 0.999,
    },
    "ppo_lambda": {
        "distribution": "uniform",
        "min": 0.90,
        "max": 0.99,
    },
    "ppo_ratio_clip": {
        "distribution": "uniform",
        "min": 0.1,
        "max": 0.3,
    },
    "ppo_entropy_scale": {
        "distribution": "log_uniform_values",
        "min": 1e-4,
        "max": 0.05,
    },
}

ALGO_PARAMETERS = {
    "SAC": SAC_PARAMETERS,
    "PPO": PPO_PARAMETERS,
}

# ── sweep builder ─────────────────────────────────────────────────────────────

def build_sweep_config(task: str, algorithm: str, num_envs: int, timesteps: int) -> dict:
    """Return a command-based wandb sweep config for the given task + algorithm.

    Command-based sweeps let `wandb agent` run train.py as the subprocess
    directly, so train.py's skrl is the single caller of wandb.init().
    Params are injected as --key value CLI args via ${args}.
    """
    if algorithm not in ALGO_PARAMETERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGO_PARAMETERS)}")

    return {
        "method": "bayes",
        "metric": {
            "name": "Reward / Total reward (mean)",
            "goal": "maximize",
        },
        # command-based: wandb agent runs this command for each trial
        "program": TRAIN_SCRIPT,
        "command": [
            "${env}",           # current Python executable
            "${program}",       # TRAIN_SCRIPT
            "--task",           task,
            "--algorithm",      algorithm,
            "--headless",
            "--wandb",
            "--num_envs",       str(num_envs),
            "--num_timesteps",  str(timesteps),
            "${args}",          # wandb injects --sac_lr VALUE ... for each trial
        ],
        "parameters": ALGO_PARAMETERS[algorithm],
    }

# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a WandB hyperparameter sweep for contractionRL.")
    parser.add_argument("--task",       type=str, default=DEFAULT_TASK,      help="IsaacLab task ID.")
    parser.add_argument("--algorithm",  type=str, default=DEFAULT_ALGORITHM, choices=list(ALGO_PARAMETERS), help="RL algorithm.")
    parser.add_argument("--project",    type=str, default=DEFAULT_PROJECT,   help="WandB project name.")
    parser.add_argument("--num_envs",   type=int, default=DEFAULT_NUM_ENVS,  help="Parallel environments per trial.")
    parser.add_argument("--timesteps",  type=int, default=DEFAULT_TIMESTEPS, help="Training timesteps per trial.")
    parser.add_argument("--count",      type=int, default=0,                 help="If >0, run this many trials after creating the sweep.")
    args = parser.parse_args()

    sweep_cfg = build_sweep_config(args.task, args.algorithm, args.num_envs, args.timesteps)
    sweep_id  = wandb.sweep(sweep_cfg, project=args.project)

    entity = wandb.api.default_entity or "UIUC-LIRA"

    print(f"\n{'='*60}")
    print(f"Created NEW wandb sweep with ID: {sweep_id}")
    print(f"  project:   {args.project}")
    print(f"  task:      {args.task}")
    print(f"  algorithm: {args.algorithm}")
    print(f"  envs/trial:{args.num_envs}   steps/trial: {args.timesteps:,}")
    print(f"")
    print(f"To run agents:")
    print(f"  wandb agent {entity}/{args.project}/{sweep_id} --count 10")
    print(f"")
    print(f"Sweep dashboard:")
    print(f"  https://wandb.ai/{entity}/{args.project}/sweeps/{sweep_id}")
    print(f"{'='*60}\n")

    if args.count > 0:
        import subprocess, os
        cmd = ["wandb", "agent", f"{entity}/{args.project}/{sweep_id}", "--count", str(args.count)]
        subprocess.run(cmd, env=os.environ.copy())

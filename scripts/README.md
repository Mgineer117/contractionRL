# Script entry points

- `list_envs.py`: environment inventory.
- `list_experiments.py`: colorful algorithm/environment inventory and key YAML settings; use `--env` and `--algorithm` to inspect a task without launching Isaac Sim.
- `skrl/train.py`, `skrl/play.py`: PPO/C2RL training and evaluation.
- `skrl/c2rl/`: metric dataset generation, preflight, verification, rate search and configuration updates.
- `skrl/utility/`: seed aggregation and GPU sizing.
- `skrl/generate_ref_traj.py`: robot reference generation.

Use the existing `env_isaaclab` Python environment. Run commands from the repository root.

Training uses skrl for PPO/SAC and their model-based variants. `--num_timesteps` counts vector environment steps; each step collects `--num_envs` transitions. `scripts/skrl/` contains the implementations.

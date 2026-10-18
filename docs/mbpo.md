# MBPO baseline

Run MBPO through the same PyTorch/skrl entrypoint, environment wrappers, evaluation, W&B sections and video options as PPO/SAC. No additional dependency is required. The implementation is in `agents/skrl/mbpo.py`; the shared task configuration is `tasks/direct/common/skrl_mbpo_cfg.yaml`. It is registered for all 20 current tasks, including analytical/PhysX Isaac aliases and the robot velocity/path tasks.

```bash
# Classic car, CPU; no Isaac Sim process.
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm mbpo --device cpu --num_envs 4 --num_timesteps 300000

# Analytical Isaac car, parallel GPU environments.
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm mbpo --mode analytical --num_envs 64 --headless

# PhysX car: the model learns from actual PhysX transitions.
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm mbpo --mode sim_to_real --num_envs 64 --headless

# Add --video --video_length 150 --video_interval 20000 for W&B videos.
python scripts/list_experiments.py --env Isaac-Car-v0 --algorithm mbpo
python scripts/skrl/play.py --classic --task classic-car-v0 --algorithm mbpo --checkpoint /path/to/agent.pt
```

These are commands, not launched experiments. Use a separate policy/checkpoint for each dynamics backend.

## Training path

1. Collect bounded stochastic controls and real transitions through the normal environment wrappers. Autoreset endpoints are replaced by the true final observation before replay storage. Only physical failures set SAC's terminal mask; time limits bootstrap.
2. Fit seven independent probabilistic MLPs on real replay, with bootstrapped training rows, a disjoint 20% holdout, early stopping and restoration of each member's best weights. Select five elites by holdout score. The Gaussian outputs predict physical state increments and reward; a binary head estimates physical termination from real `terminated` labels. Hidden gust values are never supplied to the model.
3. Branch one-step rollouts from real observations using the current stochastic policy. Sample an elite model and its predicted Gaussian noise for each branch. Store generated transitions in a separate bounded model replay.
4. Use the existing skrl SAC update, including twin critics, target critics, entropy learning and Polyak averaging. For each gradient batch, choose real replay with probability `real_data_ratio` and model replay otherwise, matching the upstream buffer-selection convention. Empty model replay falls back to real replay. Model fitting always reads real replay directly.

Tracking observations keep their declared layout. Synthetic endpoints copy the real branch's next reference and generator parameters/time, which evolve independently of the policy control. Only the physical state block is replaced by the model prediction. This keeps reference generation nominal and avoids learning or perturbing the reference. Wrapped physical angles stay wrapped; compact tracking clips remaining kinematics to the declared state bounds while preserving position excursions and marking them terminal. Other failure events use the learned terminal head.

For compact tracking, both real and synthetic rewards use `r = ||x - xref||² - ||next_x - next_xref||²`, with wrapped angular errors and `M = I`. The model's reward output is used for other task layouts. Plant transitions are learned online for both analytical and PhysX backends; MBPO never calls the plant's analytical `get_f_and_B` or C2RL/C3M's offline dynamics collector.

The compact tracking actor uses the existing structured, bounded SAC controller and 256×256 gain networks. Both Q critics use 256×256 MLPs. Flat observations use the existing squashed MLP actor. Policy and critic observations have no running normalizer. The ensemble has its own training-data mean/std buffers, which do not change policy observations. Exploration remains stochastic, with no stochasticity annealing.

## Main settings

| Setting | Default | Meaning |
|---|---:|---|
| `discount_factor` | 0.99 | SAC discount |
| `gradient_steps` / `batch_size` | 20 / 256 | SAC updates per vector step / samples per update |
| `learning_starts` / `random_timesteps` | 100 / 100 | Vector steps before updates / random warmup |
| `ensemble_size` / `num_elites` | 7 / 5 | Independent models / rollout models |
| `model_hidden` | [200, 200, 200, 200] | Each model's SiLU MLP |
| `model_train_frequency` | 250 | Vector steps between model refits; first fit occurs once learning starts and enough samples exist |
| `model_min_samples` / `model_max_samples` | 1,000 / 100,000 | Minimum real samples / maximum rows used per fit |
| `model_train_epochs` / `model_patience` | 50 / 5 | Maximum fit epochs / epochs without 1% improvement |
| `model_rollout_batch_size` | 10,000 | One-step branches generated after each fit |
| `real_data_ratio` | 0.05 | Probability of drawing a real SAC gradient batch |
| `memory.memory_size` | 100,000 | Total real transition capacity, rounded up to a multiple of `num_envs` |
| `model_buffer_size` | 100,000 | Total synthetic capacity; oldest rows are overwritten |

Override these through existing dotted options, for example `--agent.real_data_ratio=0.1 --agent.model_train_frequency=100`. The Isaac car defaults to 500,000 batched environment steps: with 64 environments this collects 32 million real transitions. Other Isaac tasks use the 1M–5M task budgets listed in the README. Explicit `--num_timesteps` overrides the task default. Adjust the budget and update/model-fit frequencies when changing parallelism; the upstream single-environment settings do not transfer unchanged in terms of real sample counts.

Checkpoints include policy/critics, SAC optimizers, the entropy scalar, ensemble, elite indices, model normalizers and model optimizer. Replay contents are not part of agent checkpoints, following the existing skrl convention. Loading a checkpoint for training starts with empty replay and collects new data; it is not an exact continuation. Playback uses the policy mean through the existing evaluator.

## Upstream relationship and scope

This is an independent skrl integration of [mbrl-lib's MBPO](https://github.com/facebookresearch/mbrl-lib/blob/3f93cccfc8d635f74e335a2f07aab6e9a48fc021/mbrl/algorithms/mbpo.py), inspected at commit `3f93cccfc8d635f74e335a2f07aab6e9a48fc021`. It follows the probabilistic ensemble, soft variance bounds/Gaussian likelihood, elite selection, branched rollouts, separate replay and SAC pattern. The upstream [Gaussian model](https://github.com/facebookresearch/mbrl-lib/blob/3f93cccfc8d635f74e335a2f07aab6e9a48fc021/mbrl/models/gaussian_mlp.py) and [model trainer](https://github.com/facebookresearch/mbrl-lib/blob/3f93cccfc8d635f74e335a2f07aab6e9a48fc021/mbrl/models/model_trainer.py) guided the model loss and validation path. No upstream runtime code or dependency is vendored.

This baseline fixes the rollout horizon at one step, retains model data with a capacity-limited ring rather than an epoch schedule, uses skrl SAC, and learns a terminal head instead of requiring an environment-specific termination callback. It is not a bit-for-bit reproduction of the upstream MuJoCo benchmark configuration. Fixed one-step rollouts also let tracking tasks reuse exact nominal reference endpoints without an extra simulator or reference approximation. Asymmetric/privileged critic states, discrete controls, and reward shaping/normalization are rejected explicitly.

Model replay writes have a local aligned batch implementation: skrl 2.1.0's single-environment multi-sample insertion advances its write cursor separately for each tensor. MBPO generates multi-sample batches, so using that path directly misaligns transition columns. Tests cover column alignment across wraparound; the installed skrl package is not patched.

# PPO configuration inheritance

Each environment owns a complete `skrl_ppo_cfg.yaml`: policy/value models, PPO optimization settings, GRU settings, normalization, optional value expansion, and the dynamics architecture and empirical fitting settings used for model rollouts. Its C2RL-PPO and ContractionPPO files inherit only that environment's PPO file.

For example, `tasks/direct/isaac_classic_car/agents/` contains:

```text
skrl_ppo_cfg.yaml                  # complete PPO config
skrl_c2rl_ppo_cfg.yaml              # extends: skrl_ppo_cfg.yaml
skrl_contraction_ppo_cfg.yaml       # extends: skrl_ppo_cfg.yaml
```

`Isaac-Car-v0`, the other paired Isaac environments, native robot tasks have their own PPO files. The CPU-only `classic-*` aliases load the corresponding `Isaac-Classic-*` config. Changing one environment's PPO settings affects its derived algorithms, without changing another environment. Native velocity-tracking tasks do not register C2RL-PPO. The common PPO files remain standalone defaults for direct loading; registered environments do not inherit them.

A task's C2RL-PPO file inherits its PPO file and adds its CMG network, metric constraints, metric training and explicit experiment overrides:

```yaml
extends: skrl_ppo_cfg.yaml
models:
  cmg:
    class: CMG
    network:
      - layers: [128, 128]
        activations: tanh
agent:
  class: C2RL-PPO
# Task-specific cm and cmg sections follow.
```

Each environment's ContractionPPO file adds the contraction policy/value classes, metric architecture, penalty and metric optimizer settings. It does not duplicate PPO optimization or context settings. Its rollout length, discount, policy/value widths and other shared defaults follow that environment's PPO file.

The loader merges dictionaries, replaces lists, and applies each shared ancestor once. Child values override parent values; in a list of parents, later parents override earlier parents. Relative paths refer to the file declaring them. Cycles and malformed parent lists fail. Registered config functions resolve this inheritance before Isaac Lab's Hydra overrides; classic training, custom `--cfg` files, sweeps and analysis use the same loader. Checkpoint configs contain resolved values and do not depend on subsequent edits to parent files.

C2RL retains the `models.critic` role used by existing checkpoints; its loader inherits PPO's `models.value` into that role. ContractionPPO and plain PPO use `models.value`. Encoder choices remain independent between actor and critic, with dimensions in the shared `models.encoder` block; see [GRU context](reference_preview.md).

## Shared runtime features

Value expansion is optional in all three PPO variants. Plain PPO uses the task's identity-metric tracking reward, C2RL uses its learned tracking metric, and ContractionPPO includes its detached contraction shaping reward in imagined targets. Actor updates and GAE retain real transitions. Critic targets and intermediate critic-only fitting use model rollouts.

Exact environments use their equations without constructing or fitting a dynamics network. PhysX PPO and ContractionPPO construct and fit the configured control-affine dynamics only when value expansion needs it. Fitting precedes the first policy observation and its model is included in checkpoints. PhysX C2RL needs its model for metric learning as well. Real environment transitions continue to use their selected backend.

## Parameters used by optimization

Shared configs contain conditional settings. For example, `models.encoder.feature_dim` is inactive when both networks use MLP, value-expansion horizon/batch size are inactive when expansion is disabled, and empirical fitting options are inactive with exact dynamics. C2RL also has settings for the unselected metric method. These defaults may remain in a resolved config, but should not be optimization dimensions for that trial.

Saved configs include `parameter_usage.active` and `parameter_usage.inactive`; each inactive path has a reason. Unsupported PPO-family agent/model parameters and unknown empirical or metric options fail validation. Dotted sweep and CLI parameters are checked after all overrides and environment selection, and inactive sampled options fail explicitly. Tune GRU dimensions only in configurations with a GRU, model fitting only when learned dynamics are used, and metric-method settings only for the selected method.

This report covers the implemented feature gates and model labels. Structural labels such as a network's input/output declaration are not optimization dimensions. It does not claim that every active parameter will improve a task or measurably change every finite run.

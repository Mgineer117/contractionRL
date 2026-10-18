# ContractionPPO baseline

This baseline ports the released [ContractionPPO implementation](https://github.com/contractionppo/ContractionPPO/tree/e63ed83516e46a2630ad203115d069d83de98f4b) to this repository's skrl training/evaluation pipeline. The associated [paper](https://arxiv.org/abs/2603.19632) is *ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers*. The algorithm slug is `contraction_ppo` (`contraction-ppo` and `ContractionPPO` are accepted aliases). It is separate from C2RL-PPO.

New tracking runs use the shared reference-error MLP controller `u = uref + W2(s) tanh(W1(s) e)` and the same tracking features in the critic. This is an intentional architecture adaptation for comparisons with our other algorithms; the released actor described below remains available with `--policy_control_mode full`. The metric loss and reward-shaping method are unchanged. Old full-control actor checkpoints remain evaluable by the comparison script.

## What the released code actually does

The implementation follows active statements in the pinned code, rather than its comments or the paper's proposed composite loss:

| Component | Released implementation used here |
|---|---|
| Metric input | Constraint error c only. The commented concatenation with robot-state minus policy-output error is inactive. |
| Metric | M(c) = Theta(c) Theta(c)^T. Theta is lower triangular with diagonal softplus(raw)+0.1. |
| Metric network | ELU MLP [128,64]; spectral normalization on every linear layer, effective weight scale 3 per layer. |
| Actor | Raw-observation ELU MLP [512,256,128], linear action mean, learnable direct standard deviation initialized at 1. No actor spectral normalization. |
| Metric loss | Separate Adam on the hinge. The derivative receives an extra constraint-scale factor during fitting, reproducing the released rollout-to-update rescaling described below. |
| Policy guidance | Add penalty_coef * (epsilon - ReLU(V_dot + alpha V + epsilon)) to rollout rewards. The added reward is detached before PPO. |
| Direct contraction gradient into actor | None: the line adding contraction loss to PPO loss is commented out upstream. |
| Two-phase schedule | Inactive: upstream's iteration counters do not gate metric updates or reward shaping. Both run throughout training. |
| Additional metric eigenvalue/Frobenius penalties | Inactive. The released algorithm does not use the regularization coefficients described in its configuration or the paper. |

Let c = g * wrap(x-x_ref)[indices], with g=`contraction_constraint_coef`. The metric loss and shaped reward are:

$$
V=c^T M(c)c,\qquad \dot V=\nabla_c V\cdot\dot c,\qquad \dot c=g\dot e,
$$

$$
h_{\mathrm{roll}}=\max(0,\dot V+\alpha V+\epsilon),\qquad \tilde r=r+\beta(\epsilon-h_{\mathrm{roll}}),
$$

$$
L_M=\operatorname{mean}\!\left[\max(0,g\dot V+\alpha V+\epsilon)\right].
$$

The active released code uses this **unnormalized** hinge. It does not divide the residual by V as in the paper's displayed normalized objective. The stored environment reward remains the repo's tracking-energy decrement; the baseline's PPO return/advantage calculation uses the augmented reward above. The metric is not installed as the environment reward metric or deployed as a controller.

Default coefficients reproduce the released handstand configuration: alpha=0.1, beta=0.005, g=100, epsilon=0.001 and metric Adam learning rate 0.001. Defaults for PPO are 24 rollout steps, five epochs, four minibatches, gamma=0.99, GAE lambda=0.95, clipping=0.2, entropy coefficient=0.01 and gradient clipping=1.0.

## Tracking adaptation and reproducibility limits

- The original task penalizes XY position relative to a stationary zero target. Here the default constraint coordinates are the environment's position coordinates, relative to its current reference. `contraction_indices` can explicitly select another subset of physical state coordinates. Both classic and Isaac tracking tasks are registered; velocity-only tasks without a reference are not registered.
- The upstream joint-PD robot interface is replaced by each local environment's existing full-control actuator interface. The policy emits the full control; no reference-control residual addition is introduced.
- The moving-reference constraint derivative is measured from one actual transition: `(wrapped_error_next - wrapped_error_current)/dt`, with appropriate angular wrapping. Every supported environment supplies `tracking_error_vector` before autoreset, so terminal samples never connect to the next episode's initial state. This needs no analytical or learned dynamics model and works with both analytical and PhysX backends.
- The released derivative routine mutates its input derivative in place (`*= constraint_scale`) while computing rollout rewards, before the transition enters storage. Metric fitting then rescales that already-scaled derivative. The port reproduces the resulting extra factor g in the metric-training derivative explicitly, without mutating measured or stored tensors. Tests compare the entire reward-to-metric-update path with upstream, including the optimizer step, in addition to value/derivative/loss/parameter-gradient parity tests. This effective scaling is a released-code detail, not an additional loss proposed by the paper.
- PPO itself uses the installed skrl engine, not the upstream RSL-RL fork. In particular, minibatch sampling, value-loss conventions, GAE/time-limit behavior and adaptive-KL scheduling are those of skrl. Metric optimizer batches are processed separately before the PPO update; their parameters and the PPO parameters are disjoint. This is a released-method baseline port, not a claim to reproduce the original robot benchmark numbers or exact random-number stream.
- Gaussian samples are stored before actuator clipping, matching the released PPO's likelihood convention. Environments enforce the actual input bounds. The metric and its optimizer are included in agent checkpoints; policy deployment does not require metric evaluation.
- Spectral normalization is a power-iteration implementation. Positive definiteness by triangular construction and sampled loss values do not by themselves establish uniform metric eigenvalue bounds or a global contraction certificate. No certificate is asserted by this port.

Each run saves `contractionppo_provenance.json`, including the upstream commit, chosen constraint coordinates, derivative construction and timestep. The verbatim reference sources and their SHA-256 hashes are in [third_party/contractionppo](../third_party/contractionppo/).

## Commands

Run with the Isaac analytical car:

```bash
/home/minjae/miniconda3/envs/env_isaaclab/bin/python scripts/skrl/train.py \
  --task Isaac-Car-v0 --mode analytical \
  --algorithm contraction_ppo --num_envs 128 --seed 42 \
  --num_timesteps 100000 --disable_disturbance \
  --wandb_project isaac-car-baselines
```

Use `--mode sim_to_real` for the PhysX setup. The baseline measures transitions directly. To use the classic analytical plant without Isaac startup, add `--classic` and select `--task classic-car-v0`.

Key settings are listed by:

```bash
python scripts/list_experiments.py --env Isaac-Car-v0 --algorithm contraction-ppo
```

PPO metrics follow the existing Evaluation/Training layout; metric loss and contraction reward mean/stddev are under `Training/Loss/ContractionPPO` and `Training/Learning/ContractionPPO`. Episode outcomes use the single `Training/Learning/episode_completion` scalar.

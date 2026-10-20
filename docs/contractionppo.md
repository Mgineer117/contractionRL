# ContractionPPO baseline

This baseline ports the released [ContractionPPO implementation](https://github.com/contractionppo/ContractionPPO/tree/e63ed83516e46a2630ad203115d069d83de98f4b) to this repository's skrl training/evaluation pipeline. The associated [paper](https://arxiv.org/abs/2603.19632) is *ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers*. The algorithm slug is `contraction_ppo` (`contraction-ppo` and `ContractionPPO` are accepted aliases). It is separate from C2RL-PPO.

New tracking runs use periodic relative-position features in both actor and critic. Structured control uses cosine/sine feedback differences; `--policy_control_mode full` uses an MLP over the same feature representation. The auxiliary metric now defaults to all physical state coordinates, minimizes over angular windings, and uses the same physical-time derivative for shaping and fitting. These are intentional tracking adaptations. Historical policy checkpoints must be evaluated with their archived source snapshot. See [periodic tracking](periodic_tracking.md).


## Current tracking adaptation

The separate metric optimizer and detached shaping reward come from the pinned implementation. The periodic state/energy adaptation is summarized below:

| Component | Current tracking adaptation |
|---|---|
| Metric input | Shared relative-position, periodic-angle tracking features (local adaptation). |
| Metric | M(z) = Theta(z) Theta(z)^T. Theta is lower triangular with diagonal softplus(raw)+0.1. |
| Metric network | ELU MLP [128,64]; spectral normalization on every linear layer, effective weight scale 3 per layer. |
| Actor | Shared periodic tracking features, structured or full-control actor; direct learnable standard deviation. |
| Metric loss | Separate Adam on the hinge, using the same physical-time derivative as rollout shaping. |
| Policy guidance | Add penalty_coef * (epsilon - ReLU(V_dot + alpha V + epsilon)) to rollout rewards. The added reward is detached before PPO. |
| Direct contraction gradient into actor | None: the line adding contraction loss to PPO loss is commented out upstream. |
| Two-phase schedule | Inactive: upstream's iteration counters do not gate metric updates or reward shaping. Both run throughout training. |
| Additional metric eigenvalue/Frobenius penalties | Inactive. The released algorithm does not use the regularization coefficients described in its configuration or the paper. |

Let c_k = g * (x-x_ref+2*pi*k)[indices] on angular coordinates, with g=`contraction_constraint_coef`, and let z be the periodic tracking features. Choose k minimizing the full quadratic. The adapted metric loss and shaped reward are:

$$
V=\min_k c_k^T M(z)c_k,\qquad \dot V=\nabla_c V\cdot g\dot e+\nabla_z V\cdot\dot z,
$$

$$
h_{\mathrm{roll}}=\max(0,\dot V+\alpha V+\epsilon),\qquad \tilde r=r+\beta(\epsilon-h_{\mathrm{roll}}),
$$

$$
L_M=\operatorname{mean}\!\left[\max(0,\dot V+\alpha V+\epsilon)\right].
$$

The active released code uses this **unnormalized** hinge. It does not divide the residual by V as in the paper's displayed normalized objective. The stored environment reward remains the repo's tracking-energy decrement; the baseline's PPO return/advantage calculation uses the augmented reward above. The metric is not installed as the environment reward metric or deployed as a controller.

Default coefficients reproduce the released handstand configuration: alpha=0.1, beta=0.005, g=100, epsilon=0.001 and metric Adam learning rate 0.001. Defaults for PPO are 24 rollout steps, five epochs, four minibatches, gamma=0.99, GAE lambda=0.95, clipping=0.2, entropy coefficient=0.01 and gradient clipping=0.5 and value-loss coefficient=2.5.

## Tracking adaptation and reproducibility limits

- The original task penalizes XY position relative to a stationary zero target. Here the default constraint coordinates are all physical state coordinates, relative to the current reference. `contraction_indices` can explicitly select another subset of physical state coordinates. Both classic and Isaac tracking tasks are registered; velocity-only tasks without a reference are not registered.
- The upstream joint-PD robot interface is replaced by each local environment's existing full-control actuator interface. The policy emits the full control; no reference-control residual addition is introduced.
- The moving-reference constraint derivative is measured from one actual transition: `(wrapped_error_next - wrapped_error_current)/dt`, with appropriate angular wrapping. Every supported environment supplies `tracking_error_vector` before autoreset, so terminal samples never connect to the next episode's initial state. This needs no analytical or learned dynamics model and works with both analytical and PhysX backends.
- The released derivative routine rescales stored derivatives in place and then rescales them again for fitting. The current periodic adaptation uses one consistent physical-time derivative in shaping and metric fitting. Standalone tests retain comparisons against the released formulas; the active tracking variant is recorded as `periodic_energy_skrl_port` in run provenance.

- `ContractionPPOAgent` subclasses the installed skrl PPO. Minibatch sampling, value-loss conventions, GAE and KL scheduling use skrl. Metric optimizer batches are processed separately before the PPO update; their parameters and the PPO parameters are disjoint. This is a periodic tracking adaptation, not a claim to reproduce the original robot benchmark numbers or exact random-number stream.
- Gaussian samples are stored before actuator clipping, matching the released PPO's likelihood convention. Environments enforce the actual input bounds. The metric and its optimizer are included in agent checkpoints; policy deployment does not require metric evaluation.
- Spectral normalization is a power-iteration implementation. Positive definiteness by triangular construction and sampled loss values do not by themselves establish uniform metric eigenvalue bounds or a global contraction certificate. No certificate is asserted by this port.

Each run saves `contractionppo_provenance.json`, including the upstream commit, chosen constraint coordinates, derivative construction and timestep. The verbatim reference sources and their SHA-256 hashes are in [third_party/contractionppo](../third_party/contractionppo/).

## Commands

Run with the Isaac analytical car:

```bash
/home/minjae/miniconda3/envs/env_isaaclab/bin/python scripts/skrl/train.py \
  --task Isaac-Classic-Car-v0 \
  --algorithm contraction_ppo --num_envs 128 --seed 42 \
  --num_timesteps 100000 --disable_disturbance \
  --wandb_project isaac-car-baselines
```

Use `--task Isaac-Car-v0` for the PhysX setup. The baseline measures transitions directly. To use the classic analytical plant without Isaac startup, add `--classic` and select `--task classic-car-v0`.

Key settings are listed by:

```bash
python scripts/list_experiments.py --env Isaac-Classic-Car-v0 --algorithm contraction-ppo
```

PPO metrics follow the existing Evaluation/Training layout; metric loss and contraction reward mean/stddev are under `Training/Loss/ContractionPPO` and `Training/Learning/ContractionPPO`. Episode outcomes use the single `Training/Learning/episode_completion` scalar.

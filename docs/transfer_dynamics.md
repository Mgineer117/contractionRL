# Transfer-path dynamics

`Isaac-Transfer-path-Quadruped-v0`, `Isaac-Transfer-path-Humanoid-v0`, and `Isaac-Transfer-path-Manipulator-v0` optimize policies in a fitted stochastic simulator. PhysX supplies the dynamics dataset and the sim-to-real evaluation. Velocity-tracking tasks and the Isaac/Isaac-Classic pairs keep their existing transitions.

Each robot's `agents/dynamics_cfg.yaml` defines its dynamics architecture and `empirical_dynamics` settings. That robot's PPO, SAC, C3M, LQR, SD-LQR and MBPO configurations inherit it. C2RL-PPO and ContractionPPO continue to inherit their own environment's PPO file. These are independent configuration files for each robot; changing one robot does not change another.

```yaml
empirical_dynamics:
  dynamics_ensemble_size: 5
  ood_penalty_coef: 1.0  # 0 disables the reward deduction.
  dataset_mode: reference  # reference or uniform
  num_controls_per_state: 4  # Only uniform collection uses this.
  dynamics_lr: 0.001
  dynamics_lr_scheduler: StepLR
  dynamics_lr_scheduler_kwargs:
    step_size: 1
    gamma: 0.999
  dynamics_batch_size: 4096
  dynamics_pretrain_epochs: 100
  dynamics_pretrain_data_path: ''
  emp_dynamics_memory_size: 100000
  dynamics_val_frac: 0.1
  dynamics_early_stop_patience: 10
```

## Dataset collection

`emp_dynamics_memory_size` is the exact number of labeled `(x, u, x_dot)` samples. An empty `dynamics_pretrain_data_path` collects fresh disturbed PhysX data. A supplied path loads an already collected dataset with matching state/control, timestep, collection-mode and disturbance provenance. Data collection never enters policy replay memory or training episode statistics.

`uniform` samples `N / n` independent realizable states and `n` uniformly sampled bounded controls per state. `N` must be divisible by `n`. Joint positions use the intersection of the task bounds and articulation limits. Floating-base position, height and velocities follow their task bounds; gravity is reconstructed from a valid orientation. The arm's end-effector position and velocity are measured from its joint state. Candidates whose reconstructed coordinates leave the task bounds are rejected. Thus the distribution is over realizable coordinates, not independent uniform samples of redundant state coordinates. All controls for one state restore that same state and actuator buffers. The control and sampled gust are held over two physics intervals. The derivative label is `(-3*x0 + 4*x1 - x2) / (2*h)`, where `h = sim.dt`.

`reference` selects `ceil(N / L)` distinct, validated reference trajectories, where `L` is the episode's number of control intervals. It starts PhysX at each selected reference's initial state and replays its nominal controls **with the task's disturbances enabled**. The recorded states are fresh disturbed rollouts, not the nominal reference-bank states. It records every state through the endpoint without autoreset, differentiates each trajectory separately using the five-point fourth-order central stencil, and uses shifted fourth-order forward/backward stencils at the first/last two states. Labels use the control sampling interval. The final trajectory is trimmed to obtain exactly `N` labels; trajectories are never concatenated before differentiation. A compatible nominal reference bank remains an upstream requirement. Abrupt control/gust changes reduce the stencil's formal accuracy even though the specified stencil is used.

## Fitting and policy optimization

Following the activation-diversity idea in [Stoyanova et al., Toward Robust Uncertainty Estimation with Random Activation Functions](https://arxiv.org/abs/2302.14552), `models.dynamics.network[0].activations` is an activation pool: `[tanh, elu, relu, leaky_relu, gelu]`. Each member receives an independent weight initialization and a distinct activation sampled without replacement using the experiment's seeded PyTorch RNG. The pool must contain at least `dynamics_ensemble_size` distinct functions drawn from those five choices. Five members are the default, including when `dynamics_ensemble_size` is omitted, so each activation is used once. The chosen activation is used throughout that member's drift, actuation and disturbance networks and remains fixed during training and inference. Activation assignments are recorded in the fit report and checkpoint and restored exactly on reload; old homogeneous checkpoints retain their original activations. This adopts activation diversity while retaining our Gaussian likelihood, bootstrapping and validation procedure; improved uncertainty calibration in these tasks still requires evaluation.

Each of the `dynamics_ensemble_size` independently initialized `NeuralDynamics` members learns a control-affine mean `f(x) + B(x)u` and a diagonal conditional Gaussian disturbance density conditioned on `(x,u)`. The networks minimize negative log-likelihood jointly within each member. Members fit independently bootstrapped training samples. Each epoch visits the shuffled bootstrap in minibatches, including the final partial batch; StepLR advances once per epoch. All members use the same held-out trajectories or sampled-state groups, selected before bootstrapping, so neighboring time points or controls of the same state do not leak between training and validation. The requested validation fraction is rounded to a whole number of groups, with at least one held-out group. Each member restores its own best validation epoch. The frozen ensemble is shared by all algorithm components; contraction synthesis uses its mean drift and actuation, which remain control affine.

Each parallel environment is assigned one fixed ensemble member when learned transitions are first enabled. The seeded assignment randomizes member labels and environment slots while keeping member counts within one of each other: with 512 environments and five members, the counts are 102, 102, 102, 103, and 103. Assignments remain fixed across control steps, episode resets and evaluation backend switches. Five shared networks serve the batched environment states on the configured device; policies share weights across all members.

Each policy action is held over `decimation` Euler substeps of duration `sim.dt`. A fresh standard Gaussian disturbance draw is sampled per environment and control interval and held across those substeps. The assigned member's state/action-conditioned mean and scale are recomputed at every substep. No extra prescribed gust is added to the learned transition. The resulting ensemble population represents a mixture of fixed-model environments; it does not switch members between transitions. This approximation does not preserve the original gust's bounded support or latent hold-duration correlation.

Value expansion recovers each root's environment slot from the time-by-environment rollout memory and uses that slot's assigned dynamics member throughout the imagined horizon, including every physics substep. Chunking does not change the root-to-member mapping. Imagined disturbance draws are independent of real transitions and do not consume training RNG. CMG learning continues to use the ensemble-mean drift and actuation. These rules apply to `Isaac-Transfer-path-*`; velocity-tracking tasks train their locomotion policies in PhysX and do not prepare this learned simulator.

## MOPO reward penalty

Following [MOPO, section 4.3](https://proceedings.neurips.cc/paper_files/paper/2020/file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf), learned-simulator policy optimization uses

$$
\widetilde r(x,u)=r_{\mathrm{task}}(x,u)-\lambda U(x,u),\qquad U(x,u)=\Delta t\max_i\|\sigma_{\dot x,i}(x,u)\|_2.
$$

Here `ood_penalty_coef` is the finite nonnegative coefficient $\lambda$, and $\Delta t=\texttt{sim.dt}\times\texttt{decimation}$ is one control interval. The score uses each member's predicted **standard deviation**, not variance or mean disagreement. Because these models predict derivatives, multiplying by the control interval approximates the transition standard deviation under the held disturbance; it does not propagate uncertainty through nonlinear substeps. The score uses the pre-transition state and bounded applied action and is deducted once per control interval. Value expansion uses the same penalty. Step information exposes `task_reward` and `ood_penalty`; the episode return during optimization includes the penalty.

The defaults of five members and coefficient `1.0` are starting settings, not tuned robot-specific values. Size `1` permits a single-model ablation. Setting the coefficient to `0` disables the deduction. MOPO's uncertainty score is a heuristic: it includes learned disturbance variability and does not guarantee detection of every OOD state/action. The norm uses the existing state coordinates without extra normalization.

Ordinary PhysX `Isaac-*` tasks do not receive this penalty. For their C2RL-PPO runs, learned dynamics supply contraction calculations and optional value-expansion rollouts; policy optimization transitions and evaluation remain PhysX. `Isaac-Classic-*` keeps its analytical transitions.

The same task implementation computes observations, reference/history context, reward, failure termination and autoreset for both transition backends. Reset uses the existing robot initialization and kinematics; learned transitions do not call the PhysX dynamics step. Initial distributions, bounds, nominal references, PD-action coordinates and physical gust settings are unchanged.

## Evaluation and saved artifacts

Checkpoint evaluation and `play.py` evaluate the same policy on both backends using the same seed/reset distribution and reference bank. C3M's periodic evaluations do the same. Both report the original task reward with the training-only MOPO deduction disabled, use the deadline/held-control evaluation behavior and report control-delay counts. Ordinary policy optimization remains synchronous.

| W&B section | Transition backend |
| --- | --- |
| `Simulation Evaluation` | Frozen learned stochastic dynamics |
| `Sim-to-Real Evaluation` | Original PhysX dynamics and configured gusts |

Continuous viewer playback evaluates the policy in PhysX. Final evaluation writes `eval_results_simulation.json`, `eval_results_sim_to_real.json` and the combined `eval_results.json`. Each run keeps `transfer_dynamics_data.npz`, `transfer_dynamics_fit.json` and `checkpoints/transfer_dynamics.pt`. The agent checkpoints also contain the shared simulator weights. Fitting history is logged under `Dynamics Training` once the agent initializes its logger. Generated datasets/reports remain in the ignored run directory.

Playback and resumed training require the fitted `transfer_dynamics.pt` beside the agent checkpoint; they never silently refit a new simulator. The configured ensemble size and network architecture must match the saved model. Legacy single-model checkpoints require `dynamics_ensemble_size: 1`. A new training run can reuse a fitted simulator through `--dynamics_checkpoint`, or reuse a compatible dataset through `dynamics_pretrain_data_path`.

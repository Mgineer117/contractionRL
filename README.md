# contractionRL

Contraction-based reinforcement learning. Algorithms — PPO, SAC, MBPO, ContractionPPO, C3M, LQR, SD-LQR,
CV-STEM-LQR, C2RL-PPO, C2RL-SAC — run through one [skrl](https://github.com/Toni-SM/skrl)
backend across two environment families that share a single interface, so any algorithm trains and
evaluates on any environment and the reported stability metrics are directly comparable.

- **Isaac envs** — quadruped / humanoid / manipulator velocity- and path-tracking. Need Isaac Lab.
- **Exact classic plants in Isaac** — `Isaac-Classic-Car-v0`, `Isaac-Classic-Car-v1`, `Isaac-Classic-Quadrotor-v0`, `Isaac-Classic-Cartpole-v0`, and `Isaac-Classic-Segway-v0` reuse the analytical equations, bounds, time step, references, and rewards, with their own algorithm configs. Isaac displays a blue body and green reference point; the markers do not add contact or rigid-body dynamics. The old Isaac `Cartpole-v0` is no longer registered.

Each Isaac task owns a directory under `tasks/direct/`, with its registration, environment, configuration, and `agents/` configs. `Classic/` contains the `isaac_classic_*/` analytical tasks. `PhyX/mobile_robots/` contains the PhysX car, cartpole, segway, and quadrotor tasks; `PhyX/complex_robots/` contains the quadruped, humanoid, and manipulator transfer tasks. Shared plant and backend support lives in `common/`. Standalone CPU-only Gym aliases are no longer registered.

Under `PhyX/complex_robots/`, transfer tasks use `isaac_transfer_path_{robot}/` and `isaac_transfer_vel_{robot}/`, where `robot` is `quadruped`, `humanoid`, or `manipulator`. Their registered environment IDs are `Isaac-Transfer-path-{Robot}-v0` and `Isaac-Transfer-vel-{Robot}-v0`.

The current state/control bounds, reference generation, rewards, disturbances, and termination rules are documented in [Isaac MDPs](docs/isaac_mdp.md). Tracking policies use compact reference parameters and MLPs, with structured nominal-plus-feedback control by default. Legacy sequence policies require their original observation layout.

PPO-family tracking policies sample `u_raw = uref + W2(s) tanh(W1(s) e) + exp(log_std) * epsilon`, with independent standard Gaussian noise per control channel. The standard deviation is learned throughout training; there is no exploration-noise schedule. The environment clips applied controls to its bounds, while PPO stores and scores the raw Gaussian samples. Deterministic evaluation uses the policy mean.

Removing privileged gust inputs changes compact observation widths (car: 40 to 35). Checkpoints trained with the older layout require their original source snapshot; new training uses the smaller layout.


List the available tasks, algorithms, and key settings without starting Isaac Sim:

```bash
python scripts/list_experiments.py
python scripts/list_experiments.py --env Isaac-Classic-Car-v1 --algorithm c2rl-ppo
```

## Install

```bash
python -m pip install -e source/contractionRL
```

Isaac envs additionally need [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). Isaac Sim 5.1 requires `numpy==1.26.0`; NumPy 2.x is incompatible with its native video-capture bindings.
The contraction-metric SDP defaults to **SCS** (bundled with `cvxpy`); **MOSEK** is tighter and faster
if you have a license — put it at `~/mosek/mosek.lic` and set `cm_solver: MOSEK`.

```bash
python scripts/list_envs.py --keyword classic   # list envs without loading Isaac Sim
```

## Quick start

```bash
# Analytical transitions rendered in Isaac Sim.
python scripts/skrl/train.py --task Isaac-Classic-Car-v0 --algorithm c2rl-ppo --num_envs 4

# Isaac: pretrain locomotion on velocity-tracking, then do contraction control on path-tracking
python scripts/skrl/train.py --task Isaac-Transfer-vel-Quadruped-v0  --algorithm ppo --num_envs 4096
python scripts/skrl/train.py --task Isaac-Transfer-path-Quadruped-v0 --algorithm c3m
# Exact car equations batched on GPU inside Isaac Lab .
python scripts/skrl/train.py --task Isaac-Classic-Car-v0 --algorithm ppo --num_envs 128
# Isaac Sim runs headless by default; pass --headon to render a window.

# Evaluate / watch
python scripts/skrl/play.py --classic --task classic-car-v0

# Multi-seed with aggregated CIs, and hyperparameter sweeps
./commands/run_seeds.sh --algorithms c2rl-ppo --env car --seeds 10 --gpu 0 -y
./commands/search.sh --algorithm c3m --env all --gpu 0 -y
```

Task names select the transition model: `Isaac-Classic-*` uses analytical equations rendered in Isaac, while `Isaac-*` uses PhysX. Both families provide Car-v0, Car-v1, Quadrotor-v0, Cartpole-v0, and Segway-v0 and support parallel GPU environments. Each pair shares the same public state, control meanings and bounds, observations, resets, reference generator, rewards, disturbances and episode rules; only dynamics processing differs. See [backend choices and MDPs](docs/isaac_backends.md) for all five tasks, commands, controls and checkpoint compatibility.

The native robot tasks use `Isaac-Transfer-path-<Robot>-v0` and `Isaac-Transfer-vel-<Robot>-v0`, where `<Robot>` is `Quadruped`, `Humanoid`, or `Manipulator`. Velocity tasks generate nominal references for the corresponding path-tracking tasks.

Isaac RL defaults are car v0/v1 **500k**, cartpole/segway **1M**, PhysX quadrotor **1M**, analytical `Isaac-Classic-Quadrotor-v0` **2M**, manipulator **3M**, quadruped **4M**, and humanoid **5M** batched training steps. Robot budgets apply to velocity and path tracking. PPO, SAC, C2RL-PPO/SAC, ContractionPPO and MBPO use these defaults; offline controllers keep their own horizons. Explicit `--num_timesteps`, `--max_iterations`, dotted/Hydra overrides and sweep budgets take precedence. These are vector steps: 500k steps with 128 environments collect 64 million transitions. The training entry point applies task defaults before overrides and saves the resolved configuration.

## Observation and action convention

The Isaac car/quadrotor and robot path-tracking tasks store physical `s = (x, xref, xi)`. Full-turn angles integrate continuously. Networks receive current-minus-reference position and cosine/sine angle features, including reference and configured preview/history context. Structured feedback also uses cosine/sine differences. Reward is the decrease in tracking energy minimized over equivalent angular windings: identity metric for PPO, learned metric for C2RL. ContractionPPO keeps its additional detached metric-shaping reward. See [periodic tracking](docs/periodic_tracking.md) for the algorithm/file map and [Isaac MDPs](docs/isaac_mdp.md) for layouts and bounds. Old policy checkpoints require their archived source snapshot because feature meanings and widths changed.


All environments enforce applied control bounds. Tracking policies add nominal control to structured feedback; SAC and bounded C3M also apply a residual-centered squash. Clipping a contraction controller can invalidate its unconstrained contraction argument, so bounded-input feasibility must be assessed separately. Explicit `--policy_control_mode full` and `residual` retain the earlier MLP ablations; new training defaults to `structured`.

`Isaac-Classic-Car-v0` and `Isaac-Classic-Car-v1` already support GPU-batched parallel environments through `--num_envs`. A real Isaac Lab/skrl check under the previous task names verified 128 cars on `cuda:0`, separate actions, and selective autoresets ([validation evidence](experiments/isaac_car_parallel_validation/20261015T0357/manifest.json)). These tasks integrate the analytical equations in PyTorch and use Isaac Sim for the scene; they are not PhysX wheel/contact vehicle models. PhysX is not required for this GPU parallelism.

## Algorithms

| Algorithm | Learns | Summary |
|---|---|---|
| PPO / SAC | policy | Baselines using the shared structured tracking actor. |
| MBPO | dynamics + policy | [Probabilistic ensemble and one-step model rollouts](docs/mbpo.md), trained with skrl SAC; `--algorithm mbpo`. |
| ContractionPPO | policy + metric | [Released-code baseline port](docs/contractionppo.md): separate metric optimizer and contraction reward shaping. |
| LQR | nothing | Constant-gain `u = uref - K·e` from a single linearization. |
| SD-LQR | nothing | Same, with `K(x)` recomputed per state. |
| CV-STEM-LQR | metric only | Tsukamoto's CV-STEM/NCM. One joint SDP over sampled states (shared `nu`, `chi`), fit `chol(W^-1)`, then deploy `u = uref - R^-1 B' M(x)·e`. No RL. |
| C3M | metric + policy | Jointly trains a dual metric `W(x)` and a `CLActor` using sampled closed-loop and projected contraction losses. |
| C2RL-PPO / C2RL-SAC | policy | See below. |

### C2RL

One policy, two phases.

1. **Synthesize the metric, then freeze it.** CV-STEM solves a joint SDP over sampled
   states and regresses the metric network onto `{x -> W*}` when `cmg_method: cvstem`.
   With `cmg_method: ccm`, the bounded M-emitting network trains directly on C3M's projected dual C2 identities,
   projected drift contraction, and bounded-input feasibility for `E = eᵀM(x)e`, including the
   reference dynamics. This method has no learned-controller contraction loss. It uses the
   environment's applied input box, which is distinct from the reference-input box.
   C2RL-PPO, C2RL-SAC and C3M metric generators use physical state x, with periodic angle embeddings. Actor and critic tracking features retain relative positions.
   The CV-STEM solve is expensive, so its result is cached
   under `data/` and the agent **refuses to run** rather than silently re-solve when no cached
   dataset matches the config.
2. **Ordinary PPO/SAC against that frozen metric,** on the reward
   `-(E_next - E_current)`, with `E = e'M(x)e` and moving-reference tracking error `e`.
   C2RL requires unit tracking weight, zero control penalty, and disabled reward variants.

PPO, C2RL-PPO and ContractionPPO optionally use [model-based value expansion](docs/value_expansion.md) with `agent.use_value_expansion: true` (disabled by default). It expands nominal analytical or learned dynamics for three steps by default, replaces critic targets, and fits intermediate imagined states with TD-k. Actor updates and GAE use real rollout data.

Each environment owns [its PPO config](docs/ppo_configuration.md), and its C2RL-PPO and ContractionPPO configs inherit that local file. Algorithm overlays contain their own additions, and saved configs report inactive settings so they are not silently swept.

C2RL [stochastic dynamics learning](docs/stochastic_dynamics.md) uses conditional Gaussian negative log likelihood for PhysX environments. Metrics use nominal f/B only. Analytical runs skip model fitting and use exact dynamics with prescribed disturbances. Classic-derived PhysX tasks train the dynamics model for metric synthesis, retain physical gusts in policy rollouts, and use learned residuals in model rollouts. References remain nominal.

`Isaac-Transfer-path-*` instead optimize every policy in a frozen learned stochastic simulator. Each robot's local `dynamics_cfg.yaml` selects uniform state/control sampling with second-order forward differences or disturbed reference rollouts with fourth-order differences. All algorithms inherit its empirical fitting settings. Reward-based optimization includes a configurable MOPO ensemble-uncertainty deduction. Evaluation compares the original task reward in the learned simulator and PhysX under `Simulation Evaluation` and `Sim-to-Real Evaluation`, including control-delay metrics. See [transfer dynamics](docs/transfer_dynamics.md).

CCM is the C2RL-PPO default. Relative-position PPO metrics reject physical-state CV-STEM caches and historical metric checkpoints require their archived source. Select CCM explicitly with `--cmg_method ccm` or `cmg_method: ccm` under `cm:`. Its training knobs are
`ccm_epochs`, `ccm_lr`, `ccm_memory_size`, and `ccm_batch_size` under `cmg:`. CV-STEM uses `cvstem_memory_size`, `cvstem_batch_size`, and `regress_*` settings. `val_frac` and `early_stop_patience` apply to both methods. The three loss weights
are `ccm_killing_weight`, `ccm_contraction_weight`, and `ccm_feasibility_weight` (all default to 1).
CCM samples `x`, `xref` and `uref` independently and uniformly over the full physical state box and full applied control box, without clipping or restricting tracking errors. This synthesis sampling does not change the bounded nominal reference generator used during episodes.
The C2 penalty sums over actuation fields, without a `1/m` factor. `cm.cm_eps` is the shared strictness margin (currently `0.01` in the task configs): the C1 penalty is `[lambda_max(C1) + cm_eps]_+²`, and input feasibility penalizes `[min_u dot(E) + 2*lambda*E + cm_eps*E]_+²`. Energy scaling preserves zero loss at exact tracking; an absolute positive input margin would be impossible there. Projected Killing remains an equality with zero target. `cm.cm_eps` also supplies the CV-STEM SDP margin. Metric bounds and contraction rates are independent: `cm.ccm_w_lb`, `cm.ccm_w_ub`, `cm.ccm_lbd` for CCM; `cm.cvstem_w_lb`, `cm.cvstem_w_ub`, `cm.cvstem_lbd` for CV-STEM. `cmg_method` selects which set is used by synthesis, the bounded metric head, and certificate logging.
`ccm_synthesis.json` records the margin, reduction, sampling domain, validation residuals and input bounds; it is not a uniform
certificate. CCM requires finite applied-input bounds and full-column-rank actuation on its samples. Derivatives are continuous-time; the finite-step environment still needs separate validation.

The **environment** computes that reward: `_inject_ccm` hands the frozen metric to the env via
`set_ccm` and the env's own `get_rewards()` returns it, so PPO's GAE and SAC's replayed critic
target are correct with no per-algorithm reward plumbing. `_inject_ccm` raises if no env accepted
the metric — quietly training on the plain baseline reward instead would be invisible in every
metric.

The default compact deployed law is `u = uref + W2(s)·tanh(W1(s)·e)`. There is no
analytic base controller in the loop and no certificate-based warm start of `pi`.

The metric and the reward always use **raw** observations: `M(x)` and `e = x - xref` are physical
quantities, and per-dimension normalization would scale `x` and `xref` independently, distorting
`e`.

## Metrics

Every algorithm logs only `auc_mean`, `auc_stddev`, `lambda`, and `lambda_stddev` under `Evaluation/Stability`. C2RL additionally reports the same keys under `Evaluation/Stability_maha`. `lambda` is the population mean empirical decay rate; it replaces the duplicate `contraction_rate_mean` series. Standard deviations describe the per-episode population. Overshoot, contraction score, running-rate aliases and percentile series are omitted from scalar logs; internal diagnostics remain available for checkpoint selection.

When no tracked episode survives the full horizon, AUC/lambda are unavailable rather than stale or zero. Episode outcomes are reported as `Training/Learning/episode_completion`: the fraction of ended episodes that reached the time horizon without failure, from 0 to 1.

**Termination.** Each env declares a certified box `[X_TERMINATION_MIN, X_TERMINATION_MAX]`, and by
default an episode ends the first step the state leaves it. The excursion is reported as
**truncation, not termination**: skrl's GAE uses `not_done = ~terminated`, so flagging it
terminated would zero the continuation value — and since every reward here is a cost, ending early
would then be strictly better than continuing. On the truncation channel with
`time_limit_bootstrap: true` the agent is indifferent to the cut. `cartpole` and `segway` ship with
the box **disabled**: both are unstable plants started far from the reference, so with it armed
every episode ended within a few steps and no stability metric could be computed at all.

## Code layout

Algorithm implementations stay in `source/contractionRL/contractionRL/agents/skrl/`. Shared code lives in `agents/utils/`:

```text
agents/
├── skrl/                       # C2RL, C3M, ContractionPPO, LQR/SD-LQR, CV-STEM-LQR
└── utils/
    ├── contraction/            # CCM/NCM synthesis, metric networks, math, stability metrics
    ├── RL/                     # Policy/value models, networks, runner, config, env, logging
    └── system_dynamics/        # State coordinates, reference layouts, learned dynamics
```

`system_dynamics/state.py` combines angle helpers and state symmetry. `system_dynamics/reference.py` combines compact and legacy reference layouts. `system_dynamics/dynamics.py` includes the learned dynamics model and data/pretraining helpers. `RL/env.py` combines the environment wrapper and preprocessors. Active legacy RNN models remain available in `RL/networks.py`.

Use the new utility paths in new code, for example `from contractionRL.agents.utils.contraction.metric_models import BoundedCCM_Generator`. Historical `agents.skrl` utility imports and pickled class paths forward to the shared implementations through compatibility aliases.

## Configuration

One yaml per (env, algorithm), next to the env: `tasks/direct/Classic/<env>/agents/skrl_<algorithm>_cfg.yaml` or `tasks/direct/PhyX/<robot_group>/<env>/agents/skrl_<algorithm>_cfg.yaml`.

**A yaml key that is not a declared field of the algorithm's `Cfg` dataclass is not applied.**
`agents.utils.RL.config.filter_cfg_fields` warns loudly rather than dropping it silently, because a run with an
ignored key trains a different algorithm than its config describes and looks completely healthy.
Add the dataclass field alongside any new yaml knob.

Sweeps go through `commands/search.sh`; the searched space lives in `search/configs/<algorithm>.yaml`
and applies to every env, so the script itself never needs editing.

## W&B logging

All algorithms use `Evaluation/Stability/*` and `Evaluation/Reward/*` for evaluation, and `Training/Loss/*`, `Training/Policy/*`, and `Training/Learning/*` for training. Reward logging uses only `Evaluation/Reward/average_reward_mean` and `average_reward_stddev`: each completed episode contributes its total raw reward divided by its actual step count, including early failures, summarized over the latest 4096 episodes. Total and discounted-return series are omitted. Evaluation populations report `_mean` and `_stddev` (population standard deviation, zero for a single sample); confidence intervals, percentiles, medians, and min/max series are omitted from W&B. Shared metric keys omit algorithm names so W&B overlays runs in the same panels: `Training/Policy/KL divergence`, `Training/Loss/Policy loss`, `Training/Loss/Value loss`, and `Training/Learning/Learning rate` (policy/shared optimizer). SAC’s separate critic LR and Q-critic loss remain distinct from PPO’s V-value loss. CMG fitting uses the separate `CMG Training/*` section (including CCM, CV-STEM regression, and synthesis diagnostics); dynamics fitting uses `Dynamics Training/*` (including loss, validation loss, and learning rate). Algorithm-specific metrics appear only when available. PPO and C2RL use `agent.experiment.write_interval` (currently 500 for Isaac Car), measured in completed vector environment steps, with one combined scalar record per interval. `global_step` is the training axis, not W&B’s internal row counter. C2RL logs offline dynamics fitting first, then CCM or CV-STEM regression, at negative `global_step` values. The policy phase starts at `global_step=0` with its initial learning rate; subsequent metrics use completed vector environment steps. Early stopping can leave gaps in the negative range. Other algorithms retain their pretraining epoch axes. Videos use `Evaluation/video` with a separate video-step axis. Episode outcomes use the single scalar `Training/Learning/episode_completion`, measured over ended episodes in reporting windows of at least 100 vector steps; it is absent before any episode ends. Clamp/reference-clamp fractions and separate early-end/failure-rate charts are omitted. Existing runs retain their earlier logged keys.

PPO and C2RL-PPO keep skrl's loss and KL-adaptive scheduler, with an additional sampled-KL update guard when `kl_threshold > 0`. A candidate update is accepted only if every logged epoch KL and its full-rollout post-update estimate are finite and at most that threshold. Rejected candidates restore model weights, Adam state, normalization statistics, rollout targets, and RNG state before retrying at half the starting learning rate (up to seven retries, respecting the scheduler's minimum LR). If none succeeds, the update is skipped. `Training/Learning/*` reports guard retries, skipped updates, and rejected-trial KL; the two policy KL curves describe accepted updates. This bounds the measured rollout estimates, not KL throughout the state space. Exploration remains stochastic. The released ContractionPPO baseline retains its own update rule.

## Citing

```bibtex
@software{cho_contractionrl,
  author  = {Cho, Minjae},
  title   = {{contractionRL}: contraction-metric reinforcement learning on a unified skrl backend},
  year    = {2026},
  license = {Apache-2.0}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

Apache-2.0. See [LICENSE](LICENSE).

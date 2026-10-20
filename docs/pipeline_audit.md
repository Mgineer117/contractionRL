# Algorithm and environment pipeline audit

What this repo does: It trains tracking policies and contraction controllers through skrl on analytical classic plants and Isaac Lab environments. This audit assumes batched CPU/GPU environments, long training runs, checkpoint playback, and multiple independent experiments. It focuses on the experiment code in the `dev` worktree, not the manuscript or standalone SOS studies.

Audit date: October 15, 2026. The findings below are **fixed in the working tree**. The installed skrl package was not modified. This report describes confirmed integration errors; it does not assign a measured share of historical performance loss to each error.

## Must fix — corrected

1. **C2RL-PPO could update twice and bypass the requested learning start** ([c2rl.py](../source/contractionRL/contractionRL/agents/skrl/c2rl.py), `C2RLSkrlTrainer.train`).
   - **What this is:** The custom trainer collects transitions and delegates learning to the inner official PPO agent.
   - **Problem:** It called a manual policy update per chunk and also advanced PPO's automatic update hook once per chunk. This added an extra update periodically, while the manual path bypassed `learning_starts`.
   - **Fix:** Advance the official hook once per transition and remove the manual update. Regression tests require exactly three updates at steps 1, 3, and 5 for six steps with rollout length two, or only step 5 with `learning_starts=4`.
   - **If left unfixed:** The actual update count differs from the configuration and baseline PPO.

2. **Compact C2RL critics discarded the reference-generator state** ([contraction_runner.py](../source/contractionRL/contractionRL/runners/contraction_runner.py), `_build_critics`).
   - **What this is:** Compact observations include current state, reference state, and reference-generator information in `xi`; plant gusts and their timers are hidden from both actor and critic.
   - **Problem:** The older reference-window critic encoded the current reference but omitted `xi`. References with different futures could therefore receive the same value estimate even when the actor observed their difference.
   - **Fix:** Use the compact tracking-feature critic for PPO's value network and all four SAC critic/target networks. Gradient tests verify their outputs depend on `xi`; historical critic checkpoints retain an explicit legacy layout.
   - **If left unfixed:** Value targets can mix different future tracking problems and give the policy a poor training signal.

3. **Autoreset states contaminated bootstrap targets and replay** ([agent_patches.py](../source/contractionRL/contractionRL/agents/utils/RL/agent_patches.py), `patch_terminal_observations`; classic/Isaac terminal-observation producers).
   - **What this is:** An environment returns a fresh reset observation after an episode ends, but learning needs the transition's actual endpoint.
   - **Problem:** PPO timeout bootstrapping and SAC replay could use the reset observation. Isaac paths also needed to expose the pre-reset endpoint consistently.
   - **Fix:** Store/bootstrap `final_observation` for ended rows while leaving the next rollout observation unchanged. True termination takes precedence if both termination and timeout flags are present. Actual PPO and SAC memory tests check the endpoint and resulting timeout reward.
   - **If left unfixed:** The learner joins unrelated episodes when forming targets.

4. **PPO action clipping disagreed with its recorded action probability** ([models.py](../source/contractionRL/contractionRL/agents/utils/RL/models.py); [contraction_ppo.py](../source/contractionRL/contractionRL/agents/skrl/contraction_ppo.py)).
   - **What this is:** PPO stores a sampled Gaussian action and its probability for later likelihood ratios.
   - **Problem:** Custom compact actors forced model-side clipping despite `clip_actions: false`. A clipped action and the original Gaussian likelihood do not describe the same distribution.
   - **Fix:** Preserve the raw Gaussian sample for PPO bookkeeping and clip only the applied plant control. Reject incompatible clipping requests on compact PPO actors. Existing regression tests check PPO, C2RL-PPO, and ContractionPPO action replay and plant clipping.
   - **If left unfixed:** PPO can optimize incorrect action likelihood ratios. This was a custom-model integration error.

5. **The spectral sigmoid could produce NaN derivatives** ([metric_models.py](../source/contractionRL/contractionRL/agents/utils/contraction/metric_models.py), `BoundedCCM_Generator._to_bounded_spd`).
   - **What this is:** The metric generator bounds eigenvalues structurally using a sigmoid.
   - **Problem:** Backpropagating through eigenvectors at repeated eigenvalues can produce undefined derivatives, although the spectral matrix function itself is smooth. Adding a multiple of the identity does not remove the degeneracy.
   - **Fix:** Evaluate the same matrix sigmoid with matrix exponential/solves and repeated doubling during differentiation. Retain the fast eigendecomposition for inference. Tests check first and second derivatives at repeated eigenvalues and agreement with the original spectral map, including large logits.
   - **If left unfixed:** CCM derivative losses can become nonfinite despite a finite, positive definite forward metric. Both spectral bounds remain structural; no upper-bound penalty was added.

6. **Some metric consumers confused M with its inverse W or mixed coordinates** ([contraction_runner.py](../source/contractionRL/contractionRL/runners/contraction_runner.py), `_make_cmg_bounds_fn`; [models.py](../source/contractionRL/contractionRL/agents/utils/RL/models.py), analytic potential).
   - **What this is:** C2RL emits the energy metric `M`; C3M emits the dual metric `W=M^-1`.
   - **Problem:** Bound/potential helpers could invert an already emitted `M`, or evaluate it using rotated errors without transforming the metric.
   - **Fix:** Respect `outputs_metric` and evaluate the potential using the wrapped physical-coordinate error. Compact critics now honor their analytic-potential option; incompatible value normalization is rejected for C2RL.
   - **If left unfixed:** Plots or critic potentials can use a different energy from the environment reward.

7. **The declared action space was smaller than the plant's applied input box** ([analytical_env.py](../source/contractionRL/contractionRL/tasks/direct/common/analytical_env.py)).
   - **What this is:** The action space determines bounded actors' output range and controllers' actuator limits.
   - **Problem:** Classic environments declared the nominal-reference input box instead of the actual actuator box. Bounded policies could lose half the available authority.
   - **Fix:** Declare `U_MIN/U_MAX`; keep the smaller reference-control bounds separate. Tests compare declared and applied bounds across classic plants.
   - **If left unfixed:** The learner, reference generator, and bounded-input feasibility loss disagree about available control.

8. **CLI overrides and sweep values did not consistently reach the algorithm** ([train.py](../scripts/skrl/train.py), `_apply_agent_overrides` and entry points).
   - **What this is:** CLI/YAML/sweep settings must resolve before constructing the model, environment, and agent.
   - **Problem:** GAE used the wrong `lambda` key; some network/critic overrides and Isaac dotted `--agent.*` arguments were dropped; Isaac sweep samples were not applied. An absent empirical-dynamics flag could overwrite YAML, and Isaac `--no_wandb` did not reliably override YAML.
   - **Fix:** Use `gae_lambda`, apply the shared overrides to both entry points, distinguish absent flags from false, and apply sweep values before construction. Derive iteration budgets after rollout overrides. Real CLI tests inspect saved effective settings for PPO, SAC, and C2RL-PPO.
   - **If left unfixed:** A command can appear to test one configuration while training another.

9. **Normalization, exploration, and scheduler options were overridden or ignored** ([config.py](../source/contractionRL/contractionRL/agents/utils/RL/config.py); [agent_patches.py](../source/contractionRL/contractionRL/agents/utils/RL/agent_patches.py)).
   - **What this is:** These settings control the scale and schedule of policy updates.
   - **Problem:** Backbone defaults overrode explicit annealing settings; standalone reward normalization was dropped; C2RL value-normalization choices could disappear. A diagnostic hook replaced non-KL schedulers, and the inner scheduler-name resolver overwrote the run-name variable.
   - **Fix:** Remove policy-noise scheduling entirely: PPO learns its exploration standard deviation. Construct the requested reward scaler, preserve configured learning-rate schedulers, and resolve scheduler classes without changing run identity. Unsupported physical-state normalization fails clearly. Tests exercise a real StepLR step, reward scaling, Gaussian sampling, and learnable exploration noise.
   - **If left unfixed:** Hyperparameter comparisons can change exploration, learning rate, or target scale unintentionally.

10. **Network settings differed between compact and legacy paths** ([runner.py](../source/contractionRL/contractionRL/agents/utils/RL/runner.py); [models.py](../source/contractionRL/contractionRL/agents/utils/RL/models.py)).
    - **What this is:** Shared model factories select policy/critic architectures from configuration.
    - **Problem:** Some legacy actors discarded configured activations. Compact paths accepted sequence options that did nothing, and full-control actors did not consistently use the shared tracking features.
    - **Fix:** Forward activations and validate names; use tracking features for new compact actors in every control mode; reject incompatible sequence settings. Keep legacy RNN code and preserve old actor feature semantics when loading/resaving historical checkpoints. Shipped compact C2RL-PPO configs now select MLPs explicitly.
    - **If left unfixed:** Architecture sweeps can silently train the same model, and old checkpoint playback can change policy behavior.

11. **C2RL checkpoints omitted inner learner state** ([c2rl.py](../source/contractionRL/contractionRL/agents/skrl/c2rl.py), checkpoint registration).
    - **What this is:** The outer C2RL agent owns metric synthesis while inner PPO/SAC owns optimizers, preprocessors, and SAC targets.
    - **Problem:** Saving only outer model references omitted important inner training state.
    - **Fix:** Register the inner checkpoint modules as well as metric/dynamics modules. Save/load tests inspect actual PPO optimizer/value-preprocessor and SAC optimizer/target-network entries.
    - **If left unfixed:** Continuation starts from a different optimizer or target network. This fix does not claim exact continuation of replay, rollout position, RNG, or scheduler history.

12. **Playback and public-runner paths could use different models or MDP settings** ([play.py](../scripts/skrl/play.py); [train_utils.py](../scripts/skrl/train_utils.py); [contraction_runner.py](../source/contractionRL/contractionRL/runners/contraction_runner.py)).
    - **What this is:** Reproducing a checkpoint requires its architecture, task, reference layout, disturbance settings, and normalization choices.
    - **Problem:** Classic training did not save a resolved MDP/model config; checkpoint discovery did not sufficiently restrict the task/algorithm; the public PPO/SAC runner bypassed the custom tracking factory. Playback could restore different defaults or use training random-action warmup.
    - **Fix:** Save effective classic settings, restore them in playback, detect legacy critic layouts, filter discovery, and use the same model factory/wrapper in the public runner. Measurement/playback bypass training warmup.
    - **If left unfixed:** Evaluation can measure a different policy or environment from the one trained. Unidentified legacy checkpoints require an explicit path.

13. **C3M ignored buffer capacity and dropped the last partial batch** ([c3m.py](../source/contractionRL/contractionRL/agents/skrl/c3m.py); [contraction_runner.py](../source/contractionRL/contractionRL/runners/contraction_runner.py)).
    - **What this is:** C3M trains over a sampled buffer in minibatches.
    - **Problem:** The configured memory size did not control actual sampling; floor division omitted the tail of non-divisible buffers.
    - **Fix:** Wire and validate memory size, and include the final partial batch. Tests require seven actual samples for capacity seven and confirm all seven participate with batch size three.
    - **If left unfixed:** Data-volume settings do not describe training, and small buffers can receive no updates.

14. **Evaluation could report misleading AUC for failed or incorrectly normalized episodes** ([train_utils.py](../scripts/skrl/train_utils.py), `_evaluate_classic_path_tracking`).
    - **What this is:** AUC integrates the tracking-error norm relative to its initial value.
    - **Problem:** The normalization could use the wrong initial error, short failed episodes could look artificially good, and evaluation did not consistently reproduce the training MDP or bypass random warmup.
    - **Fix:** Use reset-time error, retain failure/completion information, report AUC/rate over completed episodes, and return unavailable values when none finish. Copy the training reference/disturbance/timing settings. Tests check a known two-step integral and an all-failure case.
    - **If left unfixed:** A policy that fails early can appear more stable. Always read completion alongside conditional AUC.

15. **Reward logging used an unwired discount and eventually stopped updating** ([metrics.py](../source/contractionRL/contractionRL/agents/utils/contraction/metrics.py); runner initialization).
    - **What this is:** The statistics wrapper tracks discounted episode returns in a bounded history.
    - **Problem:** No caller supplied the algorithm's actual discount. Once the 4096-entry history filled, a length-based change detector stopped publishing new values.
    - **Fix:** Pass the effective discount and use a monotonic completion counter; clear unfinished return accumulators on external reset. Regression coverage checks successive completions after the history is full.
    - **If left unfixed:** Logged return can use the wrong objective and freeze during long runs.

16. **Backend/reference switches and seeds had inconsistent contracts** ([backend.py](../source/contractionRL/contractionRL/tasks/direct/common/isaac/backend.py); [train.py](../scripts/skrl/train.py); [analytical_env.py](../source/contractionRL/contractionRL/tasks/direct/common/analytical_env.py)).
    - **What this is:** Backend, trajectory layout, and seed determine dynamics/data identity.
    - **Problem:** Reapplying PhysX cache configuration nested its prefix; some compact legacy switches were accepted without effect; reference-mode validation could precede the actual override. Classic reset seeds and public custom-agent seeds were not consistently applied.
    - **Fix:** Make backend configuration idempotent, validate final reference settings, reject ignored compact reward/window/termination switches, record the actual compact layout, and seed before reference/model construction. Apply fixed-reference selection to native Isaac tracking tasks as well.
    - **If left unfixed:** Equivalent commands can select different data or fail to reproduce initial conditions. Classic reset currently seeds the global Torch RNG rather than a separate generator per environment.

17. **Native Isaac flat-observation agents selected the wrong model path** ([runner.py](../source/contractionRL/contractionRL/agents/utils/RL/runner.py); [models.py](../source/contractionRL/contractionRL/agents/utils/RL/models.py)).
    - **What this is:** Native robot velocity tasks use a flat observation instead of a tracking-reference dictionary.
    - **Problem:** Their SAC `mlp-squashed` option fell through to stock unsquashed Gaussian construction. PPO could receive custom `control_mode`/`anneal_stddev` arguments that skrl's stock model factory does not accept.
    - **Fix:** Reuse the bounded SAC actor with a flat MLP input and the existing exact squashing probability correction. Consume tracking-only defaults; PPO exploration standard deviations remain learnable. Eight tests instantiate the shipped robot/legacy cartpole PPO/SAC configs and execute real updates on representative flat tensors.
    - **If left unfixed:** PPO can fail during construction and SAC trains a distribution different from the configured bounded policy. Checkpoints from the old native SAC model need their old architecture for playback.

## Parameter contracts checked

| Setting | Actual consumer / enforced behavior |
| --- | --- |
| `rollouts`, `learning_starts`, `learning_epochs`, `mini_batches`, `gae_lambda` | Inner official PPO; transition-level update hook and CLI regression checks |
| SAC batch size, gradient steps, replay capacity, target networks | Official SAC; actual transition/update checks on all nine plants |
| Discount | Policy/critic targets plus discounted-return statistics; iteration/window metadata use the resolved value |
| Policy/critic layers and activations | Shared model factories; unsupported names fail |
| Native flat-observation PPO/SAC models | Stock Gaussian PPO / bounded squashed SAC; custom options do not leak into stock constructors |
| Compact observation / legacy encoder options | Compact MLP actor/critic consume generator state; incompatible sequence settings fail |
| Full/residual/structured control | Shared tracking features for new compact actors; environment clips applied control |
| `cmg_method`, CCM weights/rate/input bounds | CCM synthesis or the separate CV-STEM path; metric injected before RL collection |
| `w_lb`, `w_ub` | Structural dual bounds; C2RL emits M with reciprocal bounds, C3M emits W |
| `mode` | `analytical` uses exact dynamics; `sim_to_real` learns dynamics; PhysX-only tasks force `sim_to_real` |
| Disturbance / reference / timestep / horizon | Environment constructor, saved classic config, evaluation and playback restoration |
| Value/reward normalization, exploration schedule | Shared configuration translation; unsupported state normalization rejected |
| C3M memory size and minibatch size | Actual sampled buffer and all samples in each epoch |
| Checkpoint / reference-bank identity | Model-layout restoration, task/algorithm filtering, backend cache namespace |

All 95 shipped agent YAML files were checked for undeclared/untranslated agent fields. This check catches stale names, while the transition/update and CLI tests verify important values reach their runtime consumers. It is not an exhaustive proof that every possible user-supplied combination is meaningful.

## Metric formulation retained

Both C2RL and C3M metric networks receive the full physical state, including position. Their positive definiteness and lower/upper spectral bounds are structural. C2RL's CCM method uses projected dual C1/C2 conditions and bounded-input energy feasibility; it does not add a controller-Jacobian contraction loss or impose the full strong Killing identities. CV-STEM remains a separate SDP-target regression path.

The input-box minimization uses the actual applied limits: `a + sum(min(b_i*u_low_i, b_i*u_high_i))`. If the energy-decay inequality holds throughout the chosen domain, it establishes the existence of an admissible instantaneous input. Sampled training losses do not establish that uniform premise, guarantee that PPO selects the feasible input, or prove state-box invariance. A continuous-time loss also needs separate validation for held, discrete-time controls.

## Validation and performance evidence

The test suite includes real finite model outputs, transitions, replay/rollout insertion, and updates for PPO, SAC, C2RL-PPO, and C2RL-SAC on the five analytical task variants backing the Isaac environments, using compact layouts. C3M buffer/batch behavior, contraction losses, ContractionPPO, backend/model metadata, reference bounds, disturbance, and logging also have dedicated tests. Three subprocess tests exercise the real classic training CLI and inspect its saved effective configuration.

Native flat-observation model checks cover PPO and SAC configs from quadruped, humanoid, manipulator velocity tracking, and the retained legacy cartpole configuration. These tests execute learning updates with representative tensors; they do not simulate the robots.

Final verification: **233 tests passed in 46.84 seconds**, including 89 pipeline regression cases. Python compilation and `git diff --check` passed. A Pyflakes scan of scripts, package code, and tests reported no concrete undefined-name or use-before-assignment errors; 40 warnings remain, mostly registration imports/re-exports, plus unused variables and wildcard-import analysis limits. The checker was installed under `/tmp`, without changing project dependencies.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/minjae/miniconda3/envs/env_isaaclab/bin/python -m pytest tests -o addopts='' --tb=short -q
```

Tests use tiny temporary configurations and do not duplicate research experiments. The separate completed classic-car PPO study is recorded in [its diagnostic report](../experiments/classic_car_ppo/20261015-seed42/report.md), manifest, and the [private experiment dashboard](https://minjae-experiment-observatory.mj-research117.chatgpt.site). It used an earlier frozen source snapshot, before most fixes in this audit.

That study used 50,000 steps, 128 environments, no disturbance, one training seed, and a shared 512-case evaluation bank. Normalized-error AUC was 11.43 ± 2.90 for the structured actor, 11.60 ± 2.99 for the full-control MLP, and 13.41 ± 6.34 for the MLP with gamma 0.99; all completed 512/512 cases. An independent linear-feedback plant check obtained 1.80 ± 0.42. Thus PPO runs successfully, but tracking remains weak; the feedback check is not a matched training baseline. These results do not establish that MLPs, residual control, discount, or any one integration bug caused the historical C2RL gap.

The shipped car PPO/C2RL-PPO discount of 0.01 is a valid but nearly myopic modeling choice. With forward Euler, steering changes heading before it affects position; the verified one-step reward check illustrates this delay. The one-seed gamma-0.99 comparison was worse, so the discount alone is not an established explanation and was not silently retuned.

Verdict: Confirmed integration errors have been corrected with regression coverage. Reassess learning quality using new runs from this code before drawing conclusions from historical C2RL curves.

Not checked: A new live Isaac Sim/PhysX rollout after these fixes; long training runs from the final audited snapshot; every custom YAML combination; regenerated CV-STEM datasets or uniform certificates. Physics/backend code and tensor contracts were checked without claiming a live simulator validation. Cache metadata separates backends and numeric settings but does not include an immutable fingerprint of the plant source. Checkpoints retain more learner state but are not an exact replay/RNG/scheduler continuation format.

# PPO-family model-based value expansion

This is a PPO adaptation of [Feinberg et al. (2018), Model-Based Value Expansion](https://arxiv.org/abs/1803.00101), Section 3, Equation 2 and Algorithm 1. It uses the paper's finite-horizon value target and intermediate-state TD-k fitting. The paper's experiments use off-policy DDPG; this implementation keeps skrl PPO's real-data policy surrogate and GAE advantages.

Configure it under `agent:` in the environment's PPO YAML; C2RL-PPO and ContractionPPO inherit these settings:

```yaml
agent:
  rollouts: 64
  use_value_norm: true
  use_value_expansion: true
  value_expansion_horizon: 10
  value_expansion_batch_size: null  # Defaults to rollouts after CLI/sweep overrides.
```

The PPO configs enable MVE with a horizon of 10. Set `use_value_expansion: false` in the environment's PPO file, or through CLI/sweep overrides, to retain the original PPO value targets and perform no imagined rollouts or TD-k update. The feature requires compact `(x, xref, xi)` observations and separate actor/critic networks. Legacy reference-window/RNN code remains available with expansion disabled.

Equivalent command-line overrides are `--use_value_expansion true` and `--value_expansion_horizon 10`. With `value_expansion_batch_size: null` or the key omitted, expansion uses the configured `rollouts`: `--rollouts 64` gives a batch size of 64, and `--rollouts 128` gives 128. Set a positive integer or use `--value_expansion_batch_size 128` to override it explicitly. These options apply to PPO, C2RL-PPO and ContractionPPO. Batch size limits the number of roots processed at once, not the number of rollout states expanded. Every real rollout observation receives an expanded value target.

For each real state, sample actions from the rollout policy and imagine up to `H` nominal transitions. Compute the detached target

$$
\hat V_H(s)=\sum_{j=0}^{H-1}\gamma^j\hat r_j+\gamma^H V_{\mathrm{before\ update}}(\hat s_H).
$$

The reference parameters and time advance with the reference generator. Classic and analytical Isaac tasks use their exact Euler transition. PhysX tasks use a fitted control-affine `NeuralDynamics` model. C2RL reuses the model fitted for metric learning; PPO and ContractionPPO fit a model only when value expansion is enabled. Bank-based references use their nominal trajectories; generator-based reference evolution uses the selected nominal dynamics approximation.

Controls are clipped to the environment's applied bounds. Rewards use the frozen metric and wrapped tracking error, `E(current) - E(next)`, including the terminal transition. Imagined physical failures stop the return with no bootstrap. Time limits stop imagination and bootstrap the endpoint only when `time_limit_bootstrap` is enabled. Kinematic clipping and model-state failure tests follow the corresponding tracking environment. The model never resets or advances the real simulator.

The existing skrl PPO update consumes these targets through its normal value-normalization and value-loss path. GAE advantages, actor log probabilities and policy gradients use only real rollout data. Fixed TD-k targets are also calculated at each intermediate imagined state, with the remaining horizon `H-j`. After PPO, one additional critic-only Adam step fits these targets using the same optimizer learning rate, value-loss coefficient, mixed-precision scaler, and gradient clipping. Gradients accumulate over chunks to average the intermediate-state loss. Actor, metric and dynamics parameters receive no gradients from this step. `Training/Loss/Value expansion loss` reports the additional unscaled normalized-target MSE. This extra critic step is part of the PPO adaptation, rather than a verbatim DDPG implementation.

Imagination uses the stochastic policy, but preserves the real exploration RNG stream. The value normalizer is fitted by the standard real-state target path; intermediate targets use that frozen normalization state. Imagination applies the configured fixed reward scale. Custom callable reward shapers and a separate privileged `state_space` are rejected. A GRU critic may read shared observation context that an MLP actor ignores.

For PhysX environments, model rollouts draw additive noise from the fitted conditional Gaussian when environment disturbances are enabled. They never read or forecast the real simulator's hidden gust values or dwell times. References remain nominal. Disabling disturbances disables these model draws too. The Gaussian approximates the observed residual distribution; it does not reproduce temporal gust correlation. A short horizon limits accumulated model error but does not guarantee improved performance.

For learned `Isaac-Transfer-path-*` environments, each MVE root uses the same fixed ensemble member as its originating parallel environment for the entire imagined horizon. The time-by-environment rollout layout preserves this mapping across chunks. Disturbance draws remain stochastic; the MOPO reward deduction takes the maximum predicted standard-deviation norm across all members at each imagined state/action. See [transfer dynamics](transfer_dynamics.md).

Shared settings and config inheritance are documented in [PPO configuration](ppo_configuration.md).

# C2RL-PPO model-based value expansion

This is a PPO adaptation of [Feinberg et al. (2018), Model-Based Value Expansion](https://arxiv.org/abs/1803.00101), Section 3, Equation 2 and Algorithm 1. It uses the paper's finite-horizon value target and intermediate-state TD-k fitting. The paper's experiments use off-policy DDPG; this implementation keeps skrl PPO's real-data policy surrogate and GAE advantages.

Configure it under `agent:` in a C2RL-PPO YAML:

```yaml
agent:
  use_value_norm: true
  use_value_expansion: false
  value_expansion_horizon: 3
  value_expansion_batch_size: 512
```

`use_value_expansion` is disabled in all task configs and defaults to `false` when omitted. Set it to `true` to enable MVE; `false` retains the original PPO value targets and performs no imagined rollouts or TD-k update. The feature requires compact `(x, xref, xi)` observations and separate actor/critic networks. Legacy reference-window/RNN code remains available with expansion disabled.

Equivalent command-line overrides are `--use_value_expansion true`, `--value_expansion_horizon 3`, and `--value_expansion_batch_size 512`. These options affect only C2RL-PPO. Batch size limits the number of roots processed at once, not the number of rollout states expanded. Every real rollout observation receives an expanded value target.

For each real state, sample actions from the rollout policy and imagine up to `H` nominal transitions. Compute the detached target

$$
\hat V_H(s)=\sum_{j=0}^{H-1}\gamma^j\hat r_j+\gamma^H V_{\mathrm{before\ update}}(\hat s_H).
$$

The reference parameters and time advance with the reference generator. Classic/analytical Isaac tasks use exact analytical Euler transitions, including when an empirical model was separately selected for metric synthesis. PhysX tasks use the C2RL `NeuralDynamics` model trained during the existing pretraining phase; reference states and controls come from their nominal trajectory bank. No extra dynamics model or model-training phase is introduced.

Controls are clipped to the environment's applied bounds. Rewards use the frozen metric and wrapped tracking error, `E(current) - E(next)`, including the terminal transition. Imagined physical failures stop the return with no bootstrap. Time limits stop imagination and bootstrap the endpoint only when `time_limit_bootstrap` is enabled. Kinematic clipping and model-state failure tests follow the corresponding tracking environment. The model never resets or advances the real simulator.

The existing skrl PPO update consumes these targets through its normal value-normalization and value-loss path. GAE advantages, actor log probabilities and policy gradients use only real rollout data. Fixed TD-k targets are also calculated at each intermediate imagined state, with the remaining horizon `H-j`. After PPO, one additional critic-only Adam step fits these targets using the same optimizer learning rate, value-loss coefficient, mixed-precision scaler, and gradient clipping. Gradients accumulate over chunks to average the intermediate-state loss. Actor, metric and dynamics parameters receive no gradients from this step. `Training/Loss/Value expansion loss` reports the additional unscaled normalized-target MSE. This extra critic step is part of the PPO adaptation, rather than a verbatim DDPG implementation.

Imagination uses the stochastic policy, but preserves the real exploration RNG stream. The value normalizer is fitted by the standard real-state target path; intermediate targets use that frozen normalization state. If reward normalization is enabled, imagination uses its current scale without updating its statistics with synthetic rewards. Custom callable reward shapers and privileged critic states are rejected.

In `mode: sim_to_real`, model rollouts draw additive noise from the fitted conditional Gaussian when environment disturbances are enabled. They never read or forecast the real simulator's hidden gust values or dwell times. References remain nominal. Disabling disturbances disables these model draws too. The Gaussian approximates the observed residual distribution; it does not reproduce temporal gust correlation. A short horizon limits accumulated model error but does not guarantee improved performance. Value expansion remains disabled in the task configs.

# Conditional stochastic dynamics

In the learned-dynamics setup, C2RL learns the density `p(x_dot | x,u) = p_omega(x_dot - f(x) - B(x)u | x,u)` by negative log likelihood. The baseline disturbance model is a zero-mean diagonal Gaussian with learned conditional standard deviations. Zero mean fixes the ambiguity between drift and disturbance location. It does not separate physical noise from errors in an approximate nominal model.

```yaml
empirical_dynamics:
  emp_dynamics_memory_size: 8192
  dynamics_batch_size: 4096
  dynamics_pretrain_epochs: 100
  dynamics_lr: 0.001
```

The environment selects dynamics automatically. `classic-*` and `Isaac-Classic-*` use exact f/B without a learned dynamics or disturbance-density network. PhysX `Isaac-*` environments jointly fit f, B, and the conditional noise scale when required by C2RL or C3M. Real policy rollouts retain the prescribed simulator disturbances, and references remain nominal. Removed dynamics selectors are rejected rather than migrated silently.

Choose the matching task ID. C3M also fits learned dynamics by NLL, while its contraction/controller losses use nominal f/B.

For diagonal standard deviation sigma(x,u), the implemented objective averages over transitions and sums over state coordinates:

$$
\mathcal L_{\mathrm{dyn}}=\frac1N\sum_i\sum_j\left[\frac{(\dot x_{ij}-\hat f_j(x_i)-[\hat B(x_i)u_i]_j)^2}{2\sigma_j(x_i,u_i)^2}+\log\sigma_j(x_i,u_i)+\frac12\log(2\pi)\right].
$$

The standard deviation is positive, with log standard deviation clamped to [-7,5] for numerical stability. NLL can be negative; its sign is not a feasibility test. `get_f_and_B` and `predict_x_dot` return only the nominal model. `sample_disturbance` draws zero-mean residuals; `sample_x_dot` adds them to the nominal derivative. Unfitted disturbance heads cannot be sampled.

For PhysX environments, when no explicit data path is provided, C2RL collects exploratory rollouts from the real environment before metric fitting. Samples contain executed bounded controls and raw derivatives/endpoints captured before clipping and autoreset. This prevents terminal resets from being mistaken for dynamics jumps. The archive `stochastic_dynamics_data.npz` records trajectory IDs, dt, seed, and whether disturbances were enabled. Held-out trajectories are separated from training trajectories when multiple trajectory IDs are available. An explicitly provided archive declaring disturbances disabled is rejected for fitting the disturbance density when disturbances are enabled; nominal reference trajectories cannot identify a disturbance distribution.

Dynamics fitting and collection log before metric fitting at negative global steps for C2RL. The nominal and disturbance models are then frozen before metric and policy training. Analytical runs skip all dynamics data collection and fitting, regardless of whether prescribed simulator disturbances are enabled.

Policy training uses the models as follows:

- Analytical classic/Isaac plants use exact nominal equations for metric learning and simulation, with their prescribed gust when enabled. No learned model or learned disturbance head is used.
- PhysX policy rollouts use measured physical transitions. Classic-derived tasks add their prescribed held gust as `dt*d` in the same public coordinates as their analytical partners; native robot tasks retain their force/torque gusts. A learned derivative-space density is not treated as a force model.
- Learned model rollouts, including optional value expansion, add sampled learned residuals to the learned nominal f/B transition. Analytical model rollouts use exact f/B without a learned residual. Reference evolution is always nominal. With value expansion disabled, this model rollout path is inactive.
- Disabling environment disturbances disables learned disturbance injection too. Actor and critic never observe sampled future disturbances.

The conditional Gaussian is a one-step independent-noise approximation, not a model of hidden sample-and-hold gust memory. It has unbounded support, so a hard bounded-disturbance certificate does not follow from this density. Likelihood calibration and metric feasibility remain separate checks. Deterministic legacy checkpoints remain usable for nominal prediction and load with an unfitted disturbance head; noise sampling requires refitting it. Historical analytical noise-only checkpoints require the exact nominal callback when loaded standalone.

## CCM feasibility normalization

CCM retains the projected C1/C2 conditions, bounded total inputs and configured margins. The input residual is `r = min_u(dE/dt) + (2*lambda + cm_eps)*E`; its loss is `mean(relu(r / max(stop_gradient(E), 1e-6))**2)`. The positive denominator preserves the feasibility inequality and reweights the optimization without a gradient incentive to increase the denominator. C1/C2 weights and the sum over actuation fields are unchanged. Logs include signed normalized input-residual mean/stddev and `feasibility_violation_fraction`, alongside the existing raw maximum diagnostic. Standard deviations combine minibatch moments across the full epoch. Each logged point uses that epoch's summaries rather than averaging summaries from different metric weights. These sampled checks are not uniform certificates, even when no sampled violations remain.

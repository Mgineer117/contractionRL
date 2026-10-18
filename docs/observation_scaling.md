# Observation scaling

Physical observations remain unchanged for dynamics, rewards, contraction metrics and control synthesis. Scaling happens inside the neural model, so training and evaluation use the same features. `use_state_norm: false` disables a running observation preprocessor; it does not disable the compact model's fixed feature scaling. `use_value_norm` separately normalizes PPO's value targets.

| Algorithm / observation layout | Policy inputs | Critic inputs |
| --- | --- | --- |
| PPO, C2RL-PPO, SAC, C2RL-SAC, MBPO with compact tracking | Shared `TrackingFeatures` | Same features; SAC Q-functions also receive the applied action |
| ContractionPPO with compact tracking, structured or full actor | Shared `TrackingFeatures` | Same features |
| C3M with compact tracking | Shared `TrackingFeatures` | No critic |
| Legacy reference windows, including GRU/attention | Legacy relative-state / angular feature maps | Legacy feature maps; no running observation normalization |
| Native flat observations | Existing flat model inputs | Existing flat model inputs; no running observation normalization |
| LQR, SD-LQR, CV-STEM-LQR | Physical states; no neural actor normalization | No critic |

For finite nonconstant bounds, a raw feature uses `(z - center) / scale`, where `center = low/2 + high/2` and `scale = max(high/2 - low/2, 1e-6)`. A constant feature uses its fixed value as center and a divisor of one. Unbounded features use center zero and divisor one. Reversed or NaN bounds are rejected.

For Car v0/v1, Cartpole, Segway, Quadrotor, Quadruped and Humanoid compact policies, split the physical state into position p and remaining configuration z. Both actor and critic receive `(p-p_ref, z, xi_relative, z_ref, boundary_distances)`. Position-valued initial generator parameters become `p_initial-p_ref`; other generator parameters, trajectory IDs and time retain their meaning. Absolute reference-position slots are removed. Boundary distances are `(p-low)/(high-low)` and `(high-p)/(high-low)` on each position axis; they retain ground clearance and fixed-box location. Ground robots also retain base height in z. No position rotation or fixed episode origin is used.

The simulator and replay still carry the full physical `(x, xref, xi)` for dynamics, metric/reward evaluation and model rollouts. The shared differentiable input map changes only what enters policy/critic MLPs. Analytical car therefore has 35 stored observation coordinates and 37 policy/critic input features; analytical Quadrotor has 75 and 78 respectively. Cartpole/Segway add one feature, PhysX car adds two, PhysX Quadrotor adds three, and ground robots add two. Boundary distances make the transformation information-preserving within the known box. Translating the entire scene and its box preserves the features; moving within a fixed box changes the boundary features.

Ball-and-Beam, TORA, Two-link arm, Panda and Aircraft retain their previous physical tracking representation. Structured controllers multiply their gains by the physical wrapped tracking error, so controller derivatives keep physical units. Metrics continue to take physical x.


Running normalization of physical observations is unsupported and rejected consistently in both the training entry point and direct runner/agent configuration. This prevents double normalization, changing reference-control units, or silently enabling a preprocessor that is actually an identity for compact states.

PPO, C2RL-PPO and ContractionPPO value preprocessing uses a small subclass of skrl's scaler. It retains the stock epsilon-protected standard-deviation denominator, clipping, evaluation behavior and checkpoint buffers, but updates population variance (`correction=0`) in float64. One-sample and constant batches therefore have valid statistics. The same corrected update is used by the optional reward scaler and legacy observation-scaler utility. SAC-family critics do not use PPO's value-target scaler.

The new feature layout is checkpointed. Policies trained with the earlier absolute-reference or removed centered-coordinate layout require their archived source snapshots; the current loader rejects a representation mismatch instead of silently reinterpreting weights. Unchanged environments retain their existing checkpoint behavior. Frozen experiment source is never rewritten. These changes do not establish improved AUC or full Isaac/PhysX runtime behavior.

Regression checks are in `tests/test_observation_scaling.py`: constant and tiny bounds, single-reference bank observations, one-sample statistics, frozen evaluation, checkpoint round trips, shared actor/critic features, direct configuration validation, and real classic-environment observation layouts. The all-environment scaling probe uses short reference horizons; it does not validate their default full-horizon reference generators.

Translation and boundary regression tests are in `tests/test_relative_tracking_features.py`.

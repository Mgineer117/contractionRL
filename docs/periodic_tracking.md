# Periodic tracking: state, inputs and reward

The simulator keeps physical position and continuously lifted angles. It never projects the agent heading toward the reference. State dimension and ordering stay unchanged: the analytical car remains `(pos_x, pos_y, yaw, vel)`, so its metric remains 4 by 4. Quaternion backends expose continuously lifted measured angles; lifting assumes less than half a turn between measurements. Finite tilt/joint limits remain physical limits.

Network features replace named free-translation positions with current-minus-reference positions and replace each angular scalar with cosine and sine. Current, reference, initial-reference parameters, previews and histories all follow this convention. Absolute reference-position slots and boundary-distance features are removed. Consequently, actor/critic observations cannot distinguish absolute boundary proximity; translation invariance does not guarantee successful behavior near finite boundaries.

## Reward

`minimum_metric_error(error, metric, angle_idx)` selects integer full-turn shifts minimizing the full dense quadratic. `tracking_energy` evaluates that selected quadratic. With no metric argument it uses identity, equivalent to shortest-angle squared error. Each endpoint uses its own matching reference, metric and minimizing winding. Reward is previous energy minus next energy.

For one angle, with non-angular error d and block metric `[A b; b' c]`, the minimizing integer is `round((-b' d / c - delta_psi) / (2*pi))`. This also handles position/speed coupling and windings beyond the immediately adjacent turns. Multiple coupled angles use a joint Cholesky closest-lattice search, rather than independently rounding each angular component.

| Algorithm | Environment reward metric | Other learning changes |
| --- | --- | --- |
| PPO | Identity | Periodic relative actor/critic features and structured feedback |
| C2RL-PPO | Frozen learned physical-size metric | Same winding minimum in CMG feasibility, analytic critic potential and model-based value expansion |
| ContractionPPO | Identity | Keeps detached auxiliary shaping; its learned metric uses periodic relative features, defaults to all physical state coordinates, and minimizes over angular windings too |
| C3M / SAC / C2RL-SAC | Existing metric choice | Shared simulator, actor/critic features and energy evaluation; C3M still uses its differential contraction losses |
| MBPO | Identity | Periodic model inputs, physical-dimensional transition targets, and the same identity energy in imagined rewards |

Explicit legacy noncompact level-reward ablations and configured control/action penalties remain explicit overrides. The default compact tracking reward is the unscaled energy decrement.

## Where the implementation lives

| File | Change |
| --- | --- |
| [agents/utils/contraction/math_utils.py](../source/contractionRL/contractionRL/agents/utils/contraction/math_utils.py) | `minimum_metric_error`, `tracking_energy`: common periodic energy |
| [agents/utils/system_dynamics/state.py](../source/contractionRL/contractionRL/agents/utils/system_dynamics/state.py) | Continuous angle lifts and physical clamping/bound-check helpers |
| [agents/utils/system_dynamics/reference.py](../source/contractionRL/contractionRL/agents/utils/system_dynamics/reference.py) | `TrackingFeatures`: relative position, periodic current/reference angles and checkpoint marker |
| [agents/utils/RL/networks.py](../source/contractionRL/contractionRL/agents/utils/RL/networks.py) | Periodic structured feedback, preview and history encoders |
| [agents/utils/RL/models.py](../source/contractionRL/contractionRL/agents/utils/RL/models.py) | Matching actor/critic features; analytic critic uses the shared energy |
| [tasks/direct/common/analytical_env.py](../source/contractionRL/contractionRL/tasks/direct/common/analytical_env.py) | Analytical physical/reference integration, endpoint reward and reset energy |
| [tasks/direct/common/isaac/physx_env.py](../source/contractionRL/contractionRL/tasks/direct/common/isaac/physx_env.py) | PhysX continuous measured angles, matching endpoint reward and reset |
| [tasks/direct/common/path_tracking_base.py](../source/contractionRL/contractionRL/tasks/direct/common/path_tracking_base.py), [traj_buffer.py](../source/contractionRL/contractionRL/tasks/direct/common/traj_buffer.py) | Robot measured angles, continuous reference banks, reward/reset caches |
| [agents/utils/contraction/ccm_synthesis.py](../source/contractionRL/contractionRL/agents/utils/contraction/ccm_synthesis.py) | Select the same minimizing error before computing feasibility derivatives |
| [agents/utils/RL/value_expansion.py](../source/contractionRL/contractionRL/agents/utils/RL/value_expansion.py) | Imagined transitions and rewards match continuous-state real transitions |
| [agents/skrl/contraction_ppo.py](../source/contractionRL/contractionRL/agents/skrl/contraction_ppo.py) | Full-state periodic auxiliary metric and consistent derivative in shaping/fitting |
| [agents/skrl/mbpo.py](../source/contractionRL/contractionRL/agents/skrl/mbpo.py) | Periodic dynamics-model features and identity-energy imagined rewards |
| [agents/utils/system_dynamics/dynamics.py](../source/contractionRL/contractionRL/agents/utils/system_dynamics/dynamics.py) | Physical derivative targets from continuous pre-autoreset states; existing cosine/sine dynamics inputs remain |
| [scripts/skrl/train_utils.py](../scripts/skrl/train_utils.py) | Record the feature/energy convention and actual feature width in run metadata |

File paths above are relative to `source/contractionRL/contractionRL/`, except [scripts/skrl/train_utils.py](../scripts/skrl/train_utils.py).

## Compatibility and guarantee

Older policy checkpoints use different feature widths or meanings. They are rejected with a message to use their archived source snapshot. New checkpoints retain a periodic-feature marker. Existing full-size CMG weights can be warm starts, but old feasibility statistics do not validate the new energy.

The minimized energy is continuous and piecewise smooth. At equal-energy winding choices it can have a kink. Its active-branch derivative is used away from ties; there is no all-ties nonsmooth certificate. A state-dependent endpoint quadratic is generally not the CCM geodesic energy. Sampled feasibility checks and PPO training therefore do not constitute a formal global contraction guarantee.

Validation: 613 tests passed and 3 checks were skipped. No training experiment or native PhysX run was launched. CPU tests check energy minimization against exhaustive winding searches, continuity at branch changes, derivatives including reference motion, actor/critic periodicity and translation invariance, continuous simulation, and agreement among real/model/critic energies.

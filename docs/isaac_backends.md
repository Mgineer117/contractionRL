# Isaac analytical and PhysX backends

`Isaac-Classic-*` and the corresponding `Isaac-*` task use the **same task implementation**. Only the transition backend changes: analytical equations versus measured PhysX motion. The task ID selects the backend; algorithm configs have no dynamics-mode selector.

| Analytical task | PhysX task | State coordinates in both |
|---|---|---|
| Isaac-Classic-Car-v0 | Isaac-Car-v0 | position x/y, heading, forward speed (4) |
| Isaac-Classic-Car-v1 | Isaac-Car-v1 | same coordinates, lower speed domain (4) |
| Isaac-Classic-Quadrotor-v0 | Isaac-Quadrotor-v0 | position, world velocity, thrust, tilt angles and yaw (10) |
| Isaac-Classic-Cartpole-v0 | Isaac-Cartpole-v0 | position, pitch, position rate, pitch rate (4) |
| Isaac-Classic-Segway-v0 | Isaac-Segway-v0 | same four planar coordinates (4) |

Both wrappers construct the same classic task plant through `make_task_plant`. `BaseEnv` owns the state/control bounds, initial-state sampling, compact observation layout, reference generator, reward, action penalties, disturbance sampler, termination, horizon, reference previews, history, fixed references and autoresets. Configuration subclasses inherit the same task settings; registration selects the backend. Detailed task distributions and bounds are in [isaac_mdp.md](isaac_mdp.md).

Car and Quadrotor have exactly the same independent motion states and control meanings in both backends. The car has no independent lateral velocity or yaw-rate state. Quadrotor angular velocities are computed from its commanded angle rates, rather than carried as additional dynamic states. Each physical substep imposes those rates; each control transition embeds only the shared state into PhysX. There is no added actuator lag or torque servo. Gust values and hold times remain hidden in both backends. History inputs are available in both, and policies and certificates still require validation for the selected dynamics.

## References and resets

The paired Isaac tasks support `reference_mode=stabilizing`; contractive migration remains available on pure classic tasks. Both backends draw the same initial reference states and harmonic coefficients, then use the same nominal control generator and rejection rules. Each rolls out candidates with its own transition backend, without disturbances. Accepted paths must satisfy the shared state bounds and configured interior margin through the horizon endpoint. Paths, controls and acceptance rates can differ because the dynamics differ.

Both use the same reference pooling and fixed-reference behavior. PhysX reference generation snapshots and restores all live body states, including simulator velocity buffers, thrust and pending actions; generating a new reference does not advance another episode. The Segway reference generator uses the same nominal upright feedback in both backends to suppress numerical drift. No backend-specific reference-ID observation or finite saved reference bank is used.

Initial deviations use the shared sampler. In particular, each quadrotor horizontal position error independently draws an equal mixture of uniform [-1.5,-0.5] and [0.5,1.5] m; height error draws an equal mixture of uniform [-1,-0.5] and [0.5,1] m. Both apply the same remaining-coordinate errors, sign rules and subsequent domain handling.

## Disturbances and rewards

The same `BoundedGust` instance type operates on the same public state coordinates in each pair. A direction is isotropic in the active normalized channels, radius is uniform in [0,1], and channel scales set the physical units. Gusts are held for a uniform duration up to one second, discretized to control steps. Quadrotor gusts affect only world acceleration, with norm at most 0.2 m/s²; no angular gust is enabled by default. Car channels are heading rate and forward acceleration. Planar channels are position acceleration and pitch acceleration. Configuration overrides and `--disable_disturbance` apply equally to both backends.

Analytical transitions use `x_next = x + dt*(f(x)+B(x)u+d)`. PhysX integrates nominal motion over its physical substeps, then applies the same additive derivative-space increment `dt*d` to the public coordinates. These are lumped state disturbances, not a physical aerodynamic wind-force model. References and system identification remain nominal. Learned stochastic residuals are confined to learned model rollouts; see [stochastic_dynamics.md](stochastic_dynamics.md).

Rewards use the same endpoint energy decrease, wrapped angular differences, frozen metric handling and configured action penalties. Both use the same clipping, failure rules, terminal observations, independent resets and time-limit truncation. This matching does not imply equal rewards for different physical trajectories or an invariant-domain certificate.

## Physical transitions

The action interval is 0.03 s for Car/Cartpole/Segway and 0.025 s for Quadrotor. PhysX uses 0.005 s substeps; analytical transitions use one Euler step per action. Dynamics parameters remain fixed.

The car is a planar 1 kg chassis with zero gravity. Commands are yaw rate and forward acceleration, each bounded by ±3, exactly as in the analytical model. Each physical substep sets angular velocity to the commanded yaw rate and projects linear velocity onto the forward direction, removing independent sideways motion. Body-forward force is `F_x=a`; PhysX integrates translation and orientation. At control boundaries, the simulator is reconstructed from the four shared coordinates. This is an ideal constrained car model, without tire dynamics or an actuator lag state.

The quadrotor is a 1 kg thrust-driven cuboid under gravity. Commands in both backends are thrust rate and roll/pitch/yaw angle rates, bounded by ±(60,10,10,10). Its shared attitude coordinates define `R = Rx(-roll) Ry(-pitch) Rz(yaw)`, giving the analytical thrust direction `[-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]`. Both wrappers use this convention for rendering; PhysX converts it to/from its quaternion state and converts the commanded angle rates to imposed body angular velocities each physical substep. Thrust integrates each physical substep within the shared thrust bounds, and PhysX integrates thrust-driven translation and orientation. Angular velocity is cleared at control boundaries and recomputed from the next command, so it adds no independent state or actuator lag. Individual rotors and aerodynamic wind are not modeled.

Cartpole and Segway use local articulated USDA assets with a passive slider and pitch hinge. Cartpole applies the bounded cart force to a hanging pendulum, consistent with the analytical restoring-gravity convention. Segway applies coupled carriage force and body reaction torque with fixed parameters from `segway_parameters()`, calibrated to the analytical upright input/gravity linearization. These physical models do not reproduce every nonlinear term of the analytical equations. Their shared task definitions remain identical.

## Training and compatibility

```bash
python scripts/skrl/train.py --task Isaac-Classic-Car-v0 --algorithm ppo --num_envs 128
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm ppo --num_envs 128
python scripts/skrl/play.py --task Isaac-Car-v0 --algorithm ppo --checkpoint /path/to/run/checkpoints/best_agent.pt
```

PhysX reports `has_analytical_dynamics=False`; C2RL/C3M use learned dynamics, and analytical f/B are never substituted for real physical transitions or reference rollouts. Nominal system identification restores complete simulator states. Car and Quadrotor cache paths use `-physx-v4` for the canonical state/control transition; planar tasks retain `-physx-v2`. Older car/quadrotor policies and learned models require retraining because the extra motion states and actuator lag have been removed. Learned dynamics remain approximations to the selected physical integrator.

Old PhysX checkpoints used different observation layouts and reward/reset rules. They require their original source snapshot; reference-ID banks are rejected by the shared task implementation. Existing run manifests remain historical. The active Mac viewer is not restarted by a source change.

Within this project, `Isaac-Cartpole-v0` names this tracking task; Isaac Lab's stock direct task remains `Isaac-Cartpole-Direct-v0`. Native Go2/H1/Panda tasks retain their separate PhysX task definitions.

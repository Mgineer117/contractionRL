# Choosing the experiment mode

The five classic-derived Isaac tasks support `--mode analytical` (default) or `--mode sim_to_real` in both `scripts/skrl/train.py` and `scripts/skrl/play.py`:

| Task | Analytical state size | PhysX state size | PhysX body |
|---|---:|---:|---|
| Isaac-Car-v0 | 4 | 6 | Planar 1 kg chassis, lateral drag, yaw-rate actuator |
| Isaac-Car-v1 | 4 | 6 | Same chassis, lower speed domain |
| Isaac-Quadrotor-v0 | 10 | 13 | 1 kg thrust-driven cuboid, body-rate actuators |
| Isaac-Classic-Cartpole-v0 | 4 | 4 | Hanging pendulum on a force-driven slider |
| Isaac-Classic-Segway-v0 | 4 | 4 | Upright body with coupled drive force and reaction torque |

Set the single top-level YAML field `mode: analytical` or `mode: sim_to_real`. The corresponding CLI override is `--mode`. Native PhysX-only environments always use `sim_to_real`. This replaces the former backend and empirical-dynamics flags.

Both backends support parallel CUDA environments. The analytical backend preserves the classic equations and displays their states in Isaac. The PhysX backend applies forces/torques and reads measured rigid-body motion from Isaac Lab's `RigidObject`; it does not call the classic plant's transition. These are simplified bodies, not wheel/suspension or individual-rotor models. The `Isaac-Classic-*` names avoid collision with Isaac Lab's stock `Isaac-Cartpole-v0`. The existing Go2/H1/Panda tasks already use PhysX and do not accept this selector.

```bash
conda activate env_isaaclab
cd /home/minjae/research/contractionRL-dev

# Exact equations, rendered in Isaac.
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm ppo --num_envs 128 --mode analytical

# Actual PhysX rigid-body integration.
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm ppo --num_envs 128 --mode sim_to_real
python scripts/skrl/train.py --task Isaac-Car-v1 --algorithm sac --num_envs 64 --mode sim_to_real
python scripts/skrl/train.py --task Isaac-Quadrotor-v0 --algorithm ppo --num_envs 128 --mode sim_to_real

# Learn dynamics and the CCM for the selected PhysX model.
python scripts/skrl/train.py --task Isaac-Car-v0 --algorithm c2rl-ppo --cmg_method ccm --num_envs 128 --mode sim_to_real

# Playback must select the same backend and retain the saved reference bank.
python scripts/skrl/play.py --task Isaac-Car-v0 --algorithm ppo --checkpoint /path/to/run/checkpoints/best_agent.pt --mode sim_to_real
```

Add `--headon` to training for a local Isaac window, or the existing video/W&B flags to record. The equivalent Hydra environment option is `env.dynamics_backend=physx`. Explicit CLI selection takes precedence. The backend is recorded in the environment config and W&B config; PhysX training directory names include `_physx`.

## Shared PhysX tracking MDP

### State

The state is \(s=(x,x^*,\xi)\), using the same compact MLP convention as the analytical backend. Here xi contains nominal trajectory ID/time and the existing internal synthetic-control offset. Gust values and their remaining hold times are hidden from both actor and critic. Flat widths are 16 for either PhysX car, 32 for the PhysX quadrotor, and 11 for either articulated planar task (24 for their analytical backend). Reference IDs are archive-local; their meaning must stay fixed when resuming or playing a policy.

### Reference generation and bounds

Before training starts, the environment rolls out nominal controls on the **same PhysX body** and records the measured states through the horizon endpoint. Candidates violating any state bound, the 5% interior margin, or finiteness are rejected. Heading has zero interior margin for the periodic car angle. For car/quadrotor up to three candidate batches are attempted; the planar tasks record one controlled equilibrium batch. the accepted bank can be smaller than the number of parallel environments. No valid candidate raises an error. During training, resets select from this bank without advancing the other live environments. The bank is finite and does not continuously redraw new trajectory shapes.

The bank is saved as `physx_reference_bank.npz` beside the run's checkpoint directory. Resume/play load the original bank, validate its model version, task, timestep, layout and bounds, and retain the same IDs. Keep this file with the policy checkpoint. A reference bank can also be supplied programmatically with `cfg.physx_reference_path`. `reference_mode=contractive` is analytical-only and is rejected for PhysX.

### Reward

\(r_k=E_k-E_{k+1}\), \(E_k=e_k^\top M(x_k)e_k\), with wrapped angular errors, identity M for PPO/SAC and the frozen learned M for C2RL. There is no additional control penalty. Terminal transition energy is computed before reset. Kinematics are clipped for numerical reward handling; positional excursions are retained in the terminal state.

### Controller and certificate scope

The default RL controller is `u = uref + W2(s) tanh(W1(s) e)`; explicit full-control and plain residual MLP modes remain available. C3M/LQR also use nominal control plus feedback. Applied controls respect the actuator box. PhysX reports `has_analytical_dynamics=False`. C3M/C2RL use learned dynamics; LQR-family methods need a compatible learned dynamics checkpoint. Analytical metric/dynamics cache paths are separated from PhysX paths. CV-STEM still requires its offline metric dataset; selecting PhysX does not create that dataset automatically.

Analytical checkpoints and CCMs do not transfer to these different state spaces. A learned control-affine approximation is not an exact PhysX dynamics certificate: actuator saturation, sampled forces, contacts and model error need separate validation. Reference margins verify sampled nominal paths, not forward invariance for arbitrary RL policies.

### Full versus residual PPO ablation

Compact policies default to the structured controller above. For the plain MLP ablation, `--policy_control_mode full` selects `pi(s)` and `--policy_control_mode residual` selects `u_ref + pi(s)`; the same MLP inputs, network size, exploration distribution and actuator clipping are retained between these two modes. Playback supports the same flag; use the mode recorded in the checkpoint's `params/agent.yaml`.

`--disable_disturbance` turns off gust values and their random dwell state during training/playback for classic-derived Isaac environments (both backends) and classic environments. Other runs keep their configured default disturbances.

For a matched analytical-car comparison, train each mode with the same seed, budget and CCM settings, then use `scripts/skrl/compare_residual.py --full <final_full_checkpoint> --residual <final_residual_checkpoint> --output <new_output_dir>`. It checks matching metric weights and evaluates both deterministic policies on one saved bank of 512 held-out reference/initial-state cases, with disturbances off. AUC is the time integral of the normalized Euclidean tracking-error norm; lower is better. It reports mean/stddev over complete first episodes, counts failures separately, and compares paired differences on common complete episodes. Different training seeds are needed to assess variation between trained models.

### PhysX dynamics identification for C2RL and C3M

Selecting `--mode sim_to_real` enables learned dynamics for C2RL-PPO, C2RL-SAC and C3M. Before metric synthesis, the environment collects measured PhysX transitions from varied states inside the state bounds. It resets each sampled state and probes at least two different bounded controls (three by default), so equilibrium references do not leave the input response unidentified. The target derivative is the wrapped state difference over one 0.005 s physics substep. This is finite-difference data from the physical simulator, not the classic analytical equations.

The collected data fit `NeuralDynamics`, with \(\hat f(x)+\hat B(x)u\); the metric/controller losses use this learned model. Repeated dynamics requests sample the cached identification set without advancing live episodes. Collection restores physical state, current action and gust; reference indices, episode counters and disturbance timers are preserved. The full dataset and task/model/seed/timestep metadata are saved as `physx_system_id.npz` in the run directory; the existing training pipeline saves `checkpoints/dynamics.pt` and logs training/validation dynamics MSE. An explicitly configured offline dynamics dataset takes precedence over online collection.

Disturbances are enabled by default during RL and controller evaluation. Reference generation and identification are nominal, with disturbance forces temporarily zeroed, so the learned nominal plant does not absorb random gusts. This identification path covers the five classic-derived PhysX tasks below. Native robot path-tracking tasks retain their separate offline trajectory-data workflow.

## Isaac-Car-v0 — PhysX

### State and bounds

\(x=(p_x,p_y,\psi,v_{x,b},v_{y,b},\omega_z)\). Position ±45 m; heading [-pi,pi); longitudinal velocity [1,2] m/s; lateral velocity ±2 m/s; yaw rate ±4 rad/s. The chassis stays in a fixed horizontal plane: scene gravity is zero, initial height is 0.2 m, and force/torque channels preserve the planar manifold.

### Control and bounds

\(u=(\omega_{command},a_{command})\in[-3,3]^2\), units rad/s and m/s². Body-forward force is a_command N for the 1 kg body; lateral drag is -6 v_y N. Yaw torque is \(I_z12(\omega_{command}-\omega_z)\), saturated to ±1 Nm. Finite inertia causes yaw-rate lag and lateral velocity can develop during turns.

### Reference generation

Initial xy uniform in [-20,20]², heading [-1,1], forward speed 1.5 m/s, lateral/yaw rates zero. Steering uses ten normalized harmonic components with nominal command bounds ±1.5. Nominal acceleration uses a speed servo `2*(v_initial-v_x)` clipped to ±1.5 m/s². Accepted references are the resulting measured PhysX rollouts. Reset adds uniform ±0.05 noise to state coordinates, wraps heading and clips to the state domain.

### Reward

Shared energy-decrease reward above.

### Transition and disturbance

Physics dt=0.005 s, six substeps per action, action interval 0.03 s. Held body-frame acceleration gust scales: longitudinal/lateral (0.05,0.05) m/s² and yaw acceleration 0.02 rad/s². Gust values and their remaining hold times are hidden simulator state. These are external force/torque disturbances; they are not added to an analytical state transition.

### Episode termination

Raw state-domain excursion or nonfinite state terminates; other terminal kinematics are clipped for numerical reporting. A 15 s horizon (500 actions) truncates. Each environment resets independently.

## Isaac-Car-v1 — PhysX

### State and bounds

Same six-state chassis and observation layout as Car-v0, with forward velocity domain [0.2,2] m/s.

### Control and bounds

Same commanded yaw-rate/acceleration controls, force law and torque bound as Car-v0.

### Reference generation

Same procedure, except initial forward reference speed is uniform in [0.3,0.6] m/s. Full measured rollouts must satisfy the v1 interior domain. Reset noise is ±0.05 per state coordinate.

### Reward

Shared energy-decrease reward above.

### Transition and disturbance

Same 0.03 s action interval, six PhysX substeps, and held acceleration gust as Car-v0.

### Episode termination

Raw domain excursion or nonfinite state terminates; 15 s/500 actions truncates.

## Isaac-Quadrotor-v0 — PhysX

### State and bounds

\(x=(p_x,p_y,p_z,v_{x,w},v_{y,w},v_{z,w},F,\phi,\theta,\psi,\omega_{x,b},\omega_{y,b},\omega_{z,b})\). Position x/y ±30 m, height [0.15,30] m, world velocities ±1.5 m/s, mass-normalized thrust F [4.905,19.62] m/s², Euler angles ±pi/3, body angular velocities ±10 rad/s. Extra angular velocities expose actuator lag.

### Control and bounds

\(u=(\dot F,\omega_{x,command},\omega_{y,command},\omega_{z,command})\), bounded by ±(60,10,10,10), units m/s³ and rad/s. The thrust actuator integrates dot F per physics substep and clips F to its bound. PhysX receives body-z thrust F N on the 1 kg cuboid. Body-rate feedback uses `12*I*(omega_command-omega)` plus gyroscopic compensation, torque-limited to ±0.2 Nm per axis. These are body-rate targets, not directly assigned Euler-angle derivatives.

### Reference generation

Initial horizontal position [-5,5]² m, height [3,8] m, velocities and angles/rates zero, F=9.81. Nominal controls use ten normalized harmonic components scaled by 0.005. Full PhysX rollouts must stay in the interior domain. Reset adds ±0.05 state noise (±0.1 for F), then wraps angles and clips to the domain.

### Reward

Shared energy-decrease reward above.

### Transition and disturbance

Physics dt=0.005 s, five substeps per action, action interval 0.025 s, gravity (0,0,-9.81) m/s². PhysX integrates force-driven translation and torque-driven orientation. Held wind-acceleration scales are 0.2 m/s² per world axis; body angular-acceleration gust scales are 0.05 rad/s² per axis. Gust magnitude is sampled in the joint normalized channel ball and held for up to one second.

### Episode termination

Raw domain excursion, ground-clearance violation z<=0.15, roll/pitch magnitude >=pi/3, or nonfinite state terminates. The 10 s/400-action horizon truncates. Kinematic clipping is a numerical terminal rule, not an invariant-domain certificate.


## Isaac-Classic-Cartpole-v0 — analytical and PhysX

### State and bounds

Both backends use \(x=(p,\theta,\dot p,\dot\theta)\) with symmetric bounds ±(5,pi/3,1,1), in m, rad, m/s and rad/s. The policy state is the compact (x,x_ref,xi); analytical width 24, PhysX width 11. xi contains the reference-generator parameters/time (analytical) or fixed reference-bank ID/time (PhysX), plus the internal reserved control-offset fields. Plant gusts and hold times are hidden.

### Control and bounds

One full cart-force control in [-48,48] N; nominal reference controls lie in [-24,24] N. C3M/LQR still add feedback to nominal control, while PPO/SAC/C2RL generate the full bounded control.

### Reference generation and reset

The existing reference is the zero-state/zero-force equilibrium. The analytical backend preserves it exactly. PhysX records full nominal equilibrium rollouts and checks their 5% interior margin before adding them to the saved bank. Both reset from the classic initial distribution: p uniform [-2,2], tilt [0.25,0.40], velocity [-0.02,0.02], pitch rate [-0.15,-0.02], with one shared sign mirroring tilt and pitch rate. This is distinct from the stationary reference distribution.

### Reward

Exact energy decrease E_current-E_next, with identity metric for PPO/SAC and learned metric for C2RL; no action penalty.

### Transition and disturbance

Analytical: the existing classic Euler equations, dt=0.03 s, retained without changing their gravity or centrifugal signs. Acceleration disturbance scales are 0.05 m/s² and 0.05 rad/s². The classic equations have restoring gravity at theta=0, rather than the usual inverted-cartpole convention.

PhysX: six 0.005 s substeps per action. A fixed-base articulation has a passive prismatic cart joint (mass 1 kg) and a passive Y-axis hinge carrying an approximately point-mass pendulum (mass 1 kg, COM 1 m below the hinge, small positive inertia 1e-5 kg m²). Applied cart force drives the slider; the hinge receives only the disturbance torque. Position and pitch are measured joint coordinates, and velocities are measured joint velocities. Gravity is -9.81 m/s². Held disturbances apply up to 0.05 N cart force and 0.05 Nm hinge torque in the normalized disturbance ball.

The physical hanging pendulum matches the restoring convention and small-angle behavior, but it does not reproduce every nonlinear term in the existing analytical equations. In particular, the existing centrifugal terms do not coincide with the hanging point-mass mechanism. No analytical equation is silently corrected. The local authored USDA asset has no downloaded dependency; it models ideal planar constraints, not rail/contact friction.

### Episode termination

Position outside ±5 m, |theta|>=pi/3 or nonfinite state terminates. Analytical velocity coordinates are clipped; PhysX raw velocity-domain excursions also terminate before terminal kinematics are clipped for numerical reporting. Horizon: 15 s, 500 actions. Parallel environments reset independently.

## Isaac-Classic-Segway-v0 — analytical and PhysX

### State and bounds

Both backends use \(x=(p,\theta,\dot p,\dot\theta)\) with symmetric bounds ±(5,0.90,1,pi). The existing 60 s horizon is retained. Compact policy observation widths are 29 (analytical) and 16 (PhysX).

### Control and bounds

One motor-command input in [-48,48], nominal input in [-24,24]. The classic source does not supply a physical motor-unit calibration. PhysX therefore treats this as a motor command with an explicit torque gain, not as cart force or a directly assigned velocity.

### Reference generation and reset

The reference is the upright zero-state equilibrium. Analytical references follow the exact classic equations. PhysX records the actual articulated nominal rollout, using a small-angle LQR nominal controller to suppress numerical upright drift over 60 s, and accepts only complete paths within the 5% margin. Initial states follow the classic distribution: p [-2,2], tilt [0.65,0.78], velocity [0,0.05], pitch rate [-1.50,-0.60], with one shared sign mirroring tilt, velocity and pitch rate.

### Reward

E_current-E_next, identity metric for PPO/SAC and learned metric for C2RL; no extra control penalty.

### Transition and disturbance

Analytical: unchanged classic nonlinear f/B, Euler dt=0.03 s, held acceleration gust scales 0.05 m/s² and 0.05 rad/s².

PhysX: six 0.005 s substeps. A prismatic wheel-equivalent carriage and an upright body on a passive Y hinge represent ideal rolling and body tilt. The body mass is 1 kg, COM height 1 m; carriage mass, pitch inertia, motor torque gain and equivalent wheel radius are derived in `segway_parameters()` to match the classic gravity/input linearization at upright. The motor applies carriage force tau/r and body reaction torque -tau, with tau=gain*(u-drag*velocity). The velocity drag term matches the classic velocity damping near upright to within 0.4%. This is a simplified planar mechanism, not a full wheeled robot with tire contact. Disturbances are held cart force and hinge torque, scales 0.05 N and 0.05 Nm.

The upright gravity/input linearization is matched, but the full nonlinear empirical classic model is not claimed to equal this rigid-body model. Policies, learned dynamics and metrics require backend-specific training even though the four state coordinates agree.

### Episode termination

Position outside ±5 m, |theta|>=0.90 or nonfinite state terminates. Analytical remaining kinematics clip; PhysX raw velocity-domain violations also terminate. Horizon: 60 s, 2000 actions. Each environment resets independently.

```bash
python scripts/skrl/train.py --task Isaac-Classic-Cartpole-v0 --algorithm ppo --num_envs 128 --mode analytical
python scripts/skrl/train.py --task Isaac-Classic-Cartpole-v0 --algorithm ppo --num_envs 128 --mode sim_to_real
python scripts/skrl/train.py --task Isaac-Classic-Segway-v0 --algorithm ppo --num_envs 128 --mode analytical
python scripts/skrl/train.py --task Isaac-Classic-Segway-v0 --algorithm ppo --num_envs 128 --mode sim_to_real
```

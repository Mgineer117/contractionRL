# Isaac environment MDPs

The `Isaac-Classic-*` tasks use analytical transitions; the matching `Isaac-*` tasks use PhysX. Each pair shares one task implementation: state/observation layout, reset distribution, nominal reference generator, reward, bounds, disturbances, termination and horizon. The task sections below describe those shared definitions and the analytical transition; see [PhysX transition details](isaac_backends.md) for the alternative dynamics. Reference candidates use the same generator but are rolled out with their own backend.

The three `Isaac-Transfer-path-*` robot tasks train in learned stochastic dynamics and evaluate in both that simulator and PhysX. Their robot sections describe the state/control contract and physical data/evaluation plant. See [transfer dynamics](transfer_dynamics.md) for collection modes, finite differences, probabilistic fitting, decimation, checkpoints and evaluation tabs.

Implemented in the `dev` worktree. The **trajectory tracking** tasks below use compact observations and feedforward MLP policies. The three **velocity tracking** tasks remain nominal-controller training tasks for generating PhysX references; their original command rewards are listed separately. RNN and legacy reference-window implementations remain available in the source.

The robot pipeline trains `*-VelTracking-v0`, exports its genuine full-horizon PhysX rollouts to `data/<robot>/dynamics_data.npz`, then loads that archive in `*-PathTracking-v0`. Velocity training triggers export automatically; `scripts/skrl/generate_ref_traj.py` can repeat export from a saved velocity checkpoint. Use `--ref_out_dir` on training, or `--out_dir` on standalone export, to keep validation banks separate. A missing archive is an unfinished upstream generation stage. Export preserves episode failures and rejects unsafe or incomplete references. A zero quality threshold disables the reward gate; full-horizon safety checks still apply, so a safe bank from a short-run controller alone does not establish good velocity tracking.

## Shared trajectory-tracking definitions

### State and reference parameters

The conceptual state is \(s_k=(x_k,x_k^*,\xi_k)\). No future reference sequence is an observation. For both classic-derived Isaac backends, reference parameters are the initial reference state, ten harmonic coefficient vectors, and elapsed time. For robots, they are the saved nominal trajectory ID and elapsed time; that ID identifies a fixed table of physically simulated states and controls. IDs are local to the reference archive, so checkpoints must retain the same archive and ordering.

The actor and critic receive the same observation. Plant gust values and their remaining hold times are hidden simulator state and are absent from `xi`. The flat tensor order follows Gym's sorted Dict keys: `[x | xi | xref]`. Internally, xi also reserves a zero control-offset block for synthetic controller-training samples. Reference initial states have the physical state bounds, harmonic coefficients lie in [-1,1], time lies in [0,T], and trajectory IDs lie in [0,N-1]. Persistent plant disturbances remain internal to the simulator; history/RNN policies remain available for estimating their effects.

The actor/critic input map replaces free-translation position slots with current-reference errors, removes absolute reference-position slots and boundary-distance features, and translates initial position parameters relative to the current reference. Current, reference and initial-state angles become cosine/sine features. Other coordinates retain their physical meaning. Raw observations remain `(x,xref,xi)` with continuous full-turn angular lifts. Both car backends use the same compact layout, with 31 raw coordinates before optional preview/history. Absolute boundary proximity is no longer observable through these features. See [observation scaling](observation_scaling.md) and [periodic tracking](periodic_tracking.md).


### Reward

For every default tracking transition, \(r_k=E_k-E_{k+1}\). Each endpoint independently minimizes the full quadratic tracking energy over equivalent angular differences shifted by integer multiples of \(2\pi\). PPO/SAC use \(M=I\); C2RL uses its frozen learned metric. The next energy uses the matching next reference and pre-reset transition endpoint. ContractionPPO adds its detached auxiliary metric penalty. Explicit legacy level-reward ablations and configured control/action penalties remain overrides. The minimized energy is continuous but can have kinks; it is not generally the geodesic energy certified by CCM theory.


### Control

PPO, SAC, C2RL-PPO, C2RL-SAC, C3M and ContractionPPO share a structured MLP tracking controller: \(u=u^*+W_2(s)\tanh(W_1(s)e)\), followed by physical control bounds (SAC and bounded C3M use a residual-centered squash). Structured feedback uses position/speed differences and cosine/sine differences for each angle. Both gain networks receive periodic relative tracking features. Zero physical tracking error gives nominal control for the deterministic mean. Explicit `full`/`residual` modes remain available. Checkpoints with old feature layouts require their original source snapshots. LQR-family controllers retain analytical feedback. Robot controls are normalized PD-position commands, not joint torques.


### Disturbance

Reference generation never uses the plant disturbance or policy exploration noise. Analytical references follow nominal dynamics; PhysX references are simulated with gusts explicitly set to zero. Random reference parameters choose the trajectory at reset; the rollout is disturbance-free conditional on that choice. Gust values and hold durations remain private simulator state; neither the actor, critic nor reference generator receives them.

[Sun et al., Appendix E.5](https://proceedings.mlr.press/v155/sun21b/sun21b.pdf) uses piecewise-constant disturbances, random durations up to one second and random magnitudes up to a chosen maximum. Here a direction is sampled isotropically in the active normalized channels, magnitude is uniform in [0,1], and each channel is multiplied by its specified scale. Durations are sampled on the control-step grid up to one second. Directions, units and scales are implementation choices; they are not claimed to be the paper's robot parameters. References are nominal, with zero disturbance.

Analytical C2RL runs use exact dynamics and the prescribed held gust, skipping all dynamics and disturbance-density fitting. PhysX C2RL runs learn nominal f/B and a conditional residual density for metric synthesis and model rollouts. Classic-derived PhysX policy rollouts apply the same additive derivative-space gusts as their analytical partners; learned residuals apply to model rollouts only. Disturbance disabling is respected in both paths, and references stay nominal. See [stochastic dynamics learning](stochastic_dynamics.md) for collection, likelihood, and checkpoint details. The Gaussian approximation does not retain the held-gust temporal correlation or its hard support bound.

### Bounds and certification scope

References must remain in a 5% interior margin (5% of the full interval width on each nonperiodic coordinate), with the exceptions below. Invalid references are rejected, never made feasible by clipping. Robot archives additionally require compatible PhysX provenance, timestep, dimensions and PD-action coordinates, plus every state through the horizon endpoint. Existing archives are backed up before replacement.

This verifies finite-horizon sampled reference bounds. A fixed 5% margin **does not establish forward invariance for every learned policy**, nor does it certify between-sample reference safety. A rigorous invariant tracking tube requires a validated metric/controller, bounded-disturbance error radius, tube containment in the domain and a sampled-control argument; arbitrary RL controls may fail and terminate. Kinematic clipping is a numerical boundary rule and is part of the implemented transition.

Failures terminate; the horizon truncates. Autoresets start a new episode. Evaluation must not join a reset episode onto the original error curve; failure rates and full-horizon stability measurements have different populations.

### Control deadlines in transfer-path evaluation

Playback, checkpoint evaluation and C3M evaluation for `Isaac-Transfer-path-*` run policy inference in a worker. At each wall-clock sampling deadline (`step_dt`), simulation applies the latest completed control, or holds the last applied input if none is ready. A late control computed from an older observation is used at a subsequent boundary. Only one inference batch is in flight; the next request uses the latest available observation. An episode reset clears that environment's held input to zero and invalidates its pending result. Simulation and rendering stay on the main thread. CUDA inference uses a separate stream; a pending inference is never synchronized by the main thread. Physics or rendering can still run slower than real time.

Evaluation reports `control_delay_steps` (steps without a ready control computed from that step's observation), `control_hold_steps` (steps with no newly completed valid control), `control_steps` (evaluated environment transitions), `control_delay_steps_mean` (delayed steps per evaluated environment) and `control_delay_fraction`. A newly applied stale control counts as delayed but not held. Counts include the terminal transition and exclude subsequent reset episodes during a one-episode evaluation. Final evaluation saves these beside AUC in `eval_results.json` and logs them under `Evaluation/Control/`; playback saves them with its evaluation summary and prints cumulative counts in viewer mode. Counts sum over parallel environments, and therefore depend on the evaluated batch size and machine. Robot task settings are unchanged. Training rollouts retain their existing synchronous action/transition pairing.

## Isaac-Classic-Car-v0

### State and bounds

\(x=(p_x,p_y,\psi,v)\), \(s=(x,x^*,\xi)\), 31 flat observation coordinates including the internal reserved fields. Positions are [-45,45] m, heading is [-pi,pi), speed is [1,2] m/s. Heading has no reference interior margin because it is periodic.

### Control and bounds

\(u=(\omega,a)\), each in [-3,3], in rad/s and m/s² respectively. Nominal controls are bounded by [-1.5,1.5].

### Reference generation

Initial position is uniform in [-20,20]², initial heading in [-1,1], reference speed is 1.5. Steering is a sum of ten sine components at frequencies j/T cycles/s, j=1,...,10, with normalized random coefficients and bounded output; nominal acceleration is zero. The reference is generated by the same nominal Euler transition and rejected if any state violates its interior margin. Initial tracking errors are uniform in ±(1,1,1,0.5), then wrapped/clipped into the state domain.

### Reward

\(r_k=e_k^\top M(x_k)e_k-e_{k+1}^\top M(x_{k+1})e_{k+1}\); \(M=I\) for PPO/SAC.

### Transition and disturbance

Euler integration, dt=0.03 s: \(\dot p_x=v\cos\psi\), \(\dot p_y=v\sin\psi\), \(\dot\psi=\omega+d_\psi\), \(\dot v=a+d_v\). Gust scales are 0.02 rad/s and 0.05 m/s². Heading integrates continuously; speed clips. Isaac renders body/reference markers; PhysX does not replace these analytical equations.

### Episode termination

Position outside [-45,45]² or nonfinite dynamics terminates. Terminal position excursions are retained for reporting. Horizon: 15 s, 500 transitions.

## Isaac-Classic-Car-v1

### State and bounds

Same \(s=(x,x^*,\xi)\), x definition and 31-coordinate flat layout as v0. Speed bound is [0.2,2] m/s; all other bounds are unchanged.

### Control and bounds

\(u=(\omega,a)\in[-3,3]^2\), with the same units and nominal bounds as v0.

### Reference generation

Same harmonic steering generator, with initial reference speed uniform in [0.3,0.6] m/s. Speed perturbation is ±0.1 m/s; other initial errors match v0. Full reference validation uses the v1 interior bounds.

### Reward

\(r_k=E_k-E_{k+1}\), learned M for C2RL and identity for PPO/SAC.

### Transition and disturbance

Same analytical Euler equations, dt=0.03 s, speed clipping and held gust as v0. Isaac supplies visuals.

### Episode termination

Same position/nonfinite failures and 15 s/500-transition horizon as v0.

## Isaac-Classic-Quadrotor-v0

### State and bounds

\(x=(p_x,p_y,p_z,v_x,v_y,v_z,F,\phi,\theta,\psi)\), \(s=(x,x^*,\xi)\), 75 flat coordinates. Horizontal position: ±30 m; height: [0.15,30] m; world velocities: ±1.5 m/s; mass-normalized thrust F: [4.905,19.62] m/s²; all three angles: ±pi/3 rad. Both backends use the classic attitude convention \(R=R_x(-\phi)R_y(-\theta)R_z(\psi)\), with fixed-world tilt coordinates and yaw.

### Control and bounds

\(u=(\dot F,\dot\phi,\dot\theta,\dot\psi)\), bounds ±(60,10,10,10), units m/s³ and rad/s. Nominal controls have half these bounds. This is an analytical attitude-rate plant, not a motor-level quadrotor.

### Reference generation

Initial horizontal position: [-5,5]² m; height: [3,8] m; velocities: ±0.2 m/s; F=9.81; initial angles zero. Ten harmonic components per control channel use normalized random coefficients scaled by 0.005. The same nominal Euler equations generate references; full trajectories must satisfy the 5% interior margin. Both Isaac quadrotor tasks share the same reset law for these ten coordinates. Position errors independently select either sign with probability 1/2, with magnitudes uniform on [0.5,1.5] m for x/y and [0.5,1.0] m for z. Remaining errors are uniform in ±(0.15,0.2,0.3) m/s for velocity, ±0.3 m/s² for F, ±0.15 rad for roll/pitch, and ±0.5 rad for yaw. PhysX adds no initial deviation to its extra body angular-velocity coordinates. States are clipped to the domain after perturbation.

### Reward

\(r_k=E_k-E_{k+1}\), learned M for C2RL and identity for PPO/SAC.

### Transition and disturbance

Euler integration, dt=0.025 s. \(\dot p=v\); \(\dot v=(-F\sin\theta,F\cos\theta\sin\phi,F\cos\theta\cos\phi-9.81)+d_v\); the controls drive F and the three angles. This uses Isaac's positive-up world height. Both tasks use the same held world-frame acceleration gust: sample radius uniformly on [0,0.2] m/s² and an isotropic 3D unit direction, then hold the vector for a duration sampled uniformly on [0,1] s and discretized to control steps. Thus the acceleration-vector norm is at most 0.2 m/s². Other disturbance channels are zero. PhysX applies the same additive derivative-space increment after nominal integration; references remain nominal. Remaining kinematics clip at their bounds.

### Episode termination

Position outside the box, ground contact at z<=0.15, |roll| or |pitch|>=pi/3, or nonfinite dynamics terminates. Horizon: 10 s, 400 transitions. Old metrics fitted to the previous vertical sign/domain must be refitted; default metric cache paths now use `data/classic/quadrotor_zup_v2/`.

## Isaac-Transfer-path-Quadruped-v0 (Unitree Go2)

### State and bounds

\(x=(p_{xy},\psi,g_b,q-q_0,v_b,\omega_b,\dot q,h)\), n=37; \(s=(x,x^*,\xi)\), 88 flat coordinates. Bounds: p_xy ±80 m; heading [-pi,pi); gravity projection components ±1 with the physical unit-vector constraint; relative joint positions ±3.2 rad; body velocity ±(3,3,2) m/s; body angular velocity ±4 rad/s; joint velocity ±20 rad/s; base height [0.2,3] m. Adding height makes fall-relevant geometry visible.

### Control and bounds

12 normalized commands in [-1,1]. Joint PD targets are \(q_0+0.25u\) rad, subject to the robot asset's PD/effort limits.

### Reference generation

A saved, full-length **nominal Go2 PhysX rollout**, identified by trajectory ID and time. Every sample must satisfy the domain with a 5% margin; heading/gravity have zero componentwise margin, base-height clearance is 0.03 m, and projected gravity z must stay below -0.85. Reset reconstructs reference base pose/height and joint state with small pose/joint perturbations (xy/yaw ±0.1, joints ±0.05 rad). No compatible reference archive is currently present in this checkout; generate one with a trained nominal controller before running this task.

### Reward

\(r_k=E_k-E_{k+1}\), learned M for C2RL and identity for PPO/SAC.

### Transition and disturbance

PhysX articulated robot, flat ground and PD actuators; physics dt=0.005 s, four physics steps per action (0.02 s). Held world-frame force/torque is applied to the base, scales (2,2,2) N and (0.1,0.1,0.1) Nm. These share one normalized disturbance ball.

### Episode termination

Position-box exit, nonfinite state, any other raw finite-domain excursion, height below 0.2 m or projected gravity z>-0.71 terminates. Kinematic observations clip, while raw excursions still trigger termination. Horizon: 40 s, 2000 transitions.

## Isaac-Transfer-path-Humanoid-v0 (Unitree H1)

### State and bounds

\(x=(p_{xy},\psi,g_b,q-q_0,v_b,\omega_b,\dot q,h)\), n=51 for 19 joints; \(s=(x,x^*,\xi)\), 123 flat coordinates. Same component bounds as Go2, except base height [0.5,3] m. Gravity lies on the physical unit sphere.

### Control and bounds

19 normalized commands in [-1,1]. Joint PD targets: \(q_0+0.25u\), using H1 asset-specific PD and effort limits.

### Reference generation

Saved nominal H1 PhysX trajectory ID/time. Same validation and upright criteria as Go2, with 0.03 m clearance above the H1 failure height. Same reset noise magnitudes. A compatible full-length archive still needs to be generated with a trained nominal controller.

### Reward

\(r_k=E_k-E_{k+1}\), learned M for C2RL and identity for PPO/SAC.

### Transition and disturbance

PhysX, physics dt=0.005 s, decimation four, action dt=0.02 s. Held base force scales: (15,15,15) N; torque scales: (0.1,0.1,0.1) Nm, in one normalized ball.

### Episode termination

Domain excursion, nonfinite state, height below 0.5 m or projected gravity z>-0.71 terminates. Horizon: 40 s, 2000 transitions.

## Isaac-Transfer-path-Manipulator-v0 (Franka Panda)

### State and bounds

\(x=(q,\dot q,p_{ee},v_{ee},\omega_{ee,z})\), n=21; \(s=(x,x^*,\xi)\), 51 flat coordinates. Joint positions: ±4 rad; joint velocities: ±3 rad/s; local EE x/y: ±1.5 m; EE z: [0.05,1.5] m; EE world velocity: ±2 m/s; EE yaw rate: ±4 rad/s. Robot joint-limit/kinematic constraints further restrict physically realizable states.

### Control and bounds

Seven normalized commands in [-1,1], targets \(q_0+Su\), where S has 0.20 rad for joints 1–4 and 0.025 rad for joints 5–7. This matches the nominal generator's command coordinates; it is not torque control.

### Reference generation

Saved nominal Panda PhysX trajectory ID/time, complete through the endpoint and within the 5% state margin, controls within [-1,1]. Reset restores its initial arm state. A compatible archive still needs a trained nominal controller rollout; old archives with the earlier target mapping are rejected.

### Reward

\(r_k=E_k-E_{k+1}\), learned M for C2RL and identity for PPO/SAC.

### Transition and disturbance

Fixed-base PhysX Panda with position PD actuators, physics dt=1/120 s, two physics steps per action (1/60 s). Held force on the hand has scales (1,1,1) N and torque (0.1,0.1,0.1) Nm, in one normalized ball.

### Episode termination

EE position outside its workspace, nonfinite state or any other raw finite-domain excursion terminates. Kinematic observations clip. Horizon: 15 s, 900 transitions.

## Isaac-Transfer-vel-Quadruped-v0 (nominal reference generator)

### State and bounds

Policy observation: \((v_b,\omega_b,g_b,q-q_0,\dot q,c,\sin\varphi_g,\cos\varphi_g,u_{previous})\), 55 coordinates. c=(v_x^c,v_y^c,v_z^c,omega_z^c,A,omega,sin phase,cos phase), eight coordinates. This legacy policy observation is not the new full-state path observation: in particular base height is absent. Physical exports now include height (37 states). Its Gym observation box remains unbounded; numerical nonfinite values are carried forward. It is a nominal pretraining task, not a certified compact-domain tracking MDP.

### Control and bounds

12 commands in [-1,1], targets q_0+0.25u.

### Reference generation

Commands sampled each episode: v_x in [0.5,1.5] m/s, v_y=v_z=0, yaw amplitude A in [0,0.5] rad/s; omega is either 0 or 2pi/40 rad/s, equal probabilities, phase initially zero. Yaw command is A sin(omega t). Gait clock frequency is 1 Hz. Recorded physical rollouts become path references only after full-domain validation.

### Reward

Original command reward retained: 0.1 + 2 exp(-||v_xy-c_xy||²/0.25) + 0.5 exp(-(omega_z-c_yaw)²/0.25), minus 0.5||g_xy||², 0.5v_z², 0.05||omega_xy||², 0.01||u-u_previous||², 0.3||q_hip-q_hip,0||² and 10(h-0.34)², plus the centered foot-contact schedule match (scale 1). This is not the path-tracking energy reward.

### Transition and disturbance

Nominal PhysX Go2, action dt=0.02 s, no added gust during nominal reference generation.

### Episode termination

Height<0.2 m, projected gravity z>-0.71, local xy outside ±80 m, or nonfinite exported state terminates. Horizon: 40 s. Reference collection accepts only full-length nonfailed episodes and preserves the true terminal endpoint before autoreset.

## Isaac-Transfer-vel-Humanoid-v0 (nominal reference generator)

### State and bounds

Policy observation: \((v_b,\omega_b,g_b,q-q_0,\dot q,c,u_{previous})\), 74 coordinates, 19 joints and the same eight command fields. Legacy observation box remains unbounded, with a carry-forward guard. Physical exports now have 51 coordinates including base height; the policy observation itself omits height.

### Control and bounds

19 commands in [-1,1], targets q_0+0.25u.

### Reference generation

Same sampled command distributions as Go2; accepted PhysX recordings must satisfy the H1 path domain and margin.

### Reward

Original command reward: 1 + 2 exp(-||v_xy-c_xy||²/0.25) + 0.5 exp(-(omega_z-c_yaw)²/0.25), minus ||g_xy||², 0.5v_z², 0.05||omega_xy||², 1e-5||tau||² and 0.01||u-u_previous||².

### Transition and disturbance

Nominal H1 PhysX and PD control, action dt=0.02 s; no added gust for nominal reference generation.

### Episode termination

Height<0.5 m, projected gravity z>-0.71, xy outside ±80 m or nonfinite state terminates. Horizon: 40 s. Only full-length nonfailed recordings can become path references.

## Isaac-Transfer-vel-Manipulator-v0 (nominal reference generator)

### State and bounds

Policy observation: \((q,\dot q,p_{ee},v_{ee},\omega_{ee,z},c,u_{previous})\), 36 coordinates. Legacy Gym observation box is unbounded and guarded against nonfinite values. The exported physical state has 21 coordinates.

### Control and bounds

Seven commands in [-1,1], same q_0+Su mapping as Panda path tracking.

### Reference generation

Constant EE commands sampled uniformly: vx/vy in [-0.3,0.3], vz in [-0.2,0.2] m/s. Yaw amplitude in [0.1,0.5] rad/s, angular frequency [0.3,1.5] rad/s, phase [0,2pi]. Command generator fields identify the current command's future. Full nominal recordings must pass Panda path-domain validation.

### Reward

Original reward: -||v_ee-c_xyz||² - 0.5(omega_ee,z-c_yaw)² - 0.01||u-u_previous||² - 0.1 times the sum of squared normalized joint-limit excess above 90% of the soft range.

### Transition and disturbance

Nominal Panda PhysX, action dt=1/60 s; no added gust while generating nominal references.

### Episode termination

EE outside x/y ±1.5 m or z [0.05,1.5] m, or nonfinite exported state terminates. Horizon: 15 s. Only full-length nonfailed recordings can become path references.

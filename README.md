# contractionRL

Contraction-based reinforcement learning research repo built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
Integrates two complementary algorithm stacks:

| Stack | Library | Environments | Algorithms |
|-------|---------|-------------|------------|
| **skrl** (RL) | `skrl` | Isaac Sim physics envs | PPO, SAC |
| **mjrl** (contraction) | `mjrl/` (this repo) | Classic analytical + Isaac Sim | LQR, C3M |

---

## Table of Contents

1. [Installation](#installation)
2. [Listing Environments](#listing-environments)
3. [How Environments Work](#how-environments-work)
   - [Robots and task families](#robots-and-task-families)
   - [Velocity-tracking environments](#velocity-tracking-environments)
   - [Path-tracking environments](#path-tracking-environments)
   - [Base locomotion environments](#base-locomotion-environments)
   - [Classic analytical environments](#classic-analytical-environments)
4. [skrl — Isaac Sim RL (PPO / SAC)](#skrl--isaac-sim-rl-ppo--sac)
   - [Locomotion policy design](#locomotion-policy-design)
   - [Training](#training)
   - [Evaluation / Play](#evaluation--play)
5. [mjrl — Contraction Control (LQR / C3M)](#mjrl--contraction-control-lqr--c3m)
   - [Classic envs, analytical dynamics](#a-classic-envs--analytical-dynamics)
   - [Classic envs, learned dynamics](#b-classic-envs--empirically-learned-dynamics)
   - [Isaac Sim envs, learned dynamics](#c-isaac-sim-envs--learned-dynamics)
6. [Visualization](#visualization)
7. [W&B Logging](#wb-logging)
8. [Algorithm Reference](#algorithm-reference)
9. [Project Structure](#project-structure)

---

## Installation

### 1. Install Isaac Lab

Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Conda install is recommended:

```bash
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
# … follow Isaac Lab conda steps …
```

### 2. Clone this repo

```bash
git clone <repo-url> contractionRL
cd contractionRL
```

### 3. Install the contractionRL extension

```bash
python -m pip install -e source/contractionRL
```

### 4. Verify mjrl dependencies

`mjrl` is a top-level package in this repo — no separate install.
Its deps (`torch`, `scipy`, `gymnasium`, `tqdm`, `matplotlib`, `yaml`) are all
present in the Isaac Lab conda env:

```bash
python -c "import torch, scipy, gymnasium, tqdm, matplotlib, yaml; print('OK')"
```

> `scripts/mjrl/train.py` and `pretrain_dynamics.py` run without Isaac Sim.
> Only `collect_isaac_data.py` and `train_isaac.py` boot the simulator.

---

## Listing Environments

Scan task `__init__.py` files without loading Isaac Sim:

```bash
python scripts/list_envs.py

# Filter by keyword
python scripts/list_envs.py --keyword vel_tracking
python scripts/list_envs.py --keyword classic
```

---

## How Environments Work

### Robots and task families

The three Isaac Sim robots each come in three registered task variants:

| Robot | Model | Joints controlled | Registered variants |
|-------|-------|-------------------|---------------------|
| **Quadruped** | Unitree Go2 | 12 (3×4 legs: hip/thigh/calf) | base, vel_tracking, path_tracking |
| **Humanoid** | Unitree H1 | 19 (legs 11, torso 1, arms 8) | base, vel_tracking, path_tracking |
| **Manipulator** | Franka Panda | 7 (arm joints, no fingers) | base, vel_tracking, path_tracking |

Each "variant" is a **separate gym registration** with a distinct obs/reward/reset design:

- **base** — locomotion without a structured tracking goal; a starting point for custom reward shaping. Runs with skrl (PPO/SAC).
- **vel_tracking** — follow a randomly sampled velocity command each episode. Runs with skrl (PPO/SAC). Also usable for mjrl dynamics data collection.
- **path_tracking** — follow a pre-recorded state trajectory `[x_ref, u_ref]` step-by-step. Observation format `[x, x_ref, u_ref]` is contraction-compatible, so both skrl and mjrl agents can run in it.

The classic **Car-Direct-v0** is a completely separate, no-sim environment. It is not a variant of the Isaac envs — it has its own single registration and is only supported by the mjrl algorithm stack. It is described in its own section below.

---

### Velocity-tracking environments

**Tasks:** `Quadruped-VelTracking-Direct-v0`, `Humanoid-VelTracking-Direct-v0`,
`Manipulator-VelTracking-Direct-v0`

**Goal:** Follow a randomly sampled velocity command as closely as possible while
staying upright and using minimal torque.

#### Velocity commands

At each episode reset, each environment independently samples:

```
vx  ~ Uniform(vx_range)       [m/s] — forward velocity,  constant per episode
vy  ~ Uniform(vy_range)       [m/s] — lateral velocity,  constant per episode
vz    = 0                            — vertical (flat-ground assumption)
yaw_rate(t) = A·sin(ω·t + φ)       — sinusoidal yaw, makes robot curve
   A ~ Uniform(yaw_A_range)   [rad/s]  amplitude
   ω ~ Uniform(yaw_omega_range)[rad/s]  frequency
   φ ~ Uniform(0, 2π)                  phase
```

The sinusoidal yaw ensures the robot must handle turning while translating,
covering a richer portion of the state space than straight-line commands.

#### Observation space

All three vel-tracking envs follow the same layout:
`obs = [physical_state, commands(4), prev_actions]`

| Env | Physical state | + cmds | + prev_act | Total obs |
|-----|---------------|--------|-----------|-----------|
| Quadruped | lin_vel_b(3) + ang_vel_b(3) + proj_gravity_b(3) + joint_pos_rel(12) + joint_vel(12) = **33** | 4 | 12 | **49** |
| Humanoid  | lin_vel_b(3) + ang_vel_b(3) + proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19) = **47** | 4 | 19 | **70** |
| Manipulator | joint_pos(7) + joint_vel(7) + ee_pos_local(3) + ee_lin_vel(3) + ee_yaw_vel(1) = **21** | 4 | 7 | **32** |

All quantities are in the **body frame** (`_b` suffix) except manipulator
EE position which is in the env-local frame.

The **commands vector** = `[vx, vy, vz, yaw_rate(t)]`.
`vz = 0` for legged robots; manipulator uses `[vx, vy, vz, yaw_rate]` for 3D EE motion.

#### Action space

All envs: `Box(−1, 1, shape=(n_joints,))` — normalised joint position targets.

The env un-normalises actions before sending to the physics simulator:

```python
# Quadruped / Humanoid
joint_target = default_joint_pos + action_scale * action   # action_scale = 0.25 rad

# Manipulator (maps [-1,1] to within soft joint limits)
joint_target = midpoint + half_range * action
```

The simulator runs a PD controller internally to track these position targets.

#### Reward function

| Term | Formula | Scale (quadruped) | Purpose |
|------|---------|-----------------|---------|
| `rew_lin_vel` | `exp(−‖cmd_xy − vel_b_xy‖² / 0.25)` | +2.0 | Track lateral commands |
| `rew_yaw_rate` | `exp(−(cmd_yaw − ω_z)² / 0.25)` | +0.5 | Track yaw rate |
| `rew_z_vel` | `−v_z²` | −0.5 | Suppress vertical bouncing |
| `rew_ang_vel_xy` | `−‖ω_xy‖²` | −0.05 | Suppress roll/pitch oscillation |
| `rew_upright` (humanoid only) | `−‖g_proj_xy‖²` | −1.0 | Keep torso upright |
| `rew_torques` | `−‖τ‖²` | −1×10⁻⁵ | Energy efficiency |
| `rew_action_rate` | `−‖aₜ − aₜ₋₁‖²` | −0.01 | Smooth joint motion |
| `rew_alive` | `1 − terminated` | +0.5 | Survival bonus |

The exponential form for tracking rewards (`exp(−err/σ)`) gives a smooth,
dense signal: near-perfect tracking → reward ≈ scale; large error → reward → 0.

#### Termination

| Robot | Condition |
|-------|-----------|
| Quadruped | `base_height < 0.20 m` (fell) |
| Humanoid | `base_height < 0.50 m` (fell) |
| Manipulator | — (no fall termination; time-out only) |
| All | `episode_length ≥ max_episode_length` (10 s at 50 Hz decimated) |

#### Physics

All envs use `SimulationCfg(dt=1/200)` with `decimation=4` (quadruped/humanoid)
or `decimation=2` (manipulator), so the **policy step** runs at 50 Hz or 60 Hz
respectively. The physics sub-steps at 200 Hz ensure stable contact dynamics.

---

### Path-tracking environments

**Tasks:** `Quadruped-PathTracking-Direct-v0`, `Humanoid-PathTracking-Direct-v0`,
`Manipulator-PathTracking-Direct-v0`

**Goal:** Follow a pre-recorded *reference trajectory* as precisely as possible.
This is the environment used for **contraction-compatible RL** — the observation
format exactly matches what LQR and C3M expect.

#### How path tracking works

```
Episode reset:
  1. Sample a reference trajectory xref[0..T], uref[0..T] from the buffer
  2. Initialise robot state at xref[0] (teleport + velocity set)

Each step t:
  obs  = [x_current(state_dim), x_ref[t](state_dim), u_ref[t](action_dim)]
  reward = −‖x_current − x_ref[t]‖²   (pure quadratic tracking error)
  action = policy(obs)  ← policy outputs the DEVIATION from u_ref
  u_applied = u_ref[t] + action
```

The `[x_current, x_ref, u_ref]` observation format is the same one C3M and LQR
use, so an mjrl contraction policy can directly evaluate in a path-tracking env
(and vice versa — an RL policy trained here is already in the right output space).

#### Observation layout

| Env | state_dim | obs = [x, x_ref, u_ref] total |
|-----|-----------|-------------------------------|
| Quadruped path | 33 | 33 + 33 + 12 = **78** |
| Humanoid path  | 47 | 47 + 47 + 19 = **113** |
| Manipulator path | 21 | 21 + 21 + 7 = **49** |

#### Generating reference trajectories

Path-tracking envs need a pre-generated `.npz` trajectory buffer.
Use the trajectory generator script after training a velocity-tracking policy:

```bash
# Run the vel-tracking policy to collect a trajectory buffer
python scripts/generate_ref_traj.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --checkpoint logs/skrl/Quadruped-VelTracking-Direct-v0/.../best_agent.pt \
    --num_envs 64 \
    --save outputs/quadruped_ref_traj.npz \
    --headless
```

Then point the path-tracking env config at this file:

```python
# in quadruped_path_tracking/env_cfg.py
traj_path = "outputs/quadruped_ref_traj.npz"
```

---

### Base locomotion environments

**Tasks:** `Quadruped-Direct-v0`, `Humanoid-Direct-v0`, `Manipulator-Direct-v0`

Minimal locomotion environments without a structured tracking objective.
Useful for reward-shaping experiments or as a starting point for custom tasks.
Same obs/action structure as the vel-tracking envs but without explicit
velocity commands.

| Env | Obs dim | Action dim |
|-----|---------|-----------|
| `Quadruped-Direct-v0` | 48 | 12 |
| `Humanoid-Direct-v0` | 69 | 19 |
| `Manipulator-Direct-v0` | 20 | 7 |

`Contractionrl-Direct-v0` is a minimal 1D cartpole-style prototype environment
(obs=4, action=1) for rapid algorithm prototyping.

---

### Classic analytical environments

**Task:** `Car-Direct-v0`

A pure-Python gymnasium env with no Isaac Sim dependency.
Implements a Dubins-like car with exact analytical dynamics — no simulation needed.

```
State  x = [p_x, p_y, θ, v]       (position, heading, speed)
Control u = [ω, a]                 (angular rate, linear acceleration)

Dynamics (control-affine):
    ẋ = f(x) + B(x)·u
    f(x) = [v·cos(θ), v·sin(θ), 0, 0]ᵀ
    B(x) = [[0,0], [0,0], [1,0], [0,1]]

Observation = [x, x_ref, u_ref]   (concatenated for contraction compatibility)
Action      = deviation δu         (env applies u = u_ref + δu)
Episode     = 9 s at dt=0.03 s    (300 steps)
```

The reference trajectory (`x_ref`, `u_ref`) is generated at reset by integrating
sinusoidal reference controls, giving a smooth curved path.

---

## skrl — Isaac Sim RL (PPO / SAC)

### Locomotion policy design

Both PPO and SAC train a **stateless MLP** policy:
`π(obs) → action ∈ [−1, 1]^{n_joints}`

The policy never sees raw physics quantities like contact forces or absolute
position — it only sees **body-frame velocities**, **projected gravity** (tilt),
**joint positions relative to default**, **joint velocities**, and the
**current velocity command**. This makes the policy inherently invariant to
global position and heading.

```
Policy input (quadruped, 49 dims):
  [lin_vel_b(3), ang_vel_b(3), proj_gravity_b(3),   ← proprioception
   joint_pos_rel(12), joint_vel(12),                 ← joint state
   vx_cmd, vy_cmd, vz_cmd, yaw_rate(t),              ← current command
   prev_actions(12)]                                  ← action history

Policy output (12 dims):
  normalised joint position deviations ∈ [−1, 1]
  → joint_target = default_pos + 0.25 · output  [rad]
```

**Curriculum note:** The `prev_actions` channel is critical — without it the
policy has no memory of what it just commanded, making smooth gaits very hard
to learn.

#### PPO vs SAC

| | PPO | SAC |
|-|-----|-----|
| Type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Stability | High | Medium |
| Recommended for | First training runs, locomotion from scratch | Fine-tuning, sparse rewards |
| Typical `--num_envs` | 4096 | 64–256 |
| Replay buffer | None | Yes (per-env memory) |
| Entropy | Fixed clip | Adaptive (automatic temperature) |

For **locomotion from scratch**, PPO with many parallel envs (4096) is the
most reliable starting point. SAC can be faster per-sample but is more
sensitive to replay buffer sizing and learning rate.

---

### Training

```bash
# PPO — quadruped velocity tracking (recommended starting config)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --num_envs 4096 \
    --headless

# PPO — humanoid velocity tracking
python scripts/skrl/train.py \
    --task Humanoid-VelTracking-Direct-v0 \
    --algorithm ppo \
    --num_envs 4096 \
    --headless

# PPO — manipulator EE velocity tracking
python scripts/skrl/train.py \
    --task Manipulator-VelTracking-Direct-v0 \
    --algorithm ppo \
    --num_envs 2048 \
    --headless

# SAC — lower env count (replay buffer makes it sample-efficient)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm sac \
    --num_envs 64 \
    --headless

# Resume from checkpoint
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint logs/skrl/Quadruped-VelTracking-Direct-v0/<RUN>/checkpoints/best_agent.pt \
    --headless

# Limit total training steps
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --max_iterations 3000 \
    --headless

# Record videos every 2000 steps (works headless)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --video --video_length 200 --video_interval 2000 \
    --headless

# Log to W&B (videos auto-uploaded if --video also set)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm sac \
    --wandb --wandb_project contractionRL --wandb_run_name quad-sac-v1 \
    --video --video_length 200 --video_interval 2000 \
    --headless
```

**SAC hyperparameter overrides:**

```bash
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 --algorithm sac \
    --sac_lr 1e-4 --sac_batch_size 512 \
    --sac_gradient_steps 4 --sac_memory_size 100000 \
    --headless
```

**PPO hyperparameter overrides:**

```bash
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 --algorithm ppo \
    --ppo_lr 3e-4 --ppo_rollouts 24 \
    --ppo_learning_epochs 5 --ppo_mini_batches 4 \
    --headless
```

Logs and checkpoints are saved to `logs/skrl/<TASK>/<timestamp>/`.

---

### Evaluation / Play

```bash
# GUI — watch the trained policy in real-time
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint logs/skrl/Quadruped-VelTracking-Direct-v0/<RUN>/checkpoints/best_agent.pt \
    --num_envs 4

# GUI + velocity arrow overlay (blue=command, green=actual)
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint <PATH> \
    --num_envs 4 \
    --debug_vis

# Headless + record one episode as MP4
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint <PATH> \
    --num_envs 1 \
    --video --video_length 600 \
    --headless

# Path-tracking evaluation
python scripts/skrl/play.py \
    --task Quadruped-PathTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint <PATH> \
    --num_envs 4
```

---

## mjrl — Contraction Control (LQR / C3M)

`mjrl` provides model-based contraction controllers that provably converge to
any reference trajectory exponentially fast.

| Algorithm | Dynamics needed | Training | Description |
|-----------|----------------|---------|-------------|
| **LQR** | `get_f_and_B(x)` | None | Linearise + CARE solve per step |
| **C3M** | `get_f_and_B(x)` | ~5000–30000 epochs | Joint CCM + CLActor synthesis |

**Two dynamics sources** are supported:

- **Analytical** — closed-form `f(x)`, `B(x)` in the env class. Classic envs only.
- **Learned (`NeuralDynamics`)** — MLP trained on `(x, u, ẋ)` data. Required for
  Isaac Sim; optional for classic envs (useful to verify the learned pipeline).

---

### A. Classic envs — analytical dynamics

The simplest path. No pretraining needed.

#### LQR — analytical

```bash
python scripts/mjrl/train.py \
    --task Car-Direct-v0 \
    --algorithm lqr \
    --use_analytical_dynamics \
    --num_agent 4
```

**What happens step by step:**

1. `Car-Direct-v0` gymnasium env is created (no Isaac Sim)
2. 4 parallel sampler workers (processes) run independent episodes
3. At each env step the agent calls `env.get_f_and_B(x_ref)` → exact `(f, B)`
4. Jacobian `A = Df/Dx + Σ uref_j·∂B_j/∂x` computed via autograd
5. CARE solved: `AᵀP + PA − PBR⁻¹BᵀP + Q = 0` → gain `K = R⁻¹BᵀP`
6. Output `δu = −K·(x − x_ref)` (env adds `u_ref` automatically)
7. Reports `track_err_mean` and `track_err_final` across all episodes

#### C3M — analytical

```bash
# Full training run
python scripts/mjrl/train.py \
    --task Car-Direct-v0 \
    --algorithm c3m \
    --use_analytical_dynamics \
    --num_agent 4 \
    --epochs 30000

# Quick smoke-test (300 epochs, cpu)
python scripts/mjrl/train.py \
    --task Car-Direct-v0 \
    --algorithm c3m \
    --use_analytical_dynamics \
    --num_agent 2 \
    --epochs 300 \
    --device cpu
```

**What happens step by step:**

1. 65 536 random `(x, x_ref, u_ref)` triples sampled once from state/control bounds
2. Two networks jointly trained:
   - **CMG** (CCM generator): `x → W(x) = LᵀL` — PSD contraction metric inverse
   - **CLActor** (controller): `[x, x_ref, u_ref] → δu = W₂·tanh(W₁·e)`
3. Each iteration: random 1024-point mini-batch; loss enforces three conditions:
   - **Cu ≺ 0** — closed-loop contraction: `Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0`
   - **C1 ≺ 0** — drift term: `Bₗᵀ(−Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bₗ ≺ 0`
   - **C2 = 0** — compatibility: `Bₗᵀ(Ẇ_b − 2·sym(∂B/∂x·W))Bₗ = 0`
4. Eigenvalue plots logged every 500 iterations
5. Sampler eval every `eval_interval` epochs → tracking error reported

---

### B. Classic envs — empirically learned dynamics

Use this to test the full NeuralDynamics pipeline on a system where you can
cross-check against the analytical ground truth.

#### Step 1 — Pretrain NeuralDynamics

```bash
python scripts/mjrl/pretrain_dynamics.py \
    --task Car-Direct-v0 \
    --source analytical \
    --n_samples 100000 \
    --epochs 3000 \
    --hidden_dim 256 256 256 \
    --activation relu \
    --save checkpoints/car_dynamics.pt
```

**What happens:**

1. `env.get_rollout(100000, mode='dynamics')` samples `(x, u)` uniformly
   from state/control bounds, computes exact `ẋ = f(x) + B(x)u` analytically
2. `NeuralDynamics` trained: `min_θ ‖f_θ(x) + B_θ(x)·u − ẋ‖²` (MSE)
   - `f_net`: MLP `ℝ^{x_dim} → ℝ^{x_dim}` — drift
   - `B_net`: MLP `ℝ^{x_dim} → ℝ^{x_dim × u_dim}` — input matrix (flattened)
3. 10 % validation split; cosine LR annealing; best checkpoint saved
4. Architecture metadata embedded in checkpoint (self-describes on `load`)

#### Step 2a — LQR with learned dynamics

```bash
python scripts/mjrl/train.py \
    --task Car-Direct-v0 \
    --algorithm lqr \
    --dynamics_checkpoint checkpoints/car_dynamics.pt \
    --num_agent 4
```

**What happens differently from analytical:**

- `NeuralDynamics.load(checkpoint)` is called before env construction
- **Two dynamics paths** are wired up separately:
  - *Simulation path* (env.step → get_dynamics → get_f_and_B): env calls
    `env.learned_dynamics_model(x)` under `torch.no_grad()`, returns numpy
  - *Agent training path* (LQR.forward → CARE): calls
    `dynamics_model.get_f_and_B(x_ref)` with tensor input — autograd flows
    through `f_net` and `B_net` to compute Jacobians `Df/Dx`, `dB/dx`

#### Step 2b — C3M with learned dynamics

```bash
python scripts/mjrl/train.py \
    --task Car-Direct-v0 \
    --algorithm c3m \
    --dynamics_checkpoint checkpoints/car_dynamics.pt \
    --num_agent 4 \
    --epochs 30000
```

C3M contraction losses compute `Df/Dx` and `dB/dx` via autograd through the
neural networks. Everything else is identical to the analytical case.

---

### C. Isaac Sim envs — learned dynamics

No analytical dynamics for Isaac Sim; `NeuralDynamics` must be pretrained from
sim rollout data.

#### Step 1 — Collect dynamics data

```bash
python scripts/mjrl/collect_isaac_data.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --num_envs 64 \
    --n_steps 200000 \
    --save data/quadruped_dynamics_data.npz \
    --headless
```

**What happens:**

1. Isaac Sim boots with 64 parallel envs
2. At each step: uniform-random actions sampled from `action_space` bounds
3. Records `(obs_t, action_t, obs_{t+1})` tuples
4. Computes `ẋ ≈ (obs_{t+1} − obs_t) / dt` (finite-difference, 50 Hz step)
5. Masks out reset transitions (discontinuous derivative)
6. Saves `.npz`: keys `x (N, 49)`, `u (N, 12)`, `x_dot (N, 49)`

> Typical: ~10 min on RTX 3090 with 64 envs, yields ~10M transitions.
> A random policy covers the state space broadly; deterministic rollouts from
> a pre-trained policy can also be used for more realistic coverage.

#### Step 2 — Pretrain NeuralDynamics

```bash
python scripts/mjrl/pretrain_dynamics.py \
    --source npz \
    --data data/quadruped_dynamics_data.npz \
    --x_dim 49 \
    --u_dim 12 \
    --n_samples 200000 \
    --epochs 3000 \
    --hidden_dim 256 256 256 \
    --save checkpoints/quadruped_dynamics.pt
```

#### Step 3a — LQR on Isaac env

```bash
python scripts/mjrl/train_isaac.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm lqr \
    --dynamics_checkpoint checkpoints/quadruped_dynamics.pt \
    --num_envs 4 \
    --num_agent 4 \
    --Q_scaler 1.0 --R_scaler 1.0 \
    --headless
```

**What happens:**

1. Isaac Sim boots
2. `NeuralDynamics.load(checkpoint)` → x_dim=49, u_dim=12
3. `IsaacMjrlWrapper` adapts the Isaac env to the mjrl interface:
   - `num_dim_x = 49`, `num_dim_control = 12`
   - `get_f_and_B(x)` → `NeuralDynamics.get_f_and_B(x)`
   - `get_rollout('c3m')` → uniform sampling within obs/action space bounds
4. `Runner(wrapper, cfg, dynamics_model=dyn)` builds LQR agent and EvalTrainer
5. LQR linearises at each observation, solves CARE, applies `−K·e`

#### Step 3b — C3M on Isaac env

```bash
python scripts/mjrl/train_isaac.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm c3m \
    --dynamics_checkpoint checkpoints/quadruped_dynamics.pt \
    --num_envs 4 \
    --num_agent 4 \
    --epochs 30000 \
    --headless
```

**Obs/action dims for Isaac envs:**

| Task | `--x_dim` | `--u_dim` |
|------|-----------|-----------|
| `Quadruped-VelTracking-Direct-v0` | 49 | 12 |
| `Quadruped-PathTracking-Direct-v0` | 78 | 12 |
| `Humanoid-VelTracking-Direct-v0` | 70 | 19 |
| `Humanoid-PathTracking-Direct-v0` | 113 | 19 |
| `Manipulator-VelTracking-Direct-v0` | 32 | 7 |
| `Manipulator-PathTracking-Direct-v0` | 49 | 7 |

---

## Visualization

### Velocity arrows (VelTracking tasks, GUI mode)

Shows a **blue arrow** (commanded velocity direction/magnitude) and a
**green arrow** (actual body-frame velocity) floating above each robot.
Arrow length scales with speed; width is fixed so arrows remain visible at low speed.

```bash
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm ppo \
    --checkpoint <PATH> \
    --num_envs 8 \
    --debug_vis
# Do NOT add --headless — the GUI must be open for markers to render
```

Also works for Humanoid and Manipulator VelTracking tasks.

### Video recording

Works in headless mode (Isaac renders offscreen via EGL).

```bash
# Training videos — snapshot every 2000 policy steps
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 --algorithm ppo \
    --video --video_length 200 --video_interval 2000 \
    --headless

# Play — record one episode
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-Direct-v0 --algorithm ppo \
    --checkpoint <PATH> \
    --video --video_length 600 \
    --num_envs 1 --headless
```

Videos saved to `logs/skrl/<TASK>/<RUN>/videos/train/`.

---

## W&B Logging

```bash
wandb login   # once — or set WANDB_API_KEY env var

python scripts/skrl/train.py \
    --task Quadruped-VelTracking-Direct-v0 \
    --algorithm sac \
    --wandb \
    --wandb_project contractionRL \
    --wandb_run_name quad-sac-baseline \
    --video --video_length 200 --video_interval 2000 \
    --headless
```

What gets logged automatically:

- All skrl training scalars (reward, losses, entropy, LR, …) via patched
  TensorBoard writer
- MP4 videos uploaded in a background thread every `video_interval` steps
- Run config (task, algorithm, seed, hyperparams)

---

## Algorithm Reference

### LQR (Linear Quadratic Regulator)

Linearise `ẋ = f(x) + B(x)u` about the reference `x_ref`, solve the
continuous algebraic Riccati equation (CARE) for gain `K`, apply `δu = −K·e`.
The env adds `u_ref` automatically → effective law: `u = u_ref − K·e`.

No training required. Works out-of-the-box on any env with analytical or learned dynamics.

**Key hyperparameters:**

| Param | Default | Notes |
|-------|---------|-------|
| `Q_scaler` | 1.0 | `Q = Q_scaler · I` — state error weight |
| `R_scaler` | 1.0 | `R = R_scaler · I` — **must be > 0** (R=0 → huge gains → divergence at discrete dt) |

---

### C3M (Control Contraction Metric)

Jointly find a state-dependent Riemannian metric `M(x) = W(x)⁻¹` and a
tracking controller `δu = CLActor(x, x_ref, u_ref)` such that all
trajectories contract toward each other at rate `λ`.

Three matrix-valued conditions must hold everywhere in state space:

- **Cu ≺ 0** — closed-loop: `Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0`
- **C1 ≺ 0** — drift: `Bₗᵀ(−Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bₗ ≺ 0`
- **C2 = 0** — compatibility: `Bₗᵀ(Ẇ_b − 2·sym(∂B/∂x·W))Bₗ = 0`

`Bₗ = B_null` is the orthogonal complement of `col(B)` — the unactuated subspace.

**Key hyperparameters:**

| Param | Default | Notes |
|-------|---------|-------|
| `lbd` | 0.5 | Contraction rate λ (larger = faster convergence, harder to satisfy) |
| `w_ub` | 10.0 | Metric upper bound (prevent blow-up) |
| `w_lb` | 0.01 | Metric lower bound (strict PD) |
| `W_lr / u_lr` | 3e-4 | Learning rates for CMG and CLActor |
| `buffer_size` | 65536 | Size of `(x, x_ref, u_ref)` training dataset |
| `epochs` | 30000 | Training iterations |

---

### NeuralDynamics (learned dynamics model)

Learns `ẋ = f_net(x) + B_net(x) · u` from `(x, u, ẋ)` data.

- `f_net`: MLP `ℝ^{x_dim} → ℝ^{x_dim}` — autonomous drift
- `B_net`: MLP `ℝ^{x_dim} → ℝ^{x_dim × u_dim}` — input matrix (flattened)
- `B_null`: on-demand via full SVD of `B(x).detach()` — unactuated subspace
  needed for C3M's C1/C2 conditions; detached so it doesn't block the graph

Two interface modes:

- `get_f_and_B(x: Tensor)` → `(f, B, B_null)` with autograd — used by agents
- `forward(x: numpy|Tensor)` → same return, called by env under `no_grad()` for sim

**Key pretraining flags:**

| Flag | Default | Notes |
|------|---------|-------|
| `--n_samples` | 100 000 | More data → better generalisation |
| `--epochs` | 3000 | Typically converges in 2000–5000 |
| `--hidden_dim` | 256 256 256 | Use deeper/wider nets for high-dim Isaac obs |
| `--lr` | 3e-4 | Adam; cosine-annealed to 0 |

---

## Project Structure

```
contractionRL/
├── mjrl/                             # Contraction-control library (no Isaac dep)
│   ├── models/
│   │   ├── base.py                   # Autograd Jacobians, PD losses, LR schedule
│   │   ├── building_blocks.py        # MLP with xavier/orthogonal init
│   │   ├── cmg.py                    # CCM_Generator  W(x) = LᵀL
│   │   ├── actors.py                 # CLActor  δu = W₂·tanh(W₁·e)
│   │   └── dynamics.py               # NeuralDynamics  ẋ = f_net(x) + B_net(x)·u
│   ├── agents/torch/
│   │   ├── lqr.py                    # LQR — CARE solve, no training
│   │   └── c3m.py                    # C3M — joint metric + actor optimisation
│   ├── trainers/torch/
│   │   ├── base.py                   # BaseTrainer with sampler eval
│   │   ├── c3m_trainer.py            # C3MTrainer tqdm loop
│   │   └── eval_trainer.py           # EvalTrainer for no-param controllers
│   └── utils/
│       ├── sampler.py                # OnlineSampler  (num_agent == num_worker)
│       ├── isaac_wrapper.py          # IsaacMjrlWrapper — adapts Isaac env to mjrl
│       └── runner/torch/runner.py   # Runner(env, cfg, dynamics_model=).run()
│
├── scripts/
│   ├── list_envs.py                  # List all envs without Isaac Sim
│   ├── generate_ref_traj.py          # Record reference trajectories for path tracking
│   ├── skrl/
│   │   ├── train.py                  # Isaac RL training  (PPO / SAC)
│   │   └── play.py                   # Isaac RL evaluation + debug_vis
│   └── mjrl/
│       ├── train.py                  # Classic env contraction training
│       ├── train_isaac.py            # Isaac env contraction training
│       ├── pretrain_dynamics.py      # NeuralDynamics pretraining
│       └── collect_isaac_data.py     # Isaac rollout data collection
│
└── source/contractionRL/contractionRL/tasks/direct/
    ├── classic/
    │   └── car/                      # Car-Direct-v0  (analytical, no sim)
    ├── common/
    │   ├── vel_commands.py           # VelCommands — constant vxy + sinusoidal yaw
    │   ├── path_tracking_base.py     # PathTrackingBase — [x, x_ref, u_ref] obs
    │   └── traj_buffer.py            # TrajectoryBuffer — .npz trajectory loader
    ├── quadruped/                    # Quadruped-Direct-v0        (obs 48, act 12)
    ├── quadruped_vel_tracking/       # Quadruped-VelTracking-*    (obs 49, act 12)
    ├── quadruped_path_tracking/      # Quadruped-PathTracking-*   (obs 78, act 12)
    ├── humanoid/                     # Humanoid-Direct-v0         (obs 69, act 19)
    ├── humanoid_vel_tracking/        # Humanoid-VelTracking-*     (obs 70, act 19)
    ├── humanoid_path_tracking/       # Humanoid-PathTracking-*    (obs 113, act 19)
    ├── manipulator/                  # Manipulator-Direct-v0      (obs 20, act  7)
    ├── manipulator_vel_tracking/     # Manipulator-VelTracking-*  (obs 32, act  7)
    └── manipulator_path_tracking/    # Manipulator-PathTracking-* (obs 49, act  7)
```

# contractionRL

Contraction-based reinforcement learning research repo built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
All algorithms run through a unified **skrl** backend.
## Installation

1. **Create the Conda Environment:**
   We recommend using `conda` to isolate the dependencies.
   ```bash
   conda create -n contraction_rl python=3.10 -y
   conda activate contraction_rl
   ```

2. **Install IsaacLab (if not already installed):**
   Follow the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/) to set up the simulator.

3. **Install contractionRL:**
   Navigate to the root of this repository and install the package in editable mode:
   ```bash
   pip install -e source/contractionRL
   ```
   *Note: This will also install all required dependencies including `skrl`, `wandb`, and `torch`.*

## Example Commands

### 1. IsaacLab Training (Quadruped)
Train a Unitree Go2 to track a velocity command using PPO:
```bash
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo
```

### 2. Classic Environments (No Isaac Sim)
Train C3M on a classic `Car-v0` environment (uses multiprocessing for efficiency):
```bash
python scripts/skrl/train.py --classic --task Car-v0 --algorithm c3m --num_envs 4
```

### 3. Hyperparameter Sweeps (W&B)
Initialize a sweep and launch parallel agents across GPUs using the provided helper script:
```bash
# Edit search/run_c3m_sweeps.sh if necessary, then run:
./search/run_c3m_sweeps.sh
```

| Algorithm | Type | Training | Description |
|-----------|------|---------|-------------|
| **PPO** | RL | Online | Proximal Policy Optimisation |
| **SAC** | RL | Online | Soft Actor-Critic |
| **C3M** | Contraction | Offline (data buffer) | Control Contraction Metric — joint CCM + CLActor |
| **SD-LQR** | Contraction | None | State-Dependent LQR — linearise at current state |
| **LQR** | Contraction | None | LQR — linearise at reference state |
| **TEMP** | Contraction | Online | Two-policy (contracting + optimal) with Mahalanobis reward |

---

## Table of Contents

1. [Installation](#installation)
2. [Listing Environments](#listing-environments)
3. [Environments](#environments)
   - [Velocity-tracking](#velocity-tracking)
   - [Path-tracking](#path-tracking)
   - [Classic analytical environments](#classic-analytical-environments)
   - [Cartpole prototype](#cartpole-prototype)
4. [Training](#training)
   - [Velocity-tracking (locomotion pre-training)](#velocity-tracking-locomotion-pre-training)
   - [Path-tracking (contraction control)](#path-tracking-contraction-control)
   - [Classic environments](#classic-environments)
5. [Reference Trajectory Generation](#reference-trajectory-generation)
6. [Evaluation / Play](#evaluation--play)
7. [W&B Logging](#wb-logging)
8. [Algorithm Reference](#algorithm-reference)
9. [Config Files](#config-files)
10. [Project Structure](#project-structure)

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

### 4. Verify dependencies

```bash
python -c "import torch, scipy, gymnasium, skrl; print('OK')"
```

---

## Listing Environments

Scan task `__init__.py` files without loading Isaac Sim:

```bash
python scripts/list_envs.py

# Filter by keyword
python scripts/list_envs.py --keyword vel_tracking
python scripts/list_envs.py --keyword path_tracking
python scripts/list_envs.py --keyword classic
```

---

## Environments

### Velocity-tracking

**Tasks:** `Quadruped-VelTracking-v0`, `Humanoid-VelTracking-v0`, `Manipulator-VelTracking-v0`

**Goal:** Follow a randomly sampled velocity command while staying upright and using minimal torque.
These environments are used to **pre-train the locomotion policy** whose rollouts become the
reference trajectories for path-tracking.

#### Velocity commands

At each episode reset, each environment independently samples:

```
vx  ~ Uniform(vx_range)       [m/s] — forward velocity,  constant per episode
vy  ~ Uniform(vy_range)       [m/s] — lateral velocity,  constant per episode
yaw_rate(t) = A·sin(ω·t + φ)       — sinusoidal yaw, makes robot curve
   A ~ Uniform(yaw_A_range)   [rad/s]  amplitude
   ω ~ Uniform(yaw_omega_range)[rad/s]  frequency
   φ ~ Uniform(0, 2π)                  phase
```

#### Observation space

`obs = [physical_state, commands(4), prev_actions]`

| Env | Physical state | + cmds | + prev_act | Total obs |
|-----|----------------|--------|-----------|-----------|
| Quadruped | lin_vel_b(3) + ang_vel_b(3) + proj_gravity_b(3) + joint_pos_rel(12) + joint_vel(12) = **33** | 4 | 12 | **49** |
| Humanoid  | lin_vel_b(3) + ang_vel_b(3) + proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19) = **47** | 4 | 19 | **70** |
| Manipulator | joint_pos(7) + joint_vel(7) + ee_pos_local(3) + ee_lin_vel(3) + ee_yaw_vel(1) = **21** | 4 | 7 | **32** |

The **exported reference state** (`get_physical_state()`) saved to `.npz` drops body velocities:

| Robot | Reference state (exported) |
|-------|---------------------------|
| Quadruped | proj_gravity_b(3) + joint_pos_rel(12) + joint_vel(12) = **27** |
| Humanoid  | proj_gravity_b(3) + joint_pos_rel(19) + joint_vel(19) = **41** |
| Manipulator | joint_pos(7) + joint_vel(7) + ee_pos_local(3) = **17** |

#### Reward function

| Term | Formula | Scale (quadruped) | Purpose |
|------|---------|-----------------|---------|
| `rew_lin_vel` | `exp(−‖cmd_xy − vel_b_xy‖² / 0.25)` | +2.0 | Track lateral commands |
| `rew_yaw_rate` | `exp(−(cmd_yaw − ω_z)² / 0.25)` | +0.5 | Track yaw rate |
| `rew_z_vel` | `−v_z²` | −0.5 | Suppress vertical bouncing |
| `rew_ang_vel_xy` | `−‖ω_xy‖²` | −0.05 | Suppress roll/pitch oscillation |
| `rew_upright` *(humanoid)* | `−‖g_proj_xy‖²` | −1.0 | Keep torso upright |
| `rew_torques` | `−‖τ‖²` | −1×10⁻⁵ | Energy efficiency |
| `rew_action_rate` | `−‖aₜ − aₜ₋₁‖²` | −0.01 | Smooth joint motion |
| `rew_alive` | `1 − terminated` | +0.5 | Survival bonus |

Max episode reward (positive terms only, 500 steps):
`(2.0 + 0.5 + 0.5) × 500 = 1500` (quadruped) / `(2.0 + 0.5 + 1.0) × 500 = 1750` (humanoid).
The quality gate for reference-trajectory generation is set to **half of max** (750 / 875).

#### Termination

| Robot | Condition |
|-------|-----------|
| Quadruped | `base_height < 0.20 m` |
| Humanoid | `base_height < 0.50 m` |
| Manipulator | time-out only |
| All | `episode_length ≥ max_episode_length` (10 s at 50 Hz) |

---

### Path-tracking

**Tasks:** `Quadruped-PathTracking-v0`, `Humanoid-PathTracking-v0`, `Manipulator-PathTracking-v0`

**Goal:** Follow a pre-recorded reference trajectory `[x_ref, u_ref]` step-by-step.
The `[x, x_ref, u_ref]` observation format is contraction-compatible — all six algorithms can
train and evaluate here.

#### How path tracking works

```
Episode reset:
  1. Sample reference trajectory xref[0..T], uref[0..T] from .npz buffer
  2. Initialise robot joints to xref[0] + small noise

Each step t:
  obs    = [x_current, x_ref[t], u_ref[t]]
  reward = −‖x_current − x_ref[t]‖²
  action = policy(obs)        ← output is DEVIATION from u_ref
  u_applied = u_ref[t] + action
```

#### Observation layout

| Env | state_dim | obs = [x, x_ref, u_ref] total |
|-----|-----------|-------------------------------|
| Quadruped | 27 | 27 + 27 + 12 = **66** |
| Humanoid  | 41 | 41 + 41 + 19 = **101** |
| Manipulator | 21 | 21 + 21 + 7 = **49** |

#### Supported algorithms

All six algorithms work in path-tracking envs:

| Algorithm | Entry point key |
|-----------|----------------|
| PPO | `skrl_cfg_entry_point` |
| SAC | `skrl_sac_cfg_entry_point` |
| C3M | `skrl_c3m_cfg_entry_point` |
| LQR | `skrl_lqr_cfg_entry_point` |
| SD-LQR | `skrl_sdlqr_cfg_entry_point` |
| TEMP | `skrl_temp_cfg_entry_point` |

---

### Classic analytical environments

**Task:** `Car-v0`

A pure-Python gymnasium env with no Isaac Sim dependency.
Implements a Dubins-like car with exact analytical dynamics.

```
State  x = [p_x, p_y, θ, v]       (position, heading, speed)
Control u = [ω, a]                 (angular rate, linear acceleration)

Dynamics (control-affine):
    ẋ = f(x) + B(x)·u
    f(x) = [v·cos(θ), v·sin(θ), 0, 0]ᵀ
    B(x) = [[0,0], [0,0], [1,0], [0,1]]

Observation = [x, x_ref, u_ref]   (contraction-compatible)
Action      = deviation δu         (env applies u = u_ref + δu)
Episode     = 9 s at dt=0.03 s    (300 steps)
```

All six algorithms are supported. Uses `--classic` flag (no Isaac Sim needed).

---

### Cartpole prototype

**Task:** `Cartpole-v0`

Isaac Sim cartpole (obs=4, action=1). Minimal environment for rapid algorithm prototyping.
Supports PPO and SAC only.

---

## Training

### Velocity-tracking (locomotion pre-training)

Train the locomotion policy whose rollouts generate reference trajectories.
After training, reference trajectories are **auto-generated** if the policy passes the quality gate
(mean episode reward ≥ half of max theoretical reward).

```bash
# PPO — quadruped (recommended starting config)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm ppo \
    --num_envs 4096 \
    --headless

# PPO — humanoid
python scripts/skrl/train.py \
    --task Humanoid-VelTracking-v0 \
    --algorithm ppo \
    --num_envs 4096 \
    --headless

# PPO — manipulator
python scripts/skrl/train.py \
    --task Manipulator-VelTracking-v0 \
    --algorithm ppo \
    --num_envs 2048 \
    --headless

# SAC (lower env count — replay buffer makes it sample-efficient)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm sac \
    --num_envs 64 \
    --headless

# Resume from checkpoint
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --headless
```

**HP overrides:**

```bash
# PPO
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo \
    --ppo_lr 3e-4 --ppo_rollouts 64 --ppo_learning_epochs 5 --ppo_mini_batches 4 \
    --headless

# SAC
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm sac \
    --sac_lr 1e-4 --sac_batch_size 512 --sac_gradient_steps 4 --sac_memory_size 100000 \
    --headless
```

---

### Path-tracking (contraction control)

Requires reference trajectories (`logs/ref_trajs/{robot}.npz`) — generated automatically
after vel-tracking training, or manually via `scripts/generate_ref_traj.py`.

```bash
# C3M — quadruped (trains NeuralDynamics online)
python scripts/skrl/train.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm c3m \
    --headless

# SD-LQR — quadruped (analytical, no gradient training)
python scripts/skrl/train.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm sdlqr \
    --headless

# LQR — quadruped
python scripts/skrl/train.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm lqr \
    --headless

# TEMP — quadruped (two-policy online RL)
python scripts/skrl/train.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm temp \
    --headless

# PPO baseline
python scripts/skrl/train.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm ppo \
    --headless

# Same for humanoid / manipulator
python scripts/skrl/train.py --task Humanoid-PathTracking-v0 --algorithm c3m --headless
python scripts/skrl/train.py --task Manipulator-PathTracking-v0 --algorithm temp --headless
```

---

### Classic environments

No Isaac Sim needed. Pass `--classic` flag.

```bash
# C3M on Car-v0
python scripts/skrl/train.py --classic --task Car-v0 --algorithm c3m

# LQR on Car-v0
python scripts/skrl/train.py --classic --task Car-v0 --algorithm lqr

# SD-LQR
python scripts/skrl/train.py --classic --task Car-v0 --algorithm sdlqr

# TEMP
python scripts/skrl/train.py --classic --task Car-v0 --algorithm temp

# PPO / SAC
python scripts/skrl/train.py --classic --task Car-v0 --algorithm ppo --num_envs 4
python scripts/skrl/train.py --classic --task Car-v0 --algorithm sac --num_envs 4
```

---

## Reference Trajectory Generation

Path-tracking envs need a `.npz` trajectory buffer.

**Automatic** — generated at the end of every vel-tracking training run if the policy passes
the quality gate (mean episode reward ≥ 750 for quadruped, ≥ 875 for humanoid).

**Manual** — if the auto-generation was skipped or you want different settings:

```bash
python scripts/generate_ref_traj.py \
    --task Quadruped-VelTracking-v0 \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --robot quadruped \
    --num_envs 64 \
    --num_trajs 2000 \
    --headless
```

Output: `logs/ref_trajs/{robot}.npz` — matches the default `traj_path` in each path-tracking env config.

---

## Evaluation / Play

```bash
# GUI — watch the trained policy in real-time
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --num_envs 4

# Velocity arrow overlay (blue=command, green=actual)
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm ppo \
    --checkpoint <PATH> --num_envs 4 --debug_vis

# Record one episode as MP4
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 \
    --algorithm ppo \
    --checkpoint <PATH> --num_envs 1 \
    --video --video_length 600 --headless

# Path-tracking evaluation
python scripts/skrl/play.py \
    --task Quadruped-PathTracking-v0 \
    --algorithm c3m \
    --checkpoint <PATH> --num_envs 4
```

---

## W&B Logging

```bash
wandb login   # once — or set WANDB_API_KEY env var

# W&B is on by default (project=contractionRL). Pass --no_wandb to disable.
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --wandb_run_name quad-ppo-v1 \
    --video --video_length 200 --video_interval 2000 \
    --headless

# Different project
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm sac \
    --wandb_project my-project --headless

# Disable
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --no_wandb --headless
```

What gets logged: all skrl training scalars, MP4 videos (background thread), run config.

---

## Algorithm Reference

### PPO / SAC

Standard skrl implementations. See [skrl docs](https://skrl.readthedocs.io) for full parameter reference.
Configs: `agents/skrl_ppo_cfg.yaml`, `agents/skrl_sac_cfg.yaml`.

---

### C3M (Control Contraction Metric)

Jointly trains a state-dependent Riemannian metric `M(x) = W(x)⁻¹` and a tracking controller
`δu = CLActor(x, x_ref, u_ref)` such that all trajectories contract toward each other at rate `λ`.

Three matrix-valued conditions must hold everywhere in state space:

- **Cu ≺ 0** — closed-loop: `Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0`
- **C1 ≺ 0** — drift: `Bₗᵀ(−Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bₗ ≺ 0`
- **C2 = 0** — compatibility: `Bₗᵀ(Ẇ_b − 2·sym(∂B/∂x·W))Bₗ = 0`

On Isaac envs, `NeuralDynamics` (`ẋ = f_net(x) + B_net(x)·u`) is trained online from trajectory data.
On classic envs with `use_analytical_dynamics: true`, the env's exact `get_f_and_B(x)` is used.

| Param | Default | Notes |
|-------|---------|-------|
| `lbd` | 0.01 | Contraction rate λ |
| `w_ub` / `w_lb` | 10.0 / 0.1 | Metric bounds |
| `W_lr` / `u_lr` | 3e-4 | CMG / CLActor learning rates |
| `buffer_size` | 4096 | Training data buffer |

---

### SD-LQR (State-Dependent LQR)

Linearises `ẋ = f(x) + B(x)u` at the **current state** `x`, solves CARE, applies `u = uref − K(x)·e`.
No training — analytical per-step computation. Jacobians via autograd through `NeuralDynamics`.

| Param | Default | Notes |
|-------|---------|-------|
| `Q_scaler` | 1.0 | `Q = Q_scaler · I` |
| `R_scaler` | 0.01 | `R = R_scaler · I` — must be > 0 |

---

### LQR

Same as SD-LQR but linearises at the **reference state** `x_ref` instead of current state `x`.
Applies `u = −K(x_ref)·e` (env adds `u_ref` automatically).

---

### TEMP (Tracking via Entropy-regularised Model-based Planning)

Two-PPO architecture:
- **Contracting policy** (γ→0): optimises Mahalanobis tracking reward `−‖e‖²_M / std`
- **Optimal policy** (γ→0.99): optimises environment reward

CMG (CCM generator, same as C3M) is trained concurrently to shape the Mahalanobis metric.

| Param | Default | Notes |
|-------|---------|-------|
| `gamma_contracting` | 0.0 | Discount for contracting policy |
| `gamma_optimal` | 0.99 | Discount for optimal policy |
| `tracking_scaler` | 1.0 | Mahalanobis reward scale |
| `cmg_updates_per_iter` | 3 | CMG gradient steps per rollout |

---

## Config Files

Each environment owns its own agent configs, stored alongside the environment:

```
tasks/direct/
  quadruped_vel_tracking/agents/
      skrl_ppo_cfg.yaml
      skrl_sac_cfg.yaml
  quadruped_path_tracking/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
      skrl_sdlqr_cfg.yaml     skrl_temp_cfg.yaml
  humanoid_vel_tracking/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
  humanoid_path_tracking/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
      skrl_sdlqr_cfg.yaml     skrl_temp_cfg.yaml
  manipulator_vel_tracking/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
  manipulator_path_tracking/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
      skrl_sdlqr_cfg.yaml     skrl_temp_cfg.yaml
  classic/car/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
      skrl_sdlqr_cfg.yaml     skrl_temp_cfg.yaml
  cartpole/agents/
      skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
```

---

## Project Structure

```
contractionRL/
├── scripts/
│   ├── list_envs.py                  # List all envs without Isaac Sim
│   ├── generate_ref_traj.py          # Manually generate reference trajectories
│   ├── train.py                      # Unified training entry point (classic + Isaac)
│   └── skrl/
│       ├── train.py                  # skrl training (Isaac + --classic route)
│       └── play.py                   # Evaluation + debug_vis
│
└── source/contractionRL/contractionRL/
    ├── agents/skrl/
    │   ├── math_utils.py             # Jacobians, PD losses (pure PyTorch)
    │   ├── nn_modules.py             # MLP, CCM_Generator, CLActor, NeuralDynamics
    │   ├── models.py                 # skrl model wrappers (CMGModel, CLActorModel)
    │   ├── c3m.py                    # C3MAgent + C3MSkrlTrainer
    │   ├── sdlqr.py                  # SDLQRAgent + LQRAgent
    │   └── temp.py                   # TEMPAgent + TEMPSkrlTrainer
    │
    └── tasks/direct/
        ├── classic/car/              # Car-v0  (analytical, no Isaac Sim)
        ├── cartpole/                 # Cartpole-v0  (prototype, obs=4, act=1)
        ├── common/
        │   ├── vel_commands.py       # Sinusoidal yaw + constant vxy commands
        │   ├── path_tracking_base.py # [x, x_ref, u_ref] observation layout
        │   └── traj_buffer.py        # .npz trajectory loader
        ├── quadruped_vel_tracking/   # Quadruped-VelTracking-v0   (obs 49, act 12)
        ├── quadruped_path_tracking/  # Quadruped-PathTracking-v0  (obs 66, act 12)
        ├── humanoid_vel_tracking/    # Humanoid-VelTracking-v0    (obs 70, act 19)
        ├── humanoid_path_tracking/   # Humanoid-PathTracking-v0   (obs 101, act 19)
        ├── manipulator_vel_tracking/ # Manipulator-VelTracking-v0 (obs 32, act  7)
        └── manipulator_path_tracking/# Manipulator-PathTracking-v0 (obs 49, act  7)
```

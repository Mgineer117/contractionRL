# contractionRL

Contraction-based reinforcement learning research repo built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
All algorithms run through a unified **skrl** backend — Isaac Sim environments and lightweight
classic (pure-NumPy, no-Isaac) environments alike.

## Table of Contents

1. [Installation](#installation)
2. [Listing Environments](#listing-environments)
3. [Environments](#environments)
   - [Velocity-tracking (Isaac)](#velocity-tracking-isaac)
   - [Path-tracking (Isaac)](#path-tracking-isaac)
   - [Classic analytical environments](#classic-analytical-environments)
   - [Cartpole prototype (Isaac)](#cartpole-prototype-isaac)
4. [Action convention](#action-convention)
5. [Training](#training)
6. [Reference Trajectory Generation](#reference-trajectory-generation)
7. [Evaluation / Play](#evaluation--play)
8. [W&B Logging](#wb-logging)
9. [Algorithm Reference](#algorithm-reference)
10. [Config Files](#config-files)
11. [Project Structure](#project-structure)

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

Isaac Lab/Sim is only needed for the Isaac environments (humanoid/quadruped/manipulator
vel/path-tracking, the `Cartpole-v0` prototype). The `classic-*` environments are pure
NumPy/gymnasium and need none of this — see [Classic analytical environments](#classic-analytical-environments).

### 2. Clone this repo

```bash
git clone <repo-url> contractionRL
cd contractionRL
```

### 3. Install the contractionRL extension

```bash
python -m pip install -e source/contractionRL
```

*This also installs `skrl`, `wandb`, and `torch`.*

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

| Task ID | Type | Isaac Sim? |
|---|---|---|
| `Humanoid-VelTracking-v0` | velocity-tracking | yes |
| `Humanoid-PathTracking-v0` | path-tracking | yes |
| `Quadruped-VelTracking-v0` | velocity-tracking | yes |
| `Quadruped-PathTracking-v0` | path-tracking | yes |
| `Manipulator-VelTracking-v0` | velocity-tracking | yes |
| `Manipulator-PathTracking-v0` | path-tracking | yes |
| `Cartpole-v0` | prototype (obs=4, act=1) | yes |
| `classic-car-v0` | path-tracking, analytical | no |
| `classic-cartpole-v0` | path-tracking, analytical | no |
| `classic-segway-v0` | path-tracking, analytical | no |
| `classic-turtlebot-v0` | path-tracking, analytical | no |

---

## Environments

Every path-tracking environment (Isaac and classic) shares the same contraction-compatible
observation layout `obs = [x, x_ref, u_ref]` and quadratic tracking reward `r = -‖x - x_ref‖²`,
so all seven algorithms (PPO, SAC, C3M, LQR, SD-LQR, C2RL-PPO, C2RL-SAC) can train/evaluate in any of them.

### Velocity-tracking (Isaac)

**Tasks:** `Quadruped-VelTracking-v0`, `Humanoid-VelTracking-v0`, `Manipulator-VelTracking-v0`

**Purpose:** pre-train a locomotion/manipulation policy. Its rollouts are recorded and become
the reference trajectories `[x_ref, u_ref]` that the path-tracking envs track (see
[Reference Trajectory Generation](#reference-trajectory-generation)) — these envs are not
contraction-tracking envs themselves.

#### Commands sampled each episode

```
vx  ~ Uniform(vx_range)            [m/s] forward velocity, constant per episode
vy  ~ Uniform(vy_range)            [m/s] lateral velocity, constant per episode
yaw_rate(t) = A·sin(ω·t + φ)             sinusoidal yaw, makes the robot curve
   A ~ Uniform(yaw_A_range)  [rad/s]
   ω ~ Uniform(yaw_omega_range) — up to one full cycle per episode (2π / episode_length_s)
   φ ~ Uniform(0, 2π)
```
(manipulator tracks an end-effector Cartesian velocity command `[vx, vy, vz, yaw_rate]` instead.)

#### Observation, action, reward, termination

| | Quadruped | Humanoid | Manipulator |
|---|---|---|---|
| **Physical state** | `lin_vel_b(3)+ang_vel_b(3)+proj_gravity_b(3)+joint_pos_rel(12)+joint_vel(12)` = **33** | `lin_vel_b(3)+ang_vel_b(3)+proj_gravity_b(3)+joint_pos_rel(19)+joint_vel(19)` = **47** | `joint_pos(7)+joint_vel(7)+ee_pos_local(3)+ee_lin_vel(3)+ee_yaw_vel(1)` = **21** |
| **Obs** = state + cmds(4) + prev_action | **49** | **70** | **32** |
| **Action** | 12 joint targets: `default_pos + 0.25·action` | 19 joint targets: `default_pos + 0.25·action` | 7 joint targets: `mid ± half_range·action` (soft-limit midpoint form) |
| **Reward terms** | `rew_alive(0.5) + rew_lin(exp(-lin_err/0.25)·2.0) + rew_yaw(exp(-yaw_err/0.25)·0.5) + rew_flat(-0.5·Σg_xy²) + rew_z(-0.5·vz²) + rew_rp(-0.05·Σω_xy²) + rew_torque(-1e-5·Στ²) + rew_action_rate(-0.01·Σ(a-a')²)` | `rew_alive(1.0) + rew_lin(exp(-lin_err/0.1)·2.0) + rew_yaw(exp(-yaw_err/0.1)·0.5) + rew_flat(-1.0·Σg_xy²) + rew_z(-0.5·vz²) + rew_rp(-0.05·Σω_xy²) + rew_torque(-1e-5·Στ²) + rew_action_rate(-0.01·Σ(a-a')²)` | `rew_vel(-1.0·‖ee_vel-cmd‖²) + rew_yaw(-0.5·(ee_yaw_vel-cmd)²) + rew_action_rate(-0.01·Σ(a-a')²) + rew_joint_limits(-0.1·soft-limit penalty)` — no exp-shaping, no alive bonus |
| **Termination** | fall: `base_height < 0.20 m` OR tilt `proj_gravity_b[z] > -0.71` | fall: `base_height < 0.50 m` OR tilt `proj_gravity_b[z] > -0.71` | time-out only, no failure condition |
| **Decimation / control freq** | 4 @ sim dt=1/200s → **50 Hz** (step dt=0.02s) | 4 @ 1/200s → **50 Hz** | 2 @ sim dt=1/120s → **60 Hz** (step dt≈0.0167s) |
| **Episode length** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=15.0` → **900 steps** |

Episode lengths were sized so each episode covers roughly 4 contraction time-constants
(`T ≥ 4/λ`) at the target rate — `λ≈0.1` for the high-DoF locomotion envs (quadruped/humanoid,
hence 40 s) and `λ≈0.3` elsewhere (manipulator/classic, hence 15 s); see
[W&B Logging](#wb-logging) for how the achieved rate is actually measured.

**Reference-trajectory quality gate** (see [Reference Trajectory Generation](#reference-trajectory-generation)):
half of the theoretical best-case total episode reward, i.e. `0.5 · (rew_alive+rew_lin_vel+rew_yaw_rate) · T`
— **3000** for quadruped, **3500** for humanoid, **0** for manipulator (its per-step reward has no
positive term, so its best case is exactly 0 and the gate is a no-op unless you pass `--min_ref_quality`).

---

### Path-tracking (Isaac)

**Tasks:** `Quadruped-PathTracking-v0`, `Humanoid-PathTracking-v0`, `Manipulator-PathTracking-v0`

**Purpose:** follow a pre-recorded reference trajectory `[x_ref, u_ref]` step-by-step — this is
where the contraction algorithms (C3M/LQR/SD-LQR/C2RL) are actually trained/evaluated, alongside
PPO/SAC baselines.

```
Episode reset:
  1. Sample a reference trajectory xref[0..T], uref[0..T] from the .npz buffer
  2. Initialise the robot to xref[0]

Each step t:
  obs    = [x_current, x_ref[t], u_ref[t]]
  reward = -‖x_current - x_ref[t]‖²
  u_applied = action              ← the env applies the policy's action DIRECTLY
```

See [Action convention](#action-convention) — the env never adds `u_ref` back in; algorithms
that need to track `u_ref` (CLActor for C3M/C2RL, LQR/SD-LQR) fold it into their own output.

| | Quadruped | Humanoid | Manipulator |
|---|---|---|---|
| **state_dim** (exported physical state, drops body velocities) | `proj_gravity_b(3)+joint_pos_rel(12)+joint_vel(12)` = **27** | `proj_gravity_b(3)+joint_pos_rel(19)+joint_vel(19)` = **41** | `joint_pos(7)+joint_vel(7)+ee_pos_local(3)` = **17→21†** |
| **Obs** = x + x_ref + u_ref | 27+27+12 = **66** | 41+41+19 = **101** | 21+21+7 = **49** |
| **Action** | 12: `default_pos + 0.25·action` | 19: `default_pos + 0.25·action` | 7: soft-limit midpoint form |
| **Termination** | fall: `base_height < 0.20 m` | fall: `base_height < 0.50 m` | time-out only |
| **Decimation / control freq** | 4 @ 1/200s → **50 Hz** | 4 @ 1/200s → **50 Hz** | 2 @ 1/120s → **60 Hz** |
| **Episode length** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=15.0` → **900 steps** |
| **Reference file** | `logs/quadruped/dynamics_data.npz` | `logs/humanoid/dynamics_data.npz` | `logs/manipulator/dynamics_data.npz` |

† manipulator's exported state is 21-dim per the current `traj_buffer`/env code (`joint_pos(7)+joint_vel(7)+ee_pos_local(3)+ee_lin_vel(3)+ee_yaw_vel(1)`), matching its vel-tracking physical-state layout.

#### Supported algorithms

All seven algorithms work in every path-tracking env:

| Algorithm | Entry point key |
|-----------|----------------|
| PPO | `skrl_cfg_entry_point` |
| SAC | `skrl_sac_cfg_entry_point` |
| C3M | `skrl_c3m_cfg_entry_point` |
| LQR | `skrl_lqr_cfg_entry_point` |
| SD-LQR | `skrl_sdlqr_cfg_entry_point` |
| C2RL-PPO | `skrl_c2rl_ppo_cfg_entry_point` |
| C2RL-SAC | `skrl_c2rl_sac_cfg_entry_point` |

---

### Classic analytical environments

**Tasks:** `classic-car-v0`, `classic-cartpole-v0`, `classic-segway-v0`, `classic-turtlebot-v0`

Pure-Python/NumPy gymnasium envs, no Isaac Sim dependency (`tasks/direct/classic/`). All share
`common/env_base.py`'s `BaseEnv` and the same `obs = [x, x_ref, u_ref]` / `r = -0.5·‖x-x_ref‖²`
contraction-compatible interface. Unlike the Isaac envs, the reference trajectory is **not** a
recorded file — each env analytically synthesizes a fresh `xref/uref` every `reset()` by sampling
a random Fourier-sum reference control and rolling it forward through the same dynamics used at
train time (`system_reset()` in each env.py).

Dynamics are control-affine, `ẋ = f(x) + B(x)u`:

| Env | State `x` | Control `u` | dt | `time_bound` | steps | Dynamics |
|---|---|---|---|---|---|---|
| car | `[p_x,p_y,θ,v]` (4) | `[ω,a]` (2) | 0.03s | 15.0s | ≤500 | Dubins-like car with velocity state |
| cartpole | `[p,θ,v,ω]` (4) | `[F]` (1) | 0.03s | 15.0s | ≤500 | standard nonlinear cartpole ODE |
| segway | `[x,θ,v,ω]` (4) | `[τ]` (1) | 0.03s | 15.0s | ≤500 | segway balance ODE (fixed numeric coefficients) |
| turtlebot | `[x,y,θ]` (3) | `[v,ω]` (2) | 0.05s | 30.0s | ≤600 | pure kinematic unicycle (`f(x)=0`) |

Reward config for all four: `q=1.0, r=0.0`, i.e. effective reward is `-0.5·‖x-x_ref‖²` with no
control-cost term active (`control_effort` is computed but its weight `r` is 0).

**Termination** is always `False` — episodes only truncate at their sampled length (`time_steps
== episode_len - 1`), and `episode_len` can be *shorter* than `time_bound/dt` if the analytically
generated reference exits its position bounds early during `system_reset`.

Episode lengths above (15s for car/cartpole/segway, 30s for turtlebot) target a contraction rate
λ≈0.3, same reasoning as the manipulator env — see the note under
[Velocity-tracking](#velocity-tracking-isaac).

Uses `--classic` flag (no Isaac Sim needed). All six algorithms are supported.

---

### Cartpole prototype (Isaac)

**Task:** `Cartpole-v0`

Stock IsaacLab cartpole tutorial environment — a minimal scaffold for rapid PPO/SAC
prototyping, **not** part of the path/velocity-tracking contraction pipeline (no `u_ref`,
no reference trajectory).

- **Obs (4)**: `[pole_pos, pole_vel, cart_pos, cart_vel]`
- **Action (1)**: direct joint-effort target, `action·100.0 N`
- **Reward**: `rew_alive(1.0·(1-terminated)) + rew_termination(-2.0·terminated) + rew_pole_pos(-1.0·pole_pos²) + rew_cart_vel(-0.01·|cart_vel|) + rew_pole_vel(-0.005·|pole_vel|)`
- **Termination**: `|cart_pos| > 3.0 m` OR `|pole_pos| > π/2`
- **Decimation/control freq**: 2 @ sim dt=1/120s → **60 Hz**
- **Episode length**: `episode_length_s=5.0` → **300 steps**

Supports PPO and SAC only.

---

## Action convention

Every environment (Isaac and classic) applies the policy's action **directly** —
`u_applied = action`, no environment ever adds `u_ref` back in:

- Isaac locomotion envs: `default_pos + action_scale · action` (or the manipulator's soft-limit
  midpoint form) — a joint-space delta, unrelated to `u_ref`.
- Classic envs: `self.current_u = u.copy()` — applied as-is.

Instead, it's the **agent** that folds `u_ref` into its own output where relevant:

- `CLActor` (used by C3M/C2RL): `mu = uref + w2 · l1(x, xref)` — feedback added on top of `uref`,
  sliced out of the observation.
- `LQR` / `SD-LQR`: `u = uref - K·(x - xref)`.
- PPO/SAC on **all** path-tracking envs (Isaac and classic) default to `models.policy.backbone:
  control` in their yaml — this swaps the stock skrl Gaussian MLP for `CLActorModel`, so their
  policy mean is *also* `uref + feedback`, not a from-scratch `u`. Set `backbone: mlp` explicitly
  to opt out and have the policy learn the full control (this is the only option for
  vel-tracking envs, which have no `u_ref` to fold in).

---

## Training

### Velocity-tracking (locomotion pre-training)

```bash
# PPO — quadruped (recommended starting config)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --num_envs 4096 --headless

# PPO — humanoid
python scripts/skrl/train.py \
    --task Humanoid-VelTracking-v0 --algorithm ppo \
    --num_envs 4096 --headless

# PPO — manipulator
python scripts/skrl/train.py \
    --task Manipulator-VelTracking-v0 --algorithm ppo \
    --num_envs 2048 --headless

# SAC (lower env count — replay buffer makes it sample-efficient)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm sac \
    --num_envs 64 --headless

# Resume from checkpoint
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --headless
```

Reference trajectories for path-tracking are **auto-generated** at the end of a vel-tracking
run if the policy clears the quality gate — see
[Reference Trajectory Generation](#reference-trajectory-generation).

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

### Path-tracking (contraction control)

Requires a reference file at `logs/{robot}/dynamics_data.npz` (auto-generated after
vel-tracking training, or manually via `scripts/generate_ref_traj.py`).

```bash
# C3M — quadruped (trains NeuralDynamics online from the reference buffer)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m --headless

# SD-LQR / LQR — quadruped (analytical, no gradient training)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm sdlqr --headless
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm lqr --headless

# C2RL — quadruped (two-policy online RL, on top of PPO or SAC)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c2rl-ppo --headless
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c2rl-sac --headless

# PPO baseline
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm ppo --headless

# Same for humanoid / manipulator
python scripts/skrl/train.py --task Humanoid-PathTracking-v0 --algorithm c3m --headless
python scripts/skrl/train.py --task Manipulator-PathTracking-v0 --algorithm c2rl-ppo --headless
```

### Classic environments

No Isaac Sim needed. Pass `--classic` flag. `--num_envs` here is a *process-level* parallel-env
count (plain Python instances via `gymnasium.vector.SyncVectorEnv`), not GPU-batched.

```bash
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c3m --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm lqr --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm sdlqr --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-sac --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm ppo --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm sac --num_envs 4

# Same for classic-cartpole-v0 / classic-segway-v0 / classic-turtlebot-v0
```

### Hyperparameter sweeps (W&B)

**C3M — classic envs** (`search/run_c3m_sweeps.sh`):

```bash
# Launches classic-cartpole-v0 / classic-turtlebot-v0 / classic-segway-v0 / classic-car-v0
# sweeps in parallel across GPUs 0-3, 3 agents per env by default.
./search/run_c3m_sweeps.sh          # or: ./search/run_c3m_sweeps.sh <num-agents-per-env>
```

This sweep optimizes **`Stability / convergence_score`** (`= contraction_rate / overshoot`,
higher is better — see [W&B Logging](#wb-logging)), not raw reward, since maximizing reward
alone doesn't guarantee the certified contraction property C3M is meant to produce.

**PPO/SAC — Isaac locomotion pre-training** (`scripts/search/`), a separate, unrelated sweep
infrastructure for tuning the vel-tracking policies themselves (optimizes plain
`Reward / Total reward (mean)` — reward is the right target here, there's no contraction
certificate to check):

```bash
bash scripts/search/run_quadruped_vel_search.bash --algorithm SAC
bash scripts/search/run_quadruped_vel_search.bash --algorithm PPO
# similarly: run_humanoid_vel_search.bash, run_manipulator_vel_search.bash
```

---

## Reference Trajectory Generation

Isaac path-tracking envs need a `.npz` trajectory buffer at `logs/{robot}/dynamics_data.npz`
(keys `x`, `u`, `x_dot`, `lengths`, consumed by `TrajectoryBuffer`). Classic envs need no such
file — see [Classic analytical environments](#classic-analytical-environments).

**Automatic** — `_generate_ref_trajs()` in `scripts/skrl/train.py` runs at the end of every
Isaac vel-tracking training run:
1. Loads `best_agent.pt` and checks the quality gate (mean episode reward ≥ half of the
   theoretical max — see the per-robot numbers under
   [Velocity-tracking](#velocity-tracking-isaac); 0 for manipulator).
2. Rolls out a candidate pool of `--ref_oversample_factor × --ref_num_trajs` episodes across
   **every** parallel env (not just the first `num_trajs` — this maximizes the pool the
   selection gets to pick from, since Isaac can't shrink the batch anyway), keeping only
   trajectories that survive `--min_ref_traj_length_frac` (default 50%) of the episode.
3. Keeps the **longest** `--ref_num_trajs` of that pool — early termination is exactly what a
   poor rollout looks like, so ranking by survival length favors complete, high-quality
   reference data.

```bash
# Tune the collection (defaults shown)
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo --headless \
    --ref_num_trajs 1000 --ref_oversample_factor 2.0 --min_ref_traj_length_frac 0.5

# Skip the quality gate entirely
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo --headless \
    --min_ref_quality 0
```

**Manual** — if auto-generation was skipped, or you want to regenerate from an older checkpoint:

```bash
python scripts/generate_ref_traj.py \
    --task Quadruped-VelTracking-v0 \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --robot quadruped \
    --num_envs 128 \
    --num_trajs 2000 \
    --headless
```

Output defaults to `logs/{robot}/dynamics_data.npz` (matching each path-tracking env's default
`traj_path`) — override `--out_dir` only if you've also overridden `traj_path` in the env config.

---

## Evaluation / Play

Isaac only (`scripts/skrl/play.py` has no `--classic` route):

```bash
# GUI — watch the trained policy in real-time
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --num_envs 4

# Velocity arrow overlay (blue=command, green=actual)
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint <PATH> --num_envs 4 --debug_vis

# Record one episode as MP4, headless
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint <PATH> --num_envs 1 \
    --video --video_length 600 --headless

# Path-tracking evaluation
python scripts/skrl/play.py \
    --task Quadruped-PathTracking-v0 --algorithm c3m \
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

### Tabs

Metric keys are grouped into W&B "tabs" by whatever precedes the first `/` in the key. skrl's
own built-in tracker uses space-padded keys (`"Episode / Total timesteps (mean)"`), so custom
metrics use the same `"Tab / key"` spacing to land in the *same* section rather than creating a
visually-identical-but-distinct duplicate tab.

| Tab | Contents |
|---|---|
| `Reward` | skrl's own `Reward / Total reward (max/min/mean)`, plus (vel-tracking envs) `discounted_return`, `avg_reward_per_step`, `total_reward_ci95` (95% CI of total reward — `undiscounted_return` itself is intentionally **not** logged again, since it's the same quantity as skrl's own total reward). Classic C3M's periodic eval also reports `Reward / reward_mean` here. |
| `Stability` | *(path-tracking envs only)* per-episode contraction diagnostics + their 95% CIs: `auc` (Σ‖error‖ over the episode), `contraction_rate` (empirical λ, fit as `e(T)=e(0)·exp(-λT)`), `overshoot` (peak error / initial error — 1.0 = no overshoot), and **`convergence_score = contraction_rate / overshoot`** (higher is better: fast contraction with little to no overshoot). Classic C3M's periodic eval reports the same four keys here too. |
| `Episode` | skrl's own `Episode / Total timesteps (...)`, plus `contraction_flag` (fraction of steps where error strictly decreased) and `performance_score` (-mean error) for path-tracking envs, and `auc` (velocity-tracking error AUC) for vel-tracking envs. |
| `Loss` | C3M's `Loss / C3M/loss/*`, `Loss / C3M/dynamics/mse`, `Loss / Pretrain/dynamics_mse`, PPO/SAC's own loss terms. |
| `Eval` | Isaac path-tracking evaluator output (`reward_mean`, `auc`, ...) from `train.py`'s post-training evaluation. |

`Stability/*` and vel-tracking's `Reward/*` are populated from `env.extras["log"]`, forwarded by
skrl's trainer only when the value is a `torch.Tensor` (not a Python float) — keep that in mind
if you add new per-episode metrics to an env's `_reset_idx`.

For classic C3M runs specifically, `Stability/*` (and `Reward/reward_mean`) come from
`C3MSkrlTrainer.eval()`, called every `eval_interval` training steps (default 100 in each
`skrl_c3m_cfg.yaml`) — this is also what `search/run_c3m_sweeps.sh`'s sweep optimizes
(`Stability / convergence_score`, `goal: maximize`).

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

`cfg.lbd` is the *certified/target* contraction rate baked into the training loss; the
`Stability/contraction_rate` W&B metric is the *empirically measured* rate on rollouts — they
are related but not the same number, and `Stability/convergence_score` (`=contraction_rate/overshoot`)
is what the hyperparameter sweep in `search/run_c3m_sweeps.sh` actually optimizes.

| Param | Default | Notes |
|-------|---------|-------|
| `lbd` | 0.01 | Target contraction rate λ |
| `w_ub` / `w_lb` | 10.0 / 0.1 | Metric bounds |
| `W_lr` / `u_lr` | 3e-4 | CMG / CLActor learning rates |
| `buffer_size` | 4096 | Training data buffer |
| `eval_interval` | 100 | Steps between `eval()` calls that populate `Stability/*` |

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

Same as SD-LQR but linearises at the **reference state** `x_ref` instead of the current state.
Applies `u = uref − K(x_ref)·e`.

---

### C2RL (Contraction-Certified RL)

Two-policy architecture, built on top of a chosen base algorithm (`c2rl-ppo` uses two official
skrl `PPO` sub-agents, `c2rl-sac` uses two official skrl `SAC` sub-agents):
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
  quadruped_vel_tracking/agents/      skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  quadruped_path_tracking/agents/     skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  humanoid_vel_tracking/agents/        skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  humanoid_path_tracking/agents/       skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  manipulator_vel_tracking/agents/     skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  manipulator_path_tracking/agents/    skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  cartpole/agents/                    skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  classic/car/agents/                 skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  classic/cartpole/agents/            (same 7 as car)
  classic/segway/agents/              (same 7 as car)
  classic/turtlebot/agents/           (same 7 as car)
```

---

## Project Structure

```
contractionRL/
├── search/
│   └── run_c3m_sweeps.sh             # Launches classic C3M W&B bayes sweeps across GPUs
│                                     #   (optimizes Stability / convergence_score)
├── scripts/
│   ├── list_envs.py                  # List all envs without Isaac Sim
│   ├── generate_ref_traj.py          # Manually generate reference trajectories
│   ├── skrl/
│   │   ├── train.py                  # skrl training entry point (Isaac + --classic route);
│   │   │                             #   ref-traj auto-generation lives here (_generate_ref_trajs)
│   │   └── play.py                   # Evaluation + debug_vis (Isaac only)
│   └── search/                       # PPO/SAC W&B bayes sweeps for vel-tracking pre-training
│       ├── search_algo.py            #   (optimizes plain Reward / Total reward (mean))
│       ├── run_quadruped_vel_search.bash
│       ├── run_humanoid_vel_search.bash
│       └── run_manipulator_vel_search.bash
│
└── source/contractionRL/contractionRL/
    ├── agents/skrl/
    │   ├── math_utils.py             # Jacobians, PD losses (pure PyTorch)
    │   ├── nn_modules.py             # MLP, CCM_Generator, CLActor, NeuralDynamics
    │   ├── models.py                 # skrl model wrappers (CMGModel, CLActorModel)
    │   ├── c3m.py                    # C3MAgent + C3MSkrlTrainer (train/eval, Stability metrics)
    │   ├── sdlqr.py                  # SDLQRAgent + LQRAgent
    │   ├── c2rl.py                    # C2RLAgent + C2RLSkrlTrainer
    │   └── eval_metrics.py           # fit_exponential_envelope, mean_confidence_interval
    │
    └── tasks/direct/
        ├── classic/                  # car / cartpole / segway / turtlebot (analytical, no Isaac)
        │   └── common/env_base.py    # BaseEnv shared by all 4 classic envs
        ├── cartpole/                 # Cartpole-v0  (Isaac prototype, obs=4, act=1)
        ├── common/
        │   ├── vel_commands.py       # Sinusoidal yaw + constant vxy commands
        │   ├── path_tracking_base.py # [x, x_ref, u_ref] layout + Stability/Episode W&B logging
        │   ├── eval_metrics.py       # episode_metrics/batch_episode_metrics, mean_confidence_interval
        │   └── traj_buffer.py        # .npz trajectory loader
        ├── quadruped_vel_tracking/   # Quadruped-VelTracking-v0   (obs 49, act 12)
        ├── quadruped_path_tracking/  # Quadruped-PathTracking-v0  (obs 66, act 12)
        ├── humanoid_vel_tracking/    # Humanoid-VelTracking-v0    (obs 70, act 19)
        ├── humanoid_path_tracking/   # Humanoid-PathTracking-v0   (obs 101, act 19)
        ├── manipulator_vel_tracking/ # Manipulator-VelTracking-v0 (obs 32, act  7)
        └── manipulator_path_tracking/# Manipulator-PathTracking-v0 (obs 49, act  7)
```

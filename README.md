# contractionRL

Contraction-based reinforcement learning. Eight algorithms — PPO, SAC, C3M, LQR, SD-LQR,
CV-STEM-LQR, C2RL-PPO, C2RL-SAC — run through one [skrl](https://github.com/Toni-SM/skrl)
backend across two environment families that share a single interface, so any algorithm trains and
evaluates on any environment and the reported stability metrics are directly comparable.

- **Classic envs** — pure NumPy/gymnasium, no simulator. Analytically synthesize a fresh reference
  trajectory every `reset()`. Twelve are registered (`car-v0`, `car-v1`, `cartpole`, `segway`,
  `quadrotor`, `aircraft`, `auv`, `ball_and_beam`, `pvtol`, `tora`, `two_link_arm`).
- **Isaac envs** — quadruped / humanoid / manipulator velocity- and path-tracking. Need Isaac Lab.

## Install

```bash
python -m pip install -e source/contractionRL        # classic envs need nothing else
```

Isaac envs additionally need [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The
contraction-metric SDP defaults to **SCS** (bundled with `cvxpy`); **MOSEK** is tighter and faster
if you have a license — put it at `~/mosek/mosek.lic` and set `cm_solver: MOSEK`.

```bash
python scripts/list_envs.py --keyword classic   # list envs without loading Isaac Sim
```

## Quick start

```bash
# Classic (no simulator). --num_envs is a process-level SyncVectorEnv.
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo --num_envs 4

# Isaac: pretrain locomotion on velocity-tracking, then do contraction control on path-tracking
python scripts/skrl/train.py --task Quadruped-VelTracking-v0  --algorithm ppo --num_envs 4096
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m
# Isaac Sim runs headless by default; pass --headon to render a window.

# Evaluate / watch
python scripts/skrl/play.py --classic --task classic-car-v0

# Multi-seed with aggregated CIs, and hyperparameter sweeps
./commands/run_seeds.sh --algorithms c2rl-ppo --env car --seeds 10 --gpu 0 -y
./commands/search.sh --algorithm c3m --env all --gpu 0 -y
```

## Observation and action convention

Every path-tracking env, in both families, exposes the same gymnasium `Dict`:

```
obs = {x, xrefs, urefs}      xrefs[k] = xref[t + k*ref_offset],  k = 0 .. ref_length-1
```

`xrefs[0]`/`urefs[0]` are the current reference; the window is clamped at the episode end.
`agents/skrl/ref_window.py` owns this layout — `RefWindow.from_space` is how every model learns
its input shape, and `RefWindow.split` is the only code that knows the flat ordering skrl produces.

**Size `ref_length` to span the discount's effective horizon `1/(1-gamma)`.** A shorter window
makes the value function non-Markov, which is the POMDP the window exists to prevent;
`RefWindow.check_markov` warns when it happens.

**Actions:** every env applies `u_applied = action` directly — no env ever adds `uref` back in.
The *agent* folds it in: `CLActor` computes `urefs[0] + pi(s)`, LQR/SD-LQR compute
`urefs[0] - K·e`. Actor and critic are both exactly invariant to translating, and where the env is
SE(2) rotating, the whole scene.

**Never clip actions for a contraction controller.** `torch.clamp` has zero gradient at
saturation and silently collapses the certified feedback Jacobian. Use the tanh-squashed backbones
(`control-squashed` / `mlp-squashed`) when bounded actions are needed.

## Algorithms

| Algorithm | Learns | Summary |
|---|---|---|
| PPO / SAC | policy | Baselines. On path-tracking they default to the `control` backbone, so the policy mean is `uref + feedback`. |
| LQR | nothing | Constant-gain `u = uref - K·e` from a single linearization. |
| SD-LQR | nothing | Same, with `K(x)` recomputed per state. |
| CV-STEM-LQR | metric only | Tsukamoto's CV-STEM/NCM. One joint SDP over sampled states (shared `nu`, `chi`), fit `chol(W^-1)`, then deploy `u = uref - R^-1 B' M(x)·e`. No RL. |
| C3M | metric + policy | Jointly trains a Riemannian metric `W(x)` and a `CLActor` so all trajectories contract at rate `lambda`, alternating metric-only and controller-only steps on `pd_loss + c1_loss + c2_loss`. |
| C2RL-PPO / C2RL-SAC | policy | See below. |

### C2RL

One policy, two phases.

1. **Synthesize the metric, then freeze it.** `cmg_method: cvstem` solves one SDP per sampled
   state and regresses the metric network onto `{x -> W*}`; `ccm` instead trains it directly on
   Manchester's C1/C2 losses (no SDP — required for a driftless plant, which is
   CV-STEM-infeasible at every `lambda`). The `cvstem` solve is expensive, so its result is cached
   under `data/` and the agent **refuses to run** rather than silently re-solve when no cached
   dataset matches the config.
2. **Ordinary PPO/SAC against that frozen metric,** on the reward
   `tracking_scaler·(V_t - V_{t+1}) - control_scaler·||u||^2`, with `V = e'M(x)e`. The tracking
   term is the per-step *decrease* of `V` — a contraction signal, not a level penalty.

The **environment** computes that reward: `_inject_ccm` hands the frozen metric to the env via
`set_ccm` and the env's own `get_rewards()` returns it, so PPO's GAE and SAC's replayed critic
target are correct with no per-algorithm reward plumbing. `_inject_ccm` raises if no env accepted
the metric — quietly training on the plain baseline reward instead would be invisible in every
metric.

The deployed law is `u = urefs[0] + pi(s)` with `pi = W2(xrefs)·tanh(W1(x)·e)`. There is no
analytic base controller in the loop and no certificate-based warm start of `pi`.

The metric and the reward always use **raw** observations: `M(x)` and `e = x - xref` are physical
quantities, and per-dimension normalization would scale `x` and `xref` independently, distorting
`e`. `use_state_norm` is `false` in every shipped config.

## Metrics

Every algorithm reports the same four `Stability/*` metrics — AUC, contraction rate `lambda`,
overshoot, and contraction score (rate/overshoot) — through one code path
(`agents/skrl/contraction_metrics.py`), which is what makes numbers comparable across algorithms and env
families. C2RL additionally reports the `Stability_maha/*` twins computed under its own metric.

When **no** episode in a round survives to be measured, `stability_summary()` publishes only
`Stability/early_end_frac` rather than a stale AUC. Seeing that key alone is the diagnostic, not a
logging bug.

**Termination.** Each env declares a certified box `[X_TERMINATION_MIN, X_TERMINATION_MAX]`, and by
default an episode ends the first step the state leaves it. The excursion is reported as
**truncation, not termination**: skrl's GAE uses `not_done = ~terminated`, so flagging it
terminated would zero the continuation value — and since every reward here is a cost, ending early
would then be strictly better than continuing. On the truncation channel with
`time_limit_bootstrap: true` the agent is indifferent to the cut. `cartpole` and `segway` ship with
the box **disabled**: both are unstable plants started far from the reference, so with it armed
every episode ended within a few steps and no stability metric could be computed at all.

## The `toy` family: an exact metric and a global optimum

Two 2-state polynomial plants — `toy-mg-v0` (Moore–Greitzer) and `toy-duff-v0` (Duffing) — exist
to supply the two things the classic family structurally cannot:

| | classic / Isaac | toy |
|---|---|---|
| contraction metric | CV-STEM SDP at **N sampled states** | SOS identity over the **whole box** |
| optimal value `V*` | not computable (4–16 states, moving reference) | exhaustive value iteration |

Everything else is shared. SOS writes the same `{x -> W*(x)}` npz the CV-STEM SDP does, so CMG
regression, C2RL, C3M and CV-STEM-LQR consume it unchanged, and `lambda` is comparable across
families because both solve the same LMI.

```bash
# Metric synthesis. SOS is a TOOL for finding W(x), not an algorithm — so it runs
# through the entry point every metric uses, and its knobs live in the `cm:` block.
python scripts/build_cm_dataset.py --task toy-mg-v0
python scripts/build_cm_dataset.py --task toy-mg-v0 --bisect-lbd   # largest certifiable rate

# Global optimum of the C2RL objective, with a Richardson error estimate.
python scripts/toy_solve.py --task toy-mg-v0 --algorithm vi --save

# Training is the ordinary --classic path; the toy envs register the same entry points.
python scripts/skrl/train.py --classic --task toy-mg-v0 --algorithm c2rl-ppo --headless
```

**Certify below the ceiling, not at it.** Maximising a *uniform* rate drives the SDP to equalise
`lambda(x)`, which destroys the state-dependence these envs exist to study. Measured on `mg`:

    lbd      12.852    6.425    3.212    1.285
    spread    1.30x    1.44x    1.87x    5.86x

The shipped `lbd` is 10% of the ceiling, putting `mg` at 6.02x spread (vs car_v1's 5.61x) and
`duff` at 2.02x (vs cartpole's 2.19x) — like-for-like stand-ins with an exact metric.
`python scripts/env_taxonomy.py --markdown` prints the full table.

## Measuring suboptimality

The contraction results assume an optimal policy; training gives empirical
convergence. The `toy` family exists to measure the size of that assumption.

```bash
# 1. Certify the metric (exact, over the whole box) — writes the coefficients too.
python scripts/build_cm_dataset.py --task toy-mg-v0

# 2. Solve the GLOBAL optimum, before any training.
python scripts/precompute_vstar.py --task toy-mg-v0 --richardson

# 3. Train. The run loads the SAME references V* was solved for, and reports
#    Optimality/gap_* at the end.
python scripts/skrl/train.py --classic --task toy-mg-v0 --algorithm c2rl-ppo --headless
```

**Why V\* is computable here.** With the reference *fixed*, tracking is a
finite-horizon, time-varying MDP whose state is `x` alone — time enters only
through which reference point is current. So V\* is **one backward sweep**:

```
V*_T(x) = 0
V*_t(x) = min_u [ c_t(x,u) + gamma * V*_{t+1}(x') ],   x' = clamp(x + dt(f(x) + B(x)u))
c_t     = -q(e' M(x) e - e'' M(x') e'') + r||u||^2
```

`c_t` is C2RL's reward negated, at C2RL's own `discount_factor`, against the same
`M`. No fixed-point iteration, no tolerance, no contraction argument — exact for
the discretised MDP. `tests/test_toy_envs.py` asserts the stage cost equals
`-env.get_rewards` step for step, because an off-by-one in the reference index
produces a plausible-looking gap that is indistinguishable from a real one.

**Why 25 references × 25 envs.** Cost is one sweep per *reference*, so 625 tasks
cost 25 sweeps rather than 625. The trajectories ship as an artifact
(`data/toy/<key>/vstar.npz`) and the trainer installs them: two processes
agreeing on a seed is not evidence they drew the same trajectories, and if they
did not, the reported gap is against the optimum of a different problem.

**What is reported.** `V^pi` is the realised discounted return of one episode,
negated into V\*'s cost units. With deterministic dynamics and the policy mean as
the action that is not an *estimate* of `V^pi`, it **is** `V^pi` — no policy
evaluation, no critic. `Optimality/gap_mean` is `V^pi - V*`, which is `>= 0` by
construction; `below_v_star_frac` reports how often discretisation breaks that,
and is the honest size of the grid error.

Note what a Bellman residual would *not* tell you here: `T^pi` is a
gamma-contraction with a unique fixed point for **every** policy, so a perfectly
fitted critic on a terrible policy has zero residual. An NLP or MPC lookahead
cannot substitute for V\* either — both return a feasible trajectory, so both
only bound `J*` from above.

**The metric is analytic.** Toy C2RL runs `metric_source: sos`, evaluating the
certified polynomial directly instead of regressing a CMG onto samples of it
(measured median relative error in the deployed `M`: `2.5e-6` vs `2.5e-2`). The
offline solve uses the same `M`, so a mismatch cannot masquerade as policy
suboptimality.

The classic envs use the **same** two scripts — `precompute_vstar.py` then
`policy_gap.py`. What makes a 4-state env affordable is the discount: `V*_0`
depends on `V*_k` only through `gamma^k`, so at `gamma = 0.01` the sweep is exact
after 9 steps and the remaining ~490 of an episode cannot change the answer
(measured difference between a 9-step and a 100-step sweep: `2.2e-16`, one double
epsilon). Beyond that, value iteration is bounded by **memory, not dimension**
(`solvers.check_budget`): the 10-state quadrotor is refused with the arithmetic
that says why.

The critic is deliberately **not** the comparison. It is a fitted, on-policy,
moving-target estimate of `V^pi`, so a gap computed from it mixes the policy's
suboptimality with the critic's fit error. `policy_gap.py` prints it only to show
the size of that bias.

## Configuration

One yaml per (env, algorithm), next to the env: `tasks/direct/<env>/agents/skrl_<algorithm>_cfg.yaml`.

**A yaml key that is not a declared field of the algorithm's `Cfg` dataclass is not applied.**
`rl_glue.filter_cfg_fields` warns loudly rather than dropping it silently, because a run with an
ignored key trains a different algorithm than its config describes and looks completely healthy.
Add the dataclass field alongside any new yaml knob.

Sweeps go through `commands/search.sh`; the searched space lives in `search/configs/<algorithm>.yaml`
and applies to every env, so the script itself never needs editing.

## Contributing and CI

`.github/workflows/ci.yml` runs `ruff` restricted to bug classes (style is left to `pre-commit`,
since import order is load-bearing on the Isaac route) plus one short end-to-end training run per
algorithm on `classic-car-v0`. That smoke run is the check that catches what imports cannot: a
config key that stops being applied still imports fine. Nothing in CI needs Isaac Sim.

The unit suite runs without Isaac Sim too: `python -m pytest tests -q`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the invariants each test enforces.

## Citing

```bibtex
@software{cho_contractionrl,
  author  = {Cho, Minjae},
  title   = {{contractionRL}: contraction-metric reinforcement learning on a unified skrl backend},
  year    = {2026},
  license = {Apache-2.0}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

Apache-2.0. See [LICENSE](LICENSE).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Contraction-based RL research repo built on Isaac Lab. All algorithms (PPO, SAC, C3M, LQR, SD-LQR,
CV-STEM-LQR, C2RL-PPO, C2RL-SAC) run through one unified **skrl** backend, across two env
families that share the same interface:

- **Isaac envs** (`tasks/direct/{quadruped,humanoid,manipulator}_{vel,path}_tracking`, `cartpole`) — need Isaac Sim.
- **Classic envs** (`tasks/direct/classic/{car,cartpole,segway,turtlebot}`) — pure NumPy/gymnasium,
  no Isaac Sim, analytically synthesize their own reference trajectory every `reset()`.

Every path-tracking env (Isaac and classic) shares the reference-window observation
`obs = {x, xrefs, urefs}` and a `reward = -‖x - xrefs[0]‖²`-family reward, so any algorithm
can train/eval on any env.

## Commands

```bash
# Install (classic envs + full test suite need no Isaac Sim)
python -m pip install -e source/contractionRL
pip install pytest ruff

# Test — 157 tests, ~8s, no Isaac Sim required
python -m pytest tests -q
python -m pytest tests/test_configs.py -q          # single file
python -m pytest tests -q -k contraction_metrics   # single module by keyword

# Lint / format (run before opening a PR)
ruff check source scripts
pre-commit run --all-files

# List envs without loading Isaac Sim
python scripts/list_envs.py --keyword classic

# Train — classic (no Isaac Sim, --classic flag, num_envs is process-level SyncVectorEnv)
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo --num_envs 4

# Reference window + encoder (see Observation below). Size the window to span the
# discount's effective horizon 1/(1-gamma) or the value function is non-Markov.
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo \
    --ref_length 51 --ref_offset 1 --encoder gru

# Train — Isaac vel-tracking (locomotion pretrain) then path-tracking (contraction control)
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo --num_envs 4096 --headless
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m --headless

# Eval / play (Isaac only; classic playback also supported via --classic)
python scripts/skrl/play.py --classic --task classic-car-v0

# Hyperparameter sweeps — always go through this, never write ad-hoc sweep scripts
./search/search.sh --algorithm c3m --env all --gpu 0 -y

# Reproduce paper results (classic envs, all algorithms x all envs, multi-seed)
./run_seeds.sh
python scripts/aggregate_seeds.py
```

Full command reference (HP overrides, reference-trajectory generation, W&B flags, MOSEK setup)
is in README.md — read it before guessing at a flag.

## Architecture

**Core library** (`source/contractionRL/contractionRL/agents/skrl/`):
- `math_utils.py` — Jacobians, PD-violation hinge losses (pure PyTorch), shared by C3M/C2RL/CV-STEM.
- `nn_modules.py` — `MLP`, `CCM_Generator` (the CMG), `CLActor` (`mu = uref + feedback`), `NeuralDynamics`.
- `models.py` — skrl model wrappers (`CMGModel`, `CLActorModel`) that plug the above into skrl's agent API.
- `c3m.py`, `c2rl.py`, `sdlqr.py`, `cvstem_lqr.py`, `ncm_synthesis.py` — per-algorithm agents + trainers.
- `agent_patches.py` — behavior changes to vendored skrl, applied post-construction. **Never edit
  vendored skrl code directly** — patch here, or in project-side subclasses, so upgrading skrl
  stays a dependency bump.

**Envs** (`source/contractionRL/contractionRL/tasks/direct/`):
- `classic/common/env_base.py` — `BaseEnv` shared by car/cartpole/segway/turtlebot. Analytical
  `get_f_and_B(x)`, generates its own reference trajectory each episode.
- `common/path_tracking_base.py` — the Isaac-side equivalent interface (`{x, xrefs, urefs}` layout,
  `Stability`/`Episode` W&B logging, `set_ccm` for the Mahalanobis reward).
- Any member a contraction agent discovers via `getattr` (`get_f_and_B`, `get_rollout`, `set_ccm`,
  `x_dim`, `u_dim`) **must exist with the same signature on both families** —
  `tests/test_isaac_parity.py` enforces this statically without needing Isaac Sim.

**Observation**: `obs = {x, xrefs, urefs}` (a gymnasium `Dict`), where
`xrefs[k] = xref[t + k*ref_offset]` for `k = 0..ref_length-1`, clamped at the episode end.
`xrefs[0]`/`urefs[0]` are the CURRENT reference. `agents/skrl/ref_window.py` owns the layout:
`RefWindow.from_space` is how every model learns its input shape (nothing infers a layout from
`obs_dim`), `RefWindow.split` is the ONLY place that knows the flat ordering skrl produces
(`sorted(keys)` -> `[urefs, x, xrefs]`), and `Feats` owns the relative-position /
wrapped-angle / SE(2) feature maps that every network input goes through.
`RefWindow.check_markov` warns when the window is too short to span `1/(1-gamma)` — a shorter
window makes V non-Markov, which is the POMDP the window exists to prevent.

**Actor / critic**: one architecture each, both consuming the window through the same
`--encoder` (`mlp` | `gru` | `attn`, via `PreviewSequenceEncoder`):

- actor  `u = urefs[0] + W2(xrefs) · tanh(W1(x) · e)`, `e = error(x, xrefs[0])` — `W1` reads
  the current configuration only, `W2` the reference path only.
- critic `V = MLP([phi(x, e) ‖ psi(xrefs)])` (`combine: concat`; `bilinear`/`film` are
  commented out in `models.py`).

Both are exactly invariant to translating and (where the env is SE(2)) rotating the whole
scene — enforced by `tests/test_models_window.py`.

**Action convention**: every env applies `u_applied = action` directly — no env ever adds `uref`
back in. Instead the *agent* folds `urefs[0]` into its output: `CLActor` computes
`urefs[0] + feedback`; LQR/SD-LQR compute `urefs[0] - K·e`. PPO/SAC on path-tracking envs default to the `control` backbone
(`CLActorModel`), so their policy mean is also `uref + feedback` unless `backbone: mlp` is set.

**C3M**: jointly trains a Riemannian metric `W(x)` (via the CMG) and a `CLActor` controller so all
trajectories contract at rate λ, alternating CMG-only and controller-only gradient steps against
the same `pd_loss + c1_loss + c2_loss` objective.

**C2RL**: two policies (`con_policy`, near-zero γ; `opt_policy`, standard γ) sharing one CMG, both
optimizing the same Mahalanobis reward `-‖e‖²_M/std - ‖u-uref‖²/std`. Only `con_policy`'s mean
control/Jacobian shapes the CMG; `opt_policy` is what's deployed. See README.md's "Algorithm
Reference" section for the full per-epoch workflow and normalization rules — the interaction
between `use_state_norm` and the `uref`-folding controllers is subtle and load-bearing.

**Config files** live next to each env, one yaml per algorithm
(`tasks/direct/<env>/agents/skrl_<algorithm>_cfg.yaml`). A yaml key not declared on the
algorithm's `Cfg` dataclass is **silently dropped** (`rl_glue.filter_cfg_fields`) — add the
dataclass field in the same commit as any new yaml knob; `tests/test_configs.py` enforces this.

**Sweeps**: `search/search.sh` is the one entry point; the searched space lives in
`search/configs/<algorithm>.yaml` and applies to every env, so the script itself never needs
editing. C3M sweeps optimize `Stability/contraction_score`; LQR/SD-LQR/CV-STEM/C2RL-*-cvstem
optimize `Stability/auc_mean` (minimized); PPO/SAC optimize raw reward. CV-STEM trials run
through `search/sweep_runner.py`, which records a poison AUC on SDP infeasibility instead of
leaving the trial metric-less (a metric-less trial is silently ignored by bayes bookkeeping).

**Metrics**: all algorithms report the same four `Stability/*` metrics (AUC, contraction rate λ,
overshoot, contraction score = rate/overshoot) through the same code path
(`agents/skrl/eval_metrics.py`, `tasks/direct/common/eval_metrics.py`), making numbers directly
comparable across algorithms and env families.

## House rules (each has caused a real silent failure before)

- **Never let a config key be silently ignored** — see Config files above.
- **Never let a missing capability degrade quietly.** Prefer raising over falling back — e.g.
  `C2RLSkrlTrainer._inject_ccm` raises rather than silently training on the plain baseline reward.
- **Do not modify vendored skrl code** — patch via `agent_patches.py` or project-side subclasses.
- **SDP infeasibility is a signal, not a nuisance.** If CV-STEM reports 0% feasible, fix the
  envelope (`w_lb`, `cvstem_r_scaler`), don't lower `min_feasibility_rate`. A driftless plant
  (e.g. turtlebot, `f≡0`) is infeasible for CV-STEM at every λ by construction — use
  `cmg_method: ccm` there instead.
- **Actions must not be clipped for contraction controllers.** `torch.clamp` has zero gradient at
  saturation and silently collapses the certified feedback Jacobian. Use the tanh-squashed
  backbones (`control-squashed`/`mlp-squashed`) instead when bounded actions are needed.

## Adding an environment

1. Subclass `classic/common/env_base.py` (analytical) or `common/path_tracking_base.py` (Isaac).
2. Register it with one `skrl_<algorithm>_cfg_entry_point` per supported algorithm in the
   package `__init__.py`.
3. Add its short name to `CLASSIC_ENVS` in `tests/conftest.py` so the contract/config test suites
   cover it automatically.

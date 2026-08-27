# rule.md

Standing rules for two decisions that keep getting re-litigated. Both have a
right answer that follows from the metric definitions, and both have a failure
mode where the obvious fix is the wrong one.

Read this before changing `terminate_out_of_box`, `x_termination_*`, `dt`,
`time_bound`, or `max_episode_len` on any env.

---

## 0. This file is permissioned

**Adding to, editing, or removing anything in `rule.md` requires the repo owner's
explicit permission, per rule.** Do not write to this file as a side effect of
other work, do not "tidy" it, and do not delete a rule because it is
inconvenient for a change in flight.

**Every change made to this repository must conform to the rules below.** If a
change cannot be made without violating a rule, that is a signal to stop and
raise it — not to amend the rule. A rule here exists because breaking it already
cost a real, measured failure; the rule is the cheaper half of that lesson.

To propose a change: state which rule, what the new text would be, and the
measurement that justifies it, then wait for approval.

---

## 1. What happens when the state leaves the box

### The five layers, in the order `step()` applies them

`tasks/direct/classic/common/env_base.py:642`

| # | layer | what it does | why |
|---|---|---|---|
| 1 | action sanitize | `nan_to_num(u)`, then `clamp(u, U_MIN, U_MAX)` with `U_MIN/MAX = 2 x UREF` | 2x, **not** the declared action space: `u = uref + feedback` controllers legitimately exceed the uref box, and clamping to it breaks the contraction certificate |
| 2 | non-finite carry-forward | `carry_forward_nonfinite(next_x, x_t)` — NaN/Inf state reverts to the previous one | one diverged env must not poison the batch with NaN |
| 3 | position freeze | dims `[:pos_dimension]` outside `[X_MIN, X_MAX]` revert to their previous value | hard wall on position only |
| 4 | global box clamp | `clamp(wrap_angles(next_x), X_MIN, X_MAX)` | every remaining dim is pinned to the box face |
| 5 | early-termination box | `_left_termination_box(next_x)` | ends the episode where the clamp would otherwise activate silently |

### Invariants — do not break these

- **Layer 5 must be measured on the raw integrated state, before layers 3 and 4.**
  Both erase the excursion, so a check placed after them can never fire. This
  ordering is load-bearing, not stylistic.

- **Leaving the box is truncation, not termination, by default.**
  `x_termination_terminal: false`. On a cost reward (every reward here is
  `-||e||`-family), a zeroed bootstrap is a *suicide bonus*: ending the episode
  is worth more than continuing it, so termination teaches the agent to leave
  the box on purpose. Only set `true` with a positive-reward formulation.

- **Early-ended episodes are excluded from `Stability/*`, never padded.**
  The env reports `episode_ended_early`; `StatManagerEnvWrapper` invalidates
  those slots. AUC/lambda/C are defined on the full-length normalized error
  curve, and a curve cut at step k is a different quantity — padding a flat tail
  would report a fabricated hold as if the policy had achieved it.

- **Eval always disarms the box** (`_disarm_termination_for_eval`,
  `scripts/skrl/train_utils.py:205`). Truncation *inverts* AUC: a policy that
  falls at step 20 stops accumulating error and scores a better-looking AUC than
  one that tracks imperfectly for the whole horizon, and the number stops being
  comparable to the LQR / C3M / CV-STEM baselines, all measured over full
  episodes. Training-time and eval-time box behaviour therefore differ **on
  purpose** — this is not an inconsistency to fix.

- **Never `torch.clamp` the action of a contraction controller** to the uref box.
  Zero gradient at saturation silently collapses the certified feedback
  Jacobian. Use the tanh-squashed backbones when bounded actions are needed.

### The box has two jobs, and only one of them is the clamp's

This is the rule the rest of Rule 1 exists to serve:

- **Validity domain.** Outside the box the plant model, the contraction
  certificate and the reference trajectory are all meaningless. The correct
  handling is to **end the episode** (layer 5).
- **Numerical backstop.** Stop a diverged env producing NaN/overflow before the
  episode actually ends. The correct handling is to **clamp** (layers 3-4).

**The clamp is a backstop, never a model.** It is not physics:

| layer | physically? | honest reading |
|---|---|---|
| position freeze | no | a wall that **retains velocity** — position stops while the velocity dim keeps its value, so position no longer integrates velocity. A real stop either reverses (elastic) or zeros (plastic) the normal velocity. This is an infinitely sticky wall still being pushed. |
| angle clamp | no | for a segway, ±0.90 rad *is* a real mechanical limit — but then it has **fallen over**, and the right model is termination, not "leans on an invisible wall and keeps trying". |
| velocity clamp | **no** | an infinite force/torque. There is no mechanism that bounds a velocity instantaneously. Pure backstop, no physical content. |

So the clamp is defensible **only while it fires rarely**. Once it fires often
it silently *becomes* the dynamics, and then:

- the simulated plant is no longer `xdot = f(x) + B(x)u`, so **no CV-STEM / C3M
  certificate applies to it** — the metric was solved for a different system;
- AUC / lambda / C are computed on a fabricated plant;
- PPO fits the wall instead of the plant.

**Always measure the clamp rate before trusting a contraction number.** If it is
not small, the certificate is void regardless of how the SDP was solved.

This is now logged: `Clamp/frac_any`, `Clamp/frac_<state_name>` per dimension,
plus `Clamp/frac_u_saturated` and `Clamp/frac_nonfinite`. Deliberately NOT gated
behind the stability buffer, so it still reports when no episode completes and no
AUC is published. `frac_u_saturated` separates "out of authority" (saturating)
from "failing inside its authority" (not saturating).

### Measured: segway, 2026-08-27

105,888 env-steps under the analytic CV-STEM-LQR law. Fraction of `(env, step)`
pairs whose **raw integrated** state left the box (measured before layers 3-4,
which erase the excursion):

| dim | bound | clamped | handled by |
|---|---|---|---|
| `pos_x` | ±5.0 m | 25.97 % | layer 3 (freeze) |
| **`pitch`** | ±0.90 rad (±51.6°) | **95.20 %** | layer 4 (clamp) |
| `vel_x_b` | ±1.0 m/s | 0.00 % | — |
| **`pitch_rate`** | ±pi rad/s | **94.30 %** | layer 4 (clamp) |
| **any dim** | | **95.23 %** | |
| `|u|` clipped | ±12 | 0.00 % | — |

`pos_dimension = 1`, `angle_idx = (1,)`, so the freeze only ever touches
`pos_x`; pitch and pitch_rate are raw-clamped. Note `pitch` is in `angle_idx`,
but its ±0.90 box sits far inside ±pi, so wrapping never brings it back — the
clamp binds.

**Segway spends ~95% of every episode pinned against a 51.6° tilt stop with its
pitch rate clipped**, and `terminate_out_of_box: False` means nothing ends the
episode. The backstop has become the model. This single fact explains the whole
segway picture measured the same day:

- certified `lambda = 0.0514` never realized (measured 0.011-0.013) — the metric
  certifies the ODE, not the clamped system;
- CV-STEM feedback harmful at **every** gain (91.4 AUC at zero gain vs 95.7 at
  the config gain vs 134.8 at 25x) — `K = (1/r)B'M` is derived for a plant that
  is not running;
- `AUC/T = 1.37 > 1` — a fallen, pinned segway cannot reduce its error;
- the CV-STEM-LQR regression pretrain a no-op (82.3 vs 82.0 control).

`|u|` clipped 0.00% rules out actuator saturation: the controller is not even
straining. The problem is upstream of actuation.

### `terminate_out_of_box`: on by default, and off is a real risk

Default `True`, and it defaults to the state box itself, so it fires exactly
where layer 4 already silently activates — the same event, reported instead of
hidden.

With it `False`, a diverged env is pinned at the box face and keeps emitting
off-distribution transitions for the **rest of the episode**. The cost scales
with episode length, so long-horizon envs are hit hardest:

| env | `terminate_out_of_box` | steps | junk steps from a failure at step 50 |
|---|---|---|---|
| segway | `False` | 2000 | ~1950 (97%) |
| cartpole | `False` | 500 | ~450 (90%) |
| all others | default `True` | — | 0 |

At 97% the rollout batch — and therefore the seed — decides what PPO fits.
**If you set it `False`, say why in the env module, and re-check it whenever you
lengthen the episode.** Segway's `False` predates its 15 s -> 60 s change and
has not been revalidated at 2000 steps.

---

## 2. How long the episode has to be

### The requirement

The horizon must give the controller time to reach 5% of the initial error.
For an error decaying as `||e(t)|| = C * ||e0|| * exp(-lambda*t)`:

```
||e(T)||/||e0|| = C * exp(-lambda*T) <= 0.05

    T95  =  ln(20*C) / lambda           seconds
    N95  =  T95 / dt                    steps
```

`C` is the overshoot (the `Stability/overshoot` metric), `lambda` the
contraction rate. Both are reported by every algorithm through the same path,
so use the run's own numbers.

### Interpreting AUC against the horizon

`AUC = (dt/||e0||) * sum ||e||` is a trapezoidal integral **in seconds**, so:

- `AUC/T` is the **mean normalized error** — dimensionless, horizon-independent,
  and the number to compare across envs.
- `AUC/T >= 1` means the error averaged *above where it started*. Nothing
  contracted. No horizon change fixes this.
- For the exponential model, `AUC(T) = (C/lambda) * (1 - exp(-lambda*T))`, with
  asymptote `C/lambda`. Use this for **sizing**, not prediction — real error
  curves are not clean exponentials and the model brackets rather than
  reproduces the measurement.

### Decision procedure

1. Measure `lambda` and `C` from an actual run. **Do not size from the
   certified `lbd`** — see the worked example.
2. Compute `T95 = ln(20*C)/lambda`.
3. Compare with the env's `time_bound`:
   - `time_bound >= T95` -> the horizon is adequate. A bad AUC is the
     controller's fault. **Do not extend the horizon.**
   - `time_bound < T95`, and `T95` is practical (`N95` fits in memory and the
     48 h wall) -> extend `time_bound`.
   - `time_bound < T95`, and `T95` is impractical -> **the horizon is the wrong
     diagnosis.** A controller needing 6x the horizon is not slow, it is not
     contracting. Fix the controller or the certificate.
4. If you change `time_bound`, re-check: `ref_length` AUTO tracks
   `max_episode_len`, so the observation width and GPU memory move with it, and
   `terminate_out_of_box: False` gets proportionally more expensive (Rule 1).

### Worked example — segway, 2026-08-27

| quantity | certified | measured |
|---|---|---|
| lambda | 0.0514 | 0.0117 |
| C | ~3 | ~3.2 |
| T95 | 79.7 s (2655 steps) | 355 s (11849 steps) |
| have | 60 s (2000 steps) | 60 s |
| shortfall | 1.3x | **5.9x** |

Observed `AUC/T = 82/60 = 1.37`.

Two readings, and the second is correct:

- *Naive:* 60 s < 79.7 s, so extend to 80 s.
- *Correct:* `AUC/T = 1.37 >= 1`, so the error never got below its starting
  value at all — the horizon is not what is binding. The measured rate is 4-5x
  short of the certificate, needing 355 s (5.9x) to hit 95%. Sizing from the
  *certified* lambda would have hidden this behind a plausible 1.3x tweak.

The certificate is the thing that is wrong here, not the clock. Measured
2026-08-27: every nonzero CV-STEM feedback gain scores worse than **no**
feedback (91.4 AUC at zero gain vs 95.7 at the config gain vs 134.8 at 25x),
so segway's frozen metric yields a non-stabilizing feedback direction. See
`memory/project_segway_cvstem_feedback_harmful.md`.

### Regenerating the per-env table

`lbd` per env lives in `tasks/direct/classic/<env>/agents/skrl_c2rl_ppo_cfg.yaml`
under `cm:`; `dt` and `time_bound` in `<env>/env.py`. Sizing off the yaml `lbd`
gives the *certified* horizon — treat it as an optimistic lower bound on T95
until a run confirms the rate, exactly as segway shows.

---

## Quick checklist

Before reporting a tracking result:

- [ ] **What fraction of steps hit the clamp?** If it is not small, the
      contraction certificate is void and every `Stability/*` number describes a
      fabricated plant. Check this FIRST — it invalidates everything below.
- [ ] Is `AUC/T` above 1? If so, stop — nothing is contracting, and neither
      horizon nor gain tuning is the fix.
- [ ] Was the horizon sized from a **measured** lambda, not the certified one?
- [ ] Is `terminate_out_of_box` consistent with the current episode length?
- [ ] Did eval disarm the box (it should) and training keep it armed (it should)?
- [ ] Are early-ended episodes excluded from the stability numbers?

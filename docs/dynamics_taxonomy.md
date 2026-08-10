# A taxonomy of dynamics for state-dependent contraction rate

**Related work** positions this against prior contraction work. **§1 is the
taxonomy**: one definition of the three classes, and each membership
condition stated as a proposition with its proof immediately after. §2 measures
them. §3 holds the supporting results §1 leans on. §4 is the generality boundary and
§5 is what is not claimed.

**Scope.** Contraction feasibility only. The control box plays no part: no plant
is called infeasible because its certified gain is large. `‖K‖` is reported so
the cost stays visible, never tested.

Implementation: `scripts/feasibility_certificate.py`. Program under test:
`cvstem_joint` in `agents/skrl/ncm_synthesis.py`.

**Standard results used, not reproved.** The Hautus (Popov–Belevitch–Hautus)
lemma for controllability/stabilizability; the LQR `α`-shift for a prescribed
decay rate (Anderson & Moore); existence and uniqueness of the stabilizing CARE
solution under stabilizability plus observability. Class I below is exactly
"not λ-stabilizable somewhere on the box" and is a citation, not a contribution.

**Setup.** Plant `ẋ = f(x) + B(x)u` on a compact box `X ⊂ Rⁿ`, `A(x) = ∂f/∂x`.
For a sample set `S ⊂ X` and parameters `(λ, ε, dt, r, w_lb, w_ub)`, the program
`P(S)` asks for `W̄ₖ ⪰ 0` and shared `ν, χ ≥ 0` with

```
(C1)  (W̄ₖ − I)/dt + AₖW̄ₖ + W̄ₖAₖᵀ + 2λW̄ₖ − ν(2/r)BₖBₖᵀ  ⪯  −εI
(C2)  I ⪯ W̄ₖ ⪯ χI
(C3)  ν ≤ 1/w_lb,   χ ≤ ν·w_ub
```

Write `λ*(x) = sup{λ : P({x}) feasible}` and `λ*(S)` likewise, and `λ' := λ +
1/(2dt)`.

---

# Related work

Contraction-based synthesis treats the contraction rate as a single scalar fixed
in advance: control contraction metrics [MS17] and CV-STEM/NCM [TC21] both impose
pointwise LMIs across the state space but certify one uniform `λ`, so the
state-dependence lives entirely in the *metric* `W(x)` and never in the *rate*.
The one line of work that does make the rate state-dependent — Lohmiller and
Slotine's shaping of state- and time-dependent convergence rates [LS10] — is a
synthesis result: it *prescribes* a desired rate profile for feedback-linearizable
systems, and therefore presupposes what we set out to determine. Finite-time
Lyapunov exponents measure genuinely state-varying rates but only in
time-averaged form, on the dynamical-systems rather than the certificate side
[TCS21]. We instead take the per-state certified rate `λ*(x)` as the object of
study for a *given* plant: we show the joint program decouples, so that
`λ*(S) = min_x λ*(x)` and a per-state analysis is meaningful (Proposition 3); we
give a constructive Riccati certificate that returns feasibility together with the
metric envelope the plant demands, replacing SDP search (Proposition 5); and we
classify dynamics by whether `λ*(x)` is constant or varies over the operating box,
isolating which states bind the certificate and what a nearly-uncontrollable mode
costs the shared metric scale (Corollary 4.2). The question is thus **diagnostic
rather than prescriptive** — not "what rate shall we impose?" but "does this plant
admit a uniform rate at all, and if not, why not?"

* **[LS10]** W. Lohmiller and J.-J. E. Slotine, *Shaping state and time-dependent
  convergence rates in non-linear control and observer design*, arXiv:1004.2971,
  2010.
* **[MS17]** I. R. Manchester and J.-J. E. Slotine, *Control Contraction Metrics:
  Convex and Intrinsic Criteria for Nonlinear Feedback Design*, arXiv:1503.03144.
* **[TC21]** H. Tsukamoto and S.-J. Chung, *Neural Contraction Metrics for Robust
  Estimation and Control: A Convex Optimization Approach*, arXiv:2006.04361.
* **[TCS21]** H. Tsukamoto, S.-J. Chung and J.-J. E. Slotine, *Contraction Theory
  for Nonlinear Stability Analysis and Learning-based Control: A Tutorial
  Overview*, arXiv:2110.00675.

---

# 1. The taxonomy

**Definition 1 (the three classes).** *A plant on a box `X`, at fixed
`(λ, ε, dt, r, w_lb, w_ub)`, is*

```
class I    if  P({x}) is infeasible at some x ∈ X, for every ν, χ, r, dt, envelope
class II   if  it is not class I and λ*(x) is CONSTANT on X
class III  if  it is not class I and λ*(x) is NON-CONSTANT on X
```

*The three are mutually exclusive and exhaustive by construction.*

All three are properties of the **program** `P` — hence of the pair
(plant, box) *and* of the modelling choice `A = ∂f/∂x` that `P` is fed. §4 draws
that boundary and measures it.

The classes run in order of what a certificate can do with the plant: class I
admits none, class II admits one uniform rate, class III admits a rate that a
subset of the box can improve. Definition 1 is not directly checkable — `λ*(x)`
costs an SDP bisection per state and class I quantifies over every envelope — so
each class gets a condition below.

## 1.1 Class I — the Hautus test

**Proposition 1 (class I).** *If `(A(x) + λI, B(x))` is not stabilizable for some
`x ∈ X` — equivalently, by the Hautus (PBH) lemma, if*

```
rank[ A(x) − sI ,  B(x) ] < n     for some s with Re s ≥ −λ
```

*then the plant is class I on `X`: `P({x})` is infeasible for every
`(ν, χ, r, dt, w_lb, w_ub)`.*

*Proof.* This is the Hautus lemma and is cited, not derived. `(C1)` is a
state-feedback synthesis LMI in disguise, and no such condition is satisfiable
for an unstabilizable pair; demanding a prescribed decay rate `λ` rather than
mere stability is the substitution `A ↦ A + λI`, the classical `α`-shift of LQR
(Anderson & Moore). Corollary 4.1 rederives it from this document's quantitative
inequality, purely as a consistency check. ∎

Class I is therefore a name for *"not λ-stabilizable somewhere on the box,
**in the pointwise drift-Jacobian sense**"* — and that qualifier is load-bearing,
not decorative. `(C1)` is fed `A = ∂f/∂x`, which cannot see the Lie bracket that
steers a differentially flat or nonholonomic plant: pvtol and turtlebot are both
class I here while both are perfectly controllable, and for turtlebot `f ≡ 0` so
`A ≡ 0` and the test inspects *none* of the dynamics. **§4.2 measures the gap;
§4.3 states how to read this.** The remedy for such plants is a method that sees
brackets (`cmg_method: ccm`), not a different envelope.

What is worth adding is only how to **evaluate** the test:

* **The test is finite.** Uncontrollable modes sit only at eigenvalues of `A`, so
  testing `s ∈ spec(A) ∩ {Re s ≥ −λ}` misses nothing.
* **Use `σ_min`, not `rank`.** The usable form is
  `σ_min([A(x) − sI, B(x)]) = 0`: rank is discontinuous in the data and its
  tolerance is arbitrary, while `σ_min` degrades smoothly and doubles as the
  quantitative margin of Corollary 4.2.
* **Use the SVD, never eigenvectors.** A defective `A` (pvtol's uncontrollable
  block is nilpotent) returns a near-parallel eigenbasis, and modal authority
  read off it is meaningless — measured, that route reported `1e-2` where the
  true margin is `1e-19`.

## 1.2 Class II — an orthogonal gauge

**Proposition 2 (class II).** *Suppose there is a map `T : X → O(n)` into the
**orthogonal** group with*

```
T(x) A(x) T(x)ᵀ = A₀     and     T(x) B(x)B(x)ᵀ T(x)ᵀ = B₀B₀ᵀ      for all x ∈ X
```

*for a single pair `(A₀, B₀)`. Then `λ*(x) ≡ λ*₀` on `X`, so the plant is class II
(given it is not class I), and by Corollary 3.1 no subset of `X` certifies a
faster rate.*

*Proof.* Fix `x` and let `T = T(x)`. The map `W̄ ↦ TᵀW̄T` sends feasible points of
`P({x₀})` to feasible points of `P({x})` and is a bijection, because each of the
three constraints is preserved:

**(C1).** Substituting `W̄ = TᵀW̄₀T` and using `A = TᵀA₀T`, `BBᵀ = TᵀB₀B₀ᵀT`,
`TᵀT = I`:

```
AW̄ + W̄Aᵀ = Tᵀ(A₀W̄₀ + W̄₀A₀ᵀ)T,      BBᵀ = TᵀB₀B₀ᵀT,      2λW̄ = Tᵀ(2λW̄₀)T
```

and the proxy term also transforms cleanly because `T` is orthogonal:
`(W̄ − I)/dt = Tᵀ(W̄₀ − I)T/dt`, using `TᵀIT = I`. So the whole left side of (C1)
is `Tᵀ(·)T` applied to the left side at `x₀`, and the right side satisfies
`−εI = Tᵀ(−εI)T`. Orthogonal congruence preserves the PSD order, so one holds iff
the other does.

**(C2).** Orthogonal congruence preserves eigenvalues, so `I ⪯ W̄ ⪯ χI` holds iff
`I ⪯ W̄₀ ⪯ χI` does.

**(C3).** Involves no state-dependent data.

Hence `P({x})` and `P({x₀})` are feasible at exactly the same `λ`, so
`λ*(x) = λ*(x₀) = λ*₀`. ∎

**Orthogonality is load-bearing, twice.** A general similarity preserves the `A`
and `BBᵀ` terms of (C1) but *not* the `I` inside `(W̄ − I)/dt`, nor the `−εI`, nor
(C2) — so it can move both the envelope and the rate. Proposition 2 is a
statement about the metric-normalised program, not about `(A, B)` up to arbitrary
coordinates.

**No plant in this repo satisfies Proposition 2's hypothesis** — see §2.3. It is
a valid sufficient condition with no instance here, which is worth stating
plainly rather than leaving implied.

## 1.3 Class III — no structural condition, only a one-directional screen

Class III is the default: not class I, and `λ*` not constant. There is **no**
proven structural condition for it here. What there *is* — and this is the cheap
thing to look at — is a screen that reads straight off `B(x)`:

> **Screen.** Compute the singular values of `B(x)` over the box. If they
> **vary**, the plant is class III. If they are **constant**, class II is
> *suspected* but NOT implied — see "Constant `B` does not give a constant rate"
> below.

One SVD per sample, no CARE and no SDP. Measured against the exact per-state
`λ*(x)` (one-sample SDPs), it separates all nine feasible plants:

| env | class | `sv(B)` spread | exact `λ*(x)` spread |
|---|---|---|---|
| car | **II** | **1.000** | **1.0000** |
| quadrotor | **II** | **1.000** | **1.0000** |
| tora | III | 1.062 | 3.652 |
| segway | III | 1.139 | 1.755 |
| cartpole | III | 2.213 | 2.092 |
| ball_and_beam | III | 3.470 | 1.373 |
| two_link_arm | III | 5.035 | 1.927 |
| aircraft | III | 16.936 | 2.400 |
| auv | III | 42.214 | 5.184 |

Exactly `1.000` for both class-II plants — whose `B` is in fact literally
constant — and `> 1` for all seven class-III plants, the tightest margin being
tora at `1.062`. (pvtol and turtlebot also have constant `sv(B)`, but Proposition
1 catches them first: their `B` rotates without changing its singular values.)

### Why `sv(B)` and not the simpler "`B` is state-dependent"

On these nine plants the two are equivalent — both class-II plants have a
literally constant `B` — so nothing is lost here by saying the simpler thing. But
the simpler statement is **false in general**, and its counterexample is exactly
the case Proposition 2 was written for: a `B` that **rotates**.

If `B(x) = R(x)B₀` with `R(x)` orthogonal, then `B(x)` is genuinely
state-dependent while `B(x)B(x)ᵀ = R B₀B₀ᵀ Rᵀ` has constant singular values, and
whenever that rotation extends to a gauge for `A` too, Proposition 2 gives
`λ*` constant — class II. Minimal instance, `f ≡ 0` on `R²` with `B(x) = R(x₁)`:

```
A ≡ 0,  B(x)B(x)ᵀ ≡ I    ⟹   Proposition 2 applies with T ≡ I
measured:  max|B(0) − B(π)| = 2.0000   (B really does vary)
           ρ(x) = 1.739818167          identical to 12 digits
```

The shape is not exotic and occurs in this repo: **pvtol**'s `B` depends on roll
and **turtlebot**'s on yaw, and both have `sv(B)` spread exactly `1.000`. They do
not settle the question only because Proposition 1 classifies them first, as
class I.

So `sv(B)` is the right quantity: same cost, it is the orthogonally-invariant
part of `B`, and it has no rotation counterexample.

### Constant `B` does NOT give a constant rate

The tempting strengthening — *"`B` constant ⟹ `λ*` constant"* — is **false**, and
the counterexample is one-dimensional. Take `n = m = 1`, `B ≡ 1`, drift Jacobian
`a(x)` varying (e.g. `f(x) = −x³/3`, `a(x) = −x²`). The scalar LMI is minimised at
`W̄ = 1`, so at `ν = 1/w_lb`

```
λ*(x)  =  1/(r·w_lb)  −  a(x)  −  ε/2
```

which varies exactly as much as `a(x)` does. Measured with the CARE certificate,
`a ∈ [−4, 2]` gives `ρ` from `0.1518` to `4.6520` — a spread of **30.7** with `B`
constant throughout.

Nor does adding constant `spec(A)` save it. With `A(v) = [[0, v],[0, 0]]`
(nilpotent, `spec ≡ {0,0}`) and `B = [0;1]` constant, so that **both** `spec(A)`
and `sv(B)` are constant, `ρ` still ranges over `608.8 → 5.34 → 13.39` as
`v: 0.1 → 2 → 100`, a spread of **113.9**. What moves the rate there is `A`'s
*non-normal* structure and how it couples into `B` — neither of which is visible
in the spectra taken separately.

So the screen is a **heuristic**, not a theorem, and it is refuted in the
`sv(B) constant ⟹ class II` direction by any 1-D plant with non-constant `f''`.
It survives on these nine because their class-II members (car, quadrotor) happen
to have both a constant `B` *and* an `A`-variation that does not reach
`λ_max(P)` — which §2.2 shows is a property of their particular box.

**Status: empirical, not proved**, and two limits are already visible.

* **It orders nothing.** ball_and_beam has the third-largest `sv(B)` spread
  (3.470) and the *smallest* rate spread (1.373), while tora has the smallest
  `sv(B)` spread (1.062) and the second-largest rate spread (3.652). Use it for
  the yes/no, never for the size.
* **The obvious proof route is dead** — §2.3.

**A caution that §4.2 sharpens.** The screen fires exactly when `B` varies — and
a varying `B` is precisely what makes the drift-only model `A = ∂f/∂x` an
incomplete description of the plant, since the neglected term `Σᵢ uᵢ ∂bᵢ/∂x` is
zero iff `B` is constant. So the screen is most confident exactly where the
underlying program is least faithful. It remains a correct statement *about this
program*; whether it survives on the generalized Jacobian is untested.

The partial reason it works is Proposition 6: at fixed `A`, more control
authority means a strictly easier state. The gap is that `A` is never fixed
between two states of a real plant, and §3.5 shows authority is not even the
dominant mechanism on segway and aircraft.

Failing the screen, the rigorous options are:

| test | cost | status |
|---|---|---|
| **`λ_C(x)` takes two values** — the largest rate the CARE certificate carries inside the envelope, `sup{λ : λ_max(P_λ(x)) ≤ 1/w_lb}` | ~10 Riccati solves/sample | **Evidence only** (a lower bound on `λ*`), but the **best screen available**: agrees on 9 of 9 and is in λ units. **Preferred.** |
| `ρ(x) = λ_max(P(x))` takes two values | 1 CARE solve/sample | Evidence only; same 9 of 9, but wildly off in magnitude. |
| `λ_C(x₁) > λ_E(x₂)`, with `λ_E` the modal upper bound below | ~10 Riccati + 1 eig | **Rigorous** when it fires. Fires on 2 of 9 (tora, auv). |
| `λ*(x)` by one-sample SDP at two states | 2 SDP bisections | **Rigorous, always decisive.** |

### Simplification attempts, and what they cost

Three attempts to replace the machinery with something closed-form. One helped,
two failed, and the failures are informative.

**Worked — `λ_C` instead of `ρ`.** `ρ` answers *"how large a metric does `x`
demand at a fixed λ"*; `λ_C` answers *"how fast can `x` go before the envelope
stops it"*. Same information, but in the units of the answer, so it is directly
comparable to the true spread instead of being off by orders of magnitude:

| plant | true `λ*` spread | `λ_C` spread | `ρ` spread |
|---|---|---|---|
| car | 1.0000 | **1.0000** | 1.0000 |
| quadrotor | 1.0000 | **1.0000** | 1.0000 |
| segway | 1.755 | **1.97** | 2.20 |
| cartpole | 2.092 | **2.81** | 7.26 |
| ball_and_beam | 1.373 | 3.69 | 58.6 |
| aircraft | 2.400 | **10.0** | 3.26 |
| tora | 3.652 | 125 | 1 152 |
| auv | 5.184 | **30.3** | 5 618 |

**Failed — the identity-metric probe.** Setting `W̄ = I` makes the proxy term
`(W̄ − I)/dt` vanish *exactly*, giving the closed form
`λ_I(x) = ½·λ_min(μBBᵀ − A − Aᵀ) − ε/2` with `μ = 2/(r·w_lb)`, one symmetric
eigendecomposition and no Riccati solve. It is useless in practice: `BBᵀ` is
rank-deficient for any underactuated plant, so `λ_min` is set by the unactuated
directions and `λ_I < 0` at **every** state of **every** env here (best case
`−0.05`). A non-trivial metric is not a convenience for these plants; it is the
only thing that makes a positive rate possible at all.

**Failed — folding class I into one scalar.** At an *exact* left eigenvector the
slack `δ` vanishes and `χ` drops out of (★) entirely, giving the much tighter
modal upper bound

```
λ_E(x) := min over left eigenpairs (w, s) of A(x)  [ (ν/r)·w*BBᵀw − Re s ]  −  ε/2   ≥  λ*(x)
```

`λ_E ≤ 0` would then mean "no positive rate certifies", which is class I — one
scalar covering Proposition 1 *and* bracketing `λ*`. **It does not work**, and it
fails on exactly the plant predicted in §1.1: pvtol's `A` is defective, its
eigenbasis is near-parallel, and `λ_E = 6.2 > 0` — so it declares a class-I plant
feasible. The SVD-based Hautus test cannot be absorbed into an eigenvector-based
scalar; it has to stay separate. (`λ_E` is still sound as an upper bound wherever
`A` is semisimple, which is every feasible env here, and it is what makes the
rigorous certificate above fire on 2 of 9 rather than 1 of 9.)

**Net.** The class-I criterion was already minimal. The class II-vs-III screen
improved from `ρ` to `λ_C`. No new *structural* criterion emerged, and §2.3
explains why one is unlikely to: every simple structural candidate is vacuous on
plants whose `A` varies, which is all of them.

## 1.4 The decision procedure

```
σ_min([A−sI, B]) = 0 somewhere, Re s ≥ −λ ?  ──yes──>  class I    (infeasible, any envelope)
                    │no
sv(B(x)) varies over the box ?               ──yes──>  class III  (subsets buy rate)
                    │no
                          SUSPECT class II -- verify, do not conclude
```

**Only the first branch is proved.** The second is the §1.3 screen: reliable in
the *positive* direction on these nine plants, but its negative direction is
outright false (constant `B` with varying drift is class III — §1.3), so a
constant `sv(B)` warrants a check with `ρ` or a per-state SDP, never a
conclusion. Proposition 2 would prove class II, but no plant here satisfies it.

**Class membership belongs to the pair (plant, box), never the plant alone** —
see §2.2. Shrinking a box can only raise the rate (§3.1, Remark), so plants move
*down* this list as the box shrinks and *up* as it grows.

---

# 2. Measured

## 2.1 All eleven classic envs

`λ = 0.3`, `r = 1.6`, `cm_dt = 1.0`, `q = 1` for the CARE screen; the exact test
uses one-sample SDPs at `w = [1e-3, 1e3]`, `ε = 0.1`, 6 states, bisection
tol 0.5%.

```
python scripts/feasibility_certificate.py --all --lbd 0.3 --verify -n 200
```

| env | class | Hautus margin | exact `λ*(x)` spread | `ρ` spread | `ν` | `w_lb` | `w_ub` | ‖K‖max |
|---|---|---|---|---|---|---|---|---|
| **pvtol** | **I** | 1.05e-19 | — | — | ∞ | — | — | — |
| **turtlebot** | **I** | 0.00e+00 | — | — | ∞ | — | — | — |
| car | **II** | 1.000 | **1.0000** | 1.0000 | 9.54 | 1.05e-1 | 9.53e-1 | 2.86 |
| quadrotor | **II** | 1.000 | **1.0000** | 1.0000 | 55.5 | 1.80e-2 | 5.15 | 7.27 |
| ball_and_beam | III | 3.22e-1 | 1.373 | 58.6 | 389 | 2.57e-3 | 55.2 | 26.4 |
| segway | III | 1.47e-1 | 1.755 | 2.20 | 759 | 1.32e-3 | 8.95 | 31.0 |
| two_link_arm | III | 1.24e-1 | 1.927 | 43.2 | 425 | 2.35e-3 | 19.8 | 29.6 |
| cartpole | III | 1.94e-1 | 2.092 | 7.26 | 573 | 1.74e-3 | 8.26e-1 | 27.1 |
| aircraft | III | 9.76e-2 | 2.400 | 3.26 | 2 196 | 4.55e-4 | 26.3 | 32.9 |
| tora | III | 9.71e-3 | 3.652 | 1 152 | 3.26e+5 | 3.07e-6 | 7.98e-1 | 448 |
| auv | III | 6.66e-3 | 5.184 | 5 618 | 8.57e+4 | 1.17e-5 | 3.71e-1 | 251 |

**Feasibility, guaranteed: 9 of 11**, each with the envelope shown and a verified
LMI margin `q·w_lb + 1/dt` (Proposition 5). pvtol and turtlebot are class I with
an uncontrollable mode at `s = 0` — turtlebot's margin is exactly zero because
`f ≡ 0` makes `A ≡ 0`; pvtol's `1e-19` is a numerically-zero nilpotent block,
matching the `A₂₂` with `eig = {0,0}` extracted independently in
`cvstem_feasibility_theory.md` §1.3. Both are class I **for this program**: both
plants are in fact controllable, and §4.2 measures how much of their dynamics the
drift Jacobian discards (pvtol 39×, turtlebot everything).

* **The `ρ` screen was right every time** on the class, and wildly wrong on
  magnitude (ball_and_beam: `ρ` 58.6 against `λ*` 1.37). Classify with it, never
  quantify — or use `λ_C` (§1.3), which is right on both.
* **auv has a state with `λ*(x) = 0`** at this envelope, which is why its `ν` is
  `8.6e4` and its `w_lb` `1.2e-5`.
* **The classes are stable in λ.** Re-running at `λ = 1.0` reproduces every
  assignment; only `ν` moves, upward (car 9.5 → 31.3, cartpole 573 → 4 452,
  tora 3.3e5 → 7.0e6), as Corollary 4.2 requires.

## 2.2 Class membership belongs to (plant, box)

The car's Hautus matrix at `s = 0`, after the yaw rotation, has mutually
orthogonal rows of norms `1, v, 1, 1`, so `σ = min(1, v)`. Against
`ρ(v) = λ_max(P(v))`:

| `v` | 0.01 | 0.1 | 1 | 1.5 | 2 | 10 | 1000 |
|---|---|---|---|---|---|---|---|
| `σ = min(1,v)` | 0.01 | 0.1 | 1 | 1 | 1 | 1 | 1 |
| `ρ(v)` | 60542.7 | 608.75 | 9.536 | 9.536 | 9.536 | 9.536 | 39.15 |

`ρ` rises 99.5× per decade of `v` below 1 — Corollary 4.2's `1/σ²` branch, to
0.5% — and is flat wherever `σ` is. So the shipped box `v ∈ [1,2]` is class II,
any box reaching `v < 1` is class III, and `v = 0` (where `f ≡ 0`, the turtlebot)
is class I. One plant, three classes, selected by the box. (`ρ` climbs again at
`v = 1000` with `σ` unmoved: Corollary 4.2 is a lower bound driven by weak
authority, and does not capture growth driven by `‖A‖`.)

## 2.3 Why the obvious proof of the class-III screen fails

The natural route is: orthogonal congruence preserves every orthogonal invariant
of the pair `(A(x), B(x)B(x)ᵀ)`, so if any invariant varies, no gauge exists,
Proposition 2 cannot apply, and the plant "must" be class III. **The second step
is valid; the conclusion is not.** Spread of each invariant over 200 samples:

| env | class | `sv(B)` | `spec(A)` | `tr A²` | `tr(A BBᵀAᵀ)` | `‖A‖_F` | gauge possible? |
|---|---|---|---|---|---|---|---|
| car | **II** | 1.0000 | 1.0000 | 1.0000 | **2.4838** | **1.5760** | **no** |
| quadrotor | **II** | 1.0000 | 1.0000 | 1.0000 | **21.897** | **4.4909** | **no** |
| cartpole | III | 2.2123 | 4.7488 | 22.572 | 4.9566 | 5.5458 | no |
| segway | III | 1.1394 | 1.8332 | 3.4674 | 1.5440 | 1.3771 | no |
| aircraft | III | 16.304 | 13.039 | 15.695 | 1173.98 | 12.979 | no |
| auv | III | 41.433 | 9.0984 | 2.4549 | 1225.15 | 1.1412 | no |
| tora | III | 1.0623 | 132.97 | 1.6824 | 1.5667 | 1.1261 | no |

`tr(A BBᵀAᵀ)` and `‖A‖_F` vary on the car and the quadrotor, so **no orthogonal
gauge exists for either** — yet both have `λ*(x)` constant to six digits. So
"gauge impossible ⟹ class III" is **false**, and the invariant test has no
discriminating power: it fires on all nine plants.

Two consequences, both corrections to earlier drafts of this document:

1. **Proposition 2 does not certify the car.** The gauge `T(θ) =
   blockdiag(R(θ)ᵀ, I₂)` removes `θ` from `A`, leaving
   `[[0₂ₓ₂, [[0,1],[v,0]]],[0,0]]` — still `v`-dependent, so `A₀` is not
   constant on a box where `v` varies. Proposition 2's hypothesis fails, and the
   car's `v`-invariance has a different and currently unproven cause. (The
   non-orthogonal `S = diag(1, 1/v, 1, 1)` does remove `v`, but a general
   similarity moves the envelope, so it proves nothing about `λ*`.)
2. **Proposition 2 is sufficient, not necessary, for class II** — the car is the
   counterexample. That is exactly why the class-III screen cannot be proved by
   negating it.

---

# 3. Supporting results

| result | what breaks without it |
|---|---|
| **Proposition 3** (decoupling) | Definition 1 and Propositions 1–2 are statements about *one state*. Without decoupling they say nothing about `P(S)`, the program actually solved. This is what turns "`λ*` is constant" into "**no subset** can raise λ". |
| **Proposition 4** (quantitative Hautus) | Class I itself is the Hautus lemma, cited. Prop 4 earns its place only for Corollary 4.2 — the `ν ≳ 1/σ²` price of a *nearly* uncontrollable mode, which a yes/no rank test cannot give. |
| **Proposition 5** (CARE) | `ρ(x)` is not known to be finite or computable, so the cheap class-III evidence test does not exist — and this *is* the feasibility guarantee. |
| Proposition 6 | one *mechanism* behind class III. Not a sufficient condition — see §3.5. |

## 3.1 Proposition 3 — the joint program decouples

**Proposition 3.** *At fixed `(λ, ε, dt, r, w_lb, w_ub)` with `w_lb > 0`, `P(S)`
is feasible **iff** `P({xₖ})` is feasible for every `k`; the shared scalars may
be fixed a priori at `ν = 1/w_lb`, `χ = w_ub/w_lb`.*

*Proof.* First, the feasible set is monotone in `(ν, χ)`: `ν` enters (C1) only
through `−ν(2/r)BBᵀ` with `BBᵀ ⪰ 0`, so raising `ν` moves the left side down in
the PSD order, and `χ` enters (C2) only as an upper bound, which raising relaxes.

(⟹) Restricting a feasible point to one index satisfies that index's
constraints. (⟸) Let `(W̄ₖ, νₖ, χₖ)` be feasible for `{xₖ}`. Each
`χₖ ≤ νₖ·w_ub ≤ w_ub/w_lb`, so by monotonicity every `W̄ₖ` stays feasible at the
common `ν = 1/w_lb`, `χ = w_ub/w_lb`. That collection is a feasible point of
`P(S)`. ∎

**Corollary 3.1.** `λ*(S) = min_k λ*(xₖ)`.

*Proof.* Feasibility is downward-closed in `λ`, since `λ` enters (C1) only via
`+2λW̄ₖ` with `W̄ₖ ⪰ I ≻ 0`. By Proposition 3 the feasible `λ`-set of `P(S)` is the
intersection of the per-state ones, and the supremum of an intersection of
downward-closed intervals is the min of their suprema. `S` finite ⟹ attained. ∎

**Remark (box monotonicity).** For `S' ⊆ S`, `λ*(S') ≥ λ*(S)` — a min over a
smaller set cannot be smaller. This is the whole content of "class membership
belongs to (plant, box)".

**There is no shared-`ν` penalty.** The coupling between states is exactly "the
worst state's demand", never an extra cost on top. This corrects
`subset_lambda_procedure.md` §8c, which attributed segway's joint/pointwise gap
to "the cost of one shared ν/χ": that gap was measured across *different draws*
(N=100 joint against a separate N=40 pointwise), which Corollary 3.1 does not
govern. On one draw, both sides at `w=[1e-3,1e3]`, `ε=0.1`, N=8, tol 0.5%:

| plant | `λ*(S)` joint | `min_k λ*(xₖ)` | gap |
|---|---|---|---|
| cartpole | 0.7895 | 0.7895 | 0.00% |
| car | 5.0203 | 5.0203 | 0.00% |
| segway | 0.7456 | 0.7505 | 0.65% (= bisection tol + solver) |

## 3.2 Proposition 4 — one inequality per mode

**Proposition 4.** *Let `P({x})` be feasible with `(W̄, ν, χ)`, let `s ∈ C`, let
`w ∈ Cⁿ` be a unit vector, and set `δ* := w*(A − sI)`. If `Re s + λ > 0` then*

```
2(Re s + λ) + ε  ≤  2‖δ‖·χ  +  (2ν/r)·w*BBᵀw                            (★)
```

*Proof.* Take the Hermitian form of (C1) against `w`; write `β := w*W̄w`, real and
`≥ 1` since `W̄ ⪰ I`, `‖w‖ = 1`:

```
(β − 1)/dt + w*(AW̄ + W̄Aᵀ)w + 2λβ − ν(2/r)·w*BBᵀw  ≤  −ε
```

From `w*A = s·w* + δ*` we get `w*AW̄w = sβ + δ*W̄w`, and `w*W̄Aᵀw` is its
conjugate, so

```
w*(AW̄ + W̄Aᵀ)w = 2Re(s)β + 2Re(δ*W̄w) ≥ 2Re(s)β − 2‖δ‖·‖W̄‖ ≥ 2Re(s)β − 2‖δ‖·χ
```

by `‖W̄‖ ≤ χ`. Collect the `β` terms with `c := 1/dt + 2Re(s) + 2λ`:

```
β·c  ≤  1/dt + 2‖δ‖·χ + (2ν/r)·w*BBᵀw − ε
```

`Re s + λ > 0` gives `c > 0`, so `β ≥ 1` implies `c ≤ βc`. Substituting `c` and
cancelling `1/dt` — the `dt` terms drop out exactly — gives (★). ∎

**Corollary 4.1 (the Hautus test, recovered).** *If `σ_min([A(x) − sI, B(x)]) = 0`
for some `Re s ≥ −λ`, then `P({x})` is infeasible for every
`(ν, χ, r, dt, w_lb, w_ub)` whenever `ε > 0`.*

Stated only to confirm (★) degrades to the standard test. Take `w` the left
singular vector for the zero singular value, so `w*[A − sI, B] = 0`, i.e. `δ = 0`
and `Bᵀw = 0`; then (★) reads `2(Re s + λ) + ε ≤ 0`, contradicting `Re s ≥ −λ`
with `ε > 0`.

**Corollary 4.2 (the price of a weak mode).** *With `σ := σ_min([A − sI, B])` at
some `Re s ≥ −λ`,*

```
ν  ≥  [2(Re s + λ) + ε] / (2σ·(w_ub + σ/r))        and, at fixed χ,
ν  ≥  r·[2(Re s + λ) + ε] / (2σ²)
```

*Proof.* Apply (★) at the left singular vector, where `‖δ‖ ≤ σ` and
`w*BBᵀw ≤ σ²`, then substitute `χ ≤ ν·w_ub` (first) or hold `χ` fixed (second). ∎

Since Proposition 3 saturates `ν = 1/w_lb` and `ν` is shared, a small `σ` at
**one** state raises the gain bound `‖K‖ ≤ ‖B‖/(r·w_lb)` at **every** state.
This is the quantitative content the Hautus lemma cannot supply.

## 3.3 Proposition 5 — the feasibility guarantee

**Proposition 5.** *Fix `λ, r, dt > 0, q > 0`, `λ' = λ + 1/(2dt)`. If
`(A(x) + λ'I, B(x))` is stabilizable for every `x ∈ S`, then the CARE*

```
(A + λ'I)ᵀP + P(A + λ'I) − (2/r)·P B Bᵀ P + qI = 0
```

*has a unique stabilizing `P(x) ≻ 0`, and*

```
W(x) = P(x)⁻¹,  ν = max_x λ_max(P(x)),  w_lb = 1/ν,
w_ub = 1/min_x λ_min(P(x)),  χ = ν·w_ub,  W̄(x) = ν·W(x)
```

*is a feasible point of `P(S)` for every `ε ≤ q·w_lb + 1/dt`.*

*Proof.* **(i)** `(A + λ'I, B)` stabilizable and `(A + λ'I, √q·I)` observable
(immediate, `q > 0`) give a unique stabilizing `P ≻ 0` by standard CARE theory.

**(ii)** Multiply the CARE left and right by `W = P⁻¹`:

```
(A + λ'I)W + W(A + λ'I)ᵀ − (2/r)BBᵀ = −qW²
```

**(iii) Envelope.** `λ_min(W) = 1/λ_max(P) ≥ 1/ν`, so `W̄ = νW ⪰ I`; and
`λ_max(W) = 1/λ_min(P) ≤ w_ub`, so `W̄ ⪯ χI`. (C3) holds with equality.

**(iv) LMI.** Split the proxy term `(W̄ − I)/dt = νW/dt − I/dt` and absorb `νW/dt`
into the rate:

```
(νW − I)/dt + ν[AW + WAᵀ + 2λW] − ν(2/r)BBᵀ
  = ν[(A + λ'I)W + W(A + λ'I)ᵀ − (2/r)BBᵀ] − I/dt
  = ν(−qW²) − I/dt
  ⪯ −(ν·q·λ_min(W)² + 1/dt)·I  ⪯  −(q·w_lb + 1/dt)·I    ∎
```

The envelope is an **output** of the plant, not a knob tuned until something
certifies. Step (iv) also isolates what Tsukamoto's `(W̄ − I)/dt` proxy does: the
`+νW/dt` half is a **rate penalty** (build the certificate at `λ'`, not `λ`), the
`−I/dt` half is an **exact free margin**, independent of plant and of `ν` — which
is why every verified residual comes out at `−1` for `dt = 1`.

**Definition 2.** `ρ(x) := λ_max(P(x))`, the metric scale state `x` demands under
*this* metric choice, with `ν = max_S ρ`. `ρ` upper-bounds what `x` truly demands,
because the CARE picks one metric rather than the best one — hence the §1.3 `ρ`
test is evidence, not proof.

## 3.4 Proposition 6 — a mechanism for class III, not a condition

**Proposition 6.** *If `A(x₁) = A(x₂)` and `B₁B₁ᵀ ⪰ B₂B₂ᵀ`, then every certificate
feasible at `x₂` is feasible at `x₁` with the same `(ν, χ)`; hence
`λ*(x₁) ≥ λ*(x₂)` and `ρ(x₁) ≤ ρ(x₂)`.*

*Proof.* The left side of (C1) at `x₁` equals that at `x₂` minus
`ν(2/r)(B₁B₁ᵀ − B₂B₂ᵀ) ⪯ 0`, so it is `⪯ −εI` whenever `x₂`'s is; (C2)/(C3) do
not involve `B`. For `ρ`, the stabilizing CARE solution is monotone
non-increasing in `BR⁻¹Bᵀ`, so `P₁ ⪯ P₂`. ∎

**Why this is not a condition for class III.** The hypothesis `A(x₁) = A(x₂)`
almost never holds between two states of a real plant, and authority is not even
the dominant mechanism. Spearman rank correlation between the per-state Hautus
margin `σ(x)` and `ρ(x)`, 200 samples:

| plant | `σ` spread | `ρ` spread | corr(`σ`, `ρ`) | authority-driven? |
|---|---|---|---|---|
| auv | 86.6 | 5 618 | **−1.000** | yes |
| tora | 38.1 | 1 152 | **−0.993** | yes |
| two_link_arm | 13.4 | 43.2 | −0.747 | mostly |
| cartpole | 2.88 | 7.26 | −0.722 | mostly |
| ball_and_beam | 7.92 | 58.6 | −0.582 | partly |
| **segway** | **1.06** | 2.20 | **+0.169** | **no** |
| **aircraft** | 1.87 | 3.26 | **−0.121** | **no** |
| car / quadrotor | 1.000 | 1.000 | n/a (both constant) | n/a — class II |

Segway and aircraft are class III with essentially **no** authority variation:
their `ρ` varies through `A(x)` instead.

**Remark 6.1 (why `sv(B)` predicts, and where the argument stops).** In (C1) the
metric enters the drift term `AW̄ + W̄Aᵀ` linearly and the control term
`−ν(2/r)BBᵀ` not at all. `W̄` is a free per-state variable, so it can absorb
`A(x)` up to the invariant content Corollary 4.1 isolates, but it cannot
manufacture authority. **The stop:** this argument is about the *optimal* `W̄`.
`ρ(x)` comes from one fixed metric, which does not absorb `A` — which is exactly
why segway and aircraft show `ρ` variation with `σ` flat, and why the §1.3 screen
remains empirical.

---

# 4. Generality: what the propositions cover

**The propositions are general; the program they characterize is not.** The
distinction matters for any claim about control-affine systems at large.

## 4.1 The proofs assume nothing beyond the LMI

Propositions 1–6 use only (i) differentiability of `f` and continuity of `B`,
(ii) the algebraic form of (C1)–(C3), and (iii) standard CARE theory under the
stabilizability hypothesis stated in Proposition 5. No structure of `f` or `B` is
used anywhere: no polynomial/SOS assumption, no feedback linearizability, no
bound on `n` or `m`, no normality of `A`. In that sense they hold for every
control-affine `ẋ = f(x) + B(x)u` on a compact box.

## 4.2 But the program uses the DRIFT Jacobian, not the generalized one

For a control-affine plant the differential dynamics is

```
δẋ  =  [ ∂f/∂x  +  Σᵢ uᵢ ∂bᵢ/∂x ] δx  +  B(x) δu
```

so the object a contraction certificate should see is the **generalized Jacobian**
`A(x,u) = ∂f/∂x + Σᵢ uᵢ ∂bᵢ/∂x`. This document — following
`ncm_synthesis.drift_jacobians`, which is what `cvstem_joint` is fed — uses
`A(x) = ∂f/∂x` **alone**. The neglected term `Σᵢ uᵢ ∂bᵢ/∂x` vanishes identically
**iff `B` is constant**.

Measured at the control-box vertices, 40 samples per env, spectral norms:

| env | class | `‖∂f/∂x‖` | `‖Σuᵢ∂bᵢ/∂x‖` | ratio | neglected term |
|---|---|---|---|---|---|
| car | **II** | 1.498 | **0** | 0.000 | **exactly zero** (`B` constant) |
| quadrotor | **II** | 12.84 | **0** | 0.000 | **exactly zero** (`B` constant) |
| segway | III | 14.28 | 3.22 | 0.23 | significant |
| auv | III | 1.008 | 0.586 | 0.58 | significant |
| aircraft | III | 21.09 | 20.51 | 0.97 | comparable |
| cartpole | III | 9.194 | 9.85 | 1.07 | **dominates** |
| tora | III | 1.207 | 2.625 | 2.18 | **dominates** |
| ball_and_beam | III | 11.20 | 140.5 | 12.5 | **dominates** |
| two_link_arm | III | 20.72 | 326.8 | 15.8 | **dominates** |
| pvtol | I | 1.000 | 39.24 | 39.2 | **dominates** |
| turtlebot | I | **0** | 0.44 | ∞ | **is the entire dynamics** |

The pattern is exact and unflattering: **the drift-only model is faithful precisely
on the class-II plants and wrong precisely on the class-III ones**, because
"class III" is diagnosed by `B` varying and `B` varying is exactly what creates
the neglected term. The class where the analysis is exact is the class where
nothing happens.

## 4.3 Consequences, stated plainly

* **Class I is a statement about this program, not about the plant.** pvtol is
  differentially flat and turtlebot is a standard controllable nonholonomic
  system; both are "class I" only because a frozen drift Jacobian cannot see the
  Lie bracket that actually steers them. For turtlebot `f ≡ 0`, so `A ≡ 0` and the
  program is looking at *none* of the dynamics. Read class I as **"not
  λ-stabilizable in the pointwise drift-Jacobian sense"**, never as "not
  contractible".
* **The class II/III boundary is only certified for constant-`B` plants.** For the
  seven class-III envs the certificate is computed on a model whose neglected term
  is between 0.23× and 15.8× the term retained.
* **What generalizes cleanly.** Every proposition carries over verbatim with `x`
  replaced by the pair `(x, u)` and `A(x)` by `A(x,u)`, since none of the proofs
  touch where `A` came from. The per-state rate becomes
  `λ*(x) = inf_{u ∈ U} λ*(x,u)`, Proposition 3's decoupling still holds over the
  enlarged sample set, and Proposition 5's CARE is solved per `(x,u)`. Doing this
  is the natural next step and is **not done here**; `solve_cm_metric` elsewhere in
  this repo already evaluates `A(x,u)` at `u`-box vertices, so the machinery
  exists.

So: general for control-affine systems **as a theory of the drift-Jacobian
program**, and a faithful theory of the *plant* only where `B` is constant.

---

# 5. What is not claimed

* **Sampled, not box-wide.** Every statement is over the finite `S`. Extending to
  all of `X` needs the covering argument of `subset_lambda_procedure.md` §7 —
  Proposition 5's margin `q·w_lb + 1/dt` is what feeds it, but the Lipschitz
  constant is not computed here.
* **The class-III screen is empirical, and one-directional.** It holds on nine
  plants going from varying `sv(B)` to class III; the converse is FALSE (§1.3, a
  1-D plant with constant `B` and varying drift). §2.3 shows the natural proof by
  negating Proposition 2 is also unavailable, since Proposition 2 is not
  necessary for class II.
* **Proposition 2 has no instance here.** It certifies class II in principle; no
  plant in this repo satisfies its hypothesis, the car included.
* **Feasibility, not performance.** `λ*` is a certified rate, not a measured AUC.
  Proposition 5 certifies with a *particular* metric, so its `ν` is an upper
  bound on what the SDP would find — 20× on the car.
* **Subset claims still owe an invariance argument.** `λ*(S')` certifies
  contraction *while the trajectory stays in `S'`*. The Remark in §3.1 says
  shrinking the box raises the rate; it says nothing about staying inside it.
* **The related-work claim rests on a non-systematic search.** Four queries, not
  a review. "Prior work prescribes the rate, we diagnose it" is the defensible
  form; a novelty claim needs a proper search of the incremental-stability and
  CCM literature first. Note also that "classify" is accurate while
  "characterize" would not be: Proposition 2 has no instance in these envs and
  the `sv(B)` screen is one-directional.
* **The control box is out of scope by construction.** A plant needing `ν = 3e5`
  (tora) is contraction-feasible in the same sense as one needing `9.5` (car).
  Whether the gain fits an actuator is answered by `find_uniform_lambda.py`, and
  is deliberately not allowed to influence any class.

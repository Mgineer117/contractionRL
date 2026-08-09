# A taxonomy of dynamics for state-dependent contraction rate

**§1 is the taxonomy: one definition of the three classes, plus checkable
sufficient conditions for membership.** §2 measures them. §3 proves them and says
which supporting results are load-bearing and why. §4 is what is not claimed.

**Scope.** Contraction feasibility only. The control box plays no part: no plant
is called infeasible because its certified gain is large. `‖K‖` is reported so
the cost stays visible, never tested.

Implementation: `scripts/feasibility_certificate.py`. Program under test:
`cvstem_joint` in `agents/skrl/ncm_synthesis.py`.

**Standard results used, not reproved.** The Hautus (Popov–Belevitch–Hautus)
lemma for controllability/stabilizability; the LQR `α`-shift for a prescribed
decay rate (Anderson & Moore); existence and uniqueness of the stabilizing CARE
solution under stabilizability plus observability. Class III is exactly
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

# 1. The taxonomy

The classes are **defined** by what `λ*(·)` does on the box; the two structural
conditions below are *sufficient conditions for membership*, which is a different
kind of statement and is labelled as such.

**Definition 1 (the three classes).** *A plant on a box `X`, at fixed
`(λ, ε, dt, r, w_lb, w_ub)`, is*

```
class III  if  P({x}) is infeasible at some x ∈ X for every ν, χ, r, dt, envelope
class I    if  it is not class III and λ*(x) is constant on X
class II   if  it is not class III and λ*(x) is non-constant on X
```

*The three are mutually exclusive and exhaustive by construction.*

Everything else in §1 is a **checkable condition implying one of these**, because
the definitions themselves are not directly checkable: `λ*(x)` costs an SDP
bisection per state, and class III quantifies over every envelope.

## Sufficient condition for class III — the Hautus test

> **If** `(A(x) + λI, B(x))` is not stabilizable for some `x ∈ X` — equivalently,
> by the **Hautus (PBH) lemma**, if
> ```
> rank[ A(x) − sI ,  B(x) ] < n     for some s with Re s ≥ −λ
> ```
> **then** the plant is **class III** on `X`: `P({x})` is infeasible for *every*
> `ν, χ, r, dt, w_lb, w_ub`.

**This is standard and is cited, not proved here.** `(C1)` is a state-feedback
synthesis LMI in disguise; no such condition is satisfiable for an unstabilizable
pair, and shifting `A ↦ A + λI` to demand a prescribed decay rate `λ` rather than
mere stability is the classical `α`-shift of LQR (Anderson & Moore). "Class III"
is therefore just a name for *"not λ-stabilizable somewhere on the box"*.

What is worth saying beyond the citation is only how to *evaluate* it:

* **The test is finite.** Uncontrollable modes sit only at eigenvalues of `A`, so
  testing `s ∈ spec(A) ∩ {Re s ≥ −λ}` misses nothing.
* **Use `σ_min`, not `rank`.** Numerically the useful form is
  `σ_min([A(x) − sI, B(x)]) = 0`, since rank is not a continuous function of the
  data and a tolerance on it is arbitrary, while `σ_min` degrades smoothly and
  doubles as the quantitative margin Corollary 2.2 needs.
* **Use the SVD, never eigenvectors.** A defective `A` (pvtol's uncontrollable
  block is nilpotent) returns a near-parallel eigenbasis, and any modal authority
  read off it is meaningless — measured, this reported `1e-2` where the true
  margin is `1e-19`.

The only non-standard part of §3.2 is the *quantitative* statement
(Corollary 2.2): what a nearly-uncontrollable mode costs under **this program's
envelope**, rather than whether it is fatal.

## Sufficient condition for class I — an orthogonal gauge

> **If** there is a map `T : X → O(n)` into the **orthogonal** group with
> ```
> T(x) A(x) T(x)ᵀ = A₀     and     T(x) B(x)B(x)ᵀ T(x)ᵀ = B₀B₀ᵀ      for all x ∈ X
> ```
> **then** the plant is **class I** on `X`: `λ*(x)` is the same at every state,
> and **no subset of `X` certifies a faster rate.**

The dynamics is one `(A₀, B₀)` seen from a rotating frame. Every state is exactly
as hard as every other, so there is nothing for a subset to remove.

*Checking it* means exhibiting `T`, which is analytic work, not a computation.
The car: `T(θ) = blockdiag(R(θ)ᵀ, I₂)`. Writing `A = [[0₂ₓ₂, M],[0, 0]]` with
`M = R(θ)·[[0, 1],[v, 0]]`, the congruence gives `TAT ᵀ = [[0₂ₓ₂, [[0,1],[v,0]]],
[0,0]]`, free of `θ`; and `B` is supported on the `(yaw, vel)` block where `T` is
the identity, so `TB = B`.

Proved in §3.4 (Proposition 4).

## Deciding class II — no structural condition, only tests

There is **no** structural sufficient condition for class II here, and that is a
finding rather than an omission: §3.5 shows `B(x)` variation explains it on some
of this repo's plants and not on others. So class II is settled by Definition 1
directly — `λ*(x₁) ≠ λ*(x₂)` — and the practical question is only how to
establish that. Three ways, in increasing cost and rigour:

| test | cost | status |
|---|---|---|
| **Screen.** `ρ(x) = λ_max(P(x))` from the CARE of Proposition 3 takes two different values. | 1 CARE solve/sample, no SDP | **Evidence only.** `ρ` is an *upper* bound on what a state demands, so `ρ` varying does not prove `λ*` does. Agreed with the exact test on 9 of 9 plants here. |
| **Certificate.** Corollary 2.2's floor at `x₂` exceeds `ρ(x₁)`. | same, plus one SVD | **Rigorous** when it fires. Fired on 1 of 9 (auv). |
| **Exact.** `λ*(x)` by one-sample SDP at two states. | 2 SDP bisections | **Rigorous, always decisive.** |

Mechanisms are discussed in §3.5; the honest summary is that weak control
authority is *a* cause, not *the* cause.

## The decision procedure

```
σ_min([A−sI, B]) = 0 somewhere, Re s ≥ −λ ?  ──yes──>  class III  (infeasible, any envelope)
                    │no
an orthogonal gauge T reduces (A, BBᵀ) to a constant pair ?  ──yes──>  class I  (subsets buy nothing)
                    │no
λ*(x) varies ?  ──yes──>  class II  (subsets buy rate; the worst state governs)
```

**Class membership belongs to the pair (plant, box), never the plant alone** —
see §2's car row. Shrinking a box can only raise the rate (§3.1, Remark), so
plants move *up* this list as the box shrinks and *down* as it grows.

---

# 2. Measured

`λ = 0.3`, `r = 1.6`, `cm_dt = 1.0`, `q = 1` for the CARE screen; the exact test
uses one-sample SDPs at `w = [1e-3, 1e3]`, `ε = 0.1`, 6 states, bisection
tol 0.5%.

```
python scripts/feasibility_certificate.py --all --lbd 0.3 --verify -n 200
```

| env | class | Hautus margin (class III) | exact `λ*(x)` spread | `ρ` spread (screen) | `ν` | `w_lb` | `w_ub` | ‖K‖max |
|---|---|---|---|---|---|---|---|---|
| car | **I** | 1.000 | **1.0000** | 1.0000 | 9.54 | 1.05e-1 | 9.53e-1 | 2.86 |
| quadrotor | **I** | 1.000 | **1.0000** | 1.0000 | 55.5 | 1.80e-2 | 5.15 | 7.27 |
| ball_and_beam | II | 3.22e-1 | 1.373 | 58.6 | 389 | 2.57e-3 | 55.2 | 26.4 |
| segway | II | 1.47e-1 | 1.755 | 2.20 | 759 | 1.32e-3 | 8.95 | 31.0 |
| two_link_arm | II | 1.24e-1 | 1.927 | 43.2 | 425 | 2.35e-3 | 19.8 | 29.6 |
| cartpole | II | 1.94e-1 | 2.092 | 7.26 | 573 | 1.74e-3 | 8.26e-1 | 27.1 |
| aircraft | II | 9.76e-2 | 2.400 | 3.26 | 2 196 | 4.55e-4 | 26.3 | 32.9 |
| tora | II | 9.71e-3 | 3.652 | 1 152 | 3.26e+5 | 3.07e-6 | 7.98e-1 | 448 |
| auv | II | 6.66e-3 | 5.184 | 5 618 | 8.57e+4 | 1.17e-5 | 3.71e-1 | 251 |
| **pvtol** | **III** | 1.05e-19 | — | — | ∞ | — | — | — |
| **turtlebot** | **III** | 0.00e+00 | — | — | ∞ | — | — | — |

**Feasibility, guaranteed: 9 of 11**, each with the envelope shown and a verified
LMI margin `q·w_lb + 1/dt` (Proposition 3). pvtol and turtlebot are class III with an
uncontrollable mode at `s = 0` — turtlebot's margin is exactly zero because
`f ≡ 0` makes `A ≡ 0`; pvtol's `1e-19` is a numerically-zero nilpotent block,
matching the `A₂₂` with `eig = {0,0}` extracted independently in
`cvstem_feasibility_theory.md` §1.3.

Notes on the table:

* **The cheap screen was right every time.** `ρ` spread and the exact `λ*(x)`
  spread agree on the class for all 9 feasible plants, though they disagree
  wildly on *magnitude* (ball_and_beam: `ρ` 58.6 against `λ*` 1.37). Use `ρ` to
  classify, never to quantify.
* **auv has a state with `λ*(x) = 0`** at this envelope — no positive rate
  certifies there — which is why its `ν` is `8.6e4` and its `w_lb` `1.2e-5`.
* **The classes are stable in λ.** Re-running the screen at `λ = 1.0` reproduces
  every assignment; only `ν` moves, upward (car 9.5 → 31.3, cartpole 573 → 4 452,
  tora 3.3e5 → 7.0e6), as Corollary 2.2 requires.
* **Hautus margin orders the whole table**: class I at 1.0, class II from `7e-3`
  to `3e-1`, class III at 0.

**The car is class I only because its box excludes `v ≈ 0`.** Its Hautus matrix
at `s = 0` has mutually orthogonal rows of norms `1, v, 1, 1`, so
`σ = min(1, v)`, and:

| `v` | 0.01 | 0.1 | 1 | 1.5 | 2 | 10 | 1000 |
|---|---|---|---|---|---|---|---|
| `σ = min(1,v)` | 0.01 | 0.1 | 1 | 1 | 1 | 1 | 1 |
| `ρ(v)` | 60542.7 | 608.75 | 9.536 | 9.536 | 9.536 | 9.536 | 39.15 |

`ρ` rises 99.5× per decade of `v` below 1 — Corollary 2.2's `1/σ²` branch, to
0.5% — and is exactly flat wherever `σ` is. So the shipped box `v ∈ [1,2]` is
class I, any box reaching `v < 1` is class II, and `v = 0` (where `f ≡ 0`, the
turtlebot) is class III. One plant, three classes, selected by the box. (`ρ`
climbs again at `v = 1000` with `σ` unmoved: Corollary 2.2 is a lower bound
driven by weak authority and does not capture growth driven by `‖A‖`.)

**Reconciliation with `find_uniform_lambda`'s "segway INFEASIBLE".** At `λ = 0.3`
segway needs `w_lb = 1.32e-3`, which fits the shipped `[1e-3, 1e3]`; at `λ = 1.0`
it needs `3.5e-4`, which does not. Segway is contraction-feasible at the shipped
envelope for small enough λ; the INFEASIBLE verdict comes from the *actuator*
branch driving `r` up until `ν` hits its cap, exactly as
`subset_lambda_procedure.md` §8d describes. Different questions, no conflict.

---

# 3. Why these are the criteria

Only three supporting results carry weight, and each answers a specific
objection to §1. Stated plainly, so the ones that are scaffolding can be skipped:

| result | what breaks without it |
|---|---|
| **Proposition 1** (decoupling) | The class definitions are statements about *one state*. Without decoupling they say nothing about `P(S)`, which is the program actually solved. This is what turns "`ρ` is constant" into "**no subset** can raise λ". |
| **Proposition 2** (quantitative Hautus) | Class III itself is the **Hautus lemma**, cited not proved. Prop 2 earns its place only for Corollary 2.2 — the `ν ≳ 1/σ²` price of a *nearly* uncontrollable mode, which a yes/no rank test cannot give and which the class-II certificate test uses. |
| **Proposition 3** (CARE) | `ρ(x)` is not known to be finite or computable, so the cheap class-II screen does not exist — and this *is* the feasibility guarantee. |
| Proposition 4 | proves the class-I sufficient condition. |
| Proposition 5 | one *mechanism* behind class II. Not a sufficient condition — see §3.5. |

## 3.1 Proposition 1 — the joint program decouples

**Proposition 1.** *At fixed `(λ, ε, dt, r, w_lb, w_ub)` with `w_lb > 0`, `P(S)` is
feasible **iff** `P({xₖ})` is feasible for every `k`; the shared scalars may be
fixed a priori at `ν = 1/w_lb`, `χ = w_ub/w_lb`.*

*Proof.* First, the feasible set is monotone in `(ν, χ)`: `ν` enters (C1) only
through `−ν(2/r)BBᵀ` with `BBᵀ ⪰ 0`, so raising `ν` moves the left side down in
the PSD order, and `χ` enters (C2) only as an upper bound, which raising relaxes.

(⟹) Restricting a feasible point to one index satisfies that index's
constraints. (⟸) Let `(W̄ₖ, νₖ, χₖ)` be feasible for `{xₖ}`. Each
`χₖ ≤ νₖ·w_ub ≤ w_ub/w_lb`, so by monotonicity every `W̄ₖ` stays feasible at the
common `ν = 1/w_lb`, `χ = w_ub/w_lb`. That collection is a feasible point of
`P(S)`. ∎

**Corollary 1.1.** `λ*(S) = min_k λ*(xₖ)`.

*Proof.* Feasibility is downward-closed in `λ`, since `λ` enters (C1) only via
`+2λW̄ₖ` with `W̄ₖ ⪰ I ≻ 0`. By Proposition 1 the feasible `λ`-set of `P(S)` is the
intersection of the per-state ones, and the supremum of an intersection of
downward-closed intervals is the min of their suprema. `S` finite ⟹ attained. ∎

**Remark (box monotonicity).** For `S' ⊆ S`, `λ*(S') ≥ λ*(S)` — immediate, a min
over a smaller set cannot be smaller. This is the whole content of "class
membership belongs to (plant, box)".

**There is no shared-`ν` penalty.** The coupling between states is exactly "the
worst state's demand", never an extra cost on top. This corrects
`subset_lambda_procedure.md` §8c, which attributed segway's joint/pointwise gap
to "the cost of one shared ν/χ": that gap was measured across *different draws*
(N=100 joint against a separate N=40 pointwise), which Corollary 1.1 does not
govern. On one draw, both sides at `w=[1e-3,1e3]`, `ε=0.1`, N=8, tol 0.5%:

| plant | `λ*(S)` joint | `min_k λ*(xₖ)` | gap |
|---|---|---|---|
| cartpole | 0.7895 | 0.7895 | 0.00% |
| car | 5.0203 | 5.0203 | 0.00% |
| segway | 0.7456 | 0.7505 | 0.65% (= bisection tol + solver) |

## 3.2 Proposition 2 — one inequality per mode

**The qualitative content of this section is the Hautus lemma and is cited, not
proved.** `(A(x) + λI, B(x))` unstabilizable at some `x` ⟹ class III, because
`(C1)` is a state-feedback synthesis LMI and no such LMI is satisfiable for an
unstabilizable pair; the `A ↦ A + λI` shift demanding a prescribed decay rate is
LQR's classical `α`-shift (Anderson & Moore). Proposition 2 exists for the
*quantitative* refinement in Corollary 2.2 — what a **nearly** uncontrollable
mode costs under this program's envelope — which the Hautus lemma, being a
yes/no rank test, does not give.

**Proposition 2.** *Let `P({x})` be feasible with `(W̄, ν, χ)`, let `s ∈ C`, let
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

**Corollary 2.1 (class III — the Hautus test, recovered).** *If
`σ_min([A(x) − sI, B(x)]) = 0` for some `Re s ≥ −λ`, then `P({x})` is infeasible
for every `(ν, χ, r, dt, w_lb, w_ub)` whenever `ε > 0`.*

This is the Hautus lemma; it is stated here only to confirm that (★) degrades to
it, so the quantitative bound and the standard qualitative test are consistent.
Take `w` the left singular vector for the zero singular value, so
`w*[A − sI, B] = 0`, i.e. `δ = 0` and `Bᵀw = 0`; then (★) reads
`2(Re s + λ) + ε ≤ 0`, contradicting `Re s ≥ −λ` with `ε > 0`.

**Corollary 2.2 (the price of a weak mode).** *With `σ := σ_min([A − sI, B])` at
some `Re s ≥ −λ`,*

```
ν  ≥  [2(Re s + λ) + ε] / (2σ·(w_ub + σ/r))        and, at fixed χ,
ν  ≥  r·[2(Re s + λ) + ε] / (2σ²)
```

*Proof.* Apply (★) at the left singular vector, where `‖δ‖ ≤ σ` and
`w*BBᵀw ≤ σ²`, then substitute `χ ≤ ν·w_ub` (first) or hold `χ` fixed (second). ∎

Since Proposition 1 saturates `ν = 1/w_lb` and `ν` is shared, a small `σ` at **one**
state raises the gain bound `‖K‖ ≤ ‖B‖/(r·w_lb)` at **every** state. That is the
precise sense in which one weak state is expensive everywhere.

## 3.3 Proposition 3 — the feasibility guarantee

**Proposition 3.** *Fix `λ, r, dt > 0, q > 0`, `λ' = λ + 1/(2dt)`. If
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
*this* metric choice. `ν = max_S ρ`. `ρ` is an upper bound on what `x` truly
demands, because the CARE picks one metric rather than the best one — hence
the class-II screen is evidence, not proof.

## 3.4 Proposition 4 — the class-I sufficient condition

**Proposition 4.** *If `T : X → O(n)` satisfies `T(x)A(x)T(x)ᵀ = A₀` and
`T(x)B(x)B(x)ᵀT(x)ᵀ = B₀B₀ᵀ`, then `λ*(x) ≡ λ*₀` and `ρ(x) ≡ λ_max(P₀)`.*

*Proof.* The congruence `W̄ ↦ T(x)ᵀW̄₀T(x)` maps feasible points of `P({x₀})` to
feasible points of `P({x})` bijectively: it leaves (C1) invariant because
`T` is a similarity on `A` and a congruence on `BBᵀ` simultaneously, and it
leaves (C2) invariant because **orthogonal** congruence preserves eigenvalues —
so `I ⪯ W̄ ⪯ χI` is preserved exactly. (C3) involves no state. Hence the feasible
`λ`-sets coincide and `λ*(x) = λ*₀`.

For `ρ`: substituting `P = TᵀP₀T` into the CARE at `(A(x), B(x))` reproduces the
`(A₀, B₀)` CARE conjugated by `T`, since `Tᵀ(qI)T = qI`. Similarity preserves the
stabilizing property, so by uniqueness `P(x) = T(x)ᵀP₀T(x)`, whose spectrum is
that of `P₀`. ∎

**Orthogonality is load-bearing.** A general similarity preserves (C1) but not
(C2), so it can move `ρ` and the envelope. The class-I condition is a statement about the
metric-normalised problem, not about `(A, B)` up to arbitrary coordinates.

## 3.5 Proposition 5 — a mechanism for class II, not a sufficient condition

**Proposition 5.** *If `A(x₁) = A(x₂)` and `B₁B₁ᵀ ⪰ B₂B₂ᵀ`, then every certificate
feasible at `x₂` is feasible at `x₁` with the same `(ν, χ)`; hence
`λ*(x₁) ≥ λ*(x₂)` and `ρ(x₁) ≤ ρ(x₂)`.*

*Proof.* The left side of (C1) at `x₁` equals that at `x₂` minus
`ν(2/r)(B₁B₁ᵀ − B₂B₂ᵀ) ⪯ 0`, so it is `⪯ −εI` whenever `x₂`'s is; (C2)/(C3) do
not involve `B`. For `ρ`, the stabilizing CARE solution is monotone
non-increasing in `BR⁻¹Bᵀ`, so `P₁ ⪯ P₂`. ∎

**Why this is not a sufficient condition for class II.** The hypothesis `A(x₁) = A(x₂)` almost never
holds between two states of a real plant, and the data says authority is not even
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
| car / quadrotor | 1.000 | 1.000 | n/a (both constant) | n/a — class I |

So segway and aircraft are class II with essentially **no** authority variation:
their `ρ` varies through `A(x)` instead. Remark 5.1 below is therefore a
statement about *feasibility of the optimal metric*, not about `ρ`.

**Remark 5.1 (why `σ(B)` predicts and `σ(A)` does not, and its limit).** In
(C1) the metric enters the drift term `AW̄ + W̄Aᵀ` linearly and the control term
`−ν(2/r)BBᵀ` not at all. `W̄` is a free per-state variable, so it can absorb
`A(x)` up to the invariant content Corollary 2.1 isolates, but it cannot
manufacture authority. **The limit:** this argument is about the *optimal* `W̄`.
`ρ(x)` comes from one fixed metric (the CARE's), which does not absorb `A`, which
is exactly why segway and aircraft show `ρ` variation with `σ` flat.

---

# 4. What is not claimed

* **Sampled, not box-wide.** Every statement is over the finite `S`. Extending to
  all of `X` needs the covering argument of `subset_lambda_procedure.md` §7 —
  Proposition 3's margin `q·w_lb + 1/dt` is what feeds it, but the Lipschitz constant
  is not computed here.
* **Feasibility, not performance.** `λ*` is a certified rate, not a measured AUC.
  Proposition 3 certifies with a *particular* metric, so its `ν` is an upper bound on
  what the SDP would find — 20× on the car (9.54 against a Corollary-2.2 floor of
  0.49).
* **Class II has no cheap rigorous test here.** The screen is evidence; the
  Corollary-2.2 certificate fired on 1 of 9 plants. Tightening that bound so it
  decides class II without an SDP is open.
* **Subset claims still owe an invariance argument.** `λ*(S')` certifies
  contraction *while the trajectory stays in `S'`*. The Remark in §3.1 says
  shrinking the box raises the rate; it says nothing about staying inside it.
* **The control box is out of scope by construction.** A plant needing `ν = 3e5`
  (tora) is contraction-feasible in the same sense as one needing `9.5` (car).
  Whether the gain fits an actuator is a separate question, answered by
  `find_uniform_lambda.py`, and deliberately not allowed to influence any class.

# A taxonomy of dynamics for state-dependent contraction rate

Six theorems, their proofs, and the class each classic plant lands in. Two
questions are answered separately and must not be conflated:

1. **Is the plant contraction-feasible at rate λ, guaranteed?** Theorems 2 and 3
   settle this constructively — a CARE solve per state, no SDP and no bisection.
2. **What makes the rate state-dependent?** Theorems 4 and 5 give the two
   mechanisms, and Theorem 1 is what makes a per-state answer meaningful at all.

**Scope.** Everything here is about the *contraction* LMI. The control box plays
no part: no plant is called infeasible because its certified gain is large. ‖K‖
is reported so the deployment cost is visible, never tested.

Implementation: `scripts/feasibility_certificate.py`. Synthesis under test:
`cvstem_joint` in `agents/skrl/ncm_synthesis.py`.

---

## 0. Setup

Plant `ẋ = f(x) + B(x)u` on a compact box `X ⊂ Rⁿ`, `A(x) = ∂f/∂x`. For a finite
sample set `S = {x₁..x_N} ⊂ X` and parameters `(λ, ε, dt, r, w_lb, w_ub)`, the
program `P(S)` asks for `W̄ₖ ⪰ 0` and shared scalars `ν, χ ≥ 0` with

```
(C1)  (W̄ₖ − I)/dt + AₖW̄ₖ + W̄ₖAₖᵀ + 2λW̄ₖ − ν(2/r)BₖBₖᵀ  ⪯  −εI      for all k
(C2)  I ⪯ W̄ₖ ⪯ χI                                                   for all k
(C3)  ν ≤ 1/w_lb,     χ ≤ ν·w_ub
```

Deployed: `Wₖ = W̄ₖ/ν`, `Mₖ = Wₖ⁻¹`, `Kₖ = (1/r)BₖᵀMₖ`. Write

```
λ*(x) = sup { λ : P({x}) is feasible },      λ*(S) = sup { λ : P(S) is feasible }
```

Throughout, `dt > 0`, `w_lb > 0`, `q > 0`, and `λ' := λ + 1/(2·dt)`.

---

## 1. The joint program decouples

Everything downstream is a statement about a single state, so the first thing to
establish is that a single state is the right unit of analysis.

**Lemma 1 (monotone saturation).** *The feasible set of (C1)–(C3) is monotone in
the shared scalars: if `(W̄, ν, χ)` is feasible and `ν ≤ ν' ≤ 1/w_lb`,
`χ ≤ χ' ≤ ν'·w_ub`, then `(W̄, ν', χ')` is feasible.*

*Proof.* `ν` enters (C1) only through `−ν(2/r)BBᵀ` with `BBᵀ ⪰ 0`, so raising `ν`
moves the left side down in the PSD order. `χ` enters (C2) only as an upper
bound, which raising `χ` relaxes. (C3) holds by hypothesis. ∎

**Theorem 1 (decoupling).** *At fixed `(λ, ε, dt, r, w_lb, w_ub)`, `P(S)` is
feasible **iff** `P({xₖ})` is feasible for every `k`. Moreover the shared scalars
may be fixed a priori at `ν = 1/w_lb`, `χ = w_ub/w_lb`, with no loss.*

*Proof.* (⟹) Restricting a feasible point to one index satisfies that index's
constraints. (⟸) Let `(W̄ₖ, νₖ, χₖ)` be feasible for `{xₖ}`. Each `χₖ ≤ νₖ·w_ub ≤
w_ub/w_lb`, so Lemma 1 applies with `ν' = 1/w_lb`, `χ' = w_ub/w_lb`, and every
`W̄ₖ` remains feasible at those common values. That collection is a feasible point
of `P(S)`. ∎

**Corollary 1.1.** `λ*(S) = min_{k} λ*(xₖ)`.

*Proof.* Feasibility at `λ` is downward-closed in `λ`, because `λ` enters (C1)
only through `+2λW̄ₖ` with `W̄ₖ ⪰ I ≻ 0`, so lowering `λ` relaxes it. By Theorem 1
the feasible `λ`-set of `P(S)` is the intersection of the per-state ones, and the
intersection of downward-closed intervals has supremum equal to the minimum of
the suprema. `S` is finite, so the min is attained. ∎

**There is no shared-`ν` penalty.** The coupling between states is exactly "the
worst state's demand", never an extra cost on top of it. This corrects
`subset_lambda_procedure.md` §8c, which attributed segway's joint/pointwise gap
to "the cost of one shared ν/χ across heterogeneous states": that gap was
measured across *different draws* (N=100 joint against a separate N=40 pointwise
draw), which Corollary 1.1 does not govern.

Measured on one draw, both sides at `w=[1e-3,1e3]`, `ε=0.1`, `dt=1`, `r=1.6`,
N=8, bisection tol 0.5%:

| plant | `λ*(S)` joint | `min_k λ*(xₖ)` | relative gap |
|---|---|---|---|
| cartpole | 0.7895 | 0.7895 | 0.00% |
| car | 5.0203 | 5.0203 | 0.00% |
| segway | 0.7456 | 0.7505 | 0.65% (= bisection tol + solver) |

---

## 2. The necessary condition: one inequality per mode

**Theorem 2 (Hautus–envelope inequality).** *Let `P({x})` be feasible with
`(W̄, ν, χ)`. Let `s ∈ C`, let `w ∈ Cⁿ` be a unit vector, and set
`δ* := w*(A − sI)`. If `Re s + λ > 0` then*

```
2(Re s + λ) + ε  ≤  2‖δ‖·χ  +  (2ν/r)·w*BBᵀw                            (★)
```

*Proof.* Take the Hermitian form of (C1) against `w`, and write `β := w*W̄w`,
which is real and `≥ 1` because `W̄ ⪰ I` and `‖w‖ = 1`:

```
(β − 1)/dt + w*(AW̄ + W̄Aᵀ)w + 2λβ − ν(2/r)·w*BBᵀw  ≤  −ε
```

From `w*A = s·w* + δ*` we get `w*AW̄w = sβ + δ*W̄w`, and `w*W̄Aᵀw` is its complex
conjugate, so

```
w*(AW̄ + W̄Aᵀ)w = 2Re(s)·β + 2Re(δ*W̄w) ≥ 2Re(s)·β − 2‖δ‖·‖W̄‖ ≥ 2Re(s)·β − 2‖δ‖·χ
```

using `‖W̄‖ ≤ χ` from (C2). Substituting and collecting the `β` terms with
`c := 1/dt + 2Re(s) + 2λ`:

```
β·c  ≤  1/dt + 2‖δ‖·χ + (2ν/r)·w*BBᵀw − ε
```

`Re s + λ > 0` gives `c > 0`, so `β ≥ 1` implies `c ≤ β·c`. Substituting `c` and
cancelling `1/dt` — the `dt` terms drop out exactly — yields (★). ∎

**Corollary 2.1 (class III: exact structural obstruction).** *If for some `x`
there is an `s` with `Re s ≥ −λ` and `σ_min([A(x) − sI, B(x)]) = 0`, then `P({x})`
is infeasible for **every** `(ν, χ, r, dt, w_lb, w_ub)` whenever `ε > 0`.*

*Proof.* Let `w` be the left singular vector of the Hautus matrix for the zero
singular value: `w*[A − sI, B] = 0`, i.e. `δ = 0` and `Bᵀw = 0`. Then (★) reads
`2(Re s + λ) + ε ≤ 0`, contradicting `Re s ≥ −λ` and `ε > 0`. (For `Re s = −λ`
exactly, `Re s + λ > 0` fails; run the same computation directly and the
conclusion still holds, since the left side of (C1) is then `⪰ 0` on `span(w)`
while the right side is `−ε < 0`.) ∎

This is PBH λ-stabilizability, and it is a *finite* check: uncontrollable modes
can only sit at eigenvalues of `A`, so evaluating `σ_min([A − sI, B])` at
`spec(A) ∩ {Re s ≥ −λ}` is exact.

*Use the SVD, not eigenvectors.* pvtol's uncontrollable block is nilpotent, so
`np.linalg.eig` returns a nearly parallel basis and any modal authority computed
from it is meaningless — an earlier version of the script reported pvtol's
authority as `1e-2` when the true Hautus margin is `1e-19`.

**Corollary 2.2 (how expensive a weak mode is).** *With `σ := σ_min([A − sI, B])`
at some `Re s ≥ −λ`, feasibility forces*

```
ν  ≥  [2(Re s + λ) + ε] / (2σ·(w_ub + σ/r))       and, at fixed χ,
ν  ≥  r·[2(Re s + λ) + ε] / (2σ²)
```

*Proof.* Apply (★) at the left singular vector, where `‖δ‖ ≤ σ` and
`w*BBᵀw ≤ σ²`, then substitute `χ ≤ ν·w_ub` (first form) or hold `χ` fixed
(second). ∎

So a small `σ` at ONE state forces a large `ν`, and by Theorem 1's saturation
`ν = 1/w_lb` is shared — which is the precise sense in which one weak state is
expensive everywhere, via `‖M‖ ≤ ν` and `‖K‖ ≤ ‖B‖/(r·w_lb)`.

**Measured, and the 1/σ² branch is exact in the limit.** The car's Hautus matrix
at `s = 0` (after the yaw gauge of Theorem 4) has mutually orthogonal rows of
norms `1, v, 1, 1`, so `σ = min(1, v)`. Against `ρ(v) = λ_max(P(v))`:

| `v` | 0.01 | 0.1 | 1 | 1.5 | 2 | 10 | 1000 |
|---|---|---|---|---|---|---|---|
| `σ = min(1,v)` | 0.01 | 0.1 | 1 | 1 | 1 | 1 | 1 |
| `ρ(v)` | 60542.7 | 608.75 | 9.536 | 9.536 | 9.536 | 9.536 | 39.15 |

`ρ` rises by 99.5× per decade of `v` below 1 — the `1/σ²` branch, to 0.5% — and
is exactly flat wherever `σ` is flat. Note `ρ` climbs again at `v = 1000`
*without* `σ` moving: Corollary 2.2 is a lower bound driven by weak authority and
does not capture growth driven by `‖A‖` itself.

---

## 3. The sufficient condition: a certificate, not a search

**Theorem 3 (constructive feasibility).** *Fix `λ, r, dt > 0, q > 0` and let
`λ' = λ + 1/(2dt)`. Suppose `(A(x) + λ'I, B(x))` is stabilizable for every
`x ∈ S`. Then the CARE*

```
(A + λ'I)ᵀP + P(A + λ'I) − (2/r)·P B Bᵀ P + qI = 0
```

*has a unique stabilizing solution `P(x) ≻ 0` at each `x`, and*

```
W(x) = P(x)⁻¹,   ν = max_x λ_max(P(x)),   w_lb = 1/ν,
w_ub = 1/min_x λ_min(P(x)),   χ = ν·w_ub,   W̄(x) = ν·W(x)
```

*is a feasible point of `P(S)` for every `ε ≤ q·w_lb + 1/dt`.*

*Proof.* **(i)** `(A + λ'I, B)` stabilizable and `(A + λ'I, √q·I)` observable
(immediate, `q > 0`) give a unique stabilizing `P ≻ 0` by standard CARE theory.

**(ii)** Multiply the CARE left and right by `W = P⁻¹`:

```
W(A + λ'I)ᵀ + (A + λ'I)W − (2/r)BBᵀ + qW² = 0
⟹  (A + λ'I)W + W(A + λ'I)ᵀ − (2/r)BBᵀ = −qW²
```

**(iii) Envelope.** `λ_min(W) = 1/λ_max(P) ≥ 1/ν`, so `W̄ = νW ⪰ I`. And
`λ_max(W) = 1/λ_min(P) ≤ w_ub`, so `W̄ ⪯ ν·w_ub·I = χI`. (C2) holds, and (C3)
holds with equality by construction.

**(iv) LMI.** Expand (C1) at `W̄ = νW`, splitting the proxy term
`(W̄ − I)/dt = νW/dt − I/dt` and absorbing `νW/dt` into the rate:

```
(νW − I)/dt + ν[AW + WAᵀ + 2λW] − ν(2/r)BBᵀ
  = ν[(A + λ'I)W + W(A + λ'I)ᵀ − (2/r)BBᵀ]  −  I/dt
  = ν(−qW²) − I/dt
  ⪯ −(ν·q·λ_min(W)² + 1/dt)·I
  ⪯ −(q/ν + 1/dt)·I   =   −(q·w_lb + 1/dt)·I
```

using `λ_min(W) ≥ 1/ν` in the last step. ∎

Two things fall out of step (iv) that are worth naming, because they are the
whole role of Tsukamoto's `(W̄ − I)/dt` proxy:

* the `+νW/dt` half is a **rate penalty** — it is why the certificate must be
  built at `λ' = λ + 1/(2dt)`, not at `λ`;
* the `−I/dt` half is an **exact free margin**, independent of the plant and of
  `ν`. It is why every verified residual below comes out at `−1` for `dt = 1`.

**Corollary 3.1 (the guarantee).** *Pointwise λ'-stabilizability on the box is
sufficient for CV-STEM feasibility at rate λ, with an explicit envelope, an
explicit margin, and no search.* The envelope is an **output** of the plant, not
an input to be tuned until something certifies.

**Corollary 3.2 (sandwich).** The metric scale the plant demands at `x` obeys

```
[2(Re s + λ) + ε] / (2σ(w_ub + σ/r))   ≤   ν_req(x)   ≤   λ_max(P(x))
```

the left from Corollary 2.2 and the right from Theorem 3.

---

## 4. The taxonomy coordinate

**Definition.** `ρ(x) := λ_max(P(x)) ∈ (0, ∞]`, the metric scale state `x`
demands. By Theorem 3, `ν = max_{x∈S} ρ(x)` and `w_lb = 1/max_S ρ`; by
Corollary 1.1 the certified rate is set by the worst state alone.

| class | definition | consequence |
|---|---|---|
| **I** | `ρ` constant on `X` | every state equally hard; **no subset can raise λ** |
| **II** | `ρ` non-constant on `X` | a subset dropping every argmax strictly raises λ |
| **III** | `ρ = ∞` at some `x` (Cor 2.1) | infeasible at **every** envelope |

Class membership is decided by two cheap computations — one eigen-decomposition
plus one SVD for class III, one CARE solve per sample for I vs II.

---

## 5. What makes `ρ` flat, and what makes it vary

**Theorem 4 (class I by orthogonal gauge).** *Suppose there is
`T : X → O(n)` with `T(x)A(x)T(x)ᵀ = A₀` and `T(x)B(x)B(x)ᵀT(x)ᵀ = B₀B₀ᵀ` for all
`x`. Then `ρ(x) ≡ λ_max(P₀)` and `λ*(x) ≡ λ*₀`, so `X` is class I.*

*Proof.* Let `P₀` be the stabilizing CARE solution for `(A₀, B₀)`. Substituting
`P = TᵀP₀T` into the CARE for `(A(x), B(x))` and using `TᵀT = I` reproduces the
`(A₀, B₀)` CARE conjugated by `T`, because both weights are congruence-invariant
under orthogonal `T`: `Tᵀ(qI)T = qI`. Similarity preserves the stabilizing
property, so by uniqueness `P(x) = T(x)ᵀP₀T(x)`, whose spectrum equals that of
`P₀`. The same congruence `W̄ ↦ TᵀW̄₀T` maps feasible points of `P({x})` to
feasible points bijectively and preserves (C2) exactly, since orthogonal
congruence preserves eigenvalues. ∎

*Orthogonality is load-bearing.* A general similarity preserves the LMI but not
(C2), so it can move `ρ` and the envelope. This is why the classification is a
statement about the *metric-normalized* problem, not about `(A, B)` up to
arbitrary coordinates.

The car satisfies Theorem 4 in yaw with `T(θ) = blockdiag(R(θ)ᵀ, I₂)`: writing
`A = [[0₂ₓ₂, M],[0, 0]]` with `M = R(θ)·[[0, 1],[v, 0]]`, the congruence gives
`TAT ᵀ = [[0₂ₓ₂, [[0,1],[v,0]]],[0,0]]`, free of `θ`; and `B` is supported on the
`(yaw, vel)` block where `T` acts as the identity, so `TB = B`. Measured: `ρ`
constant to 6 digits across yaw and across both positions.

**Theorem 5 (class II by control authority).** *Suppose `A(x₁) = A(x₂)` and
`B₁B₁ᵀ ⪰ B₂B₂ᵀ`. Then every certificate feasible at `x₂` is feasible at `x₁` with
the same `(ν, χ)`; hence `λ*(x₁) ≥ λ*(x₂)` and `ρ(x₁) ≤ ρ(x₂)`.*

*Proof.* The left side of (C1) at `x₁` equals that at `x₂` minus
`ν(2/r)(B₁B₁ᵀ − B₂B₂ᵀ) ⪯ 0`, so it is `⪯ −εI` whenever `x₂`'s is; (C2) and (C3)
do not involve `B`. For `ρ`, the stabilizing CARE solution is monotone
non-increasing in `BR⁻¹Bᵀ`, giving `P₁ ⪯ P₂` and hence `ρ(x₁) ≤ ρ(x₂)`. ∎

**Corollary 5.1 (why `σ(B)` predicts and `σ(A)` does not).** Under (C1) the
metric enters the drift term `AW̄ + W̄Aᵀ` linearly and the control term
`−ν(2/r)BBᵀ` not at all — `W̄` is a free per-state variable, so it can absorb
`A(x)` up to the invariant content Corollary 2.1 isolates, but it cannot
manufacture authority. Theorem 5 is the monotone form of that asymmetry, and
Theorem 4 the exact form of "absorbed".

---

## 6. Class membership is a property of (plant, box)

**Theorem 6 (box-dependence).** *For `S' ⊆ S`, `λ*(S') ≥ λ*(S)` and
`max_{S'} ρ ≤ max_S ρ`. Consequently a class-II plant becomes class I on any box
where `ρ` is flat, and a class-I plant becomes class II on any box that reaches
states of different `ρ`.*

*Proof.* Immediate from Corollary 1.1: a min over a smaller set cannot be
smaller. ∎

This is not a technicality; it is the correct reading of the car:

* on its shipped box `v ∈ [1, 2]`, `σ ≡ 1` and `ρ ≡ 9.536` — **class I**;
* on any box reaching `v < 1`, `σ = v` and `ρ ∝ 1/v²` — **class II**, by
  Corollary 2.2 with the measured table of §2;
* at `v = 0` exactly, `f ≡ 0` and the plant *is* the driftless turtlebot —
  **class III**, by Corollary 2.1.

One plant, three classes, selected by the box. So "the car is class I" is only
ever shorthand for "the car on `v ∈ [1,2]` is class I", and the same caution
applies to every row of the table below.

---

## 7. Measured: every classic env

`λ = 0.3`, `r = 1.6`, `cm_dt = 1.0`, `q = 1`, `N = 200` uniform box samples,
seed 0. `--verify` re-evaluates the repo's exact (C1) expression at the
constructed certificate; all feasible rows pass.

```
python scripts/feasibility_certificate.py --all --lbd 0.3 --verify -n 200
```

| env | class | Hautus margin | `ν` (Thm 3) | `w_lb` | `w_ub` | `ρ` spread | ‖K‖max |
|---|---|---|---|---|---|---|---|
| car | **I** | 1.000 | 9.536 | 1.049e-01 | 9.525e-01 | 1.0000 | 2.86 |
| quadrotor | **I** | 1.000 | 55.53 | 1.801e-02 | 5.153e+00 | 1.0000 | 7.27 |
| aircraft | II | 9.76e-02 | 2 196 | 4.553e-04 | 2.633e+01 | 3.26 | 32.9 |
| segway | II | 1.47e-01 | 758.8 | 1.318e-03 | 8.952e+00 | 2.20 | 31.0 |
| cartpole | II | 1.94e-01 | 573.3 | 1.744e-03 | 8.264e-01 | 7.26 | 27.1 |
| two_link_arm | II | 1.24e-01 | 424.8 | 2.354e-03 | 1.984e+01 | 43.2 | 29.6 |
| ball_and_beam | II | 3.22e-01 | 388.5 | 2.574e-03 | 5.524e+01 | 58.6 | 26.4 |
| auv | II | 6.66e-03 | 8.57e+04 | 1.167e-05 | 3.709e-01 | 5 618 | 251 |
| tora | II | 9.71e-03 | 3.26e+05 | 3.070e-06 | 7.978e-01 | 1 152 | 448 |
| **pvtol** | **III** | 1.05e-19 | ∞ | — | — | — | — |
| **turtlebot** | **III** | 0.00e+00 | ∞ | — | — | — | — |

**Answer to "are these environments contraction-feasible, guaranteed?"** — 9 of
11, yes, with the envelope in the table and a verified LMI margin of
`q·w_lb + 1/dt`. The two exceptions are not tuning failures and cannot be fixed
by any envelope, `r`, `dt` or `λ`: both have an uncontrollable mode at `s = 0`
(Corollary 2.1). turtlebot's margin is exactly zero because `f ≡ 0` makes
`A ≡ 0`; pvtol's `1e-19` is a numerically-zero nilpotent block, matching the
independently-extracted `A₂₂` with `eig = {0,0}` in
`cvstem_feasibility_theory.md` §1.3. For both, the fix is a method that sees Lie
brackets — `cmg_method: ccm` — not a different envelope.

The Hautus margin orders the whole table: class I sits at 1.0, class II between
`7e-3` and `3e-1`, class III at 0. It costs one eigendecomposition and one SVD
per sample and needs no SDP, which makes it the cheapest classifier available.

**The classes are stable in λ.** Re-running at `λ = 1.0` reproduces the
assignment exactly, env for env; only `ν` moves, and upward, as Corollary 2.2
requires (car 9.5 → 31.3, cartpole 573 → 4 452, tora 3.3e5 → 7.0e6). The Hautus
margins are unchanged to three digits, since raising λ only widens the set of
eigenvalues tested.

**Reconciliation with `find_uniform_lambda`'s "segway INFEASIBLE".** At `λ = 0.3`
segway needs `w_lb = 1.318e-3`, which *fits* the shipped envelope `[1e-3, 1e3]`;
at `λ = 1.0` it needs `3.5e-4`, which does not. So segway is contraction-feasible
at the shipped envelope for small enough λ, and the repo's INFEASIBLE verdict
comes from the *actuator* branch driving `r` up until `ν` hits its cap — exactly
the mechanism `subset_lambda_procedure.md` §8d describes. The two results do not
conflict; they answer different questions, which is why the control box is kept
out of this document.

---

## 8. What is not claimed

* **Sampled, not box-wide.** Every statement is over the finite `S`. Extending to
  all of `X` needs the covering argument of `subset_lambda_procedure.md` §7: an
  `ε` margin buys radius `ε/L` for an `L`-Lipschitz `x ↦ S(x)`. Theorem 3's
  margin `q·w_lb + 1/dt` is what feeds that argument, but the Lipschitz constant
  is not computed here.
* **Feasibility, not performance.** `λ*` is a certified rate, not a measured AUC.
  Theorem 3 chooses a *particular* metric (the CARE one); it certifies but does
  not optimize, so its `ν` is an upper bound on what the SDP would find — the
  gap is 20× on the car (9.54 against a Corollary-2.2 floor of 0.49).
* **Invariance is still owed for subset claims.** `λ*(S')` certifies contraction
  while the trajectory stays in `S'`. Theorem 6 says shrinking the box raises the
  rate; it says nothing about whether the closed loop remains in the smaller box.
* **The control box is out of scope by construction.** A plant with `ν = 3e5`
  (tora) is contraction-feasible in exactly the same sense as one with `ν = 9.5`
  (car). Whether its gain fits an actuator is a separate question, answered by
  `find_uniform_lambda.py`, and it is deliberately not allowed to influence any
  class here.

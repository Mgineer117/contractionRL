"""Certified global optimum of the toy tracking problem, by the sparse
moment-SOS (Lasserre) hierarchy.

Why a hierarchy and not branch-and-bound
----------------------------------------
Spatial B&B costs grow exponentially in the number of nonconvex variables, and
here that number IS the horizon: gamma=0.01 truncates to 8 steps (24 variables)
but gamma=0.99 needs the full 100 (300 variables, 200 cubic equalities). A
discount SWEEP would then report certified optima at low gamma and timeouts at
high gamma. This problem is a CHAIN -- x_{t+1} couples only to (x_t, u_t) -- so
correlative sparsity (Waki et al. 2006; Lasserre 2006) splits it into cliques of
five variables whose size does not depend on the horizon at all. Cost becomes
linear in T: gamma=0.99 is 100 small SDPs instead of 8, not an exponential wall.

The problem, per (reference, x_0) task
--------------------------------------
    min  sum_t gamma^t c_t(x_t, u_t)
    s.t. x_{t+1} = x_t + dt (f(x_t) + B(x_t) u_t)
         u_t in [U_MIN, U_MAX],  x_t in [X_MIN, X_MAX]

``c_t`` is C2RL's reward negated, transcribed from ``BaseEnv.get_rewards``:
``c_t = -q (V_t - V_{t+1}) + r ||u_t||^2`` with ``V_t = e_t' M(x_t) e_t``,
``e_t = x_t - xref_t``.

Two reformulations make that a polynomial program.

1. TELESCOPE. The decrement sum collapses:

       sum_{t<T} g^t (V_t - V_{t+1}) = V_0 - (1-g) sum_{1<=t<T} g^{t-1} V_t - g^{T-1} V_T

   so minimising the decrement cost is minimising a POSITIVE weighted sum of the
   levels, ``sum_t w_t V_t``, plus control effort -- ``V_0`` is a constant since
   ``x_0`` is fixed. Fewer cross terms, and every objective monomial then lands
   on a single clique, which is what correlative sparsity needs.

2. EPIGRAPH THE RATIONAL METRIC. The SOS certificate is written in ``W``, so
   ``W(x)`` is the degree-2 polynomial and ``M = W^-1`` is RATIONAL -- the
   objective is not polynomial as written. Introduce ``v_t`` for ``V_t`` with

       v_t det(W(x_t)) >= e_t' adj(W(x_t)) e_t          (degree 5)

   ``det W > 0`` on the box by the same certificate, and ``w_t >= 0``, so
   minimising drives this to equality: exact, not a relaxation. The objective
   becomes LINEAR in ``v``.

Cliques are then ``A_t = {x_t, u_t, x_{t+1}}`` (dynamics, boxes, control cost)
and ``B_t = {x_t, v_t}`` (the metric epigraph, level cost). Ordering
A_0, B_1, A_1, B_2, ... satisfies the running-intersection property, since each
clique meets the union of its predecessors in ``{x_t}`` or ``{x_t, v_t}``.

What comes back is a certified LOWER bound. The first-order moments give a
candidate control sequence; rolling it through the real env gives a feasible
UPPER bound, and the two together are the certificate -- see ``solve_task``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import scipy.sparse as sp
import sympy as sy

from .sos_cm import PLANTS, XS

# ─────────────────────────────────────────────────────────────────────────── #
# Sparse polynomials over a global variable set
#
# A monomial is a sorted tuple of (variable index, exponent) with every exponent
# positive; the empty tuple is 1. A polynomial is {monomial: coefficient}. Dense
# exponent vectors are not an option here -- the global variable count is 5T, so
# at T=100 every monomial would be a 500-long tuple of mostly zeros.
# ─────────────────────────────────────────────────────────────────────────── #

def mono_mul(a: tuple, b: tuple) -> tuple:
    d = dict(a)
    for k, e in b:
        d[k] = d.get(k, 0) + e
    return tuple(sorted(d.items()))


def mono_deg(m: tuple) -> int:
    return sum(e for _, e in m)


def poly_mul(p: dict, q: dict) -> dict:
    out: dict = {}
    for ma, ca in p.items():
        for mb, cb in q.items():
            m = mono_mul(ma, mb)
            out[m] = out.get(m, 0.0) + ca * cb
    return {m: c for m, c in out.items() if abs(c) > 1e-14}


def poly_add(*ps: dict) -> dict:
    out: dict = {}
    for p in ps:
        for m, c in p.items():
            out[m] = out.get(m, 0.0) + c
    return {m: c for m, c in out.items() if abs(c) > 1e-14}


def poly_scale(p: dict, s: float) -> dict:
    return {m: c * s for m, c in p.items() if abs(c * s) > 1e-14}


def poly_deg(p: dict) -> int:
    return max((mono_deg(m) for m in p), default=0)


def var(i: int) -> dict:
    """The polynomial ``x_i``."""
    return {((i, 1),): 1.0}


def const(c: float) -> dict:
    return {(): float(c)} if abs(c) > 1e-14 else {}


def monomials(vs: list[int], deg: int) -> list[tuple]:
    """Every monomial over ``vs`` of total degree <= ``deg``, degree-ordered."""
    out = [()]
    for d in range(1, deg + 1):
        for combo in itertools.combinations_with_replacement(sorted(vs), d):
            m: dict = {}
            for v in combo:
                m[v] = m.get(v, 0) + 1
            out.append(tuple(sorted(m.items())))
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# The relaxation
# ─────────────────────────────────────────────────────────────────────────── #

class SparseMomentProgram:
    """Order-``d`` sparse moment relaxation, assembled clique by clique.

    One shared moment vector ``y`` indexed by global monomials is what couples
    the cliques: two cliques that overlap read the SAME ``y`` entries for the
    monomials they share, so no explicit consistency constraint is needed.
    """

    def __init__(self):
        self.index: dict[tuple, int] = {(): 0}
        self.psd: list[tuple[sp.spmatrix, int]] = []
        self.eqs: list[sp.spmatrix] = []
        self.obj: dict[tuple, float] = {}
        self.const = 0.0

    def _idx(self, m: tuple) -> int:
        i = self.index.get(m)
        if i is None:
            i = len(self.index)
            self.index[m] = i
        return i

    def add_moment_matrix(self, vs: list[int], d: int) -> None:
        """``M_d(y)`` restricted to clique ``vs`` -- the PSD block."""
        mons = monomials(vs, d)
        m = len(mons)
        rows, cols, vals = [], [], []
        for i, a in enumerate(mons):
            for j, b in enumerate(mons):
                rows.append(i * m + j)
                cols.append(self._idx(mono_mul(a, b)))
                vals.append(1.0)
        self.psd.append((sp.coo_matrix((vals, (rows, cols)), shape=(m * m, len(self.index))), m))

    def add_localizing(self, g: dict, vs: list[int], d: int) -> None:
        """``M_{d-ceil(deg g/2)}(g y) >= 0`` for an inequality ``g >= 0``."""
        dg = d - math.ceil(poly_deg(g) / 2)
        if dg < 0:
            raise ValueError(f"relaxation order {d} is below ceil(deg g / 2) = "
                             f"{math.ceil(poly_deg(g) / 2)} for a degree-{poly_deg(g)} "
                             f"constraint; raise the order for this clique.")
        mons = monomials(vs, dg)
        m = len(mons)
        rows, cols, vals = [], [], []
        for i, a in enumerate(mons):
            for j, b in enumerate(mons):
                ab = mono_mul(a, b)
                for mb, c in g.items():
                    rows.append(i * m + j)
                    cols.append(self._idx(mono_mul(ab, mb)))
                    vals.append(c)
        self.psd.append((sp.coo_matrix((vals, (rows, cols)), shape=(m * m, len(self.index))), m))

    def add_equality(self, h: dict, vs: list[int], d: int) -> None:
        """``sum_b h_b y_{a+b} = 0`` for every ``a`` over the clique with
        ``deg a <= 2d - deg h``. Stronger than a localizing matrix of the same
        order, and it is where an equality-constrained POP gets its tightness."""
        da = 2 * d - poly_deg(h)
        if da < 0:
            raise ValueError(f"relaxation order {d} cannot carry a degree-"
                             f"{poly_deg(h)} equality; raise the order.")
        rows, cols, vals = [], [], []
        for i, a in enumerate(monomials(vs, da)):
            for mb, c in h.items():
                rows.append(i)
                cols.append(self._idx(mono_mul(a, mb)))
                vals.append(c)
        n = rows[-1] + 1 if rows else 0
        self.eqs.append(sp.coo_matrix((vals, (rows, cols)), shape=(n, len(self.index))))

    def add_objective(self, p: dict) -> None:
        for m, c in p.items():
            if m == ():
                self.const += c
            else:
                self.obj[m] = self.obj.get(m, 0.0) + c
                self._idx(m)

    def solve(self, solver: str = "MOSEK", verbose: bool = False,
              threads: int | None = None) -> dict:
        import cvxpy as cp

        n = len(self.index)
        y = cp.Variable(n)
        cons = [y[0] == 1.0]
        for S, m in self.psd:
            S = sp.csr_matrix((S.data, (S.row, S.col)), shape=(S.shape[0], n))
            cons.append(cp.reshape(S @ y, (m, m), order="C") >> 0)
        for E in self.eqs:
            E = sp.csr_matrix((E.data, (E.row, E.col)), shape=(E.shape[0], n))
            cons.append(E @ y == 0)
        c = np.zeros(n)
        for m, v in self.obj.items():
            c[self.index[m]] = v
        prob = cp.Problem(cp.Minimize(c @ y + self.const), cons)
        # threads=1 is what makes a PROCESS pool over these solves scale. MOSEK
        # sizes its own thread pool to the machine, so N concurrent workers each
        # claimed ~65 threads on a 24-core box -- 900+ threads fighting, and 14
        # workers were SLOWER than 6. Left None (MOSEK's default) for a single
        # solve, where the parallelism is free.
        kw = {} if threads is None or solver != "MOSEK" else {
            "mosek_params": {"MSK_IPAR_NUM_THREADS": int(threads)}}
        prob.solve(solver=solver, verbose=verbose, **kw)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"moment relaxation did not solve: {prob.status}")
        return {"bound": float(prob.value), "status": prob.status,
                "y": np.asarray(y.value).ravel(), "num_moments": n,
                "num_psd_blocks": len(self.psd)}

    def first_order(self, y: np.ndarray, i: int) -> float:
        """``E[x_i]`` -- the candidate minimiser's i-th coordinate, exact when
        the relaxation is tight (the optimal measure is then a Dirac)."""
        j = self.index.get(((i, 1),))
        return float(y[j]) if j is not None else float("nan")


# ─────────────────────────────────────────────────────────────────────────── #
# The toy tracking task as a polynomial program
#
# The hierarchy above knows nothing about control; this half knows the plant,
# the reward and the certificate. The dynamics are written out as polynomials
# rather than traced from the torch env -- and then CHECKED against
# ``get_f_and_B`` at construction. Transcribing an objective by hand is exactly
# how the first version of the V* pipeline came to score rollouts under a reward
# C2RL never used, so the guard is not optional.
# ─────────────────────────────────────────────────────────────────────────── #
def _coeffs(expr) -> dict:
    """sympy drift entry -> {(a, b): coefficient} for x1^a x2^b."""
    poly = sy.Poly(sy.expand(expr), *XS)
    return {tuple(m): float(c) for m, c in zip(poly.monoms(), poly.coeffs())}


# Derived from the SAME symbolic plant the SOS metric is certified against, so a
# dynamics edit cannot leave the optimum solving yesterday's system. A hand
# written third copy used to live here and did exactly that: it kept mg's old
# ``-x_2`` drift after the plant became ``3 x_1``, which check_dynamics caught
# only because the guard below exists. B is [0; 1] and constant for both plants.
POLY_F = {k: [_coeffs(e) for e in v["f"]] for k, v in PLANTS.items()}
POLY_B = [[float(b)] for b in PLANTS["mg"]["B"]]
assert all([[float(b)] for b in v["B"]] == POLY_B for v in PLANTS.values()), \
    "POLY_B assumes every toy plant shares the constant input matrix [0; 1]"


def check_dynamics(env, key: str, n: int = 512, tol: float = 1e-5) -> None:
    """The hand-written polynomial must BE the env's drift, not resemble it."""
    import torch

    rng = np.random.default_rng(0)
    lo = np.asarray(env.X_MIN, dtype=np.float64).ravel()
    hi = np.asarray(env.X_MAX, dtype=np.float64).ravel()
    X = rng.uniform(lo, hi, size=(n, 2))
    f_env, B_env, _ = env.get_f_and_B(torch.as_tensor(X, dtype=torch.float32), need_null=False)
    f_env = f_env.detach().cpu().numpy()
    B_env = B_env.detach().cpu().numpy()
    f_poly = np.stack([
        sum(c * X[:, 0] ** a * X[:, 1] ** b for (a, b), c in POLY_F[key][k].items())
        for k in range(2)], axis=1)
    err = np.abs(f_poly - f_env).max()
    berr = np.abs(B_env - np.asarray(POLY_B)[None]).max()
    if err > tol or berr > tol:
        raise ValueError(
            f"POLY_F/POLY_B for {key!r} do not match the env: max drift error "
            f"{err:.3e}, max B error {berr:.3e}. The global optimum would be for "
            f"a different plant than the policy is trained on.")


def w_entries(coeffs: dict, w_degree: int, xs: list[dict]) -> tuple[dict, dict, dict]:
    """The three entries of W(x) as polynomials in the supplied x-expressions.

    ``xs`` holds a polynomial per state dim, so the same code lifts W at a
    variable state (``var(i)``) and at the fixed ``x_0`` (``const(...)``).
    """
    out = []
    for i0, i1 in ((0, 0), (0, 1), (1, 1)):
        p: dict = {}
        for a in range(w_degree + 1):
            for b in range(w_degree + 1 - a):
                c = float(coeffs[f"w_{i0}{i1}_{a}{b}"])
                if abs(c) <= 1e-14:
                    continue
                term = const(c)
                for _ in range(a):
                    term = poly_mul(term, xs[0])
                for _ in range(b):
                    term = poly_mul(term, xs[1])
                p = poly_add(p, term)
        out.append(p)
    return out[0], out[1], out[2]


def metric_value(coeffs: dict, w_degree: int, x: np.ndarray, xref: np.ndarray) -> float:
    """``e' W(x)^-1 e`` numerically -- the same quantity ``v_t`` stands for."""
    w00, w01, w11 = (float(sum(coeffs[f"w_{i0}{i1}_{a}{b}"] * x[0] ** a * x[1] ** b
                               for a in range(w_degree + 1)
                               for b in range(w_degree + 1 - a)))
                     for i0, i1 in ((0, 0), (0, 1), (1, 1)))
    e = np.asarray(x, dtype=np.float64) - np.asarray(xref, dtype=np.float64)
    det = w00 * w11 - w01 ** 2
    return float((w11 * e[0] ** 2 - 2 * w01 * e[0] * e[1] + w00 * e[1] ** 2) / det)


def v_bound(coeffs, w_degree, xref, x_lo, x_hi, n=81):
    """A ceiling on ``e' M(x) e`` over the box, for the ``v`` variables' scale.

    Moment relaxations are scale-sensitive, so ``v`` is normalised by this and
    the hierarchy sees ``v/v_max in [0,1]`` like every other variable.
    """
    g = np.stack(np.meshgrid(np.linspace(x_lo[0], x_hi[0], n),
                             np.linspace(x_lo[1], x_hi[1], n), indexing="ij"), -1).reshape(-1, 2)
    best = 0.0
    for xr in np.asarray(xref)[::max(1, len(xref) // 16)]:
        best = max(best, max(metric_value(coeffs, w_degree, x, xr) for x in g))
    return float(best) * 1.05


def build(key, coeffs, w_degree, x0, xref, gamma, T, dt, x_lo, x_hi, u_lo, u_hi,
          q=1.0, r=0.0, v_max=None, order_a=3, order_b=4, level=False):
    """Assemble the order-(``order_a``, ``order_b``) sparse relaxation.

    Every variable is affinely rescaled into [-1,1] (``v`` into [0,1]) before the
    hierarchy sees it. Not cosmetic: unscaled, one moment matrix mixes ``u^6``
    ~ 4.7e4 with ``x^6`` ~ 7e-4, and MOSEK either fails outright or returns a
    "bound" ABOVE the true optimum -- measured, before this, at 2.7% of |J| on
    the wrong side, which is not a bound at all.

    Returns ``(program, meta)``; ``meta`` carries the maps and scales needed to
    read a point back out of the moments. Horizon ``T`` counts transitions, and
    ``xref`` must be sliced to exactly ``max_episode_len`` points -- see the
    index clamp below.
    """
    prog = SparseMomentProgram()
    nx = 2
    x_lo = np.asarray(x_lo, dtype=np.float64).ravel()
    x_hi = np.asarray(x_hi, dtype=np.float64).ravel()
    cx, sx = 0.5 * (x_lo + x_hi), 0.5 * (x_hi - x_lo)
    cu, su = 0.5 * (u_lo + u_hi), 0.5 * (u_hi - u_lo)
    if v_max is None:
        v_max = v_bound(coeffs, w_degree, xref, x_lo, x_hi)

    xi, ui, vi, nxt = {}, {}, {}, 0
    for t in range(1, T + 1):
        for k in range(nx):
            xi[(t, k)] = nxt
            nxt += 1
        vi[t] = nxt
        nxt += 1
    for t in range(T):
        ui[t] = nxt
        nxt += 1

    def xexpr(t):
        if t == 0:
            return [const(float(x0[k])) for k in range(nx)]
        return [poly_add(const(cx[k]), poly_scale(var(xi[(t, k)]), sx[k])) for k in range(nx)]

    def uexpr(t):
        return poly_add(const(cu), poly_scale(var(ui[t]), su))

    def lift(p2, xs):
        """A polynomial in (x1, x2) re-expressed in whatever ``xs`` holds."""
        out = {}
        for (a, b), c in p2.items():
            term = const(c)
            for _ in range(a):
                term = poly_mul(term, xs[0])
            for _ in range(b):
                term = poly_mul(term, xs[1])
            out = poly_add(out, term)
        return out

    def unit_box(i):
        """(z + 1)(1 - z) >= 0 -- one degree-2 constraint instead of two linear
        cuts, so the localizing matrix gets a higher order at the same
        relaxation order."""
        return poly_mul(poly_add(var(i), const(1.0)), poly_add(const(1.0), poly_scale(var(i), -1.0)))

    # ---- clique A_t: dynamics, boxes, control effort -------------------- #
    for t in range(T):
        xt, xn = xexpr(t), xexpr(t + 1)
        vs = [ui[t]] + [xi[(t + 1, k)] for k in range(nx)]
        if t > 0:
            vs = [xi[(t, k)] for k in range(nx)] + vs
        prog.add_moment_matrix(vs, order_a)
        for k in range(nx):
            rhs = poly_add(lift(POLY_F[key][k], xt), poly_scale(uexpr(t), POLY_B[k][0]))
            # Divided by sx[k]: the equality is written in the ORIGINAL state
            # units, where dt*f is ~1e-2 while the state box halfwidth is 2.5.
            # Left as is, the dynamics rows are 250x weaker than the box rows and
            # the interior-point solver drifts off them.
            prog.add_equality(poly_scale(poly_add(xn[k], poly_scale(xt[k], -1.0),
                                                  poly_scale(rhs, -dt)), 1.0 / sx[k]),
                              vs, order_a)
        prog.add_localizing(unit_box(ui[t]), vs, order_a)
        for k in range(nx):
            prog.add_localizing(unit_box(xi[(t + 1, k)]), vs, order_a)
        if r:
            u = uexpr(t)
            prog.add_objective(poly_scale(poly_mul(u, u), r * gamma ** t))

    # ---- clique B_t: the metric epigraph and the stage cost -------------- #
    # Both weightings are positive throughout, which is what makes the epigraph
    # bind at the optimum rather than merely bound it.
    #
    # ``level``   the standard RL cost of Theorem 1: stage cost is the Riemannian
    #             energy itself, C(s_t) = e_t' M(x_t) e_t, so the weight on v_t is
    #             just gamma^(t-1) (transition t-1 is scored by the energy at the
    #             state it lands in, matching get_rewards' reward_level branch).
    # otherwise   the potential-shaped decrement, telescoped onto the same levels.
    for t in range(1, T + 1):
        xt = xexpr(t)
        vs = [xi[(t, k)] for k in range(nx)] + [vi[t]]
        prog.add_moment_matrix(vs, order_b)
        w00, w01, w11 = w_entries(coeffs, w_degree, xt)
        det = poly_add(poly_mul(w00, w11), poly_scale(poly_mul(w01, w01), -1.0))
        # min(t, last): get_rewards reads xref[clamp(t_idx, max=max_episode_len-1)],
        # so the final transition targets the LAST reference point rather than
        # one past the end. ``xref`` must therefore arrive sliced to
        # max_episode_len -- a longer buffer (configure_ref_window pads it so the
        # observation window never clamps) would silently aim at a point the
        # reward never uses.
        xr = xref[min(t, len(xref) - 1)]
        e = [poly_add(xt[k], const(-float(xr[k]))) for k in range(nx)]
        num = poly_add(poly_mul(w11, poly_mul(e[0], e[0])),
                       poly_scale(poly_mul(w01, poly_mul(e[0], e[1])), -2.0),
                       poly_mul(w00, poly_mul(e[1], e[1])))
        vv = poly_scale(var(vi[t]), v_max)
        prog.add_localizing(poly_add(poly_mul(vv, det), poly_scale(num, -1.0)), vs, order_b)
        # Archimedean: v is the only variable the state box does not bound.
        prog.add_localizing(poly_mul(var(vi[t]),
                                     poly_add(const(1.0), poly_scale(var(vi[t]), -1.0))),
                            vs, order_b)
        if level:
            wt = q * gamma ** (t - 1)
        else:
            wt = q * gamma ** (T - 1) if t == T else q * (1.0 - gamma) * gamma ** (t - 1)
        prog.add_objective(poly_scale(vv, wt))

    if not level:
        # The telescoped form carries a constant -q V_0; the level form does not.
        prog.add_objective(const(-q * metric_value(coeffs, w_degree,
                                                   np.asarray(x0), xref[0])))
    meta = {"x": xi, "u": ui, "v": vi, "cx": cx, "sx": sx, "cu": cu, "su": su,
            "v_max": v_max, "T": T}
    return prog, meta


def read_point(prog, y, meta):
    """The candidate minimiser, unscaled back into plant units."""
    T, cx, sx = meta["T"], meta["cx"], meta["sx"]
    u = np.array([meta["cu"] + meta["su"] * prog.first_order(y, meta["u"][t]) for t in range(T)])
    x = np.array([[cx[k] + sx[k] * prog.first_order(y, meta["x"][(t, k)]) for k in range(2)]
                  for t in range(1, T + 1)])
    return u, x

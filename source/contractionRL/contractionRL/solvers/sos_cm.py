"""Exact contraction metric for the polynomial ``toy`` envs, by sum of squares.

Every classic env certifies its rate by SAMPLING: the joint CV-STEM SDP is
solved at N drawn states, so lambda provably holds at those N points and is
interpolated between them. For a polynomial plant that concession is
unnecessary. A polynomial is nonnegative on a box if it can be written

    p(x) = s_0(x) + sum_i s_i(x) (x_i - lo_i)(hi_i - x_i),      s_j SOS,

which is an algebraic IDENTITY: it holds at every point of the box, not at a
draw. That is the whole reason the toy family exists.

The condition is the SAME CV-STEM LMI ``ncm_synthesis`` solves for the classic
envs, so the two families' rates mean the same thing and are comparable:

    -Wdot + A W + W A' - (2/r) B B' + 2 lam W  <=  0,   Wdot = sum_i dW/dx_i f_i

Note this is NOT the B_perp-projected CCM condition. That one drops the ``r``
term, so its lambda would certify "some contracting controller exists" while the
deployed feedback is the Riccati form ``u = -(1/r) B' M e`` -- the certified and
the measured rate would then be different quantities. Matching the classic LMI
keeps ``local_lambda`` honest on both families.

lam multiplies W, so it is found by bisection: each trial lam is a linear SDP.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import sympy as sp

X1, X2 = sp.symbols("x1 x2", real=True)
XS = (X1, X2)
V1, V2 = sp.symbols("v1 v2", real=True)

# Symbolic drift per toy env, mirroring tasks/direct/toy/<key>/env.py. Kept
# separately because the env speaks torch and SOS needs sympy; certify_ccm
# cross-checks the two numerically before solving, so they cannot drift apart.
PLANTS = {
    "mg": {"f": [-X2 - sp.Rational(3, 2) * X1**2 - sp.Rational(1, 2) * X1**3,
                 3 * X1],
           "B": [0, 1]},
    "duff": {"f": [X2, -X1 + X1**3 / 3 - sp.Rational(3, 10) * X2],
             "B": [0, 1]},
}


@dataclass
class SOSCfg:
    """SOS synthesis knobs. Mirrors ``skrl_sos_cfg.yaml``'s ``agent`` block."""

    w_degree: int = 2         # total degree of each entry of W(x)
    w_lb: float = 1.0
    w_ub: float = 50.0
    r_scaler: float = 1.0     # the CV-STEM LMI's R = r I, same role as cvstem_r_scaler
    lam_hi: float = 4.0       # upper end of the bisection
    bisect_iters: int = 10
    solver: str = "MOSEK"
    verify_grid: int = 101


def _monomials(max_deg: int, nvars: int = 2):
    return [e for e in itertools.product(range(max_deg + 1), repeat=nvars)
            if sum(e) <= max_deg]


def _coeffs(expr, gens):
    """sympy expr -> {exponent tuple: coefficient}. Coefficients stay symbolic
    because they are linear in the unknown metric coefficients."""
    p = sp.Poly(sp.expand(expr), *gens)
    return {tuple(m): c for m, c in zip(p.monoms(), p.coeffs())}


class _BoxSOS:
    """Accumulates ``p(x) >= 0 on the box`` constraints into one cvxpy problem."""

    def __init__(self, lo, hi, unknowns):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.unknowns = list(unknowns)
        self.uvar = cp.Variable(len(self.unknowns))
        self.cons = []

    def _affine(self, expr):
        """sympy expr linear in self.unknowns -> cvxpy affine expression."""
        A, b = sp.linear_eq_to_matrix([sp.expand(expr)], self.unknowns)
        return np.array(A, float).reshape(-1) @ self.uvar - float(b[0])

    def require_nonneg(self, expr, vec_vars=()):
        """Require expr >= 0 on the box (and for all vec_vars, which enter
        quadratically -- that is how ``v'(W - w_lb I)v >= 0`` is expressed)."""
        gens = XS + tuple(vec_vars)
        target = _coeffs(expr, gens)
        deg = sp.Poly(sp.expand(expr), *gens).total_degree()
        deg += deg % 2

        terms = {}

        def gram(basis, mult):
            if not basis:
                return
            Q = cp.Variable((len(basis), len(basis)), PSD=True)
            self.cons.append(Q >> 0)
            for a, ea in enumerate(basis):
                for b_, eb in enumerate(basis):
                    for me, mc in mult.items():
                        k = tuple(i + j + m for i, j, m in zip(ea, eb, me))
                        terms[k] = terms.get(k, 0) + float(mc) * Q[a, b_]

        def basis_of(half):
            """Monomials of degree <= half in x, times exactly one vec_var.

            The vec_vars appear at degree 1 in the basis so every Gram entry is
            quadratic in them -- matching v'(.)v exactly, with no spurious
            v^0 or v^4 terms to be absorbed by the coefficient matching.
            """
            if half < 0:
                return []
            xs = _monomials(half, 2)
            if not vec_vars:
                return [e + (0,) * len(vec_vars) for e in xs]
            return [e + tuple(1 if k == i else 0 for k in range(len(vec_vars)))
                    for e in _monomials(max(half - 1, 0), 2)
                    for i in range(len(vec_vars))]

        one = {(0,) * len(gens): 1.0}
        gram(basis_of(deg // 2), one)                       # s_0
        for i in range(2):                                  # one s_i per box dim
            g = _coeffs((XS[i] - self.lo[i]) * (self.hi[i] - XS[i]), gens)
            gram(basis_of((deg - 2) // 2), g)

        for k in set(terms) | set(target):
            rhs = target.get(k, 0)
            self.cons.append(terms.get(k, 0) == (self._affine(rhs) if rhs != 0 else 0))


def _w_poly(deg, tag="w"):
    """Symmetric 2x2 matrix of degree-``deg`` polynomials with fresh unknowns."""
    mons = _monomials(deg, 2)
    syms, W = [], sp.zeros(2, 2)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        e = 0
        for m in mons:
            s = sp.Symbol(f"{tag}_{i}{j}_{m[0]}{m[1]}", real=True)
            syms.append(s)
            e += s * X1**m[0] * X2**m[1]
        W[i, j] = W[j, i] = e
    return W, syms


def check_plant(env, key, tol=1e-5, n=64):
    """The symbolic plant must BE the env's, or the certificate is about nothing."""
    import torch
    if key not in PLANTS:
        raise ValueError(f"no symbolic plant for {key!r}; SOS needs a polynomial "
                         f"f/B, which only the toy envs have. Known: {sorted(PLANTS)}")
    lo = env.X_MIN.cpu().numpy()
    hi = env.X_MAX.cpu().numpy()
    pts = np.random.default_rng(0).uniform(lo, hi, size=(n, 2))
    fsym = sp.lambdify(XS, PLANTS[key]["f"], "numpy")
    with torch.no_grad():
        f_env, B_env, _ = env.get_f_and_B(torch.as_tensor(pts, dtype=torch.float64),
                                          need_null=False)
    df = np.abs(f_env.numpy() - np.array([np.asarray(fsym(*p), float) for p in pts])).max()
    dB = np.abs(B_env.numpy() - np.array(PLANTS[key]["B"], float).reshape(1, 2, 1)).max()
    if max(df, dB) > tol:
        raise ValueError(f"{key}: symbolic plant disagrees with the env "
                         f"(max |df| {df:.3e}, |dB| {dB:.3e}). Fix PLANTS in "
                         f"solvers/sos_cm.py before trusting any certificate.")
    return max(df, dB)


def _lmi(key, lam, W, r_scaler):
    """The CV-STEM LMI as a symbolic 2x2 matrix; feasible means this is <= 0."""
    f = PLANTS[key]["f"]
    B = sp.Matrix(2, 1, PLANTS[key]["B"])
    A = sp.Matrix(2, 2, lambda i, j: sp.diff(f[i], XS[j]))
    Wdot = sum((sp.diff(W, XS[i]) * f[i] for i in range(2)), sp.zeros(2, 2))
    return -Wdot + A * W + W * A.T - (2 / sp.Rational(str(r_scaler))) * B * B.T + 2 * lam * W


def _feasible_at(key, lam, cfg, lo, hi):
    """One feasibility SDP at fixed lam. Returns the W coefficients or None."""
    W, wsyms = _w_poly(cfg.w_degree)
    v = sp.Matrix([V1, V2])
    b = _BoxSOS(lo, hi, wsyms)
    # Matrix condition, so it is imposed as v'(-LMI)v >= 0 for all v -- the full
    # 2x2, not one projected entry.
    b.require_nonneg(sp.expand((v.T * (-_lmi(key, lam, W, cfg.r_scaler)) * v)[0, 0]),
                     (V1, V2))
    b.require_nonneg(sp.expand((v.T * (W - cfg.w_lb * sp.eye(2)) * v)[0, 0]), (V1, V2))
    b.require_nonneg(sp.expand((v.T * (cfg.w_ub * sp.eye(2) - W) * v)[0, 0]), (V1, V2))

    prob = cp.Problem(cp.Minimize(0), b.cons)
    try:
        prob.solve(solver=cfg.solver)
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return dict(zip([str(s) for s in wsyms], np.asarray(b.uvar.value).ravel()))


def w_at(key, coeffs, w_deg, pts):
    """Evaluate the certified W(x) at ``pts`` (n, 2) -> (n, 2, 2).

    This is how an exact metric becomes the same ``{x -> W*(x)}`` dataset the
    classic envs produce by sampling the SDP: identical schema downstream, but
    every sample is drawn from a field that is certified everywhere rather than
    solved at that point alone.
    """
    W, wsyms = _w_poly(w_deg)
    Wn = W.subs({s: coeffs[str(s)] for s in wsyms})
    Wf = sp.lambdify(XS, Wn, "numpy")
    pts = np.asarray(pts, float)
    out = np.empty((len(pts), 2, 2))
    for k, (a, b_) in enumerate(pts):
        out[k] = np.array(Wf(a, b_), float)
    return out


def verify(key, lam, coeffs, cfg, lo, hi, grid=101):
    """Re-evaluate the certified LMI on a dense grid.

    The SDP says the SOS identity holds; this says the PLANT does. A Gram-matrix
    bookkeeping slip makes the program feasible for the wrong reason and the
    certificate silently vacuous, so this is the check, not a formality.
    """
    W, wsyms = _w_poly(cfg.w_degree)
    Wn = W.subs({s: coeffs[str(s)] for s in wsyms})
    Lf = sp.lambdify(XS, _lmi(key, lam, Wn, cfg.r_scaler), "numpy")
    Wf = sp.lambdify(XS, Wn, "numpy")

    g = [np.linspace(lo[i], hi[i], grid) for i in range(2)]
    G0, G1 = np.meshgrid(*g, indexing="ij")
    worst, eig = -np.inf, np.inf
    for a, b_ in zip(G0.ravel(), G1.ravel()):
        L = np.array(Lf(a, b_), float)
        worst = max(worst, float(np.linalg.eigvalsh(0.5 * (L + L.T))[-1]))
        Wv = np.array(Wf(a, b_), float)
        eig = min(eig, float(np.linalg.eigvalsh(0.5 * (Wv + Wv.T))[0]))
    return {"max_residual": worst, "min_eig_W": eig,
            "ok": worst <= 1e-6 and eig >= cfg.w_lb - 1e-6}


def certify_ccm(env, key, cfg: SOSCfg, log=print):
    """Bisect on lam; return the largest certified rate and its metric.

    Feasibility is monotone in lam (a metric certifying rate lam certifies every
    smaller one), so the largest feasible lam is well defined.
    """
    err = check_plant(env, key)
    log(f"[sos] {key}: symbolic f/B match the env to {err:.2e}")
    lo = env.X_MIN.cpu().numpy()
    hi = env.X_MAX.cpu().numpy()

    best_w = _feasible_at(key, 0.0, cfg, lo, hi)
    if best_w is None:
        raise ValueError(
            f"{key}: INFEASIBLE even at lam = 0 with deg(W) = {cfg.w_degree}. "
            f"Raise w_degree or widen [w_lb, w_ub].")

    best, low, high = 0.0, 0.0, cfg.lam_hi
    for _ in range(cfg.bisect_iters):
        mid = 0.5 * (low + high)
        w = _feasible_at(key, mid, cfg, lo, hi)
        if w is None:
            high = mid
        else:
            low, best, best_w = mid, mid, w
        log(f"[sos] lam={mid:.4f}  {'feasible' if w else 'infeasible'}"
            f"   bracket [{low:.4f}, {high:.4f}]")

    v = verify(key, best, best_w, cfg, lo, hi, cfg.verify_grid)
    if not v["ok"]:
        raise ValueError(f"{key}: certificate did not verify on a "
                         f"{cfg.verify_grid}^2 grid: {v}")
    return {"lam": best, "coeffs": best_w, "verify": v, "lo": lo, "hi": hi}

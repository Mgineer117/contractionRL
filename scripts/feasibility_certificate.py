"""Constructive CV-STEM feasibility certificate, and the dynamics taxonomy it induces.

    python scripts/feasibility_certificate.py --all --lbd 0.3
    python scripts/feasibility_certificate.py --task classic-cartpole-v0 --lbd 0.1 --verify

No SDP, no bisection. For each sampled state this solves one continuous-time
algebraic Riccati equation and reads the certificate off it, which is why the
answer is a GUARANTEE rather than a search outcome. See
``docs/dynamics_taxonomy.md`` for the definitions and proofs; the short version:

* **Proposition 3 (sufficient).** With ``lbd' = lbd + 1/(2*cm_dt)``, let ``P(x) > 0``
  solve ``(A+lbd'I)^T P + P(A+lbd'I) - (2/r) P B B^T P + q I = 0``. Then
  ``W(x) = P(x)^-1`` satisfies the CV-STEM LMI with margin ``q*w_lb`` under the
  envelope ``w_lb = 1/max_x lmax(P)``, ``w_ub = 1/min_x lmin(P)``. So the program
  is feasible whenever every ``(A(x)+lbd'I, B(x))`` is stabilizable, and the
  envelope is an OUTPUT, not a searched input.
* **Proposition 2 (necessary).** With ``sig = sigma_min([A(x) - sI, B(x)])`` at any
  ``s`` with ``Re s >= -lbd``, feasibility forces
  ``2*(Re s + lbd) + eps <= 2*sig*chi + (2*nu/r)*sig^2``. So ``sig = 0`` (an
  uncontrollable mode at or above the rate) is infeasible at EVERY envelope, and
  a small ``sig`` drives ``nu`` up like ``1/sig^2`` -- at every OTHER state too,
  since ``nu`` is shared.

``rho(x) = lmax(P(x))`` is the per-state metric scale the plant demands. The
classes are DEFINED by lambda*(x) (constant / varying / infeasible), and rho is
the cheap SCREEN for them: rho is only an UPPER bound on what a state demands, so
rho varying is evidence, not proof, that lambda* varies. It agreed with the exact
per-state SDP test on all 9 feasible envs here, but disagrees wildly on
magnitude, so classify with it and never quantify with it.

The control box plays NO part in any of this. Feasibility here is contraction
feasibility only: whether the LMI admits a metric. ``||K||`` is reported so the
deployment cost is visible, but it never disqualifies a plant.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.linalg import solve_continuous_are

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

from contractionRL.agents.skrl.ncm_synthesis import (  # noqa: E402
    drift_jacobians,
    sample_state_box,
)

# rho spreads below this count as "constant" -- CARE solves agree to ~1e-10
# relative, so 1e-3 is four orders of slack over solver noise and still far
# under the 1.65x spread of the loosest class-II plant.
FLAT_TOL = 1e-3


def classic_tasks() -> list[str]:
    from pathlib import Path
    d = Path(_ROOT) / "source/contractionRL/contractionRL/tasks/direct/classic"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "env.py").exists())


def hautus_margin(A: np.ndarray, B: np.ndarray, lbd: float):
    """Proposition 2, computable form: ``min sigma_min([A - sI, B])`` over eigenvalues
    ``s`` of ``A`` with ``Re s >= -lbd``.

    Testing the LMI with the left singular vector ``w`` of the Hautus matrix
    gives, at every such ``s``,

        2*(Re s + lbd) + eps  <=  2*sigma_min*chi + (2*nu/r)*sigma_min^2

    so a VANISHING margin is infeasible at every envelope (class III), and a
    small one forces ``nu`` up like ``1/sigma_min^2`` -- which is why one weak
    state is expensive at all the others, ``nu`` being shared.

    Evaluated at the eigenvalues because that is the only place an
    uncontrollable mode can sit, which makes the test exact and finite. Uses the
    SVD, not eigenvectors, on purpose: a DEFECTIVE ``A`` (pvtol's uncontrollable
    block is nilpotent) returns a near-parallel eigenbasis, and any authority
    computed from it is meaningless.

    Returns ``(margin, s)``.
    """
    s = np.linalg.eigvals(A)
    s = s[s.real >= -lbd]
    if s.size == 0:
        return np.inf, None
    m = [np.linalg.svd(np.hstack([A - si * np.eye(A.shape[0]), B]),
                       compute_uv=False).min() for si in s]
    k = int(np.argmin(m))
    return float(m[k]), complex(s[k])


def certify(A: np.ndarray, B: np.ndarray, lbd: float, r: float, dt: float, q: float):
    """Proposition 3: the CARE certificate at one state, or ``None`` if unstabilizable."""
    n = A.shape[0]
    Abar = A + (lbd + 0.5 / dt) * np.eye(n)
    try:
        P = solve_continuous_are(Abar, B, q * np.eye(n), (r / 2.0) * np.eye(B.shape[1]))
    except Exception:                     # scipy raises when the pair is unstabilizable
        return None
    P = 0.5 * (P + P.T)
    ev = np.linalg.eigvalsh(P)
    if not np.all(np.isfinite(ev)) or ev[0] <= 0:
        return None
    return P


def lmi_residual(A, B, W, lbd, r, dt, nu):
    """max eig of the CV-STEM LMI at ``Wbar = nu*W`` -- the repo's exact expression."""
    Wb = nu * W
    S = ((Wb - np.eye(A.shape[0])) / dt + A @ Wb + Wb @ A.T + 2.0 * lbd * Wb
         - nu * (2.0 / r) * (B @ B.T))
    return float(np.linalg.eigvalsh(0.5 * (S + S.T))[-1])


def analyse(task: str, *, lbd: float, r: float, dt: float, q: float,
            n: int, seed: int, verify: bool) -> dict:
    import gymnasium as gym

    env = gym.make(f"classic-{task}-v0", num_envs=1, device="cpu").unwrapped
    xs = sample_state_box(env.X_MIN, env.X_MAX, n=n, seed=seed)
    A, B = drift_jacobians(env.get_f_and_B, xs, device="cpu")

    hau = None                                   # the weakest Hautus margin in the box
    for k in range(n):
        m, s = hautus_margin(A[k], B[k], lbd)
        if hau is None or m < hau[0]:
            hau = (m, k, s)

    rho, Ps = [], []
    for k in range(n):
        P = certify(A[k], B[k], lbd, r, dt, q)
        if P is None:
            rho.append(np.inf)
            Ps.append(None)
        else:
            rho.append(float(np.linalg.eigvalsh(P)[-1]))
            Ps.append(P)
    rho = np.array(rho)

    out = dict(task=task, x_dim=A.shape[1], u_dim=B.shape[2], rho=rho, hau=hau,
               feasible=bool(np.all(np.isfinite(rho))))

    if not out["feasible"]:
        out["cls"] = "III"
        return out

    w_lb = 1.0 / rho.max()
    w_ub = 1.0 / min(float(np.linalg.eigvalsh(P)[0]) for P in Ps)
    nu = 1.0 / w_lb
    K = np.stack([(1.0 / r) * B[k].T @ Ps[k] for k in range(n)])
    # eps the certificate achieves: -q*W^2 scaled by nu, PLUS the exact -I/dt the
    # (Wbar - I)/dt proxy contributes. The second term is free and usually dominates.
    # ||K|| is REPORTED, never tested: the control box plays no part in contraction
    # feasibility, so no plant is disqualified by it here.
    out.update(w_lb=w_lb, w_ub=w_ub, nu=nu, chi=nu * w_ub, eps_cert=q * w_lb + 1.0 / dt,
               spread=float(rho.max() / rho.min()),
               k_max=float(np.linalg.norm(K, ord=2, axis=(1, 2)).max()))
    out["cls"] = "I" if out["spread"] - 1.0 < FLAT_TOL else "II"

    if verify:
        res = [lmi_residual(A[k], B[k], np.linalg.inv(Ps[k]), lbd, r, dt, nu)
               for k in range(n)]
        out["residual"] = float(np.max(res))
        out["verified"] = out["residual"] <= -out["eps_cert"] * (1 - 1e-6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, help="e.g. classic-car-v0; omit with --all")
    ap.add_argument("--all", action="store_true", help="every classic env")
    ap.add_argument("--lbd", type=float, default=0.3)
    ap.add_argument("--r", type=float, default=1.6)
    ap.add_argument("--cm-dt", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0, help="CARE state weight; sets the certified margin q*w_lb")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="re-evaluate the LMI at the certificate")
    args = ap.parse_args()

    import contractionRL.tasks.direct.classic  # noqa: F401  registers classic envs

    if args.all:
        tasks = classic_tasks()
    elif args.task:
        tasks = [args.task.replace("classic-", "").replace("-v0", "")]
    else:
        ap.error("pass --task or --all")

    hdr = (f"{'env':16s} {'cls':>4s} {'hautus(Prop2)':>13s} {'nu(Prop3)':>10s} "
           f"{'w_lb':>10s} {'w_ub':>10s} {'rho spread':>11s} {'||K||max':>9s}  note")
    print(f"lbd={args.lbd}  r={args.r}  cm_dt={args.cm_dt}  q={args.q}  N={args.n}\n")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for t in tasks:
        try:
            res = analyse(t, lbd=args.lbd, r=args.r, dt=args.cm_dt, q=args.q,
                          n=args.n, seed=args.seed, verify=args.verify)
        except Exception as exc:  # noqa: BLE001 -- a broken env must not hide the rest
            print(f"{t:16s} {'ERR':>4s}  {type(exc).__name__}: {exc}")
            continue
        rows.append(res)
        m, k, s = res["hau"]
        if not res["feasible"]:
            print(f"{t:16s} {'III':>4s} {m:13.3e} {'inf':>10s} {'-':>10s} {'-':>10s} "
                  f"{'-':>11s} {'-':>9s}  uncontrollable mode at s={s.real:+.3f}"
                  f"{s.imag:+.3f}j (sample {k}) -- infeasible at EVERY envelope")
            continue
        note = f"verified: residual {res['residual']:.3g} <= -{res['eps_cert']:.3g}" \
            if "residual" in res and res["verified"] else \
            (f"VERIFY FAILED residual {res['residual']:.3g}" if "residual" in res else "")
        print(f"{t:16s} {res['cls']:>4s} {m:13.3e} {res['nu']:10.4g} "
              f"{res['w_lb']:10.3e} {res['w_ub']:10.3e} {res['spread']:11.4f} "
              f"{res['k_max']:9.4g}  {note}")

    print("\ncls I = rho(x) constant over the box (no subset can raise lbd)")
    print("cls II = rho(x) varies (a subset that drops the argmax raises lbd)")
    print("cls III = some state is not lbd-stabilizable (infeasible at EVERY envelope)")
    print("hautus = min_x min_s sigma_min([A-sI, B]) over Re s >= -lbd (Prop 2); "
          "0 => class III.")
    print("nu/w_lb/w_ub are OUTPUTS of Prop 3 -- the envelope this plant NEEDS, "
          "not one imposed on it.")
    print("||K||max is reported only -- the control box is NOT part of contraction "
          "feasibility here.")
    # --verify is the runnable check on Proposition 3: a certificate that does not
    # satisfy the repo's own (C1) expression at its claimed margin is a bug in
    # the proof or in the construction, so fail loudly rather than printing it.
    bad = [r["task"] for r in rows if "verified" in r and not r["verified"]]
    if bad:
        print(f"\nFAILED verification: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

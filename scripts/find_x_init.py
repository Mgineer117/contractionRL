"""Step 3 of rule.md: choose X_INIT / XREF_INIT, once lambda is known.

Two requirements, and the second is the one that gets forgotten:

1. **Hardest region.** Where the closed-loop contraction rate under the
   certified metric is LOWEST, so the reported number is not flattered by easy
   starts. The local rate at x is the largest ``lam`` satisfying

       A_cl(x)' M(x) + M(x) A_cl(x) + 2*lam*M(x)  <=  0,
       A_cl = A - B K,   K = (1/r) B' M,   M = W^-1

   i.e. ``lam(x) = -0.5 * max eig( A_cl'M + M A_cl , M )`` as a generalized
   eigenproblem. Uniform lambda is the min of this over X; the spread across X
   is what "state-dependent contraction rate" means, and if it is flat then
   rule.md says the box is arbitrary.

2. **Room to contract.** A box touching the edge of X is not a valid spawn
   region no matter how hard it is. The state clamps from step 0, the plant
   stops being f+Bu, and no certificate describes what then runs -- that is
   Rule 1, and it is exactly how segway came to clamp pitch on 100% of steps
   with X_INIT_MAX[pitch] set equal to X_MAX[pitch].

Requirement 2 is checked by ROLLOUT, not by a margin formula: the certified
controller u = uref - K(x)e is run from each candidate box and the realised
clamp fraction is measured. A candidate is admissible only if it stays off the
walls in practice.

    python scripts/find_x_init.py --task classic-segway-v0
    python scripts/find_x_init.py --task classic-segway-v0 --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import scipy.linalg as sla
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

import contractionRL.tasks.direct.classic  # noqa: E402,F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from contractionRL.agents.skrl.math_utils import jacobian  # noqa: E402


def local_rates(env, x_np, W, r):
    """lam(x) for each sampled state, under the certified metric."""
    x = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_()
    with torch.enable_grad():
        f, B, _ = env.get_f_and_B(x)
    A = jacobian(f, x, create_graph=False).detach().numpy().astype(np.float64)
    Bn = B.detach().numpy().astype(np.float64)
    lam = np.empty(len(x_np))
    for k in range(len(x_np)):
        M = np.linalg.inv(W[k])
        Acl = A[k] - Bn[k] @ ((1.0 / r) * Bn[k].T @ M)
        S = Acl.T @ M + M @ Acl
        # generalized eigenproblem (S, M): S <= -2 lam M  <=>  lam <= -0.5 max eig
        ev = sla.eigh(0.5 * (S + S.T), M, eigvals_only=True)
        lam[k] = -0.5 * ev[-1]
    return lam


def rollout_clamp(env, box_lo, box_hi, ref_lo, ref_hi, W, x_np, r, steps=400, n=128):
    """Run the certified controller from a candidate box; report clamp + error."""
    env.X_INIT_MIN = torch.as_tensor(box_lo, dtype=torch.float32)
    env.X_INIT_MAX = torch.as_tensor(box_hi, dtype=torch.float32)
    env.XREF_INIT_MIN = torch.as_tensor(ref_lo, dtype=torch.float32)
    env.XREF_INIT_MAX = torch.as_tensor(ref_hi, dtype=torch.float32)
    env.reset()
    env.clamp_summary()  # zero the accumulators
    tree = torch.as_tensor(x_np, dtype=torch.float32)
    e0 = None
    errs = []
    for t in range(steps):
        xt = env.x_t
        # nearest cached state -> its W (the deployed metric is the regressed CMG,
        # but the certified W is what the rate claim is about)
        d = torch.cdist(xt, tree)
        idx = d.argmin(dim=-1).numpy()
        xr = env.xref[:, min(t, env.xref.shape[1] - 1)]
        ur = env.uref[:, min(t, env.uref.shape[1] - 1)].numpy()
        e = env.wrap_angles(xt - xr).numpy().astype(np.float64)
        with torch.no_grad():
            _, B, _ = env.get_f_and_B(xt)
        Bn = B.numpy().astype(np.float64)
        u = np.empty_like(ur)
        for i in range(len(e)):
            M = np.linalg.inv(W[idx[i]])
            u[i] = ur[i] - ((1.0 / r) * Bn[i].T @ M) @ e[i]
        nrm = np.linalg.norm(e, axis=-1)
        if e0 is None:
            e0 = nrm.copy()
        errs.append(nrm)
        env.step(u.astype(np.float32))
    cs = env.clamp_summary(reset=False) or {}
    errs = np.array(errs)
    ratio = errs[-1] / np.maximum(e0, 1e-9)
    return cs, float(np.mean(ratio)), float(errs[0].mean()), float(errs[-1].mean())


# Fraction of |X| the spawn box must leave free on every side. Mirrors
# MIN_HEADROOM_FRAC in tests/test_rule_step3_spawn_box.py -- kept a touch
# above it so a box this tool emits is not marginal against the test.
MIN_HEADROOM_FRAC = 0.06


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--algorithm", default="c2rl-ppo")
    p.add_argument("--pctl", type=float, default=15.0,
                   help="keep states below this percentile of lam(x) as the hard region")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--synth-n", "--synth_n", type=int, default=600,
                   help="when no dataset exists, solve the metric at this N")
    p.add_argument("--no-move-xref", "--no_move_xref", dest="move_xref",
                   action="store_false", default=True,
                   help="keep XREF_INIT where it is instead of co-locating it with "
                        "X_INIT. Only correct for an env whose reference is pinned "
                        "(segway holds XREF_INIT at zero: a tilted reference is not "
                        "sustainable for an inverted pendulum).")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    short = args.task.removeprefix("classic-").removesuffix("-v0").removesuffix("-v1")
    if args.task == "classic-car-v1":
        short = "car_v1"
    cfg_path = os.path.join(_ROOT, "source/contractionRL/contractionRL/tasks/direct/classic",
                            short, "agents", f"skrl_{args.algorithm.replace('-','_')}_cfg.yaml")
    with open(cfg_path) as fh:
        cm = yaml.safe_load(fh)["cm"]
    r = float(cm["cvstem_r_scaler"]) + 1e-5

    env = gym.make(args.task, num_envs=128, device="cpu").unwrapped
    npz = sorted(glob.glob(os.path.join(_ROOT, f"data/classic/{short}/*.npz")))
    if npz:
        d = np.load(npz[0])
        x_np, W = np.asarray(d["x"], np.float64), np.asarray(d["W"], np.float64)
        print(f"metric: shipped dataset {os.path.basename(npz[0])} ({len(x_np)} states)")
    else:
        # Step 3 asks WHERE the rate is low and WHICH dims drive it. That is a
        # property of the metric family, not of the sample count, so a small
        # solve answers it in minutes instead of the ~15 h the shipped N=10000
        # build costs. The absolute rates below are therefore indicative; the
        # box they pick still gets rollout-verified against the real env.
        from contractionRL.agents.skrl.ncm_synthesis import (  # noqa: PLC0415
            cvstem_joint,
            drift_jacobians,
            sample_state_box,
        )
        n = args.synth_n
        print(f"metric: no shipped dataset — solving a joint SDP at N={n} for the "
              f"rate STRUCTURE only (absolute rates indicative, box is rollout-verified)")
        x_np = sample_state_box(env.X_MIN, env.X_MAX, n=n, seed=0).astype(np.float64)
        A0, B0 = drift_jacobians(env.get_f_and_B, x_np, device="cpu")
        # eps is a covering radius scheduled by sample count -- hardcoding 0.1 at
        # N=600 solves a STRICTER program than the one that certified the lambda
        # (the search uses 0.05 there), and reports a false infeasible. tora did
        # exactly that before this line read eps_for_n.
        # The SHIPPED cm_eps, not the search schedule. Step 3 analyses the metric
        # the env actually deploys, and eps_for_n(600)=0.05 is five times stricter
        # than the 0.01 the configs generate at -- strict enough to report tora
        # infeasible at a lambda its own dataset builds fine at.
        _eps = float(cm.get("cm_eps", 0.01))
        sol = cvstem_joint(np.asarray(A0, np.float64), np.asarray(B0, np.float64),
                           lbd=float(cm["lbd"]), eps=_eps, dt=1.0, solver=cm.get("cm_solver", "MOSEK"),
                           r_scaler=float(cm["cvstem_r_scaler"]),
                           w_lb=float(cm["w_lb"]), w_ub=float(cm["w_ub"]))
        if sol is None:
            print(f"{short}: joint SDP infeasible at the config's lbd — Step 3 cannot "
                  f"run until Step 1 is settled for this env.", file=sys.stderr)
            return 2
        W = np.asarray(sol["W"], np.float64)

    names = list(getattr(env, "state_names", None) or
                 [f"x{i}" for i in range(int(env.num_dim_x))])
    X_LO, X_HI = env.X_MIN.numpy().copy(), env.X_MAX.numpy().copy()

    print(f"=== {args.task}  lbd(cfg)={cm['lbd']}  r={r-1e-5}  N={len(x_np)}")
    lam = local_rates(env, x_np, W, r)
    print(f"local closed-loop rate lam(x) over X: min {lam.min():.4f}  p05 "
          f"{np.percentile(lam,5):.4f}  median {np.median(lam):.4f}  max {lam.max():.4f}")
    spread = lam.max() / max(lam.min(), 1e-9)
    verdict = ("STATE-DEPENDENT: X_INIT belongs in the low-rate region"
               if spread > 2 else
               "FLAT: X_INIT is arbitrary within X (rule.md Step 3)")
    print(f"spread max/min = {spread:.2f}x  ->  {verdict}")

    hard = x_np[lam <= np.percentile(lam, args.pctl)]
    print(f"\nhardest {args.pctl:g}% region ({len(hard)} states):")
    for i, nm in enumerate(names):
        print(f"   {nm:<12} [{hard[:,i].min():+.4f}, {hard[:,i].max():+.4f}]   "
              f"X = [{X_LO[i]:+.4f}, {X_HI[i]:+.4f}]")

    # Which dims actually drive hardness? Scaling the bounding box about the
    # ORIGIN is wrong whenever a dim is hard at LARGE |x| and the box is
    # symmetric: shrinking segway's pitch [-0.9, 0.9] gives [-0.36, 0.36], which
    # is the EASY region, the exact opposite of Step 3's intent. So shrink the
    # WALL MARGIN instead, and only for the dims that matter.
    # Both polarities matter. Only the first was handled, which silently sent
    # car_v1 -- the flagship state-dependent env -- down the origin-shrinking
    # fallback even though corr(lam, |vel|) = +0.804: its Hautus margin is
    # sigma = min(1, v), so authority VANISHES as v -> 0 and the hard region is
    # vel in [0.20, 0.46] of a [0.20, 2.00] box. Scaling about the origin cannot
    # express that, and nothing said so.
    drives, drives_small = [], []
    for i in range(len(names)):
        c = np.corrcoef(lam, np.abs(x_np[:, i]))[0, 1]
        tag = ""
        if c < -0.05:                      # harder as |x| grows
            drives.append(i)
            tag = "   <- drives hardness (hard at LARGE |x|)"
        elif c > 0.05:                     # harder as |x| shrinks
            drives_small.append(i)
            tag = "   <- drives hardness (hard at SMALL |x|)"
        print(f"   corr(lam, |{names[i]}|) = {c:+.3f}{tag}")
    if not drives and not drives_small:
        print("   no dim drives hardness; falling back to shrinking about the origin")

    print(f"\ncandidate X_INIT boxes (certified controller, {args.steps} steps, 128 envs):")
    print(f"{'margin':>7} {'clamp_any':>10} {'worst dim':>22} {'e(T)/e(0)':>10} {'e(0)':>8} {'e(T)':>8}")
    best = None
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        lo = np.array([hard[:, i].min() for i in range(len(names))])
        hi = np.array([hard[:, i].max() for i in range(len(names))])
        # EVERY dim gets wall headroom -- the transient overshoots in dims that do
        # not drive hardness at all (segway clamps vel_x_b 15% of steps while its
        # difficulty lives in pitch), and a clamped dim voids the certificate just
        # the same. Driving dims additionally keep their large-|x| character, so
        # headroom does not quietly relocate the spawn to the easy region.
        for i in range(len(names)):
            half = max(abs(X_LO[i]), abs(X_HI[i]))
            if i in drives:
                q = np.percentile(np.abs(hard[:, i]), 40)
                lo[i], hi[i] = q, half * shrink
                if lo[i] >= hi[i]:
                    lo[i] = 0.5 * hi[i]      # sign dims supply both polarities
            elif i in drives_small:
                # Keep the low-|x| end and cap the high end at the hard region's
                # own spread; shrink narrows TOWARD the hard end rather than
                # toward the origin, which here is the easy direction.
                q = float(np.percentile(np.abs(hard[:, i]), 60))
                lo[i] = float(hard[:, i].min())
                span = float(X_HI[i] - X_LO[i])
                hi[i] = lo[i] + shrink * max(q - lo[i], 0.1 * span)
            else:
                lo[i], hi[i] = -half * shrink, half * shrink
        lo, hi = np.maximum(lo, X_LO), np.minimum(hi, X_HI)
        # Never emit a box that spawns ON the wall. The ladder's top rung is
        # shrink=1.0, which sets hi = half exactly -- 0% headroom -- and the
        # np.minimum above re-pins any asymmetric dim back to X_HI, so both paths
        # can hand back a spawn flush against the box. That is the segway bug
        # (X_INIT[pitch] == X_MAX[pitch]): a state clamped from step 0 makes the
        # plant something other than f+Bu and no certificate describes it.
        # Capped against exactly what tests/test_rule_step3_spawn_box.py measures,
        # reach = max(|lo|,|hi|) against half, so the tool cannot emit a box its
        # own rule rejects.
        # Two-sided, against the ACTUAL bounds. The old form clipped to
        # +-half*(1-frac), which is identical for a symmetric box but leaves an
        # all-positive dim flush against its LOWER wall -- car_v1's vel starts at
        # 0.20 and its hard region starts there too, so the box the tool would
        # emit spawned exactly on the wall it is meant to clear.
        pad = np.array([MIN_HEADROOM_FRAC * max(abs(X_LO[i]), abs(X_HI[i]))
                        for i in range(len(names))])
        lo = np.clip(lo, X_LO + pad, X_HI - pad)
        hi = np.clip(hi, X_LO + pad, X_HI - pad)
        # XREF_INIT moves WITH X_INIT, which is what rule.md Step 3 says ("XREF_INIT
        # is established the same way, so the reference also starts in the low-rate
        # region") and is not optional. Moving X_INIT alone does not make the
        # operating point harder, it makes e(0) = x_0 - xref_0 enormous: on car and
        # car_v1 that produced e(0) of 20-35 against the ~1.0 their XE_INIT gives,
        # a different problem rather than a harder one, and nothing contracted.
        # Co-locating them keeps e(0) at roughly the box width while still putting
        # both endpoints in the slow region.
        if args.move_xref:
            ref_lo, ref_hi = lo, hi
        else:
            ref_lo = env.XREF_INIT_MIN.numpy()
            ref_hi = env.XREF_INIT_MAX.numpy()
        cs, ratio, ei, ef = rollout_clamp(env, lo, hi, ref_lo, ref_hi, W, x_np, r,
                                          steps=args.steps)
        fa = cs.get("frac_any", float("nan"))
        worst = max(((k, v) for k, v in cs.items() if k.startswith("frac_")
                     and k not in ("frac_any", "frac_u_saturated", "frac_nonfinite")),
                    key=lambda kv: kv[1], default=("-", 0.0))
        print(f"{shrink:>7.2f} {fa:>10.3f} {worst[0]+'='+format(worst[1],'.3f'):>22} "
              f"{ratio:>10.3f} {ei:>8.3f} {ef:>8.3f}")
        # admissible: essentially never on a wall, and the error actually shrinks
        if best is None and fa < 0.02 and ratio < 1.0:
            best = (shrink, lo.copy(), hi.copy(), ratio, fa)

    if best is None:
        print("\nNo candidate stayed off the walls while contracting. Either the "
              "hard region is unreachable without clamping (shrink X, rule.md Step 1) "
              "or the metric does not actually contract here.")
        return 1
    shrink, lo, hi, ratio, fa = best
    print(f"\nCHOSEN X_INIT (largest admissible = hardest): shrink {shrink:g}, "
          f"clamp {fa:.3f}, e(T)/e(0) {ratio:.3f}")
    print(f"  X_INIT_MIN = [{', '.join(f'{v:.4f}' for v in lo)}]")
    print(f"  X_INIT_MAX = [{', '.join(f'{v:.4f}' for v in hi)}]")
    if args.move_xref:
        print(f"  XREF_INIT_MIN = [{', '.join(f'{v:.4f}' for v in lo)}]   (co-located, Step 3)")
        print(f"  XREF_INIT_MAX = [{', '.join(f'{v:.4f}' for v in hi)}]")
    if args.apply:
        print("\n--apply is not wired: X_INIT lives in env.py next to prose that "
              "explains it, so paste the two lines above with a note on why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

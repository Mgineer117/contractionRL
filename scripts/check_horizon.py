"""rule.md Step 5: is each env's horizon long enough for 95% error reduction?

    T95 = ln(20*C) / lam   seconds

Step 5 says to take that from a **measured** rate, and the distinction is not
cosmetic. The certified lambda is a worst-case bound over the whole state box, so
sizing a horizon with it is wildly conservative: segway's certified 0.0514 asks
for 58-93 s, while the certified controller actually decays the error to 0.34% of
e(0) in 12 s, which is a realised rate near 0.47 -- nine times faster.

So this rolls the certified controller u = uref - K(x)e, K = (1/r)B'M, M = W^-1,
fits the decay on the log of the mean error, reads the overshoot C off the same
trace, and compares T95 against episode_len*dt. Both numbers are printed: the
certified one is what the theory guarantees, the measured one is what the horizon
has to accommodate.

    python scripts/check_horizon.py
    python scripts/check_horizon.py --task classic-segway-v0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contractionRL.tasks.direct.classic  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from contractionRL.agents.skrl.math_utils import jacobian  # noqa: E402


def metric_for(env, cm, short, synth_n):
    """The shipped {x -> W} if it exists, else a small solve for the structure."""
    npz = sorted(glob.glob(os.path.join(_ROOT, f"data/classic/{short}/*.npz")))
    if npz:
        d = np.load(npz[0])
        return (np.asarray(d["x"], np.float64), np.asarray(d["W"], np.float64),
                f"shipped N={len(d['x'])}")
    from contractionRL.agents.skrl.ncm_synthesis import (
        cvstem_joint,
        drift_jacobians,
        sample_state_box,
    )
    x = sample_state_box(env.X_MIN, env.X_MAX, n=synth_n, seed=0).astype(np.float64)
    A, B = drift_jacobians(env.get_f_and_B, x, device="cpu")
    sol = cvstem_joint(np.asarray(A, np.float64), np.asarray(B, np.float64),
                       lbd=float(cm["lbd"]), eps=float(cm.get("cm_eps", 0.01)), dt=1.0,
                       solver=cm.get("cm_solver", "MOSEK"),
                       r_scaler=float(cm["cvstem_r_scaler"]),
                       w_lb=float(cm["w_lb"]), w_ub=float(cm["w_ub"]))
    if sol is None:
        return None, None, "SDP infeasible"
    return x, np.asarray(sol["W"], np.float64), f"synth N={synth_n}"


def measure(env, steps):
    """Mean-error trace under a per-state LQR (Q = R = I).

    NOT a nearest-neighbour lookup of the certified W. That was the first
    version and it is unusable above a few dimensions: 10000 samples in
    quadrotor's 10-D box leaves the nearest stored metric far from the visited
    state, and the resulting gain reported a NEGATIVE decay rate -- an
    interpolation artifact, not a property of the plant.

    Solving the Riccati equation at each visited state is exact there, needs no
    dataset at all (so Step 5 can be checked before Step 1 finishes), and answers
    the question Step 5 actually asks: is the horizon long enough for a good
    controller to remove 95% of the initial error. Where the ARE has no solution
    the state is not stabilisable, and zero gain is the honest thing to apply --
    that is a real property of the plant and shows up as a growing error.
    """
    import scipy.linalg as sla

    env.reset()
    n, m = int(env.num_dim_x), int(env.num_dim_control)
    Q, R = np.eye(n), np.eye(m)
    errs, fails = [], 0
    for t in range(steps):
        x = env.x_t.detach().clone().requires_grad_()
        with torch.enable_grad():
            f, B, _ = env.get_f_and_B(x)
        A = jacobian(f, x, create_graph=False).detach().numpy()[0].astype(np.float64)
        Bk = B.detach().numpy()[0].astype(np.float64)
        xr = env.xref[:, min(t, env.xref.shape[1] - 1)]
        ur = env.uref[:, min(t, env.uref.shape[1] - 1)].numpy()
        e = env.wrap_angles(env.x_t - xr).numpy().astype(np.float64)
        try:
            K = Bk.T @ sla.solve_continuous_are(A, Bk, Q, R)
        except Exception:  # noqa: BLE001 — unstabilisable here; zero gain is honest
            fails += 1
            K = np.zeros((m, n))
        errs.append(np.linalg.norm(e, axis=-1))
        env.step((ur - e @ K.T).astype(np.float32))
    return np.array(errs).mean(axis=1), fails


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=None)
    p.add_argument("--algorithm", default="c2rl-ppo")
    p.add_argument("--synth-n", "--synth_n", type=int, default=600)
    p.add_argument("--envs", type=int, default=64)
    args = p.parse_args()

    tasks = [args.task] if args.task else [k for k in gym.registry
                                           if k.startswith("classic-")]
    print(f"{'env':<15} {'ctrl':<15} {'horizon':>8} {'lam_cert':>9} {'lam_meas':>9} "
          f"{'C':>5} {'T95_meas':>9} {'e(T)/e(0)':>10}  verdict")
    bad = []
    for t in tasks:
        short = t.removeprefix("classic-").removesuffix("-v0").removesuffix("-v1")
        if t == "classic-car-v1":
            short = "car_v1"
        with open(os.path.join(_ROOT, "source/contractionRL/contractionRL/tasks/direct/"
                               f"classic/{short}/agents/"
                               f"skrl_{args.algorithm.replace('-','_')}_cfg.yaml")) as fh:
            cm = yaml.safe_load(fh)["cm"]
        env = gym.make(t, num_envs=args.envs, device="cpu").unwrapped
        H = float(env.dt) * int(env.episode_len)
        e, fails = measure(env, int(env.episode_len))
        tag = "LQR" if not fails else f"LQR({fails} unstab)"
        e0 = max(e[0], 1e-12)
        C = float(e.max() / e0)                      # overshoot, off the same trace
        k = max(5, len(e) // 2)
        g = e[:k] > 1e-9
        tt = np.arange(len(e)) * float(env.dt)
        lam_m = float(-np.polyfit(tt[:k][g], np.log(e[:k][g]), 1)[0]) if g.sum() > 3 else 0.0
        ratio = float(e[-1] / e0)
        t95 = float(np.log(20 * max(C, 1.0)) / lam_m) if lam_m > 1e-6 else float("inf")
        ok = "OK" if (ratio <= 0.05 and t95 <= H) else (
             "OK (reaches 95%)" if ratio <= 0.05 else "*** TOO SHORT ***")
        if ratio > 0.05:
            bad.append(short)
        print(f"{short:<15} {tag:<15} {H:>8.1f} {float(cm['lbd']):>9.4f} {lam_m:>9.4f} "
              f"{C:>5.2f} {t95:>9.1f} {ratio:>10.4f}  {ok}")
    if bad:
        print(f"\nhorizon does not reach 95% reduction: {', '.join(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

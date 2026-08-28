"""Is each env's exploration sigma proportionate to the feedback its plant needs?

PPO explores by sampling around the policy mean with standard deviation
``sigma = exp(initial_log_std)``. That number is only meaningful RELATIVE to the
control the plant actually requires: the same sigma that is generous on a car is
a rounding error on a segway.

Every classic env ships ``initial_log_std: -0.5`` (sigma = 0.607), and measured
against the feedback the certified controller needs:

    car        |pi| p95 = 0.856   sigma/p95 = 0.71    fine
    car_v1     |pi| p95 = 0.591   sigma/p95 = 1.03    fine
    segway     |pi| p95 = 6.952   sigma/p95 = 0.087   11x too small
    quadrotor  |pi| p95 = 5.242   sigma/p95 = 0.116    9x too small

so the constant was tuned where it happened to fit and copied where it did not.
On segway the consequence is not subtle: the policy has to discover a ~7-unit
stabilising feedback by sampling 0.6-unit perturbations, from a state that clamps
on step 1 and therefore returns an almost flat reward. Raising it to sigma = 2.0
halved the AUC (167.2 -> 77.6) in the same 20k steps.

The reference is the certified controller's own feedback ``-K(x)e``, with
``K = (1/r) B' M`` -- i.e. what a controller that provably works has to output on
this env's own initial conditions.

    python scripts/check_exploration_scale.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

import contractionRL.tasks.direct.classic  # noqa: E402,F401
import gymnasium as gym  # noqa: E402

# sigma should land within a small factor of the feedback the plant needs. Below
# ~0.2 the policy cannot reach the useful range by sampling; far above ~2 it is
# mostly injecting noise the critic has to average away.
LO, HI = 0.2, 2.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", default="c2rl-ppo")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--envs", type=int, default=128)
    args = p.parse_args()

    print(f"{'env':<14} {'sigma':>7} {'|pi| p50':>9} {'|pi| p95':>9} {'ratio':>7} "
          f"{'suggest':>9}  reading")
    bad = []
    for t in [k for k in gym.registry if k.startswith("classic-")]:
        short = t.removeprefix("classic-").removesuffix("-v0").removesuffix("-v1")
        if t == "classic-car-v1":
            short = "car_v1"
        cfg_path = os.path.join(
            _ROOT, "source/contractionRL/contractionRL/tasks/direct/classic", short,
            "agents", f"skrl_{args.algorithm.replace('-', '_')}_cfg.yaml")
        with open(cfg_path) as fh:
            y = yaml.safe_load(fh)
        sigma = math.exp(float(y["models"]["policy"].get("initial_log_std", 0.0)))
        r = float(y["cm"]["cvstem_r_scaler"]) + 1e-5
        npz = sorted(glob.glob(os.path.join(_ROOT, f"data/classic/{short}/*.npz")))
        if not npz:
            print(f"{short:<14} {sigma:>7.3f}        --        --      --        --  "
                  f"(no certified metric yet)")
            continue
        d = np.load(npz[0])
        xs = np.asarray(d["x"], np.float64)
        W = np.asarray(d["W"], np.float64)
        env = gym.make(t, num_envs=args.envs, device="cpu").unwrapped
        env.reset()
        tree = torch.as_tensor(xs, dtype=torch.float32)
        mags = []
        for k in range(args.steps):
            xt = env.x_t
            idx = torch.cdist(xt, tree).argmin(dim=-1).numpy()
            xr = env.xref[:, min(k, env.xref.shape[1] - 1)]
            ur = env.uref[:, min(k, env.uref.shape[1] - 1)].numpy()
            e = env.wrap_angles(xt - xr).numpy().astype(np.float64)
            with torch.no_grad():
                _, B, _ = env.get_f_and_B(xt)
            Bn = B.numpy().astype(np.float64)
            fb = np.array([-((1.0 / r) * Bn[i].T @ np.linalg.inv(W[idx[i]])) @ e[i]
                           for i in range(len(e))])
            mags.append(np.abs(fb))
            env.step((ur + fb).astype(np.float32))
        m = np.concatenate(mags)
        p50, p95 = float(np.percentile(m, 50)), float(np.percentile(m, 95))
        ratio = sigma / max(p95, 1e-9)
        ok = LO <= ratio <= HI
        if not ok:
            bad.append(short)
        print(f"{short:<14} {sigma:>7.3f} {p50:>9.3f} {p95:>9.3f} {ratio:>7.3f} "
              f"{math.log(max(p95, 1e-6)):>9.2f}  "
              f"{'OK' if ok else ('SIGMA TOO SMALL' if ratio < LO else 'sigma large')}")
    if bad:
        print(f"\nsigma out of proportion on: {', '.join(bad)} — the 'suggest' column "
              f"is initial_log_std for sigma = p95.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

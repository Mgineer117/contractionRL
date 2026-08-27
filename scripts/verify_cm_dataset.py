"""Check that a generated CM dataset certifies what its config claims.

``build_cm_dataset`` already recomputes the joint LMI residual before it writes,
but that is the solver checking its own algebra: same expression, same W̄, same
ν. It cannot catch a dataset that was written under one config and is now loaded
under another, and it never checks the thing the certificate is actually FOR --
that the closed loop contracts.

So this recomputes all three from the stored file and the yaml, independently:

  (1) joint LMI      (W̄-I)/dt + AW̄ + W̄Aᵀ + 2λW̄ - ν(2/r)BBᵀ  ⪯ 0,  W̄ = νW
  (2) closed loop    A_clᵀM + M A_cl + 2λM ⪯ 0,  A_cl = A - BK,
                     K = (1/r)BᵀM,  M = W⁻¹      <-- the actual claim
  (3) envelope       w_lb·I ⪯ W ⪯ w_ub·I

(2) is implied by (1) whenever W̄ ⪰ I, since the proxy adds a PSD term, but it is
the statement anything downstream relies on and it is one eigvalsh to check.

ν is recovered from the file as max_k 1/λ_min(W_k): cvstem_joint returns W = W̄/ν
with W̄ ⪰ I tight at the binding state, and ν is not stored.

    python scripts/verify_cm_dataset.py                    # every env with a dataset
    python scripts/verify_cm_dataset.py --task classic-car-v0
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

import contractionRL.tasks.direct.classic  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from contractionRL.agents.skrl.math_utils import jacobian  # noqa: E402


def verify(short: str, task: str, algorithm: str) -> bool:
    paths = sorted(glob.glob(os.path.join(_ROOT, f"data/classic/{short}/*.npz")))
    if not paths:
        print(f"{short:10s} no dataset — nothing to verify "
              f"(scripts/build_cm_dataset.py --task {task})")
        return True
    cfg_path = os.path.join(
        _ROOT, "source/contractionRL/contractionRL/tasks/direct/classic",
        short, "agents", f"skrl_{algorithm.replace('-', '_')}_cfg.yaml")
    with open(cfg_path) as fh:
        cm = yaml.safe_load(fh)["cm"]
    lbd, r = float(cm["lbd"]), float(cm["cvstem_r_scaler"]) + 1e-5
    w_lb, w_ub = float(cm["w_lb"]), float(cm["w_ub"])

    d = np.load(paths[0])
    x_np = np.asarray(d["x"], dtype=np.float64)
    W = np.asarray(d["W"], dtype=np.float64)
    # W̄ = νW with W̄ ⪰ I tight somewhere, so ν = max_k 1/λ_min(W_k).
    nu = 1.0 / np.linalg.eigvalsh(W).min(axis=1).max()

    env = gym.make(task, num_envs=1, device="cpu").unwrapped
    x = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_()
    with torch.enable_grad():
        f, B, _ = env.get_f_and_B(x)
    A = jacobian(f, x, create_graph=False).detach().numpy().astype(np.float64)
    Bn = B.detach().numpy().astype(np.float64)

    n, x_dim = W.shape[0], W.shape[1]
    eye = np.eye(x_dim)
    lmi = np.empty(n)
    clo = np.empty(n)
    for k in range(n):
        Wb = nu * W[k]
        S = ((Wb - eye) + A[k] @ Wb + Wb @ A[k].T + 2.0 * lbd * Wb
             - nu * (2.0 / r) * (Bn[k] @ Bn[k].T))          # dt = 1.0, always
        lmi[k] = np.linalg.eigvalsh(0.5 * (S + S.T))[-1]
        M = np.linalg.inv(W[k])
        Acl = A[k] - Bn[k] @ ((1.0 / r) * Bn[k].T @ M)
        T = Acl.T @ M + M @ Acl + 2.0 * lbd * M
        clo[k] = np.linalg.eigvalsh(0.5 * (T + T.T))[-1]
    ev = np.linalg.eigvalsh(W)
    lo, hi = float(ev[:, 0].min()), float(ev[:, -1].max())

    ok = [lmi.max() < 0.0, clo.max() < 0.0, lo >= w_lb - 1e-9 and hi <= w_ub + 1e-9]
    print(f"{short:10s} N={n:<6d} lbd={lbd:<8g} r={r - 1e-5:<6g} nu={nu:.4g}  "
          f"{os.path.relpath(paths[0], _ROOT)}")
    print(f"           joint LMI   max {lmi.max():+.3e}  {'PASS' if ok[0] else '*** FAIL ***'}")
    print(f"           closed loop max {clo.max():+.3e}  {'PASS' if ok[1] else '*** FAIL ***'}")
    print(f"           envelope    eig(W) [{lo:.4g}, {hi:.4g}] vs [{w_lb:g}, {w_ub:g}]  "
          f"{'PASS' if ok[2] else '*** FAIL ***'}"
          + ("   <- pinned at w_lb; the gain cap is binding" if abs(lo - w_lb) < 1e-9 else ""))
    if len(paths) > 1:
        print(f"           NOTE: {len(paths)} datasets present, verified only the first. "
              f"Stale ones key-miss at load and are dead weight.")
    return all(ok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=None, help="one task id; default is every "
                                                "classic env that has a dataset")
    p.add_argument("--algorithm", default="c2rl-ppo")
    args = p.parse_args()

    tasks = ([args.task] if args.task else
             [t for t in gym.registry if t.startswith("classic-")])
    bad = []
    for t in tasks:
        short = t.removeprefix("classic-").removesuffix("-v0").removesuffix("-v1")
        if t == "classic-car-v1":
            short = "car_v1"
        if not verify(short, t, args.algorithm):
            bad.append(short)
    if bad:
        print(f"\nFAILED: {', '.join(bad)} — the stored metric does not certify the "
              f"config's lambda. Do not train on it.", file=sys.stderr)
        return 1
    print("\nall verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

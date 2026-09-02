"""Solve the GLOBAL optimum of the C2RL objective, offline, before training.

Writes ``data/toy/<key>/vstar.npz``: the reference trajectories the run will
replay, and V*_0 on a state grid for each of them. Training loads the same file,
so the policy is measured against the optimum of exactly the problem it faced.

Why it is computable here and nowhere else
------------------------------------------
With the reference FIXED, tracking is a finite-horizon time-varying MDP whose
state is x alone -- time enters only through which reference point is current.
So V* comes from ONE backward sweep

    V*_T(x) = 0
    V*_t(x) = min_u [ c_t(x,u) + gamma V*_{t+1}(x') ]

with c_t the C2RL reward negated, at C2RL's own discount, against the same
metric C2RL trains on. No fixed-point iteration, no tolerance, no contraction
argument: exact for the discretised MDP. The only error is discretisation, and
--richardson measures its order instead of assuming it.

The 25-way grouping is what makes this affordable: cost is one sweep per
REFERENCE, so 25 references at 625 envs is 25 sweeps, where one reference per
env slot would be 625.

    python scripts/precompute_vstar.py --task toy-mg-v0
    python scripts/precompute_vstar.py --task toy-mg-v0 --richardson
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))

import contractionRL.tasks.direct.toy as toy  # noqa: E402
import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from contractionRL import cm_data  # noqa: E402
from contractionRL.cm_data import find_npz  # noqa: E402
from contractionRL.solvers import TrackingVI  # noqa: E402


def _cfg(task: str, algorithm: str) -> dict:
    entry = (gym.spec(task).kwargs or {}).get(f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point")
    if entry is None:
        raise SystemExit(f"{task} registers no '{algorithm}' config.")
    pkg, fname = entry.split(":")
    with open(os.path.join(os.path.dirname(importlib.import_module(pkg).__file__), fname)) as fh:
        return yaml.safe_load(fh)


def _key(task: str) -> str:
    """Short env name. Toy and classic both -- the solver is the same."""
    for k in toy.TOY_ENVS:
        if toy.env_id(k) == task:
            return k
    return task.removeprefix("classic-").removesuffix("-v0").replace("-v1", "_v1")


def _attach_metric(env, key: str, cm: dict):
    """The SAME metric and reward weights C2RL trains against. See cm_data."""
    info = cm_data.attach_metric(
        env, key, tag=f"[vstar:{key}]",
        **{k: cm[k] for k in cm_data.REWARD_KEYS if k in cm})
    return float(info["lbd"])


def _sweep_depth(gamma: float, T: int, tol: float = 1e-15) -> int:
    """How far back the sweep has to run before gamma^k stops mattering.

    V*_0 depends on V*_k only through gamma^k, so at C2RL's gamma = 0.01 the
    influence is under double epsilon by k = 8 and the remaining ~490 steps of a
    classic episode contribute nothing. Truncating there is exact to machine
    precision and is what makes a 4-state env affordable at all -- the full
    horizon would be ~50x the work for no change in the answer. The realised
    return the policy is scored against truncates identically.
    """
    if gamma <= 0:
        return 1
    if gamma >= 1:
        return T
    import math
    return min(T, max(1, int(math.ceil(math.log(tol) / math.log(gamma))) + 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="toy-mg-v0 | toy-duff-v0")
    p.add_argument("--algorithm", default="c2rl-ppo",
                   help="whose discount and reward weights V* is solved under")
    p.add_argument("--groups", type=int, default=25, help="distinct reference trajectories")
    p.add_argument("--envs-per-group", "--envs_per_group", dest="epg", type=int, default=25)
    p.add_argument("--n", type=int, default=129, help="grid points per state dim")
    p.add_argument("--actions", type=int, default=81, help="grid points per input dim")
    p.add_argument("--richardson", action="store_true",
                   help="also solve at n/2 and n/4 and report the observed order")
    p.add_argument("--ref-offset", "--ref_offset", dest="ref_offset", type=int, default=1)
    p.add_argument("--horizon-tol", "--horizon_tol", dest="horizon_tol", type=float,
                   default=1e-15, help="truncate the sweep once gamma^k drops below this")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    key = _key(a.task)
    cfg = _cfg(a.task, a.algorithm)
    gamma = float(cfg["agent"]["discount_factor"])
    cm = cfg.get("cm", {})
    num_envs = a.groups * a.epg

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    env = gym.make(a.task, num_envs=num_envs, device="cpu",
                   fix_ref_trajectories=True, ref_groups=a.groups).unwrapped
    # Same window train.py builds. It sets ref_gen_len, hence the SHAPE of the
    # stored reference, so a pack solved under a different window either fails to
    # install (100 steps vs 199) or -- worse, if the shapes coincide -- installs
    # a reference the agent never sees.
    env.configure_ref_window(length=env.full_ref_length(a.ref_offset),
                             offset=a.ref_offset, gamma=gamma)
    lbd = _attach_metric(env, key, cm)
    xref, uref = env.group_references()
    T_ep = int(env.max_episode_len)
    T = _sweep_depth(gamma, T_ep, a.horizon_tol)
    print(f"[vstar] {a.task}: {a.groups} references x {a.epg} envs, episode "
          f"{T_ep} steps, sweeping {T} of them (gamma^{T} < {a.horizon_tol:g}, so the "
          f"rest cannot change V*_0), reference stored to {xref.shape[1]}, "
          f"gamma={gamma}, lbd={lbd}, q={env.tracking_scaler}, r={env.control_scaler}")

    grids = [a.n] + ([(a.n + 1) // 2, (a.n + 3) // 4] if a.richardson else [])
    sols = {}
    for n in grids:
        vi = TrackingVI(env, n, a.actions, gamma, "cpu")
        V0 = torch.stack([vi.solve(xref[g], horizon=T) for g in range(a.groups)])
        sols[n] = (vi, V0)
        print(f"[vstar] n={n:4d} ({n ** env.num_dim_x:7d} states x {a.actions} actions): "
              f"|V*_0| mean {float(V0.abs().mean()):.6e}")

    if a.richardson:
        # Nested only if (n-1) is divisible by 4; otherwise compare by resampling
        # the coarse grids' values at the fine grid's points, which is what
        # value_at already does exactly.
        vf, Vf = sols[grids[0]]
        Xf = vf.X
        d = []
        for n in grids[1:]:
            vc, Vc = sols[n]
            d.append(torch.stack([vc.value_at(Vc[g], Xf) for g in range(a.groups)]) - Vf)
        scale = max(float(Vf.abs().mean()), 1e-12)
        # Three norms, because the sup is set by the single worst point -- the
        # bang-bang switching surface, where V* is Lipschitz but not C^1 and no
        # refinement helps. Quoting sup alone cannot separate "the scheme is
        # poor" from "V* has a kink", and the second is not an error at all.
        for tag, fn in (("sup", lambda z: float(z.abs().max())),
                        ("L2", lambda z: float(z.pow(2).mean().sqrt())),
                        ("median", lambda z: float(z.abs().median()))):
            e = [fn(z) for z in d]
            order = float(np.log2(e[1] / e[0])) if e[0] > 0 else float("nan")
            print(f"[vstar] richardson {tag:6s}: |V_h/2-V_h/4|={e[0]:.4e}  "
                  f"|V_h-V_h/4|={e[1]:.4e}  p={order:+.3f}  "
                  f"(finest-grid error ~{100 * e[0] / scale:.3f}% of |V*|)")

    vi, V0 = sols[grids[0]]
    out = REPO / "data/toy" / key
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vstar.npz"
    np.savez(path,
             xref=xref.cpu().numpy(), uref=uref.cpu().numpy(),
             V0=V0.cpu().numpy(), X=vi.X.cpu().numpy(),
             x_lo=vi.lo, x_hi=vi.hi, n=a.n, actions=a.actions,
             gamma=gamma, lbd=lbd, groups=a.groups, envs_per_group=a.epg,
             horizon=T_ep, sweep_depth=T, ref_offset=a.ref_offset,
             cm_dataset=find_npz(key).name, seed=a.seed,
             # The reward signature, so optimality_gap can REFUSE a rollout
             # scored under a different objective than this V*.
             **cm_data.reward_signature(env))
    print(f"[vstar] wrote {path}  ({V0.shape[0]} references x {V0.shape[1]} grid states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

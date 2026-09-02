"""Certified global optimum of each fixed toy task, by the sparse moment-SOS
hierarchy.

The toy family trains and is evaluated on a FIXED benchmark: ONE reference
trajectory and ``num_envs`` pinned initial states -- one task per env slot.
Fixing the reference is what collapses the augmented state (x, xref) to x, so
the optimal value is a function of two variables; pinning the starts is what
makes each optimum computable, since a trajectory optimum is a statement about
one (reference, x_0) pair. This script solves every slot and writes the answers
next to the reference they share.

Each solve returns a certified LOWER bound (the relaxation) and a feasible
UPPER bound (the extracted control sequence rolled through the real env). Their
difference is the certificate -- it is reported per task, and a task whose gap
exceeds --gap-tol is flagged rather than silently averaged in.

    python scripts/precompute_global.py --task toy-mg-v0
    python scripts/precompute_global.py --task toy-mg-v0 --gamma 0.9 --jobs 20

Why not value iteration: VI is exact only for the DISCRETISED MDP. Measured
against these certified optima at gamma=0.01, its grid error runs 0.1-2.5% and
goes both ways -- on one task VI reported a value BELOW the true global optimum,
which is impossible. The C2RL gap being measured is itself ~0.4-1%, so VI's
reference was inside its own noise floor.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))

import gymnasium as gym  # noqa: E402

_W = {}


def horizon_for(gamma: float, max_len: int, tol: float) -> int:
    """Steps that can still move the objective -- but see the warning below.

    ``J_0`` depends on stage ``k`` only through ``gamma^k``, so once
    ``gamma^k < tol`` the tail cannot change the objective VALUE in double
    precision. That makes truncation safe for the bound, and UNSAFE for the
    trajectory: at the truncated stages the optimiser is indifferent, so the
    last controls are arbitrary and the state they leave behind is arbitrary
    too. Measured on duff at gamma=0.01 (H=8), E/E0 sat below the certified
    envelope through t=3 and then ran to 28.6 over t=5..7, purely an
    end-of-horizon artifact of stages carrying weight 1e-10 and less.

    So this is only a DEFAULT. Any comparison of trajectories across discounts
    must pin the same horizon with ``--horizon`` (100, the episode, here).
    """
    if gamma <= 0.0:
        return 1
    return int(min(max_len, max(1, np.ceil(np.log(tol) / np.log(gamma)))))


def _init(task, key, gamma, T, oa, ob, xref, uref, x0, ref_offset, ref_mode,
          threads, lookahead):
    from contractionRL import cm_data
    from contractionRL.solvers import moment_sos as tg

    torch.set_num_threads(1)
    import contractionRL.tasks.direct.toy  # noqa: F401

    env = gym.make(task, num_envs=len(x0), device="cpu",
                   reference_mode=ref_mode).unwrapped
    env.configure_ref_window(length=env.full_ref_length(ref_offset),
                             offset=ref_offset, gamma=gamma)
    cfg = _agent_cfg(task)
    cm_data.attach_metric(env, key, device="cpu", tag="[global]",
                          **{k: cfg["cm"][k] for k in cm_data.REWARD_KEYS
                             if k in cfg.get("cm", {})})
    tg.check_dynamics(env, key)
    env.set_group_references(xref, uref, x0)
    d = np.load(sorted(glob.glob(str(REPO / "data/toy" / key / "cm_data_*.npz")))[-1])
    _W.update(threads=threads, env=env, tg=tg, task=task, key=key, gamma=gamma, T=T,
              oa=oa, ob=ob, lookahead=lookahead,
              level=bool(getattr(env, "reward_level", False)),
              coeffs={str(n): float(v) for n, v in zip(d["sos_coeff_names"],
                                                       d["sos_coeff_values"])},
              wdeg=int(d["sos_w_degree"]),
              q=float(getattr(env, "tracking_scaler", 1.0)),
              r=float(getattr(env, "control_scaler", 0.0)))


def _window(prog_args, x, xref, horizon):
    """One moment-SOS solve of the gamma-discounted problem starting at ``x``."""
    tg, env = _W["tg"], _W["env"]
    prog, meta = tg.build(
        _W["key"], _W["coeffs"], _W["wdeg"], x, xref, _W["gamma"], horizon,
        float(env.dt), np.asarray(env.X_MIN).ravel(), np.asarray(env.X_MAX).ravel(),
        float(env.U_MIN.min()), float(env.U_MAX.max()),
        q=_W["q"], r=_W["r"], order_a=_W["oa"], order_b=_W["ob"], level=_W["level"])
    res = prog.solve(threads=_W["threads"])
    us, _xs = tg.read_point(prog, res["y"], meta)
    return us, res


def _solve(i):
    """The gamma-sighted optimal control of task ``i`` over the FULL episode.

    Two regimes, and the discount picks between them without changing what is
    executed -- ``T`` steps are applied either way, so the trajectory length is
    a property of the task and never of gamma:

    * ``lookahead >= T``  the discount still weighs the episode end, so one
      solve over the whole episode IS the optimal control, and its relaxation
      bound certifies it globally.
    * ``lookahead < T``   gamma has truncated inside the episode (at 0.01,
      gamma^8 = 1e-16). A single solve would leave the tail stages weightless
      and their controls arbitrary. So we do what RL does: at each step solve
      the gamma-discounted problem from the CURRENT state, apply only u_0, and
      advance. Since gamma^lookahead is at the tolerance, this receding-horizon
      control IS the optimal policy of the discounted problem to machine
      precision -- but the certificate is now per decision, so we report the
      worst window gap rather than a single global bound.
    """
    env = _W["env"]
    T, gamma = _W["T"], _W["gamma"]
    look = _W["lookahead"]
    xref_all = env._group_xref[env._group_of[i], :env.max_episode_len].numpy().astype(np.float64)
    t0 = time.time()

    if look >= T:
        x0 = env._fixed_x0[i].detach().clone().numpy().astype(np.float64)
        us, res = _window(None, x0, xref_all, T)
        lb, gaps, status, moments = float(res["bound"]), None, res["status"], res["num_moments"]
    else:
        env.reset()
        us, bounds, status, moments = [], [], "receding", 0
        for t in range(T):
            x = env.x_t[i].detach().clone().numpy().astype(np.float64)
            uw, res = _window(None, x, xref_all[t:], min(look, T - t))
            us.append(float(uw[0]))
            bounds.append(float(res["bound"]))
            status, moments = res["status"], max(moments, int(res["num_moments"]))
            u = torch.zeros(env.num_envs, env.num_dim_control)
            u[i, 0] = float(uw[0])
            env.step(u)
        us = np.asarray(us)
        lb, gaps = float("nan"), bounds

    # Feasible upper bound: the extracted controls, scored by the ENV itself.
    # This is the half of the certificate that cannot be fooled by a modelling
    # slip -- if the transcribed objective drifted from get_rewards, LB and UB
    # would stop bracketing.
    env.reset()
    disc, g = 0.0, 1.0
    for t in range(T):
        u = torch.zeros(env.num_envs, env.num_dim_control)
        u[i, 0] = float(us[t])
        _o, rew, _te, _tr, _in = env.step(u)
        disc += g * float(rew[i])
        g *= gamma
    return {"task": i, "lb": lb, "ub": float(-disc), "u": list(map(float, us)),
            "secs": time.time() - t0, "status": status, "moments": int(moments),
            "window_lb": gaps}


def _agent_cfg(task: str, algorithm: str = "c2rl-ppo") -> dict:
    import importlib

    import yaml
    entry = (gym.spec(task).kwargs or {})[
        f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"]
    pkg, fname = entry.split(":")
    with open(os.path.join(os.path.dirname(importlib.import_module(pkg).__file__),
                           fname)) as fh:
        return yaml.safe_load(fh)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="toy-mg-v0")
    p.add_argument("--algorithm", default="c2rl-ppo",
                   help="whose discount and reward weights the optimum is solved under")
    p.add_argument("--gamma", type=float, default=None,
                   help="override the agent config's discount_factor")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--ref-offset", "--ref_offset", dest="ref_offset", type=int, default=1)
    p.add_argument("--reference-mode", "--reference_mode", dest="ref_mode",
                   choices=["stabilizing", "contractive"], default="stabilizing",
                   help="which reference the optimum is solved for. Each mode is a "
                        "DIFFERENT task set, so each gets its own pack "
                        "(global.npz / global_contractive.npz).")
    p.add_argument("--horizon-tol", "--horizon_tol", dest="horizon_tol",
                   type=float, default=1e-15)
    p.add_argument("--order-a", "--order_a", dest="oa", type=int, default=3)
    p.add_argument("--order-b", "--order_b", dest="ob", type=int, default=4)
    p.add_argument("--gap-tol", "--gap_tol", dest="gap_tol", type=float, default=1e-4,
                   help="relative UB-LB above which a task is reported unconverged")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    p.add_argument("--solver-threads", "--solver_threads", dest="threads", type=int,
                   default=1, help="MOSEK threads PER worker; 1 so --jobs scales")
    p.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=None,
                   help="tasks (pinned initial conditions) to solve; default = the "
                        "env's default_num_envs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import contractionRL.tasks.direct.toy  # noqa: F401
    from contractionRL import cm_data

    key = a.task.removeprefix("toy-").removesuffix("-v0")
    cfg = _agent_cfg(a.task, a.algorithm)
    gamma = float(a.gamma if a.gamma is not None else cfg["agent"]["discount_factor"])

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    n_env = int(a.num_envs or (gym.spec(a.task).kwargs or {}).get("default_num_envs", 25))
    env = gym.make(a.task, num_envs=n_env, device="cpu",
                   reference_mode=a.ref_mode).unwrapped
    env.configure_ref_window(length=env.full_ref_length(a.ref_offset),
                             offset=a.ref_offset, gamma=gamma)
    cm_data.attach_metric(env, key, device="cpu", tag="[global]",
                          **{k: cfg["cm"][k] for k in cm_data.REWARD_KEYS
                             if k in cfg.get("cm", {})})
    env.reset()
    xref, uref = env.group_references()
    # One task per ENV SLOT. ref_groups == 1, so all slots share the reference
    # and differ only in their pinned x_0 -- which is exactly the task index.
    x0 = env._fixed_x0.clone()
    T = a.horizon or env.max_episode_len
    look = horizon_for(gamma, T, a.horizon_tol)
    mode = "one global solve" if look >= T else f"receding, {look}-step lookahead"
    print(f"[global] {a.task}: {env.num_envs} tasks over {env.ref_groups} "
          f"{a.ref_mode} reference(s), gamma={gamma}, "
          f"horizon {T} of {env.max_episode_len} ({mode}), "
          f"cost={'level' if getattr(env, 'reward_level', False) else 'decrement'}, "
          f"order (a={a.oa}, b={a.ob}), "
          f"q={getattr(env, 'tracking_scaler', 1.0)} r={getattr(env, 'control_scaler', 0.0)}",
          flush=True)

    args = (a.task, key, gamma, T, a.oa, a.ob, xref.numpy(), uref.numpy(), x0.numpy(),
            a.ref_offset, a.ref_mode, a.threads, look)
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.jobs, env.num_envs),
                                      initializer=_init, initargs=args) as pool:
        rows = []
        for row in pool.imap_unordered(_solve, range(env.num_envs)):
            rows.append(row)
            print(f"[global] task {row['task']:>3}  LB {row['lb']:+.9e}  "
                  f"UB {row['ub']:+.9e}  gap {row['ub'] - row['lb']:+.2e}  "
                  f"{row['secs']:.0f}s", flush=True)
    rows.sort(key=lambda r: r["task"])

    lb = np.array([r["lb"] for r in rows])
    ub = np.array([r["ub"] for r in rows])
    rel = np.abs(ub - lb) / np.maximum(np.abs(ub), 1e-12)
    bad = np.nonzero(rel > a.gap_tol)[0] if np.isfinite(rel).all() else np.array([], int)
    if np.isfinite(rel).all():
        print(f"\n[global] certificate: max relative UB-LB {rel.max():.3e}, "
              f"median {np.median(rel):.3e} over {len(rows)} tasks ({time.time() - t0:.0f}s)")
    else:
        # Receding horizon: the bound certifies each DECISION, not the episode.
        print(f"\n[global] receding horizon: no single-episode bound; "
              f"J realised in [{ub.min():.4e}, {ub.max():.4e}] over {len(rows)} "
              f"tasks ({time.time() - t0:.0f}s)")
    if bad.size:
        print(f"[global] {bad.size} task(s) above --gap-tol {a.gap_tol:g}: "
              f"{bad.tolist()} -- raise --order-a/--order-b for these, or treat "
              f"their J* as a bound rather than the optimum.")

    stem = "global" if a.ref_mode == "stabilizing" else f"global_{a.ref_mode}"
    # V* is gamma-dependent, so an overridden gamma is a DIFFERENT pack, not a
    # newer one. Suffixing keeps both on disk and lets train.py pick by discount.
    if a.gamma is not None:
        stem += f"_g{gamma:g}"
    out = pathlib.Path(a.out or (REPO / "data/toy" / key / f"{stem}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xref=xref.numpy(), uref=uref.numpy(), x0=x0.numpy(),
             j_lb=lb, j_ub=ub, j_star=ub, rel_gap=rel,
             u_star=np.array([r["u"] for r in rows]), gamma=gamma, horizon=T,
             order_a=a.oa, order_b=a.ob, groups=env.ref_groups,
             envs_per_group=env.num_envs // env.ref_groups,
             ref_offset=env.ref_window.offset, seed=a.seed,
             cm_dataset=key, solver="moment-sos", reference_mode=a.ref_mode,
             **cm_data.reward_signature(env))
    print(f"[global] wrote {out}  ({env.num_envs} certified optima)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

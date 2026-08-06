"""Generate C2RL's offline ``{x → W*(x)}`` contraction-metric dataset.

C2RL no longer solves the per-state CV-STEM SDP at agent construction — it
LOADS a dataset produced here (see ``c2rl._synthesize_cmg_cvstem``). Run this
once per (task, algorithm) whose ``lbd``/``w_lb``/``w_ub``/``cvstem_r_scaler``/
``cm_eps``/``cm_solver``/``cmg_memory_size`` differ; everything else about a run
— gamma, seed, network sizes, timesteps — is absent from the cache key, so one
dataset serves an entire sweep.

    python scripts/build_cm_dataset.py --task classic-cartpole-v0 --algorithm c2rl-ppo
    python scripts/build_cm_dataset.py --task cartpole --algorithm c2rl-ppo --check

The destination path and the cache key both come from ``c2rl.cm_dataset_target``
— the same function the agent loads through — so a file written here cannot
key-miss on load. Do not reimplement either here.

Feasibility is a RESULT, not a warning: if fewer than ``min_feasibility_rate``
of states solve, this exits non-zero rather than writing a dataset the CMG would
regress onto a biased subset of the state space.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSIC = ROOT / "source/contractionRL/contractionRL/tasks/direct/classic"


def _short(task: str) -> str:
    """'classic-cartpole-v0' -> 'cartpole'; already-short names pass through."""
    return task.removeprefix("classic-").removesuffix("-v0")


def _load_cfg(task: str, algorithm: str):
    """Rebuild the flat cfg namespace the agent sees, then filter it to the
    dataclass — mirroring ContractionRunner's block merge
    ``{**cm, **cmg, **empirical_dynamics, **agent}``. Reading the yaml blocks in
    a different order here would change which duplicate key wins."""
    from contractionRL.agents.skrl.c2rl import C2RLPPOCfg, C2RLSACCfg
    from contractionRL.agents.skrl.rl_glue import filter_cfg_fields

    path = CLASSIC / _short(task) / "agents" / f"skrl_{algorithm.replace('-', '_')}_cfg.yaml"
    if not path.exists():
        raise SystemExit(f"no config at {path}")
    raw = yaml.safe_load(open(path))
    merged = {**raw.get("cm", {}), **raw.get("cmg", {}),
              **raw.get("empirical_dynamics", {}), **raw.get("agent", {})}
    merged.pop("class", None)
    CfgCls = C2RLSACCfg if "sac" in algorithm else C2RLPPOCfg
    return CfgCls(**filter_cfg_fields(merged, CfgCls, context="build_cm_dataset")), path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="classic-cartpole-v0 or cartpole")
    p.add_argument("--algorithm", default="c2rl-ppo", choices=["c2rl-ppo", "c2rl-sac"])
    p.add_argument("--check", action="store_true",
                   help="report the target path and whether it already loads, then exit")
    p.add_argument("--force", action="store_true",
                   help="re-solve and overwrite even if a matching dataset exists")
    args = p.parse_args()

    import contractionRL.tasks.direct.classic  # noqa: F401
    from contractionRL.agents.skrl.c2rl import cm_dataset_target
    from contractionRL.agents.skrl.ncm_synthesis import (
        build_cm_dataset, load_cached_cm_dataset, save_cm_dataset,
        sample_state_box,
    )

    cfg, cfg_path = _load_cfg(args.task, args.algorithm)
    if cfg.cmg_method != "cvstem":
        raise SystemExit(f"cmg_method is {cfg.cmg_method!r}; only 'cvstem' uses this "
                         f"dataset ('ccm' trains the CMG directly, no SDP).")
    cache_path, cache_kwargs = cm_dataset_target(cfg)
    if cache_path is None:
        raise SystemExit(f"{cfg_path}: no cm_data_path — nowhere to write the dataset.")

    print(f"[build_cm] config    {cfg_path}")
    print(f"[build_cm] target    {cache_path}")
    print(f"[build_cm] key       lbd={cfg.lbd} w=[{cfg.w_lb},{cfg.w_ub}] "
          f"r={cfg.cvstem_r_scaler} eps={cfg.cm_eps} solver={cfg.cm_solver} "
          f"N={cfg.cmg_memory_size}", flush=True)

    existing = load_cached_cm_dataset(cache_path, **cache_kwargs)
    if existing is not None and not args.force:
        print(f"[build_cm] already present and matching "
              f"({len(existing['x'])} states, feasibility "
              f"{existing['feasibility_rate']:.1%}) — nothing to do. --force to redo.")
        return 0
    if args.check:
        print("[build_cm] NOT present (or key mismatch) — run without --check to build.")
        return 1

    task = args.task if args.task.startswith("classic-") else f"classic-{args.task}-v0"
    env = gym.make(task, num_envs=1, device="cpu").unwrapped
    x_samples = sample_state_box(env.X_MIN, env.X_MAX, n=cfg.cmg_memory_size,
                                 seed=getattr(cfg, "cm_seed", 0))

    t0 = time.time()
    ds = build_cm_dataset(
        None, env.get_f_and_B,
        x_dim=int(env.num_dim_x),
        lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
        num_samples=cfg.cmg_memory_size, solver=cfg.cm_solver,
        device="cpu", tag="[build_cm]",
        x_samples=x_samples,
        random_ratio=cache_kwargs["random_ratio"],
        min_feasibility_rate=cfg.min_feasibility_rate,
        r_scaler=cfg.cvstem_r_scaler,
        max_lambda_reductions=cfg.max_lambda_reductions,
        chi_weight=cfg.cm_chi_weight,
        nu_weight=cfg.cm_nu_weight, wdot_dt=cfg.cm_wdot_dt,
    )
    el = time.time() - t0

    rate = ds["feasibility_rate"]
    print(f"[build_cm] solved {len(ds['x'])} states in {el/60:.1f} min — "
          f"feasibility {rate:.1%}, lambda-reduced {ds.get('lambda_reduced_rate', 0.0):.1%}, "
          f"LMI residual mean {ds['residual_mean']:.3e} max {ds['residual_max']:.3e}")
    if rate < cfg.min_feasibility_rate:
        print(f"[build_cm] FAILED: feasibility {rate:.1%} < min_feasibility_rate "
              f"{cfg.min_feasibility_rate:.1%}. Nothing written — lower lbd or widen "
              f"the envelope rather than lowering the threshold.", file=sys.stderr)
        return 2

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_cm_dataset(cache_path, ds, **cache_kwargs)
    print(f"[build_cm] wrote {cache_path}")
    if load_cached_cm_dataset(cache_path, **cache_kwargs) is None:
        print("[build_cm] FAILED: the file just written does not load back under the "
              "same key.", file=sys.stderr)
        return 3
    print("[build_cm] verified: reloads under the agent's own cache key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

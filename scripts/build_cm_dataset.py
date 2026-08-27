"""Generate C2RL's offline ``{x → W*(x)}`` contraction-metric dataset.

C2RL no longer solves the per-state CV-STEM SDP at agent construction — it
loads a dataset produced here (see ``c2rl._synthesize_cmg_cvstem``). Run this
once per (task, algorithm) whose ``lbd``/``w_lb``/``w_ub``/``cvstem_r_scaler``/
``cm_eps``/``cm_solver``/``cmg_memory_size`` differ; everything else about a run
— gamma, seed, network sizes, timesteps — is absent from the cache key, so one
dataset serves an entire sweep.

    python scripts/build_cm_dataset.py --task classic-cartpole-v0 --algorithm c2rl-ppo
    python scripts/build_cm_dataset.py --task cartpole --algorithm c2rl-ppo --check

The destination path and the cache key both come from ``c2rl.cm_dataset_target``
— the same function the agent loads through — so a file written here cannot
key-miss on load. Do not reimplement either here.

Feasibility is a result, not a warning. One joint SDP covers every sample, so it
either certifies the rate over the whole draw or raises -- this exits non-zero and
writes nothing rather than leaving the CMG to regress onto a metric that
certifies nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
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
    # Probes for locating the feasibility boundary. A lambda certified at
    # N=100/eps=0.1 can still be infeasible at the shipped N=10000 (measured on
    # segway), and each attempt is a multi-hour solve, so the candidates have to
    # be drivable without rewriting the env yaml between runs. The filename and
    # cache key both derive from the EFFECTIVE values, so a probe writes its own
    # file rather than shadowing the config's.
    p.add_argument("--lbd", type=float, default=None,
                   help="override the config's contraction rate for this build")
    p.add_argument("--num-samples", "--num_samples", type=int, default=None,
                   help="override cmg_memory_size (cost is ~N^1.95, so a smaller "
                        "N is how you locate the boundary cheaply before "
                        "committing to the shipped size)")
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
    if args.lbd is not None:
        print(f"[build_cm] lbd OVERRIDE {cfg.lbd} -> {args.lbd} (probe; the config "
              f"still says {cfg.lbd})", flush=True)
        cfg.lbd = args.lbd
    if args.num_samples is not None:
        print(f"[build_cm] N OVERRIDE {cfg.cmg_memory_size} -> {args.num_samples} "
              f"(probe; the shipped size is {cfg.cmg_memory_size})", flush=True)
        cfg.cmg_memory_size = args.num_samples
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
    # A key MISS on an existing file is a clobber, not a rebuild. The filename
    # carries only (lbd, w_lb, w_ub, r_scaler) while the key also carries
    # num_samples/eps/solver/chi/nu — so a --num-samples probe resolves to the
    # SHIPPED path and overwrites a 10000-state dataset with 20 states. Measured:
    # a probe run destroyed data/classic/car/...npz (recovered from git).
    if existing is None and cache_path.exists() and not args.force:
        print(f"[build_cm] {cache_path} EXISTS but does not match this key "
              f"(N={cfg.cmg_memory_size} eps={cfg.cm_eps} solver={cfg.cm_solver}). "
              f"Writing would overwrite it. Pass --force if that is intended.",
              file=sys.stderr)
        return 2

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
        r_scaler=cfg.cvstem_r_scaler,
        chi_weight=cfg.cm_chi_weight,
        nu_weight=cfg.cm_nu_weight, wdot_dt=0.0,
    )
    el = time.time() - t0

    rate = ds["feasibility_rate"]
    print(f"[build_cm] solved {len(ds['x'])} states in {el/60:.1f} min — "
          f"feasibility {rate:.1%}, lambda-reduced {ds.get('lambda_reduced_rate', 0.0):.1%}, "
          f"LMI residual mean {ds['residual_mean']:.3e} max {ds['residual_max']:.3e}")
    # No feasibility THRESHOLD to check: one joint program either certifies lambda
    # over the whole draw or build_cm_dataset raises, so rate is always 1.0 here
    # and any `rate < threshold` test is unreachable. Infeasibility arrives as an
    # exception, not as a low rate.

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

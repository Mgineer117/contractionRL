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
TOY = ROOT / "source/contractionRL/contractionRL/tasks/direct/toy"


def _short(task: str) -> str:
    """'classic-cartpole-v0' -> 'cartpole', 'toy-mg-v0' -> 'mg'; short names pass through."""
    return task.removeprefix("classic-").removeprefix("toy-").removesuffix("-v0")


# Imported, not re-listed: a third toy env added to the package but missed here
# would silently take the CV-STEM path and produce a SAMPLED metric where an
# exact one was available, with nothing in the output saying so.
sys.path.insert(0, str(ROOT / "source" / "contractionRL"))
from contractionRL.tasks.direct.toy import TOY_ENVS as TOY_KEYS  # noqa: E402


def _is_toy(task: str) -> bool:
    return _short(task) in TOY_KEYS


def _task_id(task: str) -> str:
    """Accept either the short key or the full gym id, for both families."""
    if task.startswith(("classic-", "toy-")):
        return task
    return f"toy-{task}-v0" if task in TOY_KEYS else f"classic-{task}-v0"


def _agents_dir(task: str) -> Path:
    fam = TOY if _is_toy(task) else CLASSIC
    return fam / _short(task) / "agents"


def _load_cfg(task: str, algorithm: str):
    """Rebuild the flat cfg namespace the agent sees, then filter it to the
    dataclass — mirroring ContractionRunner's block merge
    ``{**cm, **cmg, **empirical_dynamics, **agent}``. Reading the yaml blocks in
    a different order here would change which duplicate key wins."""
    from contractionRL.agents.skrl.c2rl import C2RLPPOCfg, C2RLSACCfg
    from contractionRL.agents.skrl.rl_glue import filter_cfg_fields

    path = _agents_dir(task) / f"skrl_{algorithm.replace('-', '_')}_cfg.yaml"
    if not path.exists():
        raise SystemExit(f"no config at {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    merged = {**raw.get("cm", {}), **raw.get("cmg", {}),
              **raw.get("empirical_dynamics", {}), **raw.get("agent", {})}
    merged.pop("class", None)
    CfgCls = C2RLSACCfg if "sac" in algorithm else C2RLPPOCfg
    return CfgCls(**filter_cfg_fields(merged, CfgCls, context="build_cm_dataset")), path



def _plant_signature(key: str) -> str:
    """The symbolic (f, B) a toy certificate was solved for, as one string."""
    from contractionRL.solvers.sos_cm import PLANTS
    pl = PLANTS[key]
    return f"f={[str(e) for e in pl['f']]} B={[str(e) for e in pl['B']]}"


def _build_via_sos(env, key, cfg, x_samples, bisect: bool):
    """Metric synthesis for the POLYNOMIAL toy envs.

    Same output contract as the CV-STEM path -- an ``{x -> W*(x)}`` dataset the
    CMG regresses -- and the same LMI. What differs is where the guarantee comes
    from: CV-STEM solves the LMI at N sampled states, so it holds at those states;
    SOS proves it as an algebraic identity over the whole box, so every sample is
    drawn from a field that is already certified everywhere.

    The dataset is built AT the config's ``lbd``, not at whatever the bisection
    could reach: ``lbd`` is part of the cache key and the filename, so a dataset
    certifying a different rate than the config names is exactly the key-miss
    (or worse, silent mismatch) cm_dataset_target exists to prevent. Use
    ``--bisect-lbd`` to discover the largest feasible rate, then put it in the
    yaml -- the same two-step find_uniform_lambda gives the classic envs.
    """
    import numpy as np
    from contractionRL.solvers.sos_cm import SOSCfg, _feasible_at, verify, w_at

    scfg = SOSCfg(w_degree=int(getattr(cfg, "sos_w_degree", 2)),
                  w_lb=float(cfg.w_lb), w_ub=float(cfg.w_ub),
                  r_scaler=float(cfg.cvstem_r_scaler),
                  solver=str(cfg.cm_solver),
                  verify_grid=int(getattr(cfg, "sos_verify_grid", 101)))
    lo = env.X_MIN.cpu().numpy()
    hi = env.X_MAX.cpu().numpy()

    if bisect:
        lam_hi = float(getattr(cfg, "sos_lam_hi", 4.0))
        # If the top of the bracket is itself feasible the bisection converges to
        # lam_hi and reports it as "the ceiling", which is a bracket artifact, not
        # a property of the plant. Say so instead.
        def _certifies(lam):
            """Feasible AND verifying. The SDP alone is not enough at the
            boundary: right at the ceiling the solver returns 'optimal' with
            slack a few 1e-6 wide, which then fails the grid check and leaves a
            config whose lbd can never be built. Accepting only rates that
            verify makes the reported ceiling one that actually ships."""
            w = _feasible_at(key, lam, scfg, lo, hi)
            if w is None:
                return False
            return verify(key, lam, w, scfg, lo, hi, 41)["ok"]

        if _certifies(lam_hi):
            raise SystemExit(
                f"[build_cm] {key}: lam = {lam_hi} (cm.sos_lam_hi) is itself "
                f"certifiable, so the ceiling is above the search bracket and any "
                f"bisection result would just be lam_hi. Raise cm.sos_lam_hi.")
        low, high, best = 0.0, lam_hi, None
        for _ in range(int(getattr(cfg, "sos_bisect_iters", 10))):
            mid = 0.5 * (low + high)
            if _certifies(mid):
                low, best = mid, mid
            else:
                high = mid
            print(f"[build_cm] SOS bisect lam={mid:.4f} -> "
                  f"[{low:.4f}, {high:.4f}]", flush=True)
        if best is None:
            raise SystemExit(f"[build_cm] {key}: nothing in (0, {lam_hi}] certified.")
        raise SystemExit(
            f"[build_cm] largest SOS-certifiable lbd for {key} at "
            f"r={scfg.r_scaler}, deg(W)={scfg.w_degree} is {best:.4f}. "
            f"Set cm.lbd to it in the yaml, then rerun without --bisect-lbd.")

    coeffs = _feasible_at(key, float(cfg.lbd), scfg, lo, hi)
    if coeffs is None:
        raise SystemExit(
            f"[build_cm] {key}: the SOS program is INFEASIBLE at the config's "
            f"lbd={cfg.lbd} (r={scfg.r_scaler}, deg(W)={scfg.w_degree}). "
            f"Infeasibility is the answer, not a nuisance: lower cm.lbd (find the "
            f"ceiling with --bisect-lbd) or raise sos_w_degree. Do NOT widen "
            f"[w_lb, w_ub] first -- that changes what is being certified.")

    v = verify(key, float(cfg.lbd), coeffs, scfg, lo, hi, scfg.verify_grid)
    if not v["ok"]:
        raise SystemExit(f"[build_cm] {key}: certificate did not verify on a "
                         f"{scfg.verify_grid}^2 grid: {v}. Not writing.")
    print(f"[build_cm] SOS verified on {scfg.verify_grid}^2 grid: max LMI eig "
          f"{v['max_residual']:+.3e} (want <= 0), min eig(W) {v['min_eig_W']:.4f}")

    W = w_at(key, coeffs, scfg.w_degree, x_samples)
    # The COEFFICIENTS go in the npz alongside the samples. The samples exist so
    # this dataset is interchangeable with the CV-STEM path; the coefficients let
    # C2RL skip the regression entirely (cmg_method: sos) and evaluate the exact
    # polynomial, which is the whole point of having an analytic certificate --
    # a 2.5% regression error in M is not "the exact metric".
    return {"x": np.asarray(x_samples, np.float64), "W": W,
            # Two plain arrays, not one object array: an object array needs
            # allow_pickle=True to read back, and np.load defaults to False, so
            # the loader raised instead of returning the metric.
            "sos_coeff_names": np.array(sorted(coeffs), dtype="U24"),
            "sos_coeff_values": np.array([coeffs[k] for k in sorted(coeffs)], np.float64),
            "sos_w_degree": int(scfg.w_degree),
            # WHICH PLANT this certificate is about. The cache key is built from
            # the solver knobs (lbd, w_lb, w_ub, r, eps, N) and knows nothing
            # about f and B, so editing the dynamics leaves a matching filename
            # holding a metric for the OLD system -- and --force is the only way
            # to notice, if you think to use it. Measured 2026-09-01: mg's drift
            # lost a term and every downstream lam, band and optimum was read
            # through the superseded W without one warning. cm_data.attach_metric
            # compares this against the live PLANTS entry and refuses a mismatch.
            "plant_signature": np.array(_plant_signature(key), dtype="U256"),
            # 1.0 by construction: the identity holds everywhere, so no sample
            # can be infeasible. Kept so the npz schema matches the CV-STEM path.
            "feasibility_rate": 1.0,
            "residual_mean": float(v["max_residual"]),
            "residual_max": float(v["max_residual"]),
            "lambda_reduced_count": 0, "lambda_reduced_rate": 0.0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   help="classic-cartpole-v0 | cartpole | toy-mg-v0 | mg")
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
    p.add_argument("--bisect-lbd", "--bisect_lbd", action="store_true",
                   help="toy/SOS only: report the largest certifiable lbd and exit "
                        "without writing. The analogue of find_uniform_lambda.py.")
    p.add_argument("--force", action="store_true",
                   help="re-solve and overwrite even if a matching dataset exists")
    args = p.parse_args()

    import contractionRL.tasks.direct.classic  # noqa: F401
    import contractionRL.tasks.direct.toy  # noqa: F401
    from contractionRL.agents.skrl.c2rl import cm_dataset_target
    from contractionRL.agents.skrl.ncm_synthesis import (
        build_cm_dataset,
        load_cached_cm_dataset,
        sample_state_box,
        save_cm_dataset,
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
    cache_path, cache_kwargs = cm_dataset_target(cfg)
    if cache_path is None:
        raise SystemExit(f"{cfg_path}: no cm_data_path — nowhere to write the dataset.")
    # A --num-samples probe gets its OWN filename. cm_dataset_filename encodes only
    # (lbd, w_lb, w_ub, r_scaler), so without this a probe resolves to the SHIPPED
    # path and writes a 100-state file where the agent expects 10000 -- which then
    # key-misses at load and sends training back into a 15 h SDP. --lbd probes are
    # already distinct because lbd IS in the name; N is not. The agent never
    # constructs this name, so a probe can never be picked up by mistake.
    if args.num_samples is not None:
        cache_path = cache_path.with_name(
            f"{cache_path.stem}_N{cfg.cmg_memory_size}{cache_path.suffix}")
        print(f"[build_cm] probe path (N override): {cache_path}")

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

    env = gym.make(_task_id(args.task), num_envs=1, device="cpu").unwrapped
    x_samples = sample_state_box(env.X_MIN, env.X_MAX, n=cfg.cmg_memory_size,
                                 seed=getattr(cfg, "cm_seed", 0))

    t0 = time.time()
    if _is_toy(args.task):
        ds = _build_via_sos(env, _short(args.task), cfg, x_samples, args.bisect_lbd)
        el = time.time() - t0
        print(f"[build_cm] SOS certificate + {len(ds['x'])} samples in {el/60:.1f} min")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_cm_dataset(cache_path, ds, **cache_kwargs)
        print(f"[build_cm] wrote {cache_path}")
        return 0

    # env.get_rollout, not None: with cmg_random_ratio in (0,1) the mixer asks the
    # rollout for the reference-structured share and a None here is a TypeError
    # mid-solve. At the shipped ratio of 0.0 the offline pool is used wholesale
    # and this is never called.
    ds = build_cm_dataset(
        env.get_rollout, env.get_f_and_B,
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

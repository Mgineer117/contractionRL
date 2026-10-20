#!/usr/bin/env python
"""Assert every env in a batch has the CM dataset its own yaml asks for.

The 2026-08-24 relaunch lost 22 runs to a single missing file: the car_weak ->
car_v1 rename moved the expected dataset path, the renamed .npz never reached the
cluster, and every job died in `synthesize_cmg` after wandb had already created
the run -- 22 W&B entries with zero steps and no output.log. That failure is
detectable in under a second, but only if something checks it BEFORE 120 jobs go
into the queue.

This reuses the agent's own resolver and loader (`cm_dataset_target` +
`load_cached_cm_dataset`), so a pass here means the identical call inside
`_synthesize_cmg_cvstem` cannot return None. Re-deriving the filename or the
cache key locally is precisely the mistake `cm_dataset_target`'s docstring warns
about -- a check that agrees with itself and disagrees with the agent is worse
than no check.

Usage:
    python scripts/skrl/c2rl/preflight_cm_data.py classic-cartpole-v0 classic-car-v1 ...
    python scripts/skrl/c2rl/preflight_cm_data.py --batch        # the four gamma-batch envs
Exit status is 1 if any env would fail, so it can gate a submit script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))

# The envs of the gamma batch, in the order they are submitted.
BATCH = ["classic-cartpole-v0", "classic-segway-v0", "classic-car-v0", "classic-car-v1"]


def cfg_for(task: str, algorithm: str):
    """Build the algorithm Cfg exactly as train.py does: yaml -> filtered dataclass."""
    from contractionRL.agents.skrl.c2rl import C2RLPPOCfg, C2RLSACCfg
    from contractionRL.agents.utils.RL.config import filter_cfg_fields, migrate_c2rl_metric_cfg

    import contractionRL.tasks.direct.common.tracking_tasks  # noqa: F401
    algorithm = algorithm.replace("-", "_")
    from contractionRL.agents.utils.RL.config import load_registered_agent_config
    y = Path(f"skrl_{algorithm}_cfg.yaml")
    raw = load_registered_agent_config(task, f"skrl_{algorithm}_cfg_entry_point")
    # Section merge order is ContractionRunner._setup_c2rl's, agent last so it
    # wins (its line 870). Reversing it would read a `cm:` value the agent does
    # not use, and the whole point of this check is to agree with the agent.
    merged = {**(raw.get("cm") or {}), **(raw.get("cmg") or {}),
              **(raw.get("empirical_dynamics") or {}), **(raw.get("agent") or {})}
    merged.pop("class", None)
    merged = migrate_c2rl_metric_cfg(merged)
    merged.setdefault("cmg_method", "ccm" if algorithm == "c2rl_ppo" else "cvstem")
    Cfg = C2RLSACCfg if algorithm == "c2rl_sac" else C2RLPPOCfg
    return Cfg(**filter_cfg_fields(merged, Cfg, context="preflight")), y


def check(task: str, algorithm: str = "c2rl_ppo", cmg_method: str | None = None) -> tuple[bool, str]:
    from contractionRL.agents.skrl.c2rl import cm_dataset_target
    from contractionRL.agents.utils.contraction.ncm_synthesis import load_cached_cm_dataset

    cfg, y = cfg_for(task, algorithm)
    if cmg_method is not None:
        cfg.cmg_method = cmg_method
    if cfg.cmg_method == "ccm":
        return True, "CCM trains the metric directly; no CV-STEM dataset is required"
    path, kwargs = cm_dataset_target(cfg)
    if path is None:
        return False, f"{y.name} sets neither cm_data_path nor dynamics_pretrain_data_path"
    abs_path = path if path.is_absolute() else REPO / path
    if not abs_path.is_file():
        return False, (f"MISSING {path}  (lbd={cfg.lbd} w_lb={cfg.w_lb} "
                       f"w_ub={cfg.w_ub} r={cfg.cvstem_r_scaler})")
    # Present but possibly solved under a different config -- the agent treats
    # that identically to absent, so this check must too.
    if load_cached_cm_dataset(abs_path, **kwargs) is None:
        return False, (f"KEY MISMATCH {path} exists but was not solved at "
                       f"(lbd={cfg.lbd}, eps={cfg.cm_eps}, solver={cfg.cm_solver}, "
                       f"N={cfg.cmg_memory_size})")
    return True, f"ok {path.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="task ids, e.g. classic-car-v1")
    ap.add_argument("--batch", action="store_true", help="check the four gamma-batch envs")
    ap.add_argument("--algorithm", default="c2rl_ppo")
    ap.add_argument("--cmg-method", choices=["cvstem", "ccm"])
    a = ap.parse_args()
    tasks = a.tasks or (BATCH if a.batch else [])
    if not tasks:
        ap.error("give task ids or --batch")

    bad = 0
    for t in tasks:
        try:
            ok, msg = check(t, a.algorithm, a.cmg_method)
        except Exception as e:  # a broken yaml is also a launch blocker
            ok, msg = False, f"{type(e).__name__}: {e}"
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {t:22s} {msg}")
    print(f"\n{len(tasks) - bad}/{len(tasks)} ready" + ("" if not bad else "  -- DO NOT SUBMIT"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

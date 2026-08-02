"""Smoke-test every algorithm on every environment.

Runs each (env, algorithm) pair through the real ``scripts/skrl/train.py`` entry
point for a handful of timesteps and reports pass/fail. The point is not to
learn anything — it is to catch the failures that a short run surfaces just as
well as a long one, and that a unit test cannot see because they live in the
wiring rather than the math:

  * a config key silently dropped by ``rl_glue.filter_cfg_fields``
  * an observation-layout mismatch between an env and a model
  * a missing ``skrl_<algo>_cfg_entry_point`` registration
  * a capability an agent discovers via ``getattr`` that one env family lacks
  * a shape bug that only appears once the trainer's eval loop runs

Eval is deliberately NOT skipped: several past regressions (stale reset caches,
AUC blow-ups, tracker slicing) were only observable in the post-training
evaluation, not in the training loop.

Usage
-----
    python scripts/smoke_test.py --family classic
    python scripts/smoke_test.py --family isaac
    python scripts/smoke_test.py --family classic --algorithms ppo,c3m
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Isaac Sim lives in its own conda env; the classic family runs in whatever
# interpreter invoked this script.
ISAAC_PYTHON = os.path.expanduser("~/miniconda3/envs/env_isaaclab/bin/python")

# Full stdout+stderr of every failing run lands here.
FAIL_LOG_DIR = os.path.join(REPO, "logs", "smoke_failures")


def _PY(family: str) -> str:
    return ISAAC_PYTHON if family == "isaac" and os.path.exists(ISAAC_PYTHON) else sys.executable

CLASSIC_ENVS = ["car", "cartpole", "segway", "turtlebot", "quadrotor"]
CLASSIC_ALGOS = ["ppo", "sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"]

# Isaac path-tracking envs support the same algorithm set as classic; the
# vel-tracking envs are the locomotion pretrain stage and only run PPO/SAC
# (they have no reference trajectory, hence no contraction problem to certify).
ISAAC_PATH_ENVS = ["Quadruped-PathTracking-v0", "Humanoid-PathTracking-v0",
                   "Manipulator-PathTracking-v0"]
ISAAC_VEL_ENVS = ["Quadruped-VelTracking-v0", "Humanoid-VelTracking-v0",
                  "Manipulator-VelTracking-v0"]

# turtlebot is driftless (f == 0), which is infeasible for CV-STEM's SDP at
# every lambda by construction — the house rule is to use cmg_method: ccm there
# rather than to loosen the feasibility threshold, so the CV-STEM-LQR agent has
# nothing to solve. Skipping is the documented behaviour, not a workaround.
SKIP = {("turtlebot", "cvstem-lqr")}

# CV-STEM-LQR does not scale to Isaac state dimensions. The pointwise metric SDP
# is FEASIBLE there, just far too slow: measured with SCS at lbd=0.5/w_lb=0.01/
# w_ub=10, one solve takes 0.30s at the classic car's n=6, 23.4s at the
# quadruped's n=36, and 111.6s at the humanoid's n=50. metric_source=online
# needs one solve per env per step, and metric_source=pretrained needs a dataset
# of thousands of states (~64 h for the quadruped, single-threaded). This is a
# measured capability limit, not a wiring bug — hence no
# skrl_cvstem_lqr_cfg.yaml is registered for the Isaac path-tracking envs, and
# the pair is skipped rather than reported as a failure.
ISAAC_UNSUPPORTED = {"cvstem-lqr"}

# The analytic controllers consume get_f_and_B directly. Classic envs have it in
# closed form; Isaac envs do not, so these need a NeuralDynamics from a prior
# C3M/C2RL run injected via --dynamics_checkpoint.
NEEDS_DYNAMICS = ("lqr", "sdlqr", "cvstem-lqr")


def _latest_dynamics_ckpt() -> str | None:
    """Most recently written ``checkpoints/dynamics.pt`` under logs/."""
    hits = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "logs")):
        if "dynamics.pt" in files and os.path.basename(root) == "checkpoints":
            p = os.path.join(root, "dynamics.pt")
            hits.append((os.path.getmtime(p), p))
    return max(hits)[1] if hits else None


def run_one(family: str, env: str, algo: str, timesteps: int, num_envs: int,
            timeout: int, dynamics_ckpt: str | None = None) -> tuple[str, str, float]:
    """Returns (status, detail, seconds)."""
    cmd = [_PY(family), "scripts/skrl/train.py", "--algorithm", algo,
           "--num_timesteps", str(timesteps), "--num_envs", str(num_envs),
           "--no_wandb", "--seed", "0"]
    if family == "classic":
        cmd += ["--classic", "--task", f"classic-{env}-v0", "--device", "cpu"]
    else:
        cmd += ["--task", env, "--headless"]
        # Isaac envs have no analytical get_f_and_B, so the analytic controllers
        # need a NeuralDynamics from an earlier C3M/C2RL run. That is a real
        # ordering requirement of the pipeline, not a test artefact — C3M runs
        # first below and its dynamics.pt is threaded in here.
        if dynamics_ckpt and algo in ("lqr", "sdlqr", "cvstem-lqr"):
            cmd += ["--dynamics_checkpoint", dynamics_ckpt]

    # Fixed setup costs, paid before training and NOT scaled by
    # --num_timesteps, so a short run needs them shortened too. These are
    # properties of the ALGORITHM, not of the env family — classic C3M/C2RL pay
    # the same NeuralDynamics-pretrain and full-buffer-epoch cost as Isaac.
    if algo in ("c3m", "c2rl-ppo", "c2rl-sac"):
        cmd += ["--dynamics_pretrain_epochs", "5"]
    # C2RL's Phase-A CMG synthesis is the other fixed cost: 1000 epochs at
    # ~8.6 s/it on the quadruped is ~2.4 h before the first env step.
    if algo in ("c2rl-ppo", "c2rl-sac"):
        cmd += ["--cmg_regress_epochs", "5"]
    # C3M does a FULL pass over its static buffer every update
    # (memory_size/batch_size batches), so the default 131072 is ~128 batches
    # per timestep. Shrink the buffer, not the batch size, so the per-batch
    # shapes the contraction loss sees stay realistic. NOT applied to c2rl-ppo:
    # its memory_size must equal agent.rollouts.
    if algo == "c3m":
        cmd += ["--memory_size", "8192"]

    env_vars = dict(os.environ)
    # The editable install points at the MAIN checkout; without this a worktree
    # run silently imports the other tree's library and "fixes" appear to fail.
    env_vars["PYTHONPATH"] = os.path.join(REPO, "source", "contractionRL")

    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=REPO, env=env_vars, timeout=timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"exceeded {timeout}s", time.time() - t0
    dt = time.time() - t0
    out = p.stdout + p.stderr

    def _keep_log() -> str:
        """Persist the WHOLE output. The one-line summary is often the least
        useful line in it (hydra's "set HYDRA_FULL_ERROR=1" trailer), and
        re-running an Isaac failure to see its traceback costs 30+ minutes."""
        log = os.path.join(FAIL_LOG_DIR, f"{env}_{algo}.log".replace("/", "_"))
        os.makedirs(FAIL_LOG_DIR, exist_ok=True)
        with open(log, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + out)
        return os.path.relpath(log, REPO)

    if p.returncode != 0:
        # Prefer the last exception line that is not hydra's trailer.
        tb = [ln for ln in re.findall(r"^\w*(?:Error|Exception)\b.*$", out, re.M)
              if "HYDRA_FULL_ERROR" not in ln]
        detail = tb[-1] if tb else (out.strip().splitlines() or ["no output"])[-1]
        return "FAIL", f"{detail[:220]}  (log: {_keep_log()})", dt

    # A zero exit code is necessary but not sufficient: a NaN reward or a
    # non-finite Stability metric means the run "succeeded" into garbage.
    #
    # Match nan only where it is a reported VALUE ("auc : nan +- nan",
    # "reward=nan"), never merely a word. The divergence guard's own message
    # ("... produced a non-finite (NaN/Inf) physical state ... carrying the last
    # finite value forward") says "NaN" while describing the guard WORKING, and
    # a bare \bnan\b flagged those runs as failures even though every metric
    # came out finite.
    metric_nan = re.compile(r"(?:[:=]\s*|\s)nan(?:\s*(?:±|\+/-)|\s*$|[,)])", re.I | re.M)
    bad = [ln for ln in out.splitlines()
           if metric_nan.search(ln) and "WARNING" not in ln]
    if bad:
        # A NaN needs its log just as much as a crash does — more, since there
        # is no traceback pointing at the cause.
        return "NAN", f"{bad[-1].strip()[:220]}  (log: {_keep_log()})", dt
    return "PASS", "", dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["classic", "isaac"], required=True)
    ap.add_argument("--algorithms", default=None, help="comma-separated subset")
    ap.add_argument("--envs", default=None, help="comma-separated subset")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.family == "classic":
        envs = (args.envs or ",".join(CLASSIC_ENVS)).split(",")
        pairs = [(e, a) for e in envs
                 for a in (args.algorithms or ",".join(CLASSIC_ALGOS)).split(",")]
        timesteps = args.timesteps or 200
        num_envs = args.num_envs or 4
    else:
        path_envs = (args.envs or ",".join(ISAAC_PATH_ENVS)).split(",")
        vel_envs = [] if args.envs else ISAAC_VEL_ENVS
        algos = (args.algorithms or ",".join(CLASSIC_ALGOS)).split(",")
        pairs = [(e, a) for e in path_envs for a in algos]
        pairs += [(e, a) for e in vel_envs for a in algos if a in ("ppo", "sac")]
        timesteps = args.timesteps or 60
        num_envs = args.num_envs or 16

    # C3M writes <log_dir>/checkpoints/dynamics.pt, which the analytic
    # controllers need on Isaac — so run it before them rather than reporting a
    # missing-prerequisite error as if it were a bug.
    order = {"c3m": 0}
    pairs.sort(key=lambda p: (p[0], order.get(p[1], 1)))

    results = []
    dynamics_ckpt: dict[str, str] = {}
    for env, algo in pairs:
        short = env.split("-")[0].lower()
        if (short, algo) in SKIP or (env, algo) in SKIP:
            print(f"  SKIP  {env:32s} {algo:12s}  (documented incompatibility)", flush=True)
            results.append(("SKIP", env, algo, "", 0.0))
            continue
        if args.family == "isaac" and algo in ISAAC_UNSUPPORTED:
            print(f"  SKIP  {env:32s} {algo:12s}  (SDP intractable at this state dim)", flush=True)
            results.append(("SKIP", env, algo, "", 0.0))
            continue
        print(f"  ....  {env:32s} {algo:12s}", end="\r", flush=True)
        status, detail, dt = run_one(args.family, env, algo, timesteps, num_envs,
                                     args.timeout, dynamics_ckpt.get(env))
        if status == "PASS" and algo == "c3m":
            found = _latest_dynamics_ckpt()
            if found:
                dynamics_ckpt[env] = found
        note = detail
        if args.family == "isaac" and algo in NEEDS_DYNAMICS and env not in dynamics_ckpt:
            note = (note + "  [no dynamics.pt available]").strip()
        print(f"  {status:6s}{env:32s} {algo:12s} {dt:6.1f}s  {note}", flush=True)
        results.append((status, env, algo, note, dt))

    bad = [r for r in results if r[0] not in ("PASS", "SKIP")]
    n_pass = sum(1 for r in results if r[0] == "PASS")
    print(f"\n{n_pass}/{len(results)} passed, {len(bad)} failed")
    for status, env, algo, detail, _ in bad:
        print(f"  {status}: {env} {algo}: {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

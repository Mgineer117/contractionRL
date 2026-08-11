"""Check all six requested changes against the files, not from memory."""
import glob
import pathlib
import re

import yaml

cfgs = sorted(glob.glob("search/configs/c2rl-ppo-g*.yaml"))
cfgs = [c for c in cfgs if "gamma-seeds" not in c]
envs = sorted(glob.glob("source/contractionRL/contractionRL/tasks/direct/classic/*/"
                        "agents/skrl_c2rl_ppo_cfg.yaml"))

print("=" * 72)
gammas = []
for c in cfgs:
    d = yaml.safe_load(pathlib.Path(c).read_text())
    gammas.append(d["parameters"]["agent.discount_factor"]["value"])
want = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
ok1 = sorted(gammas) == sorted(want)
print(f"1. gamma set        {'PASS' if ok1 else 'FAIL'}  {sorted(gammas)}")
print(f"   files: {[pathlib.Path(c).name for c in cfgs]}")

print("=" * 72)
BANNED = ["agent.use_state_norm", "agent.use_value_norm", "agent.use_reward_norm",
          "agent.learning_rate_scheduler",
          "agent.learning_rate_scheduler_kwargs.kl_threshold"]
# The setting LIVES in the env yamls; the sweep's job is merely not to touch it.
# Pinning it in both places would be one decision stored twice.
ok2 = True
for c in cfgs:
    p_ = yaml.safe_load(pathlib.Path(c).read_text())["parameters"]
    for k in BANNED:
        if k in p_:
            print(f"   FAIL {pathlib.Path(c).name}: {k} present in sweep")
            ok2 = False
env_bad = []
for e in envs:
    s_ = pathlib.Path(e).read_text()
    for k in ("use_state_norm", "use_value_norm", "use_reward_norm"):
        if not re.search(rf"^  {k}: false", s_, re.M):
            env_bad.append(f"{e.split('/')[-3]}:{k}")
    if not re.search(r"^  learning_rate_scheduler: null", s_, re.M):
        env_bad.append(f"{e.split('/')[-3]}:scheduler")
ok2 = ok2 and not env_bad
print(f"2. norms+KLAdaptive {'PASS' if ok2 else 'FAIL'}  "
      f"absent from all {len(cfgs)} sweeps, off in all {len(envs)} env yamls"
      + (f"  bad={env_bad}" if env_bad else ""))

print("=" * 72)
ok3 = True
for c in cfgs:
    p_ = yaml.safe_load(pathlib.Path(c).read_text())["parameters"]
    for k in ("agent.caps_temporal_scale", "agent.caps_spatial_scale",
              "agent.caps_spatial_std"):
        if k in p_:
            print(f"   FAIL {pathlib.Path(c).name}: {k} present in sweep")
            ok3 = False
caps_bad = [e.split("/")[-3] for e in envs
            if not (re.search(r"^  caps_temporal_scale: 0\.0\b", pathlib.Path(e).read_text(), re.M)
                    and re.search(r"^  caps_spatial_scale: 0\.0\b", pathlib.Path(e).read_text(), re.M))]
ok3 = ok3 and not caps_bad
print(f"3. caps == 0        {'PASS' if ok3 else 'FAIL'}  "
      f"absent from all sweeps, 0.0 in all {len(envs)} env yamls"
      + (f"  bad={caps_bad}" if caps_bad else ""))

print("=" * 72)
cm_keys = set()
for c in cfgs:
    cm_keys |= {k for k in yaml.safe_load(pathlib.Path(c).read_text())["parameters"] if k.startswith("cm.")}
print(f"4. cm.* in search    {sorted(cm_keys)}")
print("   ^^ NOTE: originally you said REMOVE these; your later 'Fix 4 ... as it")
print("      needs to pretrain' had me restore them with cm.cm_build_if_missing.")

print("=" * 72)
ADDED = ["agent.learning_rate", "agent.kl_threshold", "agent.std_dev_annealing"]
ok5 = True
for c in cfgs:
    p = yaml.safe_load(pathlib.Path(c).read_text())["parameters"]
    for k in ADDED:
        if k not in p or ("values" not in p[k] and "distribution" not in p[k]):
            print(f"   FAIL {pathlib.Path(c).name}: {k} not searched")
            ok5 = False
print(f"5. lr/kl/anneal     {'PASS' if ok5 else 'FAIL'}  searched in all {len(cfgs)} sweeps")

print("=" * 72)
pat = re.compile(r"residual_pretrain|_pretrain_residual|residual_contraction_pretrain")
hits = []
for f in list(pathlib.Path().rglob("*.py")) + list(pathlib.Path().rglob("*.yaml")):
    # skip this file: its own pattern string would match
    if ".git" in str(f) or "worktrees" in str(f) or f.name == pathlib.Path(__file__).name:
        continue
    t = f.read_text(errors="ignore")
    for i, ln in enumerate(t.splitlines(), 1):
        if pat.search(ln) and not ln.strip().startswith("#"):
            hits.append(f"{f}:{i}")
print(f"6. pretrain removed {'PASS' if not hits else 'FAIL'}  "
      f"non-comment references: {hits or 'none'}")
print("=" * 72)

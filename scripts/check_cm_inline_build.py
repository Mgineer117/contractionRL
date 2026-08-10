"""Exercise C2RLAgent._build_cm_dataset_inline for real, then load the result back.

Uses a SMALL cmg_memory_size and an unused lbd so the cache genuinely misses, and
checks the round trip that matters: the file the builder writes must load against
the SAME cache key the consumer computes, since both go through
cm_dataset_target(). A mismatch there is the silent failure this flag exists to
avoid.
"""
import pathlib
import tempfile

import contractionRL.tasks.direct.classic  # noqa: F401
import gymnasium as gym
from contractionRL.agents.skrl.c2rl import C2RLAgent, C2RLPPOCfg, cm_dataset_target
from contractionRL.agents.skrl.ncm_synthesis import load_cached_cm_dataset

tmpdir = pathlib.Path(tempfile.mkdtemp())
cfg = C2RLPPOCfg()
cfg.cm_data_path = str(tmpdir / "cm_data.npz")
cfg.cm_build_if_missing = True
cfg.cmg_memory_size = 64          # small: this is a plumbing test, not a solve test
cfg.lbd = 0.123456                # unused elsewhere -> guaranteed cache miss
cfg.w_lb, cfg.w_ub = 0.001, 1000.0
cfg.cvstem_r_scaler = 1.6
cfg.cm_solver = "SCS"             # no licence needed for the test

path, kwargs = cm_dataset_target(cfg)
print(f"target {path}")
assert load_cached_cm_dataset(path, **kwargs) is None, "fixture should start empty"

env = gym.make("classic-car-v0", num_envs=1, device="cpu").unwrapped
agent = C2RLAgent.__new__(C2RLAgent)     # no full construction: only this method
agent._cfg = cfg
agent._env = env

agent._build_cm_dataset_inline(path, kwargs)

assert path.exists(), f"builder did not write {path}"
ds = load_cached_cm_dataset(path, **kwargs)
assert ds is not None, "WROTE a file the consumer cannot load -- key mismatch"
print(f"PASS round trip: {len(ds['x'])} states, feasibility {ds['feasibility_rate']:.1%}")
assert len(ds["x"]) > 0
# no temp files left behind
leftover = list(tmpdir.glob("*.tmp"))
assert not leftover, f"temp files leaked: {leftover}"
print("PASS no .tmp leftovers (atomic rename cleaned up)")

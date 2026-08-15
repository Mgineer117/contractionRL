"""Self-check for the X_TERMINATION_* early-termination box.

    python scripts/check_x_termination.py

Covers the four things that can silently break:
  1. OFF by default — every env keeps the old never-terminating behaviour.
  2. ON — leaving the box ends the episode, on `truncated` (not `terminated`),
     so skrl's GAE keeps bootstrapping and there is no suicide bonus.
  3. A termination box wider than [X_MIN, X_MAX] RAISES instead of no-opping
     (step() clamps first, so a wider bound could never fire).
  4. StatManagerEnvWrapper drops early-ended episodes from AUC/lambda instead of
     padding them into a full-length curve, and reports the dropped fraction.
"""
import pathlib
import sys

import contractionRL.tasks  # noqa: F401  (registers classic-*-v0)
import gymnasium as gym
import torch
from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper
from contractionRL.tasks.direct.classic.segway.env import SegwayEnv

NUM_ENVS = 8
# pitch 1.0 is INSIDE the box (pi/3 = 1.047), but the rate carries it out in one
# dt: 1.0 + 3.0*0.03 = 1.09. Setting pitch alone would not move at all — pitch
# only changes through pitch_rate.
OUT_OF_BOX = torch.tensor([0.0, 1.0, 0.0, 3.0])


def _env(**kw):
    return SegwayEnv(num_envs=NUM_ENVS, device="cpu", **kw)


# ── 1. off by default ────────────────────────────────────────────────────── #
env = _env()
assert env.terminate_out_of_box is False
assert env.X_TERMINATION_MIN is not None, "constants must exist even when off"
env.reset()
# Slam the state far outside the box; with the feature off nothing ends early.
env.x_t[:] = torch.tensor([100.0, 3.0, 50.0, 50.0])
_, _, term, trunc, info = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert not term.any() and not trunc.any(), "default behaviour changed"
assert "episode_ended_early" not in info, "flag leaks when the feature is off"
print("1. off by default                     ok")

# ── 2. on -> truncation, not termination ─────────────────────────────────── #
env = _env()
env.set_terminate_out_of_box(True)
env.reset()
env.x_t[:] = OUT_OF_BOX
_, _, term, trunc, info = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert info["episode_ended_early"].all(), "pitch ran past pi/3 but nothing fired"
assert trunc.all(), "excursion must be reported as truncation"
assert not term.any(), "terminated zeroes the GAE bootstrap = suicide bonus"
assert (env.time_steps == 0).all(), "env must have auto-reset"
print("2. on -> truncated, not terminated    ok")

# opt-in terminal flag still available for anyone who wants it
env = _env()
env.set_terminate_out_of_box(True, terminal=True)
env.reset()
env.x_t[:] = OUT_OF_BOX
_, _, term, _, _ = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert term.all(), "terminal=True must report on `terminated`"
print("2b. terminal=True honoured            ok")

# ── 3. a box wider than [X_MIN, X_MAX] raises ────────────────────────────── #
env = _env()
env.X_TERMINATION_MAX = env.X_MAX + 1.0
try:
    env.set_terminate_out_of_box(True)
except ValueError as e:
    assert "inside [X_MIN, X_MAX]" in str(e)
    print("3. wider-than-state-box raises        ok")
else:
    raise AssertionError("a bound outside the clamp is a silent no-op — must raise")

# ── 4. StatManager drops early-ended episodes ────────────────────────────── #
# Through the REAL wrapper stack (train.py:737): StatManagerEnvWrapper only ever
# sees the flat observation the skrl wrapper produces, never the env's Dict.
# Half the envs are held on the reference (they run the full horizon), half are
# kicked out of the box, so those slots must be excluded rather than padded.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "skrl"))
from train_utils import BatchedGymnasiumWrapper  # noqa: E402

raw = gym.make("classic-segway-v0", num_envs=NUM_ENVS, device="cpu")
env = raw.unwrapped
env.set_terminate_out_of_box(True)
wrapped = StatManagerEnvWrapper(BatchedGymnasiumWrapper(raw), num_envs_for_eval=NUM_ENVS)
wrapped.reset()
zero_u = torch.zeros(NUM_ENVS, env.num_dim_control)
kick = torch.zeros(NUM_ENVS, dtype=torch.bool)
kick[: NUM_ENVS // 2] = True
for t in range(env.max_episode_len + 2):
    # Hold the survivors on the reference (error ~ 0) and shove the rest out.
    env.x_t[~kick] = env.xref[~kick, min(t, env.max_episode_len - 1)]
    env.x_t[kick] = OUT_OF_BOX
    wrapped.step(zero_u)
    if wrapped._compute_count > 0:
        break

summary = wrapped.stability_summary()
assert wrapped._compute_count > 0, "buffer never reduced"
frac = summary.get("early_end_frac")
assert frac is not None, "early_end_frac must always be reported"
assert frac > 0.0, f"kicked envs were not excluded (early_end_frac={frac})"
if wrapped._recent_valid_n:
    assert "auc_mean" in summary, "survivors must still produce metrics"
    for k in ("auc_mean", "contraction_rate_mean", "overshoot_mean"):
        assert summary[k] == summary[k], f"{k} is NaN — invalid rows leaked in"
else:
    assert set(summary) == {"early_end_frac"}, \
        "with no survivors the metrics must be ABSENT, not stale"
print(f"4. early-ended slots excluded         ok "
      f"(early_end_frac={frac:.2f}, valid={wrapped._recent_valid_n})")

# ── 5. feature OFF is the old behaviour, metric for metric ───────────────── #
# The regression that matters most: every published number was produced with no
# termination box, so with the flag off nothing may be dropped and every metric
# must still be reported.
raw = gym.make("classic-segway-v0", num_envs=NUM_ENVS, device="cpu")
env = raw.unwrapped
wrapped = StatManagerEnvWrapper(BatchedGymnasiumWrapper(raw), num_envs_for_eval=NUM_ENVS)
wrapped.reset()
for _ in range(env.max_episode_len + 2):
    wrapped.step(torch.zeros(NUM_ENVS, env.num_dim_control))
    if wrapped._compute_count > 0:
        break
summary = wrapped.stability_summary()
assert wrapped._compute_count > 0, "buffer never reduced with the feature off"
assert summary["early_end_frac"] == 0.0, "nothing may be dropped when the box is off"
assert wrapped._recent_valid_n == NUM_ENVS, "every slot must count when the box is off"
for k in ("auc_mean", "contraction_rate_mean", "overshoot_mean", "contraction_score_mean"):
    assert k in summary and summary[k] == summary[k], f"{k} missing/NaN with the feature off"
print(f"5. off == previous behaviour          ok (auc={summary['auc_mean']:.3f})")

print("\nall checks passed")

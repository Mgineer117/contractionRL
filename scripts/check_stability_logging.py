"""Check the new logging keys exist, are finite, and that lambda_raw is genuinely
uncensored while lambda is not.

Builds a StatManagerEnvWrapper with synthetic error trajectories rather than
running an env: 3 groups, one contracting fast enough to exceed the lambda=10
ceiling, one growing (raw lambda < 0), one ordinary. If the clamp is doing what
the comment claims, lambda_clip_hi_frac and lambda_clip_lo_frac are both > 0 and
lambda_raw_min < 0 < 10 < lambda_raw_max while lambda_min/max stay inside [0,10].
"""
import numpy as np
import torch
from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper

N, T, DT = 60, 50, 0.02
w = StatManagerEnvWrapper.__new__(StatManagerEnvWrapper)
w._num_envs_for_eval = N
w._max_ep_len = T
w._dt = DT
w._device = lambda: torch.device("cpu")

t = torch.arange(T, dtype=torch.float32) * DT
errs = torch.zeros(N, T)
for i in range(N):
    if i % 3 == 0:          # very fast decay -> raw lambda well above 10
        errs[i] = torch.exp(-40.0 * t)
    elif i % 3 == 1:        # growing -> raw lambda below 0
        errs[i] = 0.5 * torch.exp(+3.0 * t)
    else:                   # ordinary
        errs[i] = torch.exp(-2.0 * t)
errs = errs.clamp(min=1e-8)

w._eval_buffer = errs
w._time_buffer = t.unsqueeze(0).repeat(N, 1)
w._tracking_steps = torch.full((N,), T, dtype=torch.long)

m = w._metric_set(errs)

need = ["lambda_raw_mean", "lambda_clip_lo_frac", "lambda_clip_hi_frac",
        "n_eval", "n_running_lambda", "running_lambda_frac"]
for base in ("lambda", "lambda_raw", "running_lambda", "score", "peak", "auc"):
    need += [f"{base}_{k}" for k in
             ("median", "p05", "p25", "p75", "p95", "min", "max", "std")]

missing = [k for k in need if k not in m]
assert not missing, f"MISSING keys: {missing}"

bad = [k for k in need if not np.isfinite(float(m[k]))]
assert not bad, f"NON-FINITE: {bad}"

# Only the CEILING can bind. The floor cannot: C is fitted so that
# C >= max_t x_worst(t)*exp(lbd*t) for lbd drawn from linspace(0.01, 10), hence
# (ln C - ln x(t))/t >= lbd >= 0.01 for the worst curve and is more positive for
# every better one -- so min_lambdas > 0 by construction and clamp(min=0) is
# inert. clamp(max=10) is the one that censors, and it is NOT inert.
assert m["lambda_clip_hi_frac"] > 0, "fixture should exceed the lambda=10 ceiling"
assert m["lambda_clip_lo_frac"] == 0, "floor should be unreachable by construction"
assert m["lambda_raw_max"] > 10.0, f'raw max {m["lambda_raw_max"]} should exceed 10'
assert m["lambda_max"] <= 10.0 + 1e-6, "clamped escaped the ceiling"
# and the censoring materially moves the reported mean
assert abs(m["lambda_raw_mean"] - m["lambda_mean"]) > 1e-3, \
    "raw and clamped means identical -- clamp not exercised"

# _dist carries the vectors for post-hoc quantiles
for k in ("auc", "lambda", "lambda_raw", "running_lambda", "score", "peak"):
    assert k in m["_dist"], f"_dist missing {k}"

print(f"PASS {len(need)} logging keys present, finite")
print(f"  clip_lo_frac={m['lambda_clip_lo_frac']:.3f}  "
      f"clip_hi_frac={m['lambda_clip_hi_frac']:.3f}")
print(f"  lambda_mean={m['lambda_mean']:.4f} (clamped)  vs  "
      f"lambda_raw_mean={m['lambda_raw_mean']:.4f} (uncensored)")
print(f"  raw range [{m['lambda_raw_min']:.3f}, {m['lambda_raw_max']:.3f}]  "
      f"clamped range [{m['lambda_min']:.3f}, {m['lambda_max']:.3f}]")
print(f"  running_lambda_frac={m['running_lambda_frac']:.3f} "
      f"(n={m['n_running_lambda']}/{m['n_eval']})")

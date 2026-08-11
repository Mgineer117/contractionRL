"""Check the discounted-return accumulator against a closed-form value.

Two envs, constant reward 1.0, episodes of length T. Then
    G = sum_{t=0}^{T-1} gamma^t = (1 - gamma^T) / (1 - gamma)
which is exact, so any off-by-one in the exponent or a failure to reset the
gamma^t factor at the episode boundary shows up immediately.

The reset is the part worth testing: if gamma^t kept counting across episodes it
would underflow to 0 within a few hundred steps at gamma=0.99 and every later
episode would silently report G = 0.
"""
import torch

from contractionRL.agents.skrl.contraction_metrics import StatManagerEnvWrapper

GAMMA, T, N_EP = 0.9, 10, 3
w = StatManagerEnvWrapper.__new__(StatManagerEnvWrapper)
w._device = lambda: torch.device("cpu")
w.set_discount_factor(GAMMA)

for _ in range(N_EP):
    for t in range(T):
        last = t == T - 1
        w._track_discounted_return(
            torch.ones(2),
            torch.tensor([last, last]),
            torch.tensor([False, False]),
        )

s = w.discounted_return_summary()
want = (1.0 - GAMMA ** T) / (1.0 - GAMMA)
print(f"episodes recorded : {int(s['discounted_return_n'])} (expect {N_EP * 2})")
print(f"mean discounted G : {s['discounted_return_mean']:.6f}")
print(f"closed form       : {want:.6f}")
assert int(s["discounted_return_n"]) == N_EP * 2, "wrong episode count"
assert abs(s["discounted_return_mean"] - want) < 1e-5, "G does not match closed form"
assert abs(s["discounted_return_min"] - want) < 1e-5, "later episodes drifted"
assert abs(s["discounted_return_max"] - want) < 1e-5, "later episodes drifted"
print("PASS G matches sum_t gamma^t exactly, and every episode agrees")
print("     -> gamma^t resets at the episode boundary (no cross-episode underflow)")

# Absent, not 0.0, before any episode completes.
w2 = StatManagerEnvWrapper.__new__(StatManagerEnvWrapper)
w2._device = lambda: torch.device("cpu")
w2.set_discount_factor(GAMMA)
w2._track_discounted_return(torch.ones(2), torch.tensor([False, False]),
                            torch.tensor([False, False]))
assert w2.discounted_return_summary() == {}, "should be EMPTY before an episode ends"
print("PASS summary is empty (key absent) until an episode finishes")

"""Estimate sweep memory from two GRU windows and the PPO minibatch size.

Uses a historical empirical memory coefficient with headroom; placement remains
an estimate, not a simulator allocation guarantee.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

import contractionRL.tasks.direct.common.tracking_tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402

# Calibrated: a GRU trial at B=6144, seq=2000 tried to allocate 53.58 GiB.
GIB_PER_BSEQ = 53.58 / (6144 * 2000)
HEADROOM = 1.15          # the allocation lands on top of whatever the run holds

# Partitions this account can reach, with USABLE GiB per card (nominal minus the
# ~2 GiB driver/context) and the account each bills to. Smallest first, so a job
# takes the cheapest card that fits rather than the biggest one free.
FLEET = [
    ("eng-research-gpu", "huytran1-ae-eng", "A10",  22.0),
    ("IllinoisComputes-GPU", "huytran1-ic",  "A100", 38.0),
    ("csl",              "csl",             "L40S", 44.0),
]


def need_gib(task: str, encoder_window: int, num_envs: int, rollouts: int,
             min_mini_batches: int) -> tuple[float, int, int, int]:
    """Estimate a context-enabled trial with both past and future windows."""
    env = gym.make(task, num_envs=1, device="cpu").unwrapped
    ep = int(env.episode_len)
    env.close()
    seq = 2 * encoder_window
    B = max(1, int(rollouts * num_envs / max(1, min_mini_batches)))
    return B * seq * GIB_PER_BSEQ * HEADROOM, ep, seq, B


def place(gib: float, agents: int):
    """Cheapest card that fits `agents` concurrent trials of this size."""
    for part, acct, name, cap in FLEET:
        if gib * agents <= cap:
            return part, acct, name, cap
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--algorithm", default="c2rl-ppo-cvstem")
    p.add_argument("--rollouts", type=int, default=24,
                   help="agent.rollouts from the env yaml; sets the PPO batch")
    p.add_argument("--agents", type=int, default=2, help="concurrent trials per GPU")
    p.add_argument("--shell", action="store_true",
                   help="emit `PARTITION ACCOUNT AGENTS` for a launcher to read")
    args = p.parse_args()

    with open(os.path.join(_ROOT, "search/configs", f"{args.algorithm}.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    params = cfg['parameters']
    windows = params.get('models.encoder.encoder_window', {'value': None})
    windows = windows.get('values', [windows.get('value')])
    gammas = params.get('agent.discount_factor', {'value': .99})
    gammas = gammas.get('values', [gammas.get('value', .99)])
    encoder_window = max(math.ceil(round(1/(1-gamma), 10)) if value is None else value
                         for value in windows for gamma in gammas)
    mbs = cfg["parameters"].get("agent.mini_batches", {})
    min_mb = min(mbs.get("values", [mbs.get("value", 4)]))
    num_envs = int(cfg.get("num_envs", 1024))

    tasks = ([args.env] if args.env else
             [k for k in gym.registry if k.startswith("classic-")])
    if not args.shell:
        print(f"worst case = two windows of {encoder_window} x smallest mini_batches "
              f"{min_mb}, num_envs {num_envs}, rollouts {args.rollouts}, "
              f"{HEADROOM:g}x headroom")
        print(f"{'env':<24} {'episode':>8} {'seq':>6} {'B':>7} {'GiB/trial':>10}  placement")
    rc = 0
    for t in tasks:
        gib, ep, seq, B = need_gib(t, encoder_window, num_envs, args.rollouts, min_mb)
        hit = place(gib, args.agents)
        if args.shell:
            if not args.env:
                continue
            solo = hit or place(gib, 1)
            if solo:
                print(f"{solo[0]} {solo[1]} {args.agents if hit else 1}")
            else:
                print("NONE NONE 0")
                rc = 1
            continue
        if hit:
            print(f"{t:<24} {ep:>8} {seq:>6} {B:>7} {gib:>10.1f}  "
                  f"{hit[0]} ({hit[2]}, {hit[3]:g} GiB) x{args.agents}")
        else:
            solo = place(gib, 1)
            if solo:
                print(f"{t:<24} {ep:>8} {seq:>6} {B:>7} {gib:>10.1f}  "
                      f"{solo[0]} ({solo[2]}) x1 only")
            else:
                biggest = FLEET[-1]
                need_ne = int(num_envs * biggest[3] / gib)
                print(f"{t:<24} {ep:>8} {seq:>6} {B:>7} {gib:>10.1f}  "
                      f"NO CARD FITS (max {biggest[2]} {biggest[3]:g}) -> "
                      f"num_envs <= {need_ne}")
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""Pick a GPU big enough for the worst trial the sweep can sample.

The footprint is driven by the GRU/attn encoder over the reference window, and it
is set by two things the sweep samples INDEPENDENTLY:

    seq = ceil(episode_len / stride)                  the window it unrolls
    B   = rollouts * num_envs / mini_batches          the PPO mini-batch it sees

and the cost is ~linear in their PRODUCT. An earlier version of this file priced
memory per retained point and ignored B entirely; it predicted 28.8 GiB for
segway at stride 1 and the real allocation was 53.58 GiB, so segway OOM'd again
on a 44 GiB L40S. B is not a detail, it is half the product.

GIB_PER_BSEQ is calibrated from that failure -- 53.58 GiB at B=6144, seq=2000 --
rather than derived from tensor shapes, so it already contains the gate count,
the cuDNN workspace and the autograd tape.

The WORST case is what has to fit, not the average: bayes will eventually sample
the densest stride together with the smallest mini_batches, and that trial has to
run rather than take its GPU-mates down with it.

    python scripts/gpu_for_env.py --all
    python scripts/gpu_for_env.py --env classic-segway-v0 --shell
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "contractionRL"))

import contractionRL.tasks.direct.classic  # noqa: E402,F401
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


def need_gib(task: str, min_stride: int, num_envs: int, rollouts: int,
             min_mini_batches: int) -> tuple[float, int, int, int]:
    """Worst-case GiB for one trial: densest stride x smallest mini_batches."""
    env = gym.make(task, num_envs=1, device="cpu").unwrapped
    ep = int(env.episode_len)
    seq = math.ceil(ep / max(1, min_stride))
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
    strides = cfg["parameters"]["xref_encoder_stride"].get(
        "values", [cfg["parameters"]["xref_encoder_stride"].get("value", 1)])
    min_stride = min(strides)
    mbs = cfg["parameters"].get("agent.mini_batches", {})
    min_mb = min(mbs.get("values", [mbs.get("value", 4)]))
    num_envs = int(cfg.get("num_envs", 1024))

    tasks = ([args.env] if args.env else
             [k for k in gym.registry if k.startswith("classic-")])
    if not args.shell:
        print(f"worst case = densest stride {min_stride} x smallest mini_batches "
              f"{min_mb}, num_envs {num_envs}, rollouts {args.rollouts}, "
              f"{HEADROOM:g}x headroom")
        print(f"{'env':<24} {'episode':>8} {'seq':>6} {'B':>7} {'GiB/trial':>10}  placement")
    rc = 0
    for t in tasks:
        gib, ep, seq, B = need_gib(t, min_stride, num_envs, args.rollouts, min_mb)
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

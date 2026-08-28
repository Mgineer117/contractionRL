"""Pick a GPU big enough to hold an env's densest reference window.

The sweep's memory footprint is not a property of the algorithm, it is a
property of the ENV crossed with the stride the trial happened to sample, and
those differ by 5x across the envs one search config serves. Placing jobs by
hand gets this wrong in exactly one direction -- segway's 2000-step episode at
stride 1 needs 23.6 GiB and was scheduled onto a 22.06 GiB A10, where a trial
tried to allocate 26.80 GiB and died.

    retained points = ceil(episode_len / min(stride))
    VRAM           ~ points * MIB_PER_POINT_1024 * (num_envs / 1024)

MIB_PER_POINT_1024 is measured, not derived: attn at stride 1 on a 500-point
window peaked at 5.9 GiB with num_envs=1024, i.e. ~11.8 MiB per retained point.
It is a sizing heuristic and deliberately paired with HEADROOM below rather than
trusted exactly -- the observed peak on segway (26.8 GiB attempted) ran above the
23.6 GiB this predicts, because a single allocation lands on top of whatever the
run already holds.

    python scripts/gpu_for_env.py --env classic-segway-v0
    python scripts/gpu_for_env.py --all
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

MIB_PER_POINT_1024 = 11.8
HEADROOM = 1.25          # the prediction ran ~13% under the observed peak; round up

# Partitions this account can reach, with USABLE GiB per card (nominal minus the
# ~2 GiB the driver and context hold) and the account each one bills to. Ordered
# smallest-first so a job takes the cheapest card that fits rather than the
# biggest one available.
FLEET = [
    ("eng-research-gpu", "huytran1-ae-eng", "A10",  22.0),
    ("csl",              "csl",             "L40S", 45.0),
    ("IllinoisComputes-GPU", "huytran1-ic",  "A100", 38.0),
]


def need_gib(task: str, min_stride: int, num_envs: int) -> tuple[float, int, int]:
    env = gym.make(task, num_envs=1, device="cpu").unwrapped
    ep = int(env.episode_len)
    pts = math.ceil(ep / max(1, min_stride))
    gib = pts * MIB_PER_POINT_1024 * (num_envs / 1024.0) / 1024.0 * HEADROOM
    return gib, ep, pts


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
    p.add_argument("--agents", type=int, default=2, help="concurrent trials per GPU")
    p.add_argument("--shell", action="store_true",
                   help="emit `PARTITION ACCOUNT AGENTS` for a launcher to read")
    args = p.parse_args()

    with open(os.path.join(_ROOT, "search/configs", f"{args.algorithm}.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    strides = cfg["parameters"]["xref_encoder_stride"].get(
        "values", [cfg["parameters"]["xref_encoder_stride"].get("value", 1)])
    min_stride = min(strides)
    num_envs = int(cfg.get("num_envs", 1024))

    tasks = ([args.env] if args.env else
             [k for k in gym.registry if k.startswith("classic-")])
    if not args.shell:
        print(f"densest stride {min_stride}, num_envs {num_envs}, "
              f"{args.agents} agents/GPU, {HEADROOM:g}x headroom")
        print(f"{'env':<24} {'episode':>8} {'points':>7} {'GiB/trial':>10} "
              f"{'GiB/GPU':>8}  placement")
    rc = 0
    for t in tasks:
        gib, ep, pts = need_gib(t, min_stride, num_envs)
        hit = place(gib, args.agents)
        if args.shell:
            if args.env and hit:
                print(f"{hit[0]} {hit[1]} {args.agents}")
            elif args.env:
                # Nothing fits at this packing; one agent is the last resort.
                solo = place(gib, 1)
                if solo:
                    print(f"{solo[0]} {solo[1]} 1")
                else:
                    print("NONE NONE 0")
                    rc = 1
            continue
        if hit:
            print(f"{t:<24} {ep:>8} {pts:>7} {gib:>10.1f} {gib*args.agents:>8.1f}"
                  f"  {hit[0]} ({hit[2]}, {hit[3]:g} GiB)")
        else:
            solo = place(gib, 1)
            alt = (f"  -> only fits 1 agent on {solo[0]} ({solo[2]})" if solo
                   else "  -> DOES NOT FIT ANY CARD; raise the min stride or cut num_envs")
            print(f"{t:<24} {ep:>8} {pts:>7} {gib:>10.1f} {gib*args.agents:>8.1f}{alt}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

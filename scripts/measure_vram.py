"""Measure PEAK GPU VRAM of a c2rl-ppo run per environment, to size runs-per-GPU.

Runs a SHORT training for each env as a subprocess and polls
``nvidia-smi --query-compute-apps`` for that PID's memory, tracking the maximum.
Reads the real process footprint rather than ``torch.cuda.max_memory_allocated``
on purpose: allocated bytes exclude the CUDA context (~250-600 MiB), the caching
allocator's reserved-but-unused blocks, and cuDNN/cuBLAS workspaces -- all of
which occupy the card and therefore decide how many runs actually fit.

The peak is what matters, and it does NOT occur during the PPO loop: Phase A's
CMG regression does an ``eigh`` over the whole batch, and that transient is
typically the high-water mark. So the run must get past Phase A to be measured
honestly, which is why this does not simply sample the steady state.

    python scripts/measure_vram.py --envs car car_weak segway cartpole
    python scripts/measure_vram.py --envs segway --ref-length 500 --timesteps 6000

Prints a per-env table plus how many concurrent runs fit on a given card.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_total_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def proc_mib(pids: set[int]) -> int:
    """Summed VRAM of the given PIDs, per nvidia-smi. 0 if none are on the GPU."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    tot = 0
    for line in out.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
        if m and int(m.group(1)) in pids:
            tot += int(m.group(2))
    return tot


def descendants(pid: int) -> set[int]:
    """pid plus its children -- train.py may spawn workers that hold VRAM too."""
    found = {pid}
    try:
        out = subprocess.run(["ps", "-o", "pid=,ppid="], capture_output=True,
                             text=True).stdout
        kids: dict[int, list[int]] = {}
        for line in out.strip().splitlines():
            p, pp = (int(x) for x in line.split())
            kids.setdefault(pp, []).append(p)
        stack = [pid]
        while stack:
            cur = stack.pop()
            for k in kids.get(cur, []):
                if k not in found:
                    found.add(k)
                    stack.append(k)
    except Exception:
        pass
    return found


def measure(env: str, *, timesteps: int, ref_length: int, num_envs: int,
            gamma: float, poll: float, timeout: float) -> dict:
    cmd = [sys.executable, "scripts/skrl/train.py", "--classic",
           "--task", env, "--algorithm", "c2rl-ppo",
           "--num_timesteps", str(timesteps), "--discount_factor", str(gamma),
           "--seed", "0", "--ref_length", str(ref_length),
           "--num_envs", str(num_envs), "--no_wandb"]
    envv = dict(os.environ)
    envv["PYTHONPATH"] = os.path.join(ROOT, "source", "contractionRL")
    p = subprocess.Popen(cmd, cwd=ROOT, env=envv, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    peak, t0, seen_gpu = 0, time.time(), False
    while p.poll() is None:
        if time.time() - t0 > timeout:
            p.kill()
            break
        cur = proc_mib(descendants(p.pid))
        if cur > 0:
            seen_gpu = True
        peak = max(peak, cur)
        time.sleep(poll)
    tail = ""
    with contextlib.suppress(Exception):
        tail = "".join((p.stdout.read() or "").splitlines(keepends=True)[-4:])
    return {"env": env, "peak_mib": peak, "rc": p.returncode,
            "seen_gpu": seen_gpu, "elapsed_s": time.time() - t0, "tail": tail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=["car", "car_weak", "segway", "cartpole"])
    ap.add_argument("--timesteps", type=int, default=6000,
                    help="short, but MUST clear Phase A -- that is where the peak is")
    ap.add_argument("--ref-length", type=int, default=500,
                    help="match the experiment: pinned window => widest observation")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="highest gamma; with a PINNED window this does not change "
                         "the observation, but it does change ref-window compute")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--cards", nargs="+", type=int, default=[24564, 49152, 81559],
                    help="card sizes (MiB) to report packing for: A10 24G, L40S 48G, H100 80G")
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found — cannot measure", file=sys.stderr)
        return 2
    total = gpu_total_mib()
    print(f"local GPU total {total} MiB; baseline other-process usage "
          f"{proc_mib(set(range(1, 2**22))) } MiB")
    print(f"protocol: ref_length={args.ref_length} num_envs={args.num_envs} "
          f"gamma={args.gamma} timesteps={args.timesteps}\n")

    rows = []
    for e in args.envs:
        task = e if e.startswith("classic-") else f"classic-{e}-v0"
        print(f"[measure] {task} ...", flush=True)
        r = measure(task, timesteps=args.timesteps, ref_length=args.ref_length,
                    num_envs=args.num_envs, gamma=args.gamma, poll=args.poll,
                    timeout=args.timeout)
        rows.append(r)
        status = "ok" if r["rc"] == 0 else f"rc={r['rc']}"
        print(f"    peak {r['peak_mib']} MiB   {status}   {r['elapsed_s']:.0f}s")
        if r["rc"] != 0 and r["tail"]:
            print("    tail:", r["tail"].strip()[:300])

    print(f"\n{'env':16s} {'peak MiB':>9s} " +
          " ".join(f"{'/' + str(c // 1024) + 'G':>7s}" for c in args.cards))
    for r in rows:
        pk = r["peak_mib"]
        fits = [str(c // pk) if pk > 0 else "-" for c in args.cards]
        print(f"{r['env']:16s} {pk:>9d} " + " ".join(f"{f:>7s}" for f in fits))
    ok = [r for r in rows if r["rc"] == 0 and r["peak_mib"] > 0]
    if ok:
        worst = max(ok, key=lambda r: r["peak_mib"])
        print(f"\nworst env: {worst['env']} at {worst['peak_mib']} MiB")
        print("Size a shared card by the WORST env, not the mean -- one OOM takes "
              "down whatever else is packed on that GPU.")
        for c in args.cards:
            print(f"  {c // 1024}G card -> {c // worst['peak_mib']} concurrent runs "
                  f"(leave one slot spare for fragmentation)")
    bad = [r for r in rows if r["rc"] != 0 or r["peak_mib"] == 0]
    if bad:
        print("\nNOT MEASURED (treat as unknown, do not assume):")
        for r in bad:
            why = "never appeared on the GPU" if r["peak_mib"] == 0 else f"rc={r['rc']}"
            print(f"  {r['env']}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

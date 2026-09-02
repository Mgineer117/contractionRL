"""V^pi - V* for a trained checkpoint: Monte-Carlo return against the global optimum.

    V*    exhaustive value iteration -- the optimum of the discretised MDP
    V^pi  the REALISED discounted return of the policy on the same references

``V^pi`` is Monte Carlo on purpose. The critic is a biased estimate of the
policy's value -- it is fitted, on-policy, and to a moving target -- so a gap
computed from it mixes the policy's suboptimality with the critic's fit error
and cannot separate them. The realised return has neither problem: with
deterministic dynamics and the policy MEAN as the action it is not an estimate
of ``V^pi``, it IS ``V^pi``.

The critic is still printed, as a diagnostic of exactly that bias -- but it is
not the comparison.

    python scripts/policy_gap.py --task toy-mg-v0
    python scripts/policy_gap.py --task classic-segway-v0

Needs a V* pack (scripts/precompute_vstar.py) and a trained checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))
sys.path.insert(0, str(REPO / "scripts" / "skrl"))

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402


def _find_checkpoint(task: str, algorithm: str) -> str:
    """Newest ``best_agent.pt`` whose run reports this task.

    Run directories are named by timestamp only, so the task lives in each run's
    eval_results.json and nowhere else -- globbing the algorithm's log tree
    without this check silently scores another env's policy.
    """
    hits = []
    for p in glob.glob(str(REPO / "logs/classic" / algorithm / "*/*/eval_results.json")):
        try:
            with open(p) as fh:
                if json.load(fh).get("task") != task:
                    continue
        except (json.JSONDecodeError, OSError):
            continue
        ck = os.path.join(os.path.dirname(p), "checkpoints", "best_agent.pt")
        if os.path.exists(ck):
            hits.append(ck)
    if not hits:
        raise SystemExit(
            f"no {algorithm} checkpoint for {task}. Train one first:\n"
            f"    python scripts/skrl/train.py --classic --task {task} "
            f"--algorithm {algorithm} --headless")
    return max(hits, key=os.path.getmtime)


def _agent_cfg(task: str, algorithm: str) -> dict:
    entry = (gym.spec(task).kwargs or {})[
        f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"]
    pkg, fname = entry.split(":")
    with open(os.path.join(os.path.dirname(importlib.import_module(pkg).__file__),
                           fname)) as fh:
        return yaml.safe_load(fh)


def _psi_width(sd: dict) -> int | None:
    for k, w in (sd.get("value") or {}).items():
        if k.endswith("psi.mlp.net.0.weight") or k.endswith("psi.rnn.weight_ih_l0"):
            return int(w.shape[1])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--algorithm", default="c2rl-ppo")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    import contractionRL.tasks.direct.classic  # noqa: F401
    import contractionRL.tasks.direct.toy  # noqa: F401
    import play
    from contractionRL import cm_data
    from contractionRL.agents.skrl.contraction_metrics import optimality_gap

    key = a.task.removeprefix("classic-").removeprefix("toy-").removesuffix("-v0")
    pack_dir = "toy" if (REPO / "data/toy" / key).exists() else "classic"
    root = REPO / "data" / pack_dir / key
    pack_path = next((q for q in (root / "global.npz", root / "vstar.npz")
                      if q.exists()), root / "global.npz")
    if not pack_path.exists():
        raise SystemExit(f"no V* pack at {pack_path}. Solve it first:\n"
                         f"    python scripts/precompute_vstar.py --task {a.task}")
    pack = np.load(pack_path)

    cfg = _agent_cfg(a.task, a.algorithm)
    cfg["seed"] = 42
    ckpt = a.checkpoint or _find_checkpoint(a.task, a.algorithm)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    # Reference-window length is recorded nowhere, so recover it from the
    # checkpoint: build at the default, then rescale by the psi input widths.
    # Guessing it ends in a state_dict shape mismatch every time.
    env, runner = play.build_classic_eval_runner(
        a.task, a.algorithm, cfg, device=a.device, num_envs=int(pack["groups"])
        * int(pack["envs_per_group"]), num_envs_for_eval=8)
    want = _psi_width(sd)
    have = _psi_width({"value": dict(runner.agent._rl_agent.value.state_dict())})
    if want and have and want != have:
        length = round(env.unwrapped.ref_window.length * want / have)
        print(f"[gap] checkpoint was trained with ref_length={length} "
              f"(psi input {want} vs {have}); rebuilding.")
        env.close()
        env, runner = play.build_classic_eval_runner(
            a.task, a.algorithm, cfg, device=a.device,
            num_envs=int(pack["groups"]) * int(pack["envs_per_group"]),
            num_envs_for_eval=8, ref_length=length)
    runner.agent.load(ckpt)
    for m in runner.agent.models.values():
        if m is not None:
            m.eval()
    raw = env.unwrapped

    def _act(obs):
        obs_t = raw.ref_window.flatten(
            *(torch.as_tensor(obs[k], dtype=torch.float32, device=a.device)
              for k in ("x", "xrefs", "urefs")))
        actions, outputs = runner.agent.act(obs_t, None, timestep=0, timesteps=0)
        return outputs.get("mean_actions", actions)

    # C2RL's metric is injected by the TRAINER, which never runs here, so the
    # eval env starts with no metric and get_rewards would score the rollout
    # under the plain -q||e||^2 branch instead of the objective V* was solved
    # under. optimality_gap refuses that outright; attaching is the fix.
    cm_data.attach_metric(
        raw, key, device=a.device, tag="[gap]",
        **{k: cfg["cm"][k] for k in cm_data.REWARD_KEYS if k in cfg.get("cm", {})})
    print(f"[gap] {a.task} / {a.algorithm}\n[gap] checkpoint {ckpt}")
    out = optimality_gap(raw, _act, pack, device=a.device, tag="[gap]")

    # The critic, for contrast only. Its distance from the MC return is the bias
    # that makes it the wrong instrument for this measurement.
    try:
        rl = runner.agent._rl_agent
        obs, _ = raw.reset()
        obs_t = raw.ref_window.flatten(
            *(torch.as_tensor(obs[k], dtype=torch.float32, device=a.device)
              for k in ("x", "xrefs", "urefs")))
        pre = getattr(rl, "_observation_preprocessor", None) or (lambda t, **k: t)
        vpre = getattr(rl, "_value_preprocessor", None)
        with torch.no_grad():
            v, _ = rl.value.act({"observations": pre(obs_t, train=False)}, role="value")
            v = (vpre(v, inverse=True) if vpre is not None else v).double().reshape(-1)
        print(f"[gap] critic mean {float(-v.mean()):+.6e} in cost units vs MC V^pi "
              f"{out['v_pi_mean']:+.6e} -- difference {float(-v.mean()) - out['v_pi_mean']:+.3e} "
              f"is critic BIAS, which is why the gap above does not use it.")
    except Exception as e:  # noqa: BLE001
        print(f"[gap] critic diagnostic unavailable ({type(e).__name__}: {e})")

    if a.save:
        dest = REPO / "data" / pack_dir / key / f"policy_gap_{a.algorithm}.json"
        dest.write_text(json.dumps({**out, "checkpoint": ckpt, "task": a.task}, indent=2))
        print(f"[gap] wrote {dest}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

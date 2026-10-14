"""Locate and attach a shipped ``cm_data_*.npz`` contraction metric.

Three call sites used to glob for these files themselves, each with its own
family/path logic, and each free to disagree about which of several builds
counts as "the" dataset. They now share this, so a figure, a taxonomy row and a
value-iteration cost are all reading the SAME metric.
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3] / "data"


def find_npz(name: str) -> pathlib.Path:
    """Newest ``cm_data_*.npz`` for env short name ``name``, in the classic family.

    Newest, not first-alphabetically: the sorted-glob the call sites used picks
    by lbd string, so a re-derived rate silently loses to a stale higher one.
    """
    hits = [pathlib.Path(p) for fam in ("classic",)
            for p in glob.glob(str(ROOT / fam / name / "cm_data_*.npz"))]
    if not hits:
        raise FileNotFoundError(
            f"no CM dataset for '{name}'. Expected data/classic/{name}/"
            f"cm_data_*.npz -- build it with:\n"
            f"    python scripts/skrl/c2rl/build_cm_dataset.py --task <task-id>")
    return max(hits, key=lambda p: p.stat().st_mtime)


def attach_cmg(env, name: str | None = None, *, epochs: int = 80,
               device: str = "cpu", tag: str = "[cm_data]", **set_ccm_kw) -> dict:
    """Regress the shipped ``W(x)`` into the CMG C2RL deploys and set it on ``env``.

    The npz holds W at sampled states; everything downstream (the Mahalanobis
    reward, ``local_lambda``, the CV-STEM-LQR gain) needs M(x) at ARBITRARY
    states, which is what the regression provides.
    """
    from contractionRL.agents.skrl.ncm_synthesis import regress_cmg
    from contractionRL.agents.skrl.nn_modules import BoundedCCM_Generator

    path = find_npz(name or env.task)
    d = np.load(path, allow_pickle=True)
    w_lb, w_ub = float(d["w_lb"]), float(d["w_ub"])
    xd = int(env.num_dim_x)
    cmg = BoundedCCM_Generator(x_dim=xd, hidden_dim=[128, 128], activation="tanh",
                               w_lb=w_lb, w_ub=w_ub, outputs_metric=True)
    st = regress_cmg(cmg, {"x": d["x"], "W": d["W"]}, w_lb=w_lb, x_dim=xd,
                     bounded=True, epochs=epochs, lr=1e-3, batch_size=512,
                     device=device, tag=tag)
    cmg.eval()
    for q in cmg.parameters():
        q.requires_grad_(False)
    env.set_ccm(cmg, w_lb=w_lb, device=device, **set_ccm_kw)
    # Which build this is. cartpole ships an N=3000 dataset while car_v1/segway
    # use the full one; a result that silently mixes the two is unreadable later.
    env._cm_dataset = path.name
    env._migrate_r = float(d["r_scaler"])
    print(f"{tag} {path.name}: CMG attached (lbd={float(d['lbd']):.4f}, "
          f"r={env._migrate_r}, M rel-err {st.get('metric_rel_error')})", flush=True)
    return {"path": path, "lbd": float(d["lbd"]), "r_scaler": env._migrate_r,
            "w_lb": w_lb, "w_ub": w_ub, "fit": st}


REWARD_KEYS = ("tracking_scaler", "control_scaler", "reward_euclidean", "reward_level")


def attach_metric(env, name: str | None = None, *, device: str = "cpu",
                  tag: str = "[cm_data]", **reward_kw) -> dict:
    """Attach the deployed CMG metric and reward weights to a classic env."""
    result = attach_cmg(env, name, device=device, tag=tag, **reward_kw)
    return {"metric_source": "regress", "lbd": result["lbd"],
            **{k: getattr(env, k, None) for k in REWARD_KEYS}}


def reward_signature(env) -> dict:
    """What ``get_rewards`` will actually compute, as comparable scalars."""
    return {
        "uses_metric": bool(getattr(env, "ccm_gen", None) is not None),
        "tracking_scaler": float(getattr(env, "tracking_scaler", 1.0)),
        "control_scaler": float(getattr(env, "control_scaler", 0.0)),
        "reward_euclidean": bool(getattr(env, "reward_euclidean", False)),
        "reward_level": bool(getattr(env, "reward_level", False)),
    }

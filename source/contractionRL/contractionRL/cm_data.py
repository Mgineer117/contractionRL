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
    """Newest ``cm_data_*.npz`` for env short name ``name``, in either family.

    Newest, not first-alphabetically: the sorted-glob the call sites used picks
    by lbd string, so a re-derived rate silently loses to a stale higher one.
    """
    hits = [pathlib.Path(p) for fam in ("toy", "classic")
            for p in glob.glob(str(ROOT / fam / name / "cm_data_*.npz"))]
    if not hits:
        raise FileNotFoundError(
            f"no CM dataset for '{name}'. Expected data/{{toy,classic}}/{name}/"
            f"cm_data_*.npz -- build it with:\n"
            f"    python scripts/build_cm_dataset.py --task <task-id>")
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


def _check_plant(d, name: str) -> None:
    """Refuse a certificate solved for a DIFFERENT plant than the one loaded.

    The dataset's cache key is (lbd, w_lb, w_ub, r, eps, solver, N) -- the solver
    knobs. Edit f or B and the key is unchanged, so the stale npz keeps loading
    and every lam, band, envelope and optimum downstream describes the plant that
    used to be there. Nothing about the numbers looks wrong; that is the danger.
    Measured 2026-09-01 on mg, whose drift carried a stabilising -x_2 that was
    removed. An npz written before this field existed has no signature and is
    allowed through with a warning -- it cannot be checked, only rebuilt.
    """
    from contractionRL.solvers.sos_cm import PLANTS

    pl = PLANTS.get(name)
    if pl is None:
        return
    live = f"f={[str(e) for e in pl['f']]} B={[str(e) for e in pl['B']]}"
    have = str(d["plant_signature"]) if "plant_signature" in d else None
    if have is None:
        print(f"[cm_data] {name}: this metric predates plant_signature, so it "
              f"cannot be checked against the current dynamics. Rebuild it "
              f"(build_cm_dataset.py --task toy-{name}-v0 --force) if f or B "
              f"has changed since it was written.", flush=True)
        return
    if have != live:
        raise RuntimeError(
            f"[cm_data] {name}: the shipped metric was certified for a "
            f"DIFFERENT plant.\n    npz:  {have}\n    live: {live}\n"
            f"Every lambda, band and optimum read through it would describe the "
            f"old system. Rebuild:\n"
            f"    python scripts/build_cm_dataset.py --task toy-{name}-v0 --force")


def attach_metric(env, name: str | None = None, *, device: str = "cpu",
                  tag: str = "[cm_data]", **reward_kw) -> dict:
    """Attach the metric C2RL actually deploys for this env, and its reward weights.

    SOS polynomial when the dataset carries coefficients (toy), the CMG
    regression otherwise (classic). One function because the alternative is what
    happened: C2RL's CCM is injected by the TRAINER, so any eval-only path had no
    metric at all and ``get_rewards`` silently fell through to the plain
    -q||e||^2 branch -- scoring the policy on a different objective than V* was
    solved for. Measured on toy-mg: gap 247% of |V*| that way, 0.80% correctly.
    """
    d = np.load(find_npz(name or env.task))
    if "sos_coeff_names" in d:
        _check_plant(d, name or env.task)
        from contractionRL.agents.skrl.nn_modules import AnalyticSOSMetric
        coeffs = {str(k): float(v)
                  for k, v in zip(d["sos_coeff_names"], d["sos_coeff_values"])}
        m = AnalyticSOSMetric(coeffs, int(d["sos_w_degree"]), int(env.num_dim_x),
                              float(d["w_lb"]), box=(env.X_MIN, env.X_MAX))
        env.set_ccm(m, w_lb=float(d["w_lb"]), device=device, **reward_kw)
        env._cm_dataset = find_npz(name or env.task).name
        env._migrate_r = float(d["r_scaler"])
        src = "sos"
        print(f"{tag} metric = exact degree-{int(d['sos_w_degree'])} SOS polynomial",
              flush=True)
    else:
        attach_cmg(env, name, device=device, tag=tag, **reward_kw)
        src = "regress"
    return {"metric_source": src, "lbd": float(d["lbd"]),
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

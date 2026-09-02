"""State-dependent contraction rate field of a toy plant, from its SOS metric.

The certified metric of ``solvers/sos_cm.py`` is an analytic ``W(x)``, so the
local rate needs no sampling, no re-solve and no actuator gate: at every ``x``
the largest rate the DEPLOYED metric certifies is the one that makes the
certified LMI tight there,

    G(x) := -Wdot + A W + W A' - (2/r) B B',   lambda(x) = -1/2 lam_max(G, W),

a generalized eigenvalue of two 2x2 matrices. This is \\eqref{eqn:local_rate}'s
certified estimator with Wdot kept, which is the same quantity the bisection
maximized -- so min_x lambda(x) must reproduce the certified uniform rate, and
the script asserts exactly that.

    python scripts/toy_rate_field.py --task toy-mg-v0 --n 201
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))


def rate_field(key: str, n: int):
    from contractionRL.solvers import sos_cm as S

    d = np.load(sorted(glob.glob(str(REPO / "data/toy" / key / "cm_data_*.npz")))[-1])
    coeffs = {str(a): float(b) for a, b in zip(d["sos_coeff_names"], d["sos_coeff_values"])}
    W, wsyms = S._w_poly(int(d["sos_w_degree"]))
    Wn = W.subs({s: coeffs[str(s)] for s in wsyms})
    # lam=0 makes _lmi return G exactly; r must be the one that was certified.
    Gf = sp.lambdify(S.XS, S._lmi(key, 0, Wn, float(d["r_scaler"])), "numpy")
    Wf = sp.lambdify(S.XS, Wn, "numpy")

    lo, hi = d["x"].min(0), d["x"].max(0)
    g1, g2 = (np.linspace(lo[i], hi[i], n) for i in range(2))
    X1, X2 = np.meshgrid(g1, g2, indexing="ij")
    lam = np.empty((n, n))
    cond = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a, b = X1[i, j], X2[i, j]
            Wv = np.array(Wf(a, b), float).reshape(2, 2)
            Gv = np.array(Gf(a, b), float).reshape(2, 2)
            lam[i, j] = -0.5 * eigh(0.5 * (Gv + Gv.T), Wv, eigvals_only=True).max()
            cond[i, j] = np.linalg.cond(Wv)
    return dict(x1=g1, x2=g2, lam=lam, cond=cond, lbd=float(d["lbd"]),
                r_scaler=float(d["r_scaler"]), x_lo=lo, x_hi=hi)


def _envelope(X, gamma, dt, lbd, lam_of, mode):
    """The theorem's bound, beta(s_0) exp(-2 lam_bar(0,k) k dt), averaged over tasks.

    beta = (1 - gamma e^{-2 Lambda(s_0) dt})^{-1} is the overshoot the OPTIMAL
    policy is permitted in exchange for long-run return -- 1.01 at gamma=0.01
    (the overshoot-free corollary) and ~30 at gamma=0.99. Plotting exp(-2 lam t)
    instead is the CERTIFIED FEEDBACK's envelope (beta = 1) and is not a bound
    the theorem ever claimed for u*.
    """
    n_t, n_k = X.shape[0], X.shape[1]
    if mode == "uniform":
        lam_traj = np.full((n_t, n_k), lbd)
    else:
        lam_traj = lam_of(X)
    lam_bar = np.cumsum(lam_traj, 1) / np.arange(1, n_k + 1)
    beta = 1.0 / (1.0 - gamma * np.exp(-2.0 * lam_traj[:, :1] * dt))
    return (beta * np.exp(-2.0 * lam_bar * np.arange(n_k) * dt)).mean(0)


def _lam_on(key: str):
    """lam(x) as a callable on a (tasks, H, n) array -- the same generalized
    eigenvalue the field uses, so the envelope and the heat map agree."""
    from contractionRL.solvers import sos_cm as S

    d = np.load(sorted(glob.glob(str(REPO / "data/toy" / key / "cm_data_*.npz")))[-1])
    co = {str(a): float(b) for a, b in zip(d["sos_coeff_names"], d["sos_coeff_values"])}
    W, ws = S._w_poly(int(d["sos_w_degree"]))
    Wn = W.subs({s: co[str(s)] for s in ws})
    Gf = sp.lambdify(S.XS, S._lmi(key, 0, Wn, float(d["r_scaler"])), "numpy")
    Wf = sp.lambdify(S.XS, Wn, "numpy")
    n = int(np.asarray(Wf(0.0, 0.0)).shape[0])

    def lam(X):
        out = np.empty(X.shape[:2])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                a, b = X[i, j]
                Wv = np.asarray(Wf(a, b), float).reshape(n, n)
                Gv = np.asarray(Gf(a, b), float).reshape(n, n)
                out[i, j] = -0.5 * eigh(0.5 * (Gv + Gv.T), Wv, eigvals_only=True).max()
        return out
    return lam


def _lam_along(f, xr):
    """Bilinear read of the field at the reference points."""
    i = np.clip(np.interp(xr[:, 0], f["x1"], np.arange(len(f["x1"]))), 0, len(f["x1"]) - 1)
    j = np.clip(np.interp(xr[:, 1], f["x2"], np.arange(len(f["x2"]))), 0, len(f["x2"]) - 1)
    return f["lam"][np.rint(i).astype(int), np.rint(j).astype(int)]


def _optimal_rollouts(task, pk, ref_mode):
    """State trajectories AND tracking-error curves the certified u* produces.

    The pack stores u*, not x*: the moment relaxation's extracted states are
    only as good as the relaxation, whereas replaying u* through the env is the
    same feasible rollout that produced the upper bound. So what is drawn is a
    real trajectory of the plant, not a decoded moment.
    """
    if "u_star" not in pk.files:
        return None, None
    import contractionRL.tasks.direct.toy  # noqa: F401
    import gymnasium as gym
    import torch
    from contractionRL import cm_data
    key = task.removeprefix("toy-").removesuffix("-v0")
    us = np.asarray(pk["u_star"])                       # (tasks, H, u_dim) or (tasks, H)
    us = us.reshape(us.shape[0], us.shape[1], -1)
    env = gym.make(task, num_envs=us.shape[0], device="cpu",
                   reference_mode=ref_mode).unwrapped
    env.configure_ref_window(length=env.full_ref_length(int(pk["ref_offset"])),
                             offset=int(pk["ref_offset"]), gamma=float(pk["gamma"]))
    cm_data.attach_metric(env, key, device="cpu", tag="[rate]")
    env.set_group_references(pk["xref"], pk["uref"], pk["x0"])
    env.reset()
    xs, es, en = [], [], []
    for t in range(us.shape[1]):
        xr = env.xref[torch.arange(env.num_envs), env.time_steps]
        e = (env.x_t - xr).unsqueeze(-1)
        xs.append(env.x_t.clone().numpy())
        es.append(torch.norm(env.x_t - xr, dim=-1).numpy())
        # E = e' M(x) e, the RIEMANNIAN energy. This is the quantity the
        # certificate bounds; the Euclidean ||e|| beside it is not, and the two
        # disagree by up to sqrt(cond(W)) -- on mg that is 5.7x, enough for a
        # perfectly contracting run to show ||e|| RISING to 1.64 while E falls
        # monotonically. (V is reserved for the value function.)
        with torch.no_grad():
            M = env._metric_from_cmg(env.x_t)
        en.append(torch.bmm(torch.bmm(e.transpose(1, 2), M), e).squeeze(-1).squeeze(-1).numpy())
        env.step(torch.as_tensor(us[:, t], dtype=torch.float32))
    # Stop AT the episode end. One more sample and the env has auto-reset, so
    # x is back at x_0 and the error reads |e_0| again -- a converged rollout
    # then looks like it returned to where it started.
    # The stored reference is ref_gen_len long (199 at offset 1) while the
    # EPISODE is max_episode_len; plotting the tail would show lam on a stretch
    # of reference no task ever tracks.
    return (np.stack(xs, 1), env.max_episode_len,
            np.stack(es, 1), np.stack(en, 1), float(env.dt), float(pk["gamma"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="toy-mg-v0")
    p.add_argument("--n", type=int, default=201)
    p.add_argument("--reference-mode", "--reference_mode", dest="ref_mode",
                   choices=["stabilizing", "contractive"], default="stabilizing",
                   help="which optimum pack to overlay; each mode is its own task set")
    p.add_argument("--lambda-bar", "--lambda_bar", dest="lam_bar",
                   choices=["uniform", "local"], default="uniform",
                   help="Lambda(s) in the theorem's envelope. 'uniform' is the "
                        "N=1 reading of Assumption 1 -- the single rate SOS "
                        "actually certifies over the box, and the only one this "
                        "repo has a certificate for. 'local' substitutes the "
                        "pointwise lam(x) field, which is what a certified "
                        "PARTITION (scripts/lambda_subsets.py) would buy; it is "
                        "a strictly tighter bound than the theorem claims, so "
                        "read it as an aspiration, not a guarantee.")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    key = a.task.removeprefix("toy-").removesuffix("-v0")

    f = rate_field(key, a.n)
    lam, lbd = f["lam"], f["lbd"]
    print(f"[rate] {key}: lambda(x) in [{lam.min():.6f}, {lam.max():.6f}], "
          f"spread {lam.max() / max(lam.min(), 1e-12):.4f}x, "
          f"certified uniform {lbd:.6f}, cond(W) in "
          f"[{f['cond'].min():.2f}, {f['cond'].max():.2f}]")
    # The uniform rate is the worst point of the field. A field whose minimum
    # sits BELOW it would mean the SOS certificate does not hold somewhere on
    # the box, which is the one thing the whole construction claims.
    assert lam.min() >= lbd - 1e-6, f"field min {lam.min()} < certified {lbd}"

    stem = "rate_field" if a.ref_mode == "stabilizing" else f"rate_field_{a.ref_mode}"
    out = pathlib.Path(a.out or (REPO / "data/toy" / key / f"{stem}.npz"))
    np.savez(out, **f)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ext = [f["x1"][0], f["x1"][-1], f["x2"][0], f["x2"][-1]]
        # The benchmark on top of the field it lives in: where the tasks sit
        # decides whether the run ever visits the rate variation at all. Under
        # reference_mode="contractive" the reference is built to WALK across the
        # field, so the overlay is the whole point of the figure rather than a
        # sanity annotation.
        gstem = "global" if a.ref_mode == "stabilizing" else f"global_{a.ref_mode}"
        # Every discount solved for this reference, cheapest first. The rate
        # field is one plot; what CHANGES with gamma is the rollout, so the
        # second panel is a gamma comparison rather than a second field.
        packs = {}
        for gp in sorted((REPO / "data/toy" / key).glob(f"{gstem}*.npz")):
            if gp.stem not in (gstem,) and not gp.stem.startswith(f"{gstem}_g"):
                continue
            d = np.load(gp)
            packs[float(d["gamma"])] = d
        pk = packs.get(float(np.load(REPO / "data/toy" / key / f"{gstem}.npz")["gamma"])) \
            if (REPO / "data/toy" / key / f"{gstem}.npz").exists() else None
        rolls = {g: _optimal_rollouts(a.task, d, a.ref_mode) for g, d in packs.items()}
        traj, ep_len = (rolls[min(rolls)][0], rolls[min(rolls)][1]) if rolls else (None, None)

        ncol = 3 if pk is not None else 1
        fig, ax = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.6),
                               constrained_layout=True)
        ax = np.atleast_1d(ax)
        im = ax[0].imshow(lam.T, origin="lower", extent=ext, aspect="auto",
                          cmap="viridis")
        ax[0].set(title=rf"{key} ({a.ref_mode}): $\lambda(x)$",
                  xlabel="$x_1$", ylabel="$x_2$")
        fig.colorbar(im, ax=ax[0])

        # Panel 2 was kappa(W(x)), which is a property of the metric and says
        # nothing about the run. What the reader wants here is whether the
        # certified optimum CONVERGES, and how that depends on the discount --
        # so plot the mean normalised error, whose area is the AUC.
        if ncol == 3:
            # E = e'M(x)e, normalised. The certificate's claim is
            # E(t)/E(0) <= exp(-2 lam t), so it is drawn here -- a curve that
            # leaves the envelope is a controller the theorem says nothing about.
            dt = next(iter(rolls.values()))[4]
            lam_of = _lam_on(key)
            # Normalise EACH task by its own E(0) first, then average: the
            # envelope is a statement about the RATIO, and averaging raw E would
            # let one large-e0 task set the curve for all 64.
            ratios = {g: rolls[g][3] / np.maximum(rolls[g][3][:, :1], 1e-12)
                      for g in sorted(rolls)}
            curves = {g: r.mean(0) for g, r in ratios.items()}
            # A low gamma buys a SHORT horizon as well as a myopic objective
            # (horizon_for stops once gamma^k underflows), so AUC over each
            # curve's own length confounds the two. Score every gamma over the
            # SHORTEST horizon as well, which is the discount alone.
            hmin = min(len(m) for m in curves.values())
            for g, m in curves.items():
                r = ratios[g]
                # THE THEOREM'S envelope, not the certified feedback's:
                #     C(s_k) <= beta(s_0) C(s_0) exp(-2 lam_bar(0,k) k dt),
                #     beta(s) = (1 - gamma e^{-2 lam(s) dt})^{-1},
                # with lam_bar(0,k) the running mean of the LOCAL rate along
                # each trajectory. beta is the overshoot the optimal policy is
                # allowed to buy for long-run return: it is ~1 at gamma=0.01
                # (the overshoot-free corollary) and ~30 at gamma=0.99. Drawing
                # exp(-2 lam t) instead -- beta = 1, uniform lam -- tests u*
                # against a bound the theory never claimed for it.
                bnd = _envelope(rolls[g][0], g, dt, lbd, lam_of, a.lam_bar)
                alt = _envelope(rolls[g][0], g, dt, lbd, lam_of,
                                "local" if a.lam_bar == "uniform" else "uniform")
                frac_out = float((m > bnd + 1e-9).mean())
                frac_alt = float((m > alt + 1e-9).mean())
                # 95% CI of the MEAN over the 64 certified tasks. Clipped
                # strictly positive because the axis is logarithmic and the
                # lower arm of a wide band can cross zero.
                sem = r.std(0, ddof=1) / np.sqrt(r.shape[0])
                lo = np.maximum(m - 1.96 * sem, 1e-16)
                line, = ax[1].semilogy(
                    m, lw=1.7, marker="o" if len(m) <= 16 else None, ms=3.5,
                    label=rf"$u^*$, $\gamma$={g:g} (H={len(m)}): AUC/step "
                          rf"{m.mean():.3f}, {100 * frac_out:.0f}% outside")
                ax[1].fill_between(np.arange(len(m)), lo, m + 1.96 * sem,
                                   color=line.get_color(), alpha=0.22, lw=0)
                print(f"[rate] gamma={g:<5g} H={len(m):>3d}  "
                      f"E/E0 {m[0]:.3f} -> {m[-1]:.3e}  "
                      f"AUC/step {m.mean():.4f} (first {hmin}: {m[:hmin].mean():.4f})"
                      f"  outside the {a.lam_bar} envelope {100 * frac_out:.1f}%"
                      f"  (other reading {100 * frac_alt:.1f}%)")
            for g in sorted(curves):
                env_k = _envelope(rolls[g][0], g, dt, lbd, lam_of, a.lam_bar)
                b = 1.0 / (1.0 - g * np.exp(-2.0 * lbd * dt))
                ax[1].semilogy(env_k, "--", lw=1.2, color="0.45" if g < 0.5 else "0.0",
                               label=rf"bound $\beta(s_0)e^{{-2\bar\lambda k\Delta t}}$, "
                                     rf"$\gamma$={g:g} ($\beta$={b:.2f})")
            ax[1].set(title=f"{key} ({a.ref_mode}): contraction under $u^*$ "
                            r"(Riemannian, not Euclidean)",
                      xlabel="step",
                      ylabel=r"$E/E_0$,  $E=e^{\top}M(x)e$   (mean, 95% CI)")
            ax[1].legend(fontsize=7)
        if pk is not None:
            xr = pk["xref"][0][:ep_len]
            if traj is not None:
                for tr in traj:
                    ax[0].plot(tr[:, 0], tr[:, 1], "-", c="deepskyblue", lw=0.5,
                               alpha=0.7, zorder=2)
            ax[0].plot(xr[:, 0], xr[:, 1], "w-", lw=1.6, zorder=4)
            ax[0].scatter(pk["x0"][:, 0], pk["x0"][:, 1], s=6, c="r", zorder=5)
            # lam ALONG the reference: the claim "starts slow, ends fast" is a
            # statement about this curve, not about the field's own extremes.
            lr = _lam_along(f, xr)
            ax[2].plot(lr, "k-", lw=1.4)
            ax[2].axhline(lbd, color="r", ls="--", lw=1,
                          label=f"certified {lbd:.3f}")
            ax[2].set(title=rf"$\lambda(x_{{\rm ref}}(t))$: "
                            rf"{lr[0]:.2f}$\to${lr[-1]:.2f} "
                            rf"({lr.max() / lr.min():.2f}x)",
                      xlabel="step", ylabel=r"$\lambda$")
            ax[2].legend(fontsize=7)
            print(f"[rate] lam along the {a.ref_mode} reference: "
                  f"{lr[0]:.4f} -> {lr[-1]:.4f} "
                  f"(min {lr.min():.4f}, max {lr.max():.4f}, "
                  f"{lr.max() / lr.min():.3f}x)")
        fig.savefig(out.with_suffix(".png"), dpi=150)
        print(f"[rate] wrote {out} and {out.with_suffix('.png')}")
    except ImportError:
        print(f"[rate] wrote {out} (no matplotlib, skipped the figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The N certified optima, drawn as N runs -- controls, trajectories, energy.

``scripts/toy_rate_field.py`` already reports the MEAN Riemannian energy under
u*, which hides the thing a benchmark of N pinned initial states is for: the
spread. Every panel here draws all N attempts, so a mean that looks contracting
because 63 tasks converge and one diverges cannot pass as a converging mean.

Rows, one column per discount:
  1. ``u*(t)``      -- the certified optimal control of each attempt.
  2. ``x(t)``       -- where that control takes the plant, against the reference.
  3. ``E/E_0``      -- ``E = e' M(x) e``, the RIEMANNIAN energy, which is what
                       the certificate bounds; the mean and its 95% CI over the
                       N attempts, over the faint per-attempt curves, against
                       the theorem's envelope beta(s_0) exp(-2 lam_bar k dt).

Read across a row and the discount is the only thing that changed: same plant,
same contractive reference, same N pinned starts, same horizon. Pinning the
horizon is not cosmetic -- a low gamma's default horizon stops once gamma^k
underflows, and comparing an H=8 curve to an H=100 one reads the truncation as
if it were the discount. Pass --horizon (or solve every pack with
``precompute_global.py --horizon 100``) so every column is the same window.

    python scripts/toy_optimal_runs.py --task toy-mg-v0 --reference-mode contractive
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "contractionRL"))
sys.path.insert(0, str(REPO / "scripts"))

from toy_rate_field import _envelope, _lam_on, _optimal_rollouts  # noqa: E402


def packs(key: str, ref_mode: str) -> dict:
    """{gamma: pack}. A gamma solved twice (suffixed and not) keeps the LONGER
    horizon -- the unsuffixed pack is the default-horizon solve, which for a low
    discount is the truncated one this figure exists to avoid."""
    stem = "global" if ref_mode == "stabilizing" else f"global_{ref_mode}"
    out = {}
    for p in sorted((REPO / "data/toy" / key).glob(f"{stem}*.npz")):
        if p.stem != stem and not p.stem.startswith(f"{stem}_g"):
            continue
        d = np.load(p)
        g = float(d["gamma"])
        if g not in out or int(d["horizon"]) > int(out[g]["horizon"]):
            out[g] = d
    return dict(sorted(out.items()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="toy-mg-v0")
    p.add_argument("--reference-mode", "--reference_mode", dest="ref_mode",
                   choices=["stabilizing", "contractive"], default="contractive")
    p.add_argument("--gammas", nargs="+", type=float, default=None,
                   help="default: every discount solved for this reference")
    p.add_argument("--horizon", type=int, default=None,
                   help="truncate every column to the same H (default: each in full)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    key = a.task.removeprefix("toy-").removesuffix("-v0")

    got = packs(key, a.ref_mode)
    if a.gammas:
        got = {g: d for g, d in got.items() if any(abs(g - w) < 1e-12 for w in a.gammas)}
    if not got:
        print(f"[runs] no {a.ref_mode} optimum pack for {key}. Solve one:\n"
              f"    python scripts/precompute_global.py --task {a.task} "
              f"--reference-mode {a.ref_mode} --gamma 0.99 --horizon 100")
        return 1

    d0 = np.load(sorted((REPO / "data/toy" / key).glob("cm_data_*.npz"))[-1])
    lbd = float(d0["lbd"])
    lam_of = _lam_on(key)

    runs = {}
    for g, pk in got.items():
        X, ep_len, e_eu, E, dt, _ = _optimal_rollouts(a.task, pk, a.ref_mode)
        H = min(a.horizon or X.shape[1], X.shape[1])
        us = np.asarray(pk["u_star"]).reshape(len(pk["u_star"]), -1)[:, :H]
        runs[g] = dict(X=X[:, :H], E=E[:, :H], e=e_eu[:, :H], u=us, dt=dt,
                       xref=np.asarray(pk["xref"])[0][:ep_len], H=H)

    hs = {r["H"] for r in runs.values()}
    if len(hs) > 1:
        print(f"[runs] WARNING horizons differ across discounts {sorted(hs)} -- the "
              f"columns are not comparable. Re-solve the short packs with "
              f"--horizon {max(hs)}, or pass --horizon {min(hs)} to crop.")

    print(f"\n[runs] {key} / {a.ref_mode} reference, N={len(next(iter(runs.values()))['X'])} "
          f"certified optima per discount")
    print("  gamma      H    E/E0 end (mean)   [95% CI]                 AUC/step   "
          "worst attempt   outside envelope")
    for g, r in runs.items():
        ratio = r["E"] / np.maximum(r["E"][:, :1], 1e-300)
        m = ratio.mean(0)
        sem = ratio.std(0, ddof=1) / np.sqrt(len(ratio))
        bnd = _envelope(r["X"], g, r["dt"], lbd, lam_of, "uniform")
        r.update(ratio=ratio, m=m, sem=sem, bnd=bnd)
        print(f"  {g:<8g} {r['H']:>4d}  {m[-1]:>14.3e}   "
              f"[{max(m[-1] - 1.96 * sem[-1], 0):.3e}, {m[-1] + 1.96 * sem[-1]:.3e}]  "
              f"{m.mean():>9.4f}   {ratio[:, -1].max():>13.3e}   "
              f"{100 * (m > bnd + 1e-9).mean():>5.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(runs)
    fig, ax = plt.subplots(3, nc, figsize=(4.3 * nc, 9.4), squeeze=False,
                           constrained_layout=True)
    for c, (g, r) in enumerate(runs.items()):
        t = np.arange(r["H"])
        for u in r["u"]:
            ax[0][c].plot(t, u, "-", lw=0.5, alpha=0.45, c="tab:blue")
        ax[0][c].set(title=rf"{key} ({a.ref_mode}): $u^*$, $\gamma$={g:g}, "
                           rf"N={len(r['u'])}, H={r['H']}",
                     xlabel="step", ylabel="$u^*$")

        ax[1][c].plot(r["xref"][:, 0], r["xref"][:, 1], "k-", lw=2, zorder=4,
                      label="reference")
        for X in r["X"]:
            ax[1][c].plot(X[:, 0], X[:, 1], "-", lw=0.5, alpha=0.5, c="tab:blue")
        ax[1][c].scatter(r["X"][:, 0, 0], r["X"][:, 0, 1], s=8, c="r", zorder=5,
                         label="$x_0$")
        ax[1][c].set(title=rf"N attempts under $u^*$, $\gamma$={g:g}",
                     xlabel="$x_1$", ylabel="$x_2$")
        ax[1][c].legend(fontsize=7)

        for row in r["ratio"]:
            ax[2][c].semilogy(t, np.maximum(row, 1e-16), "-", lw=0.4, alpha=0.3,
                              c="tab:blue")
        # Clipped strictly positive: the axis is logarithmic and the lower arm
        # of a wide band crosses zero.
        lo = np.maximum(r["m"] - 1.96 * r["sem"], 1e-16)
        ax[2][c].semilogy(t, r["m"], "-", lw=2.2, c="tab:red", zorder=5,
                          label=rf"mean, AUC/step {r['m'].mean():.3f}")
        ax[2][c].fill_between(t, lo, r["m"] + 1.96 * r["sem"], color="tab:red",
                              alpha=0.3, lw=0, zorder=4, label="95% CI of the mean")
        ax[2][c].semilogy(t, r["bnd"], "--", lw=1.3, c="0.25",
                          label=r"$\beta(s_0)e^{-2\bar\lambda k\Delta t}$")
        ax[2][c].set(title=rf"Riemannian energy, $\gamma$={g:g}", xlabel="step",
                     ylabel=r"$E/E_0$,  $E=e^{\top}M(x)e$")
        ax[2][c].legend(fontsize=7)

    stem = f"optimal_runs_{a.ref_mode}"
    out = pathlib.Path(a.out or (REPO / "data/toy" / key / f"{stem}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    np.savez(out.with_suffix(".npz"),
             gammas=np.array(list(runs)), lbd=lbd,
             **{f"{n}_g{g:g}": r[n] for g, r in runs.items()
                for n in ("X", "E", "e", "u", "ratio", "m", "sem", "bnd")})
    print(f"[runs] wrote {out} and {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

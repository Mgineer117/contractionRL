"""Global V* by exhaustive value iteration, with a Richardson error estimate.

Why not an NLP or an MPC lookahead: both return a feasible TRAJECTORY, so both
give ``J* <= J_found``. That direction can prove a policy is bad and never that
it is good. The Bellman optimality operator is a gamma-contraction with a unique
fixed point, so value iteration has no local minima to get stuck in -- it returns
V* of the discretised MDP by construction.

What is solved here is the C2RL objective, not a convenient stand-in: the stage
cost is ``env_base.get_rewards`` negated, with the reference pinned at a trim
(xref, uref) so the problem is time-invariant and the state is x alone. Pinning
is the ONLY approximation of the RL problem; discretisation error is measured
separately by ``richardson``.

Cost scales as n^x_dim * actions^u_dim, so the limit is memory, not dimension --
see ``check_budget``, which refuses up front rather than running until someone
gives up. Two states (the toy envs) is comfortable; four states at a coarse grid
is affordable; the 10-state quadrotor is not.

This module was reconstructed on 2026-09-01 from the compiled ``.pyc`` of the
deleted ``value_iteration.py`` -- the docstrings and constants are the originals,
and the invariants below are re-checked by ``tests/test_vi.py``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VICfg:
    """Value-iteration knobs. Mirrors ``skrl_vi_cfg.yaml``'s ``agent`` block."""

    n: int = 65
    actions: int = 41
    gamma: float = 0.01
    tol: float = 1e-12
    max_iter: int = 20000
    device: str = "cpu"
    max_gb: float = 8.0


def check_budget(x_dim: int, u_dim: int, n: int, actions: int,
                 max_gb: float = 8.0) -> float:
    """Raise unless the (state x action) tables fit in ``max_gb``.

    The old guard was a hard ``x_dim > 2`` wall, which is the wrong invariant:
    a 4-state plant on a 25-point grid is cheaper than a 2-state plant on a
    1025-point one. Budget is the thing that actually binds.
    """
    cells, A = float(n) ** x_dim, float(actions) ** u_dim
    pairs = cells * A
    # one float64 successor coordinate per state dim (16 B: the value and its
    # interpolation index/fraction) plus the cost table itself.
    gb = pairs * (x_dim * 16 + 8) / 1073741824
    if gb > max_gb:
        raise ValueError(
            f"value iteration needs ~{gb:.1f} GB for a {x_dim}-state / {u_dim}"
            f"-input plant at n={n}, actions={actions} ({pairs:.3g} state-action "
            f"pairs), over the {max_gb} GB budget. Lower --n or --actions, raise "
            f"--max-gb, or pick a lower-dimensional env.")
    return gb


def _action_box(env):
    """The box the ENV actually applies, which is not the reference-control box.

    ``env_base`` sets ``U_MIN/U_MAX = 2 * UREF_MIN/UREF_MAX`` and ``step()``
    clamps to those, so a search over UREF gives V* half the authority the policy
    has -- and the policy then beats "the global optimum" on most tasks. Measured
    on toy-mg before this was fixed: mean gap -0.084, with 85.8% of tasks below
    V*. That is the whole reason ``optimality_gap`` reports below_v_star_frac.
    """
    lo = getattr(env, "U_MIN", None)
    hi = getattr(env, "U_MAX", None)
    if lo is None:
        lo = env.UREF_MIN
    if hi is None:
        hi = env.UREF_MAX
    return lo, hi


def _mesh(axes) -> torch.Tensor:
    """Cartesian product of 1-D axes as a (prod(len), len(axes)) tensor."""
    g = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(g, -1).reshape(-1, len(axes))


def _maha(env, X: torch.Tensor, e: torch.Tensor,
          chunk: int = 1048576) -> torch.Tensor:
    """e^T M(X) e, chunked -- X can be tens of millions of rows."""
    out = torch.empty(X.shape[0], dtype=torch.float64, device=X.device)
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            M = env._metric_from_cmg(X[i:i + chunk].float()).double()
            ei = e[i:i + chunk]
            out[i:i + chunk] = torch.einsum("bi,bij,bj->b", ei, M, ei)
    return out


def _corner_interp(V, i0, frac, shape, stride, nd, device):
    """Multilinear V at each successor state, summing the 2^nd corners.

    Corners are formed here rather than stored: at 4 states that is 16 tables,
    which is what decides whether the grid fits in memory at all.
    """
    Vf = V.reshape(-1)
    out = torch.zeros(shape, dtype=torch.float64, device=device)
    for c in itertools.product((0, 1), repeat=nd):
        idx = sum((i0[..., d] + c[d]) * s for d, s in enumerate(stride))
        w = torch.ones(shape, dtype=torch.float64, device=device)
        for d in range(nd):
            fr = frac[..., d]
            w = w * (fr if c[d] else (1.0 - fr))
        out = out + w * Vf[idx.reshape(-1)].view(shape)
    return out


class GridVI:
    """Value iteration on a uniform state grid with multilinear interpolation.

    Interpolation weights depend on the grid and the dynamics but not on V, so
    the base index and fraction are built once; the 2^x_dim corners are formed
    inside the sweep rather than stored, which is what keeps a 4-state grid in
    memory at all.
    """

    def __init__(self, env, n: int, n_act: int, gamma: float, device: str,
                 xref=None, uref=None, max_gb: float = 8.0):
        nd, nu = int(env.num_dim_x), int(env.num_dim_control)
        check_budget(nd, nu, n, n_act, max_gb)
        self.nd, self.n, self.gamma, self.device = nd, n, gamma, device

        lo = env.X_MIN.double().cpu().numpy()
        hi = env.X_MAX.double().cpu().numpy()
        self.lo, self.hi = lo, hi
        f64 = dict(dtype=torch.float64, device=device)

        self.X = _mesh([torch.linspace(float(lo[i]), float(hi[i]), n, **f64)
                        for i in range(nd)])
        u_lo, u_hi = _action_box(env)
        self.U = _mesh([torch.linspace(float(u_lo[i]), float(u_hi[i]), n_act, **f64)
                        for i in range(nu)])

        if xref is None:
            xref = torch.zeros(nd, **f64)
        if uref is None:
            uref = torch.zeros(nu, **f64)
        self.xref = torch.as_tensor(xref, **f64).reshape(-1)
        self.uref = torch.as_tensor(uref, **f64).reshape(-1)

        dt = float(env.dt)
        with torch.no_grad():
            f, B, _ = env.get_f_and_B(self.X.float(), need_null=False)
        f, B = f.double(), B.double()
        # (cells, actions, nd): every grid state under every grid action.
        XN = (self.X[:, None, :] + dt * (f[:, None, :]
                                         + torch.einsum("cij,aj->cai", B, self.U)))
        XN = torch.stack([XN[..., i].clamp(float(lo[i]), float(hi[i]))
                          for i in range(nd)], -1)
        self.shape = XN.shape[:2]

        self.cost = self._stage_cost(env, self.X, XN, self.U)

        h = torch.as_tensor((hi - lo) / (n - 1), **f64)
        t = (XN - torch.as_tensor(lo, **f64)) / h
        self.i0 = torch.floor(t).long().clamp(0, n - 2)
        self.frac = t - self.i0.double()
        stride, k = [], 1
        for _ in range(nd):
            stride.append(k)
            k *= n
        self.stride = list(reversed(stride))

    def _stage_cost(self, env, X, XN, U):
        """``-env_base.get_rewards`` with the reference pinned at (xref, uref).

        Mirrors that method branch for branch. A cost that merely resembles the
        reward makes ``J^pi - J*`` a comparison between two different problems,
        which is the failure this whole measurement exists to avoid.
        """
        q = float(getattr(env, "tracking_scaler", 1.0))
        r = float(getattr(env, "control_scaler", 0.0))
        eff = (U ** 2).sum(-1)                      # (actions,)

        e = env.wrap_angles(X.float() - self.xref.float()).double()
        en = env.wrap_angles(XN.float() - self.xref.float()).double()
        flat = en.reshape(-1, self.nd)

        if getattr(env, "reward_euclidean", False) or getattr(env, "ccm_gen", None) is None:
            V = (e ** 2).sum(-1)
            VN = (flat ** 2).sum(-1).view(self.shape)
        else:
            V = _maha(env, X, e)
            VN = _maha(env, XN.reshape(-1, self.nd), flat).view(self.shape)

        if getattr(env, "reward_level", False):
            # level reward: -q * E(x'), no decrement
            return q * VN + r * eff[None, :]
        # decrement reward: -q * (E(x) - E(x')), negated for a COST
        return -q * (V[:, None] - VN) + r * eff[None, :]

    def _interp(self, V, i0, frac, shape):
        """Multilinear V at each successor state. Corners are formed here rather
        than stored: at 4 states that is 16 tables, which is what decides whether
        the grid fits in memory at all."""
        return _corner_interp(V, i0, frac, shape, self.stride, self.nd, self.device)

    def solve(self, tol: float = 1e-12, max_iter: int = 20000):
        V = torch.zeros(self.n ** self.nd, dtype=torch.float64, device=self.device)
        resid = float("inf")
        for k in range(max_iter):
            Vnew = (self.cost + self.gamma
                    * self._interp(V, self.i0, self.frac, self.shape)).min(dim=1).values
            resid = (Vnew - V).abs().max().item()
            V = Vnew
            if resid <= tol:
                return V, k + 1, resid
        return V, max_iter, resid

    def greedy_action(self, V: torch.Tensor) -> torch.Tensor:
        """argmin_u of the Bellman backup at every grid point -> (cells, u_dim)."""
        Vn = self.cost + self.gamma * self._interp(V, self.i0, self.frac, self.shape)
        return self.U[Vn.argmin(dim=1)]

    def snap(self, u: torch.Tensor):
        """Nearest action-grid index for each row of ``u``, and the snap error.

        Policy evaluation reuses the precomputed successor tables, which exist
        only at grid actions. The error is REPORTED rather than assumed small:
        a coarse action grid can make a good policy look suboptimal, and that
        would be indistinguishable from the gap being measured.
        """
        d = (u[:, None, :] - self.U[None, :, :]).pow(2).sum(-1)
        idx = d.argmin(dim=1)
        return idx, float((u - self.U[idx]).abs().max())

    def evaluate(self, a_idx: torch.Tensor, tol: float = 1e-12,
                 max_iter: int = 20000):
        """J^pi for the deterministic grid policy ``a_idx`` -- T^pi, not T*.

        Same operator as ``solve`` with the min removed, so J^pi and V* differ in
        exactly one place and their difference is the policy's suboptimality
        rather than an artefact of two different discretisations.
        """
        rows = torch.arange(a_idx.shape[0], device=self.device)
        cost = self.cost[rows, a_idx]
        i0 = self.i0[rows, a_idx]
        frac = self.frac[rows, a_idx]
        shape = cost.shape
        J = torch.zeros_like(cost)
        resid = float("inf")
        for k in range(max_iter):
            Jnew = cost + self.gamma * self._interp(J, i0, frac, shape)
            resid = (Jnew - J).abs().max().item()
            J = Jnew
            if resid <= tol:
                return J, k + 1, resid
        return J, max_iter, resid


def solve_vi(env, cfg: VICfg, xref=None, uref=None):
    """Solve on three NESTED grids ``n, 2(n-1)+1, 4(n-1)+1``.

    Nested so the coarse points are exactly every 2nd/4th point of the finer
    grids: the three solutions are then compared without the comparison itself
    introducing interpolation error.
    """
    out = []
    for n in (cfg.n, 2 * (cfg.n - 1) + 1, 4 * (cfg.n - 1) + 1):
        g = GridVI(env, n, cfg.actions, cfg.gamma, cfg.device, xref, uref,
                   cfg.max_gb)
        V, iters, resid = g.solve(cfg.tol, cfg.max_iter)
        out.append({"n": n, "V": V, "iters": iters, "resid": resid, "grid": g})
    return out


def richardson(sols, interior_frac: float = 0.1):
    """Observed order of accuracy and the extrapolated V*.

        p     = log2( ||V_h - V_h/2|| / ||V_h/2 - V_h/4|| )
        V_ext = V_h/4 + (V_h/4 - V_h/2) / (2^p - 1)

    Reported in three norms deliberately. The sup norm is what an error bound
    needs, but it is set by the single worst point -- typically on the bang-bang
    switching surface, where V* is Lipschitz but not C^1 and no amount of
    refinement helps. L2 and an interior-only sup norm say whether the scheme is
    converging faster than that one locus suggests. Reporting sup alone cannot
    distinguish "the method is poor" from "V* has a kink".
    """
    nd = sols[0]["grid"].nd
    n0 = sols[0]["n"]
    # subsample the finer grids onto the coarse points: nested, so this is exact
    Vc = sols[0]["V"].reshape(*([n0] * nd))
    Vm = sols[1]["V"].reshape(*([sols[1]["n"]] * nd))[(slice(None, None, 2),) * nd]
    Vf = sols[2]["V"].reshape(*([sols[2]["n"]] * nd))[(slice(None, None, 4),) * nd]

    m = max(int(n0 * interior_frac), 1)
    interior = (slice(m, n0 - m),) * nd
    norms = {
        "sup": lambda D: D.abs().max().item(),
        "L2": lambda D: float(D.pow(2).mean().sqrt()),
        "sup-interior": lambda D: D[interior].abs().max().item(),
    }

    orders, diffs = {}, {}
    for tag, fn in norms.items():
        e1, e2 = fn(Vc - Vm), fn(Vm - Vf)
        diffs[tag] = (e1, e2)
        orders[tag] = (float(np.log2(e1 / e2)) if e2 > 0 and np.isfinite(e1 / e2)
                       else float("nan"))

    p = orders["sup"]
    V_ext = (Vf + (Vf - Vm) / (2 ** p - 1)) if np.isfinite(p) and p > 0 else Vf
    err_est = (V_ext - Vf).abs().max().item()
    return {"orders": orders, "diffs": diffs, "V_ext": V_ext, "err_est": err_est,
            "V_fine": sols[2]["V"], "converging": bool(np.isfinite(p) and p > 0)}


def grid_value_at(V, x, lo, hi, n, nd, device="cpu"):
    """Multilinear interpolation of a grid function at arbitrary states.

    Standalone so a consumer with only the saved arrays (data/toy/<k>/vstar.npz)
    can read V*(x_0) without rebuilding the solver's dynamics tables -- which is
    every eval, and would otherwise cost more than the rollout it annotates.
    """
    f64 = dict(dtype=torch.float64, device=device)
    V = torch.as_tensor(V, **f64).reshape(-1)
    x = torch.as_tensor(x, **f64).reshape(-1, nd)
    lo = torch.as_tensor(np.asarray(lo, dtype=np.float64), **f64)
    hi = torch.as_tensor(np.asarray(hi, dtype=np.float64), **f64)

    h = (hi - lo) / (n - 1)
    t = ((x - lo) / h).clamp(0.0, float(n - 1))
    i0 = torch.floor(t).long().clamp(0, n - 2)
    frac = t - i0.double()

    stride, k = [], 1
    for _ in range(nd):
        stride.append(k)
        k *= n
    stride = list(reversed(stride))

    out = torch.zeros(x.shape[0], **f64)
    for c in itertools.product((0, 1), repeat=nd):
        idx = sum((i0[:, d] + c[d]) * s for d, s in enumerate(stride))
        w = torch.ones_like(out)
        for d in range(nd):
            fr = frac[:, d]
            w = w * (fr if c[d] else (1.0 - fr))
        out = out + w * V[idx]
    return out


class TrackingVI:
    """V* for ONE fixed reference trajectory, by a backward sweep.

    With the reference frozen the tracking problem is a finite-horizon,
    time-varying MDP whose state is x alone -- time enters only through which
    reference point is current. So

        V*_T(x) = 0
        V*_t(x) = min_u [ c_t(x,u) + gamma * V*_{t+1}(x') ],
        x'      = clamp(x + dt (f(x) + B(x) u)),
        c_t     = -q (e^T M(x) e - e'^T M(x') e') + r ||u||^2

    is ONE pass backward from t = T. Not a fixed-point iteration: exact for the
    discretised MDP, no tolerance, no contraction argument, and cheaper than the
    infinite-horizon regulation problem it replaces. Pinning the reference at a
    trim and iterating T*V = V (the earlier approach) answers a different
    question than the agent is trained on.

    The successor tables do not depend on t -- only the cost does -- so the
    dynamics are built once and each of the T backups is a gather plus a min.
    """

    def __init__(self, env, n: int, n_act: int, gamma: float,
                 device: str = "cpu", max_gb: float = 8.0):
        nd, nu = int(env.num_dim_x), int(env.num_dim_control)
        check_budget(nd, nu, n, n_act, max_gb)
        self.env, self.nd, self.nu = env, nd, nu
        self.n, self.gamma, self.device = n, gamma, device

        lo = env.X_MIN.double().cpu().numpy()
        hi = env.X_MAX.double().cpu().numpy()
        self.lo, self.hi = lo, hi
        f64 = dict(dtype=torch.float64, device=device)

        self.X = _mesh([torch.linspace(float(lo[i]), float(hi[i]), n, **f64)
                        for i in range(nd)])
        u_lo, u_hi = _action_box(env)
        self.U = _mesh([torch.linspace(float(u_lo[i]), float(u_hi[i]), n_act, **f64)
                        for i in range(nu)])

        dt = float(env.dt)
        with torch.no_grad():
            f, B, _ = env.get_f_and_B(self.X.float(), need_null=False)
        f, B = f.double(), B.double()
        XN = (self.X[:, None, :] + dt * (f[:, None, :]
                                         + torch.einsum("cij,aj->cai", B, self.U)))
        self.XN = torch.stack([XN[..., i].clamp(float(lo[i]), float(hi[i]))
                               for i in range(nd)], -1)
        self.shape = self.XN.shape[:2]

        h = torch.as_tensor((hi - lo) / (n - 1), **f64)
        t = (self.XN - torch.as_tensor(lo, **f64)) / h
        self.i0 = torch.floor(t).long().clamp(0, n - 2)
        self.frac = t - self.i0.double()
        stride, k = [], 1
        for _ in range(nd):
            stride.append(k)
            k *= n
        self.stride = list(reversed(stride))

        self.q = float(getattr(env, "tracking_scaler", 1.0))
        self.r = float(getattr(env, "control_scaler", 0.0))
        self.eff = (self.U ** 2).sum(-1)
        self.maha = getattr(env, "ccm_gen", None) is not None
        if self.maha:
            self._MX = self._metric(self.X)
            self._MN = self._metric(self.XN.reshape(-1, nd)).view(
                *self.shape, nd, nd)

    def _metric(self, X, chunk: int = 1048576):
        out = torch.empty(X.shape[0], self.nd, self.nd, dtype=torch.float64,
                          device=self.device)
        with torch.no_grad():
            for i in range(0, X.shape[0], chunk):
                out[i:i + chunk] = self.env._metric_from_cmg(
                    X[i:i + chunk].float()).double()
        return out

    def _cost(self, xr_t, xr_next):
        """Stage cost at one timestep: env_base.get_rewards, negated."""
        e = self.env.wrap_angles(self.X.float() - xr_t.float()).double()
        en = self.env.wrap_angles(self.XN.float() - xr_next.float()).double()
        if getattr(self.env, "reward_euclidean", False) or not self.maha:
            V = e.norm(dim=-1) ** 2
            VN = en.norm(dim=-1) ** 2
        else:
            V = torch.einsum("ci,cij,cj->c", e, self._MX, e)
            VN = torch.einsum("cai,caij,caj->ca", en, self._MN, en)
        if getattr(self.env, "reward_level", False):
            return self.q * VN + self.r * self.eff[None, :]
        return -self.q * (V[:, None] - VN) + self.r * self.eff[None, :]

    def solve(self, xref, uref=None, horizon: int | None = None,
              keep_all: bool = False):
        """Backward sweep over a reference ``xref`` of shape (>=T, x_dim).

        ``horizon`` is the EPISODE length. It is not ``len(xref)``: the stored
        reference runs past the episode end so the observation window never
        clamps, and sweeping those extra steps would solve a longer problem than
        the agent ever plays.

        ``uref`` is unused: the reward penalises ``||u||^2`` (control_scaler is 0
        in every shipped config anyway), and the reference control enters the
        agent through ``u = urefs[0] + pi``, not through the cost. Accepted so
        callers can pass the pair they already have.
        """
        xref = torch.as_tensor(xref).double().to(self.device)
        T = int(xref.shape[0]) if horizon is None else int(horizon)
        if xref.shape[0] < T:
            raise ValueError(f"horizon {T} exceeds the reference's "
                             f"{xref.shape[0]} steps.")
        V = torch.zeros(self.n ** self.nd, dtype=torch.float64, device=self.device)
        stack = [V.clone()] if keep_all else None
        for t in range(T - 1, -1, -1):
            c = self._cost(xref[t], xref[min(t + 1, xref.shape[0] - 1)])
            Vn = _corner_interp(V, self.i0, self.frac, self.shape, self.stride,
                                self.nd, self.device)
            V = (c + self.gamma * Vn).min(dim=1).values
            if keep_all:
                stack.append(V.clone())
        return torch.stack(list(reversed(stack))) if keep_all else V

    def value_at(self, V, x):
        """Multilinear V at arbitrary states -- how V*(x_0) is read for an env slot."""
        return grid_value_at(V, x, self.lo, self.hi, self.n, self.nd, self.device)


def solve_tracking(env, xrefs, cfg: VICfg, log=print):
    """V*_0 on the grid for each of ``xrefs`` (G, T, x_dim). One sweep each."""
    vi = TrackingVI(env, cfg.n, cfg.actions, cfg.gamma, cfg.device, cfg.max_gb)
    out = []
    for g in range(int(xrefs.shape[0])):
        out.append(vi.solve(xrefs[g]))
        if log:
            log(f"[vstar] reference {g + 1}/{int(xrefs.shape[0])}: |V*_0| mean "
                f"{float(out[-1].abs().mean()):.6e}")
    return torch.stack(out), vi

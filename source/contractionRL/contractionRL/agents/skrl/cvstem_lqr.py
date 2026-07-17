"""CV-STEM-LQR — Tsukamoto's CV-STEM contraction controller as a skrl Agent.

Analytical (no learnable RL parameters — ``update()`` is a no-op), like
``sdlqr.py``'s SD-LQR/LQR, but the feedback gain comes from a CV-STEM
*contraction metric* instead of a per-state CARE (algebraic Riccati) solve. The
control law is Tsukamoto's (Neural Contraction Metric work, his ``ncm.control``)::

    u = u_ref - K(x)·(x - x_ref),   K(x) = R⁻¹·B(x)ᵀ·M(x)

evaluated at the CURRENT state ``x`` (the "SD" linearization point, matching his
original), with the SAME ``R`` used in the metric's Riccati term (``r_scaler``)
so ``K`` is the CERTIFIED CV-STEM gain — not an arbitrary LQR gain that happens
to share the metric.

Why this differs from SD-LQR
----------------------------
SD-LQR's ``K = R⁻¹BᵀP`` uses ``P`` from ``solve_continuous_are`` — a LOCAL
optimal-control solution at the linearized ``(A, B)`` with only a
stabilizability guarantee. Here ``M(x) = W(x)⁻¹`` is a CV-STEM contraction
metric (``ncm_synthesis.solve_cm_metric``): the SDP that produced it enforced

    A(x)·W + W·Aᵀ - 2·B R⁻¹ Bᵀ + 2λ·W ⪯ -ε·I,   w_lb·I ⪯ W ⪯ w_ub·I

whose ``-2·B R⁻¹ Bᵀ`` term is EXACTLY the closed loop ``A - B R⁻¹BᵀM`` carried
through the ``W = M⁻¹`` congruence. So ``M`` doubles as a state-dependent
Riccati solution and this ``K`` renders the system contracting at rate ``λ`` —
a global incremental-stability certificate, not a local one. See
``ncm_synthesis.py``'s module docstring.

Metric source (``metric_source`` config)
-----------------------------------------
* ``"online"`` (default): solve the CV-STEM SDP fresh at every step, per env,
  for the current ``x`` (``_solve_cm_metric_with_backoff``). Truest to the
  certificate — every deployed ``M`` is a verified feasible metric — but needs
  cvxpy/an SDP solver at deploy time and runs one solve per env per step (a
  Python loop on CPU, like SD-LQR's CARE loop). Infeasible states fall back to
  zero feedback (``u = u_ref``), same as SD-LQR's non-stabilizable fallback.
* ``"pretrained"``: synthesize a frozen CMG network ONCE at construction
  (Tsukamoto NCM — ``build_cm_dataset`` per-state SDP + ``regress_cmg`` MSE fit,
  cached to ``cm_data_path``), then ``M(x)`` is a cheap batched network forward
  pass and the whole step is vectorized on-device. The certificate is then only
  as tight as the regression fit, but rollouts are orders of magnitude faster
  and need no solver at deploy time. This reuses C2RL's exact Phase-A synthesis.

Normalization: none, and none is meaningful — there are no learned RL weights
whose input distribution could drift. ``"online"`` pins its compute to CPU
(cvxpy is numpy/CPU-only); ``"pretrained"`` runs on the env's device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from skrl.agents.torch.base import Agent, AgentCfg

from .angle_utils import wrap_diff
from .math_utils import bound_W, jacobian, spd_inverse
from .ncm_synthesis import _solve_cm_metric_with_backoff
from .rl_glue import filter_cfg_fields


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CVSTEMLQRCfg(AgentCfg):
    # "online" (per-step SDP solve) | "pretrained" (frozen CMG network). See
    # module docstring.
    metric_source: str = "online"

    # R = r_scaler·I. Used in BOTH the metric SDP's Riccati term AND the deployed
    # gain K = R⁻¹BᵀM — they MUST share the same R for K to be the certified
    # CV-STEM gain. A single knob enforces that (mirrors sdlqr.py's R_scaler and
    # ncm_synthesis' r_scaler, deliberately unified here). Strictly positive:
    # r_scaler → 0 gives huge gains → discrete-dt divergence.
    r_scaler: float = 1.0

    # ── CV-STEM SDP knobs (both metric sources — online solve AND the offline
    #    dataset synthesis for "pretrained") — see ncm_synthesis.solve_cm_metric ──
    lbd: float = 0.5              # contraction rate λ
    w_lb: float = 0.1            # W eigenvalue lower bound (M ≤ 1/w_lb)
    w_ub: float = 10.0           # W eigenvalue upper bound (M ≥ 1/w_ub)
    cm_eps: float = 0.01         # strict-definiteness margin on the LMI
    cm_solver: str = "SCS"       # cvxpy SDP solver (SCS | CLARABEL | MOSEK)
    max_lambda_reductions: int = 5  # per-state λ-backoff budget on infeasibility

    # ── "pretrained" only: offline CMG synthesis (mirrors the yaml `cmg:` block
    #    and C2RL's _synthesize_cmg_cvstem) ──────────────────────────────────── #
    min_feasibility_rate: float = 0.5
    cm_data_path: str = ""       # {x, W} dataset cache (skips the per-state solve)
    cmg_memory_size: int = 131072
    cmg_regress_epochs: int = 10
    cmg_regress_lr: float = 1.0e-3
    cmg_regress_batch_size: int = 1024
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict | None = None
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class CVSTEMLQRAgent(Agent):
    """CV-STEM contraction controller wrapped as a native skrl Agent.

    Extra constructor kwargs:
      ``get_f_and_B``: ``(x) -> (f, B, Bbot)`` (analytical env dynamics or a
        loaded NeuralDynamics).
      ``get_rollout``: only needed for ``metric_source="pretrained"`` state
        sampling (same source C2RL/C3M use); may be ``None`` for ``"online"``.
      ``models``: ``{"cmg": MetricModel}`` for ``"pretrained"`` (a
        BoundedCCM_Generator, ``constrain_eigenvalues=True``); ``{}`` for
        ``"online"``.
    """

    def __init__(
        self,
        *,
        cfg: CVSTEMLQRCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_f_and_B: Callable,
        get_rollout: Callable | None = None,
        x_dim: int | None = None,
        u_dim: int | None = None,
        angle_idx: list | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = CVSTEMLQRCfg(**filter_cfg_fields(cfg, CVSTEMLQRCfg, context="CVSTEMLQRAgent"))
        super().__init__(
            cfg=cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        obs_dim = int(observation_space.shape[0])
        u_dim_inferred = int(action_space.shape[0])
        if u_dim is None:
            u_dim = u_dim_inferred
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._angle_idx = angle_idx or []
        self._cfg = cfg
        self._get_f_and_B = get_f_and_B
        self._get_rollout = get_rollout

        self._metric_source = str(cfg.metric_source).lower()
        if self._metric_source not in ("online", "pretrained"):
            raise ValueError(
                f"CVSTEMLQRAgent: metric_source must be 'online' or 'pretrained', "
                f"got {cfg.metric_source!r}."
            )

        if self._metric_source == "online":
            # cvxpy is numpy/CPU-only — the per-env SDP solve loop runs on CPU
            # regardless of the env's device (same reason sdlqr.py pins CPU for
            # scipy's CARE). The metric is re-solved every step, so there is no
            # network to build.
            self._compute_device = "cpu"
            self._ccm_gen = None
        else:  # "pretrained"
            self._compute_device = device
            cmg_model = models.get("cmg")
            if cmg_model is None:
                raise ValueError(
                    "CVSTEMLQRAgent: metric_source='pretrained' requires a 'cmg' model "
                    "(MetricModel) in `models`."
                )
            self._ccm_gen = cmg_model.ccm_gen
            self._synthesize_pretrained_cmg()

    # ── pretrained-CMG synthesis (Phase A, once at construction) ───────────── #

    def _synthesize_pretrained_cmg(self) -> None:
        """Solve the CV-STEM SDP over sampled states and MSE-regress the frozen
        CMG onto ``{x → W*}`` — the exact ``build_cm_dataset`` + ``regress_cmg``
        pipeline C2RL uses for ``cmg_method='cvstem'`` (see
        ``ncm_synthesis.py``), cached to ``cm_data_path``. Runs once, then the
        CMG is frozen for the whole rollout."""
        from pathlib import Path

        from .ncm_synthesis import (
            build_cm_dataset,
            load_cached_cm_dataset,
            regress_cmg,
            save_cm_dataset,
        )

        cfg = self._cfg
        tag = "[CVSTEM-LQR]"
        cache_path = Path(cfg.cm_data_path) if cfg.cm_data_path else None
        # Cache-identity knobs — must match load/save so a config change re-solves
        # rather than silently reusing stale W* targets (see load_cached_cm_dataset).
        cache_kwargs = dict(
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            solver=cfg.cm_solver, num_samples=cfg.cmg_memory_size, tag=tag,
            r_scaler=cfg.r_scaler, chi_weight=None, nu_weight=1.0,
            wdot_dt=0.0, random_ratio=0.0, wdot_trajectory=False, temporal_dt=0.0,
        )

        dataset = load_cached_cm_dataset(cache_path, **cache_kwargs) if cache_path else None
        if dataset is None:
            dataset = build_cm_dataset(
                self._get_rollout, self._get_f_and_B,
                x_dim=self._x_dim, lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub,
                eps=cfg.cm_eps, num_samples=cfg.cmg_memory_size, solver=cfg.cm_solver,
                device=self.device, tag=tag,
                min_feasibility_rate=cfg.min_feasibility_rate,
                r_scaler=cfg.r_scaler, max_lambda_reductions=cfg.max_lambda_reductions,
            )
            if cache_path is not None:
                save_cm_dataset(cache_path, dataset, **cache_kwargs)

        bounded = getattr(self._ccm_gen, "bounded", False)
        regress_cmg(
            self._ccm_gen, dataset,
            w_lb=cfg.w_lb, x_dim=self._x_dim, bounded=bounded,
            epochs=cfg.cmg_regress_epochs, lr=cfg.cmg_regress_lr,
            batch_size=cfg.cmg_regress_batch_size,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            device=self.device, tag=tag,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
        )
        for p in self._ccm_gen.parameters():
            p.requires_grad_(False)
        self._ccm_gen.eval()
        print(f"{tag} Phase A complete — CMG frozen "
              f"(feasibility_rate={dataset['feasibility_rate']:.1%}).")

    # ── action computation ─────────────────────────────────────────────────── #

    def _split_obs(self, obs: torch.Tensor):
        x_dim, u_dim = self._x_dim, self._u_dim
        x = obs[:, :x_dim]
        xref = obs[:, x_dim : 2 * x_dim]
        uref = obs[:, 2 * x_dim : 2 * x_dim + u_dim]
        return x, xref, uref

    def _compute_action_pretrained(self, obs: torch.Tensor) -> torch.Tensor:
        """Vectorized on-device gain from the frozen CMG:
        M(x) = spd_inverse(bound_W(CMG(x))), K = R⁻¹BᵀM, u = uref - K·e."""
        cfg = self._cfg
        r = cfg.r_scaler + 1e-5  # strictly positive — mirrors sdlqr.py's guard
        x, xref, uref = self._split_obs(obs.to(self._compute_device))

        with torch.no_grad():
            _f, B, _Bbot = self._get_f_and_B(x)          # B: (b, x_dim, u_dim)
            B = B.to(torch.float32)
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, cfg.w_lb, self._x_dim, getattr(self._ccm_gen, "bounded", False))
            M = spd_inverse(W)                            # (b, x_dim, x_dim)
            K = (1.0 / r) * torch.bmm(B.transpose(1, 2), M)  # (b, u_dim, x_dim)
            e = wrap_diff(x - xref, self._angle_idx).unsqueeze(-1)  # (b, x_dim, 1)
            u = uref - torch.bmm(K, e).squeeze(-1)        # (b, u_dim)
        return u

    def _compute_action_online(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-env CV-STEM SDP solve at the current state, then K = R⁻¹BᵀM.

        Uses the DRIFT Jacobian A(x) = ∂f/∂x (not SD-LQR's generalized
        A = ∂f/∂x + Σ uref·∂B/∂x) — that is exactly what solve_cm_metric's LMI
        expects, with control entering only through the Riccati term."""
        cfg = self._cfg
        r = cfg.r_scaler + 1e-5
        x_dim, u_dim = self._x_dim, self._u_dim
        batch_size = obs.shape[0]

        x = obs[:, :x_dim].float().to(self._compute_device).requires_grad_()
        xref = obs[:, x_dim : 2 * x_dim].float().to(self._compute_device)
        uref = obs[:, 2 * x_dim : 2 * x_dim + u_dim].float().to(self._compute_device)

        with torch.enable_grad():
            f, B, _ = self._get_f_and_B(x)
        f = f.float().to(self._compute_device)
        B = B.float().to(self._compute_device)
        DfDx = jacobian(f, x, create_graph=False)         # (b, x, x) — drift Jacobian
        B = B.detach()

        DfDx_np = DfDx.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()

        actions = torch.zeros(batch_size, u_dim, device=self._compute_device)
        e_batch = wrap_diff(x.detach() - xref, self._angle_idx)
        for i in range(batch_size):
            W, _lbd_used, _red = _solve_cm_metric_with_backoff(
                DfDx_np[i], B_np[i],
                lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
                solver=cfg.cm_solver, r_scaler=cfg.r_scaler,
                max_lambda_reductions=cfg.max_lambda_reductions,
            )
            if W is None:
                # Infeasible at this state even after λ-backoff → zero feedback
                # (u = uref), rather than aborting the batch. Same policy as
                # SD-LQR's non-stabilizable CARE fallback.
                actions[i] = uref[i]
                continue
            M = np.linalg.inv(W.astype(np.float64))       # W = M⁻¹ → M
            K = (1.0 / r) * (B_np[i].T @ M)               # (u, x)
            K_t = torch.as_tensor(K, dtype=torch.float32, device=self._compute_device)
            actions[i] = uref[i] - K_t @ e_batch[i]
        return actions

    def act(self, observations, states, *, timestep: int, timesteps: int):
        orig_device = observations.device
        if self._metric_source == "pretrained":
            actions = self._compute_action_pretrained(observations)
        else:
            with torch.enable_grad():  # Jacobian needs grad; cvxpy solve is non-diff
                actions = self._compute_action_online(observations)
        actions = actions.detach().to(orig_device)
        log_prob = torch.zeros(actions.shape[0], 1, device=orig_device)
        return actions, {"log_prob": log_prob}

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def record_transition(
        self, *, observations, states, actions, rewards, next_observations,
        next_states, terminated, truncated, infos, timestep, timesteps,
    ) -> None:
        super().record_transition(
            observations=observations, states=states, actions=actions,
            rewards=rewards, next_observations=next_observations,
            next_states=next_states, terminated=terminated, truncated=truncated,
            infos=infos, timestep=timestep, timesteps=timesteps,
        )

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass  # analytical — no gradient updates

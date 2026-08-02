"""The one tracking reward, shared by the classic and Isaac env families.

Both families reward the same thing: drive ``x`` onto ``xrefs[0]``. They used to
say so in two hand-written copies that had quietly diverged — the Isaac copy
ignored ``tracking_scaler``/``control_scaler`` entirely and had no control
term, so a C2RL sweep over ``cm.tracking_scaler`` was a silent no-op on Isaac
and the two families optimized different objectives under the same config.

Everything here is written in terms of a scalar potential

    V = ‖e‖²_M = eᵀ M e        (M = I when no contraction metric is injected)

and the reward is one of two forms of it:

    potential (default)  r = q·(V - V′) - r·‖u‖²
    level                r = -q·V′      - r·‖u‖²

The potential form is potential-based shaping in the Ng et al. sense (Φ = -V,
γ = 1): it telescopes over an episode to ``q·(V₀ - V_T)``, so it cannot change
which policy is optimal, only how fast the value function learns. The level
form is the plain quadratic cost; it is what a shorter-horizon run wants, since
the decrement's per-step signal vanishes near the optimum exactly where the
advantage estimate needs it (see the C2RL single-update-collapse note in
env_base.get_rewards).
"""
from __future__ import annotations

import torch


def mahalanobis_sq(e: torch.Tensor, M: torch.Tensor | None) -> torch.Tensor:
    """``eᵀMe`` per batch element, ``(N, d) x (N, d, d) -> (N,)``.

    ``M=None`` means the identity metric, i.e. the plain squared Euclidean norm
    — that is the only difference between the Euclidean and Mahalanobis rewards,
    so both callers go through this one function rather than branching.
    """
    if M is None:
        return e.pow(2).sum(dim=-1)
    return torch.einsum("bi,bij,bj->b", e, M, e)


def metric_from_ccm(ccm_gen, x: torch.Tensor, w_lb: float) -> torch.Tensor:
    """``M(x) = bound_W(W(x), w_lb)⁻¹`` for a frozen CMG.

    The bound-then-invert dance was open-coded in five places across the two env
    bases (reward, reset seeding, residual anchor, x2 families); an ``x_dim``
    that disagreed with ``x.shape[-1]`` in any one of them silently produced a
    wrongly-shaped identity offset.
    """
    from contractionRL.agents.skrl.math_utils import bound_W, spd_inverse
    W_raw, _ = ccm_gen(x)
    W = bound_W(W_raw, w_lb, x.shape[-1], getattr(ccm_gen, "bounded", False))
    return spd_inverse(W)


def shaped_reward(V: torch.Tensor, V_next: torch.Tensor, control_effort: torch.Tensor,
                  *, tracking_scaler: float, control_scaler: float,
                  level: bool) -> torch.Tensor:
    """``q·(V - V′) - r·‖u‖²`` (potential) or ``-q·V′ - r·‖u‖²`` (level).

    ``V`` is ignored in the level form; callers that only have the post-
    transition potential may pass it as ``V_next``.
    """
    shaped = -tracking_scaler * V_next if level else tracking_scaler * (V - V_next)
    if not control_scaler:
        # NOT `- 0.0 * control_effort`: the Isaac path-tracking configs set
        # control_scaler to 0.0, and 0.0 * nan is nan, so a single non-finite
        # action from a diverging policy would poison the whole reward (and with
        # it the episode return and every downstream normalizer) even though the
        # control term is supposed to be switched OFF. Skip the term entirely.
        return shaped
    return shaped - control_scaler * control_effort


def tracking_reward(error: torch.Tensor, next_error: torch.Tensor,
                    control_effort: torch.Tensor, *,
                    tracking_scaler: float, control_scaler: float,
                    M: torch.Tensor | None = None,
                    next_M: torch.Tensor | None = None,
                    reward_level: bool = False,
                    reward_euclidean: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """THE tracking reward. Both env families call exactly this.

    ``error``/``next_error`` are the pre- and post-transition tracking errors,
    already angle-wrapped by the caller (each family wraps differently: classic
    by ``angle_idx`` on the full state, Isaac by ``wrap_diff``).

    ``M``/``next_M`` are the contraction metric at those two states, or ``None``
    for the identity metric. ``reward_euclidean`` forces the identity metric
    even when a metric is supplied — the AUC-aligned variant that lets a learned
    residual minimize the TRUE tracking error rather than the frozen-CMG proxy
    its analytic base already ~minimizes.

    Returns ``(reward, next_V)``, where ``next_V`` is the Mahalanobis tracking
    error eᵀMe for the metrics tab (``None`` whenever the metric is the
    identity, since then it would just duplicate the Euclidean error curve).
    """
    if reward_euclidean:
        M = next_M = None
    # The POTENTIAL form needs a potential worth telescoping: the frozen CMG's
    # V, or an explicitly requested Euclidean decrement. With neither (plain
    # PPO/SAC/LQR/C3M baselines) the reward is the level quadratic cost, which
    # is what it has always been in both families — reward_level is a C2RL knob
    # and must not silently reshape the baselines.
    shaping_requested = next_M is not None or reward_euclidean
    level = reward_level or not shaping_requested

    if reward_euclidean and level:
        # The one form that is NOT ‖e‖²_M: LEVEL-euclidean uses the LINEAR norm
        # r = -q‖e‖. AUC = ∫‖e‖/‖e0‖ dt, so the discounted sum of -‖e‖ IS
        # (minus) the error integral — the tightest possible alignment, and
        # squaring would reweight long tails away from what AUC measures. The
        # DECREMENT form telescopes to the ENDPOINT error e0²-eT², which a
        # dawdle-then-settle policy games while keeping AUC high (measured
        # plateau at 0.96).
        reward = -tracking_scaler * torch.norm(next_error, p=2, dim=-1) \
            - control_scaler * control_effort
        return reward, None

    V = mahalanobis_sq(error, M)
    next_V = mahalanobis_sq(next_error, next_M)
    reward = shaped_reward(V, next_V, control_effort,
                           tracking_scaler=tracking_scaler,
                           control_scaler=control_scaler, level=level)
    return reward, (next_V if next_M is not None else None)


def _self_check() -> None:
    """Assert the shared reward reproduces both families' original branches.

    Run: ``python -m contractionRL.tasks.direct.common.reward``
    """
    torch.manual_seed(0)
    n, d = 6, 4
    e, e2, ce = torch.randn(n, d), torch.randn(n, d), torch.rand(n)
    A, B = torch.randn(n, d, d), torch.randn(n, d, d)
    M, M2 = A @ A.transpose(1, 2) + torch.eye(d), B @ B.transpose(1, 2) + torch.eye(d)
    q, r = 1.7, 0.3

    def original(euclidean, level, ccm):
        """The pre-refactor branches, transcribed from env_base.get_rewards."""
        if euclidean:
            if level:
                return -q * e2.norm(dim=-1) - r * ce
            return q * (e.norm(dim=-1) ** 2 - e2.norm(dim=-1) ** 2) - r * ce
        if ccm:
            V = torch.einsum("bi,bij,bj->b", e, M, e)
            V2 = torch.einsum("bi,bij,bj->b", e2, M2, e2)
            return (-q * V2 - r * ce) if level else (q * (V - V2) - r * ce)
        return -q * (e2.norm(dim=-1) ** 2) - r * ce   # plain: LEVEL, always

    for euclidean in (False, True):
        for level in (False, True):
            for ccm in (False, True):
                if euclidean and not ccm:
                    continue          # only ever set alongside a CMG
                got, _ = tracking_reward(
                    e, e2, ce, tracking_scaler=q, control_scaler=r,
                    M=(M if ccm else None), next_M=(M2 if ccm else None),
                    reward_level=level, reward_euclidean=euclidean)
                want = original(euclidean, level, ccm)
                assert torch.allclose(got, want, atol=1e-5), (
                    f"euclidean={euclidean} level={level} ccm={ccm}: "
                    f"max|diff|={(got - want).abs().max():.3e}")

    # The plain baseline must stay the LEVEL quadratic even at reward_level=False
    # — reward_level is a C2RL knob and must not reshape PPO/SAC/LQR/C3M.
    plain, maha = tracking_reward(e, e2, ce, tracking_scaler=q, control_scaler=r)
    assert torch.allclose(plain, -q * (e2.norm(dim=-1) ** 2) - r * ce, atol=1e-5)
    assert maha is None, "identity metric must not report a Mahalanobis error"
    print("reward self-check OK: all 6 branches match the originals")


if __name__ == "__main__":
    _self_check()

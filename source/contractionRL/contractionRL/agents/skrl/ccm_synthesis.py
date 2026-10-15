"""Direct CCM metric training: Killing, unactuated contraction, input feasibility.

E(x, xref) = wrap(x-xref)' M(x) wrap(x-xref), matching the C2RL reward.
All three conditions are sampled losses, not uniform certificates. There is
no learned controller or controller-Jacobian contraction loss in this module.
"""

import torch

from .angle_utils import wrap_diff
from .math_utils import EarlyStopper, Jacobian_Matrix, jacobian, b_jacobian, train_val_split, quadratic_energy


def input_box_minimum(a, b, low, high):
    """Exact min_u a + b'u over a componentwise box (also asymmetric boxes)."""
    return a + torch.minimum(b * low, b * high).sum(-1)


def ccm_residuals(metric, get_f_and_B, x, xref, uref, low, high, *, lbd, angle_idx=()):
    """Per-example full Killing matrices, projected drift matrix, and CLF residual.

    N spans ker(B' M). The nullspace of B' is computed from the actual B,
    rather than assuming the actuated state coordinates are the final m rows.
    Dynamics are fixed during CMG training; gradients through M and D M remain.
    """
    x = x.detach().requires_grad_(True)
    xref, uref = xref.detach().to(x), uref.detach().to(x)
    M, _ = metric(x)
    dM = Jacobian_Matrix(M, x)
    f, B, _ = get_f_and_B(x)
    Jf = jacobian(f, x, create_graph=False).detach()
    Jb = b_jacobian(B, x, B.shape[-1], create_graph=False).detach()
    f, B = f.detach(), B.detach()
    fref, Bref, _ = get_f_and_B(xref)
    ref_velocity = (fref + (Bref @ uref.unsqueeze(-1)).squeeze(-1)).detach()
    DfM = torch.einsum("bijk,bk->bij", dM, f)
    Q = DfM + Jf.mT @ M + M @ Jf + 2 * lbd * M
    DbM = torch.einsum("bijk,bku->buij", dM, B)
    Jb = Jb.permute(0, 3, 1, 2)
    killing = DbM + Jb.mT @ M[:, None] + M[:, None] @ Jb

    n, m = B.shape[-2:]
    if m > n:
        raise ValueError("CCM training requires independent actuation fields (m <= n).")
    U, singular, _ = torch.linalg.svd(B, full_matrices=True)
    tolerance = torch.finfo(B.dtype).eps * max(n, m) * singular[:, :1]
    if bool((singular <= tolerance).any()):
        raise ValueError("CCM training requires full-column-rank B on the sampled domain.")
    if n > m:
        N = torch.linalg.solve(M, U[:, :, m:])
        N = torch.linalg.qr(N, mode="reduced").Q
        projected = N.mT @ Q @ N
        projected = 0.5 * (projected + projected.mT)
    else:
        N = M[:, :, :0]
        projected = M[:, :0, :0]

    error = wrap_diff(x - xref, angle_idx)
    E = quadratic_energy(error, M)
    # Exact derivative of this quadratic tracking energy, including the moving
    # reference. The box constrains TOTAL input u, not the residual u-uref.
    a = (torch.einsum("bi,bij,bj->b", error, DfM, error)
         + 2 * torch.einsum("bi,bij,bj->b", error, M, f - ref_velocity))
    b = (torch.einsum("bi,buij,bj->bu", error, DbM, error)
         + 2 * torch.einsum("bi,bij,bju->bu", error, M, B))
    feasibility = input_box_minimum(a, b, low, high) + 2 * lbd * E
    return {"killing": killing, "projected": projected, "feasibility": feasibility,
            "energy": E, "a": a, "b": b, "N": N}


def ccm_loss(residuals, *, eps, killing_weight=1.0, contraction_weight=1.0, feasibility_weight=1.0):
    killing = residuals["killing"].square().sum(dim=(-1, -2)).mean()
    projected = residuals["projected"]
    if projected.shape[-1]:
        largest = torch.linalg.eigvalsh(projected)[:, -1]
        contraction = (largest + eps).clamp_min(0).square().mean()
        contraction_max = largest.max()
    else:
        contraction = killing * 0
        contraction_max = contraction.detach()
    residual = residuals["feasibility"]
    feasibility = residual.clamp_min(0).square().mean()
    loss = killing_weight * killing + contraction_weight * contraction + feasibility_weight * feasibility
    stats = {"loss": loss.detach(), "killing_loss": killing.detach(),
             "contraction_loss": contraction.detach(), "feasibility_loss": feasibility.detach(),
             "killing_max": residuals["killing"].detach().abs().max(),
             "contraction_max": contraction_max.detach(), "feasibility_max": residual.detach().max()}
    return loss, {key: float(value) for key, value in stats.items()}


def train_cmg_ccm(metric, get_f_and_B, data, low, high, *, cfg, device, angle_idx=(), on_epoch=None):
    """Train only the metric, restoring the best held-out-loss weights."""
    if not getattr(metric, "bounded", False) or not getattr(metric, "outputs_metric", False):
        raise ValueError("CCM synthesis requires a bounded M-emitting CMG.")
    if cfg.cmg_ccm_epochs < 1 or cfg.cmg_ccm_batch_size < 1 or cfg.cmg_ccm_lr <= 0:
        raise ValueError("CCM epochs, batch size, and learning rate must be positive.")
    weights = (cfg.ccm_killing_weight, cfg.ccm_contraction_weight, cfg.ccm_feasibility_weight)
    if any(weight <= 0 for weight in weights) or cfg.cm_eps <= 0 or cfg.lbd < 0:
        raise ValueError("CCM loss weights and cm_eps must be positive; lbd must be nonnegative.")
    tensors = {key: torch.as_tensor(data[key], dtype=torch.float32).detach().cpu()
               for key in ("x", "xref", "uref")}
    count = len(tensors["x"])
    if count < 1 or any(len(value) != count or not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("CCM samples must be finite, nonempty, aligned x/xref/uref arrays.")
    low, high = (torch.as_tensor(bound, dtype=torch.float32, device=device).flatten() for bound in (low, high))
    if low.numel() != tensors["uref"].shape[-1] or high.shape != low.shape:
        raise ValueError("CCM control bounds must match the input dimension.")
    if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not bool((low < high).all()):
        raise ValueError("CCM input feasibility requires finite, strictly ordered control bounds.")
    if bool(((tensors["uref"].to(device) < low) | (tensors["uref"].to(device) > high)).any()):
        raise ValueError("CCM reference inputs must lie in the applied input box.")
    train_idx, val_idx = train_val_split(count, cfg.cmg_val_frac, device="cpu")
    if not len(train_idx):
        raise ValueError("CCM validation split leaves no training samples.")
    stopper = EarlyStopper(patience=cfg.cmg_early_stop_patience if len(val_idx) else 0)
    optimizer = torch.optim.Adam(metric.parameters(), lr=cfg.cmg_ccm_lr)

    def evaluate(indices, train):
        totals, maxima, seen = {}, {}, 0
        for start in range(0, len(indices), cfg.cmg_ccm_batch_size):
            selection = indices[start:start + cfg.cmg_ccm_batch_size]
            batch = {key: value[selection].to(device) for key, value in tensors.items()}
            # Validation needs state derivatives as well; no torch.no_grad().
            with torch.enable_grad():
                residuals = ccm_residuals(metric, get_f_and_B, **batch, low=low, high=high,
                                          lbd=cfg.lbd, angle_idx=angle_idx)
                loss, stats = ccm_loss(residuals, eps=cfg.cm_eps, killing_weight=weights[0],
                                       contraction_weight=weights[1], feasibility_weight=weights[2])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite CCM metric loss.")
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(metric.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step()
            for key, value in stats.items():
                if key.endswith("_max"):
                    maxima[key] = max(maxima.get(key, -float("inf")), value)
                else:
                    totals[key] = totals.get(key, 0.0) + value * len(selection)
            seen += len(selection)
        return {**{key: value / seen for key, value in totals.items()}, **maxima}

    for epoch in range(cfg.cmg_ccm_epochs):
        metric.train()
        train_stats = evaluate(train_idx[torch.randperm(len(train_idx))], True)
        metric.eval()
        validation = evaluate(val_idx, False) if len(val_idx) else evaluate(train_idx, False)
        stop = stopper.step(validation["loss"], metric, epoch)
        if on_epoch:
            on_epoch(epoch, train_stats, validation)
        if stop:
            break
    stopper.restore_best(metric)
    metric.eval()
    validation = evaluate(val_idx if len(val_idx) else train_idx, False)
    return {"method": "ccm", "epochs": epoch + 1, "validation": validation,
            "energy": "wrapped_tracking_error_quadratic", "closed_loop_loss": False, "certified": False,
            "training_samples": len(train_idx), "validation_samples": len(val_idx),
            "u_min": low.detach().cpu().tolist(), "u_max": high.detach().cpu().tolist()}

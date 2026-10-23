"""Learned control-affine dynamics, transition datasets, and dynamics pretraining."""
from __future__ import annotations

from collections.abc import Sequence
import os
import sys
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as _tqdm

from contractionRL.agents.utils.RL.networks import MLP, _ACT_MAP
from contractionRL.agents.utils.contraction.math_utils import EarlyStopper, train_val_split
from contractionRL.agents.utils.system_dynamics.state import StateSymmetry, embed_angles, embedded_dim


class ErfActivation(nn.Module):
    def forward(self, x):
        return torch.erf(x)


RAF_ACTIVATIONS = ("tanh", "elu", "relu", "leaky_relu", "gelu")
_DYNAMICS_ACT_MAP = _ACT_MAP | {
    "gelu": nn.GELU, "softsign": nn.Softsign, "silu": nn.SiLU,
    "selu": nn.SELU, "erf": ErfActivation, "identity": nn.Identity,
}


def dynamics_environment_metadata(env) -> dict:
    """Physical derivative rows and applied-control coordinates used by a model."""
    if not hasattr(env, "x_dim") and getattr(env, "envs", None):
        env = env.envs[0].unwrapped
    dt = getattr(env, "dt", None) or getattr(env, "step_dt", None)
    coordinates = getattr(env, "action_coordinates", None)
    if coordinates is None:
        coordinates = getattr(getattr(env, "_traj_buf", None), "action_coordinates", None)
    cfg = getattr(env, "cfg", None)
    scales = {name: torch.as_tensor(getattr(cfg, name)).tolist()
              for name in sorted(dir(cfg)) if name.startswith("action_scale") and getattr(cfg, name) is not None}
    return {"x_dim": int(env.x_dim), "u_dim": int(env.u_dim),
            "state_names": list(getattr(env, "state_names", ())),
            "angle_idx": list(getattr(env, "angle_idx", ())),
            "dt": None if dt is None else float(dt), "action_coordinates": coordinates,
            "action_scales": scales or None}


def validate_dynamics_metadata(saved: dict, expected: dict) -> None:
    missing = []
    for name, value in expected.items():
        if value is None or (name == "state_names" and not value):
            continue
        actual = saved.get(name)
        if actual is None or (name == "state_names" and not actual):
            missing.append(name)
        elif not (math.isclose(float(actual), float(value), rel_tol=1e-6, abs_tol=1e-9)
                  if name == "dt" else actual == value):
            raise ValueError(f"Dynamics checkpoint {name} does not match the environment")
    if missing:
        warnings.warn("Legacy dynamics checkpoint lacks " + ", ".join(missing)
                      + "; physical state/control compatibility is unverified. Use its original environment and source.",
                      RuntimeWarning, stacklevel=2)


@torch.no_grad()
def collect_stochastic_rollouts(env, count, *, low, high, device, dt, angle_idx=(), path=None):
    """Collect executed exploratory controls and disturbed plant derivatives.

    The raw simulator is used so warmup transitions never enter PPO memory or
    evaluation statistics. Terminal derivatives precede autoreset and clipping.
    """
    if count < 2 or dt <= 0:
        raise ValueError("Stochastic dynamics collection requires at least two samples and positive dt")
    if path and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite stochastic rollout evidence: {path}")
    low, high = (torch.as_tensor(v, device=device) for v in (low, high))
    def physical():
        if hasattr(env, "get_physical_state"):
            return env.get_physical_state().clone()
        if hasattr(env, "_get_physical_state"):
            return env._get_physical_state().clone()
        return env.x_t.clone()
    env.reset()
    groups = torch.arange(env.num_envs, device=device)
    next_group = env.num_envs
    records = {key: [] for key in ("x", "u", "x_dot", "trajectory_id")}
    seen = 0
    for _ in range(max(100, 100 * math.ceil(count / env.num_envs))):
        x = physical()
        u = low + torch.rand(env.num_envs, low.numel(), device=device) * (high - low)
        _, _, terminated, truncated, infos = env.step(u)
        if "dynamics_x_dot" in infos:
            derivative = infos["dynamics_x_dot"]
        else:
            endpoint = infos.get("dynamics_x_next", infos.get("terminal_physical_state"))
            if endpoint is None:
                raise RuntimeError("Simulator must expose raw pre-autoreset endpoints for stochastic dynamics learning")
            derivative = (endpoint - x) / dt
        valid = torch.isfinite(x).all(-1) & torch.isfinite(derivative).all(-1)
        for key, value in (("x", x), ("u", u), ("x_dot", derivative), ("trajectory_id", groups)):
            records[key].append(value[valid].clone())
        seen += int(valid.sum())
        done = (terminated | truncated).reshape(-1).bool()
        n_done = int(done.sum())
        groups[done] = torch.arange(next_group, next_group + n_done, device=device)
        next_group += n_done
        if seen >= count:
            break
    else:
        raise RuntimeError("Unable to collect enough finite disturbed rollout transitions")
    data = {key: torch.cat(values)[:count].cpu().numpy() for key, values in records.items()}
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **{key: value[:, None] for key, value in data.items()},
                            dt=dt, source="stochastic_exploration_rollouts",
                            coordinate_frame="world",
                            seed=torch.initial_seed(),
                            disturbance_enabled=bool(getattr(getattr(env, "disturbance", None), "enabled", False)))
    env.reset()
    return data


def load_offline_dynamics_data(data_path: str, tag: str = "", *, require_disturbed: bool = False) -> dict:
    """Load + flatten an offline ``dynamics_data.npz`` (keys ``x``, ``u``,
    ``x_dot``, optional ``lengths``) into flat ``(n, dim)`` arrays.

    Filters out whole episodes containing any NaN, and — when a ``lengths``
    array is present — masks out padding steps (the last valid state repeated
    past each episode's true length) so downstream fits/samples aren't biased
    toward artificial ``x_dot ~ 0`` padding. Shared by ``pretrain_dynamics``
    (NeuralDynamics fit) and C2RL's ``synthesize_cmg`` (CMG SDP-dataset states),
    so both read the same offline states the same way.
    """
    print(f"{tag} Loading dynamics pretrain data from {data_path}")
    npz = np.load(data_path)
    if require_disturbed and (
        str(npz.get("reference_source", "")) == "physx_nominal_rollout"
        or ("disturbance_enabled" in npz and not bool(npz["disturbance_enabled"]))
    ):
        npz.close()
        raise ValueError("Nominal dynamics data cannot identify the disturbance density; supply disturbed rollouts or omit dynamics_pretrain_data_path to collect them")
    if require_disturbed and "disturbance_enabled" not in npz:
        npz.close()
        raise ValueError("Disturbed dynamics fitting requires explicit disturbance_enabled=True provenance; regenerate actual disturbed rollouts or omit dynamics_pretrain_data_path to collect them")
    if str(npz.get("coordinate_frame", "world")) != "world":
        npz.close()
        raise ValueError("Dynamics training requires world-coordinate data; use the archive's original source for other frames")

    arrays = [npz[key] for key in ("x", "u", "x_dot")]
    if any(a.ndim not in (2, 3) or a.shape[:-1] != arrays[0].shape[:-1] for a in arrays):
        raise ValueError("Dynamics data must contain aligned flat or trajectory-shaped x/u/x_dot arrays")
    nan_mask = np.logical_or.reduce([~np.isfinite(a).all(axis=tuple(range(1, a.ndim))) for a in arrays])
    groups = npz["trajectory_id"] if "trajectory_id" in npz else None
    if groups is not None:
        groups = np.broadcast_to(groups.reshape(-1, 1) if groups.ndim == 1 and arrays[0].ndim == 3 else groups,
                                 arrays[0].shape[:-1])
    lengths_arr = npz["lengths"] if "lengths" in npz else None
    if nan_mask.any():
        print(f"{tag} WARNING: Found NaNs in {nan_mask.sum()} offline episodes! Filtering them out...")
        valid_mask = ~nan_mask
        x_arr = npz["x"][valid_mask]
        u_arr = npz["u"][valid_mask]
        x_dot_arr = npz["x_dot"][valid_mask]
        if lengths_arr is not None:
            lengths_arr = lengths_arr[valid_mask]
        if groups is not None:
            groups = groups[valid_mask]
    else:
        x_arr, u_arr, x_dot_arr = npz["x"], npz["u"], npz["x_dot"]

    if lengths_arr is not None:
        T_len = x_arr.shape[1]
        step_mask = np.arange(T_len)[None, :] < lengths_arr[:, None]  # (N, T)
        x_arr = x_arr[step_mask]
        u_arr = u_arr[step_mask]
        x_dot_arr = x_dot_arr[step_mask]
        if groups is not None:
            groups = groups[step_mask]
        print(f"{tag} Unpacked {step_mask.sum()} valid samples from {step_mask.shape[0]} trajectories (padding masked)")

    x_arr = x_arr.reshape(-1, x_arr.shape[-1])
    u_arr = u_arr.reshape(-1, u_arr.shape[-1])
    x_dot_arr = x_dot_arr.reshape(-1, x_dot_arr.shape[-1])
    result = {
        "x": x_arr.astype(np.float32),
        "u": u_arr.astype(np.float32),
        "x_dot": x_dot_arr.astype(np.float32),
    }
    if groups is not None:
        result["trajectory_id"] = groups.reshape(-1).copy()
    npz.close()
    return result


def pretrain_dynamics(agent, *, epochs: int, data_path: str | None,
                      timesteps: int, memory_size: int | None = None,
                      num_controls_per_state: int | None = None,
                      log_interval: int | None = None,
                      tag: str = "",
                      val_frac: float = 0.1,
                      early_stop_patience: int = 10,
                      step_offset: int | None = None,
                      require_disturbed: bool = False) -> None:
    """Pretrain ``agent._neural_dynamics`` for ``epochs`` epochs over a fixed
    ``memory_size``-sized buffer of ``(x, u, x_dot)`` samples, drawn once before
    the epoch loop — the same "sample a fixed dataset, then multi-epoch it"
    structure ``ncm_synthesis.build_cm_dataset``/``regress_cmg`` use for CMG
    synthesis (see ``C2RLAgent._sample_cmg_x``/``synthesize_cmg``), so both
    pretraining phases behave the same way.

    The buffer comes from ``load_offline_dynamics_data(data_path)`` when
    ``data_path`` points at an offline ``dynamics_data.npz`` — uniformly
    subsampled (without replacement) to ``memory_size``, with a warning and a
    cap to what's on disk if ``memory_size`` asks for more (Isaac envs
    typically only have this fixed on-disk supply). Otherwise it's drawn fresh
    via ``agent._get_rollout(memory_size, "dynamics", num_control_per_state=...)``
    — classic envs can feasibly sample any ``memory_size`` since it's synthetic
    analytic sampling, no cap needed; Isaac envs without a data_path instead
    use their environment's data collector. Classic-derived PhysX tasks probe
    varied controls from reset states rather than fitting only nominal paths.

    No-op when the agent has no NeuralDynamics or epochs<=0. Analytical
    dynamics may still have a learned noise head. Metrics (dynamics NLL + LR) are flushed to the
    agent's writer on ``dynamics_epoch`` by default. With ``step_offset``, epoch
    indices instead use ``global_step = step_offset + epoch``, allowing C2RL to
    place dynamics before metric fitting at negative steps. The flush cadence
    is derived from ``epochs`` — not
    the agent's training ``write_interval``, which is typically
    ``timesteps//100`` and would never fire within a short pretraining loop —
    and the final epoch always flushes. ``log_interval`` optionally caps that
    cadence; ``tag`` labels console prints.

    ``val_frac`` holds out that fraction of the (fixed, once-sampled) buffer as
    a validation split never trained on; ``early_stop_patience`` stops the loop
    once the held-out NLL hasn't improved for that many consecutive epochs,
    restoring the best-val-epoch weights (see ``math_utils.EarlyStopper``).
    ``val_frac<=0`` or a validation split with 0 samples disables both and
    falls back to always running the full ``epochs`` budget.
    """
    if agent._neural_dynamics is None or epochs <= 0:
        return

    # ~100 wandb points regardless of the (huge) training write_interval.
    log_every = max(1, epochs // 100)
    if log_interval and log_interval > 0:
        log_every = min(log_every, log_interval)
    from contractionRL.agents.utils.RL.logging import pretraining_logging_enabled
    has_writer = pretraining_logging_enabled(agent)

    dev = agent._neural_dynamics.device

    supplied = getattr(agent, "_dynamics_rollout_data", None)
    data_groups = None
    if supplied is not None:
        x_np, u_np, xdot_np = (supplied[key] for key in ("x", "u", "x_dot"))
        data_groups = supplied.get("trajectory_id")
    elif data_path:
        offline = load_offline_dynamics_data(data_path, tag=tag, require_disturbed=require_disturbed)
        x_all, u_all, xdot_all = offline["x"], offline["u"], offline["x_dot"]
        n_avail = x_all.shape[0]
        n_samples = memory_size if memory_size is not None else n_avail
        if n_samples > n_avail:
            print(f"{tag} WARNING: emp_dynamics_memory_size={n_samples} exceeds the "
                  f"{n_avail} available offline dynamics samples — using {n_avail} instead.")
            n_samples = n_avail
        idx = np.random.choice(n_avail, size=n_samples, replace=False)
        x_np, u_np, xdot_np = x_all[idx], u_all[idx], xdot_all[idx]
        if "trajectory_id" in offline:
            data_groups = offline["trajectory_id"][idx]
        # C3M's later dynamics updates must keep using the accepted dataset.
        agent._dynamics_rollout_data = {key: value[idx] for key, value in offline.items()}
    else:
        n_samples = memory_size if memory_size is not None else agent._cfg.dynamics_batch_size
        data = agent._get_rollout(n_samples, "dynamics", num_control_per_state=num_controls_per_state)
        x_np, u_np, xdot_np = data["x"], data["u"], data["x_dot"]

    x = torch.as_tensor(x_np).to(torch.float32).to(dev)
    u = torch.as_tensor(u_np).to(torch.float32).to(dev)
    x_dot = torch.as_tensor(xdot_np).to(torch.float32).to(dev)
    n = x.shape[0]

    groups = data_groups
    if groups is not None and val_frac > 0 and len(np.unique(groups)) > 1:
        groups = torch.as_tensor(groups, device=dev)
        unique = groups.unique()
        unique = unique[torch.randperm(len(unique), device=dev)]
        heldout = unique[:max(1, min(len(unique) - 1, int(len(unique) * val_frac)))]
        mask = torch.isin(groups, heldout)
        train_idx, val_idx = (~mask).nonzero().flatten(), mask.nonzero().flatten()
    else:
        train_idx, val_idx = train_val_split(n, val_frac, device=dev)
    if not len(train_idx):
        raise ValueError("Dynamics validation split leaves no training samples")
    n_train = train_idx.shape[0]
    n_val = val_idx.shape[0]
    x_val, u_val, xdot_val = x[val_idx], u[val_idx], x_dot[val_idx]
    stopper = EarlyStopper(patience=early_stop_patience if n_val > 0 else 0)
    batches_per_epoch = max(1, n_train // agent._cfg.dynamics_batch_size)

    dyn_pbar = _tqdm.tqdm(range(epochs), desc="Pretraining dynamics", file=sys.stdout)
    for epoch in dyn_pbar:
        for _ in range(batches_per_epoch):
            dbz = min(agent._cfg.dynamics_batch_size, n_train)
            idx = train_idx[torch.randint(0, n_train, (dbz,), device=dev)]
            loss_val = agent._train_dynamics({"x": x[idx], "u": u[idx], "x_dot": x_dot[idx]})

        postfix = {"loss": f"{loss_val:.3g}"}

        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent._dynamics_lr_scheduler.step()

        probabilistic = hasattr(agent._neural_dynamics, "negative_log_likelihood")
        objective = "nll" if probabilistic else "mse"
        agent.track_data(f"Loss / Pretrain/dynamics_{objective}", loss_val)
        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_lr_scheduler.get_last_lr()[0])
        else:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_optimizer.param_groups[0]["lr"])

        stop = False
        if n_val > 0:
            with torch.no_grad():
                val_loss = (agent._neural_dynamics.negative_log_likelihood(x_val, u_val, xdot_val)
                            if probabilistic else F.mse_loss(
                                agent._neural_dynamics.predict_x_dot(x_val, u_val), xdot_val)).item()
            agent.track_data(f"Loss / Pretrain/dynamics_val_{objective}", val_loss)
            postfix["val"] = f"{val_loss:.3g}"
            stop = stopper.step(val_loss, agent._neural_dynamics, epoch)

        dyn_pbar.set_postfix(**postfix)

        if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1 or stop):
            from contractionRL.agents.utils.RL.logging import metric_batch
            step = epoch + 1 if step_offset is None else step_offset + epoch
            axis = "dynamics_epoch" if step_offset is None else "global_step"
            with metric_batch(step=step, axis=axis):
                agent.write_tracking_data(timestep=step, timesteps=timesteps)

        if stop:
            print(f"{tag} Dynamics pretraining early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val {objective.upper()} {stopper.best:.4g} @ epoch {stopper.best_epoch + 1}).")
            dyn_pbar.close()
            break

    if n_val > 0:
        # Restore the best-val-epoch weights whether the loop was early-stopped
        # or simply ran out its full epoch budget (the final epoch is not
        # necessarily the best one either way) — this print always fires so
        # the actually-used epoch/MSE is visible even when the loop ran to
        # completion without early stopping ever triggering.
        stopper.restore_best(agent._neural_dynamics)
        print(f"{tag} Dynamics pretraining: using best-val epoch "
              f"{stopper.best_epoch + 1}/{epochs} (val {objective.upper()} {stopper.best:.4g}).")



def _sym_from_names(names, mode=None):
    """Rebuild the symmetry a checkpoint was trained with, so a reload cannot
    silently change the network input layout."""
    if not names:
        return None
    symmetry = StateSymmetry.from_names(names)
    if mode == "translation":
        symmetry = symmetry.demote_to_translation()
    elif mode == "none":
        symmetry = symmetry.disabled()
    return symmetry


class NeuralDynamics(nn.Module):
    """Control-affine conditional Gaussian: xdot = f(x) + B(x)u + omega.

    omega has zero conditional mean and diagonal state/action-dependent
    covariance. An optional exact nominal callback keeps f/B analytical while
    fitting only the noise density. get_f_and_B never samples disturbances.
    """

    _dtype = torch.float32

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: Sequence[int] = (256, 256),
        activation: str = "relu",
        device: str | None = None,
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
        nominal_dynamics=None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.null_dim = x_dim - u_dim
        # f and B provably do not depend on the position dims (that is the very
        # property the quotient rests on), so the nets should not see them.
        self.sym = sym
        self._hidden_dim = list(hidden_dim)
        self._activation_str = activation
        self.device = torch.device(device or "cpu")
        self.angle_idx = list(angle_idx)
        self.nominal_dynamics = nominal_dynamics
        self.transition_metadata = {}

        if not isinstance(activation, str) or activation.lower() not in _DYNAMICS_ACT_MAP:
            raise ValueError(f"Unsupported dynamics activation: {activation!r}")
        act = _DYNAMICS_ACT_MAP[activation.lower()]()
        # Nets see the continuous embedding of x's angle dims; f/B are still
        # shaped/indexed by the raw x_dim (outputs are raw-coordinate ẋ / rows).
        emb_dim = (sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx))
        self.f_net = MLP(emb_dim, list(hidden_dim), x_dim, activation=act) if nominal_dynamics is None else None
        self.B_net = MLP(emb_dim, list(hidden_dim), x_dim * u_dim, activation=act) if nominal_dynamics is None else None
        self.disturbance_net = MLP(emb_dim + u_dim, list(hidden_dim), x_dim, activation=act)
        self.register_buffer("disturbance_fitted", torch.tensor(False))
        self.to(self.device)

    def get_f_and_B(self, x: torch.Tensor, *, need_null=True):
        """Return (f, B, B_null) — autodiff-compatible for C3M Jacobians."""
        if self.nominal_dynamics is not None:
            return self.nominal_dynamics(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self._dtype).to(self.device)
        x_emb = (self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        f = self.f_net(x_emb)
        B = self.B_net(x_emb).reshape(-1, self.x_dim, self.u_dim)
        B_null = self._compute_B_null(B) if need_null else None
        return f, B, B_null

    def _compute_B_null(self, B: torch.Tensor) -> torch.Tensor:
        n = B.shape[0]
        if self.null_dim <= 0:
            return torch.zeros(n, self.x_dim, 1, device=B.device, dtype=B.dtype)
        with torch.no_grad():
            U, _, _ = torch.linalg.svd(B.detach(), full_matrices=True)
        return U[:, :, self.u_dim:].contiguous()

    def predict_x_dot(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        f, B, _ = self.get_f_and_B(x, need_null=False)
        return f + (B @ u.unsqueeze(-1)).squeeze(-1)

    def disturbance_log_std(self, x, u):
        x = x.to(device=self.device, dtype=self._dtype)
        u = u.to(x)
        features = self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx)
        return self.disturbance_net(torch.cat((features, u), -1)).clamp(-7., 5.)

    def log_prob(self, x_dot, x, u):
        """log p_chi3(x_dot - f_chi1(x) - B_chi2(x)u | x,u)."""
        residual = x_dot - self.predict_x_dot(x, u)
        log_std = self.disturbance_log_std(x, u)
        return (-.5 * (residual.square() * (-2 * log_std).exp()
                       + 2 * log_std + math.log(2 * math.pi))).sum(-1)

    def negative_log_likelihood(self, x, u, x_dot):
        return -self.log_prob(x_dot, x, u).mean()

    @torch.no_grad()
    def sample_disturbance(self, x, u):
        if not bool(self.disturbance_fitted):
            raise RuntimeError("Conditional disturbance model is not fitted; collect disturbed transitions and train it first")
        return self.disturbance_log_std(x, u).exp() * torch.randn_like(x)

    @torch.no_grad()
    def sample_x_dot(self, x, u):
        return self.predict_x_dot(x, u) + self.sample_disturbance(x, u)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if prefix + "disturbance_fitted" not in state_dict:
            warnings.warn("Loading deterministic dynamics checkpoint: disturbance density is unfitted", stacklevel=2)
            for key, value in self.state_dict().items():
                if key.startswith("disturbance_"):
                    state_dict[prefix + key] = torch.zeros_like(value) if key == "disturbance_fitted" else value.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x.astype(np.float32)).to(self.device)
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0)
        return self.get_f_and_B(x)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "x_dim": self.x_dim,
                "u_dim": self.u_dim,
                "hidden_dim": self._hidden_dim,
                "activation": self._activation_str,
                "angle_idx": self.angle_idx,
                "state_names": list(getattr(self.sym, "names", ()) or ()),
                "symmetry_mode": getattr(self.sym, "mode", None),
                "nominal_source": "analytical" if self.nominal_dynamics is not None else "learned",
                "disturbance_density": ("gaussian_mixture" if hasattr(self, "members")
                                        else "zero_mean_diagonal_gaussian"),
                "ensemble_size": getattr(self, "ensemble_size", None),
                "member_activations": getattr(self, "member_activations", None),
                "transition_metadata": self.checkpoint_metadata(),
                "state_dict": self.state_dict(),
            },
            path,
        )
        print(f"[NeuralDynamics] saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None, *, nominal_dynamics=None,
             expected_metadata: dict | None = None) -> NeuralDynamics:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("transition_metadata", {key: ckpt.get(key) for key in
            ("x_dim", "u_dim", "state_names", "angle_idx", "dt", "action_coordinates", "action_scales")})
        validate_dynamics_metadata(metadata, {key: ckpt.get(key) for key in
            ("x_dim", "u_dim", "state_names", "angle_idx")})
        if expected_metadata is not None:
            validate_dynamics_metadata(metadata, expected_metadata)
        elif "transition_metadata" not in ckpt:
            warnings.warn("Legacy dynamics checkpoint has no transition_metadata; physical state/control compatibility is unverified.",
                          RuntimeWarning, stacklevel=2)
        if ckpt.get("nominal_source") == "analytical" and nominal_dynamics is None:
            raise ValueError("Analytical dynamics checkpoint requires its exact nominal_dynamics callable")
        symmetry = _sym_from_names(ckpt.get("state_names"), ckpt.get("symmetry_mode"))
        if symmetry is not None and "symmetry_mode" not in ckpt:
            weight = ckpt["state_dict"].get("f_net.net.0.weight")
            if weight is not None:
                candidates = (symmetry, symmetry.demote_to_translation(), symmetry.disabled())
                symmetry = next((candidate for candidate in candidates if candidate.single_dim() == weight.shape[1]), symmetry)
        factory = EnsembleDynamics if ckpt.get("ensemble_size") is not None else NeuralDynamics
        ensemble_kwargs = {}
        if factory is EnsembleDynamics:
            activations = ckpt.get("member_activations")
            if activations is None:
                if not isinstance(ckpt["activation"], str):
                    raise ValueError("Ensemble checkpoint is missing its member_activations assignment")
                activations = [ckpt["activation"]] * ckpt["ensemble_size"]
            ensemble_kwargs = {"ensemble_size": ckpt["ensemble_size"], "member_activations": activations}
        model = factory(
            x_dim=ckpt["x_dim"],
            u_dim=ckpt["u_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
            device=device,
            angle_idx=ckpt.get("angle_idx", []),
            sym=symmetry,
            nominal_dynamics=nominal_dynamics,
            **ensemble_kwargs,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.transition_metadata = metadata.copy()
        model.eval()
        print(f"[NeuralDynamics] loaded ← {path}  (x_dim={ckpt['x_dim']}, u_dim={ckpt['u_dim']})")
        return model

    def checkpoint_metadata(self) -> dict:
        return {"x_dim": self.x_dim, "u_dim": self.u_dim,
                "state_names": list(getattr(self.sym, "names", ()) or ()),
                "angle_idx": list(self.angle_idx), **self.transition_metadata}

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            result.device = torch.device(args[0])
        return result


class EnsembleDynamics(NeuralDynamics):
    """Independently initialized Gaussian models with distinct fixed activations."""

    def __init__(self, x_dim, u_dim, hidden_dim=(256, 256), activation=RAF_ACTIVATIONS, device=None,
                 angle_idx=(), sym=None, nominal_dynamics=None, *, ensemble_size=5, member_activations=None):
        nn.Module.__init__(self)
        if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, int) or ensemble_size < 1:
            raise ValueError("dynamics_ensemble_size must be a positive integer")
        pool = [activation] if isinstance(activation, str) else list(activation) if isinstance(activation, (list, tuple)) else []
        allowed = RAF_ACTIVATIONS if member_activations is None else _DYNAMICS_ACT_MAP
        if (not pool or any(not isinstance(name, str) or name not in allowed for name in pool)
                or len(set(pool)) != len(pool)):
            raise ValueError("Dynamics activation pool must contain distinct supported activation names")
        if member_activations is None:
            if ensemble_size > len(pool):
                raise ValueError("Dynamics activation pool needs at least dynamics_ensemble_size distinct functions")
            order = torch.randperm(len(pool))[:ensemble_size].tolist()
            member_activations = [pool[index] for index in order]
        elif (not isinstance(member_activations, (list, tuple)) or len(member_activations) != ensemble_size
              or any(name not in pool for name in member_activations)):
            raise ValueError("Checkpoint member_activations do not match the ensemble activation pool")
        self.member_activations = list(member_activations)
        # Separate constructors draw independent weights from the seeded RNG; no model is cloned.
        self.members = nn.ModuleList([
            NeuralDynamics(x_dim, u_dim, hidden_dim, name, device, angle_idx, sym, nominal_dynamics)
            for name in self.member_activations])
        first = self.members[0]
        for name in ("x_dim", "u_dim", "null_dim", "device", "angle_idx", "sym", "_hidden_dim",
                     "_activation_str", "nominal_dynamics"):
            setattr(self, name, getattr(first, name))
        self.ensemble_size = ensemble_size
        self._activation_str = activation if isinstance(activation, str) else list(activation)
        self.transition_metadata = {}
        self.register_buffer("disturbance_fitted", torch.tensor(False, device=self.device))

    def get_f_and_B(self, x, *, need_null=True):
        values = [member.get_f_and_B(x, need_null=False) for member in self.members]
        f = torch.stack([value[0] for value in values]).mean(0)
        B = torch.stack([value[1] for value in values]).mean(0)
        return f, B, self._compute_B_null(B) if need_null else None

    def member_parameters(self, x, u, member_ids):
        means = torch.stack([member.predict_x_dot(x, u) for member in self.members])
        stds = torch.stack([member.disturbance_log_std(x, u).exp() for member in self.members])
        rows = torch.arange(len(x), device=x.device)
        return means[member_ids, rows], stds[member_ids, rows]

    def uncertainty(self, x, u):
        """MOPO's max member standard-deviation norm, in derivative coordinates."""
        return torch.stack([member.disturbance_log_std(x, u).exp().norm(dim=-1)
                            for member in self.members]).amax(0)

    def disturbance_log_std(self, x, u):
        means = torch.stack([member.predict_x_dot(x, u) for member in self.members])
        variances = torch.stack([(2 * member.disturbance_log_std(x, u)).exp() for member in self.members])
        return .5 * (variances.mean(0) + means.var(0, unbiased=False)).log()

    def log_prob(self, x_dot, x, u):
        return torch.logsumexp(torch.stack([member.log_prob(x_dot, x, u) for member in self.members]), 0) - math.log(self.ensemble_size)

    @torch.no_grad()
    def sample_disturbance(self, x, u):
        if not bool(self.disturbance_fitted):
            raise RuntimeError("Conditional disturbance ensemble is not fitted")
        ids = torch.randint(self.ensemble_size, (len(x),), device=x.device)
        mean, std = self.member_parameters(x, u, ids)
        return mean + std * torch.randn_like(x) - self.predict_x_dot(x, u)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        for member in result.members:
            member.device = result.device
        return result

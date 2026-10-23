"""Shared MLP, structured policy, and legacy sequence-network building blocks."""
from __future__ import annotations

from contractionRL.agents.utils.system_dynamics.state import embed_angles

import math
import torch
import torch.nn as nn
from contractionRL.agents.utils.contraction.math_utils import rescale_residual
from contractionRL.agents.utils.system_dynamics.reference import Feats, RefWindow, TrackingFeatures

_MIN_LOG_STD = math.log(0.001)

_ACT_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):
    """Standard feedforward MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        gain = nn.init.calculate_gain("tanh")
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(in_d, out_d)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.1)
            layers += [lin, activation]
        self.output_dim = dims[-1]
        if output_dim is not None:
            lin = nn.Linear(dims[-1], output_dim)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.0)
            layers.append(lin)
            self.output_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _zero_last_linear(mlp: MLP) -> None:
    """Zero an MLP's final Linear so its output starts at exactly 0."""
    last = None
    for m in mlp.net:
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class PreviewSequenceEncoder(nn.Module):
    """Encode flattened reference rows with an MLP, GRU or attention layer.

    GRU previews reverse the nearest-first storage so the final hidden state
    sees the nearest reference last. History uses chronological order.
    Stride subsamples rows before encoding; compact GRU context requires stride 1.
    """

    MODES = ("mlp", "gru", "attn")

    def __init__(self, preview_dim, point_dim, hidden, out_dim, mode="gru", stride: int = 1, reverse: bool = True):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"PreviewSequenceEncoder: mode must be one of {self.MODES}, got {mode!r}")
        if preview_dim % point_dim != 0:
            raise ValueError(f"PreviewSequenceEncoder: preview_dim={preview_dim} is not a "
                             f"multiple of point_dim={point_dim}")
        self.num_points = preview_dim // point_dim
        self.point_dim = point_dim
        self.mode = mode
        self.stride = max(1, int(stride))
        self.reverse = reverse
        num_kept = len(range(0, self.num_points, self.stride))
        if mode == "gru":
            self.rnn = nn.GRU(input_size=point_dim, hidden_size=hidden, batch_first=True)
            self.proj = nn.Linear(hidden, out_dim)
        elif mode == "attn":
            self.key = nn.Linear(point_dim, hidden)
            self.value = nn.Linear(point_dim, hidden)
            self.query = nn.Parameter(torch.randn(hidden) * (hidden ** -0.5))
            self.proj = nn.Linear(hidden, out_dim)
        else:  # mlp
            self.mlp = MLP(num_kept * point_dim, [hidden], out_dim, activation=nn.Tanh())


    def train(self, mode: bool = True) -> PreviewSequenceEncoder:
        """Keep dropout-free GRUs in training mode so cuDNN permits backward.

        The training flag does not change their outputs because dropout is disabled.
        """
        super().train(mode)
        if self.mode == "gru":
            self.rnn.train(True)
        return self

    def forward(self, preview_flat: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        n = preview_flat.shape[0]
        points = preview_flat.reshape(n, self.num_points, self.point_dim)
        lengths = None
        if valid is not None:
            if self.mode != 'gru' or self.reverse or self.stride != 1:
                raise ValueError('Internal history padding requires an oldest-first dense GRU')
            lengths = valid.long().sum(-1)
            # History is left-padded. Pack only actual state/action pairs.
            indices = (torch.arange(self.num_points, device=points.device)[None]
                       + self.num_points - lengths[:, None]) % self.num_points
            points = points.gather(1, indices[:, :, None].expand_as(points))
            lengths_cpu = lengths.clamp_min(1).cpu()
        if self.stride > 1:
            points = points[:, ::self.stride, :]  # nearest-first, every stride-th point kept
        if self.mode == "mlp":
            return self.mlp(points.reshape(n, -1))
        if self.mode == "gru":
            seq = points.flip(dims=[1]) if self.reverse else points
            def encode(chunk, chunk_lengths=None):
                if chunk_lengths is not None:
                    from torch.nn.utils.rnn import pack_padded_sequence
                    chunk = pack_padded_sequence(chunk, chunk_lengths, batch_first=True, enforce_sorted=False)
                return self.rnn(chunk)[1]
            chunk_size = min(128, max(1, 25600 // seq.shape[1]))
            if len(seq) > chunk_size:
                from torch.utils.checkpoint import checkpoint
                # Long gamma-derived windows need bounded cuDNN workspaces and activations.
                chunks = seq.split(chunk_size)
                sizes = lengths_cpu.split(chunk_size) if lengths is not None else [None] * len(chunks)
                h = torch.cat([(checkpoint(encode, chunk, size, use_reentrant=False)
                                if torch.is_grad_enabled() else encode(chunk, size))
                               for chunk, size in zip(chunks, sizes)], dim=1)
            else:
                h = encode(seq, lengths_cpu if lengths is not None else None)
            result = self.proj(h.squeeze(0))
            return result if lengths is None else result.masked_fill((lengths == 0)[:, None], 0.)
        k = self.key(points)                                     # (n, P, hidden)
        v = self.value(points)                                   # (n, P, hidden)
        scores = torch.einsum("h,nph->np", self.query, k) / (k.shape[-1] ** 0.5)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)          # (n, P, 1)
        return self.proj((w * v).sum(dim=1))


class TemporalTrackingFeatures(TrackingFeatures):
    """Periodic current/reference, future and history features in relative position."""
    def __init__(self, window, angle_idx=(), *, encoder='gru', hidden=64, stride=1, feature_dim=None):
        TrackingFeatures.__init__(self, window, angle_idx)
        if encoder != 'gru' or stride != 1:
            raise ValueError("GRU context uses dense past and future windows")
        if window.preview_length < 1 or window.history_length != window.preview_length:
            raise ValueError("GRU context requires equally sized past and future windows")
        n = window.x_dim
        self.feature_dim = 3 * n if feature_dim is None else feature_dim
        if not isinstance(self.feature_dim, int) or isinstance(self.feature_dim, bool) or self.feature_dim < 1:
            raise ValueError('feature_dim must be a positive integer or null for 3 * state dimension')
        point_dim = n + len(self.angle_idx)
        self.preview_encoder = PreviewSequenceEncoder(window.preview_length * point_dim, point_dim, hidden, self.feature_dim, mode='gru')
        history_dim = point_dim + window.u_dim
        self.history_encoder = PreviewSequenceEncoder(window.history_length * history_dim, history_dim, hidden, self.feature_dim, mode='gru', reverse=False)
        self.output_dim += 2 * self.feature_dim

    def sequence_features(self, sequence, x):
        points = []
        for i in range(self.window.x_dim):
            value = sequence[:, :, i:i+1]
            if i in self.position:
                points.append(value - x[:, None, i:i+1])
            elif i in self.angle_idx:
                points.extend((value.cos(), value.sin()))
            else:
                points.append(value)
        return torch.cat(points, -1)

    def forward(self, obs):
        x, _, _ = self.window.extract(obs)
        future = self.sequence_features(self.window.preview(obs), x)
        base = TrackingFeatures.forward(self, obs)
        split = 2 * self.state_feature_dim
        parts = [base[:, :split], self.preview_encoder(future.flatten(1))]
        n = self.window.x_dim
        history = self.window.history(obs)
        valid = history[:, :, -1]
        states = self.sequence_features(history[:, :, :n], x)
        controls = history[:, :, n:n+self.window.u_dim]
        parts.append(self.history_encoder(torch.cat((states, controls), -1).flatten(1), valid=valid))
        parts.append(base[:, split:])
        return torch.cat(parts, -1)


class CLActor(nn.Module):
    """Structured controller with u = uref + W2(context) @ tanh(W1(context) @ error).

    Zero tracking error preserves reference feedforward. Compact tracking uses
    periodic relative-position features and optional past/future GRUs; legacy
    reference windows remain supported. Autograd supplies du/dx for contraction.
    """

    def __init__(
        self,
        window: RefWindow,
        feats: Feats,
        hidden_dim: list[int] | None = None,
        activation: nn.Module | str = nn.Tanh(),
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        feature_dim: int | None = None,
    ):
        super().__init__()
        self.window = window
        self.feats = feats
        self.x_dim = window.x_dim
        self.u_dim = window.u_dim

        if isinstance(activation, str):
            activation = _ACT_MAP[activation.lower()]()
        hidden = list(hidden_dim) if hidden_dim else [128, 128]
        self.c = 3 * self.x_dim          # bilinear latent width

        self.compact = bool(getattr(window, "compact", False))
        if self.compact:
            if encoder == 'gru':
                self.tracking_features = TemporalTrackingFeatures(window, feats.angle_idx,
                    encoder=encoder, hidden=encoder_hidden, stride=encoder_stride, feature_dim=feature_dim)
            else:
                if encoder != "mlp" or encoder_stride != 1:
                    raise ValueError("Compact sequence encoders require an explicit reference preview")
                self.tracking_features = TrackingFeatures(window, feats.angle_idx)
        self.feedback_angles = self.tracking_features.angle_idx if self.compact else feats.angle_idx
        self.w1 = MLP(self.tracking_features.output_dim if self.compact else feats.single_dim,
                      hidden, self.c * (self.x_dim + len(self.feedback_angles)), activation=activation)
        # W2 consumes the window as a sequence of `length` points, each
        # `feats.pair_dim` wide. point_dim is what makes gru/attn see points
        # rather than one flat vector.
        if self.compact:
            self.w2 = MLP(self.tracking_features.output_dim, hidden, self.u_dim * self.c, activation=activation)
        else:
            self.w2 = PreviewSequenceEncoder(
                preview_dim=window.length * feats.pair_dim,
                point_dim=feats.pair_dim,
                hidden=encoder_hidden,
                out_dim=self.u_dim * self.c,
                mode=encoder,
                stride=encoder_stride,
            )

        self.logstd = nn.Parameter(torch.zeros(1, self.u_dim))

    def trim_state(self, state: torch.Tensor):
        """``(x, xrefs[0], urefs[0])`` — the current triple, for callers that
        only need the present reference (the analytic controllers, the squashed
        wrappers). The full window stays available via ``window.split``."""
        x, xrefs, urefs = self.window.split(state)
        return x, xrefs[:, 0], urefs[:, 0]

    def _feedback(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw (unbounded) feedback ``W2(xrefs) @ tanh(W1(x) @ e)`` and ``urefs[0]``."""
        x, xrefs, urefs = self.window.split(state)
        n = x.shape[0]
        if self.compact:
            features = self.tracking_features(state)
            e = self.tracking_features.feedback_error(x, xrefs[:, 0]).unsqueeze(-1)
            w1 = self.w1(features).reshape(n, self.c, self.x_dim + len(self.feedback_angles))
            w2 = self.w2(features).reshape(n, self.u_dim, self.c)
        else:
            e = (embed_angles(x, self.feats.angle_idx) - embed_angles(xrefs[:, 0], self.feats.angle_idx)).unsqueeze(-1)
            seq = self.feats.sequence(x, xrefs).reshape(n, -1)
            w1 = self.w1(self.feats.single(x)).reshape(n, self.c, self.x_dim + len(self.feedback_angles))
            w2 = self.w2(seq).reshape(n, self.u_dim, self.c)
        feedback = torch.matmul(w2, torch.tanh(torch.matmul(w1, e))).squeeze(-1)
        return feedback, urefs[:, 0]

    def mean_control(self, state: torch.Tensor) -> torch.Tensor:
        feedback, uref = self._feedback(state)
        return uref + feedback

    def mean_control_squashed(self, state: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Deterministic control law bounded into ``(low, high)`` by construction.

        ``u = urefs[0] + rescale_residual(tanh(feedback), urefs[0], low, high)`` —
        the feedback (not ``uref``) is squashed and ``uref`` added after, so
        ``feedback == 0`` still gives exactly ``u == uref`` (see
        ``math_utils.rescale_residual``). Unlike ``torch.clamp``, ``tanh``'s
        gradient never hits exact zero, so ``K = jacobian(u, x)`` stays
        trainable even where the raw feedback would have saturated — see
        project memory (project_c3m_clip_actions_divergence.md).
        """
        feedback, uref = self._feedback(state)
        action, _ = rescale_residual(torch.tanh(feedback), uref, low, high)
        return action

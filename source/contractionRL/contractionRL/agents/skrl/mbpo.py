"""One-step MBPO using skrl SAC and a bootstrapped probabilistic ensemble.

Algorithm reference: facebookresearch/mbrl-lib, commit
3f93cccfc8d635f74e335a2f07aab6e9a48fc021. See docs/mbpo.md for adaptations.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import compute_space_size

from contractionRL.agents.utils.system_dynamics.reference import RefWindow
from contractionRL.agents.utils.system_dynamics.state import wrap_diff


@dataclass(kw_only=True)
class MBPOCfg(SAC_CFG):
    ensemble_size: int = 7
    num_elites: int = 5
    model_hidden: list[int] = field(default_factory=lambda: [200, 200, 200, 200])
    model_learning_rate: float = 0.001
    model_weight_decay: float = 0.00001
    model_batch_size: int = 256
    model_train_frequency: int = 250
    model_train_epochs: int = 50
    model_patience: int = 5
    model_holdout_ratio: float = 0.2
    model_min_samples: int = 1000
    model_max_samples: int = 100000
    model_rollout_batch_size: int = 10000
    model_buffer_size: int = 100000
    real_data_ratio: float = 0.05


class _MixedReplay(RandomMemory):
    """Store only real transitions; select a real/model batch for SAC updates."""
    model_memory = None
    real_data_ratio = 1.0

    def sample_real(self, names, *, batch_size):
        return super().sample(names=names, batch_size=batch_size)[0]

    def sample(self, names, *, batch_size, mini_batches=1, sequence_length=1):
        if self.model_memory is not None and len(self.model_memory) and torch.rand(()).item() >= self.real_data_ratio:
            return self.model_memory.sample(names=names, batch_size=batch_size,
                                            mini_batches=mini_batches, sequence_length=sequence_length)
        return super().sample(names=names, batch_size=batch_size,
                              mini_batches=mini_batches, sequence_length=sequence_length)


class _ModelReplay(RandomMemory):
    def add_samples(self, **tensors):
        # skrl 2.1's single-env batch path advances its cursor per column.
        # Synthetic batches need one shared cursor for the entire transition.
        count = len(tensors["observations"])
        if count > self.memory_size or any(len(value) != count for value in tensors.values()):
            raise ValueError("Inconsistent or oversized model replay batch")
        indices = (torch.arange(count, device=self.device) + self.memory_index) % self.memory_size
        for name, value in tensors.items():
            self.tensors_view[name].index_copy_(0, indices, value)
        self.filled |= self.memory_index + count >= self.memory_size
        self.memory_index = (self.memory_index + count) % self.memory_size


class _GaussianMember(nn.Module):
    def __init__(self, input_dim, target_dim, hidden):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, 2 * target_dim + 1))
        self.net = nn.Sequential(*layers)
        self.max_logvar = nn.Parameter(torch.full((target_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((target_dim,), -10.0))
        self.target_dim = target_dim

    def forward(self, inputs):
        output = self.net(inputs)
        mean, logvar = output[..., :self.target_dim], output[..., self.target_dim:-1]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar, output[..., -1:]


class MBPOEnsemble(nn.Module):
    """Normalized delta-state/reward Gaussians and a learned terminal head."""
    def __init__(self, input_dim, target_dim, cfg):
        super().__init__()
        self.members = nn.ModuleList([_GaussianMember(input_dim, target_dim, cfg.model_hidden)
                                      for _ in range(cfg.ensemble_size)])
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))
        self.register_buffer("elites", torch.arange(cfg.num_elites))
        self.register_buffer("fits", torch.zeros((), dtype=torch.long))

    def forward(self, inputs):
        inputs = (inputs - self.input_mean) / self.input_std
        return tuple(torch.stack(values) for values in zip(*(member(inputs) for member in self.members)))

    @torch.no_grad()
    def sample(self, inputs):
        mean, logvar, terminal_logits = self(inputs)
        member = self.elites[torch.randint(len(self.elites), (len(inputs),), device=inputs.device)]
        rows = torch.arange(len(inputs), device=inputs.device)
        values = mean[member, rows] + (0.5 * logvar[member, rows]).exp() * torch.randn_like(mean[member, rows])
        values = values * self.target_std + self.target_mean
        terminated = torch.rand(len(inputs), 1, device=inputs.device) < terminal_logits[member, rows].sigmoid()
        return values, terminated


class _EntropyState:
    """skrl saves the entropy optimizer but not its optimized scalar."""
    def __init__(self, agent):
        self.agent = agent

    def state_dict(self):
        return {"log_entropy_coefficient": self.agent.log_entropy_coefficient.detach().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.agent.log_entropy_coefficient.copy_(state["log_entropy_coefficient"])
        self.agent._entropy_coefficient = self.agent.log_entropy_coefficient.detach().exp()


class MBPOAgent(SAC):
    """SAC's optimizer is unchanged; its batches may come from a learned model."""
    def __init__(self, *, cfg, memory, observation_space, state_space=None, action_space, device=None, **kwargs):
        cfg = MBPOCfg(**cfg) if isinstance(cfg, dict) else cfg
        positive = ("ensemble_size", "num_elites", "model_batch_size", "model_train_frequency",
                    "model_train_epochs", "model_patience", "model_min_samples", "model_max_samples",
                    "model_rollout_batch_size", "model_buffer_size")
        if any(getattr(cfg, name) < 1 for name in positive):
            raise ValueError("MBPO model counts, capacities and intervals must be positive")
        if not 1 <= cfg.num_elites <= cfg.ensemble_size:
            raise ValueError("num_elites must be between 1 and ensemble_size")
        if not 0 < cfg.model_holdout_ratio < 1 or not 0 <= cfg.real_data_ratio <= 1:
            raise ValueError("Invalid model_holdout_ratio or real_data_ratio")
        if cfg.model_min_samples < 2 or cfg.model_max_samples < cfg.model_min_samples:
            raise ValueError("Model fitting requires at least two samples and a sufficient model_max_samples")
        if not cfg.model_hidden or any(width < 1 for width in cfg.model_hidden):
            raise ValueError("model_hidden must contain positive layer widths")
        if not math.isfinite(cfg.model_learning_rate) or cfg.model_learning_rate <= 0 or not math.isfinite(cfg.model_weight_decay) or cfg.model_weight_decay < 0:
            raise ValueError("Invalid model learning rate or weight decay")
        if cfg.rewards_shaper is not None:
            raise ValueError("MBPO uses the environment reward; reward shaping/normalization is unsupported")
        if state_space is not None:
            raise ValueError("MBPO currently requires actor/critic observations to coincide (no privileged state_space)")
        if not isinstance(action_space, gym.spaces.Box) or not torch.isfinite(torch.as_tensor(action_space.low)).all() or not torch.isfinite(torch.as_tensor(action_space.high)).all():
            raise ValueError("MBPO requires finite continuous action bounds")
        if memory is None or memory.memory_size < cfg.model_min_samples:
            raise ValueError("MBPO requires a replay capacity at least model_min_samples")
        # MBPO's YAML memory_size counts total transitions, independent of num_envs.
        real_memory = _MixedReplay(memory_size=math.ceil(memory.memory_size / memory.num_envs),
                                   num_envs=memory.num_envs, device=device, replacement=True,
                                   export=memory.export, export_format=memory.export_format,
                                   export_directory=memory.export_directory)
        super().__init__(cfg=cfg, memory=real_memory, observation_space=observation_space,
                         state_space=state_space, action_space=action_space, device=device, **kwargs)
        self.model_memory = _ModelReplay(memory_size=cfg.model_buffer_size, num_envs=1,
                                         device=self.device, replacement=True)
        self.memory.model_memory = self.model_memory
        self.memory.real_data_ratio = cfg.real_data_ratio
        self.window = None
        if isinstance(observation_space, gym.spaces.Dict) and (hasattr(observation_space, "reference_state") or {"x", "xrefs", "urefs"} <= set(observation_space.spaces)):
            self.window = RefWindow.from_space(observation_space)
        obs_dim = compute_space_size(observation_space)
        start = 0 if self.window is None or getattr(self.window, "compact", False) else self.window.length * self.window.u_dim
        count = obs_dim if self.window is None else self.window.x_dim
        self.physical_slice = slice(start, start + count)
        self.dynamics = MBPOEnsemble(obs_dim + compute_space_size(action_space), count + 1, cfg).to(self.device)
        self.model_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=cfg.model_learning_rate,
                                                  weight_decay=cfg.model_weight_decay)
        self.checkpoint_modules.update(dynamics=self.dynamics, model_optimizer=self.model_optimizer)
        if cfg.learn_entropy:
            self.checkpoint_modules["mbpo_entropy"] = _EntropyState(self)
        self.angle_idx = []
        self.physical_bounds = None
        self.pos_dimension = 0

    def bind_env(self, env):
        from contractionRL.runners.contraction_runner import _env_attrs, _unwrap_env
        raw = _unwrap_env(env)
        (angles,) = _env_attrs(raw, "angle_idx", default=())
        self.angle_idx = list(angles or []) if self.window is not None else []
        low, high, positions = _env_attrs(raw, "X_MIN", "X_MAX", "pos_dimension")
        if getattr(self.window, "compact", False) and low is not None and high is not None and positions is not None:
            self.physical_bounds = (torch.as_tensor(low, device=self.device), torch.as_tensor(high, device=self.device))
            self.pos_dimension = int(positions)
        from contractionRL.agents.utils.RL.agent_patches import patch_terminal_observations
        patch_terminal_observations(self)

    def init(self, *, trainer_cfg=None):
        super().init(trainer_cfg=trainer_cfg)
        for name in self.memory.get_tensor_names():
            value = self.memory.get_tensor_by_name(name)
            self.model_memory.create_tensor(name=name, size=value.shape[-1], dtype=value.dtype)

    def _dataset(self):
        size = len(self.memory)
        # Unique train/holdout rows; never sample synthetic transitions to fit the model.
        indices = torch.randperm(size, device=self.device)[:self.cfg.model_max_samples]
        obs, actions, rewards, next_obs, terminated = self.memory.sample_by_index(
            names=["observations", "actions", "rewards", "next_observations", "terminated"], indexes=indices)[0]
        delta = wrap_diff(next_obs[:, self.physical_slice] - obs[:, self.physical_slice], self.angle_idx)
        return torch.cat((obs, actions), -1), torch.cat((delta, rewards), -1), terminated.float()

    def fit_model(self):
        inputs, targets, terminated = self._dataset()
        split = max(1, min(len(inputs) - 1, int(len(inputs) * self.cfg.model_holdout_ratio)))
        val_x, train_x = inputs[:split], inputs[split:]
        val_y, train_y = targets[:split], targets[split:]
        val_t, train_t = terminated[:split], terminated[split:]
        with torch.no_grad():
            for data, mean, std in ((train_x, self.dynamics.input_mean, self.dynamics.input_std),
                                    (train_y, self.dynamics.target_mean, self.dynamics.target_std)):
                mean.copy_(data.mean(0))
                deviation = data.std(0, unbiased=False)
                std.copy_(torch.where(deviation < 1e-6, torch.ones_like(deviation), deviation))
        train_y = (train_y - self.dynamics.target_mean) / self.dynamics.target_std
        val_y = (val_y - self.dynamics.target_mean) / self.dynamics.target_std
        best_scores = torch.full((self.cfg.ensemble_size,), float("inf"), device=self.device)
        best_weights = [copy.deepcopy(member.state_dict()) for member in self.dynamics.members]
        stale = 0
        self.dynamics.train()
        for _ in range(self.cfg.model_train_epochs):
            bootstrap = torch.randint(len(train_x), (self.cfg.ensemble_size, len(train_x)), device=self.device)
            for start in range(0, len(train_x), self.cfg.model_batch_size):
                losses = []
                for i, member in enumerate(self.dynamics.members):
                    ids = bootstrap[i, start:start + self.cfg.model_batch_size]
                    mean, logvar, logits = member((train_x[ids] - self.dynamics.input_mean) / self.dynamics.input_std)
                    nll = ((mean - train_y[ids]).square() * (-logvar).exp() + logvar).mean()
                    losses.append(nll + F.binary_cross_entropy_with_logits(logits, train_t[ids])
                                  + .01 * (member.max_logvar.sum() - member.min_logvar.sum()))
                loss = torch.stack(losses).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("MBPO dynamics loss became non-finite")
                self.model_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.)
                self.model_optimizer.step()
            with torch.no_grad():
                scores = torch.zeros_like(best_scores)
                for start in range(0, len(val_x), self.cfg.model_batch_size):
                    stop = start + self.cfg.model_batch_size
                    mean, _, logits = self.dynamics(val_x[start:stop])
                    mse = (mean - val_y[start:stop]).square().mean((1, 2))
                    bce = F.binary_cross_entropy_with_logits(logits, val_t[start:stop].expand_as(logits), reduction="none").mean((1, 2))
                    scores += (mse + bce) * len(val_x[start:stop]) / len(val_x)
                if not torch.isfinite(scores).all():
                    raise RuntimeError("MBPO holdout scores became non-finite")
                improved = scores < best_scores * .99
                for i in improved.nonzero().flatten().tolist():
                    best_scores[i] = scores[i]
                    best_weights[i] = copy.deepcopy(self.dynamics.members[i].state_dict())
                stale = 0 if improved.any() else stale + 1
            if stale >= self.cfg.model_patience:
                break
        for member, weights in zip(self.dynamics.members, best_weights):
            member.load_state_dict(weights)
        self.dynamics.elites.copy_(best_scores.argsort()[:self.cfg.num_elites])
        self.dynamics.fits.add_(1)
        self.dynamics.eval()
        self.track_data("Loss / Dynamics loss", loss.item() / self.cfg.ensemble_size)
        self.track_data("Loss / Dynamics holdout mean", best_scores.mean().item())
        self.track_data("Loss / Dynamics holdout stddev", best_scores.std(unbiased=False).item())

    @torch.no_grad()
    def rollout_model(self):
        generated = 0
        # Fixed one-step rollouts: reference/xi endpoint comes from the real
        # branch, so no plant-model noise ever changes the nominal reference.
        chunk_size = min(self.cfg.model_batch_size, self.cfg.model_buffer_size)
        for start in range(0, self.cfg.model_rollout_batch_size, chunk_size):
            size = min(chunk_size, self.cfg.model_rollout_batch_size - start)
            obs, reference_next = self.memory.sample_real(["observations", "next_observations"], batch_size=size)
            actions, _ = self.policy.act({"observations": self._observation_preprocessor(obs), "states": None}, role="policy")
            predicted, terminated = self.dynamics.sample(torch.cat((obs, actions), -1))
            next_obs = reference_next.clone()
            next_obs[:, self.physical_slice] = wrap_diff(obs[:, self.physical_slice] + predicted[:, :-1], self.angle_idx)
            if self.physical_bounds is not None:
                low, high = self.physical_bounds
                physical = next_obs[:, self.physical_slice]
                p = self.pos_dimension
                outside = ((physical[:, :p] < low[:p]) | (physical[:, :p] > high[:p])).any(-1)
                terminated = terminated | outside[:, None]
                physical[:, p:] = physical[:, p:].clamp(low[p:], high[p:])
            if getattr(self.window, "history_length", 0):
                history = self.window.history(obs)
                applied = actions.clamp(torch.as_tensor(self.action_space.low, device=self.device),
                                        torch.as_tensor(self.action_space.high, device=self.device))
                row = torch.cat((obs[:, self.physical_slice], applied, obs.new_ones(len(obs), 1)), -1)
                self.window.history(next_obs).copy_(torch.cat((history[:, 1:], row[:, None]), 1))
            rewards = predicted[:, -1:]
            if getattr(self.window, "compact", False):
                x, xr, _ = self.window.extract(obs)
                nx, nxr, _ = self.window.extract(next_obs)
                rewards = (wrap_diff(x - xr, self.angle_idx).square().sum(-1)
                           - wrap_diff(nx - nxr, self.angle_idx).square().sum(-1)).unsqueeze(-1)
            valid = torch.isfinite(next_obs).all(-1) & torch.isfinite(rewards).all(-1) & torch.isfinite(actions).all(-1)
            if valid.any():
                self.model_memory.add_samples(observations=obs[valid], actions=actions[valid],
                    rewards=rewards[valid], next_observations=next_obs[valid], terminated=terminated[valid])
                generated += int(valid.sum())
        if not generated:
            raise RuntimeError("MBPO produced no finite model transitions")
        self.track_data("Learning / Model transitions generated", generated)
        self.track_data("Learning / Model replay size", len(self.model_memory))

    def update(self, *, timestep, timesteps):
        if len(self.memory) < self.cfg.model_min_samples:
            return
        if not self.dynamics.fits.item() or (timestep + 1) % self.cfg.model_train_frequency == 0:
            self.fit_model()
            self.rollout_model()
        super().update(timestep=timestep, timesteps=timesteps)

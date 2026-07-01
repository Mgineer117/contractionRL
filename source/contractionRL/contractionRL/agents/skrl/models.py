"""skrl-compatible model wrappers for contractionRL custom actors.

CLActorModel: wraps CLActor (C3M_U contracting controller) in the skrl
GaussianMixin interface so it can be passed to skrl PPO / TEMP runners.

Observation layout assumed: [x (x_dim), xref (x_dim), uref (u_dim)]
  → obs_dim = 2*x_dim + u_dim  →  x_dim = (obs_dim - u_dim) / 2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
except ImportError:
    raise ImportError("skrl is required. Install it or use the local developer copy.")

from .nn_modules import CCM_Generator, CLActor

_MIN_LOG_STD = math.log(0.01)  # ≈ -4.605; matches CLActor annealing floor


class CLDeterministicActorModel(DeterministicMixin, Model):
    """Contracting C3M_U actor wrapped as a skrl Deterministic policy model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        if x_dim is None:
            x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            mode="deterministic",
            anneal_stddev=False,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
        )

        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        return self.cl_actor.mean_control(state), {}


class CMGModel(Model):
    """CCM_Generator wrapped as a skrl Model for checkpointing.

    The underlying CCM_Generator is accessed via ``self.ccm_gen`` by C3MAgent
    and TEMPAgent for the contraction loss computation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mode: str = "deterministic",
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        self.obs_dim = int(observation_space.shape[0])
        self.u_dim = int(action_space.shape[0])
        if x_dim is None:
            x_dim = (self.obs_dim - self.u_dim) // 2
        self.x_dim = x_dim

        self.ccm_gen = CCM_Generator(
            x_dim=x_dim,
            hidden_dim=hidden_dim or [256, 256],
            activation=activation,
            mode=mode,
            device=str(device) if not isinstance(device, str) else device,
        )

    def compute(self, inputs: dict, role: str = "cmg"):
        x = inputs["observations"][:, : self.x_dim]
        W, info = self.ccm_gen(x)
        return W.reshape(W.shape[0], -1), {}

    def act(self, inputs: dict, role: str = "cmg"):
        output, extra = self.compute(inputs, role)
        return output, None, extra

    def forward(self, inputs: dict, role: str = "cmg"):
        output, _ = self.compute(inputs, role)
        return output


class CLActorModel(GaussianMixin, Model):
    """Contracting C3M_U actor wrapped as a skrl Gaussian policy model.

    Args:
        observation_space: gymnasium observation space ([x, xref, uref]).
        action_space:      gymnasium action space (u).
        device:            torch device.
        clip_actions:      whether to clip sampled actions to action space bounds.
        clip_log_std:      whether to clip log_std to [min_log_std, max_log_std].
        min_log_std:       lower clip bound for log_std.
        max_log_std:       upper clip bound for log_std.
        initial_log_std:   initial value for the global log_std parameter.
        hidden_dim:        hidden layer sizes for the CLActor weight generators.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = _MIN_LOG_STD,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = int(self.observation_space.shape[0])
        u_dim = int(self.action_space.shape[0])
        x_dim = (obs_dim - u_dim) // 2

        self.cl_actor = CLActor(
            x_dim=x_dim,
            u_dim=u_dim,
            mode="stochastic",
            anneal_stddev=True,
            hidden_dim=hidden_dim or [128, 128],
            activation=activation,
        )
        self.log_std_parameter = self.cl_actor.logstd

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)

        # Model.__init__ moves to self.device before cl_actor exists; re-sync here.
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        mean = self.cl_actor.mean_control(state)
        return mean, {"log_std": self.log_std_parameter}

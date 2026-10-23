"""Tensor-preserving environment wrapper and value preprocessing."""
from __future__ import annotations

from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.resources.preprocessors.torch import RunningStandardScaler as _RunningStandardScaler
import gymnasium
import torch


class RunningStandardScaler(_RunningStandardScaler):
    """skrl's scaler with population variance, including one-sample updates."""

    def _compute(self, x, *, train=False, inverse=False):
        if train and x.numel():
            if x.ndim not in (2, 3):
                raise ValueError("RunningStandardScaler expects (batch, features) or (time, batch, features)")
            dims = tuple(range(x.ndim - 1))
            values = x.double()
            self._parallel_variance(values.mean(dim=dims), values.var(dim=dims, correction=0),
                                    x.numel() // x.shape[-1])
        return super()._compute(x, train=False, inverse=inverse)


class BatchedGymnasiumWrapper(GymnasiumWrapper):
    """Overrides SKRL's default GymnasiumWrapper to prevent tensor-copy warnings.

    Our classic environments natively output PyTorch tensors for speed. SKRL's default
    wrapper forces torch.tensor() on the outputs, which throws a UserWarning in PyTorch
    when the output is already a tensor. This wrapper safely applies torch.as_tensor
    instead, fully implementing PyTorch's recommendation without modifying SKRL's library.
    """
    def step(self, actions: torch.Tensor):
        from skrl.utils.spaces.torch import (
            flatten_tensorized_space,
            tensorize_space,
            unflatten_tensorized_space,
            untensorize_space,
        )

        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to torch using .clone().detach() or as_tensor (implementing the PyTorch recommendation)
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))

        # Here we fix the SKRL warning by checking if it's already a tensor!
        if torch.is_tensor(reward):
            reward = reward.clone().detach().to(self.device).view(self.num_envs, -1)
            terminated = terminated.clone().detach().to(self.device).view(self.num_envs, -1)
            truncated = truncated.clone().detach().to(self.device).view(self.num_envs, -1)
        else:
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
            terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

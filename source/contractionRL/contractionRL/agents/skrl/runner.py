"""CLActorRunner — backbone-aware GaussianMixin factory.

YAML policy config:

    policy:
      class: GaussianMixin
      backbone: contraction       # CLActor (W2 @ tanh(W1 @ (x-x_ref)))
      clip_log_std: True
      min_log_std: -4.605
      max_log_std: 2.0
      network:                    # layers used for w1 and w2 MLPs
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: mlp               # standard skrl MLP (default)
      network:
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

When backbone is ``contraction``, ``layers`` is extracted from the first
network entry and passed as ``hidden_dim`` to CLActorModel.  All other
MLP-specific keys (``output``, ``initial_log_std``) are dropped.
"""
from __future__ import annotations

from skrl.utils.runner.torch import Runner


def _gaussian_factory(observation_space, state_space, action_space, device,
                       backbone: str = "mlp", **kwargs):
    if backbone == "contraction":
        import gymnasium
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # CLActorModel requires obs layout [x, x_ref, u_ref]: obs_dim == 2*x_dim + u_dim
        # and x_dim must be an integer
        remainder = obs_dim - act_dim
        if remainder <= 0 or remainder % 2 != 0:
            raise ValueError(
                f"backbone: contraction requires obs_dim = 2*x_dim + u_dim, but got "
                f"obs_dim={obs_dim}, act_dim={act_dim} (remainder={remainder} is not even). "
                f"Use backbone: mlp for velocity-tracking environments."
            )
        from contractionRL.agents.skrl.models import CLActorModel
        network = kwargs.pop("network", [{}])
        hidden_dim = network[0].get("layers", [128, 128]) if network else [128, 128]
        kwargs.pop("output", None)
        kwargs.pop("initial_log_std", None)
        
        # Try to extract x_dim if available
        x_dim = kwargs.pop("x_dim", None)
        
        return CLActorModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            **kwargs,
        )
    from skrl.utils.model_instantiators.torch import gaussian_model
    return gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )


class CLActorRunner(Runner):
    def _component(self, name: str):
        if name.lower() == "gaussianmixin":
            return _gaussian_factory
        return super()._component(name)

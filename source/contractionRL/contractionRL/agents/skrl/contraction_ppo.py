# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""ContractionPPO released-code baseline, ported to this repo's skrl PPO.

Source: contractionppo/ContractionPPO, e63ed83516e46a2630ad203115d069d83de98f4b.
The released code fits the metric separately and adds a detached rollout reward;
it does not backpropagate contraction loss into the actor. See docs/contractionppo.md.

Derived from BSD-3-Clause source, Copyright (c) 2021-2025 ETH Zurich and
NVIDIA CORPORATION; original files/notices are in third_party/contractionppo.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

from contractionRL.agents.utils.system_dynamics.state import wrap_diff
from contractionRL.agents.utils.system_dynamics.reference import RefWindow

UPSTREAM_COMMIT = 'e63ed83516e46a2630ad203115d069d83de98f4b'


def _activation(name):
    return {'elu': nn.ELU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}[name]()


class SpectralNormBoundedLinear(nn.Module):
    def __init__(self, linear, bound):
        super().__init__()
        self.linear = nn.utils.spectral_norm(linear)
        self.bound = float(bound)

    def forward(self, x):
        out = self.linear(x)
        if self.linear.bias is None:
            return out*self.bound
        return out*self.bound + (1-self.bound)*self.linear.bias


class ContractionMetricNetwork(nn.Module):
    """Released metric: input c, softplus Cholesky diagonal + .1, M=Theta Theta'."""
    def __init__(self, input_dim, constraint_dim, hidden_dims=(128, 64), activation='elu',
                 min_diagonal_value=.1, spectral_norm_bound=3.):
        super().__init__()
        self.constraint_dim = constraint_dim
        self.min_diagonal_value = min_diagonal_value
        layers, width = [], input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), _activation(activation)]
            width = hidden
        layers.append(nn.Linear(width, constraint_dim*(constraint_dim+1)//2))
        self.mlp = nn.Sequential(*layers)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
        for index, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                self.mlp[index] = SpectralNormBoundedLinear(layer, spectral_norm_bound)

    def forward(self, c):
        vector = self.mlp(c)
        rows, cols = torch.tril_indices(self.constraint_dim, self.constraint_dim, device=c.device)
        vector = torch.where(rows == cols, F.softplus(vector)+self.min_diagonal_value, vector)
        theta = c.new_zeros(len(c), self.constraint_dim, self.constraint_dim)
        theta[:, rows, cols] = vector
        return theta @ theta.mT


def contraction_terms(metric, error, error_dot, *, alpha, epsilon, scale):
    """Released unnormalized hinge and shaping reward; no in-place data scaling."""
    with torch.enable_grad():
        c = (error.detach()*scale).requires_grad_(True)
        V = torch.einsum('bi,bij,bj->b', c, metric(c), c)
        # The release makes a second forward pass for the derivative. Keep that
        # behavior, including spectral normalization's power-iteration buffers.
        c_derivative = (error.detach()*scale).requires_grad_(True)
        V_derivative = torch.einsum('bi,bij,bj->b', c_derivative, metric(c_derivative), c_derivative)
        gradient = torch.autograd.grad(V_derivative.sum(), c_derivative, create_graph=True)[0]
        V_dot = (gradient*(error_dot.detach()*scale)).sum(-1)
        penalty = F.relu(V_dot+alpha*V+epsilon)
    return V, V_dot, penalty


@dataclass(kw_only=True)
class ContractionPPOCfg(PPO_CFG):
    alpha: float = .1
    penalty_coef: float = .005
    contraction_constraint_coef: float = 100.
    contraction_metric_learning_rate: float = 1e-3
    contraction_hidden_dims: tuple[int, ...] = (128, 64)
    contraction_activation: str = 'elu'
    epsilon_contraction: float = 1e-3
    spectral_norm_bound: float = 3.
    min_diagonal_value: float = .1
    contraction_indices: tuple[int, ...] | None = None


def _plain_mlp(space, output_dim, hidden_dims, name, device):
    from gymnasium.spaces import flatten_space
    width = space if isinstance(space, int) else flatten_space(space).shape[0]
    layers = []
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), _activation(name)]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers).to(device)


class ContractionPPOPolicy(GaussianMixin, Model):
    """Shared structured tracking actor; explicit full mode reloads old baselines."""
    def __init__(self, observation_space, action_space, device, network=None,
                 initial_log_std=0., control_mode='structured', **kwargs):
        if control_mode not in {'full', 'structured'}:
            raise ValueError('ContractionPPO supports full or structured control')
        self.control_mode = control_mode
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=False)
        spec = (network or [{}])[0]
        if control_mode == 'structured':
            from contractionRL.agents.utils.RL.models import _build_cl_actor
            self.cl_actor = _build_cl_actor(observation_space, action_space, None,
                                            kwargs.get('angle_idx'), kwargs.get('sym'),
                                            spec.get('layers',[512,256,128]), spec.get('activations','elu'),
                                            'mlp', 64, 1)
            self.cl_actor.logstd.requires_grad_(False)
        else:
            input_size = observation_space
            if hasattr(observation_space, 'reference_state'):
                from contractionRL.agents.utils.system_dynamics.reference import TrackingFeatures
                self.tracking_features = TrackingFeatures(observation_space.reference_state, kwargs.get('angle_idx', ()))
                self.register_buffer('_uses_tracking_features', torch.tensor(True))
                input_size = self.tracking_features.output_dim
            self.net = _plain_mlp(input_size, action_space.shape[0], spec.get('layers',[512,256,128]), spec.get('activations','elu'), device)
        self.std = nn.Parameter(torch.full((action_space.shape[0],),math.exp(float(initial_log_std)),device=device))
        self.to(device)

    def compute(self, inputs, role='policy'):
        if self.control_mode == 'structured':
            return self.cl_actor.mean_control(inputs['observations']), {'log_std': self.std.log()}
        obs = inputs['observations']
        if hasattr(self, 'tracking_features') and bool(self._uses_tracking_features):
            obs = self.tracking_features(obs)
        return self.net(obs), {'log_std':self.std.log()}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if hasattr(self, 'tracking_features') and '_uses_tracking_features' not in state_dict:
            if self.tracking_features.position:
                raise ValueError("Load historical policies with their archived source snapshot")
            state_dict = dict(state_dict)
            state_dict['_uses_tracking_features'] = self._uses_tracking_features.new_tensor(False)
            state_dict.update({'tracking_features.' + key: value for key, value in self.tracking_features.state_dict().items()})
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class ContractionPPOValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, network=None, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        spec = (network or [{}])[0]
        input_size = observation_space
        if hasattr(observation_space, 'reference_state'):
            from contractionRL.agents.utils.system_dynamics.reference import TrackingFeatures
            self.tracking_features = TrackingFeatures(observation_space.reference_state, kwargs.get('angle_idx', ())).to(device)
            input_size = self.tracking_features.output_dim
        self.net = _plain_mlp(input_size,1,spec.get('layers',[512,256,128]),spec.get('activations','elu'),device)

    def compute(self, inputs, role='value'):
        obs = inputs['observations']
        if hasattr(self, 'tracking_features'):
            obs = self.tracking_features(obs)
        return self.net(obs), {}


class ContractionPPOAgent(PPO):
    def __init__(self, *, cfg=None, **kwargs):
        cfg = ContractionPPOCfg(**(cfg or {})) if isinstance(cfg, (dict,type(None))) else cfg
        if not all(math.isfinite(value) for value in (cfg.alpha,cfg.penalty_coef,
                cfg.contraction_constraint_coef,cfg.contraction_metric_learning_rate,
                cfg.epsilon_contraction,cfg.min_diagonal_value,cfg.spectral_norm_bound)):
            raise ValueError('ContractionPPO coefficients must be finite')
        if (cfg.alpha <= 0 or cfg.penalty_coef < 0 or cfg.contraction_constraint_coef <= 0
                or cfg.contraction_metric_learning_rate <= 0 or cfg.epsilon_contraction < 0
                or cfg.min_diagonal_value <= 0 or cfg.spectral_norm_bound <= 0):
            raise ValueError('Invalid ContractionPPO configuration')
        super().__init__(cfg=cfg, **kwargs)
        self.layout = RefWindow.from_space(self.observation_space)
        self.contraction_metric = None

    def bind_env(self, env):
        """Use privileged transition errors, with dimensions inferred from the plant."""
        def find(*names):
            pending, seen = [env], set()
            while pending:
                target = pending.pop()
                if id(target) in seen: continue
                seen.add(id(target))
                for name in names:
                    value = getattr(target,name,None)
                    if value is not None: return value
                for name in ('unwrapped','env','plant'):
                    value = getattr(target,name,None)
                    if value is not None: pending.append(value)
            return None
        self.dt = find('step_dt','dt')
        self.angle_idx = list(find('angle_idx') or [])
        position_dim = find('pos_dimension')
        if self.dt is None or self.dt <= 0 or position_dim is None:
            raise ValueError('ContractionPPO requires a tracking environment with dt and pos_dimension')
        self.indices = list(range(int(position_dim)) if self.cfg.contraction_indices is None else self.cfg.contraction_indices)
        if not self.indices or len(set(self.indices)) != len(self.indices) or min(self.indices)<0 or max(self.indices)>=self.layout.x_dim:
            raise ValueError('Invalid contraction_indices')
        self.contraction_metric = ContractionMetricNetwork(len(self.indices),len(self.indices),
            hidden_dims=self.cfg.contraction_hidden_dims,activation=self.cfg.contraction_activation,
            min_diagonal_value=self.cfg.min_diagonal_value,spectral_norm_bound=self.cfg.spectral_norm_bound).to(self.device)
        self.metric_optimizer = torch.optim.Adam(self.contraction_metric.parameters(),lr=self.cfg.contraction_metric_learning_rate)
        self.checkpoint_modules.update(contraction_metric=self.contraction_metric,metric_optimizer=self.metric_optimizer)

    def init(self, *, trainer_cfg=None):
        if self.contraction_metric is None:
            raise RuntimeError('ContractionPPO must bind its tracking environment before initialization')
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            for name in ('contraction_error','contraction_error_dot'):
                self.memory.create_tensor(name=name,size=len(self.indices),dtype=torch.float32)
            self.memory.create_tensor(name='contraction_valid',size=1,dtype=torch.bool)
        if getattr(self,'experiment_dir',None):
            Path(self.experiment_dir).mkdir(parents=True,exist_ok=True)
            Path(self.experiment_dir,'contractionppo_provenance.json').write_text(json.dumps({
                'upstream_commit':UPSTREAM_COMMIT,'variant':'released_code_skrl_port',
                'constraint_indices':self.indices,'constraint_scale':self.cfg.contraction_constraint_coef,
                'derivative':'measured_relative_error_forward_difference','dt':float(self.dt),
                'policy_contraction_gradient':False,'two_phase_schedule':False,
                'inplace_derivative_scaling':False,
                'metric_derivative_extra_scale':self.cfg.contraction_constraint_coef},indent=2))

    def _terms(self, error, derivative):
        return contraction_terms(self.contraction_metric,error,derivative,alpha=self.cfg.alpha,
            epsilon=self.cfg.epsilon_contraction,scale=self.cfg.contraction_constraint_coef)

    def record_transition(self, *, observations, rewards, next_observations, terminated, truncated, infos, **kwargs):
        if not self.training:
            return super().record_transition(observations=observations,rewards=rewards,
                next_observations=next_observations,terminated=terminated,truncated=truncated,infos=infos,**kwargs)
        x, refs, _ = self.layout.split(observations)
        error = wrap_diff(x-refs[:,0],self.angle_idx)
        endpoint = infos.get('tracking_error_vector') if isinstance(infos,dict) else None
        if endpoint is None:
            raise ValueError('ContractionPPO requires pre-autoreset tracking_error_vector in transition info')
        endpoint = torch.as_tensor(endpoint,device=self.device)
        derivative = wrap_diff(endpoint-error,self.angle_idx)/self.dt
        error, derivative = error[:,self.indices].detach(), derivative[:,self.indices].detach()
        valid = torch.isfinite(error).all(-1) & torch.isfinite(derivative).all(-1)
        intrinsic = torch.zeros(len(error),device=self.device)
        if valid.any():
            _, _, penalty = self._terms(error[valid],derivative[valid])
            intrinsic[valid] = (self.cfg.epsilon_contraction-penalty.detach())*self.cfg.penalty_coef
        # Write auxiliary fields at the same slot as PPO's single add_samples call.
        index = self.memory.memory_index
        for name, value in [('contraction_error',error),('contraction_error_dot',derivative),('contraction_valid',valid[:,None])]:
            self.memory.get_tensor_by_name(name)[index].copy_(value)
        self.latest_contraction_rewards = intrinsic.detach().clone()
        shaped = rewards + intrinsic.reshape_as(rewards)
        self.track_data('Learning / ContractionPPO/contraction_reward_mean',intrinsic.mean().item())
        self.track_data('Learning / ContractionPPO/contraction_reward_stddev',intrinsic.std(unbiased=False).item())
        super().record_transition(observations=observations,rewards=shaped.detach(),
            next_observations=next_observations,terminated=terminated,truncated=truncated,infos=infos,**kwargs)

    def update(self, *, timestep, timesteps):
        # Metric and PPO objectives have disjoint parameters. Fit the metric
        # separately; PPO uses the detached shaping rewards already in its buffer.
        losses = []
        names=['contraction_error','contraction_error_dot','contraction_valid']
        for _ in range(self.cfg.learning_epochs):
            for error, derivative, valid in self.memory.sample(names=names,batch_size=len(self.memory),mini_batches=self.cfg.mini_batches):
                mask = valid.reshape(-1).bool()
                if not mask.any(): continue
                # Released rollout reward mutates c_dot by g before storage;
                # metric fitting multiplies it by g again. Preserve the effective
                # scaling explicitly, while keeping measured data immutable.
                _, _, penalty = self._terms(error[mask],derivative[mask]*self.cfg.contraction_constraint_coef)
                loss = penalty.mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite ContractionPPO metric loss')
                self.metric_optimizer.zero_grad()
                loss.backward()
                self.metric_optimizer.step()
                losses.append(loss.item())
        if losses: self.track_data('Loss / ContractionPPO/metric',sum(losses)/len(losses))
        self.track_data('Learning / ContractionPPO/metric_learning_rate',self.cfg.contraction_metric_learning_rate)
        super().update(timestep=timestep,timesteps=timesteps)

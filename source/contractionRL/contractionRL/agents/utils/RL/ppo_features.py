"""Shared PPO feature normalization and model-based critic targets."""
from pathlib import Path
from types import SimpleNamespace

import torch


class PPOFeatures:
    """Adapt the shared tracking model to stock PPO and ContractionPPO."""

    def __init__(self, agent, env, cfg):
        from contractionRL.runners.contraction_runner import _unwrap_env
        from contractionRL.agents.utils.RL.value_expansion import install_value_expansion
        raw = _unwrap_env(env)
        self._rl_agent = agent
        self._window = raw.ref_window
        self._angle_idx = list(getattr(raw, 'angle_idx', ()))
        space = agent.action_space
        self._u_min = torch.as_tensor(space.low, device=agent.device)
        self._u_max = torch.as_tensor(space.high, device=agent.device)
        self._raw_cfg = dict(cfg.get('agent', {}))
        settings = dict(use_value_expansion=False, value_expansion_horizon=3,
                        value_expansion_batch_size=None, rewards_shaper_scale=1., action_bound_penalty=0.,
                        dynamics_lr=.001, dynamics_lr_scheduler='', dynamics_lr_scheduler_kwargs={},
                        dynamics_batch_size=4096, dynamics_pretrain_epochs=100,
                        dynamics_pretrain_data_path='', emp_dynamics_memory_size=8192,
                        num_controls_per_state=3, dynamics_val_frac=.1, dynamics_early_stop_patience=10)
        settings.update(cfg.get('empirical_dynamics', {}))
        settings.update(self._raw_cfg)
        self._cfg = SimpleNamespace(**settings)
        self._ccm_gen = None
        self._neural_dynamics = None
        self._get_f_and_B = getattr(raw, 'get_f_and_B', None)
        self._get_rollout = getattr(raw, 'get_rollout', None)
        self._dynamics_lr_scheduler = None
        self.track_data = agent.track_data
        self.write_tracking_data = agent.write_tracking_data
        self.raw = raw
        if self._cfg.use_value_expansion:
            if not getattr(self._window, 'compact', False):
                raise ValueError('PPO value expansion requires compact tracking observations')
            if getattr(raw, 'transition_backend', None) == 'learned':
                self._neural_dynamics = raw._dynamics_model
            elif not getattr(raw, 'has_analytical_dynamics', hasattr(raw, '_f_logic')):
                self._build_dynamics(cfg)
            install_value_expansion(self, raw)
        original_enable = agent.enable_training_mode

        def enable_training_mode(enabled=True):
            original_enable(enabled)
            if (enabled and getattr(raw, 'transition_backend', None) != 'learned'
                    and self._neural_dynamics is not None and not bool(self._neural_dynamics.disturbance_fitted)):
                self._pretrain()

        agent.enable_training_mode = enable_training_mode

    def _build_dynamics(self, cfg):
        from contractionRL.agents.utils.system_dynamics.dynamics import NeuralDynamics, dynamics_environment_metadata
        from contractionRL.agents.utils.contraction.math_utils import build_lr_scheduler
        from contractionRL.runners.contraction_runner import _resolve_symmetry, _dynamics_symmetry
        block = cfg.get('models', {}).get('dynamics')
        if block is None:
            raise ValueError('PhysX PPO value expansion requires models.dynamics')
        net = block.get('network', [{}])[0]
        sym = _dynamics_symmetry(_resolve_symmetry(self.raw, self._window.x_dim, self._angle_idx))
        model = NeuralDynamics(self._window.x_dim, self._window.u_dim, hidden_dim=net.get('layers', [128,128]),
                               activation=net.get('activations', 'tanh'), device=self._rl_agent.device,
                               angle_idx=self._angle_idx, sym=sym)
        model.transition_metadata = dynamics_environment_metadata(self.raw)
        self._neural_dynamics = model
        self._get_f_and_B = model.get_f_and_B
        self._dynamics_optimizer = torch.optim.Adam(model.parameters(), lr=float(self._cfg.dynamics_lr))
        self._dynamics_lr_scheduler = build_lr_scheduler(self._dynamics_optimizer,
            self._cfg.dynamics_lr_scheduler, self._cfg.dynamics_lr_scheduler_kwargs)
        self._rl_agent.checkpoint_modules['dynamics'] = model

    def _train_dynamics(self, data):
        model = self._neural_dynamics
        x, u, dx = [torch.as_tensor(data[key], dtype=torch.float32, device=model.device)
                    for key in ('x', 'u', 'x_dot')]
        loss = model.negative_log_likelihood(x, u, dx)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PPO dynamics likelihood')
        self._dynamics_optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Non-finite PPO dynamics gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        self._dynamics_optimizer.step()
        model.disturbance_fitted.fill_(True)
        return loss.item()

    def _pretrain(self):
        from contractionRL.agents.utils.system_dynamics.dynamics import collect_stochastic_rollouts, pretrain_dynamics
        cfg, agent = self._cfg, self._rl_agent
        if cfg.dynamics_pretrain_epochs < 1:
            raise ValueError('PhysX PPO value expansion needs fitted dynamics or positive dynamics_pretrain_epochs')
        data_path = cfg.dynamics_pretrain_data_path or None
        disturbed = bool(getattr(getattr(self.raw, 'disturbance', None), 'enabled', False))
        if not data_path:
            self._dynamics_rollout_data = collect_stochastic_rollouts(self.raw, cfg.emp_dynamics_memory_size,
                low=self._u_min, high=self._u_max, device=agent.device,
                dt=getattr(self.raw, 'dt', None) or self.raw.step_dt, angle_idx=self._angle_idx,
                path=str(Path(agent.experiment_dir)/'stochastic_dynamics_data.npz'))
        self.writer = getattr(agent, 'writer', None)
        # Collection and fitting are isolated from the first policy rollout.
        pretrain_dynamics(self, epochs=cfg.dynamics_pretrain_epochs, data_path=data_path,
            timesteps=1, memory_size=cfg.emp_dynamics_memory_size, num_controls_per_state=cfg.num_controls_per_state,
            val_frac=cfg.dynamics_val_frac, early_stop_patience=cfg.dynamics_early_stop_patience,
            step_offset=-cfg.dynamics_pretrain_epochs, require_disturbed=disturbed, tag='[PPO]')
        self._neural_dynamics.eval()
        for parameter in self._neural_dynamics.parameters():
            parameter.requires_grad_(False)
        # Clear the collector's final state before policy transitions are recorded.
        self.raw.reset()

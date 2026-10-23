"""PhysX datasets and the frozen dynamics used by transfer-path training."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np
import torch


def is_transfer_path(env):
    task = getattr(getattr(env, 'spec', None), 'id', '') or ''
    return task.split(':')[-1].startswith('Isaac-Transfer-path-')


def integrate_learned(model, x, u, dt, decimation, *, stochastic=True, member_ids=None):
    noise = torch.randn_like(x) if stochastic else None
    members = member_ids
    if members is not None:
        if (members.shape != (len(x),) or members.dtype != torch.long or members.device != x.device
                or not hasattr(model, 'member_parameters')):
            raise ValueError('Dynamics member_ids must provide one device-local int64 index per state')
    elif stochastic and hasattr(model, 'member_parameters'):
        members = torch.randint(model.ensemble_size, (len(x),), device=x.device)
    for _ in range(decimation):
        if members is not None:
            derivative, std = model.member_parameters(x, u, members)
            if noise is not None:
                derivative = derivative + std * noise
        else:
            derivative = model.predict_x_dot(x, u)
            if noise is not None:
                derivative = derivative + model.disturbance_log_std(x, u).exp() * noise
        x = x + dt * derivative
    return x


def ood_reward_penalty(env, x, u):
    """One control-interval uncertainty deduction; never modify PhysX rewards."""
    coef = getattr(env, '_transfer_settings', {}).get('ood_penalty_coef', 0.)
    if (not is_transfer_path(env) or getattr(env, 'transition_backend', None) != 'learned'
            or not getattr(env, '_ood_penalty_enabled', True) or not coef):
        return x.new_zeros(len(x))
    model = env._dynamics_model
    uncertainty = (model.uncertainty(x, u) if hasattr(model, 'uncertainty')
                   else model.disturbance_log_std(x, u).exp().norm(dim=-1))
    # A held derivative disturbance produces dt * sigma over one control interval.
    return float(coef) * env.step_dt * uncertainty


class TransferDynamicsMixin:
    transition_backend = 'physx'

    def set_transition_backend(self, backend):
        if backend not in ('learned', 'physx'):
            raise ValueError('Transfer transition backend must be learned or physx')
        if backend == 'learned' and self._dynamics_model is None:
            raise RuntimeError('Fit/load transfer dynamics before selecting learned transitions')
        if backend == 'learned' and self.transition_backend != 'learned':
            self._learned_state = self._get_physical_state().detach().clone()
        if backend == 'learned' and hasattr(self._dynamics_model, 'member_parameters'):
            if getattr(self, '_dynamics_member_ids', None) is None:
                n, k = len(self._learned_state), self._dynamics_model.ensemble_size
                device = self._learned_state.device
                labels = torch.randperm(k, device=device)
                ids = labels[torch.arange(n, device=device) % k]
                self._dynamics_member_ids = ids[torch.randperm(n, device=device)]
        self.transition_backend = backend

    def _capture_reset_state(self, env_ids):
        previous = self.transition_backend
        self.transition_backend = 'physx'
        try:
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            state = self._get_physical_state().detach().clone()
        finally:
            self.transition_backend = previous
        self._learned_state[env_ids] = state[env_ids]

    def _step_learned(self, actions):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        members = getattr(self, '_dynamics_member_ids', None)
        if hasattr(self._dynamics_model, 'member_parameters') and members is None:
            raise RuntimeError('Select the learned backend before stepping its assigned dynamics members')
        with torch.no_grad():
            self._ood_reward_penalty = ood_reward_penalty(self, self._learned_state, actions)
            self._learned_state = integrate_learned(self._dynamics_model, self._learned_state, actions,
                                                     self.cfg.sim.dt, self.cfg.decimation, member_ids=members)
        self._sim_step_counter += self.cfg.decimation
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        ids = self.reset_buf.nonzero(as_tuple=True)[0]
        if len(ids):
            self._reset_idx(ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _collection_step(self, controls, gust, substeps):
        self._pre_physics_step(controls)
        body = self._ee_id[:1] if self.u_dim == 7 else [0]
        self._robot.set_external_force_and_torque(gust[:, None, :3], gust[:, None, 3:],
                                                   body_ids=body, is_global=True)
        for _ in range(substeps):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        return self._get_physical_state().detach().clone()

    def _write_collection_state(self, state):
        """Write independent coordinates; PhysX reconstructs dependent kinematics."""
        from isaaclab.utils.math import quat_from_euler_xyz, quat_apply
        n = len(state)
        ids = torch.arange(n, device=self.device)
        self.scene.reset(ids)
        root = self._robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        joint_pos = self._robot.data.default_joint_pos[ids].clone()
        joint_vel = self._robot.data.default_joint_vel[ids].clone()
        if self.u_dim == 7:
            joint_pos[:, self._arm_ids] = state[:, :7]
            joint_vel[:, self._arm_ids] = state[:, 7:14]
        else:
            m = self.u_dim
            gravity = state[:, 3:6]
            roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
            pitch = torch.asin(gravity[:, 0].clamp(-1, 1))
            quat = quat_from_euler_xyz(roll, pitch, state[:, 2])
            root[:, :2] = self.scene.env_origins[ids, :2] + state[:, :2]
            root[:, 2] = state[:, -1]
            root[:, 3:7] = quat
            root[:, 7:10] = quat_apply(quat, state[:, 6+m:9+m])
            root[:, 10:13] = quat_apply(quat, state[:, 9+m:12+m])
            joint_pos[:, self._joint_ids] += state[:, 6:6+m]
            joint_vel[:, self._joint_ids] = state[:, 12+m:12+2*m]
        self._robot.write_root_pose_to_sim(root[:, :7], ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:13], ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._continuous_state = None
        if self.angle_idx:
            self._continuous_state = self._get_physical_state().detach().clone()
            self._continuous_state[:n, list(self.angle_idx)] = state[:, list(self.angle_idx)]

    def _sample_collection_states(self, count):
        low, high = [torch.as_tensor(v, device=self.device) for v in self._state_bounds()]
        state = low + torch.rand(count, self.x_dim, device=self.device) * (high - low)
        joints = self._arm_ids if self.u_dim == 7 else self._joint_ids
        limits = self._robot.data.soft_joint_pos_limits[:count, joints]
        default = 0 if self.u_dim == 7 else self._robot.data.default_joint_pos[:count, joints]
        start = 0 if self.u_dim == 7 else 6
        joint_low = torch.maximum(low[start:start+self.u_dim], limits[..., 0] - default)
        joint_high = torch.minimum(high[start:start+self.u_dim], limits[..., 1] - default)
        if (joint_low >= joint_high).any():
            raise ValueError('Configured state bounds do not overlap realizable joint limits')
        state[:, start:start+self.u_dim] = joint_low + torch.rand_like(joint_low) * (joint_high - joint_low)
        if self.u_dim != 7:
            direction = torch.randn(count, 3, device=self.device)
            state[:, 3:6] = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return state

    @torch.no_grad()
    def collect_uniform_dynamics(self, count, controls_per_state):
        if not isinstance(controls_per_state, int) or isinstance(controls_per_state, bool) or controls_per_state < 1:
            raise ValueError('num_controls_per_state must be a positive integer')
        if count % controls_per_state:
            raise ValueError('Uniform dataset size must be divisible by num_controls_per_state')
        low, high = [torch.as_tensor(v, device=self.device) for v in self._control_bounds()]
        records = {key: [] for key in ('x', 'u', 'x_dot', 'trajectory_id')}
        total_states = count // controls_per_state
        for first in range(0, total_states, self.num_envs):
            n = min(self.num_envs, total_states - first)
            states = self._sample_collection_states(n)
            low_x, high_x = [torch.as_tensor(v, device=self.device) for v in self._state_bounds()]
            for _ in range(100):
                self._write_collection_state(states)
                measured = self._get_physical_state()[:n]
                valid = torch.isfinite(measured).all(-1) & ((measured >= low_x) & (measured <= high_x)).all(-1)
                if valid.all():
                    break
                states[~valid] = self._sample_collection_states(n)[~valid]
            else:
                raise RuntimeError('Could not sample realizable states inside the configured bounds')
            for _ in range(controls_per_state):
                self._write_collection_state(states)
                x0 = self._get_physical_state().detach().clone()
                u = low + torch.rand(self.num_envs, self.u_dim, device=self.device) * (high - low)
                self.disturbance.reset(torch.arange(self.num_envs, device=self.device))
                gust = self.disturbance.advance()
                x1 = self._collection_step(u, gust, 1)
                x2 = self._collection_step(u, gust, 1)
                dx = (-3 * x0.double() + 4 * x1.double() - x2.double()) / (2 * self.physics_dt)
                for key, value in (('x', x0[:n]), ('u', u[:n]), ('x_dot', dx[:n]),
                                   ('trajectory_id', torch.arange(first, first+n, device=self.device))):
                    records[key].append(value.cpu().numpy())
        return {key: np.concatenate(values).astype(np.int64 if key == 'trajectory_id' else np.float32)
                for key, values in records.items()}

    @torch.no_grad()
    def collect_reference_dynamics(self, trajectory_ids, length):
        states, controls = [], []
        for first in range(0, len(trajectory_ids), self.num_envs):
            ids = trajectory_ids[first:first+self.num_envs]
            n = len(ids)
            initial = self._traj_buf.initial_state(ids)
            self._write_collection_state(initial)
            self.disturbance.reset(torch.arange(self.num_envs, device=self.device))
            xs, us = [self._get_physical_state()[:n].detach().clone()], []
            for k in range(length):
                control = torch.zeros(self.num_envs, self.u_dim, device=self.device)
                control[:n] = self._traj_buf._actions[ids, k]
                us.append(control[:n].clone())
                xs.append(self._collection_step(control, self.disturbance.advance(), self.cfg.decimation)[:n])
            states.append(torch.stack(xs, 1))
            controls.append(torch.stack(us, 1))
        return torch.cat(states), torch.cat(controls)


def fourth_order_derivative(states, dt):
    """Five-point derivatives, including shifted stencils at both endpoints."""
    if states.shape[-2] < 5 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Fourth-order differences require at least five states and positive dt')
    result = torch.empty_like(states)
    result[..., 2:-2, :] = (states[..., :-4, :] - 8 * states[..., 1:-3, :]
                           + 8 * states[..., 3:-1, :] - states[..., 4:, :]) / (12 * dt)
    result[..., 0, :] = (-25 * states[..., 0, :] + 48 * states[..., 1, :]
                        - 36 * states[..., 2, :] + 16 * states[..., 3, :] - 3 * states[..., 4, :]) / (12 * dt)
    result[..., 1, :] = (-3 * states[..., 0, :] - 10 * states[..., 1, :]
                        + 18 * states[..., 2, :] - 6 * states[..., 3, :] + states[..., 4, :]) / (12 * dt)
    result[..., -1, :] = (25 * states[..., -1, :] - 48 * states[..., -2, :]
                         + 36 * states[..., -3, :] - 16 * states[..., -4, :] + 3 * states[..., -5, :]) / (12 * dt)
    result[..., -2, :] = (3 * states[..., -1, :] + 10 * states[..., -2, :]
                         - 18 * states[..., -3, :] + 6 * states[..., -4, :] - states[..., -5, :]) / (12 * dt)
    return result


def reference_dataset(env, count):
    length = int(env.max_episode_length)
    needed = math.ceil(count / length)
    bank = env._traj_buf
    if needed > len(bank._valid_ids):
        raise ValueError(f'Need {needed} distinct full reference trajectories; bank has {len(bank._valid_ids)}')
    ids = bank._valid_ids[torch.randperm(len(bank._valid_ids), device=env.device)[:needed]]
    states, controls = env.collect_reference_dynamics(ids, length)
    states = states.double()
    if states.shape[1] != length + 1 or (bank._lengths[ids] < length + 1).any():
        raise ValueError('Reference data must include every state through the episode endpoint')
    derivative = fourth_order_derivative(states, float(env.step_dt))[:, :length]
    return {'x': states[:, :length].reshape(-1, env.x_dim)[:count].float().cpu().numpy(),
            'u': controls.reshape(-1, env.u_dim)[:count].cpu().numpy(),
            'x_dot': derivative.reshape(-1, env.x_dim)[:count].float().cpu().numpy(),
            'trajectory_id': ids[:, None].expand(-1, length).reshape(-1)[:count].cpu().numpy()}


def validate_dataset(data, env, count):
    for key, width in (('x', env.x_dim), ('u', env.u_dim), ('x_dot', env.x_dim)):
        value = np.asarray(data[key])
        if value.ndim != 2 or value.shape[1] != width or len(value) != count or not np.isfinite(value).all():
            raise ValueError(f'Dynamics dataset {key} must contain exactly {count} finite rows of width {width}')
    low, high = env._control_bounds()
    if ((data['u'] < low) | (data['u'] > high)).any():
        raise ValueError('Dynamics dataset contains controls outside the configured bounds')
    if 'trajectory_id' not in data or np.asarray(data['trajectory_id']).shape != (count,):
        raise ValueError('Dynamics dataset requires one trajectory/state group per sample')


def collect_dataset(env, settings, path=None):
    if path is not None and Path(path).exists():
        raise FileExistsError(f'Refusing to overwrite dynamics dataset: {path}')
    count = settings['emp_dynamics_memory_size']
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError('emp_dynamics_memory_size must be an integer >= 2')
    mode = settings['dataset_mode']
    if mode == 'reference':
        data = reference_dataset(env, count)
        interval, scheme = env.step_dt, 'central4_forward4_backward4'
    elif mode == 'uniform':
        data = env.collect_uniform_dynamics(count, settings['num_controls_per_state'])
        interval, scheme = env.cfg.sim.dt, 'forward2'
    else:
        raise ValueError('empirical_dynamics.dataset_mode must be uniform or reference')
    validate_dataset(data, env, count)
    metadata = dict(dataset_mode=mode, derivative_scheme=scheme, derivative_dt=float(interval),
                    sample_count=count, disturbance_enabled=bool(env.disturbance.enabled),
                    state_names=list(env.state_names), action_coordinates='pd_offset_v1',
                    physics_dt=float(env.cfg.sim.dt), control_dt=float(env.step_dt),
                    disturbance_scales=env.disturbance.scales.cpu().tolist())
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite dynamics dataset: {path}')
        np.savez_compressed(path, **data, metadata=json.dumps(metadata))
    return data, metadata


def fit_dynamics(model, data, settings, *, split=None, bootstrap=False):
    from contractionRL.agents.utils.contraction.math_utils import EarlyStopper, build_lr_scheduler
    fraction = float(settings['dynamics_val_frac'])
    epochs, batch_size = settings['dynamics_pretrain_epochs'], settings['dynamics_batch_size']
    if not 0 <= fraction < 1 or epochs < 1 or batch_size < 1:
        raise ValueError('Require positive dynamics epochs/batch size and 0 <= dynamics_val_frac < 1')
    groups = np.asarray(data['trajectory_id'])
    unique = np.unique(groups)
    if fraction and len(unique) < 2:
        raise ValueError('Validation requires at least two independent trajectories/states; increase dataset size')
    if split is None:
        permutation = np.random.permutation(unique)
        n_val = max(1, min(len(unique) - 1, int(len(unique) * fraction))) if fraction else 0
        val_mask = np.isin(groups, permutation[:n_val])
        train = torch.as_tensor(np.flatnonzero(~val_mask), device=model.device)
        val = torch.as_tensor(np.flatnonzero(val_mask), device=model.device)
    else:
        train, val = split
    if hasattr(model, 'members'):
        reports = [fit_dynamics(member, data, settings, split=(train, val), bootstrap=True)
                   for member in model.members]
        model.disturbance_fitted.fill_(True)
        model.eval().requires_grad_(False)
        return {'ensemble_size': model.ensemble_size, 'member_activations': model.member_activations, 'members': reports,
                'train_samples': len(train), 'val_samples': len(val)}
    if bootstrap:
        train = train[torch.randint(len(train), (len(train),), device=model.device)]
    x, u, dx = [torch.as_tensor(data[key], device=model.device, dtype=torch.float32) for key in ('x', 'u', 'x_dot')]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings['dynamics_lr']))
    scheduler = build_lr_scheduler(optimizer, settings['dynamics_lr_scheduler'], settings['dynamics_lr_scheduler_kwargs'])
    stopper = EarlyStopper(patience=settings['dynamics_early_stop_patience'] if len(val) else 0)
    history = []
    model.train()
    for epoch in range(epochs):
        order = train[torch.randperm(len(train), device=model.device)]
        total_loss = 0.
        for indices in order.split(batch_size):
            loss = model.negative_log_likelihood(x[indices], u[indices], dx[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite transfer dynamics loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        with torch.no_grad():
            val_loss = (sum(float(model.negative_log_likelihood(x[ids], u[ids], dx[ids])) * len(ids)
                            for ids in val.split(batch_size)) / len(val)) if len(val) else None
        if val_loss is not None and not math.isfinite(val_loss):
            raise FloatingPointError('Nonfinite transfer dynamics validation loss')
        history.append(dict(epoch=epoch + 1, train_nll=total_loss / len(train), val_nll=val_loss,
                            lr=optimizer.param_groups[0]['lr']))
        if scheduler is not None:
            scheduler.step()
        if val_loss is not None and stopper.step(val_loss, model, epoch):
            break
    if len(val):
        stopper.restore_best(model)
    model.disturbance_fitted.fill_(True)
    model.eval().requires_grad_(False)
    return {'epochs': len(history), 'train_samples': len(train), 'val_samples': len(val),
            'best_epoch': stopper.best_epoch + 1 if len(val) else len(history), 'history': history}


def prepare_transfer_dynamics(env, cfg, directory=None, *, checkpoint=None, dynamics_checkpoint=None):
    """Fit/load once before runner construction; every algorithm consumes this model."""
    if not is_transfer_path(env):
        return None
    from contractionRL.agents.utils.system_dynamics.dynamics import EnsembleDynamics, NeuralDynamics, dynamics_environment_metadata
    settings = cfg['empirical_dynamics']
    ensemble_size = settings.get('dynamics_ensemble_size', 5)
    coefficient = settings.get('ood_penalty_coef', 0.)
    if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, int) or ensemble_size < 1:
        raise ValueError('dynamics_ensemble_size must be a positive integer')
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('ood_penalty_coef must be finite and nonnegative')
    block = cfg['models']['dynamics']
    if block.get('class') != 'NeuralDynamics' or len(block.get('network', [])) != 1:
        raise ValueError('Transfer dynamics requires one NeuralDynamics network specification')
    spec = block['network'][0]
    expected = dynamics_environment_metadata(env)
    expected.update(physics_dt=float(env.cfg.sim.dt), decimation=int(env.cfg.decimation),
                    disturbance_scales=env.disturbance.scales.cpu().tolist())
    if checkpoint or dynamics_checkpoint:
        if checkpoint and dynamics_checkpoint:
            raise ValueError('A resumed transfer agent uses its own simulator; omit dynamics_checkpoint')
        saved_path = Path(dynamics_checkpoint) if dynamics_checkpoint else Path(checkpoint).parent / 'transfer_dynamics.pt'
        if not saved_path.is_file():
            raise FileNotFoundError(f'Transfer checkpoint requires its fitted simulator: {saved_path}')
        model = NeuralDynamics.load(str(saved_path), device=env.device, expected_metadata=expected)
        if getattr(model, 'ensemble_size', 1) != ensemble_size:
            raise ValueError('Fitted simulator ensemble size does not match dynamics_ensemble_size')
        missing = set(expected) - set(model.checkpoint_metadata())
        if missing or not bool(model.disturbance_fitted):
            raise ValueError(f'Transfer simulator requires fitted probabilistic dynamics and complete provenance; missing {sorted(missing)}')
        if model._hidden_dim != list(spec['layers']) or model._activation_str != spec['activations']:
            raise ValueError('Fitted simulator architecture does not match models.dynamics')
    else:
        from contractionRL.agents.utils.system_dynamics.state import StateSymmetry
        symmetry = StateSymmetry.from_names(env.state_names).demote_to_translation()
        model = EnsembleDynamics(env.x_dim, env.u_dim, hidden_dim=spec['layers'], activation=spec['activations'],
                                 device=env.device, angle_idx=env.angle_idx, sym=symmetry, ensemble_size=ensemble_size)
        model.transition_metadata = expected
        source = settings['dynamics_pretrain_data_path']
        if source:
            with np.load(source, allow_pickle=False) as archive:
                data = {key: archive[key] for key in ('x', 'u', 'x_dot', 'trajectory_id')}
                metadata = json.loads(str(archive['metadata']))
            validate_dataset(data, env, settings['emp_dynamics_memory_size'])
            if metadata.get('state_names') != list(env.state_names) or metadata.get('action_coordinates') != 'pd_offset_v1':
                raise ValueError('Offline dynamics state/control provenance does not match this task')
            if (metadata.get('disturbance_enabled') != bool(env.disturbance.enabled)
                    or metadata.get('disturbance_scales') != env.disturbance.scales.cpu().tolist()
                    or not math.isclose(metadata.get('physics_dt', -1), env.cfg.sim.dt)
                    or not math.isclose(metadata.get('control_dt', -1), env.step_dt)
                    or metadata.get('dataset_mode') != settings['dataset_mode']):
                raise ValueError('Offline dynamics disturbance, timestep or collection mode does not match this task')
        else:
            data, metadata = collect_dataset(env, settings, Path(directory) / 'transfer_dynamics_data.npz')
        report = fit_dynamics(model, data, settings)
        env._transfer_fit_report = report
        report['dataset'] = metadata
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / 'transfer_dynamics_fit.json').write_text(json.dumps(report, indent=2))
        model.save(str(Path(directory) / 'checkpoints' / 'transfer_dynamics.pt'))
    model.eval().requires_grad_(False)
    if (checkpoint or dynamics_checkpoint) and directory is not None:
        destination = Path(directory) / 'checkpoints' / 'transfer_dynamics.pt'
        if destination.resolve() != saved_path.resolve():
            if destination.exists():
                raise FileExistsError(f'Refusing to overwrite the fitted simulator: {destination}')
            model.save(str(destination))
    env.set_dynamics_model(model)
    env._transfer_settings = dict(settings)
    if not checkpoint and not dynamics_checkpoint:
        env._transfer_dataset = data
    env.set_transition_backend('learned')
    return model


def attach_transfer_agent(env, agent):
    if is_transfer_path(env):
        agent.checkpoint_modules['transfer_dynamics'] = env._dynamics_model
        inner = getattr(agent, '_rl_agent', None)
        if inner is not None:
            inner.checkpoint_modules['transfer_dynamics'] = env._dynamics_model
        if not getattr(agent, '_transfer_attached', False):
            agent._transfer_attached = True
            original = agent.init
            def init(*args, **kwargs):
                original(*args, **kwargs)
                from contractionRL.agents.utils.RL.logging import log_metrics
                report = getattr(env, '_transfer_fit_report', {})
                for index, member in enumerate(report.get('members', [report])):
                    prefix = f'Dynamics Training/member_{index}/' if 'members' in report else 'Dynamics Training/'
                    for row in member.get('history', []):
                        values = {prefix + key: value for key, value in row.items()
                                  if key != 'epoch' and value is not None}
                        log_metrics(values, step=row['epoch'], axis='dynamics_epoch')
            agent.init = init


def force_wrapper_reset(env):
    current = env
    for _ in range(12):
        if '_reset_once' in vars(current):
            current._reset_once = True
            return
        current = getattr(current, 'env', getattr(current, '_env', None))
        if current is None:
            return


def evaluate_backend_pair(env, callback):
    from contractionRL.agents.utils.RL.logging import evaluation_section
    results = {}
    for backend, section, key in (('learned', 'Simulation Evaluation', 'simulation'),
                                   ('physx', 'Sim-to-Real Evaluation', 'sim_to_real')):
        with evaluation_backend(env.unwrapped, backend), evaluation_section(section):
            force_wrapper_reset(env)
            results[key] = callback()
    return results


@contextmanager
def evaluation_backend(env, backend):
    """Use matching random scenarios for each backend without consuming training RNG."""
    previous = env.transition_backend
    previous_penalty = getattr(env, '_ood_penalty_enabled', True)
    device = torch.device(env.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            seed = int(env.cfg.seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            env.set_transition_backend(backend)
            env._ood_penalty_enabled = False
            yield
    finally:
        env._ood_penalty_enabled = previous_penalty
        env.set_transition_backend(previous)

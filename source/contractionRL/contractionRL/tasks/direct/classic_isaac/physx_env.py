"""GPU-parallel PhysX bodies with measured nominal trajectory references.

Car is an ideal planar chassis, without wheel/suspension geometry. Quadrotor
is a thrust-driven cuboid with body-rate actuators, without rotor aerodynamics.
All motion is integrated by PhysX; analytical classic transitions are not used.
"""
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_apply

from contractionRL.agents.utils.system_dynamics.reference import ReferenceState
from contractionRL.agents.utils.system_dynamics.state import wrap_diff
from contractionRL.agents.utils.contraction.math_utils import quadratic_energy, bound_W, spd_inverse
from ..common.disturbance import BoundedGust
from .physx_model import domain, body_wrench


class PhysXTrackingEnv(DirectRLEnv):
    has_analytical_dynamics = False
    mode = 'sim_to_real'
    model_version = 'physx-body-v1'
    terminate_out_of_box = True

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.quad = cfg.classic_task == 'classic-quadrotor-v0'
        low, high, control, self.state_names, self.angle_idx, scales, margin = domain(cfg.classic_task)
        self.x_dim, self.u_dim = len(low), len(control)
        self.num_dim_x, self.num_dim_control = self.x_dim, self.u_dim
        self.pos_dimension = 1 if self.x_dim == 4 else (3 if self.quad else 2)
        reference_count = cfg.scene.num_envs
        if cfg.physx_reference_path:
            with np.load(cfg.physx_reference_path) as bank:
                reference_count = len(bank["x"])
        self.ref_window = ReferenceState(self.x_dim, self.u_dim, 1, self._reference_control,
                                         parameter_bounds=([0.], [reference_count-1.]),
                                         time_bound=cfg.episode_length_s,
                                         state_names=self.state_names)
        cfg.observation_space = self.ref_window.space(low, high)
        cfg.action_space = gym.spaces.Box(-control, control, dtype=np.float32)
        super().__init__(cfg, render_mode, **kwargs)
        self.X_MIN = torch.as_tensor(low, device=self.device)
        self.X_MAX = torch.as_tensor(high, device=self.device)
        self.U_MAX = torch.as_tensor(control, device=self.device)
        self.U_MIN = -self.U_MAX
        self.reference_margin = torch.as_tensor(margin, device=self.device)
        self.time_bound, self.dt = cfg.episode_length_s, self.step_dt
        size = torch.tensor([.25, .25, .08] if self.quad else [.6, .3, .15], device=self.device)
        self.inertia = (size.square().sum() - size.square()) / 12  # 1 kg cuboid
        self._thrust = torch.full((self.num_envs,), 9.81, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.u_dim, device=self.device)
        self._gust = torch.zeros(self.num_envs, self.x_dim, device=self.device)
        self.disturbance = BoundedGust(self.num_envs, scales if cfg.disturbance_enabled else scales*0, self.step_dt, self.device)
        self._traj_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamics_model = self.ccm_gen = None
        self._system_id_data = None
        self.fix_ref_trajectories = False
        self._fixed_ids = torch.full_like(self._traj_ids, -1)
        self._prev_energy = torch.zeros(self.num_envs, device=self.device)
        self.init_tracking_error = torch.zeros_like(self._prev_energy)
        self.init_maha_error = torch.zeros_like(self._prev_energy)
        try:
            if cfg.physx_reference_path:
                self._load_references(cfg.physx_reference_path)
            else:
                self._build_references()
            if getattr(cfg, "log_dir", None):
                target = Path(cfg.log_dir) / "physx_reference_bank.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite reference identities: {target}")
                np.savez_compressed(target, x=self._refs.cpu().numpy(), u=self._ref_controls.cpu().numpy(),
                                    dt=self.step_dt, model=self.model_version, task=cfg.classic_task)
                cfg.physx_reference_path = str(target)
        except BaseException:
            self.close()
            raise
        self.ref_window.parameter_bounds = ([0.], [len(self._refs)-1.])
        cfg.observation_space = self.ref_window.space(low, high)
        self._configure_gym_env_spaces()
        # One-time nominal rollouts end before any training episode begins.
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    @property
    def time_steps(self):
        return self.episode_length_buf

    @property
    def xref(self):
        return self._refs[self._traj_ids]

    @property
    def uref(self):
        return self._ref_controls[self._traj_ids]

    def _setup_scene(self):
        body_cfg = RigidObjectCfg(
            prim_path='/World/envs/env_.*/Body',
            spawn=sim_utils.CuboidCfg(
                size=(.25, .25, .08) if self.quad else (.6, .3, .15),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=not self.quad,
                                                            linear_damping=0., angular_damping=0., max_depenetration_velocity=1.),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1, .5, 1.))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 4. if self.quad else .2)))
        self._body = RigidObject(body_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.rigid_objects['body'] = self._body
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/Ground', ground)
        self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
        light = sim_utils.DomeLightCfg(intensity=2000.)
        light.func('/World/Light', light)
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path='/World/Reference', markers={'target': sim_utils.SphereCfg(
                radius=.12, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2, .9, .3)))}))

    def get_physical_state(self):
        p = self._body.data.root_pos_w - self.scene.env_origins
        rpy = wrap_diff(torch.stack(euler_xyz_from_quat(self._body.data.root_quat_w), -1), [0, 1, 2])
        if self.quad:
            return torch.cat((p, self._body.data.root_lin_vel_w, self._thrust[:, None], rpy,
                              self._body.data.root_ang_vel_b), -1)
        return torch.cat((p[:, :2], rpy[:, 2:3], self._body.data.root_lin_vel_b[:, :2],
                          self._body.data.root_ang_vel_b[:, 2:3]), -1)

    def _write_state(self, ids, x):
        p = torch.zeros(len(ids), 3, device=self.device)
        p[:, :self.pos_dimension] = x[:, :self.pos_dimension]
        if not self.quad:
            p[:, 2] = .2
        angles = x[:, 7:10] if self.quad else torch.stack((torch.zeros_like(x[:, 2]), torch.zeros_like(x[:, 2]), x[:, 2]), -1)
        q = quat_from_euler_xyz(*angles.unbind(-1))
        velocity = torch.zeros(len(ids), 6, device=self.device)
        if self.quad:
            velocity[:, :3] = x[:, 3:6]
            velocity[:, 3:] = quat_apply(q, x[:, 10:13])
            self._thrust[ids] = x[:, 6]
        else:
            vb = torch.cat((x[:, 3:5], torch.zeros(len(ids), 1, device=self.device)), -1)
            velocity[:, :3] = quat_apply(q, vb)
            velocity[:, 5] = x[:, 5]
        self._body.write_root_pose_to_sim(torch.cat((p + self.scene.env_origins[ids], q), -1), ids)
        self._body.write_root_velocity_to_sim(velocity, ids)
        self._body.reset(ids)

    def _pre_physics_step(self, actions):
        self._actions = actions.nan_to_num().clamp(self.U_MIN, self.U_MAX)
        self._gust = self.disturbance.advance()

    def _apply_action(self):
        if self.quad:
            self._thrust = (self._thrust + self.physics_dt * self._actions[:, 0]).clamp(self.X_MIN[6], self.X_MAX[6])
        x = self.get_physical_state()
        force, torque = body_wrench(x, self._actions, self.inertia, self.quad)
        if self.quad:
            force = quat_apply(self._body.data.root_quat_w, force) + self._gust[:, 3:6]
            torque = quat_apply(self._body.data.root_quat_w, torque + self.inertia * self._gust[:, 10:13])
        else:
            force[:, :2] += self._gust[:, 3:5]
            torque[:, 2] += self.inertia[2] * self._gust[:, 5]
            force = quat_apply(self._body.data.root_quat_w, force)
            torque = quat_apply(self._body.data.root_quat_w, torque)
        self._body.set_external_force_and_torque(force[:, None], torque[:, None], is_global=True)

    def _simulate_nominal(self, action):
        # Nominal references and identification transitions must exclude plant gusts.
        gust = self._gust
        self._gust = torch.zeros_like(gust)
        try:
            self._actions = action
            for _ in range(self.cfg.decimation):
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
        finally:
            self._gust = gust

    def _build_references(self):
        # Generate with the same PhysX plant, before training starts. Reset only
        # selects from this validated bank, so other live envs are never advanced.
        ids = torch.arange(self.num_envs, device=self.device)
        refs, controls = [], []
        for attempt in range(3):
            x0 = torch.zeros(self.num_envs, self.x_dim, device=self.device)
            if self.quad:
                x0[:, :2] = torch.rand(self.num_envs, 2, device=self.device) * 10 - 5
                x0[:, 2] = torch.rand(self.num_envs, device=self.device) * 5 + 3
                x0[:, 6] = 9.81
            else:
                x0[:, :2] = torch.rand(self.num_envs, 2, device=self.device) * 40 - 20
                x0[:, 2] = torch.rand(self.num_envs, device=self.device) * 2 - 1
                x0[:, 3] = torch.rand(self.num_envs, device=self.device) * .3 + .3 if self.cfg.classic_task.endswith('v1') else 1.5
            self._write_state(ids, x0)
            weights = torch.randn(self.num_envs, 10, self.u_dim, device=self.device)
            weights /= weights.norm(dim=1, keepdim=True).clamp_min(1e-9)
            if self.quad:
                weights *= .005
            states, actions = [self.get_physical_state().clone()], []
            for k in range(self.max_episode_length):
                phases = torch.arange(1, 11, device=self.device) * (2 * np.pi * k / self.max_episode_length)
                u = (weights * phases.sin()[None, :, None]).sum(1)
                if not self.quad:
                    # Compensate drag-induced speed loss with a nominal speed servo.
                    u[:, 1] = 2 * (x0[:, 3] - self.get_physical_state()[:, 3])
                u = u.clamp(self.U_MIN / 2, self.U_MAX / 2)
                actions.append(u.clone())
                self._simulate_nominal(u)
                states.append(self.get_physical_state().clone())
            xs, us = torch.stack(states, 1), torch.stack(actions + [actions[-1]], 1)
            valid = torch.isfinite(xs).all(dim=(1, 2))
            valid &= ((xs >= self.X_MIN + self.reference_margin) & (xs <= self.X_MAX - self.reference_margin)).all(dim=(1, 2))
            if self.quad:
                valid &= (xs[:, :, 7:9].abs() < np.pi / 3).all(dim=(1, 2))
            if not valid.all():
                violations = ((xs < self.X_MIN + self.reference_margin) | (xs > self.X_MAX - self.reference_margin)).any(1).sum(0)
                print(f"[PhysX] Reference attempt {attempt+1}: {int(valid.sum())}/{self.num_envs} accepted; violations by state: {violations.tolist()}", flush=True)
            refs.append(xs[valid]); controls.append(us[valid])
            if sum(len(r) for r in refs) >= self.num_envs:
                break
        self._refs = torch.cat(refs)[:self.num_envs]
        self._ref_controls = torch.cat(controls)[:self.num_envs]
        if not len(self._refs):
            raise RuntimeError('No in-domain PhysX nominal reference; reference generation failed closed')
        print(f'[PhysX] Validated {len(self._refs)} full-length nominal trajectories from actual rigid-body rollouts')
        self._actions.zero_()

    def _load_references(self, path):
        with np.load(path) as bank:
            if (str(bank['model']) != self.model_version or str(bank['task']) != self.cfg.classic_task
                    or not np.isclose(float(bank['dt']), self.step_dt)):
                raise ValueError('Incompatible PhysX reference bank')
            self._refs = torch.as_tensor(bank['x'], device=self.device)
            self._ref_controls = torch.as_tensor(bank['u'], device=self.device)
        if (self._refs.shape[1:] != (self.max_episode_length+1, self.x_dim) or
                self._ref_controls.shape != (*self._refs.shape[:2], self.u_dim)):
            raise ValueError('Incompatible PhysX reference bank dimensions/horizon')
        if (not torch.isfinite(self._refs).all() or not torch.isfinite(self._ref_controls).all() or
                not ((self._refs >= self.X_MIN+self.reference_margin) &
                     (self._refs <= self.X_MAX-self.reference_margin)).all() or
                not (self._ref_controls.abs() <= self.U_MAX).all()):
            raise ValueError('PhysX reference bank violates state/control bounds or margins')

    def _reference_control(self, xref, parameters, time):
        ids = parameters[:, 0].long().clamp(0, len(self._refs)-1)
        step = (time / self.step_dt).round().long().clamp(0, self.max_episode_length)
        return self._ref_controls[ids, step]

    def _current_ref(self):
        return self._refs[self._traj_ids, self.episode_length_buf.clamp(max=self.max_episode_length)]

    def _energy(self, x, xr):
        error = wrap_diff(x - xr, self.angle_idx)
        if self.ccm_gen is None:
            return error.square().sum(-1)
        with torch.no_grad():
            raw, _ = self.ccm_gen(x)
            metric = bound_W(raw, self.w_lb, self.x_dim, getattr(self.ccm_gen, 'bounded', False))
            if not getattr(self.ccm_gen, 'outputs_metric', False):
                metric = spd_inverse(metric)
            return quadratic_energy(error, metric)

    def _get_observations(self):
        x = self.get_physical_state().nan_to_num().clamp(self.X_MIN, self.X_MAX)
        xi = torch.cat((self._traj_ids[:, None].float(), (self.episode_length_buf * self.step_dt)[:, None],
                        x.new_zeros(self.num_envs, self.u_dim)), -1)
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            p = torch.zeros(self.num_envs, 3, device=self.device)
            p[:, :self.pos_dimension] = self._current_ref()[:, :self.pos_dimension]
            if not self.quad:
                p[:, 2] = getattr(self, "reference_height", .2)
            self.markers.visualize(p + self.scene.env_origins)
        return {'policy': {'x': x, 'xref': self._current_ref(), 'xi': xi}}

    def _get_dones(self):
        x = self.get_physical_state()
        failed = (~torch.isfinite(x)).any(-1) | ((x < self.X_MIN) | (x > self.X_MAX)).any(-1)
        if self.quad:
            failed |= (x[:, 2] <= .15) | (x[:, 7:9].abs() >= np.pi/3).any(-1)
        self.extras['episode_ended_early'] = failed.clone()
        self.extras['terminal_physical_state'] = x.clone()
        timeout = self.episode_length_buf >= self.max_episode_length
        done = failed | timeout
        self.extras.pop('final_observation', None)
        self.extras['_final_observation'] = done.clone()
        if done.any():
            self.extras['final_observation'] = self._get_observations()['policy']
        return failed, timeout

    def _get_rewards(self):
        raw = self.get_physical_state().nan_to_num()
        x, xr = raw.clamp(self.X_MIN, self.X_MAX), self._current_ref()
        x[:, :self.pos_dimension] = raw[:, :self.pos_dimension]
        next_energy = self._energy(x, xr)
        reward = self._prev_energy - next_energy
        self._prev_energy = next_energy
        self.extras['tracking_error_vector'] = wrap_diff(x - xr, self.angle_idx).detach().clone()
        error = self.extras['tracking_error_vector'].square().sum(-1)
        self.extras['tracking_error'] = error.clone()
        self.extras['control_effort'] = self._actions.square().sum(-1)
        self.extras['terminal_tracking_error'] = error.sqrt()
        if self.ccm_gen is not None:
            self.extras['maha_tracking_error'] = next_energy.clone()
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, '_refs'):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        chosen = torch.randint(len(self._refs), (len(ids),), device=self.device)
        if self.fix_ref_trajectories:
            chosen = torch.where(self._fixed_ids[ids] >= 0, self._fixed_ids[ids], chosen)
            self._fixed_ids[ids] = chosen
        self._traj_ids[ids] = chosen
        x = self._sample_initial_state(self._refs[chosen, 0])
        self._write_state(ids, x)
        self.disturbance.reset(ids)
        initial = self.get_physical_state()[ids]
        self.init_tracking_error[ids] = wrap_diff(initial - self._refs[chosen, 0], self.angle_idx).square().sum(-1)
        self._prev_energy[ids] = self._energy(initial, self._refs[chosen, 0])
        self.init_maha_error[ids] = self._prev_energy[ids]

    def _sample_initial_state(self, reference):
        noise = .05 * torch.rand_like(reference) * 2 - .05
        if self.quad:
            noise[:, 6] *= 2
        return wrap_diff(reference + noise, self.angle_idx).clamp(self.X_MIN, self.X_MAX)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info['tracking_error'] = self.init_tracking_error.clone()
        if self.ccm_gen is not None:
            info['maha_tracking_error'] = self.init_maha_error.clone()
        return obs, info

    def get_tracking_error(self):
        return wrap_diff(self.get_physical_state() - self._current_ref(), self.angle_idx).norm(dim=-1)

    def set_ccm(self, ccm_gen, w_lb, device, **kwargs):
        self.ccm_gen, self.w_lb = ccm_gen, w_lb
        self._prev_energy = self._energy(self.get_physical_state(), self._current_ref())
        self.init_maha_error = self._prev_energy.clone()

    def set_dynamics_model(self, model):
        self._dynamics_model = model

    def get_f_and_B(self, x, need_null=True):
        if self._dynamics_model is None:
            raise RuntimeError('PhysX requires a learned dynamics model; analytical classic f/B are not valid')
        return self._dynamics_model.get_f_and_B(x)

    def get_rollout(self, buffer_size, mode, num_control_per_state=None):
        if mode == 'dynamics':
            if self._system_id_data is None:
                from .system_id import collect_system_id
                self._system_id_data = collect_system_id(self, max(buffer_size, 2*self.num_envs),
                                                         max(2, int(num_control_per_state or 3)))
            available = len(self._system_id_data['x'])
            rows = (torch.randperm(available, device=self.device)[:buffer_size] if buffer_size <= available
                    else torch.randint(available, (buffer_size,), device=self.device))
            return {key:self._system_id_data[key][rows].cpu().numpy().astype(np.float32)
                    for key in ('x','u','x_dot')}
        ids = torch.randint(len(self._refs), (buffer_size,), device=self.device)
        steps = torch.randint(self.max_episode_length, (buffer_size,), device=self.device)
        xr, u = self._refs[ids, steps], self._ref_controls[ids, steps]
        if mode == 'c3m':
            result = {'x': wrap_diff(xr + torch.randn_like(xr)*.05, self.angle_idx).clamp(self.X_MIN, self.X_MAX), 'xref': xr, 'uref': u}
        else:
            raise ValueError(f'Unsupported rollout mode {mode}')
        return {key: value.cpu().numpy().astype(np.float32) for key, value in result.items()}

    def get_reference_trajectory(self):
        return self.xref

    def configure_ref_window(self, *args, **kwargs):
        pass  # Generator ID/time replace the old sequence layout.

    def full_ref_length(self, *args):
        return 1

    def set_fix_ref_trajectories(self, flag):
        self.fix_ref_trajectories = bool(flag)
        self._fixed_ids.fill_(-1)

    def _set_debug_vis_impl(self, debug_vis):
        self.markers.set_visibility(debug_vis)

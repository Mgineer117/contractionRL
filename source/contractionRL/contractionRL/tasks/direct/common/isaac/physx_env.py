"""PhysX transitions for the shared classic-derived Isaac tracking tasks.

Task states, reference generators, observations, resets, gusts and rewards are
owned by the same BaseEnv used by Isaac-Classic. Rate controls are imposed
kinematically, so rigid-body velocities add no independent task states.
"""
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_apply

from contractionRL.agents.utils.system_dynamics.state import wrap_diff, unwrap_state_angles
from .env import ClassicIsaacEnv, make_task_plant
from .physx_model import body_wrench, commanded_body_rates, quadrotor_angles, quadrotor_quat


class PhysXTrackingEnv(ClassicIsaacEnv):
    has_analytical_dynamics = False
    model_version = "physx-body-v4-canonical-state"

    def __init__(self, cfg, render_mode=None, **kwargs):
        if getattr(cfg, "physx_reference_path", ""):
            raise ValueError("Legacy reference-ID banks are incompatible with the shared task layout; use a newly trained policy")
        self.quad = cfg.classic_task == "classic-quadrotor-v0"
        self.plant = make_task_plant(cfg, transition_backend=self, defer_reset=True)
        cfg.observation_space = self.plant.observation_space
        cfg.action_space = self.plant.action_space
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        if abs(self.step_dt - self.plant.dt) > 1e-12 or abs(cfg.episode_length_s - self.plant.time_bound) > 1e-12:
            raise ValueError("PhysX action interval and horizon must match the shared task")
        self._thrust = torch.full((self.num_envs,), 9.81, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.u_dim, device=self.device)
        self._gust = torch.zeros_like(self.plant.x_t)
        self._dynamics_model = self._system_id_data = None
        self._initialize_body()
        try:
            self.plant.reset()
        except BaseException:
            self.close()
            raise

    def _initialize_body(self):
        pass

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
        q = self._body.data.root_quat_w
        angles = quadrotor_angles(q) if self.quad else torch.stack(euler_xyz_from_quat(q), -1)
        rpy = wrap_diff(angles, [0, 1, 2])
        if self.quad:
            x = torch.cat((p, self._body.data.root_lin_vel_w, self._thrust[:, None], rpy), -1)
        else:
            x = torch.cat((p[:, :2], rpy[:, 2:3], self._body.data.root_lin_vel_b[:, :1]), -1)
        x = unwrap_state_angles(x, self.__dict__.get("_continuous_state"), self.angle_idx)
        self._continuous_state = x.detach().clone()
        return x

    def _write_state(self, ids, x):
        if x.shape != (len(ids), self.x_dim):
            raise ValueError(f"Expected exactly {self.x_dim} state coordinates, got {tuple(x.shape)}")
        p = torch.zeros(len(ids), 3, device=self.device)
        p[:, :self.pos_dimension] = x[:, :self.pos_dimension]
        if not self.quad:
            p[:, 2] = .2
        angles = x[:, 7:10] if self.quad else torch.stack((torch.zeros_like(x[:, 2]), torch.zeros_like(x[:, 2]), x[:, 2]), -1)
        q = quadrotor_quat(angles) if self.quad else quat_from_euler_xyz(*angles.unbind(-1))
        velocity = torch.zeros(len(ids), 6, device=self.device)
        if self.quad:
            velocity[:, :3] = x[:, 3:6]
            self._thrust[ids] = x[:, 6]
        else:
            vb = torch.stack((x[:, 3], torch.zeros_like(x[:, 3]), torch.zeros_like(x[:, 3])), -1)
            velocity[:, :3] = quat_apply(q, vb)
        self._body.write_root_pose_to_sim(torch.cat((p + self.scene.env_origins[ids], q), -1), ids)
        self._body.write_root_velocity_to_sim(velocity, ids)
        self._body.reset(ids)
        if self.__dict__.get("_continuous_state") is not None:
            self._continuous_state[ids] = x

    def reset_state(self, ids, x):
        self._write_state(ids, x)

    def sync_state(self, ids, x):
        self._write_state(ids, x)

    def _snapshot_simulation(self):
        continuous = self.__dict__.get("_continuous_state")
        return dict(root=self._body.data.root_state_w.clone(), thrust=self._thrust.clone(),
                    continuous=None if continuous is None else continuous.clone(), actions=self._actions.clone(),
                    gust=self._gust.clone())

    def _restore_simulation(self, saved):
        ids = torch.arange(self.num_envs, device=self.device)
        self._body.write_root_pose_to_sim(saved["root"][:, :7], ids)
        self._body.write_root_velocity_to_sim(saved["root"][:, 7:], ids)
        self._body.reset(ids)
        self._thrust = saved["thrust"]
        self._continuous_state = saved["continuous"]
        self._actions, self._gust = saved["actions"], saved["gust"]

    def _apply_action(self):
        x = self.get_physical_state()
        q = self._body.data.root_quat_w
        velocity = x.new_zeros(self.num_envs, 6)
        if self.quad:
            velocity[:, :3] = x[:, 3:6]
        else:
            # The analytical car has no independent lateral velocity.
            forward = torch.stack((x[:, 3], torch.zeros_like(x[:, 3]), torch.zeros_like(x[:, 3])), -1)
            velocity[:, :3] = quat_apply(q, forward)
        velocity[:, 3:] = quat_apply(q, commanded_body_rates(x, self._actions, self.quad))
        self._body.write_root_velocity_to_sim(velocity)
        if self.quad:
            self._thrust = (self._thrust + self.physics_dt * self._actions[:, 0]).clamp(self.X_MIN[6], self.X_MAX[6])
            x[:, 6] = self._thrust
        force, torque = body_wrench(x, self._actions, self.quad)
        force = quat_apply(self._body.data.root_quat_w, force)
        torque = quat_apply(self._body.data.root_quat_w, torque)
        self._body.set_external_force_and_torque(force[:, None], torque[:, None], is_global=True)

    def _simulate_nominal(self, action):
        self._actions = action
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

    def advance(self, x, u, noise):
        ids = torch.arange(len(x), device=self.device)
        self.sync_state(ids, x)
        actions = u.new_zeros(self.num_envs, self.u_dim)
        actions[:len(u)] = u
        self._simulate_nominal(actions)
        # Same discrete additive-derivative gust as the analytical task, in the
        # same units. These are lumped disturbances, not aerodynamic forces.
        next_x = self.get_physical_state()[:len(x)] + self.dt * noise
        return next_x, (next_x - x) / self.dt

    def rollout_reference(self, plant, x0, freqs, weights):
        saved = self._snapshot_simulation()
        results, parameters, valid = [], [], []
        try:
            for start in range(0, len(x0), self.num_envs):
                initial, mixing = x0[start:start+self.num_envs], weights[start:start+self.num_envs]
                self.reset_state(torch.arange(len(initial), device=self.device), initial)
                results.append(plant._integrate_reference(initial, freqs, mixing))
                parameters.append(plant._reference_parameters_draw.clone())
                valid.append(plant._reference_valid_draw.clone())
        finally:
            self._restore_simulation(saved)
        plant._reference_parameters_draw = torch.cat(parameters)
        plant._reference_valid_draw = torch.cat(valid)
        return torch.cat([r[0] for r in results]), torch.cat([r[1] for r in results]), results[0][2]

    def _draw(self):
        plant = self.plant
        rows = torch.arange(self.num_envs, device=self.device)
        ref = plant.xref[rows, plant.time_steps.clamp(max=plant.xref.shape[1]-1)]
        p = ref.new_zeros(self.num_envs, 3)
        p[:, :self.pos_dimension] = ref[:, :self.pos_dimension]
        if not self.quad:
            p[:, 2] = getattr(self, "reference_height", .2)
        self.markers.visualize(p + self.scene.env_origins)

    def _advance_scene(self):
        # advance() already integrated all physical substeps.
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

    def _get_observations(self):
        return {"policy": self.plant.construct_state(self.plant.x_t)}

    def set_dynamics_model(self, model):
        self._dynamics_model = model

    def get_f_and_B(self, x, need_null=True):
        if self._dynamics_model is None:
            raise RuntimeError("PhysX requires a learned dynamics model; analytical classic f/B are not valid")
        return self._dynamics_model.get_f_and_B(x)

    def get_rollout(self, buffer_size, mode, num_control_per_state=None):
        if mode == "c3m":
            return self.plant.get_rollout(buffer_size, mode, num_control_per_state)
        if mode != "dynamics":
            raise ValueError(f"Unsupported rollout mode {mode}")
        if self._system_id_data is None:
            from .system_id import collect_system_id
            self._system_id_data = collect_system_id(self, max(buffer_size, 2*self.num_envs),
                                                     max(2, int(num_control_per_state or 3)))
        available = len(self._system_id_data["x"])
        rows = (torch.randperm(available, device=self.device)[:buffer_size] if buffer_size <= available
                else torch.randint(available, (buffer_size,), device=self.device))
        return {key: self._system_id_data[key][rows].cpu().numpy().astype(np.float32)
                for key in ("x", "u", "x_dot")}

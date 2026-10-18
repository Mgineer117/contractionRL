"""Local articulated cartpole and wheel-equivalent planar segway PhysX models."""
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .physx_env import PhysXTrackingEnv
from .physx_model import segway_parameters


class PlanarTrackingEnv(PhysXTrackingEnv):
    model_version = 'physx-planar-v1'

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.segway = cfg.classic_task == 'classic-segway-v0'
        self.reference_height = .3 if self.segway else 1.5
        from ..classic.segway.env import ENV_CONFIG as SEGWAY
        from ..classic.cartpole.env import ENV_CONFIG as CARTPOLE
        self.initial_cfg = SEGWAY if self.segway else CARTPOLE
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        asset = Path(__file__).parent / 'assets' / ('segway.usda' if self.segway else 'cartpole.usda')
        cfg = ArticulationCfg(
            prim_path='/World/envs/env_.*/Robot',
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4, sleep_threshold=0., stabilization_threshold=0.)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={'.*': 0.}),
            actuators={'effort': ImplicitActuatorCfg(joint_names_expr=['.*'],
                                                     effort_limit_sim=1000., stiffness=0., damping=0.)})
        self._robot = Articulation(cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot'] = self._robot
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/Ground', ground)
        self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
        light = sim_utils.DomeLightCfg(intensity=2000.)
        light.func('/World/Light', light)
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path='/World/Reference', markers={'target': sim_utils.SphereCfg(
                radius=.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2,.9,.3)))}))

    def _joint_indices(self):
        self._cart = self._robot.find_joints('cart_slide')[0][0]
        self._pitch = self._robot.find_joints('body_pitch')[0][0]

    def get_physical_state(self):
        return torch.stack((self._robot.data.joint_pos[:, self._cart],
                            self._robot.data.joint_pos[:, self._pitch],
                            self._robot.data.joint_vel[:, self._cart],
                            self._robot.data.joint_vel[:, self._pitch]), -1)

    def _write_state(self, ids, x):
        q = self._robot.data.default_joint_pos[ids].clone()
        v = torch.zeros_like(q)
        q[:, self._cart], q[:, self._pitch] = x[:, 0], x[:, 1]
        v[:, self._cart], v[:, self._pitch] = x[:, 2], x[:, 3]
        self._robot.write_joint_state_to_sim(q, v, env_ids=ids)
        self._robot.reset(ids)

    def _apply_action(self):
        efforts = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        u = self._actions[:, 0]
        if self.segway:
            _, _, gain, radius, drag = segway_parameters()
            torque = gain * (u - drag*self.get_physical_state()[:, 2])
            efforts[:, self._cart] = torque / radius
            efforts[:, self._pitch] = -torque
        else:
            efforts[:, self._cart] = u
        efforts[:, self._cart] += self._gust[:, 2]
        efforts[:, self._pitch] += self._gust[:, 3]
        self._robot.set_joint_effort_target(efforts)

    def _build_references(self):
        self._joint_indices()
        ids = torch.arange(self.num_envs, device=self.device)
        self._write_state(ids, torch.zeros(self.num_envs, 4, device=self.device))
        # Both classic tasks have an equilibrium reference. The segway needs
        # nominal feedback to stop numerical upright drift over its 60 s horizon.
        K = torch.zeros(1, 4, device=self.device)
        if self.segway:
            _, _, _, _, drag = segway_parameters()
            B = np.array([[0.], [0.], [12.7/23.7], [-47.9/23.7]])
            A = np.zeros((4,4)); A[0,2] = A[1,3] = 1.
            A[2,1], A[3,1] = -9.8/23.7, 208.3/23.7
            A[:,2] -= B[:,0] * drag
            P = solve_continuous_are(A, B, np.diag([1.,20.,1.,1.]), np.eye(1))
            K = torch.tensor(B.T @ P, device=self.device, dtype=torch.float32)
        refs, controls = [self.get_physical_state().clone()], []
        for _ in range(self.max_episode_length):
            u = (-self.get_physical_state() @ K.T).clamp(-24.,24.)
            controls.append(u.clone())
            self._simulate_nominal(u)
            refs.append(self.get_physical_state().clone())
        self._refs = torch.stack(refs, 1)
        self._ref_controls = torch.stack(controls + [controls[-1]], 1)
        valid = torch.isfinite(self._refs).all(dim=(1,2))
        valid &= ((self._refs >= self.X_MIN+self.reference_margin) &
                  (self._refs <= self.X_MAX-self.reference_margin)).all(dim=(1,2))
        self._refs, self._ref_controls = self._refs[valid], self._ref_controls[valid]
        if not valid.any():
            raise RuntimeError('No in-domain articulated PhysX equilibrium reference')
        self._actions.zero_()
        print(f'[PhysX] Validated {len(self._refs)} articulated nominal references', flush=True)

    def _load_references(self, path):
        self._joint_indices()
        super()._load_references(path)

    def _sample_initial_state(self, reference):
        lo, hi = [reference.new_tensor(self.initial_cfg[key]) for key in ('x_init_min', 'x_init_max')]
        x = lo + torch.rand_like(reference)*(hi-lo)
        sign = torch.randint(0,2,(len(x),1),device=self.device)*2-1
        x[:, self.initial_cfg['x_init_sign_dims']] *= sign
        return x.clamp(self.X_MIN, self.X_MAX)

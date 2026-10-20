"""Local articulated cartpole and wheel-equivalent planar segway PhysX models."""
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .physx_env import PhysXTrackingEnv
from .physx_model import segway_parameters


class PlanarTrackingEnv(PhysXTrackingEnv):
    model_version = 'physx-planar-v2-shared-task'

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.segway = cfg.classic_task == 'classic-segway-v0'
        self.reference_height = .3 if self.segway else 1.5
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        task = 'segway' if self.segway else 'cartpole'
        asset = Path(__file__).parents[2] / f'isaac_{task}' / 'assets' / f'{task}.usda'
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

    def _initialize_body(self):
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
        self._robot.set_joint_effort_target(efforts)

    def _snapshot_simulation(self):
        return dict(joint_pos=self._robot.data.joint_pos.clone(), joint_vel=self._robot.data.joint_vel.clone(),
                    actions=self._actions.clone(), gust=self._gust.clone())

    def _restore_simulation(self, saved):
        self._robot.write_joint_state_to_sim(saved["joint_pos"], saved["joint_vel"])
        self._robot.reset()
        self._actions, self._gust = saved["actions"], saved["gust"]

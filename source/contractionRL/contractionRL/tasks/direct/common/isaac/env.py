"""Exact classic transitions, rewards and resets, rendered in an Isaac scene.

The markers carry no contact/rigid-body dynamics. The classic environment owns
the plant; PhysX advances the scene clock without changing the plant's state.
"""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from contractionRL.tasks.direct.common.isaac.physx_model import quadrotor_quat


def make_task_plant(cfg, *, transition_backend=None, defer_reset=False):
    if cfg.reference_mode != "stabilizing":
        raise ValueError("Paired Isaac tasks support reference_mode=stabilizing; contractive migration requires a pure classic task")
    task_kwargs = {key: getattr(cfg, key) for key in (
        "xe_init_min", "xe_init_max", "xe_init_sign_dims", "reference_feedback_gain", "ref_pool_size"
    ) if hasattr(cfg, key)}
    return gym.make(
        cfg.classic_task, num_envs=cfg.scene.num_envs, device=cfg.sim.device,
        reference_mode=cfg.reference_mode, compact_reference=True,
        disturbance_scales=(getattr(cfg, "disturbance_scales", None)
                            if getattr(cfg, "disturbance_enabled", True)
                            else [0.] * len(cfg.disturbance_scales)),
        reference_margin_fraction=getattr(cfg, "reference_margin_fraction", .05),
        transition_backend=transition_backend, defer_reset=defer_reset, **task_kwargs,
    ).unwrapped


class ClassicIsaacEnv(DirectRLEnv):
    has_analytical_dynamics = True

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.plant = make_task_plant(cfg)
        if cfg.decimation != 1 or abs(cfg.sim.dt - self.plant.dt) > 1e-12:
            raise ValueError("Isaac timing must match the classic plant (decimation=1).")
        if abs(cfg.episode_length_s - self.plant.time_bound) > 1e-12:
            raise ValueError("Isaac episode length must match the classic plant.")
        self.plant.configure_ref_window(length=self.plant.full_ref_length())
        cfg.observation_space = self.plant.observation_space
        cfg.action_space = self.plant.action_space
        super().__init__(cfg, render_mode, **kwargs)

    def __getattr__(self, name):
        plant = self.__dict__.get("plant")
        if plant is not None:
            return getattr(plant, name)
        raise AttributeError(name)

    def _setup_scene(self):
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/PlantMarkers",
            markers={
                "body": sim_utils.CuboidCfg(
                    size=(0.6, 0.3, 0.15),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0)),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.3)),
                ),
            },
        ))

    def configure_ref_window(self, length, offset=1, gamma=None):
        self.plant.configure_ref_window(length, offset, gamma)
        self.cfg.observation_space = self.plant.observation_space
        self._configure_gym_env_spaces()

    def configure_ref_preview(self, length, stride=1):
        self.plant.configure_ref_preview(length, stride)
        self.cfg.observation_space = self.plant.observation_space
        self._configure_gym_env_spaces()

    def configure_encoder_context(self, horizon):
        self.plant.configure_encoder_context(horizon)
        self.cfg.observation_space = self.plant.observation_space
        self._configure_gym_env_spaces()


    def get_tracking_error(self):
        plant = self.plant
        rows = torch.arange(self.num_envs, device=self.device)
        ref = plant.xref[rows, plant.time_steps.clamp(max=plant.xref.shape[1] - 1)]
        return torch.linalg.vector_norm(plant.wrap_angles(plant.x_t - ref), dim=-1)

    def get_physical_state(self):
        return self.plant.x_t.clone()

    def _draw(self):
        plant = self.plant
        rows = torch.arange(self.num_envs, device=self.device)
        ref = plant.xref[rows, plant.time_steps.clamp(max=plant.xref.shape[1] - 1)]
        if plant.task in ('cartpole', 'segway'):
            self._draw_planar(ref, rows)
            return
        positions = torch.zeros(2 * self.num_envs, 3, device=self.device)
        dim = 3 if plant.task == "quadrotor" else 2
        positions[:self.num_envs, :dim] = plant.x_t[:, :dim]
        positions[self.num_envs:, :dim] = ref[:, :dim]
        if dim == 2:
            positions[:, 2] = 0.2
        positions += self.scene.env_origins.repeat(2, 1)
        orientations = torch.zeros(2 * self.num_envs, 4, device=self.device)
        orientations[:, 0] = 1.0
        if dim == 2:
            yaw = plant.x_t[:, 2] / 2
            orientations[:self.num_envs, 0] = torch.cos(yaw)
            orientations[:self.num_envs, 3] = torch.sin(yaw)
        else:
            orientations[:self.num_envs] = quadrotor_quat(plant.x_t[:, 7:10])
        indices = torch.cat((torch.zeros_like(rows), torch.ones_like(rows)))
        self.markers.visualize(positions, orientations, marker_indices=indices)

    def _draw_planar(self, ref, rows):
        plant = self.plant
        down = plant.task == 'cartpole'
        pivot = 1.5 if down else .3
        sign = -1. if down else 1.
        positions = torch.zeros(4*self.num_envs, 3, device=self.device)
        positions[:self.num_envs, 0] = plant.x_t[:, 0]
        positions[self.num_envs:2*self.num_envs, 0] = ref[:, 0]
        positions[:2*self.num_envs, 2] = pivot
        for offset, state in [(2*self.num_envs, plant.x_t), (3*self.num_envs, ref)]:
            positions[offset:offset+self.num_envs, 0] = state[:, 0] + sign*torch.sin(state[:, 1])
            positions[offset:offset+self.num_envs, 2] = pivot + sign*torch.cos(state[:, 1])
        positions += self.scene.env_origins.repeat(4, 1)
        orientations = torch.zeros(4*self.num_envs, 4, device=self.device)
        orientations[:, 0] = 1.
        indices = torch.cat((torch.zeros_like(rows), torch.ones_like(rows), torch.zeros_like(rows), torch.ones_like(rows)))
        self.markers.visualize(positions, orientations, marker_indices=indices)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        obs, info = self.plant.reset(seed=seed, options=options)
        self.episode_length_buf.zero_()
        self._draw()
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        return {"policy": obs}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.plant.step(action)
        self.episode_length_buf.copy_(self.plant.time_steps)
        self.reset_terminated.copy_(terminated)
        self.reset_time_outs.copy_(truncated)
        self.common_step_counter += 1
        self._sim_step_counter += getattr(self.cfg, "decimation", 1)
        self._draw()
        self._advance_scene()
        return {"policy": obs}, reward, terminated, truncated, info

    def _advance_scene(self):
        self.sim.step(render=self.sim.has_gui() or self.sim.has_rtx_sensors())

    def _set_debug_vis_impl(self, debug_vis):
        self.markers.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        self._draw()

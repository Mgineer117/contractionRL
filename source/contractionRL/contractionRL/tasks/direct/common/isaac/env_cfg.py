"""Scene and reference defaults shared by paired tracking tasks."""

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    physx_reference_path: str = ""
    reference_mode: str = "stabilizing"
    disturbance_enabled: bool = True
    reference_margin_fraction: float = .05
    ref_pool_size: int = 64
    decimation = 1
    observation_space = 1
    state_space = 0
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=100.0, replicate_physics=False)
    viewer = ViewerCfg(eye=(30.0, 30.0, 30.0), lookat=(0.0, 0.0, 0.0))

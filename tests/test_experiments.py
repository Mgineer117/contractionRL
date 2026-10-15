"""Inventory and exact-plant bridge checks; no Isaac process or training launch."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import patch

import gymnasium as gym
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source/contractionRL"))
from list_experiments import environments, key_settings  # noqa: E402
import contractionRL.tasks.direct.classic  # noqa: E402,F401
from contractionRL.tasks.direct.classic.car import env as car_env  # noqa: E402,F401
from contractionRL.tasks.direct.classic.car_v1 import env as car_v1_env  # noqa: E402,F401
from contractionRL.tasks.direct.classic.quadrotor import env as quadrotor_env  # noqa: E402,F401
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper  # noqa: E402
from contractionRL.runners.contraction_runner import _resolve_dynamics_model  # noqa: E402


def test_catalog_and_config_aliases():
    rows = {row["id"]: row for row in environments()}
    assert "Cartpole-v0" not in rows
    assert len(rows) == 18
    for task, source in [("Isaac-Car-v0", "classic-car-v0"),
                         ("Isaac-Car-v1", "classic-car-v1"),
                         ("Isaac-Quadrotor-v0", "classic-quadrotor-v0")]:
        assert rows[task]["configs"] == rows[source]["configs"]
        assert len(rows[task]["configs"]) == 8
    for row in rows.values():
        for algo, path in row["configs"].items():
            assert path.is_file(), path
            assert key_settings(algo, path)
    car = rows["Isaac-Car-v1"]["configs"]
    assert car["ppo"].parent.parent.name == "car"
    assert car["c2rl-ppo"].parent.parent.name == "car_v1"
    assert key_settings("c2rl-ppo", car["c2rl-ppo"])["cmg_method"] == "cvstem"
    assert "Q_scaler" in key_settings("lqr", car["lqr"])
    import contractionRL.tasks.direct.classic_isaac  # noqa: F401
    assert "Cartpole-v0" not in gym.registry
    for task in ("Isaac-Car-v0", "Isaac-Car-v1", "Isaac-Quadrotor-v0"):
        assert "skrl_c2rl_ppo_cfg_entry_point" in gym.spec(task).kwargs


@pytest.fixture
def bridge_class():
    # Only scene/marker APIs are mocked. Transitions and rewards below use the
    # real classic plants, including out-of-box and horizon auto-resets.
    class FakeDirectEnv:
        @property
        def unwrapped(self):
            return self

        def __init__(self, cfg, render_mode=None, **kwargs):
            self.cfg = cfg
            self.device = cfg.sim.device
            self.num_envs = cfg.scene.num_envs
            self.scene = SimpleNamespace(env_origins=torch.zeros(self.num_envs, 3),
                                         clone_environments=lambda **kw: None)
            self.sim = SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: False,
                                       step=lambda **kw: None)
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
            self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
            self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
            self.common_step_counter = self._sim_step_counter = 0
            self._configure_gym_env_spaces()
            self._setup_scene()

        def _configure_gym_env_spaces(self):
            self.single_observation_space = {"policy": self.cfg.observation_space}
            self.single_action_space = self.cfg.action_space

        def seed(self, seed):
            torch.manual_seed(seed)

    class FakeMarkers:
        def __init__(self, cfg):
            pass

        def visualize(self, positions, orientations, marker_indices):
            self.positions = positions
            assert positions.shape[0] == orientations.shape[0] == marker_indices.shape[0]
            assert torch.isfinite(positions).all()
            torch.testing.assert_close(orientations.norm(dim=-1), torch.ones(len(orientations)))

    def cfg(**kwargs):
        return SimpleNamespace(func=lambda *args: None, **kwargs)

    modules = {name: ModuleType(name) for name in
               ("isaaclab", "isaaclab.envs", "isaaclab.sim", "isaaclab.markers", "isaaclab.utils", "isaaclab.utils.math")}
    modules["isaaclab"].__path__ = []
    modules["isaaclab.envs"].DirectRLEnv = FakeDirectEnv
    for name in ("DomeLightCfg", "CuboidCfg", "SphereCfg", "PreviewSurfaceCfg"):
        setattr(modules["isaaclab.sim"], name, cfg)
    modules["isaaclab.markers"].VisualizationMarkers = FakeMarkers
    modules["isaaclab.markers"].VisualizationMarkersCfg = cfg
    # Isaac's quaternion math itself is a pure tensor helper and can be imported
    # without starting Kit, but a unit quaternion suffices for this scene stub.
    modules["isaaclab.utils.math"].quat_from_euler_xyz = lambda x, y, z: torch.tensor([1., 0., 0., 0.]).repeat(len(x), 1)
    path = ROOT / "source/contractionRL/contractionRL/tasks/direct/classic_isaac/env.py"
    spec = importlib.util.spec_from_file_location("bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        yield module.ClassicIsaacEnv


@pytest.mark.parametrize("task", ["classic-car-v0", "classic-car-v1", "classic-quadrotor-v0"])
def test_exact_transitions_and_autoresets(bridge_class, task):
    import contractionRL.tasks.direct.classic  # noqa: F401
    reference = gym.make(task, num_envs=2, device="cpu").unwrapped
    cfg = SimpleNamespace(classic_task=task, reference_mode="stabilizing", decimation=1,
                          sim=SimpleNamespace(dt=reference.dt, device="cpu"),
                          scene=SimpleNamespace(num_envs=2), episode_length_s=reference.time_bound)
    bridge = bridge_class(cfg)
    reference.configure_ref_window(reference.full_ref_length())
    torch.manual_seed(7)
    obs, _ = bridge.reset()
    torch.manual_seed(7)
    expected, _ = reference.reset()
    for key in expected:
        torch.testing.assert_close(obs["policy"][key], expected[key], rtol=0, atol=0)
    for step in range(4):
        if step == 2:
            # Exercise a genuine horizon auto-reset, not only interior steps.
            bridge.plant.time_steps[:] = bridge.plant.episode_len - 1
            reference.time_steps[:] = reference.episode_len - 1
        action = torch.full((2, reference.num_dim_control), 0.05 * step)
        rng = torch.random.get_rng_state()
        actual = bridge.step(action)
        torch.random.set_rng_state(rng)
        wanted = reference.step(action)
        for key in wanted[0]:
            torch.testing.assert_close(actual[0]["policy"][key], wanted[0][key], rtol=0, atol=0)
        for a, b in zip(actual[1:4], wanted[1:4]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("tracking_error", "control_effort", "relative_tracking_error"):
            torch.testing.assert_close(actual[4][key], wanted[4][key], rtol=0, atol=0)
        torch.testing.assert_close(bridge.get_physical_state(), reference.x_t, rtol=0, atol=0)
    assert bridge.common_step_counter == 4
    bridge.configure_ref_window(3, offset=2)
    assert bridge.single_observation_space["policy"] == bridge.plant.observation_space
    torch.testing.assert_close(bridge.get_tracking_error(), torch.linalg.vector_norm(
        bridge.plant.wrap_angles(bridge.plant.x_t - bridge.plant.xref[:, 0]), dim=-1))


def test_timing_mismatch_is_rejected(bridge_class):
    import contractionRL.tasks.direct.classic  # noqa: F401
    cfg = SimpleNamespace(classic_task="classic-car-v0", reference_mode="stabilizing", decimation=2,
                          sim=SimpleNamespace(dt=0.03, device="cpu"),
                          scene=SimpleNamespace(num_envs=1), episode_length_s=15.0)
    with pytest.raises(ValueError, match="timing must match"):
        bridge_class(cfg)


def test_skrl_spaces_and_analytical_controller(bridge_class):
    cfg = SimpleNamespace(classic_task="classic-car-v0", reference_mode="stabilizing", decimation=1,
                          sim=SimpleNamespace(dt=0.03, device="cpu"),
                          scene=SimpleNamespace(num_envs=2), episode_length_s=15.0)
    bridge = bridge_class(cfg)
    bridge.configure_ref_window(3)
    wrapped = IsaacLabWrapper(bridge)
    obs, _ = wrapped.reset()
    assert obs.shape == (2, bridge.plant.ref_window.flat_dim)
    next_obs, reward, terminated, truncated, _ = wrapped.step(torch.zeros(2, 2))
    assert next_obs.shape == obs.shape
    assert reward.shape == terminated.shape == truncated.shape == (2, 1)
    assert _resolve_dynamics_model("lqr", {}, bridge.has_analytical_dynamics, "cpu") is None
    with pytest.raises(ValueError, match="dynamics_checkpoint"):
        _resolve_dynamics_model("lqr", {}, False, "cpu")

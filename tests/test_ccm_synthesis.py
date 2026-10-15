"""Check CCM residuals against derivatives and exact bounded-input minimization."""

from itertools import product
from pathlib import Path
import sys

import gymnasium as gym
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/contractionRL"))
from contractionRL.agents.skrl.ccm_synthesis import ccm_residuals, ccm_loss, input_box_minimum, train_cmg_ccm
from contractionRL.agents.skrl.c2rl import C2RLPPOCfg, C2RLSkrlTrainer
from contractionRL.agents.skrl.math_utils import quadratic_energy, energy_decrease
from contractionRL.agents.skrl.nn_modules import BoundedCCM_Generator
from contractionRL.runners import ContractionRunner
import contractionRL.tasks.direct.classic  # noqa: F401


class SmoothMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def forward(self, x):
        a = 2 + self.scale * x[:, 0].square()
        d = 3 + self.scale * x[:, 1].square()
        c = self.scale * torch.sin(x[:, 0])
        return torch.stack((a, c, c, d), -1).reshape(-1, 2, 2), {}


def dynamics(x):
    f = torch.stack((x[:, 1], -x[:, 0] - 0.3 * x[:, 1]), -1)
    B = torch.stack((1.5 + 0.5 * torch.sin(x[:, 0]), 1 + 0.2 * x[:, 1]), -1).unsqueeze(-1)
    return f, B, None


def test_box_minimum_matches_all_vertices():
    torch.manual_seed(3)
    a, b = torch.randn(5, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    low, high = torch.tensor([-2., -1., 0.]), torch.tensor([1., 3., 4.])
    vertices = torch.tensor(list(product(*zip(low.tolist(), high.tolist()))), dtype=torch.float64)
    brute = (a[:, None] + b @ vertices.T).min(-1).values
    torch.testing.assert_close(input_box_minimum(a, b, low, high), brute)
    torch.testing.assert_close(input_box_minimum(a, b, -high, high), a - (b.abs() * high).sum(-1))


def test_energy_derivative_with_moving_reference_and_state_dependent_B():
    metric = SmoothMetric()
    x = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float64)
    ref = torch.tensor([[0.1, 0.2], [0.2, -0.1]], dtype=torch.float64)
    uref, u = torch.tensor([[0.2], [-0.3]], dtype=torch.float64), torch.tensor([[0.6], [-0.5]], dtype=torch.float64)
    residuals = ccm_residuals(metric, dynamics, x, ref, uref, -torch.ones(1), torch.ones(1), lbd=0.2)
    f, B, _ = dynamics(x)
    fr, Br, _ = dynamics(ref)
    v = f + (B @ u.unsqueeze(-1)).squeeze(-1)
    vr = fr + (Br @ uref.unsqueeze(-1)).squeeze(-1)
    h = 1e-5
    def energy(signed_h):
        point, reference = x + signed_h * v, ref + signed_h * vr
        return quadratic_energy(point - reference, metric(point)[0])
    derivative = (energy(h) - energy(-h)) / (2 * h)
    torch.testing.assert_close(residuals["a"] + (residuals["b"] * u).sum(-1), derivative, rtol=1e-7, atol=1e-8)
    orthogonality = B.mT @ metric(x)[0] @ residuals["N"]
    torch.testing.assert_close(orthogonality, torch.zeros_like(orthogonality), rtol=0, atol=1e-12)
    same = ccm_residuals(metric, dynamics, x, x, uref, -torch.ones(1), torch.ones(1), lbd=0.2)
    torch.testing.assert_close(same["energy"], torch.zeros(2, dtype=x.dtype))
    torch.testing.assert_close(same["feasibility"], torch.zeros(2, dtype=x.dtype))


def test_full_killing_condition_and_backpropagation():
    metric = SmoothMetric()
    x = torch.tensor([[0.3, 0.2], [0.2, 0.1]], dtype=torch.float64)
    residuals = ccm_residuals(metric, dynamics, x, x * 0.5, torch.zeros(2, 1),
                              -torch.ones(1), torch.ones(1), lbd=0.2)
    loss, stats = ccm_loss(residuals, eps=0.01)
    loss.backward()
    assert torch.isfinite(metric.scale.grad) and metric.scale.grad.abs() > 0
    assert set(stats) == {"loss", "killing_loss", "contraction_loss", "feasibility_loss",
                          "killing_max", "contraction_max", "feasibility_max"}
    assert "controller_loss" not in stats
    torch.testing.assert_close(loss.detach(), torch.tensor(
        stats["killing_loss"] + stats["contraction_loss"] + stats["feasibility_loss"], dtype=x.dtype))


def test_fully_actuated_integrator_input_feasibility():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return (1 + 0 * x).unsqueeze(-1), {}
    def integrator(x):
        return x * 0, torch.ones(len(x), 1, 1, dtype=x.dtype), None
    x = torch.tensor([[1.], [3.]], dtype=torch.float64)
    result = ccm_residuals(Identity(), integrator, x, x * 0, x * 0,
                           torch.tensor([-1.]), torch.tensor([1.]), lbd=0.5)
    torch.testing.assert_close(result["feasibility"], torch.tensor([-1., 3.], dtype=x.dtype))
    assert result["projected"].shape == (2, 0, 0)
    _, stats = ccm_loss(result, eps=0.01)
    assert stats["contraction_loss"] == 0
    assert stats["feasibility_loss"] > 0


def test_reward_uses_energy_decrease_with_both_metric_evaluations():
    metric = SmoothMetric()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    nxt = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    ref, next_ref = torch.tensor([[0.1, 0.1]]), torch.tensor([[0.12, 0.08]])
    E = quadratic_energy(x - ref, metric(x)[0])
    next_E = quadratic_energy(nxt - next_ref, metric(nxt)[0])
    torch.testing.assert_close(energy_decrease(E, next_E), -(next_E - E))
    assert energy_decrease(E, next_E).item() > 0


def test_ccm_minibatch_training_and_invalid_box():
    torch.manual_seed(4)
    metric = BoundedCCM_Generator(2, [4], w_lb=0.5, w_ub=2.0, outputs_metric=True)
    x = torch.randn(6, 2) * 0.2
    data = {"x": x, "xref": x * 0.5, "uref": torch.zeros(6, 1)}
    cfg = C2RLPPOCfg(cmg_method="ccm", cmg_ccm_epochs=1, cmg_ccm_batch_size=2,
                     cmg_val_frac=0.33, w_lb=0.5, w_ub=2.0)
    result = train_cmg_ccm(metric, dynamics, data, [-1.], [1.], cfg=cfg, device="cpu")
    assert result["method"] == "ccm" and result["epochs"] == 1
    assert result["training_samples"] + result["validation_samples"] == 6
    assert all(torch.isfinite(torch.tensor(v)) for v in result["validation"].values())
    with pytest.raises(ValueError, match="finite"):
        train_cmg_ccm(metric, dynamics, data, [-float("inf")], [1.], cfg=cfg, device="cpu")


def small_ccm_runner(algorithm="ppo", **overrides):
    env = gym.make("classic-car-v0", num_envs=1, device="cpu").unwrapped
    env.configure_ref_window(2)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / f"source/contractionRL/contractionRL/tasks/direct/classic/car/agents/skrl_c2rl_{algorithm}_cfg.yaml").read_text())
    cfg["memory"]["memory_size"] = -1 if algorithm == "ppo" else 8
    cfg["agent"].update(cmg_method="ccm", rollouts=2)
    if algorithm == "ppo":
        cfg["agent"]["mini_batches"] = 1
    cfg["agent"].update(overrides)
    cfg["agent"]["experiment"]["wandb"] = False
    cfg["models"]["policy"].update(encoder="mlp", encoder_stride=1)
    runner = ContractionRunner(env, cfg, is_classic=True)
    return runner, env


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_runner_passes_applied_bounds_and_dispatches_ccm(monkeypatch, algorithm):
    runner, env = small_ccm_runner(algorithm)
    agent = runner._agent
    torch.testing.assert_close(torch.as_tensor(agent._u_min), env.U_MIN)
    torch.testing.assert_close(torch.as_tensor(agent._u_max), env.U_MAX)
    assert agent._cfg.cmg_method == "ccm"
    monkeypatch.setattr(agent, "_synthesize_cmg_cvstem", lambda **kw: pytest.fail("CCM must not load the CV-STEM cache"))
    monkeypatch.setattr(agent, "_synthesize_cmg_ccm", lambda **kw: {"method": "ccm"})
    monkeypatch.setattr(agent, "_attach_critic_potential", lambda: None)
    assert agent.synthesize_cmg()["method"] == "ccm"


def test_ccm_phase_a_freezes_metric_and_preserves_policy(tmp_path):
    runner, env = small_ccm_runner()
    agent = runner._agent
    agent._cfg.cmg_memory_size = 4
    agent._cfg.cmg_ccm_epochs = 1
    agent._cfg.cmg_ccm_batch_size = 2
    agent.experiment_dir = str(tmp_path)
    policy_before = {key: value.clone() for key, value in agent._policy_model.state_dict().items()}
    result = agent.synthesize_cmg()
    assert result["method"] == "ccm" and result["closed_loop_loss"] is False
    assert all(not param.requires_grad for param in agent._ccm_gen.parameters())
    for key, value in agent._policy_model.state_dict().items():
        torch.testing.assert_close(value, policy_before[key], rtol=0, atol=0)
    assert (tmp_path / "ccm_synthesis.json").is_file()
    # The frozen metric is the same object deployed by the reward path.
    C2RLSkrlTrainer._inject_ccm(env, agent)
    env.reset()
    assert env.ccm_gen is not None


@pytest.mark.parametrize("overrides", [{"reward_level": True}, {"reward_euclidean": True},
                                      {"control_scaler": 0.1}, {"tracking_scaler": 2.0}])
def test_c2rl_rejects_other_reward_forms(overrides):
    with pytest.raises(ValueError, match="reward is exactly"):
        small_ccm_runner(**overrides)


def test_actual_environment_reward_matches_moving_reference_energy():
    env = gym.make("classic-car-v0", num_envs=2, device="cpu").unwrapped
    metric = BoundedCCM_Generator(4, [4], w_lb=0.5, w_ub=2.0, outputs_metric=True)
    env.set_ccm(metric, w_lb=0.5, device="cpu", tracking_scaler=1., control_scaler=0.)
    env.reset()
    old_x = env.x_t.clone()
    old_error = env.wrap_angles(old_x - env.xref[:, 0])
    old_M = metric(old_x)[0]
    _, reward, terminated, truncated, _ = env.step(torch.zeros(2, 2))
    assert not (terminated | truncated).any()
    new_error = env.wrap_angles(env.x_t - env.xref[:, 1])
    new_M = metric(env.x_t)[0]
    expected = -(quadratic_energy(new_error, new_M) - quadratic_energy(old_error, old_M))
    torch.testing.assert_close(reward, expected)
    raw_decrease = -(quadratic_energy(env.x_t, new_M) - quadratic_energy(old_x, old_M))
    assert not torch.allclose(reward, raw_decrease)

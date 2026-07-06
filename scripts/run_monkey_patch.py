import sys
import gymnasium as gym
import contractionRL.tasks.direct.classic
from gymnasium.vector import SyncVectorEnv
from skrl.envs.wrappers.torch import wrap_env
from contractionRL.agents.skrl.runner import CLActorRunner as Runner
import yaml
import torch

with open("source/contractionRL/contractionRL/tasks/direct/classic/car/agents/skrl_ppo_cfg.yaml") as f:
    agent_cfg = yaml.safe_load(f)

agent_cfg["seed"] = 42

vec_env = SyncVectorEnv([lambda: gym.make("classic-car-v0")] * 4)
env = wrap_env(vec_env, wrapper="gymnasium")

_a = agent_cfg["agent"]
_a.pop("std_dev_annealing", None)
_a.pop("std_dev_annealing_kwargs", None)

runner = Runner(env, agent_cfg)

# Monkey-patch memory.add_samples
orig_add_samples = runner.agent.memory.add_samples
def add_samples_patch(**kwargs):
    for k, v in kwargs.items():
        if v is not None:
            expected_shape = runner.agent.memory.tensors[k][0:4].shape
            if expected_shape != v.shape:
                print(f"SHAPE MISMATCH! name: {k} | expected: {expected_shape} | actual: {v.shape}")
                print(f"v: {v}")
    orig_add_samples(**kwargs)

runner.agent.memory.add_samples = add_samples_patch

runner.run()

import sys
import yaml
from contractionRL.tasks.direct.classic.cartpole.env import CartPoleEnv

env = CartPoleEnv()
print("BaseEnv module:", type(env).__module__)
if hasattr(env, "unwrapped"):
    print("BaseEnv unwrapped module:", type(env.unwrapped).__module__)

from skrl.envs.wrappers.torch import wrap_env
skrl_env = wrap_env(env, wrapper="gym")
print("Skrl wrapped env module:", type(skrl_env).__module__)
if hasattr(skrl_env, "unwrapped"):
    print("Skrl wrapped env unwrapped module:", type(skrl_env.unwrapped).__module__)
else:
    print("Skrl wrapped env has no unwrapped property")

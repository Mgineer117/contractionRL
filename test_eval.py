import torch
import sys
import yaml
import os
import copy
import numpy as np

# Load train.py to reuse its setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.skrl.train import args_cli

# We just want to mock the args to run Quadruped-VelTracking-v0
args_cli.task = "Quadruped-VelTracking-v0"
args_cli.algorithm = "ppo"
args_cli.num_envs = 10
args_cli.headless = True
args_cli.ml_framework = "torch"
args_cli.ref_num_trajs = 0
args_cli.min_ref_quality = 0

from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
from isaaclab.app import AppLauncher
# Actually, I can't easily launch isaaclab from python like this if it's not the main script.
# I will just grep the wandb logs or the stdout!

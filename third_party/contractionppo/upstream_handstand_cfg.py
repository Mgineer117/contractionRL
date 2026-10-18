# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ContractionPPO.contraction_ppo.config import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlContractionPpoAlgorithmCfg
from ContractionPPO.contraction_ppo.config.contraction_cfg import ContractionCfg as RslRlContractionCfg

@configclass
class UnitreeA1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "unitree_go1_boatdeck_handstand_cont"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlContractionPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        contraction_cfg=RslRlContractionCfg(
            alpha=0.1,                          # Contraction rate parameter
            contraction_logging_scale=1.0,
            penalty_coef=0.005,                  # Penalty coefficient
            # penalty_coef=0.0,
            contraction_metric_learning_rate=1e-3,
            contraction_constraint_coef=100.0,  # Constraint coefficient
            constraint_dim=2,                   # Dimension of constraint (obs - des) - XY position
            action_dim=12,                      # Dimension of robot state/action (joint positions)
            hidden_dims=[128, 64],             # Hidden layers for contraction metric MLP
            activation="elu",                   # Activation function
            epsilon_contraction=1e-3,           # Small positive constant for strict contraction
            lyapunov_regularization=1e-4,       # Regularization for metric stability
        )
    )

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contraction Theory-based Proximal Policy Optimization (PPO) algorithm.

This module extends the standard PPO algorithm with contraction theory guarantees
using Lyapunov functions and contraction metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from ContractionPPO.contraction_ppo.config.contraction_cfg import ContractionCfg
from ContractionPPO.contraction_ppo.algorithms.ppo import PPO
from ContractionPPO.contraction_ppo.modules import ContractionMetricNetwork
from ContractionPPO.contraction_ppo.storage import RolloutStorage

def compute_quaternion_derivative(q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Computes the time derivative of a quaternion given the body's angular velocity.

    This function is designed to work with batches of data.

    Args:
        q (torch.Tensor): A batch of current orientation quaternions.
                          The format is assumed to be [w, x, y, z].
                          Shape: (N, 4), where N is the batch size.
        omega (torch.Tensor): A batch of angular velocities in the body frame.
                              The format is [wx, wy, wz].
                              Shape: (N, 3).

    Returns:
        torch.Tensor: The time derivative of each quaternion in the batch (q_dot).
                      Shape: (N, 4).
    """
    # Ensure inputs are on the same device
    if q.device != omega.device:
        omega = omega.to(q.device)

    # Get the batch size
    N = q.shape[0]

    # Unpack angular velocities for clarity
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]

    # Construct the 4x4 Omega matrix for the entire batch.
    # This avoids loops and is highly efficient on GPU.
    zeros = torch.zeros(N, device=q.device)
    Omega = torch.stack([
        torch.stack([zeros, -wx,   -wy,   -wz],   dim=1),
        torch.stack([wx,    zeros,  wz,   -wy],   dim=1),
        torch.stack([wy,    -wz,   zeros,  wx],   dim=1),
        torch.stack([wz,    wy,    -wx,   zeros], dim=1)
    ], dim=1)

    # Reshape the quaternion tensor for batch matrix-vector multiplication.
    # The shape needs to change from (N, 4) to (N, 4, 1).
    q_reshaped = q.unsqueeze(-1)

    # Perform the core calculation: q_dot = 0.5 * Omega @ q
    # torch.bmm is used for batch matrix multiplication.
    q_dot_reshaped = 0.5 * torch.bmm(Omega, q_reshaped)

    # Squeeze the last dimension to return to the standard (N, 4) shape.
    return q_dot_reshaped.squeeze(-1)

class ContractionPPO(PPO):
    """
    Contraction Theory-based PPO algorithm.
    
    This algorithm extends PPO with contraction theory guarantees by:
    1. Learning a contraction metric M(x) via an MLP that outputs Θ such that M = ΘΘᵀ
    2. Computing Lyapunov function V = x^T M(x) x where x = [c; state - output_ppo]
    3. Enforcing the contraction condition V̇ < -αV through a penalty term
    4. Two-phase training: first learn the metric, then integrate with PPO
    """

    def __init__(
        self,
        policy,
        # Standard PPO parameters
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Contraction theory parameters
        contraction_cfg: dict | None = None,
    ):
        # Initialize base PPO
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        if contraction_cfg is None:
            raise ValueError("contraction_cfg is required")
        self.contraction_cfg = contraction_cfg
        
        self.alpha = self.contraction_cfg['alpha']
        self.contraction_logging_scale = self.contraction_cfg['contraction_logging_scale']
        self.contraction_penalty_coef = self.contraction_cfg['penalty_coef']

        constraint_dim = self.contraction_cfg['constraint_dim']
        action_dim = self.contraction_cfg['action_dim']
        # For the new formulation, the input to the MLP is x = [c; e] where:
        # c has dimension constraint_dim and e has dimension equal to action space
        mlp_input_dim = constraint_dim
        
        hidden_dims = self.contraction_cfg['hidden_dims']
        activation = self.contraction_cfg['activation']

        self.contraction_metric = ContractionMetricNetwork(
            input_dim=mlp_input_dim,
            constraint_dim=mlp_input_dim,  # Output dimension matches input for x^T M(x) x
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(self.device)

        self.contraction_metric_optimizer = optim.Adam(self.contraction_metric.parameters(), lr=self.contraction_cfg['contraction_metric_learning_rate'])
            
        # Training phase tracking
        self.total_iterations = 0
        self.metric_learning_iterations = 0
        
        # Storage for previous robot state and policy output (for derivative computation)
        self.prev_robot_state = None
        self.prev_policy_output = None
        
    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        """Initialize storage with contraction-specific fields."""
        contraction_obs_shape = (self.contraction_cfg['constraint_dim'],)
        # For the new formulation, robot_state_shape will be determined by the environment
        # For now, we'll set it to match the action space (e.g., joint positions)
        robot_state_shape = actions_shape  # This will be updated based on actual robot state
       
        # Determine RND state shape
        rnd_state_shape = [self.rnd.num_states] if self.rnd else None
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        
        # Initialize storage with contraction shapes
        from ContractionPPO.contraction_ppo.storage import RolloutStorage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape=rnd_state_shape,
            contraction_obs_shape=contraction_obs_shape,
            robot_state_shape=robot_state_shape,
            device=self.device,
        )
    
    def act(self, obs, critic_obs, contraction_data: Dict[str, torch.Tensor]):
        """
        Compute actions with contraction data collection.
        
        Args:
            obs: Policy observations
            critic_obs: Critic observations  
            contraction_data: Dictionary containing:
                - 'contraction_obs': Current constraint observations
                - 'contraction_des': Desired constraint values
                - 'contraction_c_dot': Constraint dynamics
                - 'robot_state': Current robot state (e.g., joint positions)
        """
        # Standard PPO action computation
        actions = super().act(obs, critic_obs)

        # Store constraint data
        if contraction_data is not None:
            if 'contraction_obs' in contraction_data and contraction_data['contraction_obs'] is not None:
                self.transition.contraction_obs = self._normalize_contraction_tensor(contraction_data['contraction_obs'].detach())
            if 'contraction_des' in contraction_data and contraction_data['contraction_des'] is not None:
                self.transition.contraction_des = self._normalize_contraction_tensor(contraction_data['contraction_des'].detach())
            if 'contraction_c_dot' in contraction_data and contraction_data['contraction_c_dot'] is not None:
                self.transition.contraction_c_dot = self._normalize_contraction_tensor(contraction_data['contraction_c_dot'].detach())
            
            # Store robot state and policy output for the new Lyapunov function
            current_robot_state = None
            if 'robot_state' in contraction_data and contraction_data['robot_state'] is not None:
                current_robot_state = contraction_data['robot_state'].detach()
                self.transition.robot_state = current_robot_state
            
            # Store policy output (actions)
            current_policy_output = None
            if actions is not None:
                current_policy_output = actions.detach()
                self.transition.policy_output = current_policy_output
            
            # Store previous values for derivative computation
            if self.prev_robot_state is not None:
                self.transition.prev_robot_state = self.prev_robot_state.clone()
            else:
                # First step - use current values as previous (no derivative for first step)
                if current_robot_state is not None:
                    self.transition.prev_robot_state = current_robot_state.clone()
                    
            if self.prev_policy_output is not None:
                self.transition.prev_policy_output = self.prev_policy_output.clone()
            else:
                # First step - use current values as previous
                if current_policy_output is not None:
                    self.transition.prev_policy_output = current_policy_output.clone()
            
            # Update stored previous values for next iteration
            self.prev_robot_state = current_robot_state
            self.prev_policy_output = current_policy_output
            
        return actions
    
    def _normalize_contraction_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize contraction tensor to have shape [batch_size, constraint_dim].
        
        Args:
            tensor: Input tensor with potentially extra dimensions
            
        Returns:
            Normalized tensor with shape [batch_size, constraint_dim]
        """
        if tensor.dim() == 2:
            return tensor
        elif tensor.dim() == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        else:
            return tensor.view(tensor.shape[0], -1)
    
    def compute_lyapunov_derivative(self, contraction_obs_batch: torch.Tensor, contraction_des_batch: torch.Tensor, 
                                    contraction_c_dot_batch: torch.Tensor, robot_state_batch: torch.Tensor, 
                                    policy_output_batch: torch.Tensor, prev_robot_state_batch: torch.Tensor, 
                                    prev_policy_output_batch: torch.Tensor, dt: float = 0.04) -> torch.Tensor:
        """
        Compute proper Lyapunov derivative V̇ = ∂V/∂x * ẋ using new formulation.
        
        New formulation: V = x^T M(x) x where x = [c; state - output_ppo]
        Derivative: V̇ = ∂V/∂x * ẋ where ẋ = [ċ; (e_{t+1} - e_t)/Δt]
        
        Args:
            contraction_obs_batch: Constraint observations [batch_size, constraint_dim] 
            contraction_des_batch: Desired constraint values [batch_size, constraint_dim]
            contraction_c_dot_batch: Constraint dynamics [batch_size, constraint_dim]
            robot_state_batch: Current robot state [batch_size, state_dim]
            policy_output_batch: Current policy output [batch_size, action_dim]
            prev_robot_state_batch: Previous robot state [batch_size, state_dim]
            prev_policy_output_batch: Previous policy output [batch_size, action_dim]
            dt: Time step for finite difference
            
        Returns:
            Lyapunov derivative V̇ [batch_size]
        """
        with torch.enable_grad():
            # Compute constraint violation c = obs - des
            c = (contraction_obs_batch - contraction_des_batch) * self.contraction_cfg['contraction_constraint_coef']  # [batch_size, constraint_dim]
            
            # Compute policy error: e = state - output_ppo
            e_current = robot_state_batch - policy_output_batch  # [batch_size, action_dim]
            e_previous = prev_robot_state_batch - prev_policy_output_batch  # [batch_size, action_dim]
            
            # Compute time derivative of policy error using finite difference
            e_dot = (e_current - e_previous) / dt  # [batch_size, action_dim]
            
            # Construct the new state vector x = [c; e]
            # x = torch.cat([c, e], dim=-1)  # [batch_size, constraint_dim + action_dim]
            x = c.clone().requires_grad_(True)
            # x = x.clone().requires_grad_(True)
            
            # Compute the contraction metric M(x)
            M = self.contraction_metric(x)  # [batch_size, x_dim, x_dim]
            
            # Compute Lyapunov function: V = x^T M(x) x
            V = torch.bmm(x.unsqueeze(-2), torch.bmm(M, x.unsqueeze(-1))).squeeze(-1).squeeze(-1)

            # Compute gradient ∂V/∂x
            grad_V_x = torch.autograd.grad(
                outputs=V.sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]  # Shape: [batch_size, x_dim]
            
            contraction_c_dot_batch *= self.contraction_cfg['contraction_constraint_coef']
            
            # Construct time derivative vector x_dot = [c_dot; e_dot]
            # x_dot = torch.cat([contraction_c_dot_batch, e_dot], dim=-1)  # [batch_size, x_dim]
            x_dot = contraction_c_dot_batch.clone().requires_grad_(True)
            
            # Compute V̇ = ∂V/∂x * ẋ
            V_dot = torch.sum(grad_V_x * x_dot, dim=-1)  # [batch_size]
        
        return V_dot
    
    def compute_contraction_penalty(self, lyapunov_values: torch.Tensor, lyapunov_derivatives: torch.Tensor) -> torch.Tensor:
        """
        Compute the contraction penalty term for the condition V̇ + αV ≤ -ε.
        
        THEORETICAL FIX: For contraction theory, we need V̇ + αV ≤ -ε for convergence guarantees.
        
        Args:
            lyapunov_values: Current Lyapunov values V [batch_size]
            lyapunov_derivatives: Time derivatives V̇ [batch_size]
            
        Returns:
            Contraction penalty [batch_size]
        """
        # Contraction condition: V̇ + αV ≤ -ε (rearranged: V̇ + αV + ε ≤ 0)
        contraction_violation = (lyapunov_derivatives + self.alpha * lyapunov_values) + self.contraction_cfg['epsilon_contraction']
        
        # Use ReLU to penalize violations (when V̇ + αV + ε > 0)
        # This ensures we enforce the strict inequality V̇ + αV ≤ -ε
        penalty = torch.nn.functional.relu(contraction_violation)
        
        return penalty
    
    def update_contraction_metric(self, contraction_obs_batch: torch.Tensor | None, contraction_des_batch: torch.Tensor | None, 
                                contraction_c_dot_batch: torch.Tensor | None, robot_state_batch: torch.Tensor | None, 
                                policy_output_batch: torch.Tensor | None, prev_robot_state_batch: torch.Tensor | None,
                                prev_policy_output_batch: torch.Tensor | None) -> torch.Tensor:
        """
        Update the contraction metric network using SGD on the contraction penalty.
        
        Args:
            contraction_obs_batch: Current constraint observations
            contraction_des_batch: Desired constraint values
            contraction_c_dot_batch: Constraint dynamics
            robot_state_batch: Current robot state
            policy_output_batch: Current policy output
            prev_robot_state_batch: Previous robot state for derivative computation
            prev_policy_output_batch: Previous policy output for derivative computation
        Returns:
            Contraction loss value
        """
        # PERFORMANCE FIX: Detach input tensors to prevent gradients from flowing back
        contraction_obs_batch = contraction_obs_batch.detach() if contraction_obs_batch is not None else None
        contraction_des_batch = contraction_des_batch.detach() if contraction_des_batch is not None else None
        contraction_c_dot_batch = contraction_c_dot_batch.detach() if contraction_c_dot_batch is not None else None
        robot_state_batch = robot_state_batch.detach() if robot_state_batch is not None else None
        policy_output_batch = policy_output_batch.detach() if policy_output_batch is not None else None
        prev_robot_state_batch = prev_robot_state_batch.detach() if prev_robot_state_batch is not None else None
        prev_policy_output_batch = prev_policy_output_batch.detach() if prev_policy_output_batch is not None else None
        
        if contraction_obs_batch is None or robot_state_batch is None:
            print("Warning: Missing essential contraction data (contraction_obs or robot_state), returning zero loss")
            return torch.tensor(0.0, device=self.device)
        
        # Ensure both tensors have the correct shape before computing constraint vector
        if contraction_obs_batch is not None:
                contraction_obs_batch = self._normalize_contraction_tensor(contraction_obs_batch)
        if contraction_des_batch is not None:
                contraction_des_batch = self._normalize_contraction_tensor(contraction_des_batch)
        
        # Compute constraint vector c = obs - des
        if contraction_obs_batch is not None and contraction_des_batch is not None:
                c = (contraction_obs_batch - contraction_des_batch) * self.contraction_cfg['contraction_constraint_coef']
        else:
            # If we don't have proper constraint data, return zero loss
            print("Warning: Missing contraction_obs_batch or contraction_des_batch for constraint computation. Returning zero loss.")
            return torch.tensor(0.0, device=self.device)
        
        # Compute policy error e = state - output_ppo
        if robot_state_batch is not None and policy_output_batch is not None:
                e = robot_state_batch - policy_output_batch
        else:
            print("Warning: Missing robot_state_batch or policy_output_batch for policy error computation. Returning zero loss.")
            return torch.tensor(0.0, device=self.device)
        
        # Construct the new state vector x = [c; e]
        # x = torch.cat([c, e], dim=-1)  # [batch_size, constraint_dim + action_dim]
        x = c.clone().requires_grad_(True)
        
        # Compute contraction metric M(x)
        M = self.contraction_metric(x)
        
        # Compute Lyapunov function: V = x^T M(x) x
        lyapunov_values = torch.bmm(x.unsqueeze(-2), torch.bmm(M, x.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        
        # Compute time derivative using true previous values
        if (contraction_c_dot_batch is not None and robot_state_batch is not None and 
            policy_output_batch is not None and prev_robot_state_batch is not None and prev_policy_output_batch is not None):
            lyapunov_derivatives = self.compute_lyapunov_derivative(
                contraction_obs_batch, contraction_des_batch, contraction_c_dot_batch,
                robot_state_batch, policy_output_batch, prev_robot_state_batch, prev_policy_output_batch
            )
        else:
            missing_data = []
            if contraction_c_dot_batch is None:
                missing_data.append("contraction_c_dot_batch")
            if robot_state_batch is None:
                missing_data.append("robot_state_batch")
            if policy_output_batch is None:
                missing_data.append("policy_output_batch")
            if prev_robot_state_batch is None:
                missing_data.append("prev_robot_state_batch")
            if prev_policy_output_batch is None:
                missing_data.append("prev_policy_output_batch")
            
            print(f"Warning: Missing data for Lyapunov derivative computation: {missing_data}. Using zero derivatives.")
            lyapunov_derivatives = torch.zeros_like(lyapunov_values)
        
        # Compute contraction penalty
        contraction_penalty = self.compute_contraction_penalty(lyapunov_values, lyapunov_derivatives)
        
        # Loss is the mean contraction penalty
        contraction_loss = contraction_penalty.mean()
        
        # Note: we do not perform a separate optimizer step here. The gradients
        # computed from this loss will be applied during the main PPO optimizer
        # step because the contraction metric parameters were added to the same
        # optimizer. We therefore simply return the tensor so it can be added to
        # the overall loss.

        # Update contraction metric
        self.contraction_metric_optimizer.zero_grad()
        contraction_loss.backward()
        self.contraction_metric_optimizer.step()

        return contraction_loss
    
    def update(self):
        """
        Update method with contraction theory integration.
        
        This method implements a two-phase training approach:
        
        Phase 1 - Metric Learning (first 80% of training):
        - Learn the contraction metric M(o) from environment data
        - Continue normal PPO learning with standard rewards
        - No contraction penalties applied to PPO loss
        
        Phase 2 - Integrated Training (last 20% of training):
        - Continue learning the contraction metric
        - Add contraction penalties to PPO loss to enforce contraction condition
        - Full contraction-aware policy optimization
        """
        self.total_iterations += 1
        
        # Standard PPO losses
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_contraction_metric_loss = 0
        mean_contraction_c_error = 0
        
        # RND and symmetry losses (from parent class)
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        
        # Get mini-batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # Iterate over mini-batches
        for batch_data in generator:
            # Unpack batch data (including contraction data if available)
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, 
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
             hid_states_batch, masks_batch, rnd_state_batch) = batch_data[:12]
            
            # Extract contraction data if available
            contraction_data_available = len(batch_data) > 12
            if contraction_data_available:
                if len(batch_data) >= 19:  # Updated to handle previous values
                    (contraction_obs_batch, contraction_des_batch, contraction_c_dot_batch,
                     robot_state_batch, policy_output_batch, prev_robot_state_batch, prev_policy_output_batch) = batch_data[12:19]
                else:
                    print("Warning: Expected 19 elements in batch_data for contraction PPO with previous values, got", len(batch_data))
                    # Fallback to old format without previous values
                    (contraction_obs_batch, contraction_des_batch, contraction_c_dot_batch,
                     robot_state_batch, policy_output_batch) = batch_data[12:17]
                    prev_robot_state_batch = None
                    prev_policy_output_batch = None
                
                # PERFORMANCE FIX: Detach contraction data early
                if contraction_obs_batch is not None:
                    contraction_obs_batch = contraction_obs_batch.detach()
                else:
                    print("Warning: contraction_obs_batch is None in update method")
                
                if contraction_des_batch is not None:
                    contraction_des_batch = contraction_des_batch.detach()
                else:
                    print("Warning: contraction_des_batch is None in update method")
                
                if contraction_c_dot_batch is not None:
                    contraction_c_dot_batch = contraction_c_dot_batch.detach()
                else:
                    print("Warning: contraction_c_dot_batch is None in update method")
                
                if robot_state_batch is not None:
                    robot_state_batch = robot_state_batch.detach()
                else:
                    print("Warning: robot_state_batch is None in update method")
                
                if policy_output_batch is not None:
                    policy_output_batch = policy_output_batch.detach()
                else:
                    print("Warning: policy_output_batch is None in update method")
                    
                if prev_robot_state_batch is not None:
                    prev_robot_state_batch = prev_robot_state_batch.detach()
                else:
                    print("Warning: prev_robot_state_batch is None in update method, using current values")
                    prev_robot_state_batch = robot_state_batch
                    
                if prev_policy_output_batch is not None:
                    prev_policy_output_batch = prev_policy_output_batch.detach()
                else:
                    print("Warning: prev_policy_output_batch is None in update method, using current values")
                    prev_policy_output_batch = policy_output_batch
            
            # Handle data augmentation and batch processing (from parent class)
            num_aug = 1
            original_batch_size = obs_batch.shape[0]
            
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            
            # Symmetry augmentation (if enabled)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Apply symmetry augmentation logic from parent class
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
            
            # Recompute actions and values for current batch
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            
            # Get policy outputs for original batch size (before augmentation)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            
            # Adaptive learning rate (from parent class)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            
            # Standard PPO losses
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Base PPO loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            # ------------------------------------------------------------
            # Contraction metric loss (shared optimizer)
            # ------------------------------------------------------------
            if contraction_data_available:
                contraction_loss = self.update_contraction_metric(
                    contraction_obs_batch,
                    contraction_des_batch,
                    contraction_c_dot_batch,
                    robot_state_batch,
                    policy_output_batch,
                    prev_robot_state_batch,
                    prev_policy_output_batch,
                )
                # Add weighted contraction loss to PPO loss
                # loss += self.contraction_cfg['contraction_loss_coef'] * contraction_loss
                
                # Logging helpers
                mean_contraction_metric_loss += contraction_loss.item() * self.contraction_logging_scale
                mean_contraction_c_error += torch.norm(contraction_obs_batch - contraction_des_batch, dim=-1).mean()  # type: ignore
            
            # Symmetry loss (from parent class)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            
            # RND loss (from parent class)
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.rnd and self.rnd_optimizer is not None:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            
            # Gradient synchronization for multi-GPU
            if self.is_multi_gpu:
                self.reduce_parameters()
            
            # Apply gradients (clip both policy and contraction metric parameters)
            nn.utils.clip_grad_norm_(list(self.policy.parameters()), self.max_grad_norm)
            self.optimizer.step()
            
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            

            
            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
        
        # Compute average losses
        num_updates = self.num_mini_batches * self.num_learning_epochs
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_contraction_metric_loss /= num_updates
        mean_contraction_c_error /= num_updates
        
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        
        # Clear the storage
        self.storage.clear()
        
        # Return loss dictionary
        loss_dict = {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "contraction_metric_loss": mean_contraction_metric_loss,
            "contraction_c_error": mean_contraction_c_error,
        }
        
        if mean_rnd_loss is not None:
            loss_dict["rnd_loss"] = mean_rnd_loss
        if mean_symmetry_loss is not None:
            loss_dict["symmetry_loss"] = mean_symmetry_loss
            
        return loss_dict
    
    def set_total_training_iterations(self, total_iterations: int):
        """Set the total number of training iterations for phase scheduling."""
        self.metric_learning_iterations = total_iterations
    
    def cleanup_memory(self):
        """
        PERFORMANCE FIX: Explicit memory cleanup to prevent memory leaks.
        Call this periodically during training to free unused memory.
        """
        # Force garbage collection of PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear any accumulated gradients

        self.optimizer.zero_grad()
        self.contraction_metric_optimizer.zero_grad()
        
        if hasattr(self, 'rnd_optimizer') and self.rnd_optimizer is not None:
            self.rnd_optimizer.zero_grad()
    
    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        """
        Override to add contraction-based rewards and ensure proper memory management during environment steps.
        """
        # PERFORMANCE FIX: Detach rewards and dones to prevent gradient accumulation
        rewards_detached = rewards.detach() if rewards.requires_grad else rewards
        dones_detached = dones.detach() if dones.requires_grad else dones
        
        # Compute contraction-based rewards if contraction data is available
        contraction_rewards = torch.zeros_like(rewards_detached)
        contraction_data = self._extract_contraction_data_from_transition()
        if contraction_data is not None:
            contraction_rewards = self._compute_contraction_rewards(contraction_data).detach()
        
        # Store latest contraction rewards for external logging
        self.latest_contraction_rewards = contraction_rewards
        
        # Add contraction rewards to extrinsic rewards
        total_rewards = rewards_detached + contraction_rewards
        # total_rewards = rewards_detached
        
        # Call parent implementation with modified rewards
        super().process_env_step(total_rewards, dones_detached, infos)
        
        # PERFORMANCE FIX: Periodic memory cleanup during long rollouts
        if hasattr(self.storage, 'step') and self.storage.step > 0 and self.storage.step % 50 == 0:
            # Light cleanup every 50 steps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _extract_contraction_data_from_transition(self) -> dict | None:
        """Extract contraction data from the current transition."""
        try:
            if not hasattr(self.transition, 'observations') or not hasattr(self.transition, 'actions'):
                return None
            obs = self.transition.observations
            actions = self.transition.actions
            
            contraction_obs = None
            contraction_des = None
            contraction_c_dot = None
            robot_state = None
            policy_output = None
            
            # Try to extract contraction data from observations
            if hasattr(self.transition, 'contraction_obs'):
                contraction_obs = self.transition.contraction_obs
            
            if hasattr(self.transition, 'contraction_des'):
                contraction_des = self.transition.contraction_des
            
            if hasattr(self.transition, 'contraction_c_dot'):
                contraction_c_dot = self.transition.contraction_c_dot
            
            if hasattr(self.transition, 'robot_state'):
                robot_state = self.transition.robot_state
            
            if hasattr(self.transition, 'policy_output'):
                policy_output = self.transition.policy_output
            
            if hasattr(self.transition, 'prev_robot_state'):
                prev_robot_state = self.transition.prev_robot_state
            
            if hasattr(self.transition, 'prev_policy_output'):
                prev_policy_output = self.transition.prev_policy_output
            
            return {
                'obs': obs,
                'actions': actions,
                'contraction_obs': contraction_obs,
                'contraction_des': contraction_des,
                'contraction_c_dot': contraction_c_dot,
                'robot_state': robot_state,
                'policy_output': policy_output,
                'prev_robot_state': prev_robot_state,
                'prev_policy_output': prev_policy_output
            }
        except Exception as e:
            print(f"Warning: Failed to extract contraction data: {e}")
            return None
    
    def _compute_contraction_rewards(self, contraction_data: dict) -> torch.Tensor:
        """Compute contraction-based rewards from contraction data."""
        try:
            obs = contraction_data['obs']
            actions = contraction_data['actions']
            contraction_obs = contraction_data['contraction_obs']
            contraction_des = contraction_data['contraction_des']
            contraction_c_dot = contraction_data['contraction_c_dot']
            robot_state = contraction_data['robot_state']
            policy_output = contraction_data['policy_output']
            prev_robot_state = contraction_data['prev_robot_state']
            prev_policy_output = contraction_data['prev_policy_output']
            
            # Check if we have the required data
            if contraction_obs is None or robot_state is None:
                return torch.zeros(obs.shape[0], device=obs.device)
            
            # Normalize contraction tensors
            obs_batch_corrected = self._normalize_contraction_tensor(contraction_obs)
            des_batch_corrected = self._normalize_contraction_tensor(contraction_des)
            
            # Compute constraint violation and policy error
            c = (obs_batch_corrected - des_batch_corrected) * self.contraction_cfg['contraction_constraint_coef']
            e = robot_state - policy_output
            
            # Construct the new state vector x = [c; e]
            # x = torch.cat([c, e], dim=-1)
            x = c.clone().requires_grad_(True)
            
            # Compute Lyapunov function using new formulation
            with torch.no_grad():
                M = self.contraction_metric(x.detach())
            current_lyapunov = torch.bmm(x.unsqueeze(-2), torch.bmm(M, x.unsqueeze(-1))).squeeze(-1).squeeze(-1)
            
            lyapunov_derivatives = self.compute_lyapunov_derivative(
                obs_batch_corrected, des_batch_corrected, contraction_c_dot,
                robot_state, policy_output, prev_robot_state, prev_policy_output
            )

            # print(f"\033[92m🌈✨ [Contraction] current_lyapunov: shape={tuple(current_lyapunov.shape)}, mean={current_lyapunov.mean():.4f} | lyapunov_derivatives: shape={tuple(lyapunov_derivatives.shape)}, mean={lyapunov_derivatives.mean():.4f} 🚀😃\033[0m")
            
            # Compute contraction penalty
            contraction_penalty = self.compute_contraction_penalty(current_lyapunov, lyapunov_derivatives)
            
            # Convert penalty to reward (negative penalty becomes positive reward)
            contraction_rewards = (self.contraction_cfg['epsilon_contraction'] - contraction_penalty ) * self.contraction_penalty_coef
            
            return contraction_rewards
            
        except Exception as e:
            print(f"Warning: Failed to compute contraction rewards: {e}")
            return torch.zeros(obs.shape[0], device=obs.device)
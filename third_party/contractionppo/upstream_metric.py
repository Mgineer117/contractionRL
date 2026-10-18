"""
Contraction Metric Network for computing positive definite contraction metrics.

This module implements an MLP that outputs a lower triangular matrix Θ such that
the contraction metric M = ΘΘᵀ is guaranteed to be positive definite.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ContractionPPO.contraction_ppo.utils import resolve_nn_activation
from torch.nn.utils import spectral_norm


class SpectralNormBoundedLinear(nn.Module):
    """Linear layer wrapped with spectral normalization and a configurable Lipschitz bound.

    The spectral normalization ensures ||W||_2 = 1, and we rescale the weight by
    `bound` during the forward pass so that the effective spectral norm is <= bound.
    The bias is left unchanged, preserving the requirement to rescale weights only.
    """

    def __init__(self, linear: nn.Linear, bound: float):
        super().__init__()
        self.linear = spectral_norm(linear)
        self.bound = float(bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.bound == 1.0:
            return out
        # Scale only the weight contribution while leaving the bias unchanged:
        # desired: out = (bound * W) x + b
        # compute: bound * (W x + b) + (1 - bound) * b = bound * W x + b
        if self.linear.bias is None:
            return out * self.bound
        return out * self.bound + (1.0 - self.bound) * self.linear.bias


class ContractionMetricNetwork(nn.Module):
    """
    Multi-Layer Perceptron for computing contraction metrics.
    
    This network takes observations and outputs a lower triangular matrix Θ
    such that M = ΘΘᵀ forms a positive definite contraction metric.
    """

    def __init__(
        self,
        input_dim: int,
        constraint_dim: int,
        hidden_dims: list[int] = [128, 128],
        activation: str = "elu",
        min_diagonal_value: float = 0.1,
        spectral_norm_bound: float = 3.0,
        **kwargs,
    ):
        """
        Initialize the contraction metric network.
        
        Args:
            input_dim: Dimension of input observations for the MLP
            constraint_dim: Dimension of the constraint vector (obs - des)
            hidden_dims: Hidden layer dimensions for the MLP
            activation: Activation function name
            min_diagonal_value: Minimum value for diagonal elements to ensure positive definiteness
            spectral_norm_bound: Per-layer Lipschitz bound applied via spectral normalization (<= 1.0 recommended)
        """
        if kwargs:
            print(
                "ContractionMetricNetwork.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.constraint_dim = constraint_dim
        self.min_diagonal_value = min_diagonal_value
        self.spectral_norm_bound = float(spectral_norm_bound)
        
        # Calculate output dimension for lower triangular matrix
        # For an n×n matrix, we need n(n+1)/2 elements
        self.output_dim = constraint_dim * (constraint_dim + 1) // 2
        
        activation_fn = resolve_nn_activation(activation)
        
        # Build the MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        print(f"ContractionMetricNetwork: {self.mlp}")

        # Initialize weights
        self._initialize_weights()

        # Apply spectral normalization with scaling to enforce per-layer Lipschitz bound
        self._apply_spectral_norm()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                nn.init.constant_(module.bias, 0.0)
    
    def _apply_spectral_norm(self) -> None:
        """Wrap all Linear layers with spectral normalization and scale to the desired bound."""
        for index, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                self.mlp[index] = SpectralNormBoundedLinear(module, self.spectral_norm_bound)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the contraction metric.
        
        Args:
            observations: Input observations [batch_size, input_dim]
            
        Returns:
            Contraction metric M = ΘΘᵀ [batch_size, constraint_dim, constraint_dim]
        """
        batch_size = observations.shape[0]
        
        # Get raw outputs from MLP
        raw_output = self.mlp(observations)  # [batch_size, output_dim]
        
        # PERFORMANCE FIX: Use more efficient matrix operations
        # Pre-allocate theta tensor for better memory usage
        device = observations.device
        dtype = observations.dtype
        
        # Convert to lower triangular matrix Θ - OPTIMIZED VERSION
        theta = self._vector_to_lower_triangular_optimized(raw_output, batch_size, device, dtype)
        
        # Compute M = ΘΘᵀ to ensure positive definiteness
        # PERFORMANCE FIX: Use more efficient batch matrix multiplication
        metric = torch.bmm(theta, theta.transpose(-2, -1))

        # return torch.eye(self.constraint_dim, device=metric.device).repeat(batch_size, 1, 1)
        
        return metric
    
    def _vector_to_lower_triangular_optimized(self, vector: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Optimized version: Convert a vector to a lower triangular matrix.
        
        Args:
            vector: Flattened lower triangular elements [batch_size, output_dim]
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
            
        Returns:
            Lower triangular matrix Θ [batch_size, constraint_dim, constraint_dim]
        """
        n = self.constraint_dim
        
        # AUTOGRAD FIX: Create tensor that can participate in gradient computation
        # Use requires_grad based on input vector to maintain gradient flow
        theta = torch.zeros(batch_size, n, n, device=device, dtype=dtype, requires_grad=vector.requires_grad)
        
        # PERFORMANCE FIX: Cache tril_indices to avoid recomputation
        if not hasattr(self, '_cached_tril_indices') or self._cached_tril_indices[0].device != device:
            self._cached_tril_indices = torch.tril_indices(n, n, device=device)
        
        tril_indices = self._cached_tril_indices
        
        # Fill lower triangular part - AUTOGRAD FIX: Clone both tensors to avoid inference tensor issues
        if vector.requires_grad:
            # Create a clone that can be modified during gradient computation
            theta_clone = theta.clone()
            vector_clone = vector.clone()  # Clone the vector to avoid inference tensor issues
            theta_clone[:, tril_indices[0], tril_indices[1]] = vector_clone
            theta = theta_clone
        else:
            # For inference, clone the vector to avoid any potential issues
            vector_clone = vector.clone().detach()
            theta[:, tril_indices[0], tril_indices[1]] = vector_clone
        
        # PERFORMANCE FIX: Cache diagonal mask to avoid recomputation
        if not hasattr(self, '_cached_diagonal_mask') or self._cached_diagonal_mask.device != device:
            self._cached_diagonal_mask = torch.eye(n, device=device, dtype=torch.bool)
        
        diagonal_mask = self._cached_diagonal_mask
        
        # Ensure diagonal elements are positive by applying softplus + min_value
        # PERFORMANCE FIX: Apply softplus only to diagonal elements to reduce computation
        theta[:, diagonal_mask] = torch.nn.functional.softplus(theta[:, diagonal_mask]) + self.min_diagonal_value
        
        return theta
    
    def _vector_to_lower_triangular(self, vector: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Convert a vector to a lower triangular matrix.
        
        Args:
            vector: Flattened lower triangular elements [batch_size, output_dim]
            batch_size: Batch size
            
        Returns:
            Lower triangular matrix Θ [batch_size, constraint_dim, constraint_dim]
        """
        # PERFORMANCE FIX: Delegate to optimized version
        return self._vector_to_lower_triangular_optimized(vector, batch_size, vector.device, vector.dtype)
    
    def get_theta(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get the lower triangular matrix Θ directly.
        
        Args:
            observations: Input observations [batch_size, input_dim]
            
        Returns:
            Lower triangular matrix Θ [batch_size, constraint_dim, constraint_dim]
        """
        batch_size = observations.shape[0]
        raw_output = self.mlp(observations)
        return self._vector_to_lower_triangular(raw_output, batch_size)
"""
Policy Networks for DC-Ada Multi-Robot Systems

This module provides:
1. SharedPolicy: Base policy network that can be shared across robots
2. Utility functions for action sampling and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union, Optional


class SharedPolicy(nn.Module):
    """
    Shared policy network for multi-robot systems.
    
    Uses a simple MLP architecture with separate heads for mean and log_std.
    Can be used as-is for homogeneous robots or combined with transformation
    layers for heterogeneous robots.
    """
    
    def __init__(
        self,
        obs_dim: int = 68,  # Default for H1 heterogeneity
        action_dim: int = 2,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Build network layers
        layers = []
        in_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights.

        NOTE: Orthogonal initialization can be unusually slow on some CPU/BLAS
        configurations for 256x256 matrices. Xavier init is fast and stable
        for MLP policies and keeps the smoke tests + large sweeps practical.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # Small initialization for output heads (stabilizes early action scale)
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            x: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,)
            
        Returns:
            mean: Action mean tensor
            log_std: Action log standard deviation tensor
        """
        # Handle single observation
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Sanitize non-finite inputs (can occur if a method diverges numerically)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -10.0, 10.0)

            
        # Forward through backbone
        features = self.backbone(x)
        
        # Get action distribution parameters
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        
        return mean, log_std
    
    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Get action for a single observation.
        
        Args:
            obs: Single observation array or tensor
            deterministic: If True, return mean action; otherwise sample
            
        Returns:
            action: Action array of shape (action_dim,)
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                action = torch.tanh(mean + std * noise)
            
            return action.squeeze(0).numpy()
    
    def get_actions(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False
    ) -> List[np.ndarray]:
        """
        Get actions for multiple observations (one per robot).
        
        Args:
            observations: List of observation arrays, one per robot
            deterministic: If True, return mean actions
            
        Returns:
            actions: List of action arrays, one per robot
        """
        actions = []
        for obs in observations:
            action = self.get_action(obs, deterministic=deterministic)
            actions.append(action)
        return actions
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            actions: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
            mean: Mean of action distribution
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        # Compute log probability (for tanh-squashed Gaussian)
        # Using the formula: log_prob = log_prob_gaussian - sum(log(1 - tanh(u)^2))
        # Numerical safety: guard against NaNs/Infs in actions
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.999, neginf=-0.999)
        actions = torch.clamp(actions, -0.999, 0.999)

        u = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob_gaussian = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob_gaussian = log_prob_gaussian.sum(dim=-1)
        
        # Correction for tanh squashing
        log_prob_correction = torch.log(1 - actions ** 2 + 1e-6).sum(dim=-1)
        log_prob = log_prob_gaussian - log_prob_correction
        
        # Compute entropy (approximate for tanh-squashed Gaussian)
        entropy = 0.5 * (1 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        
        return log_prob, entropy, mean
    
    def get_params_vector(self) -> torch.Tensor:
        """Get all parameters as a single flattened vector."""
        return torch.cat([p.view(-1) for p in self.parameters()])
    
    def set_params_vector(self, params: torch.Tensor):
        """Set all parameters from a single flattened vector."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(params[idx:idx + numel].view(p.shape))
            idx += numel
    
    def clone(self) -> 'SharedPolicy':
        """Create a deep copy of this policy."""
        new_policy = SharedPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        new_policy.load_state_dict(self.state_dict())
        return new_policy


def create_policy(obs_dim: int, action_dim: int = 2, hidden_dim: int = 256) -> SharedPolicy:
    """Factory function to create a policy network."""
    return SharedPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)

"""
Transformation Layers for DC-Ada

These layers transform heterogeneous robot observations into a common
latent space that the shared policy can process. Each robot has its own
transformation layer that adapts to its specific sensor configuration.

Key Design Principles:
1. Lightweight: Small parameter count for efficient zeroth-order optimization
2. Residual: Preserves information through skip connections
3. Adaptive: Can handle varying observation dimensions through padding
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class TransformationLayer(nn.Module):
    """
    Transformation layer that maps robot-specific observations to a common space.
    
    Architecture:
    - Input projection to latent space
    - Single hidden layer with ReLU
    - Output projection back to observation space
    - Residual connection (when dimensions match)
    
    This is intentionally lightweight for efficient zeroth-order optimization.
    """
    
    def __init__(
        self,
        obs_dim: int = 68,
        latent_dim: int = 32,
        use_residual: bool = True,
        init_std: float = 0.01,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        self.init_std = float(init_std)
        
        # Encoder: obs_dim -> latent_dim
        self.encoder = nn.Linear(obs_dim, latent_dim)
        
        # Hidden layer
        self.hidden = nn.Linear(latent_dim, latent_dim)
        
        # Decoder: latent_dim -> obs_dim
        self.decoder = nn.Linear(latent_dim, obs_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # Initialize close to (or exactly) identity.
        self._init_near_identity()
        
    def _init_near_identity(self):
        """Initialize weights so the layer starts close to identity mapping."""
        std = float(self.init_std)
        if std <= 0.0:
            # Exact identity: delta(x)=0 so output=x (via residual).
            nn.init.zeros_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
            return

        # Small random initialization (residual is the main identity path).
        nn.init.normal_(self.encoder.weight, std=std)
        nn.init.zeros_(self.encoder.bias)

        nn.init.normal_(self.hidden.weight, std=std)
        nn.init.zeros_(self.hidden.bias)

        nn.init.normal_(self.decoder.weight, std=std)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform observation through the layer.
        
        Args:
            x: Input observation tensor of shape (batch_size, obs_dim) or (obs_dim,)
            
        Returns:
            Transformed observation of same shape as input
        """
        # Handle single observation
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        # Encode
        latent = torch.relu(self.encoder(x))
        latent = self.layer_norm(latent)
        
        # Hidden
        latent = torch.relu(self.hidden(latent))
        
        # Decode
        output = self.decoder(latent)
        
        # Residual connection
        if self.use_residual:
            output = output + x
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output
    
    def get_params_vector(self) -> torch.Tensor:
        """
        Get all parameters as a single flattened vector.
        
        Returns:
            1D tensor containing all parameters
        """
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)
    
    def set_params_vector(self, params: torch.Tensor):
        """
        Set all parameters from a single flattened vector.
        
        Args:
            params: 1D tensor containing all parameters
        """
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(params[idx:idx + numel].view(p.shape))
            idx += numel
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def perturb(self, noise_scale: float = 0.01) -> 'TransformationLayer':
        """
        Create a perturbed copy of this layer.
        
        Args:
            noise_scale: Standard deviation of Gaussian noise
            
        Returns:
            New TransformationLayer with perturbed parameters
        """
        new_layer = TransformationLayer(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            use_residual=self.use_residual,
            init_std=self.init_std,
        )
        
        # Copy and perturb parameters
        params = self.get_params_vector()
        noise = torch.randn_like(params) * noise_scale
        new_layer.set_params_vector(params + noise)
        
        return new_layer
    
    def clone(self) -> 'TransformationLayer':
        """Create a deep copy of this layer."""
        new_layer = TransformationLayer(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            use_residual=self.use_residual,
            init_std=self.init_std,
        )
        new_layer.load_state_dict(self.state_dict())
        return new_layer


class RobotTransformationModule:
    """
    Manages transformation layers for all robots in a multi-robot system.
    
    Each robot gets its own transformation layer that can be independently
    optimized using zeroth-order methods.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 68,
        latent_dim: int = 32
    ):
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Create transformation layer for each robot
        self.layers = [
            TransformationLayer(obs_dim=obs_dim, latent_dim=latent_dim)
            for _ in range(num_robots)
        ]
        
    def transform(self, observations: list) -> list:
        """
        Transform observations for all robots.
        
        Args:
            observations: List of observation arrays, one per robot
            
        Returns:
            List of transformed observation arrays
        """
        transformed = []
        for i, obs in enumerate(observations):
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float()
            else:
                obs_tensor = obs
                
            with torch.no_grad():
                trans_obs = self.layers[i](obs_tensor)
                
            if isinstance(obs, np.ndarray):
                transformed.append(trans_obs.numpy())
            else:
                transformed.append(trans_obs)
                
        return transformed
    
    def get_layer(self, robot_id: int) -> TransformationLayer:
        """Get transformation layer for a specific robot."""
        return self.layers[robot_id]
    
    def set_layer(self, robot_id: int, layer: TransformationLayer):
        """Set transformation layer for a specific robot."""
        self.layers[robot_id] = layer
    
    def get_all_params(self) -> torch.Tensor:
        """Get parameters from all layers as a single vector."""
        all_params = []
        for layer in self.layers:
            all_params.append(layer.get_params_vector())
        return torch.cat(all_params)
    
    def set_all_params(self, params: torch.Tensor):
        """Set parameters for all layers from a single vector."""
        idx = 0
        for layer in self.layers:
            num_params = layer.get_num_params()
            layer.set_params_vector(params[idx:idx + num_params])
            idx += num_params
    
    def clone(self) -> 'RobotTransformationModule':
        """Create a deep copy of this module."""
        new_module = RobotTransformationModule(
            num_robots=self.num_robots,
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim
        )
        for i, layer in enumerate(self.layers):
            new_module.layers[i] = layer.clone()
        return new_module

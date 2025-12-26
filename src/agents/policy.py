"""
Shared Policy Network for Multi-Robot Systems

This module implements the frozen shared policy that all robots execute.
The policy is a feedforward MLP that maps observations to actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SharedPolicy(nn.Module):
    """
    Shared policy network for multi-robot coordination.
    
    Architecture:
    - Input: Transformed observation vector
    - Hidden: Two layers of 256 units with ReLU
    - Output: Mean and log_std for Gaussian action distribution
    """
    
    def __init__(
        self,
        obs_dim: int = 56,
        action_dim: int = 2,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Policy network
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Smaller initialization for output layers
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            mean: Action mean of shape (batch_size, action_dim)
            log_std: Action log standard deviation of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action from the policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action; otherwise sample
            
        Returns:
            action: Action tensor
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            action = mean + std * noise
            return torch.tanh(action)
    
    def get_action_numpy(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action as numpy array."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.get_action(obs_tensor, deterministic)
            return action.squeeze(0).numpy()


class TransformationLayer(nn.Module):
    """
    Per-robot data transformation layer for DC-Ada.
    
    This layer transforms the raw observation before passing it to the shared policy.
    It implements three types of transformations:
    1. Affine transformation (scale and shift)
    2. Modality weighting (soft attention over sensor modalities)
    3. Learnable dropout mask
    """
    
    def __init__(
        self,
        obs_dim: int = 56,
        lidar_dim: int = 32,
        camera_dim: int = 16,
        odom_dim: int = 4,
        pkg_dim: int = 4
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.lidar_dim = lidar_dim
        self.camera_dim = camera_dim
        self.odom_dim = odom_dim
        self.pkg_dim = pkg_dim
        
        # Transformation parameters (phi_i in the paper)
        # Affine parameters: scale and bias for each dimension
        self.scale = nn.Parameter(torch.ones(obs_dim))
        self.bias = nn.Parameter(torch.zeros(obs_dim))
        
        # Modality weights (before sigmoid)
        self.modality_weights_raw = nn.Parameter(torch.zeros(4))  # lidar, camera, odom, pkg
        
        # Calibration offsets for each modality
        self.lidar_offset = nn.Parameter(torch.zeros(lidar_dim))
        self.camera_offset = nn.Parameter(torch.zeros(camera_dim))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to observation.
        
        Args:
            obs: Raw observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            transformed_obs: Transformed observation tensor
        """
        # Split observation into modalities
        lidar = obs[..., :self.lidar_dim]
        camera = obs[..., self.lidar_dim:self.lidar_dim + self.camera_dim]
        odom = obs[..., self.lidar_dim + self.camera_dim:self.lidar_dim + self.camera_dim + self.odom_dim]
        pkg = obs[..., -self.pkg_dim:]
        
        # Apply calibration offsets
        lidar = lidar + self.lidar_offset
        camera = camera + self.camera_offset
        
        # Apply modality weights (soft gating)
        weights = torch.sigmoid(self.modality_weights_raw)
        
        lidar = lidar * weights[0]
        camera = camera * weights[1]
        odom = odom * weights[2]
        pkg = pkg * weights[3]
        
        # Reconstruct observation
        transformed = torch.cat([lidar, camera, odom, pkg], dim=-1)
        
        # Apply affine transformation
        transformed = transformed * self.scale + self.bias
        
        return transformed
    
    def get_params_vector(self) -> np.ndarray:
        """Get all transformation parameters as a flat vector."""
        params = []
        params.append(self.scale.detach().cpu().numpy())
        params.append(self.bias.detach().cpu().numpy())
        params.append(self.modality_weights_raw.detach().cpu().numpy())
        params.append(self.lidar_offset.detach().cpu().numpy())
        params.append(self.camera_offset.detach().cpu().numpy())
        return np.concatenate(params)
    
    def set_params_vector(self, params: np.ndarray):
        """Set all transformation parameters from a flat vector."""
        idx = 0
        
        with torch.no_grad():
            self.scale.copy_(torch.from_numpy(params[idx:idx + self.obs_dim]).float())
            idx += self.obs_dim
            
            self.bias.copy_(torch.from_numpy(params[idx:idx + self.obs_dim]).float())
            idx += self.obs_dim
            
            self.modality_weights_raw.copy_(torch.from_numpy(params[idx:idx + 4]).float())
            idx += 4
            
            self.lidar_offset.copy_(torch.from_numpy(params[idx:idx + self.lidar_dim]).float())
            idx += self.lidar_dim
            
            self.camera_offset.copy_(torch.from_numpy(params[idx:idx + self.camera_dim]).float())
    
    def get_num_params(self) -> int:
        """Get total number of transformation parameters."""
        return self.obs_dim * 2 + 4 + self.lidar_dim + self.camera_dim
    
    def get_modality_weights(self) -> np.ndarray:
        """Get the current modality weights (after sigmoid)."""
        with torch.no_grad():
            return torch.sigmoid(self.modality_weights_raw).cpu().numpy()


class DCAdaAgent:
    """
    DC-Ada Agent: Combines frozen shared policy with learnable transformation.
    """
    
    def __init__(
        self,
        robot_id: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None
    ):
        self.robot_id = robot_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared policy (frozen)
        if shared_policy is None:
            self.policy = SharedPolicy(obs_dim, action_dim)
        else:
            self.policy = shared_policy
        
        # Freeze policy parameters
        for param in self.policy.parameters():
            param.requires_grad = False
        
        # Per-robot transformation layer
        self.transform = TransformationLayer(obs_dim)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action for the robot."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            transformed_obs = self.transform(obs_tensor)
            action = self.policy.get_action(transformed_obs, deterministic)
            return action.squeeze(0).numpy()
    
    def get_transform_params(self) -> np.ndarray:
        """Get transformation parameters."""
        return self.transform.get_params_vector()
    
    def set_transform_params(self, params: np.ndarray):
        """Set transformation parameters."""
        self.transform.set_params_vector(params)
    
    def get_num_transform_params(self) -> int:
        """Get number of transformation parameters."""
        return self.transform.get_num_params()


def pretrain_shared_policy(
    env,
    policy: SharedPolicy,
    num_episodes: int = 100,
    learning_rate: float = 3e-4
) -> SharedPolicy:
    """
    Pre-train the shared policy using simple policy gradient.
    
    This creates a reasonable baseline policy that DC-Ada will adapt.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        while not done:
            # Collect actions for all robots
            actions = {}
            log_probs = []
            
            for robot_id in range(env.num_robots):
                robot_obs = obs[f"robot_{robot_id}"]
                obs_tensor = torch.FloatTensor(robot_obs).unsqueeze(0)
                
                mean, log_std = policy(obs_tensor)
                std = torch.exp(log_std)
                
                # Sample action
                noise = torch.randn_like(mean)
                action_raw = mean + std * noise
                action = torch.tanh(action_raw)
                
                # Compute log probability
                log_prob = -0.5 * ((action_raw - mean) / std).pow(2).sum(-1)
                log_prob -= 0.5 * np.log(2 * np.pi) * action.shape[-1]
                log_prob -= log_std.sum(-1)
                # Correction for tanh squashing
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
                
                actions[f"robot_{robot_id}"] = action.squeeze(0).detach().numpy()
                log_probs.append(log_prob)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_log_probs.append(torch.stack(log_probs).mean())
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = 0
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss -= log_prob * G
        
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        if (episode + 1) % 10 == 0:
            print(f"Pretrain Episode {episode + 1}: Total Reward = {sum(episode_rewards):.2f}")
    
    return policy


if __name__ == "__main__":
    # Test the policy and transformation
    policy = SharedPolicy()
    transform = TransformationLayer()
    
    # Test forward pass
    obs = torch.randn(1, 56)
    transformed = transform(obs)
    mean, log_std = policy(transformed)
    
    print(f"Policy output - Mean: {mean.shape}, Log_std: {log_std.shape}")
    print(f"Transform params: {transform.get_num_params()}")
    print(f"Modality weights: {transform.get_modality_weights()}")
    
    # Test DC-Ada agent
    agent = DCAdaAgent(robot_id=0)
    obs_np = np.random.randn(56).astype(np.float32)
    action = agent.get_action(obs_np)
    print(f"Agent action: {action}")

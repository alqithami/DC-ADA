"""
DC-Ada and Baseline Methods Implementation

This module implements:
1. DC-Ada: Data-Centric Decentralized Adaptation
2. Shared Policy (No Adaptation) baseline
3. Random Perturbation baseline
4. Local Fine-Tuning baseline
5. Gradient-Based Fine-Tuning (Centralized) baseline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

try:
    from .policy import SharedPolicy, TransformationLayer, DCAdaAgent
except ImportError:
    from policy import SharedPolicy, TransformationLayer, DCAdaAgent


class BaseMethod:
    """Base class for all methods."""
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None
    ):
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create or use provided shared policy
        if shared_policy is None:
            self.shared_policy = SharedPolicy(obs_dim, action_dim)
        else:
            self.shared_policy = shared_policy
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions for all robots."""
        raise NotImplementedError
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """Update method based on team score."""
        raise NotImplementedError
    
    def reset(self):
        """Reset method state for new experiment."""
        pass
    
    def get_metrics(self) -> Dict:
        """Get method-specific metrics."""
        return {}


class DCAdaMethod(BaseMethod):
    """
    DC-Ada: Data-Centric Decentralized Adaptation
    
    Each robot has a learnable transformation layer that adapts its sensor
    inputs to work better with the frozen shared policy.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None,
        # DC-Ada hyperparameters
        num_candidates: int = 16,
        perturbation_scale: float = 0.1,
        step_size: float = 0.01,
        acceptance_margin: float = 0.01
    ):
        super().__init__(num_robots, obs_dim, action_dim, shared_policy)
        
        self.num_candidates = num_candidates
        self.perturbation_scale = perturbation_scale
        self.step_size = step_size
        self.acceptance_margin = acceptance_margin
        
        # Create per-robot transformation layers
        self.transforms = [TransformationLayer(obs_dim) for _ in range(num_robots)]
        
        # Freeze shared policy
        for param in self.shared_policy.parameters():
            param.requires_grad = False
        
        # Track adaptation state
        self.current_robot_idx = 0
        self.baseline_score = None
        self.best_perturbation = None
        self.best_score = -float('inf')
        self.candidate_idx = 0
        
        # Metrics
        self.update_counts = [0] * num_robots
        self.modality_weights_history = []
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions for all robots using transformed observations."""
        actions = {}
        
        with torch.no_grad():
            for i in range(self.num_robots):
                obs = observations[f"robot_{i}"]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Apply transformation
                transformed_obs = self.transforms[i](obs_tensor)
                
                # Get action from shared policy
                action = self.shared_policy.get_action(transformed_obs, deterministic=False)
                actions[f"robot_{i}"] = action.squeeze(0).numpy()
        
        return actions
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """
        Update transformation parameters using zeroth-order optimization.
        
        Implements the coordinate-descent style adaptation from the paper.
        """
        update_info = {
            "updated": False,
            "robot_updated": None,
            "score_improvement": 0.0
        }
        
        # First episode: establish baseline
        if self.baseline_score is None:
            self.baseline_score = team_score
            self.best_score = team_score
            return update_info
        
        # Check if current perturbation is better
        if team_score > self.best_score:
            self.best_score = team_score
            self.best_perturbation = self._get_current_perturbation()
        
        self.candidate_idx += 1
        
        # After evaluating all candidates for current robot
        if self.candidate_idx >= self.num_candidates:
            # Check if we found an improvement
            if self.best_score > self.baseline_score + self.acceptance_margin:
                # Apply the best perturbation
                if self.best_perturbation is not None:
                    self._apply_perturbation(self.current_robot_idx, self.best_perturbation)
                    self.update_counts[self.current_robot_idx] += 1
                    
                    update_info["updated"] = True
                    update_info["robot_updated"] = self.current_robot_idx
                    update_info["score_improvement"] = self.best_score - self.baseline_score
                
                self.baseline_score = self.best_score
            
            # Move to next robot
            self.current_robot_idx = (self.current_robot_idx + 1) % self.num_robots
            self.candidate_idx = 0
            self.best_score = self.baseline_score
            self.best_perturbation = None
            
            # Generate new perturbation for next robot
            self._generate_perturbation(self.current_robot_idx)
        else:
            # Generate next candidate perturbation
            self._generate_perturbation(self.current_robot_idx)
        
        # Record modality weights
        weights = [t.get_modality_weights() for t in self.transforms]
        self.modality_weights_history.append(weights)
        
        return update_info
    
    def _get_current_perturbation(self) -> np.ndarray:
        """Get current perturbation vector."""
        return self.transforms[self.current_robot_idx].get_params_vector()
    
    def _generate_perturbation(self, robot_idx: int):
        """Generate a random perturbation for the specified robot."""
        current_params = self.transforms[robot_idx].get_params_vector()
        perturbation = np.random.randn(len(current_params)) * self.perturbation_scale
        new_params = current_params + perturbation
        self.transforms[robot_idx].set_params_vector(new_params)
    
    def _apply_perturbation(self, robot_idx: int, params: np.ndarray):
        """Apply perturbation with step size."""
        current_params = self.transforms[robot_idx].get_params_vector()
        new_params = current_params + self.step_size * (params - current_params)
        self.transforms[robot_idx].set_params_vector(new_params)
    
    def reset(self):
        """Reset for new experiment."""
        self.transforms = [TransformationLayer(self.obs_dim) for _ in range(self.num_robots)]
        self.current_robot_idx = 0
        self.baseline_score = None
        self.best_perturbation = None
        self.best_score = -float('inf')
        self.candidate_idx = 0
        self.update_counts = [0] * self.num_robots
        self.modality_weights_history = []
    
    def get_metrics(self) -> Dict:
        """Get DC-Ada specific metrics."""
        return {
            "update_counts": self.update_counts.copy(),
            "modality_weights": [t.get_modality_weights() for t in self.transforms]
        }


class SharedPolicyMethod(BaseMethod):
    """
    Shared Policy (No Adaptation) Baseline
    
    All robots use the same frozen policy with identity transformation.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None
    ):
        super().__init__(num_robots, obs_dim, action_dim, shared_policy)
        
        # Freeze policy
        for param in self.shared_policy.parameters():
            param.requires_grad = False
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions using shared policy directly."""
        actions = {}
        
        with torch.no_grad():
            for i in range(self.num_robots):
                obs = observations[f"robot_{i}"]
                action = self.shared_policy.get_action_numpy(obs, deterministic=False)
                actions[f"robot_{i}"] = action
        
        return actions
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """No updates for this baseline."""
        return {"updated": False}


class RandomPerturbationMethod(BaseMethod):
    """
    Random Perturbation Baseline
    
    Transformation parameters are updated with random noise,
    independent of the team reward.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None,
        perturbation_scale: float = 0.1
    ):
        super().__init__(num_robots, obs_dim, action_dim, shared_policy)
        
        self.perturbation_scale = perturbation_scale
        self.transforms = [TransformationLayer(obs_dim) for _ in range(num_robots)]
        
        # Freeze policy
        for param in self.shared_policy.parameters():
            param.requires_grad = False
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions with transformed observations."""
        actions = {}
        
        with torch.no_grad():
            for i in range(self.num_robots):
                obs = observations[f"robot_{i}"]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                transformed_obs = self.transforms[i](obs_tensor)
                action = self.shared_policy.get_action(transformed_obs, deterministic=False)
                actions[f"robot_{i}"] = action.squeeze(0).numpy()
        
        return actions
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """Apply random perturbations regardless of score."""
        for i in range(self.num_robots):
            current_params = self.transforms[i].get_params_vector()
            perturbation = np.random.randn(len(current_params)) * self.perturbation_scale
            self.transforms[i].set_params_vector(current_params + perturbation)
        
        return {"updated": True, "random": True}
    
    def reset(self):
        """Reset transformations."""
        self.transforms = [TransformationLayer(self.obs_dim) for _ in range(self.num_robots)]


class LocalFineTuningMethod(BaseMethod):
    """
    Local Fine-Tuning Baseline
    
    Each robot fine-tunes its own copy of the policy using local experience,
    without any communication.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None,
        learning_rate: float = 1e-4
    ):
        super().__init__(num_robots, obs_dim, action_dim, shared_policy)
        
        self.learning_rate = learning_rate
        
        # Each robot has its own policy copy
        self.local_policies = [deepcopy(self.shared_policy) for _ in range(num_robots)]
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=learning_rate)
            for policy in self.local_policies
        ]
        
        # Store episode data for each robot
        self.episode_buffers = [[] for _ in range(num_robots)]
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions from local policies."""
        actions = {}
        
        for i in range(self.num_robots):
            obs = observations[f"robot_{i}"]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = self.local_policies[i].get_action(obs_tensor, deterministic=False)
            
            actions[f"robot_{i}"] = action.squeeze(0).numpy()
            
            # Store for training
            self.episode_buffers[i].append({
                "obs": obs,
                "action": actions[f"robot_{i}"]
            })
        
        return actions
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """Update local policies using stored experience."""
        for i in range(self.num_robots):
            if len(self.episode_buffers[i]) == 0:
                continue
            
            # Simple policy gradient update
            policy = self.local_policies[i]
            optimizer = self.optimizers[i]
            
            # Compute loss
            loss = 0
            for data in self.episode_buffers[i]:
                obs = torch.FloatTensor(data["obs"]).unsqueeze(0)
                action = torch.FloatTensor(data["action"]).unsqueeze(0)
                
                mean, log_std = policy(obs)
                std = torch.exp(log_std)
                
                # Negative log likelihood
                nll = 0.5 * ((action - mean) / std).pow(2).sum(-1)
                nll += log_std.sum(-1)
                
                # Weight by team score (simple REINFORCE)
                loss += nll * (-team_score / len(self.episode_buffers[i]))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            # Clear buffer
            self.episode_buffers[i] = []
        
        return {"updated": True}
    
    def reset(self):
        """Reset local policies."""
        self.local_policies = [deepcopy(self.shared_policy) for _ in range(self.num_robots)]
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=self.learning_rate)
            for policy in self.local_policies
        ]
        self.episode_buffers = [[] for _ in range(self.num_robots)]


class GradientFineTuningMethod(BaseMethod):
    """
    Gradient-Based Fine-Tuning (Centralized) Baseline
    
    A centralized server aggregates gradients from all robots
    to update the shared policy.
    """
    
    def __init__(
        self,
        num_robots: int,
        obs_dim: int = 56,
        action_dim: int = 2,
        shared_policy: Optional[SharedPolicy] = None,
        learning_rate: float = 3e-4,
        update_frequency: int = 10
    ):
        super().__init__(num_robots, obs_dim, action_dim, shared_policy)
        
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        
        self.optimizer = optim.Adam(self.shared_policy.parameters(), lr=learning_rate)
        
        # Gradient buffer
        self.gradient_buffer = []
        self.episode_count = 0
        
        # Episode data storage
        self.all_episode_data = []
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions from shared policy."""
        actions = {}
        step_data = {"obs": {}, "actions": {}}
        
        for i in range(self.num_robots):
            obs = observations[f"robot_{i}"]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = self.shared_policy.get_action(obs_tensor, deterministic=False)
            
            actions[f"robot_{i}"] = action.squeeze(0).numpy()
            step_data["obs"][i] = obs
            step_data["actions"][i] = actions[f"robot_{i}"]
        
        self.all_episode_data.append(step_data)
        
        return actions
    
    def update(self, team_score: float, episode_data: Dict) -> Dict:
        """Aggregate gradients and update shared policy."""
        self.episode_count += 1
        
        # Compute gradients for this episode
        if len(self.all_episode_data) > 0:
            loss = 0
            for step_data in self.all_episode_data:
                for i in range(self.num_robots):
                    obs = torch.FloatTensor(step_data["obs"][i]).unsqueeze(0)
                    action = torch.FloatTensor(step_data["actions"][i]).unsqueeze(0)
                    
                    mean, log_std = self.shared_policy(obs)
                    std = torch.exp(log_std)
                    
                    # Negative log likelihood weighted by reward
                    nll = 0.5 * ((action - mean) / std).pow(2).sum(-1)
                    nll += log_std.sum(-1)
                    loss += nll * (-team_score / (len(self.all_episode_data) * self.num_robots))
            
            # Store gradients
            self.optimizer.zero_grad()
            loss.backward()
            
            # Save gradients
            gradients = []
            for param in self.shared_policy.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.clone())
            self.gradient_buffer.append(gradients)
        
        self.all_episode_data = []
        
        # Update every update_frequency episodes
        update_info = {"updated": False}
        if self.episode_count % self.update_frequency == 0 and len(self.gradient_buffer) > 0:
            # Average gradients
            self.optimizer.zero_grad()
            
            for i, param in enumerate(self.shared_policy.parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                
                avg_grad = torch.zeros_like(param)
                for grads in self.gradient_buffer:
                    if i < len(grads):
                        avg_grad += grads[i]
                avg_grad /= len(self.gradient_buffer)
                param.grad = avg_grad
            
            torch.nn.utils.clip_grad_norm_(self.shared_policy.parameters(), 0.5)
            self.optimizer.step()
            
            self.gradient_buffer = []
            update_info["updated"] = True
        
        return update_info
    
    def reset(self):
        """Reset optimizer and buffers."""
        self.optimizer = optim.Adam(self.shared_policy.parameters(), lr=self.learning_rate)
        self.gradient_buffer = []
        self.episode_count = 0
        self.all_episode_data = []


def create_method(
    method_name: str,
    num_robots: int,
    shared_policy: SharedPolicy,
    **kwargs
) -> BaseMethod:
    """Factory function to create methods."""
    
    methods = {
        "dc_ada": DCAdaMethod,
        "shared_policy": SharedPolicyMethod,
        "random_perturbation": RandomPerturbationMethod,
        "local_finetuning": LocalFineTuningMethod,
        "gradient_finetuning": GradientFineTuningMethod
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    
    return methods[method_name](
        num_robots=num_robots,
        shared_policy=deepcopy(shared_policy),
        **kwargs
    )


if __name__ == "__main__":
    # Test all methods
    import sys
    sys.path.insert(0, "/home/ubuntu/dc-ada-experiments/src")
    from envs.warehouse_env import make_env
    
    env = make_env("warehouse", num_robots=4, seed=42)
    shared_policy = SharedPolicy()
    
    methods_to_test = ["dc_ada", "shared_policy", "random_perturbation", 
                       "local_finetuning", "gradient_finetuning"]
    
    for method_name in methods_to_test:
        print(f"\nTesting {method_name}...")
        method = create_method(method_name, num_robots=4, shared_policy=shared_policy)
        
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(50):
            actions = method.get_actions(obs)
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        team_score = env.get_team_score()
        update_info = method.update(team_score, {})
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Team score: {team_score:.2f}")
        print(f"  Update info: {update_info}")

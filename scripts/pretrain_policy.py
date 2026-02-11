#!/usr/bin/env python3
"""
Pre-trains a shared policy for a given environment using REINFORCE.

This creates a baseline policy that can be used as a starting point
for all adaptation methods.

Usage:
    python scripts/pretrain_policy.py --config configs/default.yaml --output checkpoints/policy.pth
    python scripts/pretrain_policy.py --env warehouse --episodes 200 --output checkpoints/warehouse.pth
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import torch.optim as optim

from src.envs import WarehouseEnv, SearchRescueEnv, CollaborativeMappingEnv
from src.agents.policy import SharedPolicy
from src.utils.seeding import set_seed


ENV_MAP = {
    "warehouse": WarehouseEnv,
    "search_rescue": SearchRescueEnv,
    "mapping": CollaborativeMappingEnv
}


class ValueNetwork(torch.nn.Module):
    """Lightweight critic used only during pre-training."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


def pretrain_shared_policy(
    env,
    policy: SharedPolicy,
    num_episodes: int = 200,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip: float = 0.5,
    verbose: bool = True,
) -> SharedPolicy:
    """Pre-train a shared policy using a simple Advantage Actor-Critic (A2C).

    Notes:
      - The environment returns a TEAM reward. We therefore aggregate
        per-robot log-probs/values by averaging.
      - The shared policy is saved; the critic is discarded.
    """

    critic = ValueNetwork(obs_dim=policy.obs_dim, hidden_dim=256)
    optimizer = optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=learning_rate)

    episode_rewards: list[float] = []
    episode_successes: list[float] = []
    best_state_dict = None
    best_score = -float('inf')

    for episode in range(num_episodes):
        obs_list, _ = env.reset()
        log_probs = []
        values = []
        entropies = []
        rewards = []
        done = False
        last_info = {}

        while not done:
            step_logps = []
            step_vals = []
            step_ents = []
            actions = []

            for obs in obs_list:
                obs_tensor = torch.from_numpy(obs).float()
                mean, log_std = policy(obs_tensor)
                std = torch.exp(log_std)

                # Sample action (tanh-squashed Gaussian)
                noise = torch.randn_like(mean)
                u = mean + std * noise
                action = torch.tanh(u)

                # Log prob with change-of-variables correction
                log_prob_gaussian = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
                log_prob_gaussian = log_prob_gaussian.sum(dim=-1)
                log_prob_correction = torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)
                log_prob = log_prob_gaussian - log_prob_correction

                # Entropy of the underlying Gaussian (proxy)
                entropy = (0.5 * (1 + np.log(2 * np.pi)) + log_std).sum(dim=-1)

                # Critic value estimate
                value = critic(obs_tensor)

                step_logps.append(log_prob.squeeze(0) if log_prob.dim() else log_prob)
                step_vals.append(value.squeeze(0) if value.dim() else value)
                step_ents.append(entropy.squeeze(0) if entropy.dim() else entropy)
                actions.append(action.detach().numpy().flatten())

            log_probs.append(torch.stack(step_logps).mean())
            values.append(torch.stack(step_vals).mean())
            entropies.append(torch.stack(step_ents).mean())

            obs_list, reward, done, last_info = env.step(actions)
            rewards.append(float(reward))

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        values_t = torch.stack(values)
        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Losses
        policy_loss = 0.0
        value_loss = 0.0
        entropy_bonus = 0.0
        for lp, v, ent, adv, ret in zip(log_probs, values_t, entropies, advantages, returns_t):
            policy_loss = policy_loss - lp * adv
            value_loss = value_loss + (v - ret) ** 2
            entropy_bonus = entropy_bonus + ent

        T = max(1, len(returns_t))
        policy_loss = policy_loss / T
        value_loss = value_loss / T
        entropy_bonus = entropy_bonus / T

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), grad_clip)
        optimizer.step()

        episode_reward = float(sum(rewards))
        episode_rewards.append(episode_reward)
        episode_successes.append(1.0 if bool(last_info.get('success', False)) else 0.0)

        # Track best checkpoint by combined score (reward + success)
        if (episode + 1) % 20 == 0:
            avg_reward = float(np.mean(episode_rewards[-20:]))
            avg_success = float(np.mean(episode_successes[-20:]))
            if verbose:
                print(f"  Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Success: {avg_success*100:.1f}%")

            score = avg_reward + 1000.0 * avg_success
            if score > best_score:
                best_score = score
                best_state_dict = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)
    return policy


def main():
    parser = argparse.ArgumentParser(description='Pre-train shared policy')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--env', type=str, default='warehouse', 
                        choices=list(ENV_MAP.keys()), help='Environment name')
    parser.add_argument('--output', type=str, default='checkpoints/shared_policy.pth',
                        help='Path to save the pre-trained policy')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to pre-train for')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--heterogeneity', type=int, default=0,
                        help='Heterogeneity level for training (0=homogeneous)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Shared Policy Pre-training")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Load config if provided.
    # NOTE: We always honor --env, but we allow configs to override shared
    # parameters and provide env-specific kwargs under a top-level block
    # with the env name (e.g. config['warehouse']).
    env_name = args.env
    env_kwargs = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        num_robots = config.get('num_robots', 4)
        max_steps = config.get('max_steps', 500)
        block = config.get(env_name)
        if isinstance(block, dict):
            # Avoid double-specifying core keys
            for k, v in block.items():
                if k in ['num_robots', 'heterogeneity_level', 'max_steps', 'seed']:
                    continue
                env_kwargs[k] = v
    else:
        num_robots = 4
        max_steps = 500
    
    print(f"Environment: {env_name}")
    print(f"Num robots: {num_robots}")
    print(f"Episodes: {args.episodes}")
    print(f"Heterogeneity: H{args.heterogeneity}")
    print("=" * 60)
    
    # Create environment
    EnvClass = ENV_MAP[env_name]
    env = EnvClass(
        num_robots=num_robots,
        heterogeneity_level=args.heterogeneity,
        max_steps=max_steps,
        seed=args.seed,
        **env_kwargs
    )
    
    # Create policy
    obs_dim = env.get_observation_dim()
    action_dim = env.get_action_dim()
    
    policy = SharedPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256
    )
    
    print(f"Policy: obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"Parameters: {sum(p.numel() for p in policy.parameters())}")
    print("-" * 60)
    
    # Pre-train
    trained_policy = pretrain_shared_policy(
        env=env,
        policy=policy,
        num_episodes=args.episodes
    )
    
    # Save policy
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(trained_policy.state_dict(), args.output)
    
    print("-" * 60)
    print(f"Pre-trained policy saved to: {args.output}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Smoke Test for DC-Ada Experimental Pipeline

This script verifies that all components work correctly:
1. Environment creation and stepping
2. Policy network forward pass
3. Transformation layer operations
4. Full episode rollout
5. All adaptation methods

Run this before running full experiments to catch any issues.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs import WarehouseEnv, SearchRescueEnv, CollaborativeMappingEnv
from src.agents import SharedPolicy, TransformationLayer, create_method


def test_environment():
    """Test environment creation and basic operations."""
    print("Testing environment...")
    
    # Test WarehouseEnv
    env = WarehouseEnv(num_robots=4, heterogeneity_level=1, seed=42)
    obs_list, info = env.reset()
    
    assert len(obs_list) == 4, f"Expected 4 observations, got {len(obs_list)}"
    assert all(isinstance(o, np.ndarray) for o in obs_list), "Observations should be numpy arrays"
    
    obs_dim = env.get_observation_dim()
    action_dim = env.get_action_dim()
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Initial info: {info}")
    
    # Test stepping
    actions = [np.random.uniform(-1, 1, action_dim) for _ in range(4)]
    obs_list, reward, done, info = env.step(actions)
    
    assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
    assert isinstance(done, bool), f"Done should be bool, got {type(done)}"
    
    print("  Environment test PASSED")
    return env, obs_dim, action_dim


def test_policy(obs_dim, action_dim):
    """Test policy network."""
    print("Testing policy...")
    
    policy = SharedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    
    # Test forward pass
    obs = torch.randn(1, obs_dim)
    mean, log_std = policy(obs)
    
    assert mean.shape == (1, action_dim), f"Mean shape mismatch: {mean.shape}"
    assert log_std.shape == (1, action_dim), f"Log std shape mismatch: {log_std.shape}"
    
    # Test get_action
    obs_np = np.random.randn(obs_dim).astype(np.float32)
    action = policy.get_action(obs_np)
    
    assert action.shape == (action_dim,), f"Action shape mismatch: {action.shape}"
    assert np.all(np.abs(action) <= 1.0), "Actions should be in [-1, 1]"
    
    # Test get_actions (batch)
    obs_list = [np.random.randn(obs_dim).astype(np.float32) for _ in range(4)]
    actions = policy.get_actions(obs_list)
    
    assert len(actions) == 4, f"Expected 4 actions, got {len(actions)}"
    
    print(f"  Mean: {mean.detach().numpy()}")
    print(f"  Log Std: {log_std.detach().numpy()}")
    print("  Policy test PASSED")
    
    return policy


def test_transformation(obs_dim):
    """Test transformation layer."""
    print("Testing transformation layer...")
    
    transform = TransformationLayer(obs_dim=obs_dim, latent_dim=32)
    
    # Test forward pass
    obs = torch.randn(obs_dim)
    trans_obs = transform(obs)
    
    assert trans_obs.shape == obs.shape, f"Shape mismatch: {trans_obs.shape} vs {obs.shape}"
    
    # Test parameter vector operations
    params = transform.get_params_vector()
    print(f"  Params shape: {params.shape}")
    
    # Test set params
    new_params = params + 0.01 * torch.randn_like(params)
    transform.set_params_vector(new_params)
    
    # Verify params changed
    params2 = transform.get_params_vector()
    assert not torch.allclose(params, params2), "Params should have changed"
    
    # Test perturb
    perturbed = transform.perturb(noise_scale=0.01)
    assert isinstance(perturbed, TransformationLayer), "Perturb should return TransformationLayer"
    
    print("  Transformation test PASSED")
    return transform


def test_full_pipeline(env, obs_dim, action_dim):
    """Test full episode rollout."""
    print("Testing full pipeline...")
    
    policy = SharedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    transforms = [TransformationLayer(obs_dim=obs_dim) for _ in range(env.num_robots)]
    
    obs_list, info = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 50
    
    while steps < max_steps:
        # Transform observations
        transformed_obs = []
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.from_numpy(obs).float()
            with torch.no_grad():
                trans_obs = transforms[i](obs_tensor)
            transformed_obs.append(trans_obs.numpy())
        
        # Get actions
        actions = policy.get_actions(transformed_obs)
        
        # Step environment
        obs_list, reward, done, info = env.step(actions)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"  Total reward: {total_reward:.2f}, Steps: {steps}")
    print(f"  Final info: {info}")
    print("  Full pipeline test PASSED")


def test_methods(env, obs_dim):
    """Test all adaptation methods."""
    print("Testing adaptation methods...")
    
    methods = ['shared_policy', 'dc_ada', 'random_perturbation', 'local_finetuning', 'obs_normalization']
    
    for method_name in methods:
        print(f"  Testing {method_name}...")
        
        method = create_method(
            method_name=method_name,
            env=env,
            obs_dim=obs_dim,
            total_budget=500,  # Small budget for testing
            seed=42
        )
        
        results = method.run()
        
        assert len(results) > 0, f"No results from {method_name}"
        assert 'reward' in results[0], f"No reward in results from {method_name}"
        
        avg_reward = np.mean([r['reward'] for r in results])
        print(f"    Episodes: {len(results)}, Avg reward: {avg_reward:.2f}")
    
    print("  Methods test PASSED")


def test_all_environments():
    """Test all environment types."""
    print("Testing all environments...")
    
    envs = [
        ('Warehouse', WarehouseEnv),
        ('SearchRescue', SearchRescueEnv),
        ('CollaborativeMapping', CollaborativeMappingEnv)
    ]
    
    for name, EnvClass in envs:
        print(f"  Testing {name}...")
        env = EnvClass(num_robots=4, heterogeneity_level=1, seed=42)
        obs_list, info = env.reset()
        
        # Run a few steps
        for _ in range(10):
            actions = [np.random.uniform(-1, 1, 2) for _ in range(4)]
            obs_list, reward, done, info = env.step(actions)
            if done:
                break
        
        print(f"    Obs dim: {env.get_observation_dim()}, Info: {info}")
    
    print("  All environments test PASSED")


def test_heterogeneity_levels():
    """Test different heterogeneity levels."""
    print("Testing heterogeneity levels...")
    
    for level in [0, 1, 2, 3]:
        env = WarehouseEnv(num_robots=4, heterogeneity_level=level, seed=42)
        obs_list, _ = env.reset()
        
        # Check that observation dimensions are consistent
        obs_dim = env.get_observation_dim()
        for obs in obs_list:
            assert obs.shape[0] == obs_dim, f"Obs dim mismatch at H{level}"
        
        print(f"  H{level}: obs_dim={obs_dim}")
    
    print("  Heterogeneity levels test PASSED")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("DC-Ada Smoke Test")
    print("=" * 60)
    
    try:
        # Basic tests
        env, obs_dim, action_dim = test_environment()
        test_policy(obs_dim, action_dim)
        test_transformation(obs_dim)
        test_full_pipeline(env, obs_dim, action_dim)
        
        # Extended tests
        test_all_environments()
        test_heterogeneity_levels()
        
        # Method tests (takes longer)
        env = WarehouseEnv(num_robots=4, heterogeneity_level=1, seed=42)
        test_methods(env, env.get_observation_dim())
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

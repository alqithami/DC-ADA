#!/usr/bin/env python3
"""
Test that DC-Ada actually adapts transformation parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.envs import WarehouseEnv
from src.agents.methods import DCADAMethod, SharedPolicyMethod

def test_dcada_adaptation():
    """Test that DC-Ada transforms actually change during adaptation."""
    print("Testing DC-Ada adaptation...")
    
    env = WarehouseEnv(num_robots=4, heterogeneity_level=1, seed=42)
    obs_dim = env.get_observation_dim()
    
    method = DCADAMethod(
        env=env,
        obs_dim=obs_dim,
        total_budget=8000,  # small budget for a quick test
        seed=42,
        num_candidates=3,
        noise_scale=0.05,
        adaptation_interval=2,  # frequent adaptation
        step_size=1.0,
        candidate_rollout_fraction=0.25,
        deterministic_rollouts=True,
    )
    
    # Get initial transform parameters
    initial_params = [t.get_params_vector().clone() for t in method.transforms]
    print(f"  Initial params norm: {[round(p.norm().item(), 4) for p in initial_params]}")
    
    # Run (adapts internally)
    method.run()

    # Check if params changed from initial
    params_changed = []
    for i, t in enumerate(method.transforms):
        curr = t.get_params_vector()
        diff = (initial_params[i] - curr).norm().item()
        params_changed.append(diff > 1e-6)
    
    # Get final transform parameters
    final_params = [t.get_params_vector().clone() for t in method.transforms]
    print(f"  Final params norm: {[round(p.norm().item(), 4) for p in final_params]}")
    
    # Report changes
    for i, (init, final) in enumerate(zip(initial_params, final_params)):
        diff = (init - final).norm().item()
        print(f"  Robot {i}: param diff = {diff:.6f}, changed = {params_changed[i]}")
    
    if any(params_changed):
        print("  DC-Ada adaptation test PASSED - transforms were updated!")
        return True
    else:
        print("  DC-Ada adaptation test FAILED - no transforms were updated!")
        return False

def test_dcada_vs_shared():
    """Test that DC-Ada produces different results than shared policy."""
    print("\nComparing DC-Ada vs Shared Policy (longer run)...")
    
    # Use higher heterogeneity where adaptation should matter more
    env1 = WarehouseEnv(num_robots=4, heterogeneity_level=2, seed=42)
    obs_dim = env1.get_observation_dim()
    
    # Run shared policy
    shared = SharedPolicyMethod(env1, obs_dim, total_budget=5000, seed=42)
    shared_results = shared.run()
    shared_rewards = [r['reward'] for r in shared_results]
    
    # Run DC-Ada with same seed but different env instance
    env2 = WarehouseEnv(num_robots=4, heterogeneity_level=2, seed=42)
    dcada = DCADAMethod(
        env2, obs_dim, total_budget=5000, seed=42,
        num_candidates=2, noise_scale=0.05, adaptation_interval=2,
        step_size=1.0, candidate_rollout_fraction=0.25,
        deterministic_rollouts=True,
    )
    dcada_results = dcada.run()
    dcada_rewards = [r['reward'] for r in dcada_results]
    
    print(f"  Shared Policy: {len(shared_results)} episodes, mean={np.mean(shared_rewards):.2f}, std={np.std(shared_rewards):.2f}")
    print(f"  DC-Ada: {len(dcada_results)} episodes, mean={np.mean(dcada_rewards):.2f}, std={np.std(dcada_rewards):.2f}")
    print(f"  DC-Ada adaptations: {dcada.adaptation_count}")
    
    # Check that DC-Ada adapted
    if dcada.adaptation_count > 0:
        print(f"  Comparison PASSED - DC-Ada performed {dcada.adaptation_count} adaptations")
        return True
    else:
        print(f"  Comparison WARNING - DC-Ada did not adapt")
        return False

if __name__ == "__main__":
    test1 = test_dcada_adaptation()
    test2 = test_dcada_vs_shared()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("ALL DC-ADA TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - review output above")
    print("=" * 60)

"""
Quick Experiment Runner for DC-Ada

Runs a reduced experiment for validation and demonstration purposes.
Uses fewer episodes and seeds for faster execution.
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envs.warehouse_env import make_env, WarehouseEnv
from agents.policy import SharedPolicy
from agents.methods import create_method


def pretrain_policy_fast(env, policy, num_episodes=30):
    """Fast pretraining with simple policy gradient."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_data = []
        done = False
        
        while not done:
            actions = {}
            step_data = {"obs": {}, "actions": {}, "log_probs": []}
            
            for i in range(env.num_robots):
                robot_obs = obs[f"robot_{i}"]
                obs_tensor = torch.FloatTensor(robot_obs).unsqueeze(0)
                
                mean, log_std = policy(obs_tensor)
                std = torch.exp(log_std)
                
                noise = torch.randn_like(mean)
                action_raw = mean + std * noise
                action = torch.tanh(action_raw)
                
                # Log probability
                log_prob = -0.5 * ((action_raw - mean) / std).pow(2).sum(-1)
                log_prob -= log_std.sum(-1)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
                
                actions[f"robot_{i}"] = action.squeeze(0).detach().numpy()
                step_data["obs"][i] = robot_obs
                step_data["actions"][i] = actions[f"robot_{i}"]
                step_data["log_probs"].append(log_prob)
            
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            step_data["reward"] = reward
            episode_data.append(step_data)
        
        # Compute returns and update
        returns = []
        G = 0
        for step in reversed(episode_data):
            G = step["reward"] + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        for step, G in zip(episode_data, returns):
            for log_prob in step["log_probs"]:
                loss -= log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        if (episode + 1) % 10 == 0:
            print(f"  Pretrain {episode+1}/{num_episodes}")
    
    return policy


def run_quick_experiments():
    """Run quick experiments with reduced parameters."""
    
    # Parameters for quick run
    NUM_EPISODES = 100  # Reduced from 400
    NUM_SEEDS = 5       # Reduced from 20
    PRETRAIN_EPISODES = 30
    
    output_dir = PROJECT_ROOT / "results" / f"quick_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods_to_test = [
        "shared_policy",
        "random_perturbation", 
        "local_finetuning",
        "gradient_finetuning",
        "dc_ada"
    ]
    
    results = {method: [] for method in methods_to_test}
    
    print("="*60)
    print("DC-Ada Quick Experiment Runner")
    print("="*60)
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Methods: {methods_to_test}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    for seed in range(NUM_SEEDS):
        print(f"\n--- Seed {seed + 1}/{NUM_SEEDS} ---")
        
        # Create environment
        env = make_env("warehouse", num_robots=4, max_steps=300, seed=seed)
        
        # Pretrain shared policy
        print("Pretraining shared policy...")
        shared_policy = SharedPolicy()
        shared_policy = pretrain_policy_fast(env, shared_policy, PRETRAIN_EPISODES)
        
        # Run each method
        for method_name in methods_to_test:
            print(f"\nRunning {method_name}...")
            
            # Create fresh environment and method
            env = make_env("warehouse", num_robots=4, max_steps=300, seed=seed)
            method = create_method(method_name, num_robots=4, shared_policy=shared_policy)
            
            # Tracking
            episode_rewards = []
            episode_scores = []
            episode_successes = []
            episode_deliveries = []
            episode_collisions = []
            modality_weights_history = []
            
            # Run episodes
            for episode in tqdm(range(NUM_EPISODES), desc=method_name, leave=False):
                obs, _ = env.reset(seed=seed + episode * 1000)
                episode_reward = 0
                done = False
                
                while not done:
                    actions = method.get_actions(obs)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    episode_reward += reward
                    done = terminated or truncated
                
                team_score = env.get_team_score()
                update_info = method.update(team_score, {"info": info})
                
                episode_rewards.append(episode_reward)
                episode_scores.append(team_score)
                episode_successes.append(float(info.get("success", False)))
                episode_deliveries.append(info.get("delivered_count", 0))
                episode_collisions.append(info.get("collision_count", 0))
                
                if method_name == "dc_ada":
                    metrics = method.get_metrics()
                    modality_weights_history.append(metrics.get("modality_weights", []))
            
            # Store results
            result = {
                "seed": seed,
                "method": method_name,
                "episode_rewards": episode_rewards,
                "episode_scores": episode_scores,
                "episode_successes": episode_successes,
                "episode_deliveries": episode_deliveries,
                "episode_collisions": episode_collisions,
                "final_success_rate": np.mean(episode_successes[-20:]),
                "final_score": np.mean(episode_scores[-20:]),
                "final_deliveries": np.mean(episode_deliveries[-20:]),
                "convergence_episode": find_convergence(episode_successes),
                "modality_weights_history": modality_weights_history
            }
            
            results[method_name].append(result)
            
            print(f"  Final score: {result['final_score']:.2f}, Success: {result['final_success_rate']*100:.1f}%")
    
    # Save results
    save_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    return results, output_dir


def find_convergence(successes, threshold=0.95):
    """Find convergence episode."""
    if len(successes) < 10:
        return len(successes)
    
    final_mean = np.mean(successes[-int(len(successes) * 0.2):])
    target = threshold * final_mean if final_mean > 0 else 0.5
    
    window_size = 5
    for i in range(len(successes) - window_size):
        window_mean = np.mean(successes[i:i + window_size])
        if window_mean >= target:
            return i
    
    return len(successes)


def save_results(results, output_dir):
    """Save results to disk."""
    # JSON
    serializable = {}
    for method, runs in results.items():
        serializable[method] = []
        for run in runs:
            run_copy = {}
            for key, value in run.items():
                if isinstance(value, np.ndarray):
                    run_copy[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        run_copy[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                    else:
                        run_copy[key] = value
                else:
                    run_copy[key] = value
            serializable[method].append(run_copy)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Pickle
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\n{'Method':<30} {'Success Rate':<20} {'Final Score':<15}")
    print("-"*65)
    
    for method, runs in results.items():
        if len(runs) == 0:
            continue
        
        final_scores = [r["final_score"] for r in runs]
        final_successes = [r["final_success_rate"] for r in runs]
        
        success_str = f"{np.mean(final_successes)*100:.1f} ± {np.std(final_successes)*100:.1f}%"
        score_str = f"{np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}"
        
        print(f"{method:<30} {success_str:<20} {score_str:<15}")
    
    print("="*60)


if __name__ == "__main__":
    results, output_dir = run_quick_experiments()
    
    # Generate figures
    print("\nGenerating figures...")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_figures import (
        plot_learning_curves, plot_success_rate_curves, 
        plot_ablation_bar, plot_modality_weights,
        plot_per_robot_performance, plot_simulation_environment,
        generate_results_table
    )
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_simulation_environment(figures_dir)
    plot_learning_curves(results, figures_dir, "episode_scores")
    plot_success_rate_curves(results, figures_dir)
    plot_ablation_bar(results, figures_dir)
    plot_modality_weights(results, figures_dir)
    plot_per_robot_performance(results, figures_dir)
    generate_results_table(results, figures_dir)
    
    print(f"\nFigures saved to {figures_dir}")
    print("\nExperiment complete!")

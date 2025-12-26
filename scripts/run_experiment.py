"""
Main Experiment Runner for DC-Ada

This script runs complete experiments comparing DC-Ada against baselines
across multiple seeds and environments.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envs.warehouse_env import make_env
from agents.policy import SharedPolicy, pretrain_shared_policy
from agents.methods import create_method


class ExperimentRunner:
    """
    Runs experiments comparing DC-Ada against baselines.
    """
    
    def __init__(
        self,
        env_type: str = "warehouse",
        num_robots: int = 4,
        num_episodes: int = 400,
        max_steps: int = 500,
        num_seeds: int = 20,
        methods: List[str] = None,
        output_dir: str = None,
        pretrain_episodes: int = 50
    ):
        self.env_type = env_type
        self.num_robots = num_robots
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_seeds = num_seeds
        self.pretrain_episodes = pretrain_episodes
        
        if methods is None:
            self.methods = [
                "shared_policy",
                "random_perturbation",
                "local_finetuning",
                "gradient_finetuning",
                "dc_ada"
            ]
        else:
            self.methods = methods
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = PROJECT_ROOT / "results" / f"{env_type}_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {method: [] for method in self.methods}
    
    def run_all_experiments(self):
        """Run experiments for all methods and seeds."""
        print(f"\n{'='*60}")
        print(f"DC-Ada Experiment Runner")
        print(f"{'='*60}")
        print(f"Environment: {self.env_type}")
        print(f"Robots: {self.num_robots}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Seeds: {self.num_seeds}")
        print(f"Methods: {self.methods}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for seed in range(self.num_seeds):
            print(f"\n--- Seed {seed + 1}/{self.num_seeds} ---")
            
            # Create environment for pretraining
            env = make_env(self.env_type, num_robots=self.num_robots, 
                          max_steps=self.max_steps, seed=seed)
            
            # Pretrain shared policy
            print("Pretraining shared policy...")
            shared_policy = SharedPolicy()
            shared_policy = pretrain_shared_policy(
                env, shared_policy, num_episodes=self.pretrain_episodes
            )
            
            # Run each method
            for method_name in self.methods:
                print(f"\nRunning {method_name}...")
                result = self.run_single_experiment(
                    method_name, shared_policy, seed
                )
                self.results[method_name].append(result)
                
                # Save intermediate results
                self.save_results()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def run_single_experiment(
        self,
        method_name: str,
        shared_policy: SharedPolicy,
        seed: int
    ) -> Dict:
        """Run a single experiment for one method and seed."""
        
        # Create fresh environment
        env = make_env(self.env_type, num_robots=self.num_robots,
                      max_steps=self.max_steps, seed=seed)
        
        # Create method
        method = create_method(method_name, self.num_robots, shared_policy)
        
        # Tracking metrics
        episode_rewards = []
        episode_scores = []
        episode_successes = []
        episode_deliveries = []
        episode_collisions = []
        episode_steps = []
        modality_weights_history = []
        
        # Run episodes
        for episode in tqdm(range(self.num_episodes), desc=method_name, leave=False):
            obs, _ = env.reset(seed=seed + episode * 1000)
            
            episode_reward = 0
            done = False
            
            while not done:
                # Get actions from method
                actions = method.get_actions(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                episode_reward += reward
                done = terminated or truncated
            
            # Get team score
            team_score = env.get_team_score()
            
            # Update method
            update_info = method.update(team_score, {"info": info})
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_scores.append(team_score)
            episode_successes.append(float(info.get("success", False)))
            episode_deliveries.append(info.get("delivered_count", 0))
            episode_collisions.append(info.get("collision_count", 0))
            episode_steps.append(info.get("step_count", 0))
            
            # Record modality weights for DC-Ada
            if method_name == "dc_ada":
                metrics = method.get_metrics()
                modality_weights_history.append(metrics.get("modality_weights", []))
        
        # Compile results
        result = {
            "seed": seed,
            "method": method_name,
            "episode_rewards": episode_rewards,
            "episode_scores": episode_scores,
            "episode_successes": episode_successes,
            "episode_deliveries": episode_deliveries,
            "episode_collisions": episode_collisions,
            "episode_steps": episode_steps,
            "final_success_rate": np.mean(episode_successes[-50:]),
            "final_score": np.mean(episode_scores[-50:]),
            "final_deliveries": np.mean(episode_deliveries[-50:]),
            "convergence_episode": self._find_convergence(episode_successes),
            "modality_weights_history": modality_weights_history
        }
        
        return result
    
    def _find_convergence(self, successes: List[float], threshold: float = 0.95) -> int:
        """Find the episode where performance converges."""
        if len(successes) < 20:
            return len(successes)
        
        final_mean = np.mean(successes[-int(len(successes) * 0.2):])
        target = threshold * final_mean
        
        window_size = 10
        for i in range(len(successes) - window_size):
            window_mean = np.mean(successes[i:i + window_size])
            if window_mean >= target:
                # Check if it stays above
                remaining = successes[i:]
                if np.mean(remaining) >= target:
                    return i
        
        return len(successes)
    
    def save_results(self):
        """Save results to disk."""
        results_file = self.output_dir / "results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, runs in self.results.items():
            serializable_results[method] = []
            for run in runs:
                run_copy = {}
                for key, value in run.items():
                    if isinstance(value, np.ndarray):
                        run_copy[key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], np.ndarray):
                            run_copy[key] = [v.tolist() for v in value]
                        else:
                            run_copy[key] = value
                    else:
                        run_copy[key] = value
                serializable_results[method].append(run_copy)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save as pickle for full fidelity
        pickle_file = self.output_dir / "results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
    
    def generate_summary(self):
        """Generate summary statistics."""
        summary = {}
        
        for method, runs in self.results.items():
            if len(runs) == 0:
                continue
            
            final_scores = [r["final_score"] for r in runs]
            final_successes = [r["final_success_rate"] for r in runs]
            convergence_eps = [r["convergence_episode"] for r in runs]
            
            summary[method] = {
                "final_score_mean": np.mean(final_scores),
                "final_score_std": np.std(final_scores),
                "final_success_mean": np.mean(final_successes),
                "final_success_std": np.std(final_successes),
                "convergence_mean": np.mean(convergence_eps),
                "convergence_std": np.std(convergence_eps),
                "num_runs": len(runs)
            }
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"\n{'Method':<25} {'Success Rate':<20} {'Final Score':<20}")
        print("-"*65)
        
        for method, stats in summary.items():
            success = f"{stats['final_success_mean']*100:.1f} ± {stats['final_success_std']*100:.1f}%"
            score = f"{stats['final_score_mean']:.2f} ± {stats['final_score_std']:.2f}"
            print(f"{method:<25} {success:<20} {score:<20}")
        
        print("="*60)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run DC-Ada experiments")
    parser.add_argument("--env", type=str, default="warehouse",
                       choices=["warehouse", "search_rescue", "mapping"],
                       help="Environment type")
    parser.add_argument("--robots", type=int, default=4,
                       help="Number of robots")
    parser.add_argument("--episodes", type=int, default=400,
                       help="Number of episodes per experiment")
    parser.add_argument("--seeds", type=int, default=20,
                       help="Number of random seeds")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                       help="Methods to run")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--pretrain", type=int, default=50,
                       help="Pretraining episodes")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        env_type=args.env,
        num_robots=args.robots,
        num_episodes=args.episodes,
        num_seeds=args.seeds,
        methods=args.methods,
        output_dir=args.output,
        pretrain_episodes=args.pretrain
    )
    
    runner.run_all_experiments()


if __name__ == "__main__":
    main()

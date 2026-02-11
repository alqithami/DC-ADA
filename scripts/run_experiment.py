#!/usr/bin/env python3
"""
Main Experiment Runner for DC-Ada

This script runs experiments comparing DC-Ada against baselines
across different environments and heterogeneity levels.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --env warehouse --heterogeneity 2 --seeds 5
"""

import sys
import os
import argparse
import json
import time
import platform
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from src.envs import WarehouseEnv, SearchRescueEnv, CollaborativeMappingEnv
from src.agents import create_method


# Environment mapping
ENV_MAP = {
    'warehouse': WarehouseEnv,
    'search_rescue': SearchRescueEnv,
    'mapping': CollaborativeMappingEnv
}

# Default methods to compare
DEFAULT_METHODS = [
    'shared_policy',
    'dc_ada',
    'random_perturbation',
    'local_finetuning',
    'obs_normalization'
]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(
    env_name: str,
    heterogeneity_level: int,
    seed: int,
    env_params: dict | None = None,
    **kwargs
):
    """Create environment instance.

    If env_params is provided, it should be a dict mapping env names to
    constructor kwargs, e.g. env_params['warehouse'] = {'target_deliveries': 2}.
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_MAP.keys())}")

    EnvClass = ENV_MAP[env_name]

    # Base kwargs (shared across envs)
    env_kwargs = {
        'num_robots': kwargs.get('num_robots', 4),
        'heterogeneity_level': heterogeneity_level,
        'max_steps': kwargs.get('max_steps', 500),
        'seed': seed,
    }

    # Environment-specific overrides from YAML
    if isinstance(env_params, dict):
        overrides = env_params.get(env_name)
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                # Avoid double-specifying the core keys
                if k in env_kwargs:
                    continue
                env_kwargs[k] = v

    return EnvClass(**env_kwargs)


def run_single_experiment(
    env_name: str,
    method_name: str,
    heterogeneity_level: int,
    seed: int,
    total_budget: int = 50000,
    method_params: dict | None = None,
    pretrained_policies: dict | None = None,
    env_params: dict | None = None,
    **kwargs
) -> dict:
    """
    Run a single experiment with one method, environment, and seed.
    
    Returns:
        Dictionary with experiment results
    """
    print(f"  Running {method_name} on {env_name} H{heterogeneity_level} seed={seed}...")
    
    # Create environment
    env = create_env(env_name, heterogeneity_level, seed, env_params=env_params, **kwargs)
    obs_dim = env.get_observation_dim()
    
    # Build method kwargs.
    # 1) Start with global kwargs (excluding env-construction keys)
    method_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ['num_robots', 'max_steps', 'method_params', 'pretrained_policies']
    }

    # 2) Add method-specific hyperparameters from config (if provided)
    if isinstance(method_params, dict) and method_name in method_params:
        mp = method_params.get(method_name) or {}
        if isinstance(mp, dict):
            method_kwargs.update(mp)

    # 3) Add pretrained policy checkpoint for this environment (optional)
    if isinstance(pretrained_policies, dict):
        pretrained_path = pretrained_policies.get(env_name)
        if pretrained_path:
            method_kwargs['pretrained_path'] = pretrained_path
    method = create_method(
        method_name=method_name,
        env=env,
        obs_dim=obs_dim,
        total_budget=total_budget,
        seed=seed,
        **method_kwargs
    )
    
    # Run experiment
    start_time = time.time()
    results = method.run()
    total_time = time.time() - start_time
    
    # Compute summary statistics
    rewards = [r['reward'] for r in results]
    successes = [r['success'] for r in results]

    # Optional task-progress metrics (useful when success_rate is 0.0)
    extra_summary = {}
    extra_print = []
    if results:
        if 'delivered_count' in results[0]:
            delivered_counts = [r.get('delivered_count', 0) for r in results]
            delivery_ratios = [r.get('delivery_ratio', 0.0) for r in results]
            extra_summary['mean_delivered_count'] = float(np.mean(delivered_counts))
            extra_summary['mean_delivery_ratio'] = float(np.mean(delivery_ratios))
            extra_print.append(
                f"Delivered: {extra_summary['mean_delivered_count']:.2f} "
                f"(ratio={extra_summary['mean_delivery_ratio']:.2f})"
            )
        if 'rescued_count' in results[0]:
            rescued_counts = [r.get('rescued_count', 0) for r in results]
            rescue_ratios = [r.get('rescue_ratio', 0.0) for r in results]
            extra_summary['mean_rescued_count'] = float(np.mean(rescued_counts))
            extra_summary['mean_rescue_ratio'] = float(np.mean(rescue_ratios))
            extra_print.append(
                f"Rescued: {extra_summary['mean_rescued_count']:.2f} "
                f"(ratio={extra_summary['mean_rescue_ratio']:.2f})"
            )
        if 'coverage' in results[0]:
            coverages = [r.get('coverage', 0.0) for r in results]
            extra_summary['mean_coverage'] = float(np.mean(coverages))
            extra_print.append(f"Coverage: {extra_summary['mean_coverage']:.2f}")
    
    summary = {
        'env_name': env_name,
        'method_name': method_name,
        'heterogeneity_level': heterogeneity_level,
        'seed': seed,
        'total_budget': total_budget,
        'num_episodes': len(results),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'success_rate': float(np.mean(successes)),
        'total_time': total_time,
        'steps_used': int(getattr(method, 'steps_used', 0)),
        'communication_bytes': int(getattr(method, 'communication_bytes', 0)),
        'episode_results': results,
        **extra_summary,
    }
    
    msg = (
        f"    Episodes: {len(results)}, Mean reward: {summary['mean_reward']:.2f}, "
        f"Success rate: {summary['success_rate']:.1%}, "
        f"Steps: {summary['steps_used']}, Time: {summary['total_time']:.1f}s"
    )
    if extra_print:
        msg += ", " + ", ".join(extra_print)
    print(msg)
    
    return summary


def run_experiment_suite(
    env_names: list,
    method_names: list,
    heterogeneity_levels: list,
    seeds: list,
    total_budget: int = 50000,
    output_dir: str = 'results',
    method_params: dict | None = None,
    pretrained_policies: dict | None = None,
    env_params: dict | None = None,
    **kwargs
) -> dict:
    """
    Run full experiment suite across all combinations.
    
    Returns:
        Dictionary with all results
    """
    # Provenance / reproducibility metadata (helpful for reviewers)
    try:
        import torch  # Optional dependency elsewhere, but present in this repo
        torch_version = torch.__version__
    except Exception:
        torch_version = None

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'env_names': env_names,
            'method_names': method_names,
            'heterogeneity_levels': heterogeneity_levels,
            'seeds': seeds,
            'total_budget': total_budget,
            'num_robots': kwargs.get('num_robots', 4),
            'max_steps': kwargs.get('max_steps', 500),
            'pretrained_policies': pretrained_policies or {},
            'method_params': method_params or {},
            'env_params': env_params or {},
            'system': {
                'python': sys.version,
                'platform': platform.platform(),
                'numpy': np.__version__,
                'torch': torch_version,
            },
        },
        'experiments': []
    }
    
    total_runs = len(env_names) * len(method_names) * len(heterogeneity_levels) * len(seeds)
    current_run = 0
    
    for env_name in env_names:
        for h_level in heterogeneity_levels:
            for method_name in method_names:
                for seed in seeds:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] {env_name} H{h_level} {method_name} seed={seed}")
                    
                    try:
                        result = run_single_experiment(
                            env_name=env_name,
                            method_name=method_name,
                            heterogeneity_level=h_level,
                            seed=seed,
                            total_budget=total_budget,
                            method_params=method_params,
                            pretrained_policies=pretrained_policies,
                            env_params=env_params,
                            **kwargs
                        )
                        all_results['experiments'].append(result)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        all_results['experiments'].append({
                            'env_name': env_name,
                            'method_name': method_name,
                            'heterogeneity_level': h_level,
                            'seed': seed,
                            'error': str(e)
                        })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'results_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def print_summary(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Group by environment and heterogeneity level
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    
    for exp in results['experiments']:
        if 'error' in exp:
            continue
        key = (exp['env_name'], exp['heterogeneity_level'])
        grouped[key][exp['method_name']].append(exp['mean_reward'])
    
    for (env_name, h_level), methods in sorted(grouped.items()):
        print(f"\n{env_name} H{h_level}:")
        print("-" * 50)
        for method_name, rewards in sorted(methods.items()):
            mean = np.mean(rewards)
            std = np.std(rewards)
            print(f"  {method_name:25s}: {mean:8.2f} ± {std:6.2f}")


def main():
    parser = argparse.ArgumentParser(description='Run DC-Ada experiments')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--env', type=str, default='warehouse', 
                        choices=list(ENV_MAP.keys()), help='Environment name')
    parser.add_argument('--methods', type=str, nargs='+', default=DEFAULT_METHODS,
                        help='Methods to compare')
    parser.add_argument('--heterogeneity', type=int, nargs='+', default=[1],
                        help='Heterogeneity levels (0-3)')
    parser.add_argument('--heterogeneity_level', type=int, default=None,
                        help='Single heterogeneity level (alias for --heterogeneity)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--budget', type=int, default=50000, help='Total environment steps budget')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--num-robots', type=int, default=4, help='Number of robots')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        env_names = config.get('environments', [args.env])
        method_names = config.get('methods', args.methods)
        heterogeneity_levels = config.get('heterogeneity_levels', args.heterogeneity)
        seeds = list(range(config.get('num_seeds', args.seeds)))
        total_budget = config.get('total_budget', args.budget)
        num_robots = config.get('num_robots', args.num_robots)
        max_steps = config.get('max_steps', args.max_steps)

        # Optional: per-method hyperparameters, e.g. config['dc_ada']
        method_params = {}
        for m in method_names:
            block = config.get(m)
            if isinstance(block, dict):
                method_params[m] = block

        # Optional: per-environment parameters (top-level blocks named after envs)
        env_params = {}
        for e in env_names:
            block = config.get(e)
            if isinstance(block, dict):
                env_params[e] = block

        # Optional: per-environment pretrained policy checkpoints
        pretrained_policies = config.get('pretrained_policies')
        if not isinstance(pretrained_policies, dict):
            pretrained_policies = None

        output_dir = config.get('output_dir', args.output)
    else:
        env_names = [args.env]
        method_names = args.methods
        # Handle --heterogeneity_level alias
        if args.heterogeneity_level is not None:
            heterogeneity_levels = [args.heterogeneity_level]
        else:
            heterogeneity_levels = args.heterogeneity
        seeds = list(range(args.seeds))
        total_budget = args.budget
        num_robots = args.num_robots
        max_steps = args.max_steps

        method_params = None
        env_params = None
        pretrained_policies = None
        output_dir = args.output
    
    print("=" * 80)
    print("DC-Ada Experiment Runner")
    print("=" * 80)
    print(f"Environments: {env_names}")
    print(f"Methods: {method_names}")
    print(f"Heterogeneity levels: {heterogeneity_levels}")
    print(f"Seeds: {seeds}")
    print(f"Budget: {total_budget} steps")
    print(f"Robots: {num_robots}")
    if pretrained_policies:
        print(f"Pretrained policies: {pretrained_policies}")
    print("=" * 80)
    
    # Run experiments
    results = run_experiment_suite(
        env_names=env_names,
        method_names=method_names,
        heterogeneity_levels=heterogeneity_levels,
        seeds=seeds,
        total_budget=total_budget,
        output_dir=output_dir,
        method_params=method_params,
        pretrained_policies=pretrained_policies,
        env_params=env_params,
        num_robots=num_robots,
        max_steps=max_steps
    )
    
    # Print summary
    print_summary(results)

    # Fail the run if any experiment crashed. This prevents accidentally
    # incorporating partial results into the paper.
    errors = [e for e in results.get('experiments', []) if isinstance(e, dict) and 'error' in e]
    if errors:
        print("\n" + "=" * 80)
        print(f"ERROR SUMMARY: {len(errors)} experiment runs failed")
        print("=" * 80)
        # Show up to 10 representative errors
        for i, e in enumerate(errors[:10], start=1):
            print(
                f"[{i}] {e.get('env_name')} H{e.get('heterogeneity_level')} "
                f"{e.get('method_name')} seed={e.get('seed')}: {e.get('error')}"
            )
        if len(errors) > 10:
            print(f"... ({len(errors)-10} more)")
        print("=" * 80)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

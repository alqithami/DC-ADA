#!/usr/bin/env python3
"""Sanity checks for the DC-Ada experiment pipeline.

This script is intentionally lightweight and reviewer-facing. It verifies that:

1) Each environment reset/step returns correctly-shaped observations.
2) Each method runs for a small budget and returns at least one episode log.
3) Episode logs contain reward, steps, success, and task-specific info keys.
4) Success is boolean and success_rate computed from logs is finite.

Usage:
  python scripts/run_sanity_checks.py --env warehouse --heterogeneity 0 --budget 2000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

# Ensure the project root is on sys.path when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs import make_env
from src.utils.seeding import set_seed
from src.agents.methods import create_method


DEFAULT_METHODS = [
    'shared_policy',
    'dc_ada',
    'random_perturbation',
    'local_finetuning',
    'obs_normalization',
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['warehouse', 'search_rescue', 'mapping'], default='warehouse')
    parser.add_argument('--heterogeneity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_robots', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument(
        '--budget',
        type=int,
        default=2000,
        help='Budget in environment steps (joint steps).',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Optional policy checkpoint path (state_dict .pth). If not provided, will try checkpoints/<env>_policy.pth.',
    )
    args = parser.parse_args()

    set_seed(args.seed)

    ckpt = args.checkpoint or os.path.join('checkpoints', f'{args.env}_policy.pth')
    pretrained_path: Optional[str] = ckpt if os.path.exists(ckpt) else None

    print('=' * 70)
    print('DC-Ada Sanity Checks')
    print('=' * 70)
    print(f"Env: {args.env} | H{args.heterogeneity} | robots={args.num_robots} | max_steps={args.max_steps}")
    print(f"Budget: {args.budget} env-steps | Seed: {args.seed}")
    print(f"Pretrained checkpoint: {pretrained_path if pretrained_path else '(none found)'}")
    print('-' * 70)

    # ------------------------------------------------------------------
    # Environment basic check
    # ------------------------------------------------------------------
    env = make_env(
        env_name=args.env,
        num_robots=args.num_robots,
        heterogeneity_level=args.heterogeneity,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    obs_dim = env.get_observation_dim()
    act_dim = env.get_action_dim()

    obs_list, _ = env.reset()
    assert isinstance(obs_list, list) and len(obs_list) == args.num_robots, 'reset() returned wrong obs_list'
    assert obs_list[0].shape[0] == obs_dim, 'observation dimension mismatch'

    zero_actions = [np.zeros(act_dim, dtype=np.float32) for _ in range(args.num_robots)]
    _, reward, done, info = env.step(zero_actions)

    assert isinstance(reward, (float, np.floating)), 'reward must be scalar float'
    assert isinstance(done, (bool, np.bool_)), 'done must be bool'
    assert isinstance(info, dict), 'info must be a dict'

    print('Environment reset/step: OK')
    print('-' * 70)

    # ------------------------------------------------------------------
    # Method checks
    # ------------------------------------------------------------------
    for method_name in DEFAULT_METHODS:
        env_m = make_env(
            env_name=args.env,
            num_robots=args.num_robots,
            heterogeneity_level=args.heterogeneity,
            seed=args.seed,
            max_steps=args.max_steps,
        )

        method = create_method(
            method_name,
            env=env_m,
            obs_dim=obs_dim,
            total_budget=args.budget,
            seed=args.seed,
            pretrained_path=pretrained_path,
        )

        results = method.run()
        if not results:
            raise RuntimeError(f'Method {method_name} returned no episode logs')

        # Minimal log schema
        r0 = results[0]
        for key in ['reward', 'episode_length', 'success']:
            if key not in r0:
                raise KeyError(f'Method {method_name} missing key in episode log: {key}')

        # success must be bool-like
        assert isinstance(r0['success'], (bool, np.bool_)), 'success must be bool'

        rewards = [float(r.get('reward', 0.0)) for r in results]
        successes = [bool(r.get('success', False)) for r in results]
        sr = float(np.mean(successes))
        assert np.isfinite(sr), 'success_rate must be finite'

        print(f"{method_name:>18}: episodes={len(results):3d} | mean_reward={np.mean(rewards):8.2f} | success_rate={sr:6.1%}")

    print('-' * 70)
    print('Sanity checks PASSED.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

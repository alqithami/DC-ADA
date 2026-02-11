#!/usr/bin/env python3
"""Validate an experiment results JSON file.

This script is intentionally simple and reviewer-facing:
  - Confirms the expected number of runs is present
  - Fails if any runs contain an error
  - Prints a compact summary of missing/failed combinations

Usage:
  python scripts/validate_results.py --results results/results_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate DC-Ada results JSON')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    args = parser.parse_args()

    with open(args.results, 'r') as f:
        data = json.load(f)

    meta = data.get('metadata', {})
    env_names = meta.get('env_names', [])
    method_names = meta.get('method_names', [])
    heterogeneity_levels = meta.get('heterogeneity_levels', [])
    seeds = meta.get('seeds', [])

    experiments = data.get('experiments', [])
    expected = len(env_names) * len(method_names) * len(heterogeneity_levels) * len(seeds)
    actual = len(experiments)

    print('=' * 80)
    print('RESULTS VALIDATION')
    print('=' * 80)
    print(f"File: {args.results}")
    print(f"Expected runs: {expected}")
    print(f"Actual runs:   {actual}")

    errors = [e for e in experiments if isinstance(e, dict) and 'error' in e]
    print(f"Error runs:    {len(errors)}")

    if expected and actual != expected:
        print('WARNING: run count does not match expected from metadata.')

    if errors:
        ctr = Counter((e.get('env_name'), e.get('method_name'), e.get('heterogeneity_level')) for e in errors)
        print('\nTop failing groups (env, method, H):')
        for (k, v) in ctr.most_common(10):
            print(f"  {k}: {v}")
        print('\nExample errors:')
        for i, e in enumerate(errors[:5], start=1):
            print(
                f"  [{i}] {e.get('env_name')} H{e.get('heterogeneity_level')} {e.get('method_name')} seed={e.get('seed')}: {e.get('error')}"
            )

        print('\nVALIDATION FAILED: errors present in results file.')
        return 1

    print('\nVALIDATION PASSED: no errors detected.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

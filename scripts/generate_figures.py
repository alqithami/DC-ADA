#!/usr/bin/env python3
"""
Figure Generation for DC-Ada Paper

This script generates publication-ready figures from experiment results:
1. Performance comparison bar charts
2. Learning curves
3. Heterogeneity scaling analysis
4. Communication overhead comparison

Usage:
    python scripts/generate_figures.py --results results/results_*.json --output figures/
"""

import sys
import os
import argparse
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette (colorblind-friendly)
COLORS = {
    'dc_ada': '#2E86AB',           # Blue
    'shared_policy': '#A23B72',     # Magenta
    'random_perturbation': '#F18F01', # Orange
    'local_finetuning': '#C73E1D',  # Red
    'obs_normalization': '#3B1F2B'  # Dark
}

METHOD_NAMES = {
    'dc_ada': 'DC-Ada (Ours)',
    'shared_policy': 'Shared Policy',
    'random_perturbation': 'Random Perturbation',
    'local_finetuning': 'Local Fine-Tuning',
    'obs_normalization': 'Obs. Normalization'
}


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_results(results: dict) -> dict:
    """Aggregate results by environment, method, and heterogeneity level."""
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for exp in results['experiments']:
        if 'error' in exp:
            continue
        
        env = exp['env_name']
        method = exp['method_name']
        h_level = exp['heterogeneity_level']
        
        aggregated[env][h_level][method].append({
            'mean_reward': exp['mean_reward'],
            'success_rate': exp['success_rate'],
            'total_time': exp.get('total_time', 0)
        })
    
    return aggregated


def plot_performance_comparison(aggregated: dict, output_dir: str):
    """Generate bar chart comparing methods across heterogeneity levels."""
    
    for env_name, h_levels in aggregated.items():
        fig, axes = plt.subplots(1, len(h_levels), figsize=(4 * len(h_levels), 5))
        if len(h_levels) == 1:
            axes = [axes]
        
        for ax, (h_level, methods) in zip(axes, sorted(h_levels.items())):
            method_names = []
            means = []
            stds = []
            colors = []
            
            for method, results in sorted(methods.items()):
                rewards = [r['mean_reward'] for r in results]
                method_names.append(METHOD_NAMES.get(method, method))
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
                colors.append(COLORS.get(method, '#888888'))
            
            x = np.arange(len(method_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title(f'H{h_level}: {"Homogeneous" if h_level == 0 else f"Heterogeneity Level {h_level}"}')
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(f'{env_name.replace("_", " ").title()} Environment', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'performance_{env_name}.pdf')
        plt.savefig(output_path)
        plt.savefig(output_path.replace('.pdf', '.png'))
        plt.close()
        print(f"Saved: {output_path}")


def plot_heterogeneity_scaling(aggregated: dict, output_dir: str):
    """Generate line plot showing performance vs heterogeneity level."""
    
    for env_name, h_levels in aggregated.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get all methods
        all_methods = set()
        for methods in h_levels.values():
            all_methods.update(methods.keys())
        
        for method in sorted(all_methods):
            h_values = []
            means = []
            stds = []
            
            for h_level in sorted(h_levels.keys()):
                if method in h_levels[h_level]:
                    results = h_levels[h_level][method]
                    rewards = [r['mean_reward'] for r in results]
                    h_values.append(h_level)
                    means.append(np.mean(rewards))
                    stds.append(np.std(rewards))
            
            if h_values:
                color = COLORS.get(method, '#888888')
                label = METHOD_NAMES.get(method, method)
                ax.errorbar(h_values, means, yerr=stds, label=label,
                           marker='o', capsize=5, color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('Heterogeneity Level')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'{env_name.replace("_", " ").title()}: Performance vs Heterogeneity')
        ax.set_xticks(sorted(h_levels.keys()))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'scaling_{env_name}.pdf')
        plt.savefig(output_path)
        plt.savefig(output_path.replace('.pdf', '.png'))
        plt.close()
        print(f"Saved: {output_path}")


def plot_success_rate_comparison(aggregated: dict, output_dir: str):
    """Generate success rate comparison."""
    
    for env_name, h_levels in aggregated.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        methods_list = []
        h_level_list = []
        success_rates = []
        
        for h_level, methods in sorted(h_levels.items()):
            for method, results in sorted(methods.items()):
                rates = [r['success_rate'] for r in results]
                methods_list.append(METHOD_NAMES.get(method, method))
                h_level_list.append(f'H{h_level}')
                success_rates.append(np.mean(rates) * 100)
        
        # Create grouped bar chart
        unique_methods = list(dict.fromkeys(methods_list))
        unique_h_levels = list(dict.fromkeys(h_level_list))
        
        x = np.arange(len(unique_h_levels))
        width = 0.8 / len(unique_methods)
        
        for i, method in enumerate(unique_methods):
            rates = []
            for h in unique_h_levels:
                idx = [j for j, (m, hl) in enumerate(zip(methods_list, h_level_list)) 
                       if m == method and hl == h]
                if idx:
                    rates.append(success_rates[idx[0]])
                else:
                    rates.append(0)
            
            offset = (i - len(unique_methods) / 2 + 0.5) * width
            color = COLORS.get([k for k, v in METHOD_NAMES.items() if v == method][0] 
                              if method in METHOD_NAMES.values() else method, '#888888')
            ax.bar(x + offset, rates, width, label=method, color=color, alpha=0.8)
        
        ax.set_xlabel('Heterogeneity Level')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{env_name.replace("_", " ").title()}: Task Success Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_h_levels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'success_rate_{env_name}.pdf')
        plt.savefig(output_path)
        plt.savefig(output_path.replace('.pdf', '.png'))
        plt.close()
        print(f"Saved: {output_path}")


def generate_latex_table(aggregated: dict, output_dir: str):
    """Generate LaTeX table for paper."""
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance comparison across heterogeneity levels. Mean reward $\pm$ std over 5 seeds.}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"Environment & Method & H0 & H1 & H2 & H3 \\")
    latex.append(r"\midrule")
    
    for env_name, h_levels in sorted(aggregated.items()):
        env_display = env_name.replace('_', ' ').title()
        
        # Get all methods
        all_methods = set()
        for methods in h_levels.values():
            all_methods.update(methods.keys())
        
        first_method = True
        for method in sorted(all_methods):
            method_display = METHOD_NAMES.get(method, method)
            
            row = []
            if first_method:
                row.append(env_display)
                first_method = False
            else:
                row.append("")
            
            row.append(method_display)
            
            for h_level in [0, 1, 2, 3]:
                if h_level in h_levels and method in h_levels[h_level]:
                    results = h_levels[h_level][method]
                    rewards = [r['mean_reward'] for r in results]
                    mean = np.mean(rewards)
                    std = np.std(rewards)
                    
                    # Bold the best result
                    is_best = True
                    for other_method, other_results in h_levels[h_level].items():
                        if other_method != method:
                            other_mean = np.mean([r['mean_reward'] for r in other_results])
                            if other_mean > mean:
                                is_best = False
                                break
                    
                    if is_best:
                        row.append(f"\\textbf{{{mean:.1f}}} $\\pm$ {std:.1f}")
                    else:
                        row.append(f"{mean:.1f} $\\pm$ {std:.1f}")
                else:
                    row.append("--")
            
            latex.append(" & ".join(row) + r" \\")
        
        latex.append(r"\midrule")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    output_path = os.path.join(output_dir, 'results_table.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_path}")


def plot_heatmaps(aggregated: dict, output_dir: str):
    """Generate compact heatmaps for quick reviewer-friendly comparisons.

    We generate one heatmap per environment for:
      - Mean episode reward
      - Success rate (%)

    Heatmaps provide a high-level view of how each method scales with
    heterogeneity (H0..H3) without requiring the reader to parse many plots.
    """

    for env_name, h_levels in aggregated.items():
        # Consistent ordering
        h_vals = sorted(h_levels.keys())
        all_methods = set()
        for methods in h_levels.values():
            all_methods.update(methods.keys())
        method_order = [
            'dc_ada',
            'local_finetuning',
            'obs_normalization',
            'random_perturbation',
            'shared_policy',
        ]
        methods = [m for m in method_order if m in all_methods] + [m for m in sorted(all_methods) if m not in method_order]

        def build_matrix(metric: str, scale: float = 1.0):
            mat = np.full((len(methods), len(h_vals)), np.nan, dtype=np.float32)
            for i, method in enumerate(methods):
                for j, h in enumerate(h_vals):
                    if method in h_levels[h]:
                        vals = [float(r.get(metric, np.nan)) for r in h_levels[h][method]]
                        vals = [v for v in vals if np.isfinite(v)]
                        if vals:
                            mat[i, j] = float(np.mean(vals)) * float(scale)
            return mat

        # Reward heatmap
        reward_mat = build_matrix('mean_reward', scale=1.0)
        _plot_single_heatmap(
            matrix=reward_mat,
            methods=methods,
            h_vals=h_vals,
            title=f"{env_name.replace('_', ' ').title()}: Mean Reward Heatmap",
            cbar_label='Mean Reward',
            output_path=os.path.join(output_dir, f"heatmap_reward_{env_name}.pdf"),
        )

        # Success-rate heatmap
        success_mat = build_matrix('success_rate', scale=100.0)
        _plot_single_heatmap(
            matrix=success_mat,
            methods=methods,
            h_vals=h_vals,
            title=f"{env_name.replace('_', ' ').title()}: Success Rate Heatmap",
            cbar_label='Success Rate (%)',
            output_path=os.path.join(output_dir, f"heatmap_success_{env_name}.pdf"),
        )


def _plot_single_heatmap(
    matrix: np.ndarray,
    methods: list,
    h_vals: list,
    title: str,
    cbar_label: str,
    output_path: str,
):
    """Helper: plot one heatmap with value annotations."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Robust colormap scaling when NaNs are present
    finite_vals = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    im = ax.imshow(matrix, aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('Method')
    ax.set_xticks(np.arange(len(h_vals)))
    ax.set_xticklabels([f'H{h}' for h in h_vals])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_NAMES.get(m, m) for m in methods])

    # Annotate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                txt = '--'
            else:
                # Compact formatting
                txt = f"{val:.1f}" if abs(val) >= 10 else f"{val:.2f}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate figures from experiment results')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='figures', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and aggregate results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    aggregated = aggregate_results(results)
    
    print(f"Found {len(results['experiments'])} experiments")
    print(f"Environments: {list(aggregated.keys())}")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_performance_comparison(aggregated, args.output)
    plot_heterogeneity_scaling(aggregated, args.output)
    plot_success_rate_comparison(aggregated, args.output)

    # Optional but useful for papers: compact heatmaps
    plot_heatmaps(aggregated, args.output)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(aggregated, args.output)
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

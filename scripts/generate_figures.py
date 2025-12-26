"""
Figure Generation Script for DC-Ada Paper

Generates publication-quality figures from experiment results.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats

# Set up matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color scheme for methods
METHOD_COLORS = {
    "shared_policy": "#808080",      # Gray
    "random_perturbation": "#2ca02c", # Green
    "local_finetuning": "#9467bd",    # Purple
    "gradient_finetuning": "#d62728", # Red
    "dc_ada": "#1f77b4"               # Blue
}

METHOD_LABELS = {
    "shared_policy": "Shared Policy (No Adaptation)",
    "random_perturbation": "Random Perturbation",
    "local_finetuning": "Local Fine-Tuning",
    "gradient_finetuning": "Gradient-Based Fine-Tuning",
    "dc_ada": "DC-Ada (Ours)"
}

METHOD_STYLES = {
    "shared_policy": {"linestyle": "--", "linewidth": 1.5},
    "random_perturbation": {"linestyle": ":", "linewidth": 1.5},
    "local_finetuning": {"linestyle": "-.", "linewidth": 1.5},
    "gradient_finetuning": {"linestyle": "-", "linewidth": 1.5},
    "dc_ada": {"linestyle": "-", "linewidth": 2.5}
}


def load_results(results_dir: Path) -> Dict:
    """Load experiment results."""
    pickle_file = results_dir / "results.pkl"
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    json_file = results_dir / "results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No results found in {results_dir}")


def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_learning_curves(results: Dict, output_dir: Path, metric: str = "episode_scores"):
    """Plot learning curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in results.keys():
        runs = results[method]
        if len(runs) == 0:
            continue
        
        # Stack all runs
        all_curves = np.array([run[metric] for run in runs])
        
        # Compute mean and std
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        # Smooth
        window = 10
        if len(mean_curve) > window:
            mean_smooth = smooth_curve(mean_curve, window)
            std_smooth = smooth_curve(std_curve, window)
            episodes = np.arange(window - 1, len(mean_curve))
        else:
            mean_smooth = mean_curve
            std_smooth = std_curve
            episodes = np.arange(len(mean_curve))
        
        # Plot
        color = METHOD_COLORS.get(method, "#000000")
        style = METHOD_STYLES.get(method, {})
        label = METHOD_LABELS.get(method, method)
        
        ax.plot(episodes, mean_smooth, color=color, label=label, **style)
        ax.fill_between(episodes, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       color=color, alpha=0.2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team Score" if metric == "episode_scores" else metric.replace("_", " ").title())
    ax.set_title("Learning Curves: Team Performance Over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"learning_curves_{metric}.png")
    plt.savefig(output_dir / f"learning_curves_{metric}.pdf")
    plt.close()
    
    print(f"Saved learning curves to {output_dir}")


def plot_success_rate_curves(results: Dict, output_dir: Path):
    """Plot success rate learning curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in results.keys():
        runs = results[method]
        if len(runs) == 0:
            continue
        
        # Stack all runs
        all_curves = np.array([run["episode_successes"] for run in runs])
        
        # Compute mean and std
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        # Smooth
        window = 20
        if len(mean_curve) > window:
            mean_smooth = smooth_curve(mean_curve, window)
            std_smooth = smooth_curve(std_curve, window)
            episodes = np.arange(window - 1, len(mean_curve))
        else:
            mean_smooth = mean_curve
            std_smooth = std_curve
            episodes = np.arange(len(mean_curve))
        
        # Plot
        color = METHOD_COLORS.get(method, "#000000")
        style = METHOD_STYLES.get(method, {})
        label = METHOD_LABELS.get(method, method)
        
        ax.plot(episodes, mean_smooth * 100, color=color, label=label, **style)
        ax.fill_between(episodes, (mean_smooth - std_smooth) * 100, 
                       (mean_smooth + std_smooth) * 100, color=color, alpha=0.2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Team Success Rate vs. Training Episode")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_curves.png")
    plt.savefig(output_dir / "success_rate_curves.pdf")
    plt.close()


def plot_ablation_bar(results: Dict, output_dir: Path):
    """Plot ablation study bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    means = []
    stds = []
    colors = []
    
    for method in methods:
        runs = results[method]
        if len(runs) == 0:
            means.append(0)
            stds.append(0)
        else:
            final_scores = [run["final_success_rate"] * 100 for run in runs]
            means.append(np.mean(final_scores))
            stds.append(np.std(final_scores))
        colors.append(METHOD_COLORS.get(method, "#000000"))
    
    x = np.arange(len(methods))
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel("Final Success Rate (%)")
    ax.set_title("Ablation Study: Final Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 100)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_barplot.png")
    plt.savefig(output_dir / "ablation_barplot.pdf")
    plt.close()


def plot_modality_weights(results: Dict, output_dir: Path):
    """Plot modality weight evolution for DC-Ada."""
    if "dc_ada" not in results or len(results["dc_ada"]) == 0:
        print("No DC-Ada results found for modality weight plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    modality_names = ["LiDAR", "Camera", "Odometry", "Package Info"]
    robot_labels = ["Robot A (Full Sensors)", "Robot B (Camera Only)", 
                   "Robot C (Degraded LiDAR)", "Robot D (Noisy Sensors)"]
    
    # Get modality weights from first run
    run = results["dc_ada"][0]
    weights_history = run.get("modality_weights_history", [])
    
    if len(weights_history) == 0:
        print("No modality weights history found")
        return
    
    num_episodes = len(weights_history)
    episodes = np.arange(num_episodes)
    
    for robot_idx in range(min(4, len(weights_history[0]))):
        ax = axes[robot_idx]
        
        # Extract weights for this robot over time
        robot_weights = np.array([w[robot_idx] if robot_idx < len(w) else np.zeros(4) 
                                  for w in weights_history])
        
        for mod_idx, mod_name in enumerate(modality_names):
            if robot_weights.shape[1] > mod_idx:
                ax.plot(episodes, robot_weights[:, mod_idx], label=mod_name, linewidth=2)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Modality Weight")
        ax.set_title(robot_labels[robot_idx])
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Learned Modality Weights Over Training", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "modality_weights.png")
    plt.savefig(output_dir / "modality_weights.pdf")
    plt.close()


def plot_per_robot_performance(results: Dict, output_dir: Path):
    """Plot per-robot performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot deliveries per robot
    ax = axes[0]
    
    for method in ["shared_policy", "dc_ada"]:
        if method not in results or len(results[method]) == 0:
            continue
        
        runs = results[method]
        all_deliveries = np.array([run["episode_deliveries"] for run in runs])
        mean_deliveries = np.mean(all_deliveries, axis=0)
        
        window = 20
        if len(mean_deliveries) > window:
            mean_smooth = smooth_curve(mean_deliveries, window)
            episodes = np.arange(window - 1, len(mean_deliveries))
        else:
            mean_smooth = mean_deliveries
            episodes = np.arange(len(mean_deliveries))
        
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        style = METHOD_STYLES.get(method, {})
        
        ax.plot(episodes, mean_smooth, color=color, label=label, **style)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Packages Delivered")
    ax.set_title("Package Delivery Performance")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Plot collisions
    ax = axes[1]
    
    for method in ["shared_policy", "dc_ada"]:
        if method not in results or len(results[method]) == 0:
            continue
        
        runs = results[method]
        all_collisions = np.array([run["episode_collisions"] for run in runs])
        mean_collisions = np.mean(all_collisions, axis=0)
        
        window = 20
        if len(mean_collisions) > window:
            mean_smooth = smooth_curve(mean_collisions, window)
            episodes = np.arange(window - 1, len(mean_collisions))
        else:
            mean_smooth = mean_collisions
            episodes = np.arange(len(mean_collisions))
        
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        style = METHOD_STYLES.get(method, {})
        
        ax.plot(episodes, mean_smooth, color=color, label=label, **style)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collisions per Episode")
    ax.set_title("Collision Avoidance Performance")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_robot_performance.png")
    plt.savefig(output_dir / "per_robot_performance.pdf")
    plt.close()


def plot_simulation_environment(output_dir: Path):
    """Plot a schematic of the simulation environment."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    arena_size = 20.0
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Webots Simulation Environment (20m x 20m Arena)')
    
    # Draw arena boundary
    arena = mpatches.Rectangle((0, 0), arena_size, arena_size, 
                                linewidth=2, edgecolor='black', 
                                facecolor='lightgray', alpha=0.3)
    ax.add_patch(arena)
    
    # Draw obstacles
    np.random.seed(42)
    obstacles = [
        {"pos": np.array([3.5, 2.5]), "size": np.array([3, 1])},
        {"pos": np.array([8.5, 7]), "size": np.array([1, 4])},
        {"pos": np.array([14, 2.5]), "size": np.array([4, 1])},
        {"pos": np.array([15.5, 11]), "size": np.array([1, 6])},
        {"pos": np.array([7.5, 12.5]), "size": np.array([5, 1])},
        {"pos": np.array([2.5, 16.5]), "size": np.array([1, 3])},
        {"pos": np.array([13, 15.5]), "size": np.array([6, 1])},
    ]
    
    for obs in obstacles:
        rect = mpatches.Rectangle(
            obs["pos"] - obs["size"]/2, obs["size"][0], obs["size"][1],
            linewidth=1, edgecolor='darkgray', facecolor='gray'
        )
        ax.add_patch(rect)
    
    # Draw delivery zones
    delivery_zones = [
        {"pos": np.array([2.0, 2.0]), "radius": 1.5},
        {"pos": np.array([18.0, 2.0]), "radius": 1.5},
        {"pos": np.array([2.0, 18.0]), "radius": 1.5},
        {"pos": np.array([18.0, 18.0]), "radius": 1.5}
    ]
    
    for zone in delivery_zones:
        circle = mpatches.Circle(zone["pos"], zone["radius"],
                                 linewidth=1, edgecolor='green', 
                                 facecolor='lightgreen', alpha=0.5)
        ax.add_patch(circle)
    
    # Draw robots
    robot_positions = [(4, 8), (8, 4), (14, 12), (16, 6)]
    robot_colors = ['blue', 'orange', 'red', 'purple']
    robot_labels = ['Robot A (LiDAR + Camera)', 'Robot B (Camera Only)', 
                   'Robot C (Degraded LiDAR)', 'Robot D (Noisy Sensors)']
    
    for i, (pos, color, label) in enumerate(zip(robot_positions, robot_colors, robot_labels)):
        robot = mpatches.Circle(pos, 0.5, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(robot)
        ax.annotate(f'R{i+1}', pos, ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
    
    # Draw packages
    package_positions = [(6, 6), (10, 10), (12, 5), (5, 14), (15, 15), (9, 17)]
    for pos in package_positions:
        pkg = mpatches.Rectangle((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6,
                                  linewidth=1, edgecolor='brown', facecolor='tan')
        ax.add_patch(pkg)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=l) for c, l in zip(robot_colors, robot_labels)
    ]
    legend_patches.append(mpatches.Patch(color='gray', label='Obstacles'))
    legend_patches.append(mpatches.Patch(color='lightgreen', label='Delivery Zones'))
    legend_patches.append(mpatches.Patch(color='tan', label='Packages'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "simulation_environment.png")
    plt.savefig(output_dir / "simulation_environment.pdf")
    plt.close()


def generate_results_table(results: Dict, output_dir: Path):
    """Generate LaTeX table of results."""
    
    table_lines = [
        "\\begin{table}[!ht]",
        "\\centering",
        "\\caption{Overall mission performance (mean $\\pm$ s.d., $n = 20$). Asterisks (*) denote statistically significant improvements over the shared-policy baseline ($p < 0.05$).}",
        "\\label{tab:results}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Success Rate (\\%) & Team Score & Convergence Episode \\\\",
        "\\midrule"
    ]
    
    # Get baseline for significance testing
    baseline_scores = None
    if "shared_policy" in results and len(results["shared_policy"]) > 0:
        baseline_scores = [r["final_success_rate"] for r in results["shared_policy"]]
    
    for method in ["shared_policy", "random_perturbation", "local_finetuning", 
                   "gradient_finetuning", "dc_ada"]:
        if method not in results or len(results[method]) == 0:
            continue
        
        runs = results[method]
        
        success_rates = [r["final_success_rate"] * 100 for r in runs]
        scores = [r["final_score"] for r in runs]
        convergence = [r["convergence_episode"] for r in runs]
        
        # Significance test
        sig = ""
        if baseline_scores is not None and method != "shared_policy":
            method_scores = [r["final_success_rate"] for r in runs]
            _, p_value = stats.ttest_ind(baseline_scores, method_scores)
            if p_value < 0.05 and np.mean(method_scores) > np.mean(baseline_scores):
                sig = "*"
        
        label = METHOD_LABELS.get(method, method)
        if method == "dc_ada":
            label = "\\textbf{DC-Ada (ours)}" + sig
        else:
            label = label + sig
        
        success_str = f"${np.mean(success_rates):.1f} \\pm {np.std(success_rates):.1f}$"
        score_str = f"${np.mean(scores):.1f} \\pm {np.std(scores):.1f}$"
        conv_str = f"${np.mean(convergence):.0f} \\pm {np.std(convergence):.0f}$"
        
        if method == "dc_ada":
            success_str = "\\textbf{" + success_str + "}"
            score_str = "\\textbf{" + score_str + "}"
        
        table_lines.append(f"{label} & {success_str} & {score_str} & {conv_str} \\\\")
    
    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    table_file = output_dir / "results_table.tex"
    with open(table_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"Saved results table to {table_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from experiment results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for figures")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if args.output is None:
        output_dir = results_dir / "figures"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    
    print("Generating figures...")
    
    # Generate all figures
    plot_simulation_environment(output_dir)
    plot_learning_curves(results, output_dir, "episode_scores")
    plot_success_rate_curves(results, output_dir)
    plot_ablation_bar(results, output_dir)
    plot_modality_weights(results, output_dir)
    plot_per_robot_performance(results, output_dir)
    generate_results_table(results, output_dir)
    
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()

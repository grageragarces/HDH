#!/usr/bin/env python3
"""
HDH Experiment Visualization
=============================
Generate plots from experiment results.

Creates two plots:
1. Number of qubits vs cut cost
2. Number of qubits vs heuristic time

Usage:
    python plot_results.py

Requirements:
    - pandas
    - matplotlib
    - seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ========================= CONFIGURATION =========================

# File paths (modify these for your system)
BASE_DIR = Path("/Users/mariagragera/Desktop/HDH/hdh")
RESULTS_CSV = BASE_DIR / "experiment_results.csv"
PLOTS_DIR = BASE_DIR / "plots"

# Create plots directory
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ========================= DATA LOADING =========================

def load_and_clean_data():
    """Load CSV and filter successful runs."""
    df = pd.read_csv(RESULTS_CSV)
    
    # Filter only successful runs
    df = df[df['success'] == True]
    
    print(f"Loaded {len(df)} successful experiment results")
    print(f"Models: {df['model'].unique()}")
    print(f"Qubit range: {df['num_qubits'].min()} - {df['num_qubits'].max()}")
    
    return df

# ========================= PLOTTING FUNCTIONS =========================

def plot_cut_cost(df):
    """Plot number of qubits vs cut cost."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette
    colors = sns.color_palette("Set2", n_colors=4)
    model_colors = {
        "Circuit": colors[0],
        "MBQC": colors[1],
        "QW": colors[2],
        "QCA": colors[3]
    }
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Group by num_qubits and calculate mean and std
        grouped = model_data.groupby('num_qubits')['cut_cost'].agg(['mean', 'std', 'count'])
        
        # Plot mean with error bars
        ax.errorbar(
            grouped.index, 
            grouped['mean'], 
            yerr=grouped['std'],
            marker='o',
            markersize=6,
            label=model,
            color=model_colors.get(model, 'gray'),
            capsize=3,
            capthick=1,
            linewidth=2,
            alpha=0.8
        )
    
    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cut Cost (Number of Cut Hyperedges)', fontsize=13, fontweight='bold')
    ax.set_title('HDH Cutting Performance: Cut Cost vs Workload Size', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(title='Model', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = PLOTS_DIR / 'cut_cost_vs_qubits.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    plt.close()


def plot_heuristic_time(df):
    """Plot number of qubits vs heuristic time."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette
    colors = sns.color_palette("Set2", n_colors=4)
    model_colors = {
        "Circuit": colors[0],
        "MBQC": colors[1],
        "QW": colors[2],
        "QCA": colors[3]
    }
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Group by num_qubits and calculate mean and std
        grouped = model_data.groupby('num_qubits')['heuristic_time'].agg(['mean', 'std', 'count'])
        
        # Plot mean with error bars
        ax.errorbar(
            grouped.index, 
            grouped['mean'], 
            yerr=grouped['std'],
            marker='s',
            markersize=6,
            label=model,
            color=model_colors.get(model, 'gray'),
            capsize=3,
            capthick=1,
            linewidth=2,
            alpha=0.8
        )
    
    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Heuristic Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('HDH Cutting Performance: Computation Time vs Workload Size', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(title='Model', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if times vary significantly
    max_time = grouped['mean'].max()
    min_time = grouped['mean'].min()
    if min_time > 0 and max_time / min_time > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Heuristic Time (seconds, log scale)', fontsize=13, fontweight='bold')
    
    # Save
    output_path = PLOTS_DIR / 'heuristic_time_vs_qubits.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    plt.close()

# ========================= SUMMARY STATISTICS =========================

def generate_summary_stats(df):
    """Generate and save summary statistics."""
    summary = df.groupby(['model', 'num_qubits']).agg({
        'cut_cost': ['mean', 'std', 'min', 'max'],
        'heuristic_time': ['mean', 'std', 'min', 'max'],
        'workload_id': 'count'
    }).round(4)
    
    summary_path = PLOTS_DIR / 'summary_statistics.csv'
    summary.to_csv(summary_path)
    print(f"Saved summary statistics: {summary_path}")
    
    # Print overall summary
    print("\n" + "="*60)
    print("Overall Summary by Model:")
    print("="*60)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  Total workloads: {len(model_data)}")
        print(f"  Avg cut cost: {model_data['cut_cost'].mean():.2f} ± {model_data['cut_cost'].std():.2f}")
        print(f"  Avg time: {model_data['heuristic_time'].mean():.4f}s ± {model_data['heuristic_time'].std():.4f}s")
        print(f"  Qubit range: {model_data['num_qubits'].min()} - {model_data['num_qubits'].max()}")

# ========================= MAIN =========================

def main():
    """Main plotting function."""
    print("="*60)
    print("Generating plots from experiment results")
    print("="*60)
    
    # Load data
    df = load_and_clean_data()
    
    if len(df) == 0:
        print("No successful results found in CSV!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    plot_cut_cost(df)
    plot_heuristic_time(df)
    
    # Generate summary
    generate_summary_stats(df)
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print(f"Output directory: {PLOTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Plot Brute Force vs Heuristic Results for k=3 QPUs

Reads from experiment_outputs_mqtbench/comparison_results_qubit-level.csv
and creates a focused plot showing MEAN heuristic/optimal ratio vs network overhead.
Excludes any algorithms with "random" in the name.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
CSV_FILE = Path('experiment_outputs_mqtbench/comparison_results_qubit-level.csv')
OUTPUT_DIR = Path('experiment_outputs_mqtbench/brute_force_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_FILE = OUTPUT_DIR / 'ratio_vs_overhead_k3_mean.png'

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11

def extract_algorithm_info(circuit_name: str) -> dict:
    """
    Extract algorithm name and qubit count from circuit name.
    Format: algorithmname_indep_qiskit_numqubits
    
    Returns dict with 'algorithm' and 'qubits' keys.
    """
    parts = circuit_name.split('_')
    if len(parts) >= 4 and parts[-3] == 'indep' and parts[-2] == 'qiskit':
        algorithm = parts[0]  # First part is algorithm name
        qubits = parts[-1]     # Last part is number of qubits
        return {'algorithm': algorithm, 'qubits': qubits}
    else:
        # Fallback for unexpected format
        return {'algorithm': circuit_name, 'qubits': 'unknown'}

def load_and_filter_data(csv_path: Path, k_filter: int = 3) -> pd.DataFrame:
    """Load CSV, filter for specific k value, and exclude 'random' algorithms."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows loaded: {len(df)}")
    print(f"Unique k values: {sorted(df['k'].unique())}")
    
    # Filter for k=3
    df_filtered = df[df['k'] == k_filter].copy()
    print(f"Rows with k={k_filter}: {len(df_filtered)}")
    
    if df_filtered.empty:
        raise ValueError(f"No data found for k={k_filter}!")
    
    # Filter out circuits with "random" in the name
    n_before = len(df_filtered)
    df_filtered = df_filtered[~df_filtered['circuit'].str.contains('random', case=False, na=False)].copy()
    n_after = len(df_filtered)
    print(f"Filtered out 'random' algorithms: {n_before - n_after} rows removed")
    print(f"Remaining rows: {n_after}")
    
    if df_filtered.empty:
        raise ValueError(f"No data remaining after filtering out 'random' algorithms!")
    
    return df_filtered


def print_summary_statistics(df: pd.DataFrame, k: int):
    """Print summary statistics to command line."""
    print("\n" + "="*70)
    print(f"SUMMARY STATISTICS (k={k} QPUs)")
    print("="*70)
    
    # Basic info
    print(f"\nDataset Overview:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique circuits: {len(df['circuit'].unique())}")
    print(f"  Overhead values tested: {sorted(df['overhead'].unique())}")
    
    # Extract and display algorithm information
    print(f"\nAlgorithms Analyzed:")
    circuit_names = sorted(df['circuit'].unique())
    
    # Group by algorithm name
    algo_dict = defaultdict(list)
    for circuit_name in circuit_names:
        info = extract_algorithm_info(circuit_name)
        algo_name = info['algorithm']
        qubits = info['qubits']
        algo_dict[algo_name].append(qubits)
    
    print(f"  Total unique algorithms: {len(algo_dict)}")
    print(f"\n  {'Algorithm':<25} {'Qubit Sizes':<30}")
    print(f"  {'-'*60}")
    for algo_name in sorted(algo_dict.keys()):
        qubit_sizes = ', '.join(sorted(algo_dict[algo_name], key=lambda x: int(x) if x.isdigit() else 0))
        print(f"  {algo_name:<25} {qubit_sizes:<30}")
    
    # Filter finite ratios for statistics
    df_finite = df[np.isfinite(df['ratio'])].copy()
    n_infinite = (df['ratio'] == float('inf')).sum()
    n_nan = df['ratio'].isna().sum()
    
    print(f"\nRatio Distribution:")
    print(f"  Finite ratios: {len(df_finite)}")
    print(f"  Infinite ratios: {n_infinite}")
    print(f"  NaN ratios: {n_nan}")
    
    if not df_finite.empty:
        print(f"\nRatio Statistics (finite values only):")
        print(f"  Mean:   {df_finite['ratio'].mean():.4f}")
        print(f"  Median: {df_finite['ratio'].median():.4f}")
        print(f"  Std:    {df_finite['ratio'].std():.4f}")
        print(f"  Min:    {df_finite['ratio'].min():.4f}")
        print(f"  Max:    {df_finite['ratio'].max():.4f}")
        
        # Count optimal matches
        n_optimal = (df_finite['ratio'] == 1.0).sum()
        print(f"\n  Optimal matches (ratio=1.0): {n_optimal}/{len(df_finite)} "
              f"({n_optimal/len(df_finite)*100:.1f}%)")
        
        # Statistics by overhead
        print(f"\nBreakdown by Overhead:")
        print(f"  {'Overhead':<12} {'Count':<8} {'Mean Ratio':<12} {'Std Ratio':<12} {'Optimal %':<12}")
        print(f"  {'-'*70}")
        for oh in sorted(df['overhead'].unique()):
            df_oh = df_finite[df_finite['overhead'] == oh]
            if len(df_oh) > 0:
                mean_ratio = df_oh['ratio'].mean()
                std_ratio = df_oh['ratio'].std()
                n_opt = (df_oh['ratio'] == 1.0).sum()
                opt_pct = n_opt / len(df_oh) * 100
                print(f"  {oh:<12.2f} {len(df_oh):<8} {mean_ratio:<12.4f} {std_ratio:<12.4f} {opt_pct:<12.1f}%")
    else:
        print("\n  ⚠ No finite ratios found!")
    
    # Cost statistics
    print(f"\nCost Statistics:")
    print(f"  Optimal costs:")
    print(f"    Mean:   {df['optimal_cost'].mean():.2f}")
    print(f"    Median: {df['optimal_cost'].median():.2f}")
    print(f"    Min:    {df['optimal_cost'].min():.2f}")
    print(f"    Max:    {df['optimal_cost'].max():.2f}")
    
    print(f"\n  Heuristic costs:")
    print(f"    Mean:   {df['heuristic_cost'].mean():.2f}")
    print(f"    Median: {df['heuristic_cost'].median():.2f}")
    print(f"    Min:    {df['heuristic_cost'].min():.2f}")
    print(f"    Max:    {df['heuristic_cost'].max():.2f}")
    
    # Timing
    print(f"\nTiming (seconds):")
    print(f"  Brute force average: {df['brute_force_time'].mean():.4f}s")
    print(f"  Heuristic average:   {df['heuristic_time'].mean():.4f}s")
    
    print("="*70 + "\n")


def create_plot(df: pd.DataFrame, output_path: Path, k: int):
    """Create box & whisker plot showing ratio distribution by overhead."""
    print(f"\nCreating plot...")
    
    # Filter out infinite/nan ratios for plotting
    df_finite = df[np.isfinite(df['ratio'])].copy()
    
    if df_finite.empty:
        print("⚠ Warning: No finite ratios to plot!")
        print("  All optimal costs were likely 0 (everything fit on one QPU)")
        return
    
    # Get overhead values sorted
    overhead_values = sorted(df_finite['overhead'].unique())
    
    # Prepare data for box plot
    data_by_overhead = [df_finite[df_finite['overhead'] == oh]['ratio'].values 
                        for oh in overhead_values]
    
    # Print statistics for each overhead
    print(f"\nRatio distribution by overhead:")
    for oh, data in zip(overhead_values, data_by_overhead):
        print(f"  Overhead {oh:.2f}: median={np.median(data):.4f}, "
              f"mean={np.mean(data):.4f}, n={len(data)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create box & whisker plot
    bp = ax.boxplot(data_by_overhead, 
                    positions=overhead_values,
                    widths=0.05,
                    patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7, linewidth=2),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                    medianprops=dict(color='red', linewidth=2.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', 
                                   markersize=6, alpha=0.5))
    
    # Labels and title
    ax.set_xlabel('Network Overhead', fontsize=14, fontweight='bold')
    ax.set_ylabel('Optimality Ratio', fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(overhead_values)
    ax.set_xticklabels([f'{oh:.2f}' for oh in overhead_values])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set y-axis to start slightly below minimum
    y_min = max(0.9, df_finite['ratio'].min() - 0.1)
    y_max = df_finite['ratio'].max() + 0.2
    
    x_min = min(overhead_values) - 0.05
    x_max = max(overhead_values) + 0.05
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("BRUTE FORCE vs HEURISTIC ANALYSIS (k=3 QPUs)")
    print("="*70 + "\n")
    
    # Check if file exists
    if not CSV_FILE.exists():
        print(f"❌ Error: CSV file not found at {CSV_FILE}")
        print(f"   Please run the brute force experiments first.")
        return
    
    # Load and filter data
    try:
        df = load_and_filter_data(CSV_FILE, k_filter=3)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Print statistics
    print_summary_statistics(df, k=3)
    
    # Create plot
    try:
        create_plot(df, PLOT_FILE, k=3)
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Analysis complete!\n")


if __name__ == '__main__':
    main()
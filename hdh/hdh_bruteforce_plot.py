import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.ticker as ticker

# Configuration
CSV_FILE = Path('experiment_outputs_mqtbench/results_node_level_fixed_over10.csv') 
OUTPUT_DIR = Path('experiment_outputs_mqtbench')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_FILE = OUTPUT_DIR / 'optimality_vs_overhead_k3.svg'
SIZE_PLOT_FILE = OUTPUT_DIR / 'optimality_vs_circuit_size_overhead1.svg'

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
    
    # RECALCULATE AS OPTIMALITY (optimal/heuristic)
    df_filtered['optimality'] = np.where(
        df_filtered['heuristic_cost'] > 0,
        df_filtered['optimal_cost'] / df_filtered['heuristic_cost'],
        np.where(df_filtered['optimal_cost'] == 0, 1.0, np.nan)
    )
    
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
    
    # Filter finite optimality values for statistics
    df_finite = df[np.isfinite(df['optimality'])].copy()
    n_infinite = (df['optimality'] == float('inf')).sum()
    n_nan = df['optimality'].isna().sum()
    
    print(f"\nOptimality Distribution:")
    print(f"  Finite values: {len(df_finite)}")
    print(f"  Infinite values: {n_infinite}")
    print(f"  NaN values: {n_nan}")
    
    if not df_finite.empty:
        mean_pct = df_finite['optimality'].mean() * 100
        median_pct = df_finite['optimality'].median() * 100
        std_pct = df_finite['optimality'].std() * 100
        min_pct = df_finite['optimality'].min() * 100
        max_pct = df_finite['optimality'].max() * 100
        
        print(f"\nOptimality Statistics (finite values only):")
        print(f"  Mean:   {mean_pct:.1f}%")
        print(f"  Median: {median_pct:.1f}%")
        print(f"  Std:    {std_pct:.1f} percentage points")
        print(f"  Min:    {min_pct:.1f}%")
        print(f"  Max:    {max_pct:.1f}%")
        
        # Count perfect matches
        n_perfect = (df_finite['optimality'] == 1.0).sum()
        print(f"\n  Perfect matches (100% optimal): {n_perfect}/{len(df_finite)} "
              f"({n_perfect/len(df_finite)*100:.1f}%)")
        
        # Statistics by overhead
        print(f"\nBreakdown by Overhead:")
        print(f"  {'Overhead':<12} {'Count':<8} {'Mean Optimality':<18} {'Std':<12} {'Perfect %':<12}")
        print(f"  {'-'*75}")
        for oh in sorted(df['overhead'].unique()):
            df_oh = df_finite[df_finite['overhead'] == oh]
            if len(df_oh) > 0:
                mean_opt = df_oh['optimality'].mean() * 100
                std_opt = df_oh['optimality'].std() * 100
                n_perf = (df_oh['optimality'] == 1.0).sum()
                perf_pct = n_perf / len(df_oh) * 100
                print(f"  {oh:<12.2f} {len(df_oh):<8} {mean_opt:<18.1f}% {std_opt:<12.1f}pp {perf_pct:<12.1f}%")
    else:
        print("\n  ⚠ No finite optimality values found!")
    
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
    """Create box & whisker plot showing optimality distribution by overhead."""
    print(f"\nCreating plot...")
    
    # Filter out infinite/nan optimality values for plotting
    df_finite = df[np.isfinite(df['optimality'])].copy()
    
    if df_finite.empty:
        print("⚠ Warning: No finite optimality values to plot!")
        print("  All optimal costs were likely 0 (everything fit on one QPU)")
        return
    
    # Get overhead values sorted
    overhead_values = sorted(df_finite['overhead'].unique())
    
    # Prepare data for box plot (convert to percentage)
    data_by_overhead = [(df_finite[df_finite['overhead'] == oh]['optimality'] * 100).values 
                        for oh in overhead_values]
    
    # Print statistics for each overhead
    print(f"\nOptimality distribution by overhead:")
    for oh, data in zip(overhead_values, data_by_overhead):
        print(f"  Overhead {oh:.2f}: median={np.median(data):.1f}%, "
              f"mean={np.mean(data):.1f}%, n={len(data)}")
    
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
    ax.set_ylabel('Optimality (%)', fontsize=14, fontweight='bold')
    # ax.set_title('Heuristic Performance vs Network Overhead\n(100% = Optimal, Higher is Better)', 
    #              fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(overhead_values)
    ax.set_xticklabels([f'{oh:.2f}' for oh in overhead_values])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add horizontal line at 100% (perfect optimality)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.7, 
               label='100% (Perfect)', zorder=10)
    
    # Add horizontal line at 50% (warning threshold)
    ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='50% (Poor)', zorder=10)
    
    # Set y-axis range
    y_min = max(0, (df_finite['optimality'].min() * 100) - 5)
    y_max = min(110, (df_finite['optimality'].max() * 100) + 5)
    
    x_min = min(overhead_values) - 0.05
    x_max = 1.35 #max(overhead_values) + 0.05
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    # Legend
    ax.legend(loc='lower left')
    
    # Save
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()


def create_circuit_size_plot(df: pd.DataFrame, output_path: Path, k: int, overhead_value: float = 1.0):
    """Create scatter plot showing optimality vs circuit size for a specific overhead."""
    print(f"\nCreating circuit size plot for overhead={overhead_value}...")
    
    # Filter for specific overhead
    df_overhead = df[df['overhead'] == overhead_value].copy()
    
    if df_overhead.empty:
        print(f"âš  Warning: No data found for overhead={overhead_value}")
        return
    
    # Extract qubit counts
    df_overhead['qubits'] = df_overhead['circuit'].apply(
        lambda x: extract_algorithm_info(x)['qubits']
    )
    
    # Convert qubits to integer (filter out 'unknown')
    df_overhead['qubits_int'] = pd.to_numeric(df_overhead['qubits'], errors='coerce')
    df_overhead = df_overhead.dropna(subset=['qubits_int'])
    
    # Filter out infinite/nan optimality values
    df_plot = df_overhead[np.isfinite(df_overhead['optimality'])].copy()
    
    if df_plot.empty:
        print(f"âš  Warning: No finite optimality values to plot for overhead={overhead_value}")
        return
    
    # Convert optimality to percentage
    df_plot['optimality_pct'] = df_plot['optimality'] * 100
    
    # Print statistics
    print(f"\nCircuit size analysis for overhead={overhead_value}:")
    print(f"  Total circuits: {len(df_plot)}")
    print(f"  Qubit range: {int(df_plot['qubits_int'].min())} - {int(df_plot['qubits_int'].max())}")
    print(f"  Mean heuristic cost: {df_plot['heuristic_cost'].mean():.1f}")
    print(f"  Mean optimality: {df_plot['optimality_pct'].mean():.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot - y-axis is heuristic_cost, color is optimality
    scatter = ax.scatter(df_plot['qubits_int'], 
                        df_plot['heuristic_cost'],
                        alpha=0.6, 
                        s=100,
                        c=df_plot['optimality_pct'],
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=100,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Solution Quality (%)', fontsize=12, fontweight='bold')
    
    # Calculate and plot trend line for heuristic_cost
    # z = np.polyfit(df_plot['qubits_int'], df_plot['heuristic_cost'], 1)
    # p = np.poly1d(z)
    # x_trend = np.linspace(df_plot['qubits_int'].min(), df_plot['qubits_int'].max(), 100)
    # ax.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
    
    # Labels and title
    ax.set_xlabel('Circuit Size (Number of Qubits)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Heuristic Cut Cost', fontsize=14, fontweight='bold')
    # ax.set_title(f'Heuristic Optimality vs Circuit Size\n(Overhead = {overhead_value}, k = {k} QPUs)', 
    #              fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits with padding
    x_padding = (df_plot['qubits_int'].max() - df_plot['qubits_int'].min()) * 0.05
    
    # Use 95th percentile for y-axis upper limit to avoid outliers dominating the plot
    y_upper = df_plot['heuristic_cost'].quantile(0.95)
    y_lower = df_plot['heuristic_cost'].min()
    y_padding = (y_upper - y_lower) * 0.1
    
    ax.set_xlim([df_plot['qubits_int'].min() - x_padding, 
                 df_plot['qubits_int'].max() + x_padding])
    ax.set_ylim([max(0, y_lower - y_padding), 
                 y_upper + y_padding])
    
    # Count how many points are outside the visible range
    n_outliers = (df_plot['heuristic_cost'] > y_upper + y_padding).sum()
    if n_outliers > 0:
        print(f"  Note: {n_outliers} outlier(s) exceed y-axis limit (max cost: {df_plot['heuristic_cost'].max():.1f})")
    
    # Legend
    #ax.legend(loc='best')
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add statistics text box
    # stats_text = f'n = {len(df_plot)} circuits\n'
    # stats_text += f'Median cost: {df_plot["heuristic_cost"].median():.1f}\n'
    # stats_text += f'Mean optimality: {df_plot["optimality_pct"].mean():.1f}%'
    
    # ax.text(0.02, 0.02, stats_text,
    #         transform=ax.transAxes,
    #         fontsize=10,
    #         verticalalignment='bottom',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    plt.savefig(output_path, format='svg', bbox_inches='tight')    
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("BRUTE FORCE vs HEURISTIC ANALYSIS (k=3 QPUs)")
    print("OPTIMALITY VIEW: How close to optimal are we?")
    print("="*70 + "\n")
    
    # Check if file exists
    if not CSV_FILE.exists():
        print(f" Error: CSV file not found at {CSV_FILE}")
        print(f"   Please run the brute force experiments first.")
        return
    
    # Load and filter data
    try:
        df = load_and_filter_data(CSV_FILE, k_filter=3)
    except Exception as e:
        print(f" Error loading data: {e}")
        return
    
    # Print statistics
    print_summary_statistics(df, k=3)
    
    # Create plot
    try:
        create_plot(df, PLOT_FILE, k=3)
    except Exception as e:
        print(f" Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create circuit size plot for overhead=1
    try:
        create_circuit_size_plot(df, SIZE_PLOT_FILE, k=3, overhead_value=1.0)
    except Exception as e:
        print(f" Error creating circuit size plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n Analysis complete!\n")


if __name__ == '__main__':
    main()
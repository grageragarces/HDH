#!/usr/bin/env python3
"""
Diagnostic script to identify issues with brute force comparison results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

CSV_FILE = Path('experiment_outputs_mqtbench/comparison_results_qubit-level.csv')

def extract_algorithm_name(circuit_name: str) -> str:
    """Extract just the algorithm name."""
    parts = circuit_name.split('_')
    if len(parts) >= 4 and parts[-3] == 'indep' and parts[-2] == 'qiskit':
        return parts[0]
    return circuit_name.split('_')[0]

def analyze_data():
    """Comprehensive analysis of the data."""
    
    if not CSV_FILE.exists():
        print(f"❌ File not found: {CSV_FILE}")
        return
    
    print("\n" + "="*80)
    print("BRUTE FORCE COMPARISON - DIAGNOSTIC ANALYSIS")
    print("="*80 + "\n")
    
    df = pd.read_csv(CSV_FILE)
    
    # Filter for k=3 and remove random
    df = df[df['k'] == 3].copy()
    df = df[~df['circuit'].str.contains('random', case=False, na=False)].copy()
    
    print(f"Total experiments (k=3, no random): {len(df)}")
    print(f"Unique circuits: {len(df['circuit'].unique())}")
    
    # Add algorithm column
    df['algorithm'] = df['circuit'].apply(extract_algorithm_name)
    
    # ===========================================
    # 1. TIMING ANALYSIS
    # ===========================================
    print("\n" + "="*80)
    print("1. TIMING ANALYSIS - Why is brute force faster?")
    print("="*80)
    
    print(f"\nOverall timing:")
    print(f"  Brute force avg:  {df['brute_force_time'].mean():.4f}s")
    print(f"  Heuristic avg:    {df['heuristic_time'].mean():.4f}s")
    print(f"  Speedup ratio:    {df['heuristic_time'].mean() / df['brute_force_time'].mean():.2f}x")
    
    print(f"\nCircuit size distribution:")
    print(f"  Qubits: min={df['num_qubits'].min()}, max={df['num_qubits'].max()}, mean={df['num_qubits'].mean():.1f}")
    print(f"  Nodes:  min={df['num_nodes'].min()}, max={df['num_nodes'].max()}, mean={df['num_nodes'].mean():.1f}")
    print(f"  Quantum nodes: min={df['num_quantum_nodes'].min()}, max={df['num_quantum_nodes'].max()}, mean={df['num_quantum_nodes'].mean():.1f}")
    
    # Calculate search space size
    df['search_space'] = 3 ** df['num_qubits']
    print(f"\nBrute force search space size:")
    print(f"  Min:  {df['search_space'].min():,} partitions")
    print(f"  Max:  {df['search_space'].max():,} partitions")
    print(f"  Mean: {df['search_space'].mean():,.0f} partitions")
    
    print(f"\n💡 EXPLANATION: For k=3 QPUs and {df['num_qubits'].max()} qubits max,")
    print(f"   brute force only checks {df['search_space'].max():,} partitions.")
    print(f"   This is TINY! The heuristic has overhead that dominates for such small problems.")
    
    # ===========================================
    # 2. PERFORMANCE ANALYSIS BY ALGORITHM
    # ===========================================
    print("\n" + "="*80)
    print("2. PERFORMANCE BY ALGORITHM - Who's destroying your stats?")
    print("="*80)
    
    # Filter finite ratios
    df_finite = df[np.isfinite(df['ratio'])].copy()
    
    # Group by algorithm and calculate stats
    algo_stats = []
    for algo in sorted(df_finite['algorithm'].unique()):
        df_algo = df_finite[df_finite['algorithm'] == algo]
        
        algo_stats.append({
            'algorithm': algo,
            'count': len(df_algo),
            'mean_ratio': df_algo['ratio'].mean(),
            'median_ratio': df_algo['ratio'].median(),
            'std_ratio': df_algo['ratio'].std(),
            'min_ratio': df_algo['ratio'].min(),
            'max_ratio': df_algo['ratio'].max(),
            'mean_optimal_cost': df_algo['optimal_cost'].mean(),
            'mean_heuristic_cost': df_algo['heuristic_cost'].mean(),
        })
    
    algo_df = pd.DataFrame(algo_stats).sort_values('mean_ratio', ascending=False)
    
    print(f"\n{'Algorithm':<20} {'Count':<7} {'Mean±Std Ratio':<20} {'Min-Max':<15} {'Avg Costs (O/H)':<20}")
    print("-" * 90)
    
    for _, row in algo_df.iterrows():
        ratio_str = f"{row['mean_ratio']:.3f}±{row['std_ratio']:.3f}"
        range_str = f"{row['min_ratio']:.2f}-{row['max_ratio']:.2f}"
        costs_str = f"{row['mean_optimal_cost']:.1f}/{row['mean_heuristic_cost']:.1f}"
        print(f"{row['algorithm']:<20} {row['count']:<7} {ratio_str:<20} {range_str:<15} {costs_str:<20}")
    
    # Highlight worst performers
    print(f"\n🚨 WORST PERFORMERS (mean ratio > 2.0):")
    worst = algo_df[algo_df['mean_ratio'] > 2.0]
    if len(worst) > 0:
        for _, row in worst.iterrows():
            print(f"   • {row['algorithm']}: {row['mean_ratio']:.3f} (n={row['count']})")
            
            # Show detailed breakdown
            df_worst = df_finite[df_finite['algorithm'] == row['algorithm']]
            print(f"     Overhead breakdown:")
            for oh in sorted(df_worst['overhead'].unique()):
                df_oh = df_worst[df_worst['overhead'] == oh]
                print(f"       OH={oh:.2f}: ratio={df_oh['ratio'].mean():.3f} (n={len(df_oh)})")
    else:
        print("   (none)")
    
    # ===========================================
    # 3. OVERHEAD ANALYSIS
    # ===========================================
    print("\n" + "="*80)
    print("3. OVERHEAD ANALYSIS - Pattern across network capacities")
    print("="*80)
    
    for oh in sorted(df_finite['overhead'].unique()):
        df_oh = df_finite[df_finite['overhead'] == oh]
        print(f"\nOverhead {oh:.2f} (cap={int(df_oh['capacity'].iloc[0])} qubits/QPU):")
        print(f"  Mean ratio: {df_oh['ratio'].mean():.4f} ± {df_oh['ratio'].std():.4f}")
        print(f"  Median ratio: {df_oh['ratio'].median():.4f}")
        print(f"  Range: [{df_oh['ratio'].min():.4f}, {df_oh['ratio'].max():.4f}]")
        print(f"  Optimal=1.0: {(df_oh['ratio'] == 1.0).sum()}/{len(df_oh)} ({(df_oh['ratio'] == 1.0).sum()/len(df_oh)*100:.1f}%)")
    
    # ===========================================
    # 4. COST DISTRIBUTION ANALYSIS
    # ===========================================
    print("\n" + "="*80)
    print("4. COST ANALYSIS - Are costs realistic?")
    print("="*80)
    
    # Check for zero costs
    zero_optimal = (df['optimal_cost'] == 0).sum()
    zero_heuristic = (df['heuristic_cost'] == 0).sum()
    
    print(f"\nZero cost cases:")
    print(f"  Optimal cost = 0:   {zero_optimal}/{len(df)} ({zero_optimal/len(df)*100:.1f}%)")
    print(f"  Heuristic cost = 0: {zero_heuristic}/{len(df)} ({zero_heuristic/len(df)*100:.1f}%)")
    
    if zero_optimal > 0:
        print(f"\n  💡 When optimal=0, the entire circuit fits on one QPU (no cuts needed).")
        print(f"     These are excluded from ratio statistics (would be inf or nan).")
    
    print(f"\nCost distribution (all experiments):")
    print(f"  Optimal costs:   mean={df['optimal_cost'].mean():.2f}, median={df['optimal_cost'].median():.2f}")
    print(f"  Heuristic costs: mean={df['heuristic_cost'].mean():.2f}, median={df['heuristic_cost'].median():.2f}")
    
    # Show histogram
    print(f"\nOptimal cost histogram:")
    cost_bins = [0, 1, 2, 5, 10, 20, 50, 100]
    for i in range(len(cost_bins)-1):
        count = ((df['optimal_cost'] >= cost_bins[i]) & (df['optimal_cost'] < cost_bins[i+1])).sum()
        print(f"  {cost_bins[i]:3d}-{cost_bins[i+1]:3d}: {count:3d} experiments")
    count = (df['optimal_cost'] >= cost_bins[-1]).sum()
    print(f"  {cost_bins[-1]:3d}+   : {count:3d} experiments")
    
    # ===========================================
    # 5. BOX PLOT TIGHTNESS
    # ===========================================
    print("\n" + "="*80)
    print("5. WHY ARE BOXES TIGHT? - Variance analysis")
    print("="*80)
    
    print(f"\nOverall ratio variance:")
    print(f"  Standard deviation: {df_finite['ratio'].std():.4f}")
    print(f"  IQR (Q3-Q1):       {df_finite['ratio'].quantile(0.75) - df_finite['ratio'].quantile(0.25):.4f}")
    print(f"  Range (max-min):   {df_finite['ratio'].max() - df_finite['ratio'].min():.4f}")
    
    print(f"\nVariance by overhead:")
    for oh in sorted(df_finite['overhead'].unique()):
        df_oh = df_finite[df_finite['overhead'] == oh]
        iqr = df_oh['ratio'].quantile(0.75) - df_oh['ratio'].quantile(0.25)
        print(f"  OH={oh:.2f}: std={df_oh['ratio'].std():.4f}, IQR={iqr:.4f}")
    
    print(f"\n💡 Low IQR means data is clustered tightly (narrow boxes).")
    print(f"   This suggests most algorithms have similar performance at each overhead level.")
    
    # ===========================================
    # 6. RECOMMENDATIONS
    # ===========================================
    print("\n" + "="*80)
    print("6. RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 FINDINGS:")
    print("  1. Circuits are VERY small (max 5 qubits)")
    print("  2. Brute force checks <250 partitions (faster than heuristic overhead)")
    print("  3. Some algorithms perform consistently poorly (ratio ~2-3x)")
    
    print("\n🔧 SUGGESTED ACTIONS:")
    print("  1. Investigate worst-performing algorithms individually")
    print("  2. Check if heuristic has bugs or bad assumptions for small circuits")
    print("  3. Consider testing on larger circuits (8-12 qubits) where heuristic should shine")
    print("  4. For plots: Consider showing individual algorithm performance or filtering outliers")
    print("  5. The tight boxes suggest you might want to use violin plots or strip plots instead")
    
    print("\n" + "="*80 + "\n")
    
    # Save detailed per-algorithm stats
    output_file = Path('experiment_outputs_mqtbench/brute_force_comparison/algorithm_performance.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    algo_df.to_csv(output_file, index=False)
    print(f"✓ Detailed algorithm stats saved to: {output_file}\n")

if __name__ == '__main__':
    analyze_data()
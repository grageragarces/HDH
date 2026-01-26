#!/usr/bin/env python3
"""
Diagnostic script to identify issues with brute force comparison results.
NOW WITH OPTIMALITY RATIO (optimal/heuristic) - shows how close to optimal we are!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

CSV_FILE = Path('experiment_outputs_mqtbench/comparison_results_10_qubit-level_weighted.csv')

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
    
    # RECALCULATE RATIO AS OPTIMALITY (optimal/heuristic)
    df['optimality'] = np.where(
        df['heuristic_cost'] > 0,
        df['optimal_cost'] / df['heuristic_cost'],
        np.where(df['optimal_cost'] == 0, 1.0, np.nan)
    )
    
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
    
    # Filter finite optimality values
    df_finite = df[np.isfinite(df['optimality'])].copy()
    
    # Group by algorithm and calculate stats
    algo_stats = []
    for algo in sorted(df_finite['algorithm'].unique()):
        df_algo = df_finite[df_finite['algorithm'] == algo]
        
        algo_stats.append({
            'algorithm': algo,
            'count': len(df_algo),
            'mean_optimality': df_algo['optimality'].mean(),
            'median_optimality': df_algo['optimality'].median(),
            'std_optimality': df_algo['optimality'].std(),
            'min_optimality': df_algo['optimality'].min(),
            'max_optimality': df_algo['optimality'].max(),
            'mean_optimal_cost': df_algo['optimal_cost'].mean(),
            'mean_heuristic_cost': df_algo['heuristic_cost'].mean(),
        })
    
    algo_df = pd.DataFrame(algo_stats).sort_values('mean_optimality', ascending=True)  # Ascending = worst first
    
    print(f"\n{'Algorithm':<20} {'Count':<7} {'Mean±Std Optimality':<22} {'Min-Max':<15} {'Avg Costs (O/H)':<20}")
    print("-" * 95)
    
    for _, row in algo_df.iterrows():
        opt_pct = row['mean_optimality'] * 100
        std_pct = row['std_optimality'] * 100
        opt_str = f"{opt_pct:.1f}%±{std_pct:.1f}%"
        range_str = f"{row['min_optimality']*100:.1f}%-{row['max_optimality']*100:.1f}%"
        costs_str = f"{row['mean_optimal_cost']:.1f}/{row['mean_heuristic_cost']:.1f}"
        print(f"{row['algorithm']:<20} {row['count']:<7} {opt_str:<22} {range_str:<15} {costs_str:<20}")
    
    # Highlight worst performers
    print(f"\n🚨 WORST PERFORMERS (optimality < 50%):")
    worst = algo_df[algo_df['mean_optimality'] < 0.5]
    if len(worst) > 0:
        for _, row in worst.iterrows():
            opt_pct = row['mean_optimality'] * 100
            print(f"   • {row['algorithm']}: {opt_pct:.1f}% optimal (n={row['count']})")
            
            # Show detailed breakdown
            df_worst = df_finite[df_finite['algorithm'] == row['algorithm']]
            print(f"     Overhead breakdown:")
            for oh in sorted(df_worst['overhead'].unique()):
                df_oh = df_worst[df_worst['overhead'] == oh]
                oh_pct = df_oh['optimality'].mean() * 100
                print(f"       OH={oh:.2f}: {oh_pct:.1f}% optimal (n={len(df_oh)})")
    else:
        print("   (none - all algorithms achieve >50% optimality)")
    
    # ===========================================
    # 3. OVERHEAD ANALYSIS
    # ===========================================
    print("\n" + "="*80)
    print("3. OVERHEAD ANALYSIS - Pattern across network capacities")
    print("="*80)
    
    for oh in sorted(df_finite['overhead'].unique()):
        df_oh = df_finite[df_finite['overhead'] == oh]
        mean_pct = df_oh['optimality'].mean() * 100
        std_pct = df_oh['optimality'].std() * 100
        median_pct = df_oh['optimality'].median() * 100
        min_pct = df_oh['optimality'].min() * 100
        max_pct = df_oh['optimality'].max() * 100
        
        print(f"\nOverhead {oh:.2f} (cap={int(df_oh['capacity'].iloc[0])} qubits/QPU):")
        print(f"  Mean optimality: {mean_pct:.1f}% ± {std_pct:.1f}%")
        print(f"  Median optimality: {median_pct:.1f}%")
        print(f"  Range: [{min_pct:.1f}%, {max_pct:.1f}%]")
        print(f"  Perfect matches (100%): {(df_oh['optimality'] == 1.0).sum()}/{len(df_oh)} ({(df_oh['optimality'] == 1.0).sum()/len(df_oh)*100:.1f}%)")
    
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
        print(f"     These cases have 100% optimality if heuristic also = 0.")
    
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
    
    print(f"\nOverall optimality variance:")
    print(f"  Standard deviation: {df_finite['optimality'].std():.4f} ({df_finite['optimality'].std()*100:.1f} percentage points)")
    print(f"  IQR (Q3-Q1):       {df_finite['optimality'].quantile(0.75) - df_finite['optimality'].quantile(0.25):.4f}")
    print(f"  Range (max-min):   {df_finite['optimality'].max() - df_finite['optimality'].min():.4f}")
    
    print(f"\nVariance by overhead:")
    for oh in sorted(df_finite['overhead'].unique()):
        df_oh = df_finite[df_finite['overhead'] == oh]
        iqr = df_oh['optimality'].quantile(0.75) - df_oh['optimality'].quantile(0.25)
        std_pct = df_oh['optimality'].std() * 100
        print(f"  OH={oh:.2f}: std={std_pct:.1f} pct points, IQR={iqr:.4f}")
    
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
    overall_pct = df_finite['optimality'].mean() * 100
    print(f"  3. Overall heuristic achieves {overall_pct:.1f}% optimality")
    
    print("\n🔧 SUGGESTED ACTIONS:")
    print("  1. Investigate worst-performing algorithms individually")
    print("  2. Check if heuristic has bugs or bad assumptions for small circuits")
    print("  3. Consider testing on larger circuits (8-12 qubits) where heuristic should shine")
    print("  4. For plots: Consider showing individual algorithm performance or filtering outliers")
    print("  5. The tight boxes suggest you might want to use violin plots or strip plots instead")
    
    print("\n" + "="*80 + "\n")
    
    # Save detailed per-algorithm stats
    output_file = Path('experiment_outputs_mqtbench/algorithm_performance.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    algo_df.to_csv(output_file, index=False)
    print(f"✓ Detailed algorithm stats saved to: {output_file}\n")

if __name__ == '__main__':
    analyze_data()
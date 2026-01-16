#!/usr/bin/env python3
"""
FAST PARALLELIZED VERSION: All HDH Experiments

This script runs ALL experiments with multiprocessing parallelization.
Expected runtime: ~1-2 minutes instead of 10+ minutes!

Usage:
    python hdh_experiments_FAST.py              # Run all experiments
    python hdh_experiments_FAST.py --cores 4    # Use 4 cores
    python hdh_experiments_FAST.py --quick      # Quick mode (reduced tests)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Set up paths
sys.path.insert(0, str(Path.cwd()))

from hdh import HDH
from hdh.passes import compute_cut, cost, parallelism, fair_parallelism, partition_size

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path('experiment_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ Imports successful")
print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")

# =============================================================================
# HDH Generators
# =============================================================================

def generate_random_circuit_hdh(num_qubits: int, depth: int, gate_prob: float = 0.3, seed: int = None) -> HDH:
    """Generate random circuit-like HDH"""
    if seed is not None:
        np.random.seed(seed)
    
    hdh = HDH()
    
    for q in range(num_qubits):
        for t in range(depth + 1):
            hdh.add_node(f"q{q}_t{t}", "q", t)
    
    for q in range(num_qubits):
        for t in range(depth):
            hdh.add_hyperedge({f"q{q}_t{t}", f"q{q}_t{t+1}"}, "q", "wire", role="teledata")
    
    for t in range(depth):
        for q1 in range(num_qubits - 1):
            if np.random.rand() < gate_prob:
                q2 = q1 + 1
                hdh.add_hyperedge({
                    f"q{q1}_t{t}", f"q{q1}_t{t+1}",
                    f"q{q2}_t{t}", f"q{q2}_t{t+1}"
                }, "q", "cnot", role="telegate")
    
    return hdh


def generate_mbqc_hdh(num_qubits: int, depth: int, seed: int = None) -> HDH:
    """Generate MBQC-like HDH (dense entanglement)"""
    if seed is not None:
        np.random.seed(seed)
    
    hdh = HDH()
    for q in range(num_qubits):
        for t in range(depth):
            hdh.add_node(f"q{q}_t{t}", "q", t)
    
    for t in range(depth):
        for i in range(num_qubits):
            for j in range(i+1, min(i+3, num_qubits)):
                hdh.add_hyperedge({f"q{i}_t{t}", f"q{j}_t{t}"}, "q", "entangle")
    
    return hdh


def generate_qw_hdh(num_qubits: int, depth: int, seed: int = None) -> HDH:
    """Generate QW-like HDH (linear chain)"""
    if seed is not None:
        np.random.seed(seed)
    
    hdh = HDH()
    for q in range(num_qubits):
        for t in range(depth):
            hdh.add_node(f"q{q}_t{t}", "q", t)
    
    for t in range(depth):
        for i in range(num_qubits - 1):
            hdh.add_hyperedge({f"q{i}_t{t}", f"q{i+1}_t{t}"}, "q", "hop")
    
    return hdh


def generate_qca_hdh(num_qubits: int, depth: int, seed: int = None) -> HDH:
    """Generate QCA-like HDH (2D lattice)"""
    if seed is not None:
        np.random.seed(seed)
    
    hdh = HDH()
    side = int(np.sqrt(num_qubits))
    actual_qubits = side * side
    
    for q in range(actual_qubits):
        for t in range(depth):
            hdh.add_node(f"q{q}_t{t}", "q", t)
    
    for t in range(depth):
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if j < side - 1:
                    hdh.add_hyperedge({f"q{idx}_t{t}", f"q{idx+1}_t{t}"}, "q", "lattice_h")
                if i < side - 1:
                    hdh.add_hyperedge({f"q{idx}_t{t}", f"q{idx+side}_t{t}"}, "q", "lattice_v")
    
    return hdh


# =============================================================================
# Parallel Worker Functions
# =============================================================================

def worker_overhead_test(config):
    """Worker for Experiment 1: Overhead scaling"""
    overhead, seed = config
    
    np.random.seed(seed)
    num_qubits = 50
    depth = 20
    k = 4
    
    hdh = generate_random_circuit_hdh(num_qubits, depth, 0.3, seed)
    cap = int((num_qubits / k) * overhead)
    
    partitions, cut_cost = compute_cut(hdh, k, cap)
    
    return {
        'overhead': overhead,
        'cut_cost': cut_cost,
        'num_qubits': num_qubits,
        'k': k,
        'cap': cap
    }


def worker_qpu_test(config):
    """Worker for Experiment 2: QPU count scaling"""
    k, seed = config
    
    np.random.seed(seed)
    num_qubits = 50
    depth = 20
    overhead = 1.2
    
    hdh = generate_random_circuit_hdh(num_qubits, depth, 0.3, seed)
    cap = int((num_qubits / k) * overhead)
    
    partitions, cut_cost = compute_cut(hdh, k, cap)
    
    # Count unique qubits per partition
    def count_unique_qubits(partition):
        qubits = set()
        for node in partition:
            if node.startswith('q') and '_' in node:
                q_idx = int(node.split('_')[0][1:])
                qubits.add(q_idx)
        return len(qubits)
    
    partition_sizes = [count_unique_qubits(p) for p in partitions]
    
    return {
        'k': k,
        'cut_cost': cut_cost,
        'max_partition_size': max(partition_sizes) if partition_sizes else 0,
        'min_partition_size': min(partition_sizes) if partition_sizes else 0,
        'avg_partition_size': np.mean(partition_sizes) if partition_sizes else 0
    }


def worker_model_test(config):
    """Worker for Experiment 3: Cross-model comparison"""
    model_name, num_qubits, seed = config
    
    np.random.seed(seed)
    depth = 20
    k = 4
    overhead = 1.2
    
    # Generate HDH based on model
    if model_name == 'Circuit':
        hdh = generate_random_circuit_hdh(num_qubits, depth, 0.3, seed)
    elif model_name == 'MBQC':
        hdh = generate_mbqc_hdh(num_qubits, depth, seed)
    elif model_name == 'QW':
        hdh = generate_qw_hdh(num_qubits, depth, seed)
    elif model_name == 'QCA':
        hdh = generate_qca_hdh(num_qubits, depth, seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    cap = int((num_qubits / k) * overhead)
    partitions, cut_cost = compute_cut(hdh, k, cap)
    
    return {
        'model': model_name,
        'num_qubits': num_qubits,
        'cut_cost': cut_cost,
        'k': k
    }


def worker_kahypar_test(config):
    """Worker for Experiment 4: KaHyPar comparison"""
    test_type, params, seed = config
    
    np.random.seed(seed)
    
    # Unpack parameters based on test type
    if test_type == 'size':
        num_qubits, depth, k = params
    elif test_type == 'qpu':
        num_qubits, depth, k = params
    elif test_type == 'model':
        model_name, num_qubits, depth, k = params
    
    # Generate HDH
    if test_type in ['size', 'qpu']:
        hdh = generate_random_circuit_hdh(num_qubits, depth, 0.3, seed)
        label = f"{test_type}_{num_qubits if test_type == 'size' else k}"
    else:
        if model_name == 'Circuit':
            hdh = generate_random_circuit_hdh(num_qubits, depth, 0.3, seed)
        elif model_name == 'MBQC':
            hdh = generate_mbqc_hdh(num_qubits, depth, seed)
        elif model_name == 'QW':
            hdh = generate_qw_hdh(num_qubits, depth, seed)
        elif model_name == 'QCA':
            hdh = generate_qca_hdh(num_qubits, depth, seed)
        label = f"model_{model_name}"
    
    cap = (num_qubits + k - 1) // k
    capacities = [cap] * k
    
    # Run temporal greedy
    partitions_temporal, _ = compute_cut(hdh, k, cap)
    
    # Compute metrics
    cost_q, cost_c = cost(hdh, partitions_temporal)
    par_metrics = parallelism(hdh, partitions_temporal)
    fair_metrics = fair_parallelism(hdh, partitions_temporal, capacities)
    sizes = partition_size(partitions_temporal)
    
    result_temporal = {
        'test': label,
        'k': k,
        'cap': cap,
        'num_qubits': num_qubits,
        'num_nodes': len(hdh.S),
        'num_edges': len(hdh.C),
        'cut_cost': cost_q + cost_c,
        'avg_parallelism': par_metrics['average_parallelism'],
        'avg_fair_parallelism': fair_metrics['average_fair_parallelism'],
        'balance_ratio': min(sizes) / max(sizes) if sizes and max(sizes) > 0 else 1.0,
        'method': 'temporal_greedy'
    }
    
    # Run baseline (METIS)
    try:
        bins, _, _, method = metis_telegate(hdh, k, cap)
        partitions_baseline = [set() for _ in range(k)]
        for bin_idx, qubit_set in enumerate(bins):
            for qubit_str in qubit_set:
                if qubit_str.startswith('q'):
                    q_idx = int(qubit_str[1:])
                    for node in hdh.S:
                        if node.startswith(f'q{q_idx}_'):
                            partitions_baseline[bin_idx].add(node)
        
        cost_q_b, cost_c_b = cost(hdh, partitions_baseline)
        par_metrics_b = parallelism(hdh, partitions_baseline)
        fair_metrics_b = fair_parallelism(hdh, partitions_baseline, capacities)
        sizes_b = partition_size(partitions_baseline)
        
        result_baseline = {
            'test': label,
            'k': k,
            'cap': cap,
            'num_qubits': num_qubits,
            'num_nodes': len(hdh.S),
            'num_edges': len(hdh.C),
            'cut_cost': cost_q_b + cost_c_b,
            'avg_parallelism': par_metrics_b['average_parallelism'],
            'avg_fair_parallelism': fair_metrics_b['average_fair_parallelism'],
            'balance_ratio': min(sizes_b) / max(sizes_b) if sizes_b and max(sizes_b) > 0 else 1.0,
            'method': method
        }
    except Exception as e:
        result_baseline = None
    
    return (result_temporal, result_baseline)


# =============================================================================
# Experiment Runners
# =============================================================================

def run_experiment_1(n_cores=None, quick_mode=False):
    """Experiment 1: Overhead Factor Scaling (PARALLELIZED)"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Overhead Factor Scaling (PARALLELIZED)")
    print("="*70)
    
    if quick_mode:
        overhead_values = [1.0, 1.5, 2.0]
        n_samples = 3
    else:
        overhead_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        n_samples = 5
    
    # Create configurations
    configs = [(oh, seed) for oh in overhead_values for seed in range(n_samples)]
    
    print(f"Testing {len(overhead_values)} overhead values × {n_samples} samples = {len(configs)} tests")
    
    # Run in parallel
    if n_cores is None:
        n_cores = cpu_count()
    
    start_time = time.time()
    
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_overhead_test, configs),
            total=len(configs),
            desc="Overhead tests"
        ))
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.1f}s ({len(configs)/elapsed:.1f} tests/sec)")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'overhead_analysis_results.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'overhead_analysis_results.csv'}")
    
    return df


def run_experiment_2(n_cores=None, quick_mode=False):
    """Experiment 2: QPU Count Scaling (PARALLELIZED)"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: QPU Count Scaling (PARALLELIZED)")
    print("="*70)
    
    if quick_mode:
        qpu_counts = [2, 4, 6]
        n_samples = 3
    else:
        qpu_counts = [2, 3, 4, 5, 6]
        n_samples = 5
    
    configs = [(k, seed) for k in qpu_counts for seed in range(n_samples)]
    
    print(f"Testing {len(qpu_counts)} QPU counts × {n_samples} samples = {len(configs)} tests")
    
    if n_cores is None:
        n_cores = cpu_count()
    
    start_time = time.time()
    
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_qpu_test, configs),
            total=len(configs),
            desc="QPU count tests"
        ))
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.1f}s ({len(configs)/elapsed:.1f} tests/sec)")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'qpu_scaling_results.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'qpu_scaling_results.csv'}")
    
    return df


def run_experiment_3(n_cores=None, quick_mode=False):
    """Experiment 3: Cross-Model Comparison (PARALLELIZED)"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Cross-Model Comparison (PARALLELIZED)")
    print("="*70)
    
    models = ['Circuit', 'MBQC', 'QW', 'QCA']
    
    if quick_mode:
        qubit_sizes = [16, 32, 64]
        n_samples = 2
    else:
        qubit_sizes = [8, 16, 32, 64, 128]
        n_samples = 3
    
    configs = [
        (model, nq, seed)
        for model in models
        for nq in qubit_sizes
        for seed in range(n_samples)
    ]
    
    print(f"Testing {len(models)} models × {len(qubit_sizes)} sizes × {n_samples} samples = {len(configs)} tests")
    
    if n_cores is None:
        n_cores = cpu_count()
    
    start_time = time.time()
    
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_model_test, configs),
            total=len(configs),
            desc="Cross-model tests"
        ))
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.1f}s ({len(configs)/elapsed:.1f} tests/sec)")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'cross_model_results.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'cross_model_results.csv'}")
    
    return df


def run_experiment_4(n_cores=None, quick_mode=False):
    """Experiment 4: KaHyPar Comparison (PARALLELIZED)"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Temporal Greedy vs KaHyPar/METIS (PARALLELIZED)")
    print("="*70)
    
    configs = []
    
    # Problem size scaling
    if quick_mode:
        size_tests = [32, 64]
    else:
        size_tests = [16, 32, 64, 128]
    
    for nq in size_tests:
        configs.append(('size', (nq, 20, 4), 42))
    
    # QPU count
    if quick_mode:
        qpu_tests = [2, 6]
    else:
        qpu_tests = [2, 4, 6, 8]
    
    for k in qpu_tests:
        configs.append(('qpu', (64, 20, k), 42))
    
    # Models
    models = ['Circuit', 'MBQC', 'QW', 'QCA'] if not quick_mode else ['Circuit', 'MBQC']
    for model in models:
        configs.append(('model', (model, 64, 15, 4), 42))
    
    print(f"Testing {len(configs)} configurations")
    
    if n_cores is None:
        n_cores = cpu_count()
    
    start_time = time.time()
    
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_kahypar_test, configs),
            total=len(configs),
            desc="KaHyPar comparison"
        ))
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.1f}s ({len(configs)/elapsed:.1f} tests/sec)")
    
    # Separate temporal and baseline
    results_temporal = [r[0] for r in results]
    results_baseline = [r[1] for r in results if r[1] is not None]
    
    df_temporal = pd.DataFrame(results_temporal)
    df_baseline = pd.DataFrame(results_baseline)
    
    df_temporal.to_csv(OUTPUT_DIR / 'exp4_temporal_greedy_results.csv', index=False)
    df_baseline.to_csv(OUTPUT_DIR / 'exp4_kahypar_baseline_results.csv', index=False)
    
    print(f"✓ Saved: {OUTPUT_DIR / 'exp4_temporal_greedy_results.csv'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'exp4_kahypar_baseline_results.csv'}")
    
    # Compute comparison
    if not df_baseline.empty:
        df_comparison = df_temporal.merge(
            df_baseline,
            on=['test', 'k', 'cap', 'num_qubits', 'num_nodes', 'num_edges'],
            suffixes=('_temporal', '_baseline')
        )
        
        for metric in ['cut_cost', 'avg_parallelism', 'avg_fair_parallelism', 'balance_ratio']:
            temporal_col = f'{metric}_temporal'
            baseline_col = f'{metric}_baseline'
            
            if metric == 'cut_cost':
                df_comparison[f'{metric}_ratio'] = df_comparison[temporal_col] / df_comparison[baseline_col]
            else:
                df_comparison[f'{metric}_ratio'] = df_comparison[temporal_col] / df_comparison[baseline_col]
        
        df_comparison.to_csv(OUTPUT_DIR / 'exp4_comparison_analysis.csv', index=False)
        print(f"✓ Saved: {OUTPUT_DIR / 'exp4_comparison_analysis.csv'}")
        
        return df_temporal, df_baseline, df_comparison
    
    return df_temporal, df_baseline, None


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_experiment_1(df):
    """Plot Experiment 1 results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary = df.groupby('overhead')['cut_cost'].agg(['mean', 'std'])
    
    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'],
                fmt='o-', linewidth=2.5, markersize=8, capsize=5,
                color='#06A77D', alpha=0.9)
    
    ax.set_xlabel('Overhead Factor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=13, fontweight='bold')
    ax.set_title('Overhead Factor vs Communication Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_overhead_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig1_overhead_scaling.png'}")


def plot_experiment_2(df):
    """Plot Experiment 2 results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary = df.groupby('k')['cut_cost'].agg(['mean', 'std'])
    
    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'],
                fmt='s-', linewidth=2.5, markersize=8, capsize=5,
                color='#D4A574', alpha=0.9)
    
    ax.set_xlabel('Number of QPUs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=13, fontweight='bold')
    ax.set_title('QPU Count vs Communication Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_qpu_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig2_qpu_scaling.png'}")


def plot_experiment_3(df):
    """Plot Experiment 3 results"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Circuit': '#06A77D', 'MBQC': '#D4A574', 'QW': '#2E86AB', 'QCA': '#A23B72'}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        summary = model_data.groupby('num_qubits')['cut_cost'].mean()
        
        ax.plot(summary.index, summary.values, 'o-', label=model,
                linewidth=2.5, markersize=8, color=colors.get(model, '#000000'), alpha=0.9)
    
    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Model Partitioning Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_cross_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig3_cross_model.png'}")


def plot_experiment_4(df_comparison):
    """Plot Experiment 4 results"""
    if df_comparison is None or df_comparison.empty:
        print("⚠ No comparison data to plot for Experiment 4")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: Cut cost scaling
    ax = axes[0, 0]
    size_data = df_comparison[df_comparison['test'].str.contains('size')]
    if not size_data.empty:
        size_data = size_data.sort_values('num_qubits')
        ax.plot(size_data['num_qubits'], size_data['cut_cost_temporal'],
                'o-', label='Temporal Greedy', linewidth=2.5, markersize=8, color='#06A77D')
        ax.plot(size_data['num_qubits'], size_data['cut_cost_baseline'],
                's--', label='Baseline', linewidth=2.5, markersize=8, color='#D4A574')
        ax.set_xlabel('Problem Size (Qubits)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Communication Cost', fontsize=12, fontweight='bold')
        ax.set_title('(a) Cut Cost Scaling', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    # Panel 2: Relative quality
    ax = axes[0, 1]
    if not size_data.empty and 'cut_cost_ratio' in size_data.columns:
        quality = 100.0 / size_data['cut_cost_ratio']
        ax.plot(size_data['num_qubits'], quality,
                'o-', linewidth=2.5, markersize=8, color='#2E86AB')
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.fill_between(size_data['num_qubits'], 0, quality, alpha=0.2, color='#2E86AB')
        ax.set_xlabel('Problem Size (Qubits)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Quality (%)', fontsize=12, fontweight='bold')
        ax.set_title('(b) Quality vs Problem Size', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 110)
    
    # Panel 3: Parallelism
    ax = axes[1, 0]
    qpu_data = df_comparison[df_comparison['test'].str.contains('qpu')]
    if not qpu_data.empty:
        qpu_data = qpu_data.sort_values('k')
        ax.plot(qpu_data['k'], qpu_data['avg_parallelism_temporal'],
                'o-', label='Temporal Greedy', linewidth=2.5, markersize=8, color='#06A77D')
        ax.plot(qpu_data['k'], qpu_data['avg_parallelism_baseline'],
                's--', label='Baseline', linewidth=2.5, markersize=8, color='#D4A574')
        ax.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Parallelism', fontsize=12, fontweight='bold')
        ax.set_title('(c) Parallelism Scaling', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Overall Performance Summary\n\n' +
            f"Avg Cut Cost Ratio: {df_comparison['cut_cost_ratio'].mean():.2f}x\n" +
            f"Avg Parallelism Ratio: {df_comparison['avg_parallelism_ratio'].mean():.2%}\n" +
            f"Avg Fair Parallelism Ratio: {df_comparison['avg_fair_parallelism_ratio'].mean():.2%}",
            ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_kahypar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig4_kahypar_comparison.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run parallelized HDH experiments')
    parser.add_argument('--cores', type=int, default=None, help='Number of cores to use')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (reduced tests)')
    parser.add_argument('--exp', type=str, default='all', help='Which experiment to run (1,2,3,4, or all)')
    
    args = parser.parse_args()
    
    n_cores = args.cores if args.cores else cpu_count()
    
    print("\n" + "="*70)
    print("PARALLELIZED HDH PARTITIONING EXPERIMENTS")
    print("="*70)
    print(f"Using {n_cores} CPU cores")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Experiments: {args.exp}")
    print("="*70)
    
    total_start = time.time()
    
    # Run experiments
    if args.exp in ['1', 'all']:
        df1 = run_experiment_1(n_cores, args.quick)
        plot_experiment_1(df1)
    
    if args.exp in ['2', 'all']:
        df2 = run_experiment_2(n_cores, args.quick)
        plot_experiment_2(df2)
    
    if args.exp in ['3', 'all']:
        df3 = run_experiment_3(n_cores, args.quick)
        plot_experiment_3(df3)
    
    if args.exp in ['4', 'all']:
        df4_t, df4_b, df4_c = run_experiment_4(n_cores, args.quick)
        plot_experiment_4(df4_c)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"All results saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()

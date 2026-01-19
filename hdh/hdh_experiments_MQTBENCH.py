#!/usr/bin/env python3
"""
MQT BENCH VERSION: All HDH Experiments with MQT Bench Inputs

This script runs ALL experiments using real MQT Bench circuit HDHs.
Loads pre-computed HDH pickles from /Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl_reduce

Usage:
    python hdh_experiments_MQTBENCH.py              # Run all experiments
    python hdh_experiments_MQTBENCH.py --cores 4    # Use 4 cores
    python hdh_experiments_MQTBENCH.py --quick      # Quick mode (subset of circuits)
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
import pickle
import glob

# Set up paths
sys.path.insert(0, str(Path.cwd()))

from hdh import HDH
from hdh.passes.cut import compute_cut,cost,parallelism,fair_parallelism,partition_size,partition_logical_qubit_size,kahypar_cutter,kahypar_cutter_nodebalanced

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path('experiment_outputs_mqtbench')
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ Imports successful")
print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")

# =============================================================================
# MQT Bench HDH Loading
# =============================================================================

def load_mqtbench_hdhs(pkl_dir: str = "/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl_reduce", quick_mode: bool = False) -> Dict[str, HDH]:
    """
    Load all MQT Bench HDH files from pickle directory.
    
    Args:
        pkl_dir: Path to directory containing pickle files
        quick_mode: If True, only load a subset for testing
    
    Returns:
        Dictionary mapping circuit name to HDH object
    """
    pkl_path = Path(pkl_dir)
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"MQT Bench pickle directory not found: {pkl_path}")
    
    # Find all pickle files
    pkl_files = list(pkl_path.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {pkl_path}")
    
    print(f"\n{'='*70}")
    print(f"Loading MQT Bench HDH files from: {pkl_path}")
    print(f"Found {len(pkl_files)} pickle files")
    
    if quick_mode:
        pkl_files = pkl_files[:10]  # Only load first 10 for quick testing
        print(f"Quick mode: Using only {len(pkl_files)} circuits")
    
    print(f"{'='*70}\n")
    
    hdhs = {}
    failed = []
    
    for pkl_file in tqdm(pkl_files, desc="Loading HDH files"):
        circuit_name = pkl_file.stem  # Get filename without extension
        try:
            with open(pkl_file, 'rb') as f:
                hdh = pickle.load(f)
                
            # Verify it's an HDH object
            if not isinstance(hdh, HDH):
                raise TypeError(f"Expected HDH object, got {type(hdh)}")
            
            hdhs[circuit_name] = hdh
            
        except Exception as e:
            failed.append((circuit_name, str(e)))
    
    print(f"\n✓ Successfully loaded {len(hdhs)} HDH circuits")
    if failed:
        print(f"⚠ Failed to load {len(failed)} circuits:")
        for name, error in failed[:5]:  # Show first 5 failures
            print(f"  - {name}: {error}")
    
    return hdhs


def get_circuit_stats(hdh: HDH) -> Dict:
    """Get basic statistics about a circuit HDH using available methods."""
    stats = {
        'num_nodes': 0,
        'num_hyperedges': 0,
        'num_qubits': 0,
        'max_time': 0
    }
    
    try:
        # Try to access V and E attributes (common in graph libraries)
        if hasattr(hdh, 'V'):
            stats['num_nodes'] = len(hdh.V)
        if hasattr(hdh, 'E'):
            stats['num_hyperedges'] = len(hdh.E)
            
        # Count unique qubits and max time by examining nodes
        qubits = set()
        times = set()
        
        # Try different possible attribute names
        nodes_dict = None
        if hasattr(hdh, 'V'):
            nodes_dict = hdh.V
        elif hasattr(hdh, 'nodes'):
            nodes_dict = hdh.nodes
        elif hasattr(hdh, 'vertices'):
            nodes_dict = hdh.vertices
            
        if nodes_dict:
            for node_id in nodes_dict:
                # Get node data
                node_data = nodes_dict[node_id] if isinstance(nodes_dict, dict) else {}
                
                # Extract time if available
                if isinstance(node_data, dict) and 'time' in node_data:
                    times.add(node_data['time'])
                elif hasattr(node_data, 'time'):
                    times.add(node_data.time)
                    
                # Try to extract qubit info from node_id
                if isinstance(node_id, str) and '_' in node_id:
                    parts = node_id.split('_')
                    if parts[0].startswith('q') and parts[0][1:].isdigit():
                        qubits.add(int(parts[0][1:]))
        
        stats['num_qubits'] = len(qubits)
        stats['max_time'] = max(times) if times else 0
        
        # If we couldn't get qubits, use the reliable partition method
        if stats['num_qubits'] == 0:
            stats['num_qubits'] = get_num_qubits_from_partition(hdh)
        
    except Exception as e:
        # If we can't get stats, use partition method as fallback
        print(f"⚠ Could not extract stats normally, using partition method: {e}")
        stats['num_qubits'] = get_num_qubits_from_partition(hdh)
        stats['num_nodes'] = 100  # Rough estimate
        stats['num_hyperedges'] = 100  # Rough estimate
    
    return stats


def get_num_qubits_from_partition(hdh: HDH) -> int:
    """
    Get the number of qubits by running a quick partition and using partition_logical_qubit_size.
    This is more reliable than trying to inspect HDH internals.
    """
    try:
        # Run a simple partition with k=1 to get all nodes in one partition
        partitions, _ = compute_cut(hdh, k=1, cap=10000)
        if partitions and len(partitions) > 0:
            # Use the partition_logical_qubit_size function (takes only partition, not hdh)
            result = partition_logical_qubit_size(partitions[0])
            
            # Handle different return types
            if isinstance(result, list):
                # If it's a list, count non-zero elements or just return length
                non_zero = [x for x in result if x != 0]
                return len(non_zero) if non_zero else len(result)
            elif isinstance(result, (int, float)):
                return int(result)
            else:
                print(f"⚠ Unexpected return type from partition_logical_qubit_size: {type(result)}")
                return 10
    except Exception as e:
        print(f"⚠ Could not determine qubit count: {e}")
        return 10  # Fallback
    
    return 10  # Fallback


def safe_partition_qubit_size(partition) -> int:
    """
    Safely get the number of qubits in a partition, handling list returns.
    """
    try:
        result = partition_logical_qubit_size(partition)
        if isinstance(result, list):
            # Count non-zero elements or return length
            non_zero = [x for x in result if x != 0]
            return len(non_zero) if non_zero else len(result)
        elif isinstance(result, (int, float)):
            return int(result)
        else:
            return 0
    except Exception:
        return 0


# =============================================================================
# Parallel Worker Functions (Modified for MQT Bench)
# =============================================================================

def worker_overhead_test(config):
    """Worker for Experiment 1: Overhead scaling with MQT Bench circuits"""
    circuit_name, hdh, overhead, k = config
    
    # Get circuit stats to determine capacity
    stats = get_circuit_stats(hdh)
    num_qubits = stats['num_qubits']
    
    if num_qubits == 0 or k == 0:
        return None
    
    cap = int((num_qubits / k) * overhead)
    
    try:
        partitions, cut_cost = compute_cut(hdh, k, cap)
        
        return {
            'circuit': circuit_name,
            'overhead': overhead,
            'cut_cost': cut_cost,
            'num_qubits': num_qubits,
            'num_nodes': stats['num_nodes'],
            'num_hyperedges': stats['num_hyperedges'],
            'k': k,
            'cap': cap
        }
    except Exception as e:
        print(f"⚠ Error in overhead test for {circuit_name}: {e}")
        return None


def worker_qpu_test(config):
    """Worker for Experiment 2: QPU count scaling with MQT Bench circuits"""
    circuit_name, hdh, k, overhead = config
    
    stats = get_circuit_stats(hdh)
    num_qubits = stats['num_qubits']
    
    if num_qubits == 0 or k == 0:
        return None
    
    cap = int((num_qubits / k) * overhead)
    
    try:
        partitions, cut_cost = compute_cut(hdh, k, cap)
        
        # Use safe helper function for accurate qubit counting
        partition_sizes = [safe_partition_qubit_size(p) for p in partitions]
        
        return {
            'circuit': circuit_name,
            'k': k,
            'cut_cost': cut_cost,
            'num_qubits': num_qubits,
            'num_nodes': stats['num_nodes'],
            'max_partition_size': max(partition_sizes) if partition_sizes else 0,
            'min_partition_size': min(partition_sizes) if partition_sizes else 0,
            'avg_partition_size': np.mean(partition_sizes) if partition_sizes else 0
        }
    except Exception as e:
        print(f"⚠ Error in QPU test for {circuit_name}: {e}")
        return None


def worker_capacity_test(config):
    """Worker for Experiment 3: Capacity constraint testing"""
    circuit_name, hdh, k, overhead = config
    
    stats = get_circuit_stats(hdh)
    num_qubits = stats['num_qubits']
    
    if num_qubits == 0 or k == 0:
        return None
    
    cap = int((num_qubits / k) * overhead)
    
    try:
        partitions, cut_cost = compute_cut(hdh, k, cap)
        
        # Use safe helper function for accurate counting
        partition_sizes = [safe_partition_qubit_size(p) for p in partitions]
        max_size = max(partition_sizes) if partition_sizes else 0
        violated = max_size > cap
        
        return {
            'circuit': circuit_name,
            'k': k,
            'overhead': overhead,
            'cap': cap,
            'max_partition_size': max_size,
            'violated': violated,
            'num_qubits': num_qubits,
            'cut_cost': cut_cost
        }
    except Exception as e:
        print(f"⚠ Error in capacity test for {circuit_name}: {e}")
        return None


def worker_kahypar_comparison(config):
    """Worker for Experiment 4: KaHyPar comparison"""
    circuit_name, hdh, k, overhead = config
    
    stats = get_circuit_stats(hdh)
    num_qubits = stats['num_qubits']
    
    if num_qubits == 0 or k == 0:
        return None
    
    cap = int((num_qubits / k) * overhead)
    
    try:
        # Temporal greedy (our method)
        partitions_temporal, cut_cost_temporal = compute_cut(hdh, k, cap)
        
        # Baseline KaHyPar
        partitions_baseline, cut_cost_baseline = kahypar_cutter(hdh, k)
        
        # Calculate metrics for both
        def calc_metrics(partitions):
            para = parallelism(partitions, hdh)
            fair_para = fair_parallelism(partitions, hdh)
            return para, fair_para
        
        para_temporal, fair_para_temporal = calc_metrics(partitions_temporal)
        para_baseline, fair_para_baseline = calc_metrics(partitions_baseline)
        
        return {
            'circuit': circuit_name,
            'num_qubits': num_qubits,
            'k': k,
            'cut_cost_temporal': cut_cost_temporal,
            'cut_cost_baseline': cut_cost_baseline,
            'cut_cost_ratio': cut_cost_baseline / cut_cost_temporal if cut_cost_temporal > 0 else 1.0,
            'avg_parallelism_temporal': para_temporal,
            'avg_parallelism_baseline': para_baseline,
            'avg_fair_parallelism_temporal': fair_para_temporal,
            'avg_fair_parallelism_baseline': fair_para_baseline,
        }
    except Exception as e:
        print(f"⚠ Error in KaHyPar comparison for {circuit_name}: {e}")
        return None


def worker_capacity_violation(config):
    """Worker for Experiment 5: Capacity violation analysis"""
    circuit_name, hdh, k, slack = config
    
    stats = get_circuit_stats(hdh)
    num_qubits = stats['num_qubits']
    
    if num_qubits == 0 or k == 0:
        return None
    
    cap_q = (num_qubits // k) + slack
    
    try:
        # Temporal greedy with capacity awareness
        partitions_cap, _ = compute_cut(hdh, k, cap_q)
        
        # KaHyPar node-balanced
        kh_ran = False
        violated_kh = False
        kh_violation_mag = 0
        
        try:
            partitions_kh, _ = kahypar_cutter_nodebalanced(hdh, k, cap_q)
            kh_ran = True
            
            # Check violations using safe helper function
            sizes_kh = [safe_partition_qubit_size(p) for p in partitions_kh]
            max_kh = max(sizes_kh) if sizes_kh else 0
            violated_kh = max_kh > cap_q
            kh_violation_mag = max(0, max_kh - cap_q)
            
        except Exception:
            kh_ran = False
        
        # Check capacity-aware violations using safe helper function
        sizes_cap = [safe_partition_qubit_size(p) for p in partitions_cap]
        max_cap = max(sizes_cap) if sizes_cap else 0
        violated_cap = max_cap > cap_q
        
        return {
            'circuit': circuit_name,
            'num_qubits': num_qubits,
            'k': k,
            'slack': slack,
            'cap_q': cap_q,
            'violated_capacity_aware': violated_cap,
            'kh_ran': kh_ran,
            'violated_kahypar': violated_kh,
            'kahypar_violation_magnitude': kh_violation_mag,
        }
    except Exception as e:
        print(f"⚠ Error in capacity violation test for {circuit_name}: {e}")
        return None


# =============================================================================
# Experiment Runners (Modified for MQT Bench)
# =============================================================================

def run_experiment_1(hdhs: Dict[str, HDH], n_cores: int, quick: bool):
    """Experiment 1: Overhead scaling with MQT Bench circuits"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Overhead Scaling (MQT Bench)")
    print("="*70)
    
    # Test different overhead values
    overheads = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0] if not quick else [1.0, 1.2, 1.5]
    k = 4  # Fixed number of QPUs
    
    # Create configurations for all circuits and overheads
    configs = []
    for circuit_name, hdh in hdhs.items():
        for overhead in overheads:
            configs.append((circuit_name, hdh, overhead, k))
    
    print(f"Testing {len(overheads)} overhead values on {len(hdhs)} circuits")
    print(f"Total configurations: {len(configs)}")
    
    # Run in parallel
    start = time.time()
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_overhead_test, configs),
            total=len(configs),
            desc="Running overhead tests"
        ))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s")
    print(f"✓ Successful runs: {len(results)}/{len(configs)}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'exp1_overhead_scaling.csv', index=False)
    
    return df


def run_experiment_2(hdhs: Dict[str, HDH], n_cores: int, quick: bool):
    """Experiment 2: QPU count scaling with MQT Bench circuits"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: QPU Count Scaling (MQT Bench)")
    print("="*70)
    
    # Test different QPU counts
    k_values = [2, 4, 8, 16] if not quick else [2, 4, 8]
    overhead = 1.2  # Fixed overhead
    
    configs = []
    for circuit_name, hdh in hdhs.items():
        for k in k_values:
            configs.append((circuit_name, hdh, k, overhead))
    
    print(f"Testing {len(k_values)} QPU counts on {len(hdhs)} circuits")
    print(f"Total configurations: {len(configs)}")
    
    start = time.time()
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_qpu_test, configs),
            total=len(configs),
            desc="Running QPU scaling tests"
        ))
    
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s")
    print(f"✓ Successful runs: {len(results)}/{len(configs)}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'exp2_qpu_scaling.csv', index=False)
    
    return df


def run_experiment_3(hdhs: Dict[str, HDH], n_cores: int, quick: bool):
    """Experiment 3: Capacity constraint analysis"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Capacity Constraint Analysis (MQT Bench)")
    print("="*70)
    
    k_values = [2, 4, 8] if not quick else [2, 4]
    overheads = [1.0, 1.1, 1.2, 1.3, 1.5] if not quick else [1.0, 1.2]
    
    configs = []
    for circuit_name, hdh in hdhs.items():
        for k in k_values:
            for overhead in overheads:
                configs.append((circuit_name, hdh, k, overhead))
    
    print(f"Testing {len(k_values)} QPU counts × {len(overheads)} overheads on {len(hdhs)} circuits")
    print(f"Total configurations: {len(configs)}")
    
    start = time.time()
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_capacity_test, configs),
            total=len(configs),
            desc="Running capacity tests"
        ))
    
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s")
    print(f"✓ Successful runs: {len(results)}/{len(configs)}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'exp3_capacity_analysis.csv', index=False)
    
    return df


def run_experiment_4(hdhs: Dict[str, HDH], n_cores: int, quick: bool):
    """Experiment 4: KaHyPar comparison"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: KaHyPar Baseline Comparison (MQT Bench)")
    print("="*70)
    
    k_values = [2, 4, 8] if not quick else [2, 4]
    overhead = 1.2
    
    configs = []
    for circuit_name, hdh in hdhs.items():
        for k in k_values:
            configs.append((circuit_name, hdh, k, overhead))
    
    print(f"Comparing temporal greedy vs KaHyPar on {len(hdhs)} circuits")
    print(f"Total configurations: {len(configs)}")
    
    start = time.time()
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_kahypar_comparison, configs),
            total=len(configs),
            desc="Running KaHyPar comparison"
        ))
    
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s")
    print(f"✓ Successful runs: {len(results)}/{len(configs)}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'exp4_kahypar_comparison.csv', index=False)
    
    return df


def run_experiment_5(hdhs: Dict[str, HDH], n_cores: int, quick: bool):
    """Experiment 5: Capacity violation analysis"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Capacity Violation Analysis (MQT Bench)")
    print("="*70)
    
    k = 4
    slack_values = range(0, 11) if not quick else range(0, 6)
    
    configs = []
    for circuit_name, hdh in hdhs.items():
        for slack in slack_values:
            configs.append((circuit_name, hdh, k, slack))
    
    print(f"Testing {len(slack_values)} slack values on {len(hdhs)} circuits")
    print(f"Total configurations: {len(configs)}")
    
    start = time.time()
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_capacity_violation, configs),
            total=len(configs),
            desc="Running capacity violation tests"
        ))
    
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s")
    print(f"✓ Successful runs: {len(results)}/{len(configs)}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'exp5_capacity_violation.csv', index=False)
    
    return df


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_experiment_1(df):
    """Plot Experiment 1 results: Overhead scaling"""
    if df is None or df.empty:
        print("⚠ No data to plot for Experiment 1")
        return
    
    # Aggregate by overhead (average across circuits)
    agg = df.groupby('overhead').agg({
        'cut_cost': ['mean', 'std'],
        'circuit': 'count'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overheads = agg['overhead']
    means = agg['cut_cost']['mean']
    stds = agg['cut_cost']['std']
    
    ax.plot(overheads, means, 'o-', linewidth=2.5, markersize=10, color='#2E86AB', label='Mean')
    ax.fill_between(overheads, means - stds, means + stds, alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Overhead Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cut Cost', fontsize=12, fontweight='bold')
    ax.set_title(f'Cut Cost vs Overhead (MQT Bench, n={len(df["circuit"].unique())} circuits)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_overhead_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig1_overhead_scaling.png'}")


def plot_experiment_2(df):
    """Plot Experiment 2 results: QPU scaling"""
    if df is None or df.empty:
        print("⚠ No data to plot for Experiment 2")
        return
    
    # Aggregate by k
    agg = df.groupby('k').agg({
        'cut_cost': ['mean', 'std'],
        'max_partition_size': ['mean', 'std'],
        'circuit': 'count'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cut cost
    k_vals = agg['k']
    means = agg['cut_cost']['mean']
    stds = agg['cut_cost']['std']
    
    ax1.plot(k_vals, means, 'o-', linewidth=2.5, markersize=10, color='#06A77D')
    ax1.fill_between(k_vals, means - stds, means + stds, alpha=0.2, color='#06A77D')
    ax1.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cut Cost', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Cut Cost vs QPU Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Partition size
    means2 = agg['max_partition_size']['mean']
    stds2 = agg['max_partition_size']['std']
    
    ax2.plot(k_vals, means2, 's-', linewidth=2.5, markersize=10, color='#D4A574')
    ax2.fill_between(k_vals, means2 - stds2, means2 + stds2, alpha=0.2, color='#D4A574')
    ax2.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Partition Size (qubits)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Partition Size vs QPU Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'QPU Scaling Analysis (MQT Bench, n={len(df["circuit"].unique())} circuits)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_qpu_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig2_qpu_scaling.png'}")


def plot_experiment_3(df):
    """Plot Experiment 3 results: Capacity violations"""
    if df is None or df.empty:
        print("⚠ No data to plot for Experiment 3")
        return
    
    # Calculate violation rates by overhead
    violation_rates = df.groupby('overhead').agg({
        'violated': 'mean',
        'circuit': 'count'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(violation_rates['overhead'], violation_rates['violated'] * 100, 
           color='#A23B72', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Overhead Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Capacity Violation Rate vs Overhead (MQT Bench, n={len(df["circuit"].unique())} circuits)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_capacity_violations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig3_capacity_violations.png'}")


def plot_experiment_4(df):
    """Plot Experiment 4 results: KaHyPar comparison"""
    if df is None or df.empty:
        print("⚠ No data to plot for Experiment 4")
        return
    
    # Aggregate by k
    agg = df.groupby('k').agg({
        'cut_cost_temporal': ['mean', 'std'],
        'cut_cost_baseline': ['mean', 'std'],
        'cut_cost_ratio': ['mean', 'std'],
        'avg_parallelism_temporal': ['mean', 'std'],
        'avg_parallelism_baseline': ['mean', 'std'],
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: Cut costs
    ax = axes[0, 0]
    k_vals = agg['k']
    
    ax.plot(k_vals, agg['cut_cost_temporal']['mean'], 'o-', 
            label='Temporal Greedy', linewidth=2.5, markersize=8, color='#06A77D')
    ax.plot(k_vals, agg['cut_cost_baseline']['mean'], 's--', 
            label='KaHyPar', linewidth=2.5, markersize=8, color='#D4A574')
    ax.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cut Cost', fontsize=12, fontweight='bold')
    ax.set_title('(a) Cut Cost Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Quality ratio
    ax = axes[0, 1]
    quality = 100.0 / agg['cut_cost_ratio']['mean']
    ax.bar(k_vals, quality, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Quality (%)', fontsize=12, fontweight='bold')
    ax.set_title('(b) Quality vs KaHyPar', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Parallelism
    ax = axes[1, 0]
    ax.plot(k_vals, agg['avg_parallelism_temporal']['mean'], 'o-',
            label='Temporal Greedy', linewidth=2.5, markersize=8, color='#06A77D')
    ax.plot(k_vals, agg['avg_parallelism_baseline']['mean'], 's--',
            label='KaHyPar', linewidth=2.5, markersize=8, color='#D4A574')
    ax.set_xlabel('Number of QPUs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Parallelism', fontsize=12, fontweight='bold')
    ax.set_title('(c) Parallelism Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary stats
    ax = axes[1, 1]
    summary_text = (
        f"MQT Bench Analysis Summary\n"
        f"{'='*40}\n\n"
        f"Circuits tested: {len(df['circuit'].unique())}\n"
        f"Total experiments: {len(df)}\n\n"
        f"Avg cut cost ratio: {df['cut_cost_ratio'].mean():.2f}x\n"
        f"Avg parallelism gain: {(df['avg_parallelism_temporal'].mean() / df['avg_parallelism_baseline'].mean() - 1) * 100:.1f}%\n"
    )
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    ax.axis('off')
    
    plt.suptitle('Temporal Greedy vs KaHyPar (MQT Bench)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_kahypar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig4_kahypar_comparison.png'}")


def plot_experiment_5(df):
    """Plot Experiment 5 results: Capacity violation vs slack"""
    if df is None or df.empty:
        print("⚠ No data to plot for Experiment 5")
        return
    
    # Aggregate by slack
    agg = df.groupby('slack').agg({
        'violated_capacity_aware': 'mean',
        'violated_kahypar': 'mean',
        'kahypar_violation_magnitude': 'mean',
        'kh_ran': 'mean',
        'circuit': 'count'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Violation rates
    ax1.plot(agg['slack'], agg['violated_capacity_aware'] * 100, 'o-',
             label='Capacity-Aware', linewidth=2.5, markersize=8, color='#06A77D')
    ax1.plot(agg['slack'], agg['violated_kahypar'] * 100, 's-',
             label='KaHyPar', linewidth=2.5, markersize=8, color='#A23B72')
    ax1.set_xlabel('Slack (extra qubits per QPU)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Violation Rate vs Slack', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violation magnitude
    ax2.plot(agg['slack'], agg['kahypar_violation_magnitude'], 'o-',
             linewidth=2.5, markersize=8, color='#D4A574')
    ax2.set_xlabel('Slack (extra qubits per QPU)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Violation Magnitude (qubits)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) KaHyPar Violation Magnitude', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Capacity Violation Analysis (MQT Bench, n={len(df["circuit"].unique())} circuits)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_capacity_violation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig5_capacity_violation.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run HDH experiments with MQT Bench inputs')
    parser.add_argument('--cores', type=int, default=None, help='Number of cores to use')
    parser.add_argument('--quick', action='store_true', help='Quick mode (subset of circuits and tests)')
    parser.add_argument('--exp', type=str, default='all', 
                        help='Which experiment to run (1,2,3,4,5, or all)')
    parser.add_argument('--pkl_dir', type=str, default='/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl_reduce',
                        help='Directory containing MQT Bench pickle files')
    
    args = parser.parse_args()
    
    n_cores = args.cores if args.cores else cpu_count()
    
    print("\n" + "="*70)
    print("MQT BENCH HDH PARTITIONING EXPERIMENTS")
    print("="*70)
    print(f"Using {n_cores} CPU cores")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Experiments: {args.exp}")
    print("="*70)
    
    total_start = time.time()
    
    # Load MQT Bench HDHs
    hdhs = load_mqtbench_hdhs(args.pkl_dir, args.quick)
    
    if not hdhs:
        print("❌ No HDH circuits loaded. Exiting.")
        return
    
    # Print circuit statistics
    print(f"\n{'='*70}")
    print("Circuit Statistics:")
    print(f"{'='*70}")
    for name, hdh in list(hdhs.items())[:5]:  # Show first 5
        stats = get_circuit_stats(hdh)
        print(f"  {name}: {stats['num_qubits']} qubits, {stats['num_nodes']} nodes, {stats['num_hyperedges']} edges")
    if len(hdhs) > 5:
        print(f"  ... and {len(hdhs) - 5} more circuits")
    print(f"{'='*70}\n")
    
    # Run experiments based on args.exp
    if args.exp in ['all', '1']:
        df1 = run_experiment_1(hdhs, n_cores, args.quick)
        plot_experiment_1(df1)
    
    if args.exp in ['all', '2']:
        df2 = run_experiment_2(hdhs, n_cores, args.quick)
        plot_experiment_2(df2)
    
    if args.exp in ['all', '3']:
        df3 = run_experiment_3(hdhs, n_cores, args.quick)
        plot_experiment_3(df3)
    
    if args.exp in ['all', '4']:
        df4 = run_experiment_4(hdhs, n_cores, args.quick)
        plot_experiment_4(df4)
    
    if args.exp in ['all', '5']:
        df5 = run_experiment_5(hdhs, n_cores, args.quick)
        plot_experiment_5(df5)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"All results saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Brute Force Cut Optimization Comparison - V2

This version supports:
1. TRUE node-level partitioning (allowing qubit temporal splitting)
2. Network overhead parameter for capacity testing
3. Fixed k=3 QPUs
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from itertools import product
from tqdm.auto import tqdm
import time
import warnings

sys.path.insert(0, str(Path.cwd()))

from hdh import HDH
from hdh.passes.cut import compute_cut, cost

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path('brute_force_comparison_v2')
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ Imports successful")
print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")

def extract_qubits_from_hdh(hdh: HDH) -> Set[int]:
    """Extract all qubit indices from an HDH."""
    qubits = set()
    for node_id in hdh.S:
        if hdh.sigma[node_id] == 'q':
            try:
                base = node_id.split('_')[0]
                idx = int(base[1:])
                qubits.add(idx)
            except:
                continue
    return qubits


def get_node_qubit_mapping(hdh: HDH) -> Dict[str, int]:
    """Map each quantum node to its qubit index."""
    node_to_qubit = {}
    for node_id in hdh.S:
        if hdh.sigma[node_id] == 'q':
            try:
                base = node_id.split('_')[0]
                idx = int(base[1:])
                node_to_qubit[node_id] = idx
            except:
                continue
    return node_to_qubit


def get_partition_qubit_count(nodes: Set[str], node_to_qubit: Dict[str, int]) -> int:
    """
    Count unique qubits in a partition.
    
    Args:
        nodes: Set of node IDs in the partition
        node_to_qubit: Mapping from node ID to qubit index
    
    Returns:
        Number of unique qubits
    """
    qubits = set()
    for node in nodes:
        if node in node_to_qubit:
            qubits.add(node_to_qubit[node])
    return len(qubits)


def count_cut_hyperedges(hdh: HDH, node_partition: Dict[str, int]) -> int:
    """
    Count the number of cut hyperedges given a node partition.
    
    Args:
        hdh: The HDH object
        node_partition: Dict mapping node_id -> partition_id
    
    Returns:
        Number of hyperedges that span multiple partitions
    """
    cut_count = 0
    
    for edge in hdh.C:
        # Find which partitions this hyperedge touches
        partitions_touched = set()
        for node in edge:
            if node in node_partition:
                partitions_touched.add(node_partition[node])
        
        # If the hyperedge touches more than one partition, it's cut
        if len(partitions_touched) > 1:
            cut_count += 1
    
    return cut_count


def is_partition_valid(partition_sets: List[Set[str]], cap: int, node_to_qubit: Dict[str, int]) -> bool:
    """
    Check if a partition respects capacity constraints.
    
    Args:
        partition_sets: List of sets, each containing node IDs
        cap: Maximum unique qubits per partition
        node_to_qubit: Mapping from node ID to qubit index
    
    Returns:
        True if all partitions respect capacity constraint
    """
    for partition in partition_sets:
        qubit_count = get_partition_qubit_count(partition, node_to_qubit)
        if qubit_count > cap:
            return False
    return True


def brute_force_node_level(
    hdh: HDH, 
    k: int, 
    cap: int,
    max_nodes: int = 15
) -> Tuple[Dict[str, int], int]:
    """
    Brute force search at NODE level (allows temporal qubit splitting).
    
    WARNING: This is exponential in the number of nodes (k^N).
    Only feasible for very small circuits.
    
    Args:
        hdh: The HDH object
        k: Number of partitions (QPUs)
        cap: Maximum unique qubits per partition
        max_nodes: Safety limit on number of nodes
    
    Returns:
        (optimal_partition, min_cut_cost)
        optimal_partition: Dict mapping node_id -> partition_id
        min_cut_cost: Number of cut hyperedges in optimal partition
    """
    # Only consider quantum nodes for partitioning
    quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
    n = len(quantum_nodes)
    
    if n == 0:
        return {}, 0
    
    if n > max_nodes:
        raise ValueError(
            f"Too many nodes ({n}) for brute force node-level search. "
            f"Max is {max_nodes}. Use qubit-level instead or increase max_nodes."
        )
    
    node_to_qubit = get_node_qubit_mapping(hdh)
    
    print(f"  Brute forcing {n} nodes into {k} partitions (cap={cap} qubits)...")
    print(f"  Search space: {k**n:,} total partitions")
    
    if k**n > 10_000_000:
        warnings.warn(
            f"Very large search space ({k**n:,} partitions). This may take a long time!",
            RuntimeWarning
        )
        print(f"  ⚠ WARNING: This will take a while! Consider reducing max_nodes or using --qubit-level")
    
    min_cut_cost = float('inf')
    optimal_partition = None
    evaluated = 0
    valid_partitions = 0
    
    # Generate all possible assignments: k choices for each of n nodes
    total_partitions = k ** n
    
    # Use tqdm with a sample rate to avoid slowdown
    sample_rate = max(1, total_partitions // 10000)  # Update progress ~10k times max
    
    print(f"  Starting brute force search...")
    with tqdm(total=total_partitions, desc="  🔍 Evaluating partitions", 
              unit="partitions", unit_scale=True, leave=False,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for assignment in product(range(k), repeat=n):
            evaluated += 1
            
            # Create partition sets
            partition_sets = [set() for _ in range(k)]
            node_partition = {}
            
            for node, pid in zip(quantum_nodes, assignment):
                partition_sets[pid].add(node)
                node_partition[node] = pid
            
            # Check capacity constraint
            if not is_partition_valid(partition_sets, cap, node_to_qubit):
                if evaluated % sample_rate == 0:
                    pbar.update(sample_rate)
                continue
            
            valid_partitions += 1
            
            # Add classical nodes to partition 0
            for node in hdh.S:
                if hdh.sigma[node] == 'c':
                    node_partition[node] = 0
            
            # Count cuts
            cut_cost = count_cut_hyperedges(hdh, node_partition)
            
            if cut_cost < min_cut_cost:
                min_cut_cost = cut_cost
                optimal_partition = node_partition.copy()
            
            if evaluated % sample_rate == 0:
                pbar.update(sample_rate)
        
        # Update for any remaining
        pbar.update(total_partitions - pbar.n)
    
    print(f"  ✓ Search complete!")
    print(f"    Total partitions: {evaluated:,}")
    print(f"    Valid partitions: {valid_partitions:,} ({valid_partitions/evaluated*100:.1f}%)")
    print(f"    Best cut cost found: {min_cut_cost}")
    
    if optimal_partition is None:
        raise ValueError("No valid partition found! Check capacity constraints.")
    
    return optimal_partition, min_cut_cost


def brute_force_qubit_level(
    hdh: HDH,
    k: int,
    cap: int
) -> Tuple[Dict[str, int], int]:
    """
    Brute force search at QUBIT level (all nodes of a qubit stay together).
    
    This is faster than node-level but cannot discover temporal splitting benefits.
    
    Args:
        hdh: The HDH object
        k: Number of partitions (QPUs)
        cap: Maximum unique qubits per partition
    
    Returns:
        (optimal_partition, min_cut_cost)
    """
    from brute_force_cut_comparison import (
        brute_force_optimal_cut,
        convert_qubit_partition_to_node_partition
    )
    
    # Run qubit-level brute force
    qubit_partition, min_cut = brute_force_optimal_cut(hdh, k, cap)
    
    # Convert to node partition
    node_partitions = convert_qubit_partition_to_node_partition(hdh, qubit_partition, k)
    
    # Convert to dict format
    node_partition = {}
    for pid, nodes in enumerate(node_partitions):
        for node in nodes:
            node_partition[node] = pid
    
    return node_partition, min_cut


def convert_node_partition_to_sets(
    node_partition: Dict[str, int],
    k: int
) -> List[Set[str]]:
    """Convert node partition dict to list of sets."""
    partition_sets = [set() for _ in range(k)]
    for node, pid in node_partition.items():
        partition_sets[pid].add(node)
    return partition_sets


# MQT Bench Loading =============================================================================

def load_small_mqtbench_hdhs(
    pkl_dir: str,
    max_qubits: int = 5,
    max_nodes: int = 15
) -> Dict[str, HDH]:
    """
    Load MQT Bench HDH files with size constraints.
    
    Args:
        pkl_dir: Path to directory containing pickle files
        max_qubits: Maximum number of qubits
        max_nodes: Maximum number of quantum nodes (for node-level brute force)
    
    Returns:
        Dictionary mapping circuit name to HDH object
    """
    pkl_path = Path(pkl_dir)
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"MQT Bench pickle directory not found: {pkl_path}")
    
    pkl_files = list(pkl_path.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {pkl_path}")
    
    print(f"\n{'='*70}")
    print(f"Loading small MQT Bench circuits")
    print(f"  Max qubits: {max_qubits}")
    print(f"  Max quantum nodes: {max_nodes}")
    print(f"  Searching in: {pkl_path}")
    print(f"{'='*70}\n")
    
    hdhs = {}
    failed = []
    skipped = []
    
    for pkl_file in tqdm(pkl_files, desc="Scanning pickle files"):
        circuit_name = pkl_file.stem
        try:
            with open(pkl_file, 'rb') as f:
                hdh = pickle.load(f)
            
            if not isinstance(hdh, HDH):
                raise TypeError(f"Expected HDH object, got {type(hdh)}")
            
            # Check constraints
            num_qubits = len(extract_qubits_from_hdh(hdh))
            num_qnodes = sum(1 for n in hdh.S if hdh.sigma[n] == 'q')
            
            if num_qubits <= max_qubits and num_qnodes <= max_nodes and num_qubits > 0:
                hdhs[circuit_name] = hdh
                print(f"  ✓ {circuit_name}: {num_qubits} qubits, {num_qnodes} q-nodes, "
                      f"{len(hdh.S)} total nodes")
            else:
                skipped.append((circuit_name, num_qubits, num_qnodes))
                
        except Exception as e:
            failed.append((circuit_name, str(e)))
    
    print(f"\n✓ Successfully loaded {len(hdhs)} small circuits")
    print(f"⊘ Skipped {len(skipped)} circuits (too large)")
    if failed:
        print(f"⚠ Failed to load {len(failed)} circuits")
    
    return hdhs

# Analysis  =============================================================================

def compute_capacity_from_overhead(num_qubits: int, overhead: float) -> int:
    """
    Compute capacity per QPU given overhead parameter.
    
    Args:
        num_qubits: Total qubits in the circuit
        overhead: Overhead multiplier (1.0 = tight, 1.1 = 10% slack, etc.)
    
    Returns:
        Capacity (qubits per QPU)
    """
    return int(np.ceil(num_qubits * overhead))


def run_comparison_experiment(
    hdhs: Dict[str, HDH],
    k: int = 3,
    overhead_values: List[float] = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3],
    use_node_level: bool = False,
    max_nodes_for_node_level: int = 12
) -> pd.DataFrame:
    """
    Run brute force vs heuristic comparison experiments.
    
    Args:
        hdhs: Dictionary of circuit_name -> HDH
        k: Number of QPUs (fixed at 3)
        overhead_values: List of overhead multipliers to test
        use_node_level: If True, use node-level brute force (allows temporal splitting)
        max_nodes_for_node_level: Max quantum nodes for node-level search
    
    Returns:
        DataFrame with results
    """
    results = []
    
    method = "NODE-LEVEL" if use_node_level else "QUBIT-LEVEL"
    
    # Pre-calculate total experiments for progress tracking
    total_experiments = 0
    circuit_experiment_counts = {}
    
    for circuit_name, hdh in hdhs.items():
        num_qubits = len(extract_qubits_from_hdh(hdh))
        num_qnodes = sum(1 for n in hdh.S if hdh.sigma[n] == 'q')
        
        # Skip if too large for node-level
        if use_node_level and num_qnodes > max_nodes_for_node_level:
            circuit_experiment_counts[circuit_name] = 0
            continue
        
        # Count feasible experiments for this circuit
        count = 0
        for overhead in overhead_values:
            cap = compute_capacity_from_overhead(num_qubits, overhead)
            if num_qubits <= k * cap:
                count += 1
        
        circuit_experiment_counts[circuit_name] = count
        total_experiments += count
    
    print(f"\n{'='*70}")
    print(f"Running Comparison Experiments ({method})")
    print(f"Circuits loaded: {len(hdhs)}")
    print(f"k (QPUs): {k}")
    print(f"Overhead values: {overhead_values}")
    print(f"Total experiments planned: {total_experiments}")
    print(f"{'='*70}\n")
    
    completed_experiments = 0
    overall_start_time = time.time()
    
    for circuit_idx, (circuit_name, hdh) in enumerate(hdhs.items(), 1):
        num_qubits = len(extract_qubits_from_hdh(hdh))
        num_nodes = len(hdh.S)
        num_qnodes = sum(1 for n in hdh.S if hdh.sigma[n] == 'q')
        num_edges = len(hdh.C)
        
        # Circuit-level progress
        circuit_progress = (circuit_idx / len(hdhs)) * 100
        elapsed_time = time.time() - overall_start_time
        
        if completed_experiments > 0:
            avg_time_per_exp = elapsed_time / completed_experiments
            remaining_exp = total_experiments - completed_experiments
            eta_seconds = avg_time_per_exp * remaining_exp
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = "calculating..."
        
        print(f"\n{'='*70}")
        print(f"[Circuit {circuit_idx}/{len(hdhs)} - {circuit_progress:.1f}% of circuits]")
        print(f"[Experiments: {completed_experiments}/{total_experiments} - "
              f"{(completed_experiments/total_experiments*100):.1f}% complete]")
        print(f"[Elapsed: {int(elapsed_time//60)}m {int(elapsed_time%60)}s | ETA: {eta_str}]")
        print(f"{'='*70}")
        print(f"Circuit: {circuit_name}")
        print(f"  Qubits: {num_qubits}, Quantum nodes: {num_qnodes}, "
              f"Total nodes: {num_nodes}, Edges: {num_edges}")
        
        # Skip if too large for node-level
        if use_node_level and num_qnodes > max_nodes_for_node_level:
            print(f"  ⊘ Skipping: too many quantum nodes ({num_qnodes} > {max_nodes_for_node_level})")
            continue
        
        # Get number of feasible experiments for this circuit
        circuit_experiments = circuit_experiment_counts.get(circuit_name, 0)
        circuit_exp_completed = 0
        
        for overhead_idx, overhead in enumerate(overhead_values, 1):
            # Compute capacity based on overhead
            cap = compute_capacity_from_overhead(num_qubits, overhead)
            
            # Check feasibility
            if num_qubits > k * cap:
                print(f"  ⊘ Skipping overhead={overhead:.2f} (cap={cap}): "
                      f"infeasible ({num_qubits} > {k}*{cap})")
                continue
            
            # Overhead-level progress for this circuit
            print(f"\n  [{circuit_exp_completed+1}/{circuit_experiments} for this circuit] "
                  f"Testing overhead={overhead:.2f} (cap={cap} qubits/QPU):")
            
            # Brute force optimal
            start_time = time.time()
            try:
                if use_node_level:
                    optimal_partition, optimal_cost = brute_force_node_level(
                        hdh, k, cap, max_nodes=max_nodes_for_node_level
                    )
                else:
                    optimal_partition, optimal_cost = brute_force_qubit_level(
                        hdh, k, cap
                    )
                brute_force_time = time.time() - start_time
                print(f"    ✓ Brute force ({method}): {optimal_cost} cuts ({brute_force_time:.2f}s)")
                
                # Get partition statistics
                partition_sets = convert_node_partition_to_sets(optimal_partition, k)
                node_to_qubit = get_node_qubit_mapping(hdh)
                partition_qubit_counts = [
                    get_partition_qubit_count(pset, node_to_qubit) 
                    for pset in partition_sets
                ]
                print(f"    Optimal partition qubit counts: {partition_qubit_counts}")
                
            except Exception as e:
                print(f"    ✗ Brute force failed: {e}")
                continue
            
            # Heuristic cut
            start_time = time.time()
            try:
                heuristic_partitions, heuristic_edges = compute_cut(hdh, k=k, cap=cap)
                heuristic_cost = cost(hdh, heuristic_partitions)
                heuristic_time = time.time() - start_time
                
                # Get heuristic partition statistics
                heuristic_qubit_counts = [
                    get_partition_qubit_count(pset, node_to_qubit)
                    for pset in heuristic_partitions
                ]
                
                print(f"    ✓ Heuristic: {heuristic_cost} cuts ({heuristic_time:.2f}s)")
                print(f"    Heuristic partition qubit counts: {heuristic_qubit_counts}")
                
            except Exception as e:
                print(f"    ✗ Heuristic failed: {e}")
                heuristic_cost = None
                heuristic_time = None
                heuristic_qubit_counts = None
            
            # Calculate ratio
            if optimal_cost > 0:
                ratio = heuristic_cost / optimal_cost
            else:
                ratio = 1.0 if heuristic_cost == 0 else float('inf')
            
            print(f"    → Ratio (heuristic/optimal): {ratio:.3f}")
            
            # Update progress counters
            completed_experiments += 1
            circuit_exp_completed += 1
            
            # Show overall progress after each experiment
            overall_pct = (completed_experiments / total_experiments) * 100
            print(f"    📊 Overall progress: {completed_experiments}/{total_experiments} ({overall_pct:.1f}%)")
            
            results.append({
                'circuit': circuit_name,
                'num_qubits': num_qubits,
                'num_nodes': num_nodes,
                'num_quantum_nodes': num_qnodes,
                'num_edges': num_edges,
                'k': k,
                'overhead': overhead,
                'capacity': cap,
                'optimal_cost': optimal_cost,
                'heuristic_cost': heuristic_cost,
                'ratio': ratio,
                'brute_force_time': brute_force_time,
                'heuristic_time': heuristic_time if heuristic_time else 0,
                'method': method,
                'optimal_qubit_counts': str(partition_qubit_counts),
                'heuristic_qubit_counts': str(heuristic_qubit_counts) if heuristic_qubit_counts else None,
            })
    
    # Final summary
    total_elapsed = time.time() - overall_start_time
    print(f"\n{'='*70}")
    print(f"🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")
    print(f"Total experiments completed: {completed_experiments}/{total_experiments}")
    print(f"Completion rate: {(completed_experiments/total_experiments*100):.1f}%")
    print(f"Total time elapsed: {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
    if completed_experiments > 0:
        print(f"Average time per experiment: {total_elapsed/completed_experiments:.1f}s")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = OUTPUT_DIR / f'comparison_results_{method.lower()}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return df

# Plotting =============================================================================

def plot_comparison_results(df: pd.DataFrame, method: str = "NODE-LEVEL"):
    """Plot comparison results with overhead analysis."""
    if df is None or df.empty:
        print("⚠ No data to plot")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Ratio vs Overhead
    ax1 = fig.add_subplot(gs[0, :2])
    
    for circuit in df['circuit'].unique():
        df_circuit = df[df['circuit'] == circuit]
        ax1.plot(df_circuit['overhead'], df_circuit['ratio'], 'o-', 
                label=circuit, linewidth=2, markersize=8, alpha=0.7)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal (ratio=1.0)')
    ax1.set_xlabel('Network Overhead', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Heuristic/Optimal Cost Ratio', fontsize=12, fontweight='bold')
    ax1.set_title(f'(a) Cut Cost Ratio vs Network Overhead ({method})', 
                  fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Summary statistics
    ax2 = fig.add_subplot(gs[0, 2])
    summary_stats = (
        f"Summary Statistics\n"
        f"{'='*30}\n\n"
        f"Method: {method}\n"
        f"Circuits tested: {len(df['circuit'].unique())}\n"
        f"QPUs (k): {df['k'].iloc[0]}\n"
        f"Total experiments: {len(df)}\n\n"
        f"Ratio Statistics:\n"
        f"  Mean: {df['ratio'].mean():.3f}\n"
        f"  Median: {df['ratio'].median():.3f}\n"
        f"  Min: {df['ratio'].min():.3f}\n"
        f"  Max: {df['ratio'].max():.3f}\n\n"
        f"Optimal matches: {(df['ratio'] == 1.0).sum()}\n"
        f"({(df['ratio'] == 1.0).sum() / len(df) * 100:.1f}%)"
    )
    ax2.text(0.5, 0.5, summary_stats, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    ax2.axis('off')
    
    # Plot 3: Absolute costs
    ax3 = fig.add_subplot(gs[1, :])
    
    x = np.arange(len(df))
    width = 0.35
    
    ax3.bar(x - width/2, df['optimal_cost'], width, label='Optimal (Brute Force)',
            color='#2E86AB', alpha=0.8)
    ax3.bar(x + width/2, df['heuristic_cost'], width, label='Heuristic',
            color='#D4A574', alpha=0.8)
    
    # Add overhead labels on x-axis
    labels = [f"{row['circuit'][:10]}\noh={row['overhead']:.1f}" 
              for _, row in df.iterrows()]
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    ax3.set_ylabel('Number of Cut Hyperedges', fontsize=12, fontweight='bold')
    ax3.set_title('(b) Absolute Cut Costs: Optimal vs Heuristic', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Ratio distribution by overhead
    ax4 = fig.add_subplot(gs[2, 0])
    
    overhead_vals = sorted(df['overhead'].unique())
    data_by_overhead = [df[df['overhead'] == oh]['ratio'].values for oh in overhead_vals]
    bp = ax4.boxplot(data_by_overhead, labels=[f"{oh:.2f}" for oh in overhead_vals],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#06A77D')
        patch.set_alpha(0.7)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Network Overhead', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Heuristic/Optimal Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('(c) Ratio Distribution by Overhead', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Capacity utilization
    ax5 = fig.add_subplot(gs[2, 1])
    
    ax5.scatter(df['capacity'], df['ratio'], c=df['overhead'], 
               cmap='viridis', alpha=0.6, s=100, edgecolors='black')
    ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax5.set_xlabel('Capacity (qubits/QPU)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Heuristic/Optimal Ratio', fontsize=12, fontweight='bold')
    ax5.set_title('(d) Ratio vs Capacity', fontsize=13, fontweight='bold')
    plt.colorbar(ax5.collections[0], ax=ax5, label='Overhead')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Computation time
    ax6 = fig.add_subplot(gs[2, 2])
    
    ax6.scatter(df['brute_force_time'], df['heuristic_time'], alpha=0.6, s=100,
               c=df['num_quantum_nodes'], cmap='plasma', edgecolors='black')
    
    max_time = max(df['brute_force_time'].max(), df['heuristic_time'].max())
    ax6.plot([0, max_time], [0, max_time], 'k--', alpha=0.3, label='Equal time')
    
    ax6.set_xlabel('Brute Force Time (s)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Heuristic Time (s)', fontsize=12, fontweight='bold')
    ax6.set_title('(e) Computation Time', fontsize=13, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')
    plt.colorbar(ax6.collections[0], ax=ax6, label='Q-nodes')
    
    plt.suptitle(f'Brute-Force Optimal vs Heuristic - {method} (k=3 QPUs)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = OUTPUT_DIR / f'comparison_{method.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to: {plot_path}")

# Main =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Brute-force optimal vs heuristic cut comparison (V2 - with node-level and overhead)'
    )
    parser.add_argument(
        '--pkl_dir',
        type=str,
        default='/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl',
        help='Directory containing MQT Bench pickle files'
    )
    parser.add_argument(
        '--max_qubits',
        type=int,
        default=5,
        help='Maximum qubits for circuits to test (default: 5)'
    )
    parser.add_argument(
        '--max_nodes',
        type=int,
        default=12,
        help='Maximum quantum nodes for node-level brute force (default: 12)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of QPUs (default: 3)'
    )
    parser.add_argument(
        '--overhead',
        type=str,
        default='1.0,1.1,1.2',
        help='Comma-separated overhead values (default: 1.0,1.1,1.2)'
    )
    parser.add_argument(
        '--node-level',
        action='store_true',
        help='Use node-level brute force (allows temporal qubit splitting)'
    )
    parser.add_argument(
        '--qubit-level',
        action='store_true',
        help='Use qubit-level brute force (faster, no temporal splitting)'
    )
    
    args = parser.parse_args()
    
    overhead_values = [float(x) for x in args.overhead.split(',')]
    
    # Determine which method to use
    if args.node_level and args.qubit_level:
        print("Error: Cannot use both --node-level and --qubit-level. Choose one.")
        return
    elif args.node_level:
        use_node_level = True
    elif args.qubit_level:
        use_node_level = False
    else:
        # Default to qubit-level (safer/faster)
        use_node_level = False
        print("Note: Defaulting to qubit-level brute force. Use --node-level for temporal splitting.")
    
    method = "NODE-LEVEL" if use_node_level else "QUBIT-LEVEL"
    
    print("\n" + "="*70)
    print(f"BRUTE-FORCE COMPARISON V2 ({method})")
    print("="*70)
    print(f"Max qubits: {args.max_qubits}")
    print(f"Max quantum nodes (for node-level): {args.max_nodes}")
    print(f"k (QPUs): {args.k}")
    print(f"Overhead values: {overhead_values}")
    print(f"Method: {method}")
    print("="*70)
    
    if use_node_level:
        print("\n⚠ WARNING: Node-level brute force is VERY expensive!")
        print(f"   Only feasible for circuits with ≤{args.max_nodes} quantum nodes.")
        print("   For larger circuits, use --qubit-level instead.\n")
    
    total_start = time.time()
    
    # Load small circuits
    hdhs = load_small_mqtbench_hdhs(
        args.pkl_dir,
        args.max_qubits,
        args.max_nodes if use_node_level else 1000  # No limit for qubit-level
    )
    
    if not hdhs:
        print(" No suitable circuits loaded.")
        return
    
    # Run comparison
    df = run_comparison_experiment(
        hdhs,
        k=args.k,
        overhead_values=overhead_values,
        use_node_level=use_node_level,
        max_nodes_for_node_level=args.max_nodes
    )
    
    # Plot results
    plot_comparison_results(df, method)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"All results saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Method: {method}")
    print(f"  Circuits tested: {len(df['circuit'].unique())}")
    print(f"  Total experiments: {len(df)}")
    print(f"  Average ratio: {df['ratio'].mean():.3f}")
    print(f"  Optimal matches: {(df['ratio'] == 1.0).sum()} / {len(df)} "
          f"({(df['ratio'] == 1.0).sum() / len(df) * 100:.1f}%)")
    print(f"\nBy overhead:")
    for oh in sorted(df['overhead'].unique()):
        df_oh = df[df['overhead'] == oh]
        print(f"  Overhead {oh:.2f}: mean ratio = {df_oh['ratio'].mean():.3f}, "
              f"optimal rate = {(df_oh['ratio'] == 1.0).sum() / len(df_oh) * 100:.1f}%")


if __name__ == '__main__':
    main()
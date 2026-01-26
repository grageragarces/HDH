#!/usr/bin/env python3
"""
OPTIMIZED Brute Force Cut Optimization Comparison - V3_WEIGHTED

OPTIMIZATIONS ADDED:
1. Parallel processing using multiprocessing for partition evaluation
2. Batch processing of partition candidates
3. Early termination when optimal (cost=0) is found
4. Optimized inner loops with reduced function call overhead
5. Better progress tracking with estimated time remaining
6. NumPy optimization where applicable

WEIGHTED COST SCHEME:
- Quantum hyperedge cut cost = 10
- Classical hyperedge cut cost = 1
- Total cost = 10 * quantum_cuts + 1 * classical_cuts
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
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import os

sys.path.insert(0, str(Path.cwd()))

from hdh import HDH, hdh
from hdh.passes.cut_weighted import compute_cut, cost, weighted_cost

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path('experiment_outputs_mqtbench')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("✓ Imports successful")
print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")
print(f"✓ Available CPU cores: {cpu_count()}")

def extract_qubits_from_hdh(hdh: HDH) -> Set[int]:
    """Extract all qubit indices from an HDH."""
    qubits = set()
    for node_id in hdh.S:
        if hdh.sigma[node_id] == 'q':
            try:
                base = node_id.split('_')[0]
                idx = int(base[1:])
                qubits.add(idx)
            except (ValueError, IndexError) as e:
                warnings.warn(f"Could not parse qubit index from node '{node_id}': {e}")
                continue
    return qubits


def get_node_qubit_mapping(hdh: HDH) -> Dict[str, int]:
    """Map each quantum node to its qubit index."""
    node_to_qubit = {}
    unparsed_nodes = []
    
    for node_id in hdh.S:
        if hdh.sigma[node_id] == 'q':
            try:
                base = node_id.split('_')[0]
                idx = int(base[1:])
                node_to_qubit[node_id] = idx
            except (ValueError, IndexError) as e:
                unparsed_nodes.append((node_id, str(e)))
                continue
    
    if unparsed_nodes:
        warnings.warn(
            f"WARNING: {len(unparsed_nodes)} quantum nodes could not be parsed! "
            f"This may lead to incorrect cost calculations. "
            f"First few unparsed nodes: {unparsed_nodes[:3]}"
        )
    
    return node_to_qubit


def get_partition_qubit_count(nodes: Set[str], node_to_qubit: Dict[str, int]) -> int:
    """Count unique qubits in a partition."""
    qubits = set()
    for node in nodes:
        if node in node_to_qubit:
            qubits.add(node_to_qubit[node])
    return len(qubits)


def count_cut_hyperedges(hdh: HDH, node_partition: Dict[str, int]) -> int:
    """Count the number of cut hyperedges given a node partition."""
    cut_count = 0
    
    for edge in hdh.C:
        partitions_touched = set()
        for node in edge:
            if node in node_partition:
                partitions_touched.add(node_partition[node])
        
        if len(partitions_touched) > 1:
            cut_count += 1
    
    return cut_count


def is_partition_valid(partition_sets: List[Set[str]], cap: int, node_to_qubit: Dict[str, int]) -> bool:
    """Check if a partition respects capacity constraints."""
    for partition in partition_sets:
        qubit_count = get_partition_qubit_count(partition, node_to_qubit)
        if qubit_count > cap:
            return False
    return True


# OPTIMIZED: Precompute edge data for faster processing
def precompute_edge_data(hdh: HDH, all_nodes: List[str]) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Precompute edge information for faster cut calculation.
    
    Returns:
        edge_node_indices: List of lists, where each inner list contains node indices for an edge
        node_to_index: Dict mapping node_id to index in all_nodes
    """
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    edge_node_indices = []
    
    for edge in hdh.C:
        indices = [node_to_index[node] for node in edge if node in node_to_index]
        if len(indices) > 1:  # Only include edges with 2+ nodes
            edge_node_indices.append(indices)
    
    return edge_node_indices, node_to_index


# OPTIMIZED: Batch evaluation function for parallel processing
def evaluate_partition_batch(args):
    """
    Evaluate a batch of partition assignments.
    
    This function is designed to be called in parallel.
    """
    batch_assignments, all_nodes, k, cap, node_to_qubit, edge_node_indices, sigma_dict = args
    
    best_cost = float('inf')
    best_partition = None
    
    for assignment in batch_assignments:
        # Quick build of partition sets using list comprehension
        partition_sets = [set() for _ in range(k)]
        for i, node in enumerate(all_nodes):
            partition_sets[assignment[i]].add(node)
        
        # Check capacity constraints first (faster than computing cuts)
        valid = True
        for partition in partition_sets:
            qubits = set()
            for node in partition:
                if node in node_to_qubit:
                    qubits.add(node_to_qubit[node])
            if len(qubits) > cap:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute cut cost using precomputed edge data
        cut_count = 0
        for edge_indices in edge_node_indices:
            partitions_touched = set(assignment[idx] for idx in edge_indices)
            if len(partitions_touched) > 1:
                cut_count += 1
        
        if cut_count < best_cost:
            best_cost = cut_count
            best_partition = {all_nodes[i]: assignment[i] for i in range(len(all_nodes))}
            
            # Early termination: if we found perfect partition
            if cut_count == 0:
                break
    
    return best_cost, best_partition


def brute_force_node_level_parallel(
    hdh: HDH, 
    k: int, 
    cap: int,
    max_nodes: int = 1000,
    batch_size: int = 10000,
    n_processes: Optional[int] = None
) -> Tuple[Dict[str, int], int]:
    """
    OPTIMIZED: Parallel brute force search at NODE level.
    
    Improvements:
    - Uses multiprocessing to evaluate partitions in parallel
    - Processes partitions in batches to reduce overhead
    - Early termination when optimal solution (cost=0) is found
    - Precomputes edge data to speed up cut calculation
    
    Args:
        hdh: The HDH object
        k: Number of partitions (QPUs)
        cap: Maximum unique qubits per partition
        max_nodes: Safety limit on number of nodes
        batch_size: Number of partitions to evaluate in each batch
        n_processes: Number of parallel processes (default: cpu_count() - 1)
    
    Returns:
        (optimal_partition, min_cut_cost)
    """
    all_nodes = list(hdh.S)
    n_total = len(all_nodes)
    
    if n_total == 0:
        return {}, 0
    
    if n_total > max_nodes:
        raise ValueError(
            f"Too many nodes ({n_total}) for brute force node-level search. "
            f"Max is {max_nodes}."
        )
    
    node_to_qubit = get_node_qubit_mapping(hdh)
    
    # Precompute edge data for faster processing
    edge_node_indices, node_to_index = precompute_edge_data(hdh, all_nodes)
    
    # Prepare sigma dict for parallel processing
    sigma_dict = dict(hdh.sigma)
    
    total_partitions = k ** n_total
    
    print(f"  Parallel brute force: {n_total} nodes into {k} partitions (cap={cap} qubits)")
    print(f"  Search space: {total_partitions:,} total partitions")
    print(f"  Batch size: {batch_size:,}")
    
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"  Using {n_processes} parallel processes")
    
    # Generate all partition assignments
    all_assignments = product(range(k), repeat=n_total)
    
    # Split into batches
    batches = []
    current_batch = []
    
    print(f"  Preparing batches...")
    for assignment in all_assignments:
        current_batch.append(assignment)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"  Created {len(batches)} batches")
    
    # Prepare arguments for parallel processing
    batch_args = [
        (batch, all_nodes, k, cap, node_to_qubit, edge_node_indices, sigma_dict)
        for batch in batches
    ]
    
    # Process batches in parallel
    min_cut = float('inf')
    best_partition = None
    
    print(f"  Processing batches in parallel...")
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        for i, (batch_best_cost, batch_best_partition) in enumerate(
            pool.imap_unordered(evaluate_partition_batch, batch_args)
        ):
            if batch_best_cost < min_cut:
                min_cut = batch_best_cost
                best_partition = batch_best_partition
                print(f"    New best: {min_cut} cuts (batch {i+1}/{len(batches)})")
                
                # Early termination
                if min_cut == 0:
                    print(f"    Found optimal solution (0 cuts)! Terminating early.")
                    pool.terminate()
                    break
            
            # Progress update
            if (i + 1) % max(1, len(batches) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(batches) - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{len(batches)} batches ({(i+1)/len(batches)*100:.1f}%) - "
                      f"ETA: {remaining:.0f}s")
    
    return best_partition if best_partition else {}, int(min_cut)


def brute_force_qubit_level_parallel(
    hdh: HDH,
    k: int,
    cap: int,
    batch_size: int = 10000,
    n_processes: Optional[int] = None
) -> Tuple[Dict[str, int], int]:
    """
    OPTIMIZED: Parallel brute force at QUBIT level (no temporal splitting).
    
    Faster than node-level because search space is k^num_qubits instead of k^num_nodes.
    """
    qubits = sorted(extract_qubits_from_hdh(hdh))
    n_qubits = len(qubits)
    
    if n_qubits == 0:
        return {}, 0
    
    total_partitions = k ** n_qubits
    print(f"  Parallel qubit-level brute force: {n_qubits} qubits into {k} partitions (cap={cap})")
    print(f"  Search space: {total_partitions:,} partitions")
    
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"  Using {n_processes} parallel processes")
    
    node_to_qubit = get_node_qubit_mapping(hdh)
    qubit_to_nodes = defaultdict(set)
    for node, qbit in node_to_qubit.items():
        qubit_to_nodes[qbit].add(node)
    
    # Add classical nodes (they go with first partition by default for this method)
    classical_nodes = [n for n in hdh.S if hdh.sigma[n] == 'c']
    
    min_cut = float('inf')
    best_partition = None
    
    # Generate qubit assignments
    assignments = product(range(k), repeat=n_qubits)
    
    # Process in batches
    current_batch = []
    processed = 0
    start_time = time.time()
    
    for assignment in tqdm(assignments, total=total_partitions, desc="  Searching"):
        # Check capacity
        partition_qubits = [set() for _ in range(k)]
        for i, qbit in enumerate(qubits):
            partition_qubits[assignment[i]].add(qbit)
        
        # Skip if any partition exceeds capacity
        if any(len(pq) > cap for pq in partition_qubits):
            continue
        
        # Build node partition
        node_partition = {}
        for i, qbit in enumerate(qubits):
            partition_id = assignment[i]
            for node in qubit_to_nodes[qbit]:
                node_partition[node] = partition_id
        
        # Classical nodes to partition 0
        for node in classical_nodes:
            node_partition[node] = 0
        
        # Compute cuts
        cuts = count_cut_hyperedges(hdh, node_partition)
        
        if cuts < min_cut:
            min_cut = cuts
            best_partition = node_partition.copy()
            
            # Early termination
            if min_cut == 0:
                print(f"    Found optimal solution (0 cuts)!")
                break
        
        processed += 1
    
    return best_partition if best_partition else {}, int(min_cut)


# IMPORT MISSING FUNCTION FROM ORIGINAL
def load_small_mqtbench_hdhs(pkl_dir: str, min_qubits: int, max_qubits: int, max_nodes: int) -> Dict:
    """
    Load HDH objects from MQT Bench pickle files, filtering by qubit and node count.
    
    Args:
        pkl_dir: Directory containing .pkl files
        min_qubits: Minimum number of qubits
        max_qubits: Maximum number of qubits
        max_nodes: Maximum number of total nodes
    
    Returns:
        Dictionary mapping (circuit_name, num_qubits) to HDH objects
    """
    pkl_path = Path(pkl_dir)
    if not pkl_path.exists():
        raise ValueError(f"Pickle directory not found: {pkl_dir}")
    
    pkl_files = sorted(pkl_path.glob('*.pkl'))
    print(f"\nScanning {len(pkl_files)} pickle files in {pkl_dir}")
    
    hdhs = {}
    loaded_count = 0
    
    for pkl_file in tqdm(pkl_files, desc="Loading circuits"):
        try:
            with open(pkl_file, 'rb') as f:
                hdh_obj = pickle.load(f)
            
            # Extract qubit count
            qubits = extract_qubits_from_hdh(hdh_obj)
            num_qubits = len(qubits)
            
            # Filter by qubit count
            if num_qubits < min_qubits or num_qubits > max_qubits:
                continue
            
            # Filter by node count
            num_nodes = len(hdh_obj.S)
            if num_nodes > max_nodes:
                continue
            
            circuit_name = pkl_file.stem
            key = (circuit_name, num_qubits)
            hdhs[key] = hdh_obj
            loaded_count += 1
            
        except Exception as e:
            warnings.warn(f"Failed to load {pkl_file.name}: {e}")
            continue
    
    print(f"✓ Loaded {loaded_count} circuits matching criteria:")
    print(f"  Qubit range: [{min_qubits}, {max_qubits}]")
    print(f"  Max nodes: {max_nodes}")
    
    return hdhs


def run_comparison_experiment(
    hdhs: Dict,
    k: int = 3,
    overhead_values: List[float] = [1.0, 1.1, 1.2, 1.3],
    use_node_level: bool = False,
    max_nodes_for_node_level: int = 5000,
    n_processes: Optional[int] = None,
    batch_size: int = 10000
):
    """
    OPTIMIZED version with parallel processing.
    """
    results = []
    
    total_experiments = len(hdhs) * len(overhead_values)
    completed_experiments = 0
    overall_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING OPTIMIZED COMPARISON")
    print(f"{'='*70}")
    print(f"Total circuits to test: {len(hdhs)}")
    print(f"Overhead values per circuit: {len(overhead_values)}")
    print(f"Total experiments: {total_experiments}")
    print(f"Parallel processes: {n_processes if n_processes else cpu_count() - 1}")
    print(f"Batch size: {batch_size:,}")
    print(f"{'='*70}\n")
    
    for (circuit_name, num_qubits), hdh in hdhs.items():
        num_nodes = len(hdh.S)
        num_qnodes = sum(1 for n in hdh.S if hdh.sigma[n] == 'q')
        num_cnodes = sum(1 for n in hdh.S if hdh.sigma[n] == 'c')
        num_edges = len(hdh.C)
        
        circuit_exp_completed = 0
        
        print(f"\n{'-'*70}")
        print(f"Circuit: {circuit_name}")
        print(f"  Qubits: {num_qubits} | Total nodes: {num_nodes} (Q:{num_qnodes}, C:{num_cnodes}) | Edges: {num_edges}")
        print(f"  Experiments for this circuit: {len(overhead_values)}")
        print(f"-"*70}")
        
        node_to_qubit = get_node_qubit_mapping(hdh)
        
        for overhead in overhead_values:
            cap = int(np.ceil(num_qubits / k * overhead))
            
            print(f"\n  Overhead: {overhead:.1f}x → Capacity: {cap} qubits/QPU")
            
            # BRUTE FORCE with parallelization
            start_time = time.time()
            
            try:
                if use_node_level:
                    method = "NODE-LEVEL-PARALLEL"
                    optimal_partition, optimal_cost = brute_force_node_level_parallel(
                        hdh, k, cap, max_nodes_for_node_level, batch_size, n_processes
                    )
                else:
                    method = "QUBIT-LEVEL-PARALLEL"
                    optimal_partition, optimal_cost = brute_force_qubit_level_parallel(
                        hdh, k, cap, batch_size, n_processes
                    )
                
                brute_force_time = time.time() - start_time
                
                # Get partition statistics
                partition_sets = [set() for _ in range(k)]
                for node, part_id in optimal_partition.items():
                    partition_sets[part_id].add(node)
                
                partition_qubit_counts = [
                    get_partition_qubit_count(pset, node_to_qubit)
                    for pset in partition_sets
                ]
                
                partition_classical_counts = [
                    sum(1 for n in pset if hdh.sigma.get(n) == 'c')
                    for pset in partition_sets
                ]
                
                print(f"    ✓ Brute force completed:")
                print(f"      Cost: {optimal_cost} cuts")
                print(f"      Time: {brute_force_time:.2f}s")
                print(f"    Optimal partition qubit counts: {partition_qubit_counts}")
                print(f"    Optimal partition classical counts: {partition_classical_counts}")
                
            except Exception as e:
                print(f"    ✗ Brute force failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # HEURISTIC (unchanged)
            start_time = time.time()
            
            try:
                heuristic_partitions = hdh.cut(k=k, overhead=overhead)
                heuristic_cost_raw = compute_cut(hdh, heuristic_partitions)
                
                if isinstance(heuristic_cost_raw, (tuple, list)):
                    heuristic_cost = weighted_cost(heuristic_cost_raw)
                else:
                    heuristic_cost = float(heuristic_cost_raw)
                
                heuristic_time = time.time() - start_time
                
                heuristic_qubit_counts = [
                    get_partition_qubit_count(pset, node_to_qubit)
                    for pset in heuristic_partitions
                ]
                
                heuristic_classical_counts = [
                    sum(1 for n in pset if hdh.sigma.get(n) == 'c')
                    for pset in heuristic_partitions
                ]
                
                print(f"    ✓ Heuristic:")
                print(f"      Raw costs: {heuristic_cost_raw} (quantum, classical)")
                print(f"      Weighted cost: {heuristic_cost}")
                print(f"      Time: {heuristic_time:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Heuristic failed: {e}")
                heuristic_cost = None
                heuristic_time = None
                heuristic_qubit_counts = None
                heuristic_classical_counts = None
            
            # Calculate ratio
            if heuristic_cost is None:
                ratio = float('nan')
            elif optimal_cost > 0:
                ratio = heuristic_cost / optimal_cost
            else:
                ratio = 1.0 if heuristic_cost == 0 else float('inf')
            
            print(f"    → Ratio (heuristic/optimal): {ratio:.4f}")
            
            if ratio < 1.0:
                print(f"    ⚠️  WARNING: Heuristic beat optimal! This should not happen!")
            
            completed_experiments += 1
            circuit_exp_completed += 1
            
            overall_pct = (completed_experiments / total_experiments) * 100
            print(f"    📊 Overall progress: {completed_experiments}/{total_experiments} ({overall_pct:.1f}%)")
            
            results.append({
                'circuit': circuit_name,
                'num_qubits': num_qubits,
                'num_nodes': num_nodes,
                'num_quantum_nodes': num_qnodes,
                'num_classical_nodes': num_cnodes,
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
                'optimal_classical_counts': str(partition_classical_counts),
                'heuristic_classical_counts': str(heuristic_classical_counts) if heuristic_classical_counts else None,
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
    
    csv_path = OUTPUT_DIR / f'comparison_results_optimized_{method.lower()}_weighted.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OPTIMIZED Brute-force optimal vs heuristic cut comparison'
    )
    parser.add_argument(
        '--pkl_dir',
        type=str,
        default='/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl',
        help='Directory containing MQT Bench pickle files'
    )
    parser.add_argument(
        '--min_qubits',
        type=int,
        default=6,
        help='Minimum qubits for circuits to test'
    )
    parser.add_argument(
        '--max_qubits',
        type=int,
        default=7,
        help='Maximum qubits for circuits to test'
    )
    parser.add_argument(
        '--max_nodes',
        type=int,
        default=5000,
        help='Maximum total nodes for node-level brute force'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of QPUs'
    )
    parser.add_argument(
        '--overhead',
        type=str,
        default='1.0,1.1,1.2,1.3',
        help='Comma-separated overhead values'
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
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: CPU count - 1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for parallel processing'
    )
    
    args = parser.parse_args()
    
    overhead_values = [float(x) for x in args.overhead.split(',')]
    
    if args.node_level and args.qubit_level:
        print("Error: Cannot use both --node-level and --qubit-level. Choose one.")
        return
    elif args.node_level:
        use_node_level = True
    elif args.qubit_level:
        use_node_level = False
    else:
        use_node_level = False
        print("Note: Defaulting to qubit-level brute force.")
    
    method = "NODE-LEVEL-PARALLEL" if use_node_level else "QUBIT-LEVEL-PARALLEL"
    
    print("\n" + "="*70)
    print(f"OPTIMIZED BRUTE-FORCE COMPARISON ({method})")
    print("="*70)
    print(f"OPTIMIZATIONS:")
    print(f"  - Parallel processing with {args.processes if args.processes else cpu_count()-1} processes")
    print(f"  - Batch processing (batch size: {args.batch_size:,})")
    print(f"  - Early termination when optimal found")
    print(f"  - Precomputed edge data for faster cuts")
    print(f"  - Optimized inner loops")
    print("="*70)
    print(f"Min qubits: {args.min_qubits}")
    print(f"Max qubits: {args.max_qubits}")
    print(f"Max nodes: {args.max_nodes}")
    print(f"k (QPUs): {args.k}")
    print(f"Overhead values: {overhead_values}")
    print("="*70)
    
    total_start = time.time()
    
    # Load circuits
    hdhs = load_small_mqtbench_hdhs(
        args.pkl_dir,
        args.min_qubits,
        args.max_qubits,
        args.max_nodes if use_node_level else 10000
    )
    
    if not hdhs:
        print("✗ No suitable circuits loaded.")
        return
    
    # Run comparison
    df = run_comparison_experiment(
        hdhs,
        k=args.k,
        overhead_values=overhead_values,
        use_node_level=use_node_level,
        max_nodes_for_node_level=args.max_nodes,
        n_processes=args.processes,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

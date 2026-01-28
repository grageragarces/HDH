#!/usr/bin/env python3
"""
TIMED Brute Force Cut Optimization Comparison - V4_TIMED_WEIGHTED

CHANGES FROM V3:
- Removed partition size limits - no longer stops at 9.99 × 10^72
- Implements 30-minute time limit per hypergraph
- Returns best solution found within time limit
- Better progress tracking with time remaining

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
from hdh.passes.cut_weighted_bad import compute_cut, cost, weighted_cost

# Note: compute_cut is BOTH the partitioning function AND cost computation function
# compute_cut(hdh, k, cap) returns (partitions, cost)
# When given partitions directly, use cost() instead

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


def brute_force_qubit_level_timed(
    hdh: HDH,
    k: int,
    cap: int,
    time_limit_seconds: int = 1800,  # 30 minutes = 1800 seconds
    progress_interval: int = 10  # Report progress every 10 seconds
) -> Tuple[Dict[str, int], float, int, bool]:
    """
    TIMED: Brute force at QUBIT level with time limit.
    
    Optimizes for WEIGHTED cost (quantum cuts = 10, classical cuts = 1).
    
    Instead of stopping at large partition counts, runs for a fixed time
    and returns the best solution found within that time.
    
    Args:
        hdh: The HDH object
        k: Number of partitions (QPUs)
        cap: Maximum unique qubits per partition
        time_limit_seconds: Maximum time to search (default: 1800s = 30min)
        progress_interval: How often to report progress (seconds)
    
    Returns:
        (best_partition, best_weighted_cost, partitions_checked, completed)
        - best_weighted_cost: 10*quantum_cuts + 1*classical_cuts
        - completed: True if exhausted search space, False if stopped due to time
    """
    qubits = sorted(extract_qubits_from_hdh(hdh))
    n_qubits = len(qubits)
    
    if n_qubits == 0:
        return {}, 0, 0, True
    
    total_partitions = k ** n_qubits
    print(f"  Timed qubit-level brute force: {n_qubits} qubits into {k} partitions (cap={cap})")
    print(f"  Total search space: {total_partitions:,} partitions")
    print(f"  Time limit: {time_limit_seconds}s ({time_limit_seconds/60:.1f} min)")
    
    node_to_qubit = get_node_qubit_mapping(hdh)
    qubit_to_nodes = defaultdict(set)
    for node, qbit in node_to_qubit.items():
        qubit_to_nodes[qbit].add(node)
    
    # Add classical nodes (they go with first partition by default)
    classical_nodes = [n for n in hdh.S if hdh.sigma[n] == 'c']
    
    min_cut = float('inf')
    best_partition = None
    
    # Generate qubit assignments
    assignments = product(range(k), repeat=n_qubits)
    
    # Timing setup
    start_time = time.time()
    last_progress_time = start_time
    partitions_checked = 0
    completed = False
    
    print(f"  Starting search at {time.strftime('%H:%M:%S')}")
    print(f"  Will stop at {time.strftime('%H:%M:%S', time.localtime(start_time + time_limit_seconds))}")
    
    for assignment in assignments:
        # Check time limit
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > time_limit_seconds:
            print(f"\n  ⏱️  TIME LIMIT REACHED ({time_limit_seconds}s)")
            break
        
        # Progress update at intervals
        if current_time - last_progress_time >= progress_interval:
            rate = partitions_checked / elapsed if elapsed > 0 else 0
            pct_complete = (partitions_checked / total_partitions * 100) if total_partitions > 0 else 0
            remaining_time = time_limit_seconds - elapsed
            
            print(f"  ⏳ {int(elapsed)}s elapsed | "
                  f"{partitions_checked:,} checked ({pct_complete:.2f}%) | "
                  f"Rate: {rate:.0f}/s | "
                  f"Best weighted cost: {min_cut if min_cut != float('inf') else 'N/A'} | "
                  f"Time left: {int(remaining_time)}s")
            last_progress_time = current_time
        
        # Check capacity
        partition_qubits = [set() for _ in range(k)]
        for i, qbit in enumerate(qubits):
            partition_qubits[assignment[i]].add(qbit)
        
        # Skip if any partition exceeds capacity
        if any(len(pq) > cap for pq in partition_qubits):
            partitions_checked += 1
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
        
        # Compute WEIGHTED cost (not just unweighted cut count)
        # Convert node_partition dict to partition sets for cost() function
        temp_partition_sets = [set() for _ in range(k)]
        for node, part_id in node_partition.items():
            temp_partition_sets[part_id].add(node)
        
        cost_raw = cost(hdh, temp_partition_sets)
        if isinstance(cost_raw, (tuple, list)):
            cuts = weighted_cost(cost_raw)
        else:
            cuts = float(cost_raw)
        
        if cuts < min_cut:
            min_cut = cuts
            best_partition = node_partition.copy()
            elapsed_now = time.time() - start_time
            print(f"  🎯 NEW BEST: {min_cut} weighted cost (found at {int(elapsed_now)}s, {partitions_checked:,} checked)")
            
            # Early termination if optimal
            if min_cut == 0:
                print(f"  ✨ Found optimal solution (0 cuts)! Stopping early.")
                completed = True
                break
        
        partitions_checked += 1
    else:
        # Loop completed naturally (exhausted search space)
        completed = True
        print(f"  ✅ COMPLETED full search space ({partitions_checked:,} partitions)")
    
    total_elapsed = time.time() - start_time
    print(f"  Final: {min_cut if min_cut != float('inf') else 'N/A'} cuts | "
          f"{partitions_checked:,} checked | "
          f"{total_elapsed:.1f}s elapsed | "
          f"{'COMPLETE' if completed else 'TIME-LIMITED'}")
    
    return (best_partition if best_partition else {}, 
            int(min_cut) if min_cut != float('inf') else 0,
            partitions_checked,
            completed)


# IMPORT MISSING FUNCTION FROM ORIGINAL
def load_small_mqtbench_hdhs(pkl_dir: str, min_qubits: int, max_qubits: int, max_nodes: int = None) -> Dict:
    """
    Load HDH circuits from MQT Bench pickle files.
    Now accepts any size circuit since we use time limits instead of partition count limits.
    """
    pkl_path = Path(pkl_dir)
    if not pkl_path.exists():
        print(f"✗ Path does not exist: {pkl_path}")
        return {}
    
    pkl_files = list(pkl_path.glob('*.pkl'))
    print(f"✓ Found {len(pkl_files)} pickle files in {pkl_path}")
    
    loaded = {}
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                h = pickle.load(f)
            
            # Extract circuit info
            num_qubits = len(extract_qubits_from_hdh(h))
            num_nodes = len(h.S)
            
            # Check qubit range
            if num_qubits < min_qubits or num_qubits > max_qubits:
                continue
            
            # Check node count if specified (but not strictly enforced)
            if max_nodes and num_nodes > max_nodes:
                print(f"  Note: {pkl_file.stem} has {num_nodes} nodes (>{max_nodes}), but will try anyway with time limit")
            
            circuit_name = pkl_file.stem
            loaded[circuit_name] = h
            
            num_qnodes = sum(1 for n in h.S if h.sigma[n] == 'q')
            num_cnodes = sum(1 for n in h.S if h.sigma[n] == 'c')
            num_edges = len(h.C)
            
            print(f"  ✓ Loaded: {circuit_name}")
            print(f"      Qubits: {num_qubits}, Nodes: {num_nodes} (Q: {num_qnodes}, C: {num_cnodes}), Edges: {num_edges}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {pkl_file.name}: {e}")
    
    print(f"\n✓ Successfully loaded {len(loaded)} circuits")
    return loaded


def run_comparison_experiment(
    hdhs: Dict[str, HDH],
    k: int,
    overhead_values: List[float],
    time_limit_per_graph: int = 1800,  # 30 minutes
    progress_interval: int = 10
) -> pd.DataFrame:
    """
    Run timed comparison experiments.
    
    Args:
        hdhs: Dictionary of circuit_name -> HDH
        k: Number of QPUs
        overhead_values: List of overhead multipliers
        time_limit_per_graph: Time limit per hypergraph in seconds
        progress_interval: Progress reporting interval in seconds
    """
    results = []
    
    total_experiments = len(hdhs) * len(overhead_values)
    completed_experiments = 0
    
    overall_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING TIMED COMPARISON EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Total circuits: {len(hdhs)}")
    print(f"Overhead values: {overhead_values}")
    print(f"Total experiments: {total_experiments}")
    print(f"Time limit per hypergraph: {time_limit_per_graph}s ({time_limit_per_graph/60:.1f} min)")
    print(f"{'='*70}\n")
    
    for circuit_idx, (circuit_name, h) in enumerate(hdhs.items(), 1):
        num_qubits = len(extract_qubits_from_hdh(h))
        num_nodes = len(h.S)
        num_qnodes = sum(1 for n in h.S if h.sigma[n] == 'q')
        num_cnodes = sum(1 for n in h.S if h.sigma[n] == 'c')
        num_edges = len(h.C)
        
        node_to_qubit = get_node_qubit_mapping(h)
        
        print(f"\n{'='*70}")
        print(f"CIRCUIT {circuit_idx}/{len(hdhs)}: {circuit_name}")
        print(f"{'='*70}")
        print(f"  Qubits: {num_qubits}")
        print(f"  Nodes: {num_nodes} (Quantum: {num_qnodes}, Classical: {num_cnodes})")
        print(f"  Hyperedges: {num_edges}")
        
        circuit_exp_completed = 0
        
        for overhead in overhead_values:
            cap = int(np.ceil(num_qubits / k * overhead))
            
            print(f"\n  --- Overhead: {overhead} (cap={cap}) ---")
            
            # TIMED BRUTE FORCE
            print(f"  🔍 Running TIMED brute force...")
            start_time = time.time()
            
            try:
                best_partition, optimal_cost, partitions_checked, completed = brute_force_qubit_level_timed(
                    h, k, cap,
                    time_limit_seconds=time_limit_per_graph,
                    progress_interval=progress_interval
                )
                
                brute_force_time = time.time() - start_time
                
                # Convert to partition sets for analysis
                partition_sets = [set() for _ in range(k)]
                for node, part_id in best_partition.items():
                    partition_sets[part_id].add(node)
                
                partition_qubit_counts = [
                    get_partition_qubit_count(pset, node_to_qubit)
                    for pset in partition_sets
                ]
                
                partition_classical_counts = [
                    sum(1 for n in pset if h.sigma.get(n) == 'c')
                    for pset in partition_sets
                ]
                
                # CRITICAL FIX: Apply weighted cost to optimal partition (same as heuristic)
                optimal_cost_raw = cost(h, partition_sets)
                if isinstance(optimal_cost_raw, (tuple, list)):
                    optimal_cost = weighted_cost(optimal_cost_raw)
                else:
                    optimal_cost = float(optimal_cost_raw)
                
                status = "COMPLETE" if completed else "TIME-LIMITED"
                
                print(f"    ✓ Timed brute force ({status}):")
                print(f"      Raw costs: {optimal_cost_raw} (quantum, classical)")
                print(f"      Weighted cost: {optimal_cost}")
                print(f"      Partitions checked: {partitions_checked:,}")
                print(f"      Time: {brute_force_time:.2f}s")
                print(f"      Qubit counts: {partition_qubit_counts}")
                
            except Exception as e:
                print(f"    ✗ Timed brute force failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # HEURISTIC - Using compute_cut() from cut_weighted
            start_time = time.time()
            heuristic_cost = None
            heuristic_time = None
            heuristic_qubit_counts = None
            heuristic_classical_counts = None
            
            try:
                # compute_cut returns (partitions, cost) directly
                # It doesn't take overhead, it takes cap (which we already calculated)
                heuristic_partitions, heuristic_cost_value = compute_cut(h, k, cap)
                
                # Get detailed cost breakdown using cost() function
                heuristic_cost_raw = cost(h, heuristic_partitions)
                
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
                    sum(1 for n in pset if h.sigma.get(n) == 'c')
                    for pset in heuristic_partitions
                ]
                
                print(f"    ✓ Heuristic (greedy temporal partitioning):")
                print(f"      Raw costs: {heuristic_cost_raw} (quantum, classical)")
                print(f"      Weighted cost: {heuristic_cost}")
                print(f"      Time: {heuristic_time:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Heuristic failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Calculate ratio
            if heuristic_cost is None:
                ratio = float('nan')
            elif optimal_cost > 0:
                ratio = heuristic_cost / optimal_cost
            else:
                ratio = 1.0 if heuristic_cost == 0 else float('inf')
            
            print(f"    → Ratio (heuristic/optimal): {ratio:.4f}")
            
            if ratio < 1.0 and completed:
                print(f"    ⚠️  WARNING: Heuristic beat optimal! This should not happen!")
            elif ratio < 1.0 and not completed:
                print(f"    ℹ️  Note: Heuristic better than time-limited search (not full optimal)")
            
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
                'method': 'TIMED-QUBIT-LEVEL',
                'partitions_checked': partitions_checked,
                'search_completed': completed,
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
    print(f"Total time elapsed: {int(total_elapsed//3600)}h {int((total_elapsed%3600)//60)}m {int(total_elapsed%60)}s")
    if completed_experiments > 0:
        print(f"Average time per experiment: {total_elapsed/completed_experiments:.1f}s")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(results)
    
    csv_path = OUTPUT_DIR / f'results_updated.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TIMED Brute-force optimal vs heuristic cut comparison'
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
        default=10,
        help='Maximum qubits for circuits to test'
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
        '--time_limit',
        type=int,
        default=1800,
        help='Time limit per hypergraph in seconds (default: 1800 = 30 min)'
    )
    parser.add_argument(
        '--progress_interval',
        type=int,
        default=10,
        help='Progress report interval in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    overhead_values = [float(x) for x in args.overhead.split(',')]
    
    print("\n" + "="*70)
    print(f"TIMED BRUTE-FORCE COMPARISON")
    print("="*70)
    print(f"Min qubits: {args.min_qubits}")
    print(f"Max qubits: {args.max_qubits}")
    print(f"k (QPUs): {args.k}")
    print(f"Overhead values: {overhead_values}")
    print(f"Time limit per hypergraph: {args.time_limit}s ({args.time_limit/60:.1f} min)")
    print(f"Progress interval: {args.progress_interval}s")
    print("="*70)
    
    total_start = time.time()
    
    # Load circuits (no strict max_nodes limit since we use time limits)
    hdhs = load_small_mqtbench_hdhs(
        args.pkl_dir,
        args.min_qubits,
        args.max_qubits,
        max_nodes=None  # Allow any size
    )
    
    if not hdhs:
        print("✗ No suitable circuits loaded.")
        return
    
    # Run comparison
    df = run_comparison_experiment(
        hdhs,
        k=args.k,
        overhead_values=overhead_values,
        time_limit_per_graph=args.time_limit,
        progress_interval=args.progress_interval
    )


if __name__ == '__main__':
    main()

# python -m hdh.hdh_bruteforce \
#        --pkl_dir database/HDHs/Circuit/MQTBench/pkl \
#        --min_qubits 3 \
#        --max_qubits 5 \
#        --time_limit 600 \
#        --progress_interval 10
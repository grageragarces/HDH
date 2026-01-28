#!/usr/bin/env python3
"""
TIMED Brute Force Cut Optimization - CORRECTLY FIXED

THE ACTUAL BUG:
Lines 303-309 in brute_force_node_level_timed() abort immediately for >30 nodes,
returning empty partition with cost=0.

THE CORRECT FIX:
Remove the max_nodes_for_full_search check. The function already has time limits,
so it will explore node-level search for the full time duration regardless of
node count, returning the best solution found.

This preserves the desired behavior:
- Nodes of same qubit CAN be split across QPUs
- Capacity based on unique qubits per partition
- Time-bounded search (your 10 minutes)
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
    time_limit_seconds: int = 1800,
    progress_interval: int = 10
) -> Tuple[Dict[str, int], float, int, bool]:
    """
    TIMED: Brute force at QUBIT level with time limit.
    All nodes of same qubit go to same partition.
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
    
    classical_nodes = [n for n in hdh.S if hdh.sigma[n] == 'c']
    
    min_cut = float('inf')
    best_partition = None
    
    assignments = product(range(k), repeat=n_qubits)
    
    start_time = time.time()
    last_progress_time = start_time
    partitions_checked = 0
    completed = False
    
    print(f"  Starting search at {time.strftime('%H:%M:%S')}")
    print(f"  Will stop at {time.strftime('%H:%M:%S', time.localtime(start_time + time_limit_seconds))}")
    
    for assignment in assignments:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > time_limit_seconds:
            print(f"\n  ⏱️  TIME LIMIT REACHED ({time_limit_seconds}s)")
            break
        
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
        
        partition_qubits = [set() for _ in range(k)]
        for i, qbit in enumerate(qubits):
            partition_qubits[assignment[i]].add(qbit)
        
        if any(len(pq) > cap for pq in partition_qubits):
            partitions_checked += 1
            continue
        
        node_partition = {}
        for i, qbit in enumerate(qubits):
            partition_id = assignment[i]
            for node in qubit_to_nodes[qbit]:
                node_partition[node] = partition_id
        
        for node in classical_nodes:
            node_partition[node] = 0
        
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
            
            if min_cut == 0:
                print(f"  ✨ Found optimal solution (0 cuts)! Stopping early.")
                completed = True
                break
        
        partitions_checked += 1
    else:
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




def brute_force_node_level_timed(
    hdh: HDH,
    k: int,
    cap: int,
    time_limit_seconds: int = 1800,
    progress_interval: int = 10,
) -> Tuple[Dict[str, int], int, int, bool]:
    """
    ============================================================================
    CORRECTLY FIXED: Brute force at NODE level with time limit.
    
    THE FIX: Removed the max_nodes_for_full_search check that caused early
    return with empty partition (cost=0). Now runs for full time limit
    regardless of node count.
    
    Nodes of same qubit CAN be split across QPUs.
    Capacity constraint is on unique qubits per partition.
    ============================================================================
    """
    nodes = list(hdh.S)
    n_nodes = len(nodes)

    if n_nodes == 0:
        return {}, 0, 0, True

    # ========== CRITICAL FIX: REMOVED THIS CHECK ==========
    # This was causing immediate return with empty partition for >30 nodes
    # 
    # OLD CODE (BUGGY):
    # if n_nodes > max_nodes_for_full_search:
    #     warnings.warn(...)
    #     return {}, 0, 0, False  # ❌ Returns cost=0
    #
    # NEW CODE: Just proceed - time limit will handle large search spaces
    # =======================================================

    node_to_qubit = get_node_qubit_mapping(hdh)

    incidence: Dict[str, List] = defaultdict(list)
    for edge in hdh.C:
        for n in edge:
            incidence[n].append(edge)

    nodes.sort(key=lambda n: len(incidence.get(n, [])), reverse=True)

    edge_type_map = getattr(hdh, "tau", {})
    edge_weight_map = getattr(hdh, "edge_weight", {})

    def edge_weighted_value(edge) -> int:
        w = int(edge_weight_map.get(edge, 1))
        t = edge_type_map.get(edge, "q")
        return (10 * w) if t == "q" else (1 * w)

    assignment: Dict[str, int] = {}
    part_qubits: List[set] = [set() for _ in range(k)]
    edge_mask: Dict = defaultdict(int)

    best_cost = float("inf")
    best_assignment: Optional[Dict[str, int]] = None

    start_time = time.time()
    last_progress = [start_time]
    states_visited = 0
    completed = False

    def mask_is_cut(m: int) -> bool:
        return (m & (m - 1)) != 0

    def dfs(i: int, lb_cost: int) -> None:
        nonlocal best_cost, best_assignment, states_visited

        now = time.time()
        if now - start_time > time_limit_seconds:
            return

        if now - last_progress[0] >= progress_interval:
            elapsed = now - start_time
            rate = states_visited / elapsed if elapsed > 0 else 0
            print(
                f"  ⏳ {int(elapsed)}s elapsed | "
                f"States visited: {states_visited:,} | "
                f"Rate: {rate:.0f}/s | "
                f"Best weighted cost: {best_cost if best_cost != float('inf') else 'N/A'}"
            )
            last_progress[0] = now

        if lb_cost >= best_cost:
            return

        if i == n_nodes:
            states_visited += 1
            parts = [set() for _ in range(k)]
            for n, pid in assignment.items():
                parts[pid].add(n)

            cost_raw = cost(hdh, parts)
            c = int(weighted_cost(cost_raw)) if isinstance(cost_raw, (tuple, list)) else int(cost_raw)

            if c < best_cost:
                best_cost = c
                best_assignment = assignment.copy()
                elapsed = time.time() - start_time
                print(f"  🎯 NEW BEST (node-level): {best_cost} (found at {int(elapsed)}s)")
            return

        node = nodes[i]

        pid_order = list(range(k))
        pid_order.sort(key=lambda pid: len(part_qubits[pid]))

        for pid in pid_order:
            q = node_to_qubit.get(node, None)
            added_qubit = False
            if q is not None and q not in part_qubits[pid]:
                if len(part_qubits[pid]) + 1 > cap:
                    continue
                part_qubits[pid].add(q)
                added_qubit = True

            assignment[node] = pid

            delta_lb = 0
            updated_edges = []
            for e in incidence.get(node, []):
                old = edge_mask[e]
                new = old | (1 << pid)
                if new != old:
                    updated_edges.append((e, old))
                    edge_mask[e] = new
                    if (not mask_is_cut(old)) and mask_is_cut(new):
                        delta_lb += edge_weighted_value(e)

            dfs(i + 1, lb_cost + delta_lb)

            for e, old in reversed(updated_edges):
                edge_mask[e] = old

            assignment.pop(node, None)
            if added_qubit:
                part_qubits[pid].remove(q)

            if time.time() - start_time > time_limit_seconds:
                return

    print(f"  Timed node-level brute force: {n_nodes} nodes into {k} partitions (cap={cap})")
    print(f"  Time limit: {time_limit_seconds}s ({time_limit_seconds/60:.1f} min)")
    print(f"  Starting search at {time.strftime('%H:%M:%S')}")
    print(f"  Will stop at {time.strftime('%H:%M:%S', time.localtime(start_time + time_limit_seconds))}")

    dfs(0, 0)

    elapsed = time.time() - start_time
    
    # Check if we found anything
    if best_cost == float('inf'):
        print(f"  ⚠️  WARNING: No valid partition found within time limit!")
        print(f"  This might happen if capacity constraints are too tight.")
    else:
        print(
            f"  Final (node-level): {best_cost} | "
            f"States visited: {states_visited:,} | "
            f"{elapsed:.1f}s elapsed | "
            f"{'COMPLETE' if completed else 'TIME-LIMITED'}"
        )

    return (best_assignment if best_assignment else {},
            int(best_cost) if best_cost != float('inf') else 0,
            states_visited,
            completed)



def load_small_mqtbench_hdhs(pkl_dir: str, min_qubits: int, max_qubits: int, max_nodes: int = None) -> Dict:
    """Load HDH circuits from MQT Bench pickle files."""
    pkl_path = Path(pkl_dir)
    if not pkl_path.exists():
        print(f"✗ Pickle directory not found: {pkl_path}")
        return {}
    
    pkl_files = sorted(pkl_path.glob('*.pkl'))
    print(f"✓ Found {len(pkl_files)} pickle files")
    
    hdhs = {}
    loaded = 0
    skipped_qubits = 0
    skipped_nodes = 0
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                h = pickle.load(f)
            
            if not isinstance(h, HDH):
                continue
            
            num_qubits = len(extract_qubits_from_hdh(h))
            num_nodes = len(h.S)
            
            if num_qubits < min_qubits or num_qubits > max_qubits:
                skipped_qubits += 1
                continue
            
            if max_nodes is not None and num_nodes > max_nodes:
                skipped_nodes += 1
                continue
            
            circuit_name = pkl_file.stem
            hdhs[circuit_name] = h
            loaded += 1
            
        except Exception as e:
            warnings.warn(f"Error loading {pkl_file}: {e}")
            continue
    
    print(f"✓ Loaded {loaded} circuits")
    if skipped_qubits > 0:
        print(f"  Skipped {skipped_qubits} circuits (qubit count out of range)")
    if skipped_nodes > 0:
        print(f"  Skipped {skipped_nodes} circuits (too many nodes)")
    
    return hdhs


def run_comparison_experiment(
    hdhs: Dict[str, HDH],
    k: int = 3,
    overhead_values: List[float] = [1.0, 1.1, 1.2, 1.3],
    time_limit_per_graph: int = 1800,
    progress_interval: int = 10
) -> pd.DataFrame:
    """Run timed comparison experiments - CORRECTLY FIXED."""
    results = []
    
    total_experiments = len(hdhs) * len(overhead_values)
    completed_experiments = 0
    
    overall_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING TIMED COMPARISON EXPERIMENTS (CORRECTLY FIXED)")
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
            
            print(f"  Running TIMED node-level brute force...")
            start_time = time.time()
            
            try:
                # Node-level brute force (now runs for full time regardless of node count)
                best_partition, optimal_cost, partitions_checked, completed = brute_force_node_level_timed(
                    h, k, cap,
                    time_limit_seconds=time_limit_per_graph,
                    progress_interval=progress_interval
                )

                brute_force_time = time.time() - start_time
                
                # Validate we got a valid partition
                if not best_partition:
                    print("  ⚠️  WARNING: No valid partition found. Trying qubit-level fallback...")
                    best_partition, optimal_cost, partitions_checked, completed = brute_force_qubit_level_timed(
                        h, k, cap,
                        time_limit_seconds=time_limit_per_graph,
                        progress_interval=progress_interval
                    )
                    brute_force_time = time.time() - start_time
                    
                    if not best_partition:
                        print("  ⚠️  ERROR: Both methods failed to find valid partition. Skipping.")
                        continue
                
                # Convert to partition sets
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
                
                # Recalculate cost for consistency
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
            
            # HEURISTIC
            start_time = time.time()
            heuristic_cost = None
            heuristic_time = None
            heuristic_qubit_counts = None
            heuristic_classical_counts = None
            
            try:
                heuristic_partitions, heuristic_cost_value = compute_cut(h, k, cap)
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

            completed_experiments += 1
            circuit_exp_completed += 1
            
            overall_pct = (completed_experiments / total_experiments) * 100
            print(f"    Overall progress: {completed_experiments}/{total_experiments} ({overall_pct:.1f}%)")
            
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
                'method': 'NODE-LEVEL-TIMED',
                'partitions_checked': partitions_checked,
                'search_completed': completed,
                'optimal_qubit_counts': str(partition_qubit_counts),
                'heuristic_qubit_counts': str(heuristic_qubit_counts) if heuristic_qubit_counts else None,
                'optimal_classical_counts': str(partition_classical_counts),
                'heuristic_classical_counts': str(heuristic_classical_counts) if heuristic_classical_counts else None,
            })
    
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
    
    csv_path = OUTPUT_DIR / f'results_node_level_fixed_over10.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CORRECTLY FIXED: Node-level timed brute-force comparison'
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
    print(f"CORRECTLY FIXED: NODE-LEVEL TIMED BRUTE-FORCE")
    print("="*70)
    print(f"Min qubits: {args.min_qubits}")
    print(f"Max qubits: {args.max_qubits}")
    print(f"k (QPUs): {args.k}")
    print(f"Overhead values: {overhead_values}")
    print(f"Time limit per hypergraph: {args.time_limit}s ({args.time_limit/60:.1f} min)")
    print(f"Progress interval: {args.progress_interval}s")
    print("="*70)
    
    total_start = time.time()
    
    hdhs = load_small_mqtbench_hdhs(
        args.pkl_dir,
        args.min_qubits,
        args.max_qubits,
        max_nodes=None
    )
    
    if not hdhs:
        print("✗ No suitable circuits loaded.")
        return
    
    df = run_comparison_experiment(
        hdhs,
        k=args.k,
        overhead_values=overhead_values,
        time_limit_per_graph=args.time_limit,
        progress_interval=args.progress_interval
    )


if __name__ == '__main__':
    main()
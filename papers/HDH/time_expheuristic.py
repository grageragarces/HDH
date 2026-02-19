import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import time
import warnings

sys.path.insert(0, str(Path.cwd()))

from hdh import HDH, hdh
from hdh.passes.cut import compute_cut, cost, weighted_cost

OUTPUT_DIR = Path('experiment_outputs_heuristic_100q')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            except (ValueError, IndexError):
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
            except (ValueError, IndexError):
                continue
    
    return node_to_qubit


def get_partition_qubit_count(nodes: Set[str], node_to_qubit: Dict[str, int]) -> int:
    """Count unique qubits in a partition."""
    qubits = set()
    for node in nodes:
        if node in node_to_qubit:
            qubits.add(node_to_qubit[node])
    return len(qubits)


def load_100q_circuits_fast(pkl_dir: str, max_circuits: int = 10) -> List[Tuple[str, HDH]]:
    """
    Quickly load the first max_circuits with ~100 qubits.
    """
    pkl_path = Path(pkl_dir)
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Directory not found: {pkl_path}")
    
    pkl_files = sorted(pkl_path.glob('*.pkl'))
    print(f"\n📂 Scanning {len(pkl_files)} pickle files...")
    
    hdhs = []
    
    for pkl_file in pkl_files:
        if len(hdhs) >= max_circuits:
            break
            
        try:
            with open(pkl_file, 'rb') as f:
                h = pickle.load(f)
            
            qubits = extract_qubits_from_hdh(h)
            num_qubits = len(qubits)
            
            # Accept circuits with 95-105 qubits
            if 95 <= num_qubits <= 105:
                num_nodes = len(h.S)
                num_qnodes = sum(1 for n in h.S if h.sigma[n] == 'q')
                num_edges = len(h.C)
                
                hdhs.append((pkl_file.stem, h))
                print(f"  ✓ {len(hdhs)}/{max_circuits}: {pkl_file.stem} - "
                      f"{num_qubits}Q, {num_nodes}N, {num_edges}E")
        
        except Exception:
            continue
    
    print(f"\n✓ Loaded {len(hdhs)} circuits")
    return hdhs


def run_quick_benchmark(hdhs: List[Tuple[str, HDH]], k: int = 3) -> pd.DataFrame:
    """
    Quick benchmark with overhead=1.0 only.
    """
    results = []
    overhead = 1.0
    
    print("\n" + "="*70)
    print("QUICK HEURISTIC BENCHMARK (overhead=1.0)")
    print("="*70)
    print(f"Circuits: {len(hdhs)}")
    print(f"k (QPUs): {k}")
    print("="*70 + "\n")
    
    for idx, (circuit_name, h) in enumerate(hdhs, 1):
        qubits = extract_qubits_from_hdh(h)
        num_qubits = len(qubits)
        num_nodes = len(h.S)
        num_edges = len(h.C)
        
        cap = int(np.ceil(num_qubits / k * overhead))
        
        print(f"{idx}/{len(hdhs)}: {circuit_name} ({num_qubits}Q, cap={cap})")
        
        node_to_qubit = get_node_qubit_mapping(h)
        
        try:
            start_time = time.time()
            partitions, _ = compute_cut(h, k, cap)
            heuristic_time = time.time() - start_time
            
            cost_raw = cost(h, partitions)
            
            if isinstance(cost_raw, (tuple, list)):
                heuristic_cost = weighted_cost(cost_raw)
            else:
                heuristic_cost = float(cost_raw)
            
            qubit_counts = [
                get_partition_qubit_count(pset, node_to_qubit)
                for pset in partitions
            ]
            
            print(f"  ✓ Time: {heuristic_time:.4f}s, Cost: {heuristic_cost}, Qubits: {qubit_counts}")
            
            results.append({
                'circuit': circuit_name,
                'num_qubits': num_qubits,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'k': k,
                'capacity': cap,
                'time': heuristic_time,
                'cost': heuristic_cost,
                'qubit_counts': str(qubit_counts),
            })
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        print(f"Average time: {df['time'].mean():.4f}s")
        print(f"Median time: {df['time'].median():.4f}s")
        print(f"Min time: {df['time'].min():.4f}s")
        print(f"Max time: {df['time'].max():.4f}s")
        print(f"Average cost: {df['cost'].mean():.2f}")
        print(f"{'='*70}\n")
        
        csv_path = OUTPUT_DIR / 'quick_benchmark_100q.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick heuristic benchmark for 100Q circuits')
    parser.add_argument('--pkl_dir', type=str, 
                       default='/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl',
                       help='Directory containing pickle files')
    parser.add_argument('--max_circuits', type=int, default=10,
                       help='Number of circuits to test (default: 10)')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of QPUs (default: 3)')
    
    args = parser.parse_args()
    
    hdhs = load_100q_circuits_fast(args.pkl_dir, args.max_circuits)
    
    if not hdhs:
        print("✗ No suitable circuits found.")
        return
    
    df = run_quick_benchmark(hdhs, k=args.k)


if __name__ == '__main__':
    main()
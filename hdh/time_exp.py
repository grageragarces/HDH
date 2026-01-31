import sys
import pickle
import time
from pathlib import Path

# Add the current directory to path to import hdh modules
sys.path.insert(0, str(Path.cwd()))

from hdh.hdh_bruteforce import (
    brute_force_qubit_level_timed,
    extract_qubits_from_hdh,
    get_node_qubit_mapping,
    get_partition_qubit_count
)

def run_single_circuit_experiment(pkl_path, k=3, overhead=1.2, time_limit_minutes=10):
    """
    Run a timed brute-force experiment on a single circuit.
    
    Args:
        pkl_path: Path to the pickle file containing the HDH
        k: Number of partitions (devices)
        overhead: Capacity overhead multiplier
        time_limit_minutes: Time limit in minutes
    
    Returns:
        Dictionary with experiment results
    """
    time_limit_seconds = time_limit_minutes * 60
    
    print("="*70)
    print(f"10-MINUTE BRUTE-FORCE PARTITION COUNTING EXPERIMENT")
    print("="*70)
    print(f"Circuit file: {pkl_path}")
    print(f"Number of devices (k): {k}")
    print(f"Overhead: {overhead}")
    print(f"Time limit: {time_limit_minutes} minutes ({time_limit_seconds} seconds)")
    print("="*70)
    print()
    
    # Load the HDH from pickle
    print(f"Loading circuit from {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            hdh = pickle.load(f)
        print(f"✓ Circuit loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: File not found at {pkl_path}")
        print(f"  Please make sure ae_indep_qiskit_10.pkl is in the current directory")
        return None
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        return None
    
    # Extract circuit statistics
    qubits = sorted(extract_qubits_from_hdh(hdh))
    num_qubits = len(qubits)
    num_nodes = len(hdh.S)
    num_quantum_nodes = sum(1 for n in hdh.S if hdh.sigma.get(n) == 'q')
    num_classical_nodes = sum(1 for n in hdh.S if hdh.sigma.get(n) == 'c')
    num_edges = len(hdh.C)
    
    print(f"\nCircuit Statistics:")
    print(f"  Logical qubits: {num_qubits}")
    print(f"  Total nodes: {num_nodes}")
    print(f"    - Quantum nodes: {num_quantum_nodes}")
    print(f"    - Classical nodes: {num_classical_nodes}")
    print(f"  Hyperedges: {num_edges}")
    
    # Calculate capacity
    cap = int(num_qubits * overhead / k)
    print(f"  Capacity per device: {cap} qubits")
    
    # Calculate theoretical search space
    theoretical_search_space = k ** num_qubits
    print(f"\nTheoretical search space: {theoretical_search_space:,} partitions")
    print(f"  (This is {k}^{num_qubits} = {k}**{num_qubits})")
    
    # Estimate in scientific notation for the footnote
    import math
    exponent = num_qubits * math.log10(k)
    print(f"  In scientific notation: ~10^{exponent:.1f}")
    
    print(f"\n{'='*70}")
    print(f"STARTING BRUTE-FORCE SEARCH")
    print(f"{'='*70}\n")
    
    # Run the timed brute-force search
    start_time = time.time()
    
    best_partition, optimal_cost, partitions_checked, completed = brute_force_qubit_level_timed(
        hdh=hdh,
        k=k,
        cap=cap,
        time_limit_seconds=time_limit_seconds,
        progress_interval=10  # Report progress every 10 seconds
    )
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    partitions_per_second = partitions_checked / total_time if total_time > 0 else 0
    percent_explored = (partitions_checked / theoretical_search_space * 100) if theoretical_search_space > 0 else 0
    
    # Estimate the order of magnitude
    if partitions_checked > 0:
        partitions_order = math.log10(partitions_checked)
    else:
        partitions_order = 0
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"\n📊 Performance Metrics:")
    print(f"  Total partitions evaluated: {partitions_checked:,}")
    print(f"  Order of magnitude: ~10^{partitions_order:.1f}")
    print(f"  Search completed: {'Yes ✓' if completed else 'No (time limit reached)'}")
    print(f"  Percent of search space explored: {percent_explored:.4f}%")
    print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Evaluation rate: {partitions_per_second:.1f} partitions/second")
    
    print(f"\n🎯 Best Solution Found:")
    print(f"  Optimal weighted cost: {optimal_cost}")
    
    if best_partition:
        # Get partition statistics
        node_to_qubit = get_node_qubit_mapping(hdh)
        partition_sets = [set() for _ in range(k)]
        for node, part_id in best_partition.items():
            partition_sets[part_id].add(node)
        
        partition_qubit_counts = [
            get_partition_qubit_count(pset, node_to_qubit)
            for pset in partition_sets
        ]
        
        print(f"  Qubit distribution: {partition_qubit_counts}")
    
    print(f"\n💡 For your footnote:")
    print(f"  \"Our implementation evaluates on the order of 10^{int(partitions_order)} capacity-respecting")
    print(f"  partitions within the 10-minute time cap.\"")
    
    print(f"\n{'='*70}\n")
    
    # Return results as a dictionary
    results = {
        'circuit_file': pkl_path,
        'num_qubits': num_qubits,
        'num_nodes': num_nodes,
        'k': k,
        'overhead': overhead,
        'capacity': cap,
        'theoretical_search_space': theoretical_search_space,
        'partitions_checked': partitions_checked,
        'partitions_order_of_magnitude': int(partitions_order),
        'completed': completed,
        'percent_explored': percent_explored,
        'total_time_seconds': total_time,
        'partitions_per_second': partitions_per_second,
        'optimal_cost': optimal_cost,
    }
    
    return results


if __name__ == '__main__':
    # Run the experiment
    # You can modify these parameters as needed
    results = run_single_circuit_experiment(
        pkl_path='/Users/mariagragera/Desktop/HDH/database/HDHs/Circuit/MQTBench/pkl/dj_indep_qiskit_10.pkl',  # The circuit file
        k=3,                                 # Number of devices
        overhead=1.2,                        # Capacity overhead
        time_limit_minutes=10                # Time limit in minutes
    )
    
    if results:
        # Optionally save results to a file
        import json
        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to experiment_results.json")

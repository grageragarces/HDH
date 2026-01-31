#!/usr/bin/env python3
"""
HDH Cutting Experiment - Parallel Version (Standalone)
======================================================
Generate random workloads from 4 quantum computing models, build HDHs,
run the heuristic cutter, and save results.

This version uses multiprocessing to parallelize across 11 cores.

Usage:
    python experiment_parallel.py

This is a standalone version designed to avoid module import issues.
"""

if __name__ == "__main__":
    import sys
    import os
    import pickle
    import time
    import random
    import csv
    from pathlib import Path
    from typing import List, Tuple, Dict
    import traceback
    import multiprocessing as mp
    from functools import partial
    import platform
    
    # Fix for macOS multiprocessing
    if platform.system() == 'Darwin':
        mp.set_start_method('fork', force=True)
    
    # Import HDH modules
    from hdh.hdh import HDH
    from hdh.models.circuit import Circuit
    from hdh.models.mbqc import MBQC
    from hdh.models.qw import QW
    from hdh.models.qca import QCA
    from hdh.passes.cut import compute_cut
    
    # ========================= CONFIGURATION =========================
    
    # Workload sizes to test
    QUBIT_SIZES = list(range(2, 201, 10))  # 2, 12, 22, ..., 192
    
    # Number of random workloads per size
    WORKLOADS_PER_SIZE = 5
    
    # Partitioning parameters
    K = 4  # Number of partitions (QPUs)
    CAP_MULTIPLIER = 1.5  # Capacity per QPU as multiple of qubits/k
    
    # Parallelization
    NUM_CORES = 11  # Number of CPU cores to use
    
    # Output paths
    BASE_DIR = Path("/Users/mariagragera/Desktop/HDH/database")
    HDH_DIR = BASE_DIR / "HDHs"
    RESULTS_CSV = BASE_DIR / "experiment_results.csv"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # ========================= WORKLOAD GENERATORS =========================
    
    def random_circuit(num_qubits: int, seed: int) -> Circuit:
        """Generate a random quantum circuit."""
        random.seed(seed)
        circuit = Circuit()
        
        num_gates = random.randint(num_qubits * 3, num_qubits * 10)
        gate_types = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        two_qubit_gates = ["cx", "cz", "swap"]
        
        for _ in range(num_gates):
            if random.random() < 0.7:
                gate = random.choice(gate_types)
                qubit = random.randint(0, num_qubits - 1)
                circuit.add_instruction(gate, [qubit])
            else:
                if num_qubits > 1:
                    gate = random.choice(two_qubit_gates)
                    q1, q2 = random.sample(range(num_qubits), 2)
                    circuit.add_instruction(gate, [q1, q2])
        
        num_measures = random.randint(1, min(5, num_qubits))
        measured_qubits = random.sample(range(num_qubits), num_measures)
        for q in measured_qubits:
            circuit.add_instruction("measure", [q], [q])
        
        return circuit
    
    def random_mbqc(num_qubits: int, seed: int) -> MBQC:
        """Generate a random MBQC pattern."""
        random.seed(seed)
        mbqc = MBQC()
        qubits = [f"q{i}" for i in range(num_qubits)]
        num_ops = random.randint(num_qubits * 2, num_qubits * 5)
        
        for i in range(num_ops):
            op_type = random.choice(["N", "E", "M", "C"])
            
            if op_type == "N":
                A = random.sample(qubits, min(2, len(qubits))) if random.random() < 0.5 else []
                b = f"q{num_qubits + i}"
                mbqc.add_operation("N", A, b)
            elif op_type == "E":
                if len(qubits) >= 2:
                    A = random.sample(qubits, 2)
                    b = random.choice(qubits)
                    mbqc.add_operation("E", A, b)
            elif op_type == "M":
                A = [random.choice(qubits)]
                b = f"c{i}"
                mbqc.add_operation("M", A, b)
            else:
                A = [f"c{j}" for j in range(max(1, i-3), i) if j >= 0][:2]
                if A:
                    b = f"c{i}"
                    mbqc.add_operation("C", A, b)
        
        return mbqc
    
    def random_qw(num_qubits: int, seed: int) -> QW:
        """Generate a random quantum walk."""
        random.seed(seed)
        qw = QW()
        qubits = [qw._new_qubit_id() for _ in range(num_qubits)]
        num_steps = random.randint(num_qubits * 2, num_qubits * 5)
        
        for _ in range(num_steps):
            if random.random() < 0.5 and qubits:
                q = random.choice(qubits)
                q_prime = qw.add_coin(q)
                qubits.append(q_prime)
            elif len(qubits) > 0:
                q = random.choice(qubits)
                b = qw.add_shift(q)
                qubits.append(b)
        
        num_measures = min(5, len(qubits))
        for _ in range(num_measures):
            if len(qubits) >= 2:
                a, b = random.sample(qubits, 2)
                qw.add_measurement(a, b)
        
        return qw
    
    def random_qca(num_qubits: int, seed: int) -> QCA:
        """Generate a random QCA."""
        random.seed(seed)
        topology = {}
        for i in range(num_qubits):
            node = f"q{i}"
            num_neighbors = random.randint(1, min(3, num_qubits - 1))
            possible_neighbors = [f"q{j}" for j in range(num_qubits) if j != i]
            neighbors = random.sample(possible_neighbors, num_neighbors)
            topology[node] = neighbors
        
        num_measures = random.randint(1, min(5, num_qubits))
        measurements = [f"q{i}" for i in random.sample(range(num_qubits), num_measures)]
        steps = random.randint(3, 10)
        
        return QCA(topology, measurements, steps)
    
    def generate_workload(model_name: str, num_qubits: int, seed: int):
        """Generate a random workload for the given model."""
        generators = {
            "Circuit": random_circuit,
            "MBQC": random_mbqc,
            "QW": random_qw,
            "QCA": random_qca
        }
        return generators[model_name](num_qubits, seed)
    
    # ========================= WORKER FUNCTION =========================
    
    def process_workload(task):
        """Process a single workload: generate, build HDH, cut, save."""
        model_name, num_qubits, workload_id, seed, task_num, total_tasks = task
        
        try:
            # Generate workload
            workload = generate_workload(model_name, num_qubits, seed)
            
            # Build HDH
            hdh = workload.build_hdh()
            actual_qubits = hdh.get_num_qubits()
            
            # Save HDH pickle to Random subdirectory
            hdh_filename = f"{model_name.lower()}_{num_qubits}q_w{workload_id}.pkl"
            hdh_path = HDH_DIR / model_name / "Random" / hdh_filename
            hdh_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(hdh_path, 'wb') as f:
                pickle.dump(hdh, f)
            
            # Calculate capacity
            cap = max(1, int((actual_qubits / K) * CAP_MULTIPLIER))
            
            # Run heuristic cutter
            start_time = time.time()
            partitions, cut_cost = compute_cut(hdh, k=K, cap=cap)
            heuristic_time = time.time() - start_time
            
            # Print progress
            print(f"[{task_num}/{total_tasks}] {model_name} {num_qubits}q w{workload_id}: "
                  f"cost={cut_cost}, time={heuristic_time:.4f}s")
            
            # Return result
            return {
                "model": model_name,
                "num_qubits": num_qubits,
                "workload_id": workload_id,
                "hdh_file": str(hdh_path),
                "cut_cost": cut_cost,
                "heuristic_time": heuristic_time,
                "k": K,
                "cap": cap,
                "success": True
            }
            
        except Exception as e:
            print(f"[{task_num}/{total_tasks}] {model_name} {num_qubits}q w{workload_id}: ERROR - {e}")
            return {
                "model": model_name,
                "num_qubits": num_qubits,
                "workload_id": workload_id,
                "hdh_file": "ERROR",
                "cut_cost": -1,
                "heuristic_time": -1,
                "k": K,
                "cap": -1,
                "success": False
            }
    
    # ========================= EXPERIMENT RUNNER =========================
    
    def run_experiment_parallel():
        """Run the full HDH cutting experiment in parallel."""
        
        # Create directories
        for model in ["Circuit", "MBQC", "QW", "QCA"]:
            (HDH_DIR / model / "Random").mkdir(parents=True, exist_ok=True)
        
        # Generate all tasks
        models = ["Circuit", "MBQC", "QW", "QCA"]
        tasks = []
        task_num = 0
        
        for model_idx, model_name in enumerate(models):
            for size_idx, num_qubits in enumerate(QUBIT_SIZES):
                for workload_id in range(1, WORKLOADS_PER_SIZE + 1):
                    task_num += 1
                    # Create unique but reproducible seed for each task
                    seed = RANDOM_SEED + model_idx * 10000 + size_idx * 100 + workload_id
                    tasks.append((model_name, num_qubits, workload_id, seed, task_num, len(models) * len(QUBIT_SIZES) * WORKLOADS_PER_SIZE))
        
        total_tasks = len(tasks)
        print(f"\nTotal tasks to process: {total_tasks}")
        print(f"Using {NUM_CORES} CPU cores")
        print(f"Starting parallel execution...\n")
        
        # Run in parallel
        start_time = time.time()
        
        with mp.Pool(processes=NUM_CORES) as pool:
            results = pool.map(process_workload, tasks)
        
        elapsed = time.time() - start_time
        
        # Write all results to CSV
        print(f"\nAll tasks completed in {elapsed:.2f} seconds")
        print(f"Writing results to CSV...")
        
        with open(RESULTS_CSV, 'w', newline='') as csv_file:
            fieldnames = [
                "model", "num_qubits", "workload_id", "hdh_file", 
                "cut_cost", "heuristic_time", "k", "cap", "success"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            
            for result in results:
                if result is not None:
                    csv_writer.writerow(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r and r['success'])
        failed = total_tasks - successful
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"{'='*60}")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Average time per task: {elapsed/total_tasks:.4f} seconds")
        print(f"Results saved to: {RESULTS_CSV}")
        print(f"HDH pickles saved to: {HDH_DIR}/{{Model}}/Random/")
        print(f"{'='*60}")
    
    # ========================= MAIN =========================
    
    print("="*60)
    print("HDH Cutting Experiment - Parallel Version")
    print("="*60)
    print(f"Qubit sizes: {QUBIT_SIZES[0]} to {QUBIT_SIZES[-1]}")
    print(f"Workloads per size: {WORKLOADS_PER_SIZE}")
    print(f"Partitions (K): {K}")
    print(f"Capacity multiplier: {CAP_MULTIPLIER}")
    print(f"CPU cores: {NUM_CORES}")
    print(f"Random seed base: {RANDOM_SEED}")
    print("="*60)
    
    run_experiment_parallel()
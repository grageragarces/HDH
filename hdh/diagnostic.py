#!/usr/bin/env python3
"""
Diagnostic script to investigate why heuristic beats brute force.
Compares them on a single circuit with detailed output.
"""

import sys
import pickle
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path.cwd()))

from hdh import HDH
from hdh.passes.cut import compute_cut, cost

# Load the test circuit
print("Loading test circuit...")
with open('database/HDHs/Circuit/MQTBench/pkl/ae_indep_qiskit_4.pkl', 'rb') as f:
    hdh = pickle.load(f)

print(f"✓ Loaded circuit")
print(f"  Total nodes: {len(hdh.S)}")
print(f"  Total hyperedges: {len(hdh.C)}")

# Analyze nodes
quantum_nodes = [n for n in hdh.S if hdh.sigma.get(n) == 'q']
classical_nodes = [n for n in hdh.S if hdh.sigma.get(n) == 'c']
other_nodes = [n for n in hdh.S if hdh.sigma.get(n) not in ['q', 'c']]

print(f"\nNode breakdown:")
print(f"  Quantum: {len(quantum_nodes)}")
print(f"  Classical: {len(classical_nodes)}")
print(f"  Other: {len(other_nodes)}")

if other_nodes:
    print(f"\n⚠️ Found {len(other_nodes)} nodes that are neither 'q' nor 'c':")
    for node in other_nodes[:5]:
        print(f"    {node}: sigma={hdh.sigma.get(node)}")

# Check node naming formats
print(f"\nChecking node naming formats...")
import re
q_pattern = re.compile(r"^q(\d+)_t\d+$")

strict_format = 0  # Matches q{N}_t{T} exactly
loose_format = 0   # Has underscore but not strict format
no_underscore = 0  # No underscore at all

for node in quantum_nodes:
    if q_pattern.match(node):
        strict_format += 1
    elif '_' in node:
        loose_format += 1
    else:
        no_underscore += 1

print(f"  Strict format (q{{N}}_t{{T}}): {strict_format}")
print(f"  Loose format (has _ but not strict): {loose_format}")
print(f"  No underscore: {no_underscore}")

if loose_format > 0:
    print(f"\n  Examples of loose format:")
    count = 0
    for node in quantum_nodes:
        if not q_pattern.match(node) and '_' in node:
            print(f"    {node}")
            count += 1
            if count >= 5:
                break

if no_underscore > 0:
    print(f"\n  Examples of no underscore:")
    count = 0
    for node in quantum_nodes:
        if '_' not in node:
            print(f"    {node}")
            count += 1
            if count >= 5:
                break

# Test with small k and cap
k = 3
cap = 2  # Small capacity to make brute force feasible

print(f"\n{'='*70}")
print(f"Testing with k={k}, cap={cap}")
print(f"{'='*70}")

# Extract qubits
qubits = set()
for node in quantum_nodes:
    try:
        base = node.split('_')[0]
        idx = int(base[1:])
        qubits.add(idx)
    except:
        pass

print(f"\nUnique qubits: {len(qubits)}")
print(f"Qubits: {sorted(qubits)}")

# Run heuristic
print(f"\nRunning heuristic...")
try:
    heuristic_partitions, _ = compute_cut(hdh, k=k, cap=cap)
    heuristic_cost_raw = cost(hdh, heuristic_partitions)
    heuristic_cost = sum(heuristic_cost_raw) if isinstance(heuristic_cost_raw, tuple) else heuristic_cost_raw
    
    print(f"✓ Heuristic result:")
    print(f"  Cost (raw): {heuristic_cost_raw}")
    print(f"  Cost (summed): {heuristic_cost}")
    
    # Analyze heuristic partitions
    print(f"\n  Partition details:")
    for i, part in enumerate(heuristic_partitions):
        q_nodes = [n for n in part if hdh.sigma.get(n) == 'q']
        c_nodes = [n for n in part if hdh.sigma.get(n) == 'c']
        print(f"    Partition {i}: {len(part)} total nodes ({len(q_nodes)} quantum, {len(c_nodes)} classical)")
        
        # Count unique qubits
        qubits_in_part = set()
        for node in q_nodes:
            try:
                base = node.split('_')[0]
                idx = int(base[1:])
                qubits_in_part.add(idx)
            except:
                pass
        print(f"      Unique qubits: {len(qubits_in_part)} - {sorted(qubits_in_part)}")
    
except Exception as e:
    print(f"✗ Heuristic failed: {e}")
    import traceback
    traceback.print_exc()

# Now let's manually create a simple partition and check its cost
print(f"\n{'='*70}")
print(f"Manual partition test")
print(f"{'='*70}")

# Create a simple partition: put all nodes in partition 0
manual_partitions = [set(hdh.S), set(), set()]
manual_cost_raw = cost(hdh, manual_partitions)
manual_cost = sum(manual_cost_raw) if isinstance(manual_cost_raw, tuple) else manual_cost_raw

print(f"\nAll nodes in partition 0:")
print(f"  Cost (raw): {manual_cost_raw}")
print(f"  Cost (summed): {manual_cost}")

# Create another partition: distribute qubits evenly
print(f"\n\nEven qubit distribution test:")
manual_partitions2 = [set(), set(), set()]
node_to_qubit = {}
for node in quantum_nodes:
    try:
        base = node.split('_')[0]
        idx = int(base[1:])
        node_to_qubit[node] = idx
    except:
        pass

qubit_list = sorted(set(node_to_qubit.values()))
print(f"  Distributing {len(qubit_list)} qubits across {k} partitions...")

for node, qubit_idx in node_to_qubit.items():
    partition_id = qubit_idx % k
    manual_partitions2[partition_id].add(node)

# Add classical nodes to partition 0
for node in classical_nodes:
    manual_partitions2[0].add(node)

manual_cost2_raw = cost(hdh, manual_partitions2)
manual_cost2 = sum(manual_cost2_raw) if isinstance(manual_cost2_raw, tuple) else manual_cost2_raw

print(f"  Cost (raw): {manual_cost2_raw}")
print(f"  Cost (summed): {manual_cost2}")

for i, part in enumerate(manual_partitions2):
    q_nodes = [n for n in part if node in node_to_qubit]
    qubits_in_part = set(node_to_qubit[n] for n in q_nodes)
    print(f"  Partition {i}: {len(qubits_in_part)} unique qubits - {sorted(qubits_in_part)}")

print(f"\n{'='*70}")
print(f"Summary:")
print(f"  Heuristic cost: {heuristic_cost}")
print(f"  All-in-one cost: {manual_cost}")
print(f"  Even-distribution cost: {manual_cost2}")
print(f"{'='*70}")
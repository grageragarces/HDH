from qiskit import QuantumCircuit
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.qiskit import from_qiskit
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_size, compute_parallelism_by_time

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.ccx(1, 2, 0)
qc.measure_all()

hdh_graph = from_qiskit(qc)
# plot_hdh(hdh_graph)
# Partition HDH using cut
num_parts = 3
partitions = compute_cut(hdh_graph, num_parts)

print(f"\nMETIS partition into {num_parts} parts:")
for i, part in enumerate(partitions):
    print(f"Partition {i}: {sorted(part)}")
    
# --- Metrics ---
cut_cost = cost(hdh_graph, partitions)
sizes = partition_sizes(partitions)
global_parallelism = compute_parallelism_by_time(hdh_graph, partitions, mode="global")
parallelism_at_t3 = compute_parallelism_by_time(hdh_graph, partitions, mode="local", time_step=3)

print("\n--- qiskit Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")
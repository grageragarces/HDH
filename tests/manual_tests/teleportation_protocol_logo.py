from qiskit import QuantumCircuit
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

# Teleportation protocol between A & B
qc = QuantumCircuit(3, 2)

# Prepa EPR pair
qc.h(1)
qc.cx(1, 2)

# Bell measurement on A
qc.cx(0, 1)
qc.h(0)

# Measurement A
qc.measure(0, 0)
qc.measure(1, 1)

# B correction based on A
qc.cx(1, 2)
qc.cz(0, 2)

hdh = from_qiskit(qc)
# plot_hdh(hdh_graph)

# Partition HDH using cut
num_parts = 3
partitions = compute_cut(hdh, num_parts)

print(f"\nMETIS partition into {num_parts} parts:")
for i, part in enumerate(partitions):
    print(f"Partition {i}: {sorted(part)}")
    
# plot_hdh(hdh)
cut_cost = cost(hdh, partitions)
sizes = partition_sizes(partitions)
global_parallelism = compute_parallelism_by_time(hdh, partitions, mode="global")
parallelism_at_t3 = compute_parallelism_by_time(hdh, partitions, mode="local", time_step=3)

print("\n--- Teleportation protocol Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")
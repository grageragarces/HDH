import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.qasm_converter import from_qasm
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

qasm_path = os.path.join(os.path.dirname(__file__), 'test_qasm_file.qasm')
hdh_graph = from_qasm('file', qasm_path)

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

print("\n--- QASM Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qw import QW
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

qw = QW()

q0 = "q0"  # initial qubit
q1 = qw.add_coin(q0)         # returns e.g. "q1"
q2 = qw.add_shift(q1)        # returns e.g. "q2"
qw.add_measurement(q2, "c0")

hdh = qw.build_hdh()

print("NODES:")
for node_id in sorted(hdh.S):
    t = hdh.time_map[node_id]
    τ = hdh.sigma[node_id]
    print(f"  {node_id}: time={t}, τ={τ}")

print("\nHYPEREDGES:")
for i, edge in enumerate(sorted(hdh.C, key=lambda e: (min(hdh.time_map[n] for n in e), len(e)))):
    τ = hdh.tau[edge]
    print(f"  edge {i}: {set(edge)} -> τ={τ}")

# plot_hdh(hdh)
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

print("\n--- QW Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")
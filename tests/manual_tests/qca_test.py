import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qca import QCA
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

topology = {
    "q0": ["q1", "q2"],
    "q1": ["q0"],
    "q2": ["q0"]
}

measurements = {"q1", "q2"}

ca = QCA(topology=topology, measurements=measurements, steps=3)
hdh = ca.build_hdh()

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

print("\n--- QCA Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")
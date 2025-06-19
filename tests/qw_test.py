import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qw import QW
from hdh.visualize import plot_hdh

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

plot_hdh(hdh)

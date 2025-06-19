import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mbqc import MBQC
from hdh.visualize import plot_hdh

mbqc = MBQC()

# Sample MBQC pattern: Init q0, q1 → entangle q0-q1 → measure q0 → classically process q0 to q2
mbqc.add_operation("N", [], "q0")      # Init
mbqc.add_operation("N", [], "q1")
mbqc.add_operation("E", ["q0", "q1"], "q1")  # Entangle
mbqc.add_operation("M", ["q0"], "c0")        # Measure q0
mbqc.add_operation("C", ["c0"], "q2")        # Feedforward

hdh = mbqc.build_hdh()

# print("Nodes:")
# for node, (t, τ) in hdh.nodes.items():
#     print(f"  {node}: time={t}, τ={τ}")

# print("\nHyperedges:")
# for i, (σs, τ) in enumerate(hdh.hyperedges):
#     print(f"  Edge {i}: {σs} -> τ={τ}")

plot_hdh(hdh)

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qca import QCA
from hdh.visualize import plot_hdh

topology = {
    "q0": ["q1", "q2"],
    "q1": ["q0"],
    "q2": ["q0"]
}

measurements = {"q1", "q2"}

ca = QCA(topology=topology, measurements=measurements, steps=3)
hdh = ca.build_hdh()

# print("=== Nodes ===")
# for s in sorted(hdh.S):
#     print(s, hdh.sigma[s], hdh.time_map[s])

# print("\n=== Hyperedges ===")
# for edge in sorted(hdh.C, key=lambda e: sorted(e)):
#     print(edge, "type:", hdh.tau[edge])

# print("\n=== Classical Output Nodes ===")
# for s in sorted(hdh.S):
#     if hdh.sigma[s] == "c":
#         print(s)

plot_hdh(hdh)

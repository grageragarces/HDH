from typing import List, Tuple, Optional, Set, Dict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Quantum Cellular Automata (QCA) Model

class QCA:
    def __init__(self, topology, measurements, steps, hdh_cls=HDH):
        self.topology = topology
        self.measurements = measurements
        self.steps = steps
        self.hdh_cls = hdh_cls

    def build_hdh(self) -> HDH:
        hdh = self.hdh_cls()
        time_map = {node: 0 for node in self.topology}

        for t in range(1, self.steps + 1):
            for node, neighbors in self.topology.items():
                inputs = [f"{n}_t{time_map[n]}" for n in neighbors + [node]]
                for n in inputs:
                    hdh.add_node(n, "q", int(n.split("_t")[1]))

                out_node = f"{node}_t{t}"
                hdh.add_node(out_node, "q", t)
                hdh.add_hyperedge(frozenset(inputs + [out_node]), "q", name="update")
                time_map[node] = t

        # Add measurement edges
        for node in self.measurements:
            t_meas = self.steps + 1  # important!
            out_node = f"{node}_t{self.steps}"
            cl_index = int(node[1:])  # assumes "q0", "q1", etc.
            c_node = f"c{cl_index}_t{t_meas}"
            hdh.add_node(c_node, "c", t_meas)
            hdh.add_hyperedge(frozenset({out_node, c_node}), "c", name="measure")

        return hdh

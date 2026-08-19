from typing import List, Tuple, Optional, Set, Dict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Quantum Cellular Automata (QCA) Model

class QCA:
    """Quantum cellular automaton (QCA) builder: a fixed neighbor topology
    updated for a set number of steps, then optionally measured.

    Unlike the other models, a `QCA` is fully specified at construction time
    rather than built up incrementally — there's no `add_*` method, just
    `build_hdh`.

    Args:
        topology: Adjacency map from each qubit label (e.g. ``"q0"``) to the
            list of neighbor labels its update rule reads from.
        measurements: Qubit labels to measure at the final timestep.
        steps: Number of update steps to simulate.
        hdh_cls: HDH class to instantiate (override for a subclass).
    """

    def __init__(self, topology, measurements, steps, hdh_cls=HDH):
        self.topology = topology
        self.measurements = measurements
        self.steps = steps
        self.hdh_cls = hdh_cls

    def build_hdh(self) -> HDH:
        """Simulate `steps` update rounds, then measure, producing an HDH.

        At each timestep, every qubit gets one hyperedge connecting its own
        and its neighbors' previous-timestep nodes to its new-timestep node
        (i.e. its update rule). After all steps, each qubit named in
        `measurements` gets a measurement hyperedge to a classical output
        node one timestep later.

        Returns:
            HDH: the built hypergraph.
        """
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

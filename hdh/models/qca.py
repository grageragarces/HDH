from typing import List, Tuple, Optional, Set, Dict
import re
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
        topology: Adjacency map from each cell label (any string, e.g.
            ``"q0"`` or ``"A"``) to the list of neighbor labels its update
            rule reads from.
        measurements: Cell labels to measure at the final timestep.
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
                inputs = []
                for n in neighbors + [node]:
                    in_node = f"{n}_t{time_map[n]}"
                    hdh.add_node(in_node, "q", time_map[n])
                    inputs.append(in_node)

                out_node = f"{node}_t{t}"
                hdh.add_node(out_node, "q", t)
                hdh.add_hyperedge(frozenset(inputs + [out_node]), "q", name="update")
                time_map[node] = t

        # Add measurement edges
        for node in self.measurements:
            t_meas = self.steps + 1  # important!
            out_node = f"{node}_t{self.steps}"
            c_node = f"{self._classical_label(node)}_t{t_meas}"
            hdh.add_node(c_node, "c", t_meas)
            hdh.add_hyperedge(frozenset({out_node, c_node}), "c", name="measure")

        return hdh

    @staticmethod
    def _classical_label(cell: str) -> str:
        """Label for the classical bit a measured cell writes to.

        Cells named ``q<int>`` keep the circuit convention (``q3`` -> ``c3``)
        so converters recognise the bit; any other name gets a ``c_`` prefix
        (``"A"`` -> ``"c_A"``).
        """
        m = re.fullmatch(r"q(\d+)", cell)
        return f"c{m.group(1)}" if m else f"c_{cell}"

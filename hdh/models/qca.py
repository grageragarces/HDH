from typing import List, Tuple, Optional, Set, Dict
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Quantum Cellular Automata (QCA) Model

class QCA:
    """Quantum cellular automaton builder: cells updated from their neighbors for
    a number of steps, then optionally measured.

    Args:
        topology: Each cell label (any string) -> the neighbor labels its
            update reads.
        measurements: Cells to measure after the last step.
        steps: Number of update steps.
        hdh_cls: HDH class to instantiate, for subclasses.

    Example:
        >>> qca = QCA(topology={"A": ["B"], "B": ["A"]}, measurements={"A"}, steps=1)
        >>> sorted(qca.build_hdh().nodes)
        ['A_t0', 'A_t1', 'B_t0', 'B_t1', 'c_A_t2']
    """

    def __init__(self, topology, measurements, steps, hdh_cls=HDH):
        self.topology = topology
        self.measurements = measurements
        self.steps = steps
        self.hdh_cls = hdh_cls

    def build_hdh(self) -> HDH:
        """Build the HDH: per step, one hyperedge per cell joining its and its
        neighbors' previous states to its new state; then one measurement
        hyperedge per measured cell.
        """
        hdh = self.hdh_cls()
        time_map = {node: 0 for node in self.topology}

        for t in range(1, self.steps + 1):
            for node, neighbors in self.topology.items():
                inputs = []
                for n in neighbors + [node]:
                    inputs.append(hdh.add_node(n, time_map[n], "q"))

                out_node = hdh.add_node(node, t, "q")
                hdh.add_hyperedge(frozenset(inputs + [out_node]), "q", name="update")
                time_map[node] = t

        # Add measurement edges
        for node in self.measurements:
            t_meas = self.steps + 1  # important!
            out_node = hdh.node_id(node, self.steps)
            c_node = hdh.add_node(self._classical_label(node), t_meas, "c")
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

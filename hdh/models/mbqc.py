from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Measurement Based Quantum Computing (MBQC) model 

class MBQC:
    """Measurement-based quantum computing (MBQC) pattern builder.

    Record N (prepare), E (entangle), M (measure) and C (classical
    correction) operations with `add_operation`, then call `build_hdh`.
    Labels are free-form and become the HDH's wires; keep each label to one
    type (never reuse a measurement result's label as a quantum state).

    Examples:
        >>> m = MBQC()
        >>> m.add_operation("N", [], "a")
        >>> m.add_operation("N", [], "b")
        >>> m.add_operation("E", ["a", "b"], "b")
        >>> m.add_operation("M", ["a"], "m_a")
        >>> hdh = m.build_hdh()
        >>> sorted({hdh.wire_of[n] for n in hdh.nodes if hdh.node_types[n] == "q"})
        ['a', 'b']
    """

    def __init__(self, hdh_cls=HDH):
        self.pattern = []  # (op_type, A, b)
        self.hdh_cls = hdh_cls

    def add_operation(self, op_type: str, A: List[str], b: str):
        """Append one N, E, M or C operation.

        Args:
            op_type: ``"N"``, ``"E"``, ``"M"`` or ``"C"`` (case-insensitive).
            A: Labels the operation reads; empty for ``"N"``.
            b: Label it produces. For ``"E"``, reuse one of the entangled labels.
        """
        self.pattern.append((op_type.upper(), A, b))

    def build_hdh(self) -> HDH:
        """Translate the recorded pattern into an HDH, one timestep per operation."""
        hdh = self.hdh_cls()
        time_map = {}
        current_time = 0

        for op_type, A, b in self.pattern:
            in_nodes = set()
            out_nodes = set()
            all_nodes = A + [b]

            # Assign time steps
            op_time = current_time
            current_time += 1

            for x in A:
                t = time_map.get(x, 0)
                in_nodes.add(hdh.add_node(x, t, self._node_type(op_type, input=True)))

            out_nodes.add(hdh.add_node(b, op_time, self._node_type(op_type, input=False)))
            time_map[b] = op_time

            edge_nodes = in_nodes | out_nodes
            hdh.add_hyperedge(edge_nodes, self._edge_type(op_type), name=op_type.lower())

        return hdh

    def _node_type(self, op_type, input=False):
        """Map an NEMC op type + input/output position to a `sigma` value."""
        if op_type == "N":
            return "c" if input else "q"
        if op_type == "E":
            return "q"
        if op_type == "M":
            return "q" if input else "c"
        if op_type == "C":
            return "c"

    def _edge_type(self, op_type):
        """Map an NEMC op type to a `tau` value: only E is quantum."""
        return "q" if op_type == "E" else "c"

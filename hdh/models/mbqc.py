from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Measurement Based Quantum Computing (MBQC) model 

class MBQC:
    """Measurement-based quantum computing (MBQC) pattern builder.

    Patterns are sequences of NEMC operations — N (auxiliary state
    preparation), E (entanglement), M (measurement), C (classical correction)
    — recorded via `add_operation` and translated into an HDH by `build_hdh`.

    Unlike `hdh.models.circuit.Circuit`, node labels here are caller-chosen
    strings rather than auto-derived from a qubit index: MBQC nodes don't
    necessarily correspond 1:1 with qubits, so there's no automatic
    ``q_``/``c_`` naming. By convention, though, a label's type must stay
    consistent across every operation that references it (e.g. always use
    ``"c0"`` for a classical output, never reuse it as a quantum input) — HDH
    node IDs still encode their `sigma` type via prefix, and `HDH.add_node`
    raises if the same label is reused with a different inferred type.
    """

    def __init__(self, hdh_cls=HDH):
        self.pattern = []  # (op_type, A, b)
        self.hdh_cls = hdh_cls

    def add_operation(self, op_type: str, A: List[str], b: str):
        """Append one NEMC operation to the pattern.

        Args:
            op_type: One of ``"N"``, ``"E"``, ``"M"``, ``"C"`` (case-insensitive).
            A: Input node label(s) the operation reads. Empty for ``"N"``
                (it has no inputs, only produces `b`).
            b: Output node label the operation produces or measures. For
                ``"E"`` (entanglement), reuse one of the entangled nodes'
                existing labels rather than introducing a new one.
        """
        self.pattern.append((op_type.upper(), A, b))

    def build_hdh(self) -> HDH:
        """Translate the recorded NEMC pattern into an HDH.

        Each operation gets its own timestep, in recording order. Node type
        (quantum vs. classical) is inferred per operation via `_node_type`:
        N produces a quantum output from a classical placeholder input, E is
        purely quantum, M consumes a quantum input and produces a classical
        output, and C is purely classical.

        Returns:
            HDH: the built hypergraph.
        """
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
                hdh.add_node(f"{x}_t{t}", self._node_type(op_type, input=True), t)
                in_nodes.add(f"{x}_t{t}")

            hdh.add_node(f"{b}_t{op_time}", self._node_type(op_type, input=False), op_time)
            out_nodes.add(f"{b}_t{op_time}")
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

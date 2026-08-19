from typing import Dict
import sys
import os
import cirq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdh.hdh import HDH
from hdh.models.circuit import Circuit

# -------- helpers --------

def _qubit_index_map(c: cirq.Circuit) -> Dict[cirq.Qid, int]:
    """Assign each qubit in `c` a stable integer index, in sorted order.

    Cirq qubits (`GridQubit`, `LineQubit`, `NamedQubit`, ...) don't share a
    common ordering, so this sorts by whichever positional attributes a given
    qubit type has (grid row/col, line x, etc.), falling back to `repr` —
    giving a deterministic index assignment regardless of qubit type.
    """
    def _key(q):
        return (
            getattr(q, "row", None),
            getattr(q, "col", None),
            getattr(q, "x", None),
            getattr(q, "y", None),
            getattr(q, "qubit_index", None),
            repr(q),
        )
    ordered = sorted(c.all_qubits(), key=_key)
    return {q: i for i, q in enumerate(ordered)}

def _gate_name(gate: cirq.Gate) -> str:
    """Map a Cirq gate to its HDH/Qiskit-style lower-case name.

    Recognised power gates (`HPowGate`, `XPowGate`, `CNotPowGate`, etc.) map
    to their standard gate names regardless of exponent; anything else falls
    back to the gate class's own name, lower-cased.
    """
    if isinstance(gate, cirq.HPowGate):
        return "h"
    if isinstance(gate, cirq.XPowGate):
        return "x"
    if isinstance(gate, cirq.YPowGate):
        return "y"
    if isinstance(gate, cirq.ZPowGate):
        return "z"
    if isinstance(gate, cirq.CNotPowGate):
        return "cx"
    if isinstance(gate, cirq.CZPowGate):
        return "cz"
    if isinstance(gate, cirq.SwapPowGate):
        return "swap"
    if isinstance(gate, cirq.MeasurementGate):
        return "measure"
    return gate.__class__.__name__.lower()

def _is_measure(op: cirq.Operation) -> bool:
    """Return whether `op` is a Cirq measurement."""
    return isinstance(op.gate, cirq.MeasurementGate)

# -------- main --------

def from_cirq(c: cirq.Circuit) -> HDH:
    """Convert a Cirq circuit to an HDH via `hdh.models.circuit.Circuit`.

    Supports standard gates and measurement, moment by moment. Classically
    conditioned gates (Cirq's equivalent of Qiskit's `IfElseOp`) aren't
    supported.

    Args:
        c: The Cirq circuit to convert.

    Returns:
        HDH: the converted circuit.
    """
    circuit = Circuit()
    qmap = _qubit_index_map(c)

    for moment in c:
        for op in moment.operations:
            if _is_measure(op):
                q_indices = [qmap[q] for q in op.qubits]
                circuit.add_instruction("measure", q_indices, None)
                continue

            name = _gate_name(op.gate)
            q_indices = [qmap[q] for q in op.qubits]
            modifies_flags = [True] * len(q_indices)
            circuit.add_instruction(name, q_indices, None, modifies_flags)

    return circuit.build_hdh()
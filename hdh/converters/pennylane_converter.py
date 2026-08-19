from typing import Dict, List, Any, Union
import warnings

import pennylane as qml
from pennylane.tape import QuantumScript, OperationRecorder
from pennylane.ops.op_math import Conditional
from pennylane.measurements import MidMeasureMP,ProbabilityMP, ExpectationMP, SampleMP

from hdh.hdh import HDH
from hdh.models.circuit import Circuit


# ---------- wire helpers ----------

def _wire_index_map(qs: QuantumScript) -> Dict[Any, int]:
    """Map each of `qs`'s wire labels (arbitrary, per PennyLane) to a
    contiguous 0..n-1 qubit index, in wire-declaration order."""
    return {w: i for i, w in enumerate(qs.wires)}

def _mk_op(name: str, params: List[Any], wires: List[int]):
    """Build a PennyLane operation from an HDH-style gate name.

    Not currently called anywhere in this module — a `to_pennylane`
    (HDH -> PennyLane) reconstruction path hasn't been implemented yet;
    this is what its gate lookup would use.
    """
    n = name.lower()
    # minimal gate LUT; extend as needed
    if n in {"h", "hadamard"}:
        return qml.Hadamard(wires=wires[0])
    if n == "x":
        return qml.PauliX(wires=wires[0])
    if n == "y":
        return qml.PauliY(wires=wires[0])
    if n == "z":
        return qml.PauliZ(wires=wires[0])
    if n == "rx":
        return qml.RX(params[0] if params else 0.0, wires=wires[0])
    if n == "ry":
        return qml.RY(params[0] if params else 0.0, wires=wires[0])
    if n == "rz":
        return qml.RZ(params[0] if params else 0.0, wires=wires[0])
    if n in {"cx", "cnot"}:
        return qml.CNOT(wires=wires[:2])
    if n == "cz":
        return qml.CZ(wires=wires[:2])
    if n == "swap":
        return qml.SWAP(wires=wires[:2])
    # fall back: Identity to keep place without failing
    warnings.warn(f"[to_pennylane] Unknown/unsupported gate '{name}', inserting Identity.")
    return qml.Identity(wires=wires[0])

# ---------- from_pennylane ----------

def from_pennylane(circ_like: Union[QuantumScript, OperationRecorder]) -> HDH:
    """Convert a PennyLane `QuantumScript`/`OperationRecorder` to an HDH.

    Supports standard gates, mid-circuit measurements (`MidMeasureMP` and
    the `ProbabilityMP`/`ExpectationMP`/`SampleMP` terminal measurements,
    each treated as a "measure" instruction), and single-condition `qml.cond`
    blocks (via PennyLane's `Conditional` operator).

    Note: since PennyLane wires can be arbitrary labels (not necessarily
    small contiguous integers), they're remapped via `_wire_index_map`
    before being handed to `Circuit`. The resulting HDH's hyperedges are
    correct, but for multi-qubit gates the resulting node layout may not
    visually resemble the equivalent circuit built directly from small
    integer wire indices (e.g. via `from_qiskit`).

    Args:
        circ_like: The PennyLane script/recorder to convert.

    Returns:
        HDH: the converted circuit.

    Raises:
        NotImplementedError: If a `qml.cond` condition isn't
            `MeasurementValue`-based.
    """
    qs = circ_like
    wire2idx = _wire_index_map(qs)

    circuit = Circuit()

    # Track which mid-measure maps to which classical bit index
    meas_mp_to_cbit: Dict[MidMeasureMP, int] = {}
    next_cbit = 0

    for op in qs.operations:
        # mid-circuit measure -> explicit "measure" instruction
        if isinstance(op, (MidMeasureMP,ProbabilityMP, ExpectationMP, SampleMP)):
            w = op.wires[0]
            cbit = meas_mp_to_cbit.setdefault(op, next_cbit)
            if cbit == next_cbit:
                next_cbit += 1
            circuit.add_instruction("measure", [wire2idx[w]], [cbit])
            continue

        # conditional op (qml.cond -> Conditional container)
        if isinstance(op, Conditional):
            then = op.base
            mval = op.meas_val
            mps = getattr(mval, "measurements", None)
            if not mps:
                raise NotImplementedError("Only MeasurementValue-based conditions are supported.")
            mp = mps[0]  # single-bit condition
            cbit = meas_mp_to_cbit.setdefault(mp, next_cbit)
            if cbit == next_cbit:
                next_cbit += 1

            qidxs = [wire2idx[w] for w in then.wires]
            circuit.add_instruction(
                then.name.lower(),
                qidxs,
                bits=[cbit],
                modifies_flags=[True] * len(qidxs),
                cond_flag="p",
            )
            continue

        # plain operation
        name = op.name.lower()
        qidxs = [wire2idx[w] for w in op.wires]
        circuit.add_instruction(name, qidxs, bits=[])

    return circuit.build_hdh()

from typing import Dict, List, Any, Optional, Set, Union
import re
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

_Q_RE = re.compile(r'^q(\d+)_t\d+$')
_C_RE = re.compile(r'^c(\d+)_t\d+$')

def _parse_qubit(node_id: str) -> Optional[int]:
    """Return the qubit index encoded in a node id like ``q3_t7``, else None."""
    m = _Q_RE.match(node_id)
    return int(m.group(1)) if m else None

def _parse_cbit(node_id: str) -> Optional[int]:
    """Return the classical bit index encoded in a node id like ``c3_t7``, else None."""
    m = _C_RE.match(node_id)
    return int(m.group(1)) if m else None

def _mk_op(name: str, params: List[Any], wires: List[int]):
    """Build a PennyLane operation from an HDH-style gate name.

    Constructing the returned op enqueues it on whatever queuing context is
    currently active (e.g. inside a `qml.tape.make_qscript`-wrapped function),
    which is how `to_pennylane` uses this.
    """
    n = name.lower()
    # minimal gate LUT; extend as needed
    if n in {"h", "hadamard"}:
        return qml.Hadamard(wires=wires[0])
    if n in {"x", "paulix"}:
        return qml.PauliX(wires=wires[0])
    if n in {"y", "pauliy"}:
        return qml.PauliY(wires=wires[0])
    if n in {"z", "pauliz"}:
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
            then_params = [float(p) for p in then.parameters] if then.parameters else None
            circuit.add_instruction(
                then.name.lower(),
                qidxs,
                bits=[cbit],
                modifies_flags=[True] * len(qidxs),
                cond_flag="p",
                params=then_params,
            )
            continue

        # plain operation
        name = op.name.lower()
        qidxs = [wire2idx[w] for w in op.wires]
        params = [float(p) for p in op.parameters] if op.parameters else None
        circuit.add_instruction(name, qidxs, bits=[], params=params)

    return circuit.build_hdh()


# ---------- to_pennylane ----------

def to_pennylane(hdh: HDH) -> QuantumScript:
    """Convert an HDH back into a PennyLane `QuantumScript`.

    Mirrors `hdh.converters.qiskit_converter.to_qiskit`: multi-qubit gates
    are stored in the HDH as three hyperedges (``<name>_stage1``, ``_stage2``,
    ``_stage3``); only the ``_stage2`` edge carries the full gate, so stages
    1 and 3 are skipped during reconstruction.

    Unlike `to_qiskit`, this doesn't build the script by appending to a
    circuit object — mid-circuit measurement and `qml.cond` both need to be
    *executed* (not just described) to enqueue correctly, so the HDH is
    replayed as a sequence of real `qml.measure`/`qml.cond`/gate calls inside
    a `qml.tape.make_qscript`-wrapped function, the same mechanism PennyLane
    itself uses to build a script from a Python function.

    Args:
        hdh: HDH object

    Returns:
        QuantumScript: PennyLane representation.
    """
    qubit_indices: Set[int] = set()
    for node_id in hdh.S:
        q = _parse_qubit(node_id)
        if q is not None:
            qubit_indices.add(q)

    qubit_map = {q: i for i, q in enumerate(sorted(qubit_indices))}

    if not qubit_map:
        return qml.tape.make_qscript(lambda: None)()

    # (sort_time, name, wires, cbit, is_cond, params)
    records: List[tuple] = []

    for edge in hdh.C:
        raw_name = hdh.gate_name.get(edge, "")
        edge_type = hdh.tau.get(edge, "q")

        if raw_name == "measure":
            q_nodes = sorted(
                (n for n in edge if hdh.sigma.get(n) == "q"),
                key=lambda n: hdh.time_map.get(n, 0),
            )
            c_nodes = sorted(
                (n for n in edge if hdh.sigma.get(n) == "c"),
                key=lambda n: hdh.time_map.get(n, 0),
            )
            if not q_nodes or not c_nodes:
                continue
            q_idx = _parse_qubit(q_nodes[0])
            c_idx = _parse_cbit(c_nodes[0])
            if q_idx is None or c_idx is None or q_idx not in qubit_map:
                continue
            sort_time = hdh.time_map.get(c_nodes[0], 0)
            records.append((sort_time, "measure", [qubit_map[q_idx]], c_idx, False, None))
            continue

        # Stages 1 and 3 only carry wire continuity; stage 2 holds the real gate.
        if raw_name.endswith("_stage1") or raw_name.endswith("_stage3"):
            continue

        # Classical edges of a conditional gate duplicate its quantum edge.
        if edge_type == "c":
            continue

        args = hdh.edge_args.get(edge)
        if args is None:
            continue

        q_with_time, c_with_time, _ = args
        actual_name = raw_name[:-7] if raw_name.endswith("_stage2") else raw_name
        wires = [qubit_map[q] for q, _ in q_with_time if q in qubit_map]
        sort_time = min((t for _, t in q_with_time), default=0)
        is_cond = hdh.phi.get(edge) == "p"
        cbit = c_with_time[0][0] if (is_cond and c_with_time) else None

        records.append((sort_time, actual_name, wires, cbit, is_cond, hdh.gate_params.get(edge)))

    records.sort(key=lambda r: r[0])

    # A multi-qubit gate contributes one record per qubit wire; keep one.
    seen: Set[tuple] = set()
    unique_records: List[tuple] = []
    for rec in records:
        sort_time, name, wires, cbit, is_cond, params = rec
        dedup_key = (sort_time, name, tuple(wires))
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_records.append(rec)

    def _replay():
        cbit_to_mp: Dict[int, Any] = {}
        for sort_time, name, wires, cbit, is_cond, params in unique_records:
            if name == "measure":
                cbit_to_mp[cbit] = qml.measure(wires[0])
                continue

            if is_cond and cbit in cbit_to_mp:
                qml.cond(cbit_to_mp[cbit], lambda n=name, p=params, w=wires: _mk_op(n, p, w))()
            else:
                if is_cond:
                    warnings.warn(
                        f"[to_pennylane] Conditional gate '{name}' at t={sort_time} has no "
                        f"matching measurement - applying unconditionally."
                    )
                _mk_op(name, params, wires)

    return qml.tape.make_qscript(_replay)()

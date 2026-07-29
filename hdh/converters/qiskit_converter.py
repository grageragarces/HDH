from __future__ import annotations

from qiskit import QuantumCircuit, ClassicalRegister
import sys
import os
from typing import List, Optional, Set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH
from qiskit.circuit import Clbit
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import (
    HGate, IGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    RXGate, RYGate, RZGate, U1Gate, U2Gate, U3Gate,
    CXGate, CYGate, CZGate, CHGate,
    SwapGate, iSwapGate,
    CCXGate, CSGate, CSwapGate,
    PhaseGate, CPhaseGate,
)
import re

from hdh.models.circuit import Circuit

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

def _bit_index_from_cond_target(qc, target):
    """
    Extract the classical bit index from a condition target.
    
    Args:
        qc: QuantumCircuit
        target: Can be Clbit or ClassicalRegister
        
    Returns:
        int: Index of the classical bit
        
    Raises:
        NotImplementedError: If target type is unsupported or register has multiple bits
    """
    if isinstance(target, Clbit):
        return qc.clbits.index(target)
    if isinstance(target, ClassicalRegister):
        if len(target) != 1:
            raise NotImplementedError("Only 1-bit ClassicalRegister conditions are supported.")
        return qc.clbits.index(target[0])
    raise NotImplementedError(f"Unsupported condition target type: {type(target)}")

def _process_if_else_op(qc, instr, circuit):
    """
    Process an IfElseOp instruction and add it to the HDH circuit.
    
    Uses the new add_conditional_gate method for cleaner, more maintainable code.
    
    Args:
        qc: QuantumCircuit
        instr: IfElseOp instance
        circuit: HDH Circuit object
        
    Raises:
        NotImplementedError: For unsupported condition types or values
    """
    # Extract condition - can be tuple or expr.Expr
    cond = instr.condition
    
    # Handle tuple condition: (Clbit|ClassicalRegister, int)
    if isinstance(cond, tuple):
        target, val = cond
        
        # Only support condition value == 1 for now
        if int(val) != 1:
            raise NotImplementedError(
                f"Only IfElseOp conditions == 1 are supported, got {val}"
            )
        
        bit_index = _bit_index_from_cond_target(qc, target)
        
        # Process true_body (blocks[0]) - executes when condition == 1
        if len(instr.blocks) > 0:
            true_body = instr.blocks[0]
            for inner_instr, inner_qargs, inner_cargs in true_body.data:
                # Skip metadata instructions
                if inner_instr.name in {"barrier", "snapshot", "delay", "label"}:
                    continue
                    
                inner_qidx = [qc.qubits.index(q) for q in inner_qargs]
                
                # Use add_conditional_gate for cleaner code
                if len(inner_qidx) == 1:
                    # Single-qubit gate
                    circuit.add_conditional_gate(
                        classical_bit=bit_index,
                        target_qubit=inner_qidx[0],
                        gate_name=inner_instr.name
                    )
                else:
                    # Multi-qubit gate
                    circuit.add_conditional_gate(
                        classical_bit=bit_index,
                        target_qubit=inner_qidx[0],
                        gate_name=inner_instr.name,
                        additional_qubits=inner_qidx[1:]
                    )
        
        # Process false_body (blocks[1]) - executes when condition == 0
        # Note: Currently uses add_instruction with cond_flag="n" since
        # add_conditional_gate doesn't support negated conditions yet
        if len(instr.blocks) > 1 and instr.blocks[1] is not None:
            false_body = instr.blocks[1]
            for inner_instr, inner_qargs, inner_cargs in false_body.data:
                # Skip metadata instructions
                if inner_instr.name in {"barrier", "snapshot", "delay", "label"}:
                    continue
                    
                inner_qidx = [qc.qubits.index(q) for q in inner_qargs]
                inner_cidx = [qc.clbits.index(c) for c in inner_cargs] if inner_cargs else []
                
                # Use add_instruction with negated condition for else block
                # TODO: Consider adding add_conditional_gate_negated() method
                circuit.add_instruction(
                    inner_instr.name,
                    inner_qidx,
                    bits=[bit_index],
                    modifies_flags=[True] * len(inner_qidx),
                    cond_flag="n"  # Negated condition
                )
    else:
        # For expr.Expr conditions, would need additional handling
        raise NotImplementedError(
            f"Expression-based conditions are not yet supported: {type(cond)}"
        )

def from_qiskit(qc: QuantumCircuit) -> HDH:
    """
    Convert a Qiskit QuantumCircuit to HDH format.
    
    Supports:
    - Standard gates (h, rx, cx, measure, etc.)
    - IfElseOp with single-bit conditions == 1
    - Both Clbit and single-bit ClassicalRegister conditions
    
    Args:
        qc: Qiskit QuantumCircuit
        
    Returns:
        HDH: HDH representation of the circuit
        
    Raises:
        NotImplementedError: For unsupported operations or condition types
    """
    circuit = Circuit()

    for instr, qargs, cargs in qc.data:
        # Skip metadata instructions
        if instr.name in {"barrier", "snapshot", "delay", "label"}:
            continue

        # Get indices
        q_indices = [qc.qubits.index(q) for q in qargs]
        c_indices = [qc.clbits.index(c) for c in cargs]

        # Handle IfElseOp
        if isinstance(instr, IfElseOp):
            _process_if_else_op(qc, instr, circuit)
            continue

        # Handle standard instructions
        if instr.name == "measure":
            circuit.add_instruction("measure", q_indices, None)  
        else:
            modifies_flags = [True] * len(q_indices)
            circuit.add_instruction(instr.name, q_indices, c_indices, modifies_flags)

    return circuit.build_hdh()

def _make_gate(name: str, params: Optional[List] = None):
    """
    Build a Qiskit gate object from an HDH gate name.

    Args:
        name: Lower-case gate name as stored in ``hdh.gate_name``
        params: Rotation angles, when the HDH carries them

    Returns:
        Gate instance, or None if the name is not recognised.
    """
    p = params or []

    _FIXED = {
        "h":       HGate(),
        "i":       IGate(),
        "id":      IGate(),
        "x":       XGate(),
        "y":       YGate(),
        "z":       ZGate(),
        "s":       SGate(),
        "sdg":     SdgGate(),
        "t":       TGate(),
        "tdg":     TdgGate(),
        "sx":      SXGate(),
        "sxdg":    SXdgGate(),
        "cx":      CXGate(),
        "cnot":    CXGate(),
        "cy":      CYGate(),
        "cz":      CZGate(),
        "ch":      CHGate(),
        "cs":      CSGate(),
        "swap":    SwapGate(),
        "iswap":   iSwapGate(),
        "ccx":     CCXGate(),
        "toffoli": CCXGate(),
        "cswap":   CSwapGate(),
        "fredkin": CSwapGate(),
    }

    if name in _FIXED:
        return _FIXED[name]

    theta = p[0] if p else 0.0
    phi   = p[1] if len(p) > 1 else 0.0
    lam   = p[2] if len(p) > 2 else 0.0

    _PARAMETRIC = {
        "rx":     RXGate(theta),
        "ry":     RYGate(theta),
        "rz":     RZGate(theta),
        "p":      PhaseGate(theta),
        "phase":  PhaseGate(theta),
        "u1":     U1Gate(lam),
        "u2":     U2Gate(phi, lam),
        "u3":     U3Gate(theta, phi, lam),
        "u":      U3Gate(theta, phi, lam),
        "cp":     CPhaseGate(theta),
        "cphase": CPhaseGate(theta),
    }

    if name in _PARAMETRIC:
        if not p:
            print(
                f"[WARNING] Gate '{name}' is parametric but no parameters were stored "
                f"in the HDH. Defaulting to theta=phi=lambda=0."
            )
        return _PARAMETRIC[name]

    return None


def to_qiskit(hdh: HDH) -> QuantumCircuit:
    """
    Convert an HDH back into a Qiskit QuantumCircuit.

    Multi-qubit gates are stored in the HDH as three hyperedges (``<name>_stage1``,
    ``_stage2``, ``_stage3``); only the ``_stage2`` edge carries the full qubit set,
    so the stage 1 and 3 wire-continuity edges are skipped during reconstruction.

    Args:
        hdh: HDH object

    Returns:
        QuantumCircuit: Qiskit representation

    Note:
        Gate parameters (rotation angles) are not preserved by the HDH
        representation and default to 0 on reconstruction.
    """
    qubit_indices: Set[int] = set()
    bit_indices:   Set[int] = set()

    for node_id in hdh.S:
        q = _parse_qubit(node_id)
        if q is not None:
            qubit_indices.add(q)
            continue
        c = _parse_cbit(node_id)
        if c is not None:
            bit_indices.add(c)

    n_qubits = max(qubit_indices) + 1 if qubit_indices else 0
    n_bits   = max(bit_indices)   + 1 if bit_indices   else 0

    if n_qubits == 0:
        return QuantumCircuit()

    qc = QuantumCircuit(n_qubits, n_bits) if n_bits > 0 else QuantumCircuit(n_qubits)

    records: List[tuple] = []

    for edge in hdh.C:
        raw_name  = hdh.gate_name.get(edge, "")
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
            if q_idx is None or c_idx is None:
                continue
            sort_time = hdh.time_map.get(c_nodes[0], 0)
            records.append((sort_time, "measure", [q_idx], [c_idx], False))
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
        q_indices   = [q for q, _ in q_with_time]
        c_indices   = [c for c, _ in c_with_time]
        sort_time   = min((t for _, t in q_with_time), default=0)
        is_cond     = hdh.phi.get(edge) == "p"

        records.append((sort_time, actual_name, q_indices, c_indices, is_cond))

    records.sort(key=lambda r: r[0])

    # A multi-qubit gate contributes one record per qubit wire; keep one.
    seen: Set[tuple] = set()
    unique_records: List[tuple] = []
    for rec in records:
        sort_time, name, q_indices, c_indices, is_cond = rec
        dedup_key = (sort_time, name, tuple(q_indices))
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_records.append(rec)

    for sort_time, name, q_indices, c_indices, is_cond in unique_records:
        if name == "measure":
            for qi, ci in zip(q_indices, c_indices):
                qc.measure(qi, ci)
            continue

        gate = _make_gate(name)
        if gate is None:
            print(f"[WARNING] Unrecognised gate '{name}' at t={sort_time} on qubits {q_indices} - skipped.")
            continue

        if any(qi >= n_qubits or qi < 0 for qi in q_indices):
            print(f"[WARNING] Gate '{name}' references out-of-range qubits {q_indices} - skipped.")
            continue

        if gate.num_qubits != len(q_indices):
            print(f"[WARNING] Gate '{name}' expects {gate.num_qubits} qubits but got {len(q_indices)} - skipped.")
            continue

        if is_cond and c_indices:
            ctrl_bit = c_indices[0]
            if ctrl_bit >= n_bits:
                print(f"[WARNING] Conditional gate '{name}' references out-of-range classical bit {ctrl_bit} - applied unconditionally.")
                qc.append(gate, q_indices)
            else:
                with qc.if_test((qc.clbits[ctrl_bit], 1)):
                    qc.append(gate, q_indices)
        else:
            qc.append(gate, q_indices)

    return qc


def _project_hdh(hdh: HDH, node_set: Set[str]) -> HDH:
    """
    Restrict an HDH to ``node_set``, keeping only hyperedges fully inside it.

    Hyperedges that straddle the boundary are dropped: those are the cut edges
    that a full DQC implementation replaces with teledata/telegate primitives.
    """
    sub = HDH()

    for nid in node_set:
        if nid not in hdh.S:
            continue
        sub.add_node(nid, hdh.sigma[nid], hdh.time_map[nid], node_real=hdh.upsilon.get(nid, "a"))

    for edge in hdh.C:
        if not edge.issubset(node_set):
            continue
        sub.add_hyperedge(
            set(edge),
            hdh.tau[edge],
            name=hdh.gate_name.get(edge),
            node_real=hdh.phi.get(edge, "a"),
            role=hdh.edge_role.get(edge),
        )
        if edge in hdh.edge_args:
            sub.edge_args[edge] = hdh.edge_args[edge]

    return sub


def partitions_to_qiskit(hdh: HDH, partitions: List[Set[str]]) -> List[QuantumCircuit]:
    """
    Recover one Qiskit QuantumCircuit per partition of a cut HDH.

    Args:
        hdh: The HDH that was partitioned
        partitions: Node sets as returned by ``hdh.passes.cut.compute_cut``

    Returns:
        One QuantumCircuit per partition, containing only the gates whose
        qubits are entirely local to that partition.
    """
    return [to_qiskit(_project_hdh(hdh, node_set)) for node_set in partitions]

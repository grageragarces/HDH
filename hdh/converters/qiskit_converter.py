from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH
from models.circuit import Circuit
from qiskit.circuit import Instruction, InstructionSet, Measure, Reset, Clbit
from qiskit.circuit.controlflow import IfElseOp
import re

from hdh.models.circuit import Circuit

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

def to_qiskit(hdh) -> QuantumCircuit:
    """
    Convert HDH format back to Qiskit QuantumCircuit.
    
    Note: No longer compatible with from_qiskit due to HDH representation changes.
    
    Args:
        hdh: HDH object
        
    Returns:
        QuantumCircuit: Qiskit representation
    """

    def resolve_qidxs(raw_q, anc_q, expected_len, edge, name):
        seen = set()
        deduped = []
        anc_pool = list(dict.fromkeys(anc_q))

        for q in raw_q:
            if q in seen:
                if not anc_pool:
                    raise ValueError(f"Edge {edge} ({name}) needs more ancillas to resolve duplicates.")
                deduped.append(anc_pool.pop(0))
            else:
                deduped.append(q)
                seen.add(q)

        remaining_anc = [a for a in anc_pool if a not in seen]
        combined = deduped + remaining_anc

        if len(set(combined)) < len(combined):
            raise ValueError(f"Edge {edge} ({name}) still has duplicate qubits after resolution: {combined}")

        return combined[:expected_len]

    # Creates global contiguous index maps
    q_nodes = sorted(n for n in hdh.S if hdh.sigma[n] == 'q')
    c_nodes = sorted(n for n in hdh.S if hdh.sigma[n] == 'c')

    node_to_qidx = {n: i for i, n in enumerate(q_nodes)}
    node_to_cidx = {n: i for i, n in enumerate(c_nodes)}

    # Also include ancilla nodes from motifs
    if hasattr(hdh, "motifs"):
        for motif in hdh.motifs.values():
            for n in motif.get("ancilla_qubits", []):
                if hdh.sigma.get(n) == 'q' and n not in node_to_qidx:
                    node_to_qidx[n] = len(node_to_qidx)
            for n in motif.get("ancilla_bits", []):
                if hdh.sigma.get(n) == 'c' and n not in node_to_cidx:
                    node_to_cidx[n] = len(node_to_cidx)

    qr = QuantumRegister(len(node_to_qidx), 'q')
    cr = ClassicalRegister(len(node_to_cidx), 'c')
    qc = QuantumCircuit(qr, cr)

    found_telegate = False
    found_teledata = False

    for edge in sorted(hdh.C, key=lambda e: hdh.edge_metadata.get(e, {}).get("timestep", 0)):
        meta = hdh.edge_metadata.get(edge, {})
        name = hdh.gate_name.get(edge, "unknown")
        raw_q_idxs = [node_to_qidx[q] for q in meta.get("qubits", []) if q in node_to_qidx]
        c_idxs = [node_to_cidx[c] for c in meta.get("cbits", []) if c in node_to_cidx]

        anc_qidxs = []
        anc_cidxs = []
        if edge in getattr(hdh, "motifs", {}):
            motif = hdh.motifs[edge]
            anc_qidxs = [node_to_qidx[n] for n in motif.get("ancilla_qubits", []) if n in node_to_qidx]
            anc_cidxs = [node_to_cidx[n] for n in motif.get("ancilla_bits", []) if n in node_to_cidx]

        anc_qidxs = [a for a in anc_qidxs if a not in raw_q_idxs]
        anc_cidxs = [c for c in anc_cidxs if c not in c_idxs]
        c_idxs += anc_cidxs

        sub = hdh.edge_args.get(edge)

        if sub is None:
            gate = meta.get("gate")
            params = meta.get("params", [])
            if gate == "h":
                qc.h(qr[raw_q_idxs[0]])
            elif gate == "rx":
                qc.rx(params[0] if params else 0.5, qr[raw_q_idxs[0]])
            elif gate == "cx":
                qc.cx(qr[raw_q_idxs[0]], qr[raw_q_idxs[1]])
            elif gate == "measure":
                if not raw_q_idxs or not c_idxs:
                    continue
                qc.measure(qr[raw_q_idxs[0]], cr[c_idxs[0]])
            else:
                continue

        try:
            if isinstance(sub, InstructionSet):
                if len(sub.instructions) != 1:
                    raise ValueError(f"InstructionSet in edge {edge} has multiple instructions.")
                single_inst = sub.instructions[0]
                inst = single_inst[0] if isinstance(single_inst, tuple) else single_inst
            elif isinstance(sub, (Instruction, Measure, Reset)):
                inst = sub
            elif hasattr(sub, "to_instruction"):
                inst = sub.to_instruction()
            else:
                raise TypeError(f"Unsupported edge_args type for edge {edge}: {type(sub)}")
        except Exception as e:
            raise RuntimeError(f"Failed to resolve instruction for edge {edge} ({name}): {e}") from e

        q_idxs = resolve_qidxs(raw_q_idxs, anc_qidxs, inst.num_qubits, edge, name)
        c_idxs = c_idxs[:inst.num_clbits]

        if len(set(q_idxs)) < len(q_idxs):
            raise ValueError(f"Edge {edge} ({name}) has duplicate qubit indices after slicing: {q_idxs}")
        if len(set(c_idxs)) < len(c_idxs):
            raise ValueError(f"Edge {edge} ({name}) has duplicate classical indices: {c_idxs}")

        qubits = [qr[i] for i in q_idxs]
        clbits = [cr[i] for i in c_idxs]

        if name == 'measure':
            if not qubits or not clbits:
                print(f"[ERROR] Cannot apply measure on edge {edge}: missing qubit or cbit idxs")
                continue
            qc.measure(qubits[0], clbits[0])
        elif isinstance(sub, QuantumCircuit):
            if name == 'telegate':
                found_telegate = True
            if name == 'teledata':
                found_teledata = True
            for g in sub.data:
                gate, qargs, cargs = g
                qidxs = [qr[q._index] for q in qargs]
                cidxs = [cr[c._index] for c in cargs] if cargs else []
                qc.append(gate, qidxs, cidxs)
        else:
            if name == 'telegate':
                found_telegate = True
            if name == 'teledata':
                found_teledata = True
            qc.append(inst, qubits, clbits)

    if not found_telegate and not found_teledata:
        print("[WARNING] No communication primitives (telegate/teledata) appended!")

    num_teledata = sum(1 for name in hdh.gate_name.values() if name == 'teledata')
    num_telegate = sum(1 for name in hdh.gate_name.values() if name == 'telegate')
    return qc
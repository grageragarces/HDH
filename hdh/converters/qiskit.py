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
    # target can be Clbit or ClassicalRegister(1)
    if isinstance(target, Clbit):
        return qc.clbits.index(target)
    if isinstance(target, ClassicalRegister):
        if len(target) != 1:
            raise NotImplementedError("Only 1-bit ClassicalRegister conditions are supported.")
        return qc.clbits.index(target[0])
    raise NotImplementedError(f"Unsupported condition target type: {type(target)}")

def from_qiskit(qc: QuantumCircuit) -> HDH:
    circuit = Circuit()

    for instr, qargs, cargs in qc.data:
        if instr.name in {"barrier", "snapshot", "delay", "label"}:
            continue

        q_indices = [qc.qubits.index(q) for q in qargs]
        c_indices = [qc.clbits.index(c) for c in cargs]

        # Handles conditionals
        if isinstance(instr, IfElseOp):
            cond = instr.condition  # (Clbit|ClassicalRegister, int)
            target, val = cond
            if int(val) != 1:
                raise NotImplementedError("Only IfElseOp conditions == 1 are supported.")
            bit_index = _bit_index_from_cond_target(qc, target)

            true_body = instr.blocks[0]
            for inner_instr, inner_qargs, _ in true_body.data:
                inner_qidx = [qc.qubits.index(q) for q in inner_qargs]
                circuit.add_instruction(
                    inner_instr.name,
                    inner_qidx,
                    bits=[bit_index],
                    modifies_flags=[True]*len(inner_qidx),
                    cond_flag="p"
                )
            continue

        # Handles standard instructions
        if instr.name == "measure":
            circuit.add_instruction("measure", q_indices, None)  
        else:
            modifies_flags = [True] * len(q_indices)
            circuit.add_instruction(instr.name, q_indices, c_indices, modifies_flags)

    return circuit.build_hdh()

def to_qiskit(hdh) -> QuantumCircuit:

    """
    No longer compatible with from_qiskit
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

    print(f"[INFO] Allocating QuantumRegister({len(node_to_qidx)}), ClassicalRegister({len(node_to_cidx)})")
    print("[DEBUG] node_to_qidx:", node_to_qidx)
    print("[DEBUG] node_to_cidx:", node_to_cidx)

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
                print(f"[DEBUG] Appending h on edge {edge}")
                qc.h(qr[raw_q_idxs[0]])
            elif gate == "rx":
                print(f"[DEBUG] Appending rx on edge {edge}")
                qc.rx(params[0] if params else 0.5, qr[raw_q_idxs[0]])
            elif gate == "cx":
                print(f"[DEBUG] Appending cx on edge {edge}")
                qc.cx(qr[raw_q_idxs[0]], qr[raw_q_idxs[1]])
            elif gate == "measure":
                if not raw_q_idxs or not c_idxs:
                    print(f"[ERROR] Cannot apply measure on edge {edge}: missing qubit or cbit idxs")
                    continue
                print(f"[DEBUG] Appending measure on edge {edge}")
                qc.measure(qr[raw_q_idxs[0]], cr[c_idxs[0]])
            else:
                print(f"[DEBUG] Appending unknown on edge {edge}")
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
            print(f"[DEBUG] Appending measure on edge {edge}")
            qc.measure(qubits[0], clbits[0])
        elif isinstance(sub, QuantumCircuit):
            print(f"[DEBUG] Appending {name} on edge {edge} (circuit with {len(sub.data)} ops)")
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
            print(f"[DEBUG] Appending {name} on edge {edge} (inst)")
            if name == 'telegate':
                found_telegate = True
            if name == 'teledata':
                found_teledata = True
            qc.append(inst, qubits, clbits)

    if not found_telegate and not found_teledata:
        print("[WARNING] No communication primitives (telegate/teledata) appended!")

    num_teledata = sum(1 for name in hdh.gate_name.values() if name == 'teledata')
    num_telegate = sum(1 for name in hdh.gate_name.values() if name == 'telegate')
    print(f"[DEBUG] teledata count: {num_teledata}, telegate count: {num_telegate}")
    print(f"[DEBUG] Final circuit has {qc.num_qubits} qubits and {qc.count_ops()} total ops")
    return qc

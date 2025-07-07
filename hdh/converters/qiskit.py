from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Set
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH
from models.circuit import Circuit
from collections import defaultdict,Counter
from qiskit.circuit import Instruction, InstructionSet, Measure, Reset
import re

def from_qiskit(qc: QuantumCircuit) -> HDH: 
    hdh = HDH()

    qstates = {}
    cstates = {}
    for i, q in enumerate(qc.qubits):
        name = f"q{i}_t0"
        hdh.add_node(name, 'q', 0)
        qstates[q] = name

    for i, c in enumerate(qc.clbits):
        name = f"c{i}_t0"
        hdh.add_node(name, 'c', 0)
        cstates[c] = name

    timestep = 1
    for instr, qargs, cargs in qc.data:
        if instr.name == "measure":
            q = qargs[0]
            c = cargs[0]

            q_old = qstates[q]
            q_new = f"q{qc.qubits.index(q)}_t{timestep}"
            hdh.add_node(q_new, 'q', timestep)

            c_old = cstates.get(c, f"c{qc.clbits.index(c)}_t0")
            c_new = f"c{qc.clbits.index(c)}_t{timestep}"
            hdh.add_node(c_new, 'c', timestep)
            cstates[c] = c_new

            edge = hdh.add_hyperedge({q_old, q_new, c_old, c_new}, edge_type='c', name='measure')

            hdh.edge_args[edge] = QuantumCircuit(1,1).measure(0,0)
            hdh.gate_name[edge] = 'measure'
            hdh.edge_metadata[edge] = {
                "gate": instr.name,
                "qubits": [qc.qubits.index(q)],
                "cbits": [qc.clbits.index(c)],
                "timestep": timestep
            }

            qstates[q] = q_new
            timestep += 1
            continue

        old = [qstates[q] for q in qargs]
        new = [f"q{qc.qubits.index(q)}_t{timestep}" for q in qargs]

        for node in new:
            hdh.add_node(node, 'q', timestep)

        edge_nodes = set(old + new)
        edge = hdh.add_hyperedge(edge_nodes, edge_type='q', name=instr.name)

        sub_qc = QuantumCircuit(len(qargs))
        sub_qc.append(instr, list(range(len(qargs))))
        hdh.edge_args[edge] = sub_qc
        hdh.gate_name[edge] = instr.name
        hdh.edge_metadata[edge] = {
            "gate": instr.name,
            "qubits": [qc.qubits.index(q) for q in qargs],
            "params": instr.params,
            "timestep": timestep
        }

        for i, q in enumerate(qargs):
            qstates[q] = new[i]

        timestep += 1

    print(f"[DEBUG] Total gates in from_qiskit: {len(hdh.C)}")
    return hdh

def to_qiskit(hdh) -> QuantumCircuit:
    def extract_index_from_node(n: str) -> int:
        m = re.search(r'(?:q|c)[A-Za-z_]*?(\d+)', n)
        if not m:
            raise ValueError(f"Cannot extract index from: {n}")
        return int(m.group(1))

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

    qubit_indices = set()
    cbit_indices = set()

    for node in hdh.S:
        idx = extract_index_from_node(node)
        if hdh.sigma[node] == 'q':
            qubit_indices.add(idx)
        elif hdh.sigma[node] == 'c':
            cbit_indices.add(idx)

    for meta in hdh.edge_metadata.values():
        qubit_indices.update(meta.get("qubits", []))
        cbit_indices.update(meta.get("cbits", []))

    if hasattr(hdh, "motifs"):
        for motif in hdh.motifs.values():
            qubit_indices.update([
                extract_index_from_node(n)
                for n in motif.get("ancilla_qubits", [])
                if hdh.sigma.get(n) == "q"
            ])
            cbit_indices.update([
                extract_index_from_node(n)
                for n in motif.get("ancilla_bits", [])
                if hdh.sigma.get(n) == "c"
            ])

    max_q = max(qubit_indices) if qubit_indices else 0
    max_c = max(cbit_indices) if cbit_indices else 0
    print(f"[INFO] Allocating QuantumRegister({max_q+1}), ClassicalRegister({max_c+1})")

    qr = QuantumRegister(max_q + 1, 'q')
    cr = ClassicalRegister(max_c + 1, 'c')
    qc = QuantumCircuit(qr, cr)

    found_telegate = False
    found_teledata = False

    for edge in sorted(hdh.C, key=lambda e: hdh.edge_metadata.get(e, {}).get("timestep", 0)):
        meta = hdh.edge_metadata.get(edge, {})
        name = hdh.gate_name.get(edge, "unknown")
        role = meta.get("role")
        raw_q_idxs = list(meta.get("qubits", []))
        c_idxs = list(meta.get("cbits", []))

        anc_qidxs = []
        anc_cidxs = []
        if edge in getattr(hdh, "motifs", {}):
            motif = hdh.motifs[edge]
            anc_qidxs = [
                extract_index_from_node(n)
                for n in motif.get("ancilla_qubits", [])
                if hdh.sigma.get(n) == "q"
            ]
            anc_cidxs = [
                extract_index_from_node(n)
                for n in motif.get("ancilla_bits", [])
                if hdh.sigma.get(n) == "c"
            ]

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
            print(f"[DEBUG] Appending measure on edge {edge}")
            qc.measure(qubits[0], clbits[0])
        elif isinstance(sub, QuantumCircuit):
            print(f"[DEBUG] Appending {name} on edge {edge} (circuit with {len(sub.data)} ops)")
            if role == 'telegate':
                found_telegate = True
            if role == 'teledata':
                found_teledata = True
            for g in sub.data:
                gate, qargs, cargs = g
                qidxs = [qr[q._index] for q in qargs]
                cidxs = [cr[c._index] for c in cargs] if cargs else []
                qc.append(gate, qidxs, cidxs)
        else:
            print(f"[DEBUG] Appending {name} on edge {edge} (inst)")
            if role == 'telegate':
                found_telegate = True
            if role == 'teledata':
                found_teledata = True
            qc.append(inst, qubits, clbits)

    if not found_telegate and not found_teledata:
        print("[WARNING] No communication primitives (telegate/teledata) appended!")

    num_teledata = sum(1 for meta in hdh.edge_metadata.values() if meta.get("role") == 'teledata')
    num_telegate = sum(1 for meta in hdh.edge_metadata.values() if meta.get("role") == 'telegate')
    print(f"[DEBUG] teledata count: {num_teledata}, telegate count: {num_telegate}")
    print(f"[DEBUG] Final circuit has {qc.num_qubits} qubits and {qc.count_ops()} total ops")
    return qc
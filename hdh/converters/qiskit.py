from qiskit import QuantumCircuit
from typing import Set
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH
from models.circuit import Circuit

def from_qiskit(circuit: QuantumCircuit) -> HDH:
    CONTROL_GATES = {
        "cx": [0],
        "ccx": [0, 1],
    }

    qc_adapter = Circuit()

    for instruction in circuit.data:
        gate = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        name = gate.name.lower()

        if name in {"barrier", "snapshot", "delay", "label"}:
            continue

        q_indices = [circuit.find_bit(q).index for q in qargs]
        c_indices = [circuit.find_bit(c).index for c in cargs]

        if name == "measure":
            modifies_flags = [False] * len(q_indices)
        elif name == "reset":
            modifies_flags = [True] * len(q_indices)
        else:
            ctrl = CONTROL_GATES.get(name, list(range(len(q_indices) - 1)))
            modifies_flags = [i not in ctrl for i in range(len(q_indices))]

        qc_adapter.add_instruction(name, q_indices, c_indices, modifies_flags)

    return qc_adapter.build_hdh()

def to_qiskit(hdh: HDH, model: str = "circuit") -> Circuit:
    if model != "circuit":
        raise NotImplementedError(f"Model '{model}' not supported yet.")

    from collections import defaultdict

    circuit = Circuit()

    # Sort hyperedges by timestamp of earliest output node
    def edge_time(edge):
        return min(hdh.time_map[n] for n in edge)

    sorted_edges = sorted(hdh.C, key=edge_time)

    for edge in sorted_edges:
        name = hdh.gate_name.get(edge, None)
        if not name or name in {"teleport", "entanglement"}:
            continue  # skip comm primitives or unknown gates

        qargs, cargs, modifies_flags = hdh.edge_args[edge]
        qubit_indices = [q for q, _ in qargs]
        clbit_indices = [c for c, _ in cargs]

        circuit.add_instruction(name, qubit_indices, clbit_indices, modifies_flags)

        for node in edge:
            if node.startswith("q"):
                q_idx = int(node.split("_")[0][1:])
                t = hdh.time_map[node]
                qargs.append((q_idx, t))
            elif node.startswith("c"):
                c_idx = int(node.split("_")[0][1:])
                t = hdh.time_map[node]
                cargs.append((c_idx, t))

        # Group by qubit index, keep only latest (output) and earliest (input)
        q_by_idx = defaultdict(list)
        for q_idx, t in qargs:
            q_by_idx[q_idx].append(t)
    
    return circuit

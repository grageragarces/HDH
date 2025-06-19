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
"""
This module is still under development and is subject to change.
It currently implements a limited amount of quantum communication primitives for Qiskit circuits.
The goal is to provide SDK agnostic primitives that can be translated from the Circuit class to other SDKs.
This work is still ongoing.
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.circuit.library import XGate

def teledata() -> QuantumCircuit:
    """
    Standard quantum teleportation protocol.
    Qubit layout:
      q[0] — input to be teleported (qin)
      q[1] — half of Bell pair (sender side)
      q[2] — half of Bell pair (receiver side)
    Classical layout:
      c[0], c[1] — measurement bits
    """
    qc = QuantumCircuit(3, 2, name="teledata")
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.cx(2, 1)
    qc.cz(2, 0)
    return qc

def telegate(gate: Instruction = XGate()) -> QuantumCircuit:
    """
    Generic remote gate application using gate teleportation.
    Qubit layout:
      q[0] — control/input qubit
      q[1] — target qubit where gate will appear
      q[2] — cat/mediator qubit
    Classical layout:
      c[0], c[1] — measurement outcomes
    """
    qc = QuantumCircuit(3, 2, name="telegate")
    qc.cx(0, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(2, 1)
    qc.append(gate, [1])
    return qc

DEFAULT_PRIMITIVES = {
    "tp": teledata,
    "cat": telegate
}

def implement_comm_primitives(partitioned_hdhs, model="circuit", sdk="qiskit", primitives=None):
    if model != "circuit" or sdk != "qiskit":
        raise NotImplementedError("Only circuit + qiskit is currently supported.")
    
    primitives = primitives or DEFAULT_PRIMITIVES
    circuits = []

    for hdh in partitioned_hdhs:
        qc = QuantumCircuit(hdh.get_num_qubits())
        
        for edge in hdh.C:
            if hdh.edge_role.get(edge) in {"teledata", "telegate"}:
                label = hdh.gate_name.get(edge)
                if label in primitives:
                    # call the primitive circuit
                    subcircuit = primitives[label](0, 1, 0)  # qubit indices placeholder
                    required_qubits = max(qc.num_qubits, subcircuit.num_qubits)

                    if len(qc.qubits) < required_qubits:
                        # Extend the circuit with additional qubits
                        extra = required_qubits - len(qc.qubits)
                        qc.add_register(QuantumRegister(extra))

                    qc.append(subcircuit.to_instruction(), qc.qubits[:subcircuit.num_qubits])
        
        circuits.append(qc)

    return circuits
  
# =====================================================================
#                        PRIMITIVES
# =====================================================================
# The following block contains “primitives / rewrite” scaffolding and helpers.

# class AncillaAllocator:
#     def __init__(self):
#         self.counter = 0
#     def new(self, base: str, time: int):
#         name = f"{base}_anc{self.counter}_t{time}"
#         self.counter += 1
#         return name

# def extract_qidx(n):
#     m = re.search(r'q(?:[A-Za-z_]*?)(\d+)', n)
#     if m:
#         return int(m.group(1))
#     raise ValueError(f"[ERROR] extract_qidx failed on: {n}")

# def extract_cidx(n):
#     m = re.search(r'c(?:[A-Za-z_]*?)(\d+)', n)
#     if m:
#         return int(m.group(1))
#     raise ValueError(f"[ERROR] extract_cidx failed on: {n}")

# def get_logical_qubit(node_id: str) -> str:
#     return node_id.split('_')[0]

# def select_comm_primitive(role, node_type, allowed):
#     if role == "teledata":
#         return "tp" if "tp" in allowed["quantum"] else "cat"
#     elif role == "telegate":
#         return "cat"
#     elif role == "classical":
#         return "ccom" if "ccom" in allowed["classical"] else "crep"
#     raise ValueError(f"Unknown role: {role}")

# def cut_and_rewrite_hdh(...):
#     ...

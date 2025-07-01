from qiskit import QuantumCircuit
from hdh.converters.qiskit import from_qiskit  # your existing converter

def from_qasm(input_type: str, qasm: str):
    if input_type == 'file':
        circuit = QuantumCircuit.from_qasm_file(qasm)
    elif input_type == 'string':
        circuit = QuantumCircuit.from_qasm_str(qasm)
    else:
        raise ValueError("Unsupported type. Use 'file' or 'string'.")
    
    return from_qiskit(circuit)

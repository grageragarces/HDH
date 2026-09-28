from qiskit import QuantumCircuit
from hdh.converters.qiskit_converter import from_qiskit  # your existing converter

def from_qasm(input_type: str, qasm: str):
    """Convert OpenQASM 2.0 to an HDH, via Qiskit's parser and `from_qiskit`.

    Args:
        input_type: ``"file"`` if `qasm` is a path, ``"string"`` if it is
            QASM source.
        qasm: The path or source.

    Raises:
        ValueError: If `input_type` is neither.

    Example:
        >>> src = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0], q[1];'
        >>> from_qasm("string", src).get_num_qubits()
        2
    """
    if input_type == 'file':
        circuit = QuantumCircuit.from_qasm_file(qasm)
    elif input_type == 'string':
        circuit = QuantumCircuit.from_qasm_str(qasm)
    else:
        raise ValueError("Unsupported type. Use 'file' or 'string'.")
    
    return from_qiskit(circuit)

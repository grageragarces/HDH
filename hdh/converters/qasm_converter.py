from qiskit import QuantumCircuit
from hdh.converters.qiskit_converter import from_qiskit  # your existing converter

def from_qasm(input_type: str, qasm: str):
    """Convert an OpenQASM 2.0 circuit to an HDH.

    Loads the circuit via Qiskit's QASM parser, then converts it exactly as
    `hdh.converters.qiskit_converter.from_qiskit` would.

    Args:
        input_type: `"file"` to load `qasm` as a path to a `.qasm` file, or
            `"string"` to parse `qasm` directly as QASM source.
        qasm: File path or QASM source, per `input_type`.

    Returns:
        HDH: the converted circuit.

    Raises:
        ValueError: If `input_type` isn't `"file"` or `"string"`.
    """
    if input_type == 'file':
        circuit = QuantumCircuit.from_qasm_file(qasm)
    elif input_type == 'string':
        circuit = QuantumCircuit.from_qasm_str(qasm)
    else:
        raise ValueError("Unsupported type. Use 'file' or 'string'.")
    
    return from_qiskit(circuit)

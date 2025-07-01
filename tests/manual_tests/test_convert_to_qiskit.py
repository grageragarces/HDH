from qiskit import QuantumCircuit
from hdh.converters import from_qiskit, to_qiskit  

def test_roundtrip():
    # Original Qiskit circuit
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(1, 0)

    # Qiskit → HDH
    hdh = from_qiskit(qc)

    # HDH → Circuit
    circuit = to_qiskit(hdh, model="circuit")

    # Expected instruction structure
    expected = [
        ('h', [0], [], [True]),
        ('cx', [0, 1], [], [False, True]),
        ('measure', [1], [0], [False]),
    ]

    assert circuit.instructions == expected, f"Mismatch:\n{circuit.instructions}\n!=\n{expected}"
    print("✅ Round-trip test passed.")

if __name__ == "__main__":
    test_roundtrip()

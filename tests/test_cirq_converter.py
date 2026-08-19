import pytest

cirq = pytest.importorskip("cirq")

from hdh.converters.cirq_converter import from_cirq


class TestCirqConverter:
    def test_single_qubit_gate(self):
        q0 = cirq.LineQubit(0)
        c = cirq.Circuit([cirq.H(q0)])

        hdh = from_cirq(c)

        assert hdh.get_num_qubits() == 1
        assert any(hdh.gate_name.get(e) == "h" for e in hdh.C)

    def test_two_qubit_gate(self):
        q0, q1 = cirq.LineQubit.range(2)
        c = cirq.Circuit([cirq.CNOT(q0, q1)])

        hdh = from_cirq(c)

        assert hdh.get_num_qubits() == 2
        gate_names = set(hdh.gate_name.values())
        assert "cx_stage2" in gate_names

    def test_measurement(self):
        q0 = cirq.LineQubit(0)
        c = cirq.Circuit([cirq.H(q0), cirq.measure(q0, key="m")])

        hdh = from_cirq(c)

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) == 1
        measure_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "measure"]
        assert len(measure_edges) == 1

    def test_named_qubits_get_stable_indices(self):
        # Cirq qubits aren't inherently orderable across types; the converter
        # must still produce a deterministic 0..n-1 assignment.
        a, b = cirq.NamedQubit("a"), cirq.NamedQubit("b")
        c = cirq.Circuit([cirq.CNOT(a, b)])

        hdh1 = from_cirq(c)
        hdh2 = from_cirq(c)

        assert hdh1.get_num_qubits() == hdh2.get_num_qubits() == 2

    def test_bell_circuit_qubits_start_at_t0(self):
        q0, q1 = cirq.LineQubit.range(2)
        c = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

        hdh = from_cirq(c)

        assert any(n == "q0_t0" for n in hdh.S)
        assert any(n == "q1_t0" for n in hdh.S)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

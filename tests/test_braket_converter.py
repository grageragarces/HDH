import pytest

braket_circuits = pytest.importorskip("braket.circuits")
from braket.circuits import Circuit as BraketCircuit

from hdh.converters.braket_converter import from_braket


class TestBraketConverter:
    def test_single_qubit_gate(self):
        c = BraketCircuit().h(0)

        hdh = from_braket(c)

        assert hdh.get_num_qubits() == 1
        assert any(hdh.gate_name.get(e) == "h" for e in hdh.C)

    def test_two_qubit_gate(self):
        c = BraketCircuit().cnot(0, 1)

        hdh = from_braket(c)

        assert hdh.get_num_qubits() == 2
        assert "cnot_stage2" in set(hdh.gate_name.values())

    def test_measurement(self):
        c = BraketCircuit().h(0).measure(0)

        hdh = from_braket(c)

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) == 1

    def test_bell_circuit_qubits_start_at_t0(self):
        c = BraketCircuit().h(0).cnot(0, 1)

        hdh = from_braket(c)

        assert any(n == "q0_t0" for n in hdh.S)
        assert any(n == "q1_t0" for n in hdh.S)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

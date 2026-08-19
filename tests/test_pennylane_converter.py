import pytest

qml = pytest.importorskip("pennylane")

from hdh.converters.pennylane_converter import from_pennylane, to_pennylane


class TestPennyLaneConverter:
    def test_single_qubit_gate(self):
        def circuit():
            qml.Hadamard(0)

        qs = qml.tape.make_qscript(circuit)()
        hdh = from_pennylane(qs)

        assert hdh.get_num_qubits() == 1
        assert any(hdh.gate_name.get(e) == "hadamard" for e in hdh.C)

    def test_two_qubit_gate(self):
        def circuit():
            qml.CNOT([0, 1])

        qs = qml.tape.make_qscript(circuit)()
        hdh = from_pennylane(qs)

        assert hdh.get_num_qubits() == 2
        assert "cnot_stage2" in set(hdh.gate_name.values())

    def test_mid_circuit_measurement(self):
        def circuit():
            qml.Hadamard(0)
            qml.measure(0)

        qs = qml.tape.make_qscript(circuit)()
        hdh = from_pennylane(qs)

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) == 1

    def test_qml_cond_conditional_gate(self):
        # Regression test: from_pennylane used to crash on qml.cond blocks
        # with AttributeError('Conditional' object has no attribute
        # 'then_op') against current PennyLane (op.then_op was renamed to
        # op.base). Verified against pennylane 0.45.1.
        def circuit():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(1)

        qs = qml.tape.make_qscript(circuit)()
        hdh = from_pennylane(qs)

        gate_names = set(hdh.gate_name.values())
        assert "measure" in gate_names
        assert "paulix" in gate_names
        assert hdh.get_num_qubits() == 2

    def test_arbitrary_wire_labels_get_contiguous_indices(self):
        def circuit():
            qml.CNOT(wires=["a", "b"])

        qs = qml.tape.make_qscript(circuit)()
        hdh = from_pennylane(qs)

        assert hdh.get_num_qubits() == 2


class TestToPennylane:
    def test_roundtrip_simple_gates(self):
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])

        qs = qml.tape.make_qscript(circuit)()
        qs2 = to_pennylane(from_pennylane(qs))

        assert [type(op).__name__ for op in qs2.operations] == ["Hadamard", "CNOT"]

    def test_roundtrip_preserves_rotation_params(self):
        def circuit():
            qml.RX(1.2345, wires=0)

        qs = qml.tape.make_qscript(circuit)()
        qs2 = to_pennylane(from_pennylane(qs))

        assert qs2.operations[0].parameters[0] == pytest.approx(1.2345)

    def test_roundtrip_preserves_conditional_gate(self):
        # PauliX is a regression case: _mk_op's lookup table only recognised
        # the short alias "x", not PennyLane's own op.name "PauliX" (lower-
        # cased "paulix"), so this used to silently fall back to Identity.
        def circuit():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(1)

        qs = qml.tape.make_qscript(circuit)()
        qs2 = to_pennylane(from_pennylane(qs))

        names = [type(op).__name__ for op in qs2.operations]
        assert names == ["Hadamard", "MidMeasure", "Conditional"]
        assert type(qs2.operations[2].base).__name__ == "PauliX"

    def test_roundtrip_is_idempotent(self):
        def circuit():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(1)

        qs = qml.tape.make_qscript(circuit)()
        qs2 = to_pennylane(from_pennylane(qs))
        qs3 = to_pennylane(from_pennylane(qs2))

        names2 = [type(op).__name__ for op in qs2.operations]
        names3 = [type(op).__name__ for op in qs3.operations]
        assert names2 == names3

    def test_to_pennylane_on_empty_hdh(self):
        qs_empty = qml.tape.make_qscript(lambda: None)()
        qs2 = to_pennylane(from_pennylane(qs_empty))

        assert qs2.operations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

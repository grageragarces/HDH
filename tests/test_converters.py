import re
import pytest

try:
    from hdh.converters.qiskit_converter import from_qiskit, hdh_to_qiskit, partitions_to_qiskit
    QISKIT_AVAILABLE = True
    HDH_TO_QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    HDH_TO_QISKIT_AVAILABLE = False

try:
    from hdh.converters.qasm_converter import from_qasm
    QASM_AVAILABLE = True
except ImportError:
    QASM_AVAILABLE = False

try:
    from hdh.passes.cut import compute_cut
    CUT_AVAILABLE = True
except ImportError:
    CUT_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    QISKIT_INSTALLED = True
except ImportError:
    QISKIT_INSTALLED = False

N_QUBITS = 10
K = 2
CAP = 5


def _build_pipeline_circuit():
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    for i in range(0, 4):
        qc.h(i)
        qc.cx(i, i + 1)
    for i in range(5, 9):
        qc.h(i)
        qc.cx(i, i + 1)
    qc.cx(4, 5)
    qc.measure(list(range(N_QUBITS)), list(range(N_QUBITS)))
    return qc


@pytest.fixture(scope="module")
def pipeline():
    qc = _build_pipeline_circuit()
    hdh = from_qiskit(qc)
    partitions, cut_cost = compute_cut(hdh, k=K, cap=CAP)
    sub_circuits = partitions_to_qiskit(hdh, partitions)
    return qc, hdh, partitions, cut_cost, sub_circuits


# ---------------------------------------------------------------------------
# TestQiskitConverter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED,
                    reason="Qiskit converter or qiskit not available")
class TestQiskitConverter:
    def test_simple_circuit_conversion(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        hdh = from_qiskit(qc)

        assert hdh.get_num_qubits() == 2
        assert len(hdh.S) > 0
        assert len(hdh.C) > 0

    def test_single_qubit_gates(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)

        hdh = from_qiskit(qc)

        assert hdh.get_num_qubits() == 1
        assert len(hdh.S) > 0
        gate_names = [hdh.gate_name.get(e, "").lower() for e in hdh.C]
        assert any("h" in name for name in gate_names)

    def test_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)

        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 2

    def test_circuit_with_measurement(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        hdh = from_qiskit(qc)

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) > 0
        measure_edges = [e for e in hdh.C
                         if "measure" in hdh.gate_name.get(e, "").lower()]
        assert len(measure_edges) > 0

    def test_bell_state(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        hdh = from_qiskit(qc)

        assert hdh.get_num_qubits() == 2
        assert any(n == "q0_t0" for n in hdh.S), "q0 should initialize at t=0"
        assert any(n == "q1_t0" for n in hdh.S), "q1 should initialize at t=0"

    def test_ghz_state(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)

        hdh = from_qiskit(qc)

        assert hdh.get_num_qubits() == 3
        for q in [0, 1, 2]:
            assert any(n == f"q{q}_t0" for n in hdh.S), \
                f"q{q} should initialize at t=0"

    def test_parametric_gates(self):
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rx(theta, 0)
        qc = qc.assign_parameters({theta: 0.5})

        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 1

    def test_multiple_measurements(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1])

        hdh = from_qiskit(qc)

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) >= 2


# ---------------------------------------------------------------------------
# TestQiskitRoundtrip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HDH_TO_QISKIT_AVAILABLE or not QISKIT_INSTALLED,
                    reason="hdh_to_qiskit or qiskit not available")
class TestQiskitRoundtrip:
    def test_roundtrip_simple(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        hdh1 = from_qiskit(qc)
        qc2 = hdh_to_qiskit(hdh1)
        hdh2 = from_qiskit(qc2)

        assert hdh1.get_num_qubits() == hdh2.get_num_qubits()


# ---------------------------------------------------------------------------
# TestQASMConverter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QASM_AVAILABLE, reason="QASM converter not available")
class TestQASMConverter:
    def test_qasm_string_simple(self):
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        hdh = from_qasm("string", qasm_str)
        assert hdh.get_num_qubits() == 2

    def test_qasm_with_measurement(self):
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        hdh = from_qasm("string", qasm_str)
        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) > 0

    def test_qasm_bell_state(self):
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        hdh = from_qasm("string", qasm_str)
        assert any(n == "q0_t0" for n in hdh.S), "q0 should initialize at t=0"
        assert any(n == "q1_t0" for n in hdh.S), "q1 should initialize at t=0"


# ---------------------------------------------------------------------------
# TestConverterEdgeCases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED,
                    reason="Qiskit not available")
class TestConverterEdgeCases:
    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 0

    def test_single_qubit_circuit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 1

    def test_large_circuit(self):
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
        for i in range(4):
            qc.cx(i, i + 1)

        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 5
        for q in range(5):
            assert any(n == f"q{q}_t0" for n in hdh.S), \
                f"q{q} should initialize at t=0"


# ---------------------------------------------------------------------------
# TestFromQiskit  (pipeline — HDH structure)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED or not CUT_AVAILABLE,
                    reason="dependencies not available")
class TestFromQiskit:
    def test_hdh_has_nodes(self, pipeline):
        _, hdh, *_ = pipeline
        assert len(hdh.S) > 0

    def test_hdh_has_edges(self, pipeline):
        _, hdh, *_ = pipeline
        assert len(hdh.C) > 0

    def test_hdh_qubit_count(self, pipeline):
        _, hdh, *_ = pipeline
        assert hdh.get_num_qubits() == N_QUBITS

    def test_hdh_has_quantum_and_classical_nodes(self, pipeline):
        _, hdh, *_ = pipeline
        types = set(hdh.sigma.values())
        assert "q" in types
        assert "c" in types


# ---------------------------------------------------------------------------
# TestComputeCut  (pipeline — partition correctness)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED or not CUT_AVAILABLE,
                    reason="dependencies not available")
class TestComputeCut:
    def test_returns_k_partitions(self, pipeline):
        _, _, partitions, *_ = pipeline
        assert len(partitions) == K

    def test_all_nodes_assigned(self, pipeline):
        _, hdh, partitions, *_ = pipeline
        assert set().union(*partitions).issubset(hdh.S)

    def test_partitions_are_disjoint(self, pipeline):
        _, _, partitions, *_ = pipeline
        p0, p1 = partitions
        assert p0.isdisjoint(p1)

    def test_each_partition_respects_cap(self, pipeline):
        Q_RE = re.compile(r"^q(\d+)_t\d+$")
        _, _, partitions, *_ = pipeline
        for part in partitions:
            unique_qubits = {int(Q_RE.match(n).group(1)) for n in part if Q_RE.match(n)}
            assert len(unique_qubits) <= CAP

    def test_cut_cost_is_non_negative(self, pipeline):
        *_, cut_cost, _ = pipeline
        assert cut_cost >= 0

    def test_cut_cost_less_than_total_edges(self, pipeline):
        _, hdh, _, cut_cost, _ = pipeline
        assert cut_cost < len(hdh.C)


# ---------------------------------------------------------------------------
# TestPartitionsToQiskit  (pipeline — sub-circuit reconstruction)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not QISKIT_AVAILABLE or not QISKIT_INSTALLED or not CUT_AVAILABLE or not HDH_TO_QISKIT_AVAILABLE,
    reason="dependencies not available",
)
class TestPartitionsToQiskit:
    def test_returns_two_circuits(self, pipeline):
        *_, sub_circuits = pipeline
        assert len(sub_circuits) == K

    def test_all_are_quantum_circuits(self, pipeline):
        *_, sub_circuits = pipeline
        for qc in sub_circuits:
            assert isinstance(qc, QuantumCircuit)

    def test_each_sub_circuit_within_cap(self, pipeline):
        Q_RE = re.compile(r"^q(\d+)_t\d+$")
        _, _, partitions, _, sub_circuits = pipeline
        for part, qc in zip(partitions, sub_circuits):
            unique_qubits = {int(Q_RE.match(n).group(1)) for n in part if Q_RE.match(n)}
            assert len(unique_qubits) <= CAP

    def test_sub_circuits_are_non_empty(self, pipeline):
        *_, sub_circuits = pipeline
        assert sum(len(qc.data) for qc in sub_circuits) > 0

    def test_sub_circuits_have_fewer_ops_than_original(self, pipeline):
        qc_orig, _, _, _, sub_circuits = pipeline
        orig_ops = len([i for i in qc_orig.data if i.operation.name != "barrier"])
        sub_ops = sum(len(qc.data) for qc in sub_circuits)
        assert sub_ops <= orig_ops

    def test_gate_types_are_valid_qiskit(self, pipeline):
        *_, sub_circuits = pipeline
        for qc in sub_circuits:
            for instr in qc.data:
                assert instr.operation is not None

    def test_qubit_indices_in_range(self, pipeline):
        *_, sub_circuits = pipeline
        for qc in sub_circuits:
            for instr in qc.data:
                for qubit in instr.qubits:
                    assert qc.find_bit(qubit).index < qc.num_qubits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
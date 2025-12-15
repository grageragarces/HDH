"""
Integration tests for HDH converters (Qiskit, QASM, etc.)

Note: These tests handle import variations across different versions
of the HDH library.
"""

import pytest

try:
    from hdh.converters import from_qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit converter not available")
class TestQiskitConverter:
    def test_simple_circuit(self):
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 2

"""
Uncomment when to_qiskit functional
"""

# try:
#     from hdh.converters import to_qiskit
#     TO_QISKIT_AVAILABLE = True
# except ImportError:
#     try:
#         from hdh.converters.qiskit_converter import to_qiskit
#         TO_QISKIT_AVAILABLE = True
#     except ImportError:
#         TO_QISKIT_AVAILABLE = False

# try:
#     from hdh.converters import from_qasm
#     QASM_AVAILABLE = True
# except ImportError:
#     try:
#         from hdh.converters import from_qasm
#         QASM_AVAILABLE = True
#     except ImportError:
#         QASM_AVAILABLE = False

# # Check if qiskit is installed
# try:
#     from qiskit import QuantumCircuit
#     QISKIT_INSTALLED = True
# except ImportError:
#     QISKIT_INSTALLED = False


# @pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED, 
#                     reason="Qiskit converter or qiskit not available")
# class TestQiskitConverter:
#     """Test Qiskit → HDH conversion"""
    
#     def test_simple_circuit_conversion(self):
#         """Test converting simple circuit to HDH"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2)
#         qc.h(0)
#         qc.cx(0, 1)
        
#         hdh = from_qiskit(qc)
        
#         assert hdh.get_num_qubits() == 2
#         assert len(hdh.S) > 0
#         assert len(hdh.C) > 0
    
#     def test_single_qubit_gates(self):
#         """Test various single-qubit gates"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(1)
#         qc.h(0)
#         qc.x(0)
#         qc.y(0)
#         qc.z(0)
        
#         hdh = from_qiskit(qc)
        
#         assert hdh.get_num_qubits() == 1
#         assert len(hdh.S) > 0
        
#         # Check gate names
#         gate_names = [hdh.gate_name.get(e, "").lower() for e in hdh.C]
#         assert any("h" in name for name in gate_names)
    
#     def test_two_qubit_gates(self):
#         """Test two-qubit gates"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2)
#         qc.cx(0, 1)
#         qc.cz(0, 1)
#         qc.swap(0, 1)
        
#         hdh = from_qiskit(qc)
        
#         assert hdh.get_num_qubits() == 2
    
#     def test_circuit_with_measurement(self):
#         """Test converting circuit with measurement"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(1, 1)
#         qc.h(0)
#         qc.measure(0, 0)
        
#         hdh = from_qiskit(qc)
        
#         # Check for classical nodes
#         c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
#         assert len(c_nodes) > 0
        
#         # Check for measurement edge
#         measure_edges = [e for e in hdh.C 
#                         if "measure" in hdh.gate_name.get(e, "").lower()]
#         assert len(measure_edges) > 0
    
#     def test_bell_state(self):
#         """Test Bell state circuit"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2)
#         qc.h(0)
#         qc.cx(0, 1)
        
#         hdh = from_qiskit(qc)
        
#         assert hdh.get_num_qubits() == 2
        
#         # Both qubits should initialize at t=0 (Issue #32 fix)
#         has_q0_t0 = any(n == "q0_t0" for n in hdh.S)
#         has_q1_t0 = any(n == "q1_t0" for n in hdh.S)
        
#         assert has_q0_t0, "q0 should initialize at t=0"
#         assert has_q1_t0, "q1 should initialize at t=0"
    
#     def test_ghz_state(self):
#         """Test 3-qubit GHZ state"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(3)
#         qc.h(0)
#         qc.cx(0, 1)
#         qc.cx(0, 2)
        
#         hdh = from_qiskit(qc)
        
#         assert hdh.get_num_qubits() == 3
        
#         # All qubits should initialize at t=0
#         for q in [0, 1, 2]:
#             assert any(n == f"q{q}_t0" for n in hdh.S), \
#                 f"q{q} should initialize at t=0"
    
#     def test_parametric_gates(self):
#         """Test gates with parameters"""
#         from qiskit import QuantumCircuit
#         from qiskit.circuit import Parameter
        
#         qc = QuantumCircuit(1)
#         theta = Parameter('θ')
#         qc.rx(theta, 0)
#         qc = qc.assign_parameters({theta: 0.5})
        
#         hdh = from_qiskit(qc)
#         assert hdh.get_num_qubits() == 1
    
#     def test_multiple_measurements(self):
#         """Test multiple measurements"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2, 2)
#         qc.h(0)
#         qc.h(1)
#         qc.measure([0, 1], [0, 1])
        
#         hdh = from_qiskit(qc)
        
#         # Should have classical nodes for both bits
#         c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
#         assert len(c_nodes) >= 2


# @pytest.mark.skipif(not TO_QISKIT_AVAILABLE or not QISKIT_INSTALLED,
#                     reason="to_qiskit converter or qiskit not available")
# class TestQiskitRoundtrip:
#     """Test HDH → Qiskit conversion"""
    
#     def test_roundtrip_simple(self):
#         """Test HDH → Qiskit → HDH preserves structure"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2)
#         qc.h(0)
#         qc.cx(0, 1)
        
#         hdh1 = from_qiskit(qc)
#         qc2 = to_qiskit(hdh1)
#         hdh2 = from_qiskit(qc2)
        
#         # Basic structure should match
#         assert hdh1.get_num_qubits() == hdh2.get_num_qubits()


# @pytest.mark.skipif(not QASM_AVAILABLE, reason="QASM converter not available")
# class TestQASMConverter:
#     """Test QASM → HDH conversion"""
    
#     def test_qasm_string_simple(self):
#         """Test converting simple QASM string"""
#         qasm_str = """
#         OPENQASM 2.0;
#         include "qelib1.inc";
#         qreg q[2];
#         h q[0];
#         cx q[0],q[1];
#         """
        
#         hdh = from_qasm('string', qasm_str)
#         assert hdh.get_num_qubits() == 2
    
#     def test_qasm_with_measurement(self):
#         """Test QASM with measurement"""
#         qasm_str = """
#         OPENQASM 2.0;
#         include "qelib1.inc";
#         qreg q[2];
#         creg c[2];
#         h q[0];
#         cx q[0],q[1];
#         measure q -> c;
#         """
        
#         hdh = from_qasm('string', qasm_str)
        
#         # Should have classical nodes
#         c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
#         assert len(c_nodes) > 0
    
#     def test_qasm_bell_state(self):
#         """Test Bell state from QASM"""
#         qasm_str = """
#         OPENQASM 2.0;
#         include "qelib1.inc";
#         qreg q[2];
#         h q[0];
#         cx q[0],q[1];
#         """
        
#         hdh = from_qasm('string', qasm_str)
        
#         # Verify timestep initialization (Issue #32)
#         has_q0_t0 = any(n == "q0_t0" for n in hdh.S)
#         has_q1_t0 = any(n == "q1_t0" for n in hdh.S)
        
#         assert has_q0_t0, "q0 should initialize at t=0"
#         assert has_q1_t0, "q1 should initialize at t=0"


# class TestConverterEdgeCases:
#     """Test edge cases and error handling"""
    
#     @pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED,
#                         reason="Qiskit not available")
#     def test_empty_circuit(self):
#         """Test converting empty circuit"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(2)
#         hdh = from_qiskit(qc)
        
#         # Should have nodes for qubit initialization
#         assert hdh.get_num_qubits() == 2
    
#     @pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED,
#                         reason="Qiskit not available")
#     def test_single_qubit_circuit(self):
#         """Test single qubit circuit"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(1)
#         qc.h(0)
        
#         hdh = from_qiskit(qc)
#         assert hdh.get_num_qubits() == 1
    
#     @pytest.mark.skipif(not QISKIT_AVAILABLE or not QISKIT_INSTALLED,
#                         reason="Qiskit not available")
#     def test_large_circuit(self):
#         """Test larger circuit"""
#         from qiskit import QuantumCircuit
        
#         qc = QuantumCircuit(5)
#         for i in range(5):
#             qc.h(i)
#         for i in range(4):
#             qc.cx(i, i+1)
        
#         hdh = from_qiskit(qc)
#         assert hdh.get_num_qubits() == 5
        
#         # All qubits should start at t=0
#         for q in range(5):
#             assert any(n == f"q{q}_t0" for n in hdh.S), \
#                 f"q{q} should initialize at t=0"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
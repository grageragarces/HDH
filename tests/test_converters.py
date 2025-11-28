import pytest
from qiskit import QuantumCircuit
from hdh.converters import from_qiskit, to_qiskit

class TestQiskitConverter:
    """Test Qiskit ↔ HDH conversion"""
    
    def test_simple_circuit_conversion(self):
        """Test converting simple circuit to HDH"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        hdh = from_qiskit(qc)
        
        assert hdh.get_num_qubits() == 2
        assert len(hdh.S) > 0
        assert len(hdh.C) > 0
    
    def test_circuit_with_measurement(self):
        """Test converting circuit with measurement"""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        hdh = from_qiskit(qc)
        
        # Check for classical nodes
        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) > 0
    
    def test_roundtrip_conversion(self):
        """Test HDH → Qiskit → HDH preserves structure"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        hdh1 = from_qiskit(qc)
        qc2 = to_qiskit(hdh1)
        hdh2 = from_qiskit(qc2)
        
        # Basic structure should match
        assert hdh1.get_num_qubits() == hdh2.get_num_qubits()
    
    def test_parametric_gates(self):
        """Test gates with parameters"""
        from qiskit.circuit import Parameter
        
        qc = QuantumCircuit(1)
        theta = Parameter('θ')
        qc.rx(theta, 0)
        qc = qc.assign_parameters({theta: 0.5})
        
        hdh = from_qiskit(qc)
        assert hdh.get_num_qubits() == 1

class TestQASMConverter:
    """Test QASM → HDH conversion"""
    
    def test_qasm_string_conversion(self):
        """Test converting QASM string"""
        from hdh.converters import from_qasm
        
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        
        hdh = from_qasm('string', qasm_str)
        assert hdh.get_num_qubits() == 2
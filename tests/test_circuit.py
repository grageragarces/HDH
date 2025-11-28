import pytest
from hdh.models.circuit import Circuit

class TestCircuitBasics:
    """Test basic circuit operations"""
    
    def test_empty_circuit(self):
        """Test empty circuit builds valid HDH"""
        circuit = Circuit()
        hdh = circuit.build_hdh()
        
        assert len(hdh.S) == 0
        assert len(hdh.C) == 0
    
    def test_single_gate(self):
        """Test single-qubit gate"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        hdh = circuit.build_hdh()
        
        assert hdh.get_num_qubits() == 1
        q0_nodes = [n for n in hdh.S if n.startswith("q0_")]
        assert len(q0_nodes) > 0
    
    def test_two_qubit_gate(self):
        """Test two-qubit gate"""
        circuit = Circuit()
        circuit.add_instruction("cx", [0, 1])
        hdh = circuit.build_hdh()
        
        assert hdh.get_num_qubits() == 2
        
        # Check both qubits have nodes
        assert any(n.startswith("q0_") for n in hdh.S)
        assert any(n.startswith("q1_") for n in hdh.S)
    
    def test_measurement(self):
        """Test measurement instruction"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        hdh = circuit.build_hdh()
        
        # Check for classical node
        c_nodes = [n for n in hdh.S if n.startswith("c0_")]
        assert len(c_nodes) > 0, "Should have classical bit node"
        
        # Check for measurement edge
        measure_edges = [e for e in hdh.C 
                        if hdh.gate_name.get(e) == "measure"]
        assert len(measure_edges) > 0, "Should have measurement edge"

class TestCircuitSequences:
    """Test sequences of operations"""
    
    def test_bell_state(self):
        """Test Bell state circuit: H-CNOT"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        hdh = circuit.build_hdh()
        
        assert hdh.get_num_qubits() == 2
        
        # Check edges exist for both operations
        h_edges = [e for e in hdh.C if "h" in hdh.gate_name.get(e, "")]
        cx_edges = [e for e in hdh.C if "cx" in hdh.gate_name.get(e, "")]
        
        assert len(h_edges) > 0, "Should have H gate"
        assert len(cx_edges) > 0, "Should have CX gate"
    
    def test_three_qubit_ghz(self):
        """Test GHZ state: H on q0, then two CNOTs"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("cx", [0, 2])
        hdh = circuit.build_hdh()
        
        assert hdh.get_num_qubits() == 3
    
    def test_sequential_gates_same_qubit(self):
        """Test multiple gates on same qubit"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("x", [0])
        circuit.add_instruction("y", [0])
        circuit.add_instruction("z", [0])
        hdh = circuit.build_hdh()
        
        q0_nodes = sorted([n for n in hdh.S if n.startswith("q0_")],
                         key=lambda n: hdh.time_map[n])
        
        # Should have increasing timesteps
        times = [hdh.time_map[n] for n in q0_nodes]
        assert times == sorted(times)

class TestConditionalOperations:
    """Test classical control and conditional operations"""
    
    def test_measure_then_conditional_gate(self):
        """Test measurement followed by conditional gate"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
        
        hdh = circuit.build_hdh()
        
        # Check for predicted/conditional edge
        predicted_edges = [e for e in hdh.C if hdh.phi[e] == "p"]
        assert len(predicted_edges) > 0, "Should have predicted edges"
    
    def test_multiple_measurements(self):
        """Test multiple measurements"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("h", [1])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("measure", [1], [1])
        
        hdh = circuit.build_hdh()
        
        # Check for two classical nodes
        c0_nodes = [n for n in hdh.S if n.startswith("c0_")]
        c1_nodes = [n for n in hdh.S if n.startswith("c1_")]
        
        assert len(c0_nodes) > 0
        assert len(c1_nodes) > 0
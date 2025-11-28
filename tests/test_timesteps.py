import pytest
from hdh.models.circuit import Circuit

class TestTimestepInitialization:
    """Tests for Issue #32: All qubits should initialize at t=0"""
    
    def test_all_qubits_initialize_at_t0(self):
        """Verify all qubits start at t=0 regardless of usage order"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("x", [0])
        circuit.add_instruction("y", [0])
        circuit.add_instruction("h", [1])  # q1 used much later
        
        hdh = circuit.build_hdh()
        
        # Check both qubits have t=0 nodes
        q0_nodes = [n for n in hdh.S if n.startswith("q0_")]
        q1_nodes = [n for n in hdh.S if n.startswith("q1_")]
        
        assert any(n == "q0_t0" for n in q0_nodes), "q0 should have a node at t=0"
        assert any(n == "q1_t0" for n in q1_nodes), "q1 should have a node at t=0"
    
    def test_three_qubits_initialize_at_t0(self):
        """Test with 3 qubits used at different times"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("ccx", [0, 1, 2])  # q2 used last
        
        hdh = circuit.build_hdh()
        
        for q in [0, 1, 2]:
            nodes = [n for n in hdh.S if n.startswith(f"q{q}_")]
            assert any(n == f"q{q}_t0" for n in nodes), \
                f"q{q} should initialize at t=0"
    
    def test_operation_ordering_preserved(self):
        """Ensure operations still execute in correct order"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("x", [0])
        
        hdh = circuit.build_hdh()
        
        q0_nodes = sorted([n for n in hdh.S if n.startswith("q0_")],
                         key=lambda n: hdh.time_map[n])
        
        # Should have nodes at increasing timesteps
        times = [hdh.time_map[n] for n in q0_nodes]
        assert times == sorted(times), "Timesteps should be in order"
        assert times[0] == 0, "Should start at t=0"

    def test_conditional_gate_with_measurement(self):
        """Test classical control still works correctly"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
        
        hdh = circuit.build_hdh()
        
        # Check q1 still initializes at t=0
        q1_nodes = [n for n in hdh.S if n.startswith("q1_")]
        assert any(n == "q1_t0" for n in q1_nodes), \
            "q1 should initialize at t=0 even with conditional gate"
        
        # Check classical control edge exists
        c_edges = [e for e in hdh.C if hdh.tau[e] == "c"]
        assert len(c_edges) > 0, "Should have classical control edges"
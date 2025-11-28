import pytest
from hdh.hdh import HDH

class TestHDHBasics:
    """Test basic HDH data structure operations"""
    
    def test_add_node(self):
        """Test adding nodes to HDH"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        
        assert "q0_t0" in hdh.S
        assert hdh.sigma["q0_t0"] == "q"
        assert hdh.time_map["q0_t0"] == 0
        assert 0 in hdh.T
    
    def test_add_hyperedge(self):
        """Test adding hyperedges"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q0_t1", "q", 1)
        
        edge = hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", name="h")
        
        assert edge in hdh.C
        assert hdh.tau[edge] == "q"
        assert hdh.gate_name[edge] == "h"
    
    def test_get_num_qubits(self):
        """Test qubit counting"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q1_t0", "q", 0)
        hdh.add_node("q2_t1", "q", 1)
        
        assert hdh.get_num_qubits() == 3
    
    def test_ancestry(self):
        """Test ancestry computation"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q0_t1", "q", 1)
        hdh.add_node("q0_t2", "q", 2)
        
        # Add edges to create path
        hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q")
        hdh.add_hyperedge({"q0_t1", "q0_t2"}, "q")
        
        ancestry = hdh.get_ancestry("q0_t2")
        assert "q0_t0" in ancestry
        assert "q0_t1" in ancestry
    
    def test_lineage(self):
        """Test lineage computation"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q0_t1", "q", 1)
        hdh.add_node("q0_t2", "q", 2)
        
        hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q")
        hdh.add_hyperedge({"q0_t1", "q0_t2"}, "q")
        
        lineage = hdh.get_lineage("q0_t0")
        assert "q0_t1" in lineage
        assert "q0_t2" in lineage

class TestHDHNodeTypes:
    """Test quantum and classical node handling"""
    
    def test_quantum_nodes(self):
        """Test quantum node properties"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0, node_real="a")
        
        assert hdh.sigma["q0_t0"] == "q"
        assert hdh.upsilon["q0_t0"] == "a"
    
    def test_classical_nodes(self):
        """Test classical node properties"""
        hdh = HDH()
        hdh.add_node("c0_t1", "c", 1, node_real="a")
        
        assert hdh.sigma["c0_t1"] == "c"
        assert hdh.time_map["c0_t1"] == 1
    
    def test_predicted_nodes(self):
        """Test predicted (non-actualized) nodes"""
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0, node_real="p")
        
        assert hdh.upsilon["q0_t0"] == "p"
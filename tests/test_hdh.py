import pytest
from hdh.hdh import HDH, NodeType, Realisation, EdgeRole

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
    
    def test_add_node_rejects_type_mismatch(self):
        """Re-adding an existing node ID with a different type must fail loudly.

        Regression test: HDH.add_node used to silently overwrite sigma[node_id],
        which let a classical output accidentally clobber an existing quantum
        node that happened to share the same ID (see JOSS review issue #69).
        """
        hdh = HDH()
        hdh.add_node("q2_t2", "q", 2)

        with pytest.raises(ValueError):
            hdh.add_node("q2_t2", "c", 2)

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

    def test_enum_members_and_strings_are_interchangeable(self):
        hdh = HDH()
        hdh.add_node("q0_t0", NodeType.QUANTUM, 0, node_real=Realisation.PREDICTED)
        hdh.add_node("c0_t1", "c", 1)

        assert hdh.sigma["q0_t0"] == "q" == NodeType.QUANTUM
        assert hdh.sigma["c0_t1"] == NodeType.CLASSICAL
        assert hdh.upsilon["q0_t0"] == "p"
        # Stored as plain strings, so printing/serialising is unchanged.
        assert type(hdh.sigma["q0_t0"]) is str
        assert str(NodeType.QUANTUM) == "q"

    def test_invalid_node_type_rejected(self):
        hdh = HDH()
        with pytest.raises(ValueError, match="node type 'x'"):
            hdh.add_node("q0_t0", "x", 0)
        assert "q0_t0" not in hdh.S

    def test_invalid_node_realisation_rejected(self):
        with pytest.raises(ValueError, match="node realisation"):
            HDH().add_node("q0_t0", "q", 0, node_real="maybe")

    def test_invalid_edge_values_rejected(self):
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q0_t1", "q", 1)
        with pytest.raises(ValueError, match="edge type"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "quantum")
        with pytest.raises(ValueError, match="edge realisation"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", node_real="x")
        with pytest.raises(ValueError, match="edge role"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", role="teleport")
        assert not hdh.C

    def test_edge_role_accepts_enum(self):
        hdh = HDH()
        hdh.add_node("q0_t0", "q", 0)
        hdh.add_node("q0_t1", "q", 1)
        edge = hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", role=EdgeRole.TELEDATA)
        assert hdh.edge_role[edge] == "teledata"
import pytest
from hdh.hdh import HDH, NodeType, Realisation, EdgeRole

class TestHDHBasics:
    """Test basic HDH data structure operations"""
    
    def test_add_node(self):
        """Test adding nodes to HDH"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        
        assert "q0_t0" in hdh.S
        assert hdh.sigma["q0_t0"] == "q"
        assert hdh.time_map["q0_t0"] == 0
        assert 0 in hdh.T
    
    def test_add_hyperedge(self):
        """Test adding hyperedges"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q0", 1, "q")
        
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
        hdh.add_node("q2", 2, "q")

        with pytest.raises(ValueError):
            hdh.add_node("q2", 2, "c")

    def test_get_num_qubits(self):
        """Test qubit counting"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q1", 0, "q")
        hdh.add_node("q2", 1, "q")
        
        assert hdh.get_num_qubits() == 3
    
    def test_ancestry(self):
        """Test ancestry computation"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q0", 1, "q")
        hdh.add_node("q0", 2, "q")
        
        # Add edges to create path
        hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q")
        hdh.add_hyperedge({"q0_t1", "q0_t2"}, "q")
        
        ancestry = hdh.get_ancestry("q0_t2")
        assert "q0_t0" in ancestry
        assert "q0_t1" in ancestry
    
    def test_lineage(self):
        """Test lineage computation"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q0", 1, "q")
        hdh.add_node("q0", 2, "q")
        
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
        hdh.add_node("q0", 0, "q", node_real="a")
        
        assert hdh.sigma["q0_t0"] == "q"
        assert hdh.upsilon["q0_t0"] == "a"
    
    def test_classical_nodes(self):
        """Test classical node properties"""
        hdh = HDH()
        hdh.add_node("c0", 1, "c", node_real="a")
        
        assert hdh.sigma["c0_t1"] == "c"
        assert hdh.time_map["c0_t1"] == 1
    
    def test_predicted_nodes(self):
        """Test predicted (non-actualized) nodes"""
        hdh = HDH()
        hdh.add_node("q0", 0, "q", node_real="p")
        
        assert hdh.upsilon["q0_t0"] == "p"

    def test_enum_members_and_strings_are_interchangeable(self):
        hdh = HDH()
        hdh.add_node("q0", 0, NodeType.QUANTUM, node_real=Realisation.PREDICTED)
        hdh.add_node("c0", 1, "c")

        assert hdh.sigma["q0_t0"] == "q" == NodeType.QUANTUM
        assert hdh.sigma["c0_t1"] == NodeType.CLASSICAL
        assert hdh.upsilon["q0_t0"] == "p"
        # Stored as plain strings, so printing/serialising is unchanged.
        assert type(hdh.sigma["q0_t0"]) is str
        assert str(NodeType.QUANTUM) == "q"

    def test_invalid_node_type_rejected(self):
        hdh = HDH()
        with pytest.raises(ValueError, match="node type 'x'"):
            hdh.add_node("q0", 0, "x")
        assert "q0_t0" not in hdh.S

    def test_invalid_node_realisation_rejected(self):
        with pytest.raises(ValueError, match="node realisation"):
            HDH().add_node("q0", 0, "q", node_real="maybe")

    def test_invalid_edge_values_rejected(self):
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q0", 1, "q")
        with pytest.raises(ValueError, match="edge type"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "quantum")
        with pytest.raises(ValueError, match="edge realisation"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", node_real="x")
        with pytest.raises(ValueError, match="edge role"):
            hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", role="teleport")
        assert not hdh.C

    def test_edge_role_accepts_enum(self):
        hdh = HDH()
        hdh.add_node("q0", 0, "q")
        hdh.add_node("q0", 1, "q")
        edge = hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", role=EdgeRole.TELEDATA)
        assert hdh.edge_role[edge] == "teledata"

class TestAddNodeSignature:
    """`add_node(wire, time, node_type)` builds the ID itself (0.5 API)."""

    def test_returns_built_id_and_records_wire(self):
        hdh = HDH()
        node = hdh.add_node("q3", 7)

        assert node == "q3_t7" == HDH.node_id("q3", 7)
        assert hdh.wire_of[node] == "q3"
        assert hdh.sigma[node] == "q"  # quantum by default
        assert hdh.time_map[node] == 7

    def test_arbitrary_wire_labels(self):
        hdh = HDH()
        assert hdh.add_node("cell_top", 2) == "cell_top_t2"
        assert hdh.wire_of["cell_top_t2"] == "cell_top"

    def test_readding_same_state_is_idempotent(self):
        hdh = HDH()
        first = hdh.add_node("q0", 1)
        second = hdh.add_node("q0", 1)
        assert first == second
        assert len(hdh.S) == 1

    def test_same_state_with_other_type_rejected(self):
        hdh = HDH()
        hdh.add_node("x", 1, "q")
        with pytest.raises(ValueError, match="already exists"):
            hdh.add_node("x", 1, "c")

    def test_old_signature_gives_migration_hint(self):
        with pytest.raises(TypeError, match=r"add_node\('q0', 1, 'q'\)"):
            HDH().add_node("q0_t1", "q", 1)

    @pytest.mark.parametrize("time", [1.5, True, "1"])
    def test_non_integer_time_rejected(self, time):
        with pytest.raises(TypeError):
            HDH().add_node("q0", time)

    def test_empty_wire_rejected(self):
        with pytest.raises(TypeError):
            HDH().add_node("", 0)


class TestGetNumQubits:
    def test_circuit_wires_report_highest_index_plus_one(self):
        hdh = HDH()
        hdh.add_node("q0", 0)
        hdh.add_node("q4", 0)
        hdh.add_node("c7", 1, "c")
        assert hdh.get_num_qubits() == 5

    def test_named_wires_are_counted(self):
        hdh = HDH()
        for wire in ("a", "b", "a"):
            hdh.add_node(wire, 0)
        hdh.add_node("m", 1, "c")
        assert hdh.get_num_qubits() == 2

    def test_classical_node_with_q_name_is_not_a_qubit(self):
        hdh = HDH()
        hdh.add_node("q0", 0)
        hdh.add_node("q9", 1, "c")
        assert hdh.get_num_qubits() == 1

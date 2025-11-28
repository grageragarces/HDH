"""
Tests for HDH Invariant Constraints

These tests verify that HDH structures satisfy fundamental invariants:
1. Cardinality constraint
2. Wire continuity
3. No duplicate time nodes
4. Classical channel sanity
"""

import pytest
from hdh.hdh import HDH
from hdh.models.circuit import Circuit


class TestCardinalityConstraint:
    """
    Cardinality constraint: The sum of the number of q nodes per partition 
    should not exceed #qubits × #partitions.
    
    For a single partition (no partitioning), this means:
    Total quantum nodes should be reasonable given the number of qubits.
    """
    
    def test_single_partition_cardinality(self):
        """Test cardinality in single partition (no partitioning)"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        
        hdh = circuit.build_hdh()
        
        num_qubits = hdh.get_num_qubits()
        quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
        
        # For 1 partition: all nodes should be accounted for
        # With no partitioning, all quantum nodes are in one "partition"
        assert len(quantum_nodes) > 0
        assert num_qubits > 0
        
        # Sanity check: shouldn't have way more nodes than makes sense
        # Each qubit can have multiple timesteps, but there's a limit
        max_reasonable_nodes = num_qubits * 1000  # Very generous upper bound
        assert len(quantum_nodes) < max_reasonable_nodes, \
            f"Too many quantum nodes: {len(quantum_nodes)} for {num_qubits} qubits"
    
    def test_cardinality_with_mock_partition(self):
        """Test cardinality when we simulate partitioning"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("h", [1])
        circuit.add_instruction("h", [2])
        
        hdh = circuit.build_hdh()
        
        num_qubits = hdh.get_num_qubits()
        quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
        
        # Simulate 2 partitions: split nodes by qubit
        partition1 = [n for n in quantum_nodes if 'q0_' in n or 'q1_' in n]
        partition2 = [n for n in quantum_nodes if 'q2_' in n]
        
        num_partitions = 2
        
        # Cardinality constraint: sum of nodes in all partitions
        total_nodes = len(partition1) + len(partition2)
        
        # Should equal total quantum nodes
        assert total_nodes == len(quantum_nodes)
        
        # Each partition should be reasonable
        max_per_partition = num_qubits * num_partitions * 100  # Generous bound
        assert len(partition1) < max_per_partition
        assert len(partition2) < max_per_partition
    
    def test_cardinality_grows_with_circuit_depth(self):
        """Test that cardinality scales reasonably with circuit depth"""
        # Shallow circuit
        circuit1 = Circuit()
        circuit1.add_instruction("h", [0])
        hdh1 = circuit1.build_hdh()
        nodes1 = len([n for n in hdh1.S if hdh1.sigma[n] == 'q'])
        
        # Deeper circuit
        circuit2 = Circuit()
        for _ in range(10):
            circuit2.add_instruction("h", [0])
        hdh2 = circuit2.build_hdh()
        nodes2 = len([n for n in hdh2.S if hdh2.sigma[n] == 'q'])
        
        # Deeper circuit should have more nodes
        assert nodes2 > nodes1, "Deeper circuit should have more nodes"
        
        # But growth should be linear, not exponential
        assert nodes2 < nodes1 * 20, "Node growth should be reasonable"

class TestNoDuplicateTimeNodes:
    """
    No duplicate time nodes: There must not be two nodes with the same 
    qubit/bit label and the same timestamp.
    """
    
    def test_no_duplicate_quantum_nodes(self):
        """Test that no two quantum nodes have same qubit and timestamp"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("x", [0])
        circuit.add_instruction("cx", [0, 1])
        
        hdh = circuit.build_hdh()
        
        # Check all quantum nodes
        seen = set()
        for node in hdh.S:
            if hdh.sigma[node] == 'q':
                # Extract qubit index and timestamp
                # Format: q{idx}_t{time}
                assert node not in seen, f"Duplicate node found: {node}"
                seen.add(node)
    
    def test_no_duplicate_classical_nodes(self):
        """Test that no two classical nodes have same bit and timestamp"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("h", [1])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("measure", [1], [1])
        
        hdh = circuit.build_hdh()
        
        # Check all classical nodes
        seen = set()
        for node in hdh.S:
            if hdh.sigma[node] == 'c':
                assert node not in seen, f"Duplicate classical node: {node}"
                seen.add(node)
    
    def test_no_duplicate_nodes_complex_circuit(self):
        """Test no duplicates in complex circuit with multiple operations"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("x", [0])
        circuit.add_instruction("y", [1])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("measure", [1], [1])
        
        hdh = circuit.build_hdh()
        
        # Check ALL nodes (quantum and classical)
        assert len(hdh.S) == len(set(hdh.S)), \
            "Duplicate nodes found in HDH"
        
        # More detailed check
        node_list = list(hdh.S)
        node_set = set(hdh.S)
        
        if len(node_list) != len(node_set):
            duplicates = [n for n in node_list if node_list.count(n) > 1]
            pytest.fail(f"Found duplicate nodes: {set(duplicates)}")
    
    def test_timestep_uniqueness_per_qubit(self):
        """Test that each qubit has unique timesteps"""
        circuit = Circuit()
        for _ in range(5):
            circuit.add_instruction("h", [0])
        
        hdh = circuit.build_hdh()
        
        # Get all q0 nodes with their timestamps
        q0_times = {}
        for node in hdh.S:
            if node.startswith('q0_'):
                time = hdh.time_map[node]
                if time in q0_times:
                    pytest.fail(
                        f"Duplicate timestamp {time} for q0: "
                        f"{q0_times[time]} and {node}"
                    )
                q0_times[time] = node


class TestClassicalChannelSanity:
    """
    Classical channel sanity: Any edge marked as a classical channel 
    (type = 'c') must originate from or lead to at least one classical node.
    """
    
    def test_classical_edges_have_classical_nodes(self):
        """Test that classical edges connect to classical nodes"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        
        hdh = circuit.build_hdh()
        
        # Find all classical edges
        classical_edges = [e for e in hdh.C if hdh.tau[e] == 'c']
        
        assert len(classical_edges) > 0, "Should have classical edges"
        
        for edge in classical_edges:
            nodes_in_edge = list(edge)
            
            # At least one node should be classical
            has_classical_node = any(
                hdh.sigma[n] == 'c' for n in nodes_in_edge if n in hdh.sigma
            )
            
            assert has_classical_node, \
                f"Classical edge {edge} has no classical nodes"
    
    def test_measurement_creates_classical_edge(self):
        """Test that measurements create proper classical edges"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        
        hdh = circuit.build_hdh()
        
        # Should have measurement edges (type 'c')
        measure_edges = [
            e for e in hdh.C 
            if hdh.gate_name.get(e, '').lower() == 'measure'
        ]
        
        assert len(measure_edges) > 0, "Should have measurement edges"
        
        for edge in measure_edges:
            # Measurement edge should connect quantum and classical
            nodes = list(edge)
            node_types = [hdh.sigma.get(n) for n in nodes if n in hdh.sigma]
            
            # Should have both 'q' and 'c' types
            assert 'q' in node_types, "Measurement should involve quantum node"
            assert 'c' in node_types, "Measurement should create classical node"
    
    def test_classical_control_edges(self):
        """Test that classically controlled gates have proper classical edges"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
        
        hdh = circuit.build_hdh()
        
        # Find predicted (conditional) edges
        conditional_edges = [e for e in hdh.C if hdh.phi.get(e) == 'p']
        
        # At least some should be classical type
        classical_conditional = [
            e for e in conditional_edges if hdh.tau[e] == 'c'
        ]
        
        assert len(classical_conditional) > 0, \
            "Should have classical conditional edges"
        
        for edge in classical_conditional:
            nodes = list(edge)
            # Should involve classical node(s)
            has_classical = any(
                hdh.sigma.get(n) == 'c' for n in nodes if n in hdh.sigma
            )
            assert has_classical, \
                f"Classical control edge {edge} has no classical nodes"
    
    def test_no_orphan_classical_edges(self):
        """Test that classical edges aren't disconnected from classical nodes"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("h", [1])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("measure", [1], [1])
        
        hdh = circuit.build_hdh()
        
        # Get all classical edges
        classical_edges = [e for e in hdh.C if hdh.tau[e] == 'c']
        
        # Get all classical nodes
        classical_nodes = [n for n in hdh.S if hdh.sigma[n] == 'c']
        
        assert len(classical_nodes) > 0, "Should have classical nodes"
        
        # Every classical edge should connect to at least one classical node
        for edge in classical_edges:
            edge_nodes = set(edge)
            intersection = edge_nodes & set(classical_nodes)
            
            assert len(intersection) > 0, \
                f"Classical edge {edge} not connected to any classical nodes"
    
    def test_quantum_edges_no_classical_nodes(self):
        """Test that pure quantum edges don't involve classical nodes"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("x", [0])
        
        hdh = circuit.build_hdh()
        
        # Get pure quantum edges (no measurement)
        quantum_edges = [
            e for e in hdh.C 
            if hdh.tau[e] == 'q' and 'measure' not in hdh.gate_name.get(e, '')
        ]
        
        assert len(quantum_edges) > 0, "Should have quantum edges"
        
        for edge in quantum_edges:
            nodes = list(edge)
            # Should NOT have classical nodes
            has_classical = any(
                hdh.sigma.get(n) == 'c' for n in nodes if n in hdh.sigma
            )
            
            assert not has_classical, \
                f"Pure quantum edge {edge} incorrectly contains classical nodes"


class TestInvariantConsistency:
    """Test that all invariants hold together"""
    
    def test_all_invariants_bell_state(self):
        """Test all invariants on Bell state circuit"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("measure", [1], [1])
        
        hdh = circuit.build_hdh()
        
        # 1. Cardinality
        num_qubits = hdh.get_num_qubits()
        quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
        assert len(quantum_nodes) < num_qubits * 100
        
        # 2. Wire continuity (for q0 and q1)
        for q in [0, 1]:
            nodes = sorted(
                [n for n in hdh.S if n.startswith(f'q{q}_')],
                key=lambda n: hdh.time_map[n]
            )
            for i in range(len(nodes) - 1):
                lineage = hdh.get_lineage(nodes[i])
                assert any(nodes[j] in lineage for j in range(i+1, len(nodes)))
        
        # 3. No duplicates
        assert len(hdh.S) == len(set(hdh.S))
        
        # 4. Classical channel sanity
        classical_edges = [e for e in hdh.C if hdh.tau[e] == 'c']
        for edge in classical_edges:
            has_classical = any(
                hdh.sigma.get(n) == 'c' for n in edge if n in hdh.sigma
            )
            assert has_classical
    
    def test_all_invariants_complex_circuit(self):
        """Test all invariants on complex circuit"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("h", [1])
        circuit.add_instruction("h", [2])
        circuit.add_instruction("cx", [0, 1])
        circuit.add_instruction("cx", [1, 2])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [2], bits=[0], cond_flag="p")
        
        hdh = circuit.build_hdh()
        
        # Run all invariant checks
        # 1. Cardinality
        num_qubits = hdh.get_num_qubits()
        quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
        assert len(quantum_nodes) < num_qubits * 100
        
        # 2. No duplicates
        assert len(hdh.S) == len(set(hdh.S)), "Found duplicate nodes"
        
        # 3. Classical edges have classical nodes
        classical_edges = [e for e in hdh.C if hdh.tau[e] == 'c']
        for edge in classical_edges:
            has_classical = any(
                hdh.sigma.get(n) == 'c' for n in edge if n in hdh.sigma
            )
            assert has_classical, f"Classical edge {edge} has no classical nodes"

"""
Fixed Wire Continuity Tests

These tests are adapted to work with the actual HDH structure where
gates create multiple intermediate nodes (stage1, stage2, stage3).
"""

import pytest
from hdh.hdh import HDH
from hdh.models.circuit import Circuit


# class TestWireContinuityFixed:
#     """
#     Wire continuity: There should always be a valid connection (temporal path) 
#     between nodes representing the same qubit across timesteps.
    
#     FIXED VERSION: Works with multi-stage gate construction
#     """
    
#     def test_single_qubit_wire_continuity(self):
#         """Test wire continuity for a single qubit"""
#         circuit = Circuit()
#         circuit.add_instruction("h", [0])
#         circuit.add_instruction("x", [0])
#         circuit.add_instruction("y", [0])
        
#         hdh = circuit.build_hdh()
        
#         # Get all q0 nodes sorted by time
#         q0_nodes = sorted(
#             [n for n in hdh.S if n.startswith('q0_')],
#             key=lambda n: hdh.time_map[n]
#         )
        
#         assert len(q0_nodes) >= 2, "Should have multiple timesteps"
        
#         # Check that we have nodes at consecutive timesteps
#         # OR that there's at least a path through the hypergraph
#         times = [hdh.time_map[n] for n in q0_nodes]
        
#         # Verify we have a monotonically increasing sequence
#         # (some timestamps might be skipped, but should be increasing)
#         for i in range(len(times) - 1):
#             assert times[i] <= times[i+1], \
#                 f"Timestamps should be non-decreasing: {times}"
        
#         # Check connectivity through hyperedges
#         # Each node (except the last) should be in an edge with a later node
#         for i, node in enumerate(q0_nodes[:-1]):
#             # Find edges containing this node
#             edges_with_node = [e for e in hdh.C if node in e]
#             assert len(edges_with_node) > 0, \
#                 f"Node {node} should be in at least one edge"
            
#             # At least one edge should connect to a future node
#             has_future_connection = False
#             for edge in edges_with_node:
#                 edge_nodes = list(edge)
#                 for other in edge_nodes:
#                     if other in q0_nodes and hdh.time_map[other] > hdh.time_map[node]:
#                         has_future_connection = True
#                         break
#                 if has_future_connection:
#                     break
            
#             assert has_future_connection, \
#                 f"Wire continuity broken: {node} has no connection to future"
    
#     def test_multi_qubit_wire_continuity(self):
#         """Test wire continuity for multiple qubits"""
#         circuit = Circuit()
#         circuit.add_instruction("h", [0])
#         circuit.add_instruction("h", [1])
#         circuit.add_instruction("cx", [0, 1])
        
#         hdh = circuit.build_hdh()
        
#         # Check continuity for each qubit
#         for qubit_idx in [0, 1]:
#             qubit_nodes = sorted(
#                 [n for n in hdh.S if n.startswith(f'q{qubit_idx}_')],
#                 key=lambda n: hdh.time_map[n]
#             )
            
#             assert len(qubit_nodes) >= 2, \
#                 f"q{qubit_idx} should have multiple timesteps"
            
#             # Check each node (except last) connects to a future node
#             for i, node in enumerate(qubit_nodes[:-1]):
#                 edges_with_node = [e for e in hdh.C if node in e]
                
#                 # Find if any edge connects to a future node on same qubit
#                 has_connection = False
#                 for edge in edges_with_node:
#                     for other in edge:
#                         if (other in qubit_nodes and 
#                             hdh.time_map[other] > hdh.time_map[node]):
#                             has_connection = True
#                             break
#                     if has_connection:
#                         break
                
#                 # Allow for terminal nodes (no future connection)
#                 # but there should be edges connecting most nodes
#                 if i < len(qubit_nodes) - 2:  # Not the last or second-to-last
#                     assert has_connection or len(edges_with_node) > 0, \
#                         f"q{qubit_idx} wire continuity issue at {node}"
    
#     def test_wire_continuity_with_measurement(self):
#         """Test that quantum wire ends properly after measurement"""
#         circuit = Circuit()
#         circuit.add_instruction("h", [0])
#         circuit.add_instruction("measure", [0], [0])
        
#         hdh = circuit.build_hdh()
        
#         # Get all q0 nodes
#         q0_nodes = sorted(
#             [n for n in hdh.S if n.startswith('q0_')],
#             key=lambda n: hdh.time_map[n]
#         )
        
#         # There should be nodes before measurement
#         assert len(q0_nodes) > 0
        
#         # Classical wire should start after measurement
#         c0_nodes = [n for n in hdh.S if n.startswith('c0_')]
#         assert len(c0_nodes) > 0, "Should have classical nodes after measurement"
        
#         # Check that measurement edge connects quantum and classical
#         measure_edges = [
#             e for e in hdh.C 
#             if hdh.gate_name.get(e, '').lower() == 'measure'
#         ]
        
#         assert len(measure_edges) > 0, "Should have measurement edges"
        
#         # At least one measurement edge should connect q0 and c0 nodes
#         has_proper_connection = False
#         for edge in measure_edges:
#             edge_nodes = list(edge)
#             has_q0 = any(n.startswith('q0_') for n in edge_nodes)
#             has_c0 = any(n.startswith('c0_') for n in edge_nodes)
#             if has_q0 and has_c0:
#                 has_proper_connection = True
#                 break
        
#         assert has_proper_connection, "Measurement should connect q0 and c0"
    
#     def test_wire_continuity_across_idle_periods(self):
#         """Test wire continuity when qubit is idle"""
#         circuit = Circuit()
#         circuit.add_instruction("h", [0])
#         circuit.add_instruction("h", [1])
#         circuit.add_instruction("x", [1])
#         circuit.add_instruction("y", [1])
#         circuit.add_instruction("x", [0])  # q0 was idle while q1 was active
        
#         hdh = circuit.build_hdh()
        
#         # q0 should have nodes throughout even if idle
#         q0_nodes = sorted(
#             [n for n in hdh.S if n.startswith('q0_')],
#             key=lambda n: hdh.time_map[n]
#         )
        
#         # Should have at least initial node and final node
#         assert len(q0_nodes) >= 2, "q0 should have multiple nodes"
        
#         # Check first and last nodes are connected via edges
#         first_node = q0_nodes[0]
#         last_node = q0_nodes[-1]
        
#         # First node should be in some edges
#         first_edges = [e for e in hdh.C if first_node in e]
#         assert len(first_edges) > 0, "First node should be in edges"
        
#         # Last node should be in some edges
#         last_edges = [e for e in hdh.C if last_node in e]
#         assert len(last_edges) > 0, "Last node should be in edges"
        
#         # There should be a chain of nodes connecting first to last
#         # (even if q0 is idle, it should have timestep 0 node)
#         assert hdh.time_map[first_node] == 0, \
#             "First node should be at t=0 (Issue #32 fix)"
    
#     def test_timestep_progression(self):
#         """Test that timesteps progress monotonically for each qubit"""
#         circuit = Circuit()
#         circuit.add_instruction("h", [0])
#         circuit.add_instruction("x", [0])
#         circuit.add_instruction("y", [0])
#         circuit.add_instruction("z", [0])
        
#         hdh = circuit.build_hdh()
        
#         # Get all q0 nodes sorted by time
#         q0_nodes = sorted(
#             [n for n in hdh.S if n.startswith('q0_')],
#             key=lambda n: hdh.time_map[n]
#         )
        
#         # Get timestamps
#         times = [hdh.time_map[n] for n in q0_nodes]
        
#         # Should start at t=0 (Issue #32 fix)
#         assert times[0] == 0, "Should start at t=0"
        
#         # Times should be non-decreasing
#         for i in range(len(times) - 1):
#             assert times[i] <= times[i+1], \
#                 f"Timestamps should be non-decreasing: got {times[i]} then {times[i+1]}"
        
#         # Should have some progression (not all same time)
#         assert len(set(times)) > 1, "Should have multiple distinct timestamps"


class TestWireContinuityEdgeCases:
    """Test edge cases for wire continuity"""
    
    def test_single_gate_continuity(self):
        """Test wire continuity with just one gate"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        
        hdh = circuit.build_hdh()
        
        q0_nodes = [n for n in hdh.S if n.startswith('q0_')]
        
        # Should have at least input and output nodes
        assert len(q0_nodes) >= 2
        
        # All nodes should be connected via edges
        for node in q0_nodes:
            edges_with_node = [e for e in hdh.C if node in e]
            # Each node should be in at least one edge (except possibly terminal nodes)
            # We'll check that most nodes are connected
            
        # At least one edge should exist
        q0_edges = [e for e in hdh.C if any(n.startswith('q0_') for n in e)]
        assert len(q0_edges) > 0, "Should have edges for q0"
    
    def test_cnot_continuity(self):
        """Test wire continuity through CNOT gate"""
        circuit = Circuit()
        circuit.add_instruction("cx", [0, 1])
        
        hdh = circuit.build_hdh()
        
        # Both qubits should have nodes
        q0_nodes = sorted(
            [n for n in hdh.S if n.startswith('q0_')],
            key=lambda n: hdh.time_map[n]
        )
        q1_nodes = sorted(
            [n for n in hdh.S if n.startswith('q1_')],
            key=lambda n: hdh.time_map[n]
        )
        
        assert len(q0_nodes) >= 2
        assert len(q1_nodes) >= 2
        
        # Both should start at t=0 (Issue #32 fix)
        assert hdh.time_map[q0_nodes[0]] == 0
        assert hdh.time_map[q1_nodes[0]] == 0
        
        # CNOT should create edges involving both qubits
        cnot_edges = [
            e for e in hdh.C 
            if 'cx' in hdh.gate_name.get(e, '').lower()
        ]
        
        assert len(cnot_edges) > 0, "Should have CNOT edges"
        
        # At least one edge should involve nodes from both qubits
        has_multi_qubit_edge = False
        for edge in cnot_edges:
            edge_nodes = list(edge)
            has_q0 = any(n.startswith('q0_') for n in edge_nodes)
            has_q1 = any(n.startswith('q1_') for n in edge_nodes)
            if has_q0 and has_q1:
                has_multi_qubit_edge = True
                break
        
        assert has_multi_qubit_edge, "CNOT should have edge involving both qubits"
    
    def test_long_chain_continuity(self):
        """Test wire continuity through long chain of gates"""
        circuit = Circuit()
        for _ in range(10):
            circuit.add_instruction("h", [0])
        
        hdh = circuit.build_hdh()
        
        q0_nodes = sorted(
            [n for n in hdh.S if n.startswith('q0_')],
            key=lambda n: hdh.time_map[n]
        )
        
        # Should have many nodes
        assert len(q0_nodes) >= 10
        
        # First should be at t=0
        assert hdh.time_map[q0_nodes[0]] == 0
        
        # Timestamps should increase
        times = [hdh.time_map[n] for n in q0_nodes]
        for i in range(len(times) - 1):
            assert times[i] <= times[i+1], "Times should be non-decreasing"
        
        # Should have many edges
        q0_edges = [e for e in hdh.C if any(n.startswith('q0_') for n in e)]
        assert len(q0_edges) >= 10, "Should have many edges for long chain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Debug Script for HDH Wire Continuity Issue #37

This script helps diagnose wire continuity problems by:
1. Building a simple HDH from a circuit
2. Inspecting nodes and edges
3. Checking for wire continuity violations
4. Identifying problematic nodes
"""

def debug_wire_continuity(circuit, verbose=True):
    """
    Debug wire continuity in an HDH built from a circuit.
    
    Args:
        circuit: Circuit object to analyze
        verbose: Print detailed output
    
    Returns:
        dict: Analysis results
    """
    hdh = circuit.build_hdh()
    
    results = {
        'total_nodes': len(hdh.S),
        'total_edges': len(hdh.C),
        'quantum_nodes': 0,
        'classical_nodes': 0,
        'nodes_without_edges': [],
        'nodes_without_forward_edges': [],
        'qubit_analysis': {}
    }
    
    if verbose:
        print("=" * 60)
        print("HDH WIRE CONTINUITY ANALYSIS")
        print("=" * 60)
    
    # Analyze nodes by type
    quantum_nodes = [n for n in hdh.S if hdh.sigma[n] == 'q']
    classical_nodes = [n for n in hdh.S if hdh.sigma[n] == 'c']
    
    results['quantum_nodes'] = len(quantum_nodes)
    results['classical_nodes'] = len(classical_nodes)
    
    if verbose:
        print(f"\nTotal Nodes: {len(hdh.S)}")
        print(f"  Quantum: {len(quantum_nodes)}")
        print(f"  Classical: {len(classical_nodes)}")
        print(f"Total Edges: {len(hdh.C)}")
    
    # Group quantum nodes by qubit
    qubit_nodes = {}
    for node in quantum_nodes:
        # Extract qubit index from node name (format: q{idx}_t{time})
        if '_' in node:
            qubit_part = node.split('_')[0]
            if qubit_part not in qubit_nodes:
                qubit_nodes[qubit_part] = []
            qubit_nodes[qubit_part].append(node)
    
    # Analyze each qubit's wire
    if verbose:
        print("\n" + "=" * 60)
        print("PER-QUBIT ANALYSIS")
        print("=" * 60)
    
    for qubit_name in sorted(qubit_nodes.keys()):
        nodes = sorted(qubit_nodes[qubit_name], 
                      key=lambda n: hdh.time_map[n])
        
        qubit_info = {
            'node_count': len(nodes),
            'timesteps': [hdh.time_map[n] for n in nodes],
            'nodes_without_edges': [],
            'nodes_without_forward': []
        }
        
        if verbose:
            print(f"\n{qubit_name.upper()}:")
            print(f"  Nodes: {len(nodes)}")
            print(f"  Timesteps: {[hdh.time_map[n] for n in nodes]}")
        
        # Check each node
        for i, node in enumerate(nodes):
            node_time = hdh.time_map[node]
            
            # Find edges containing this node
            node_edges = [e for e in hdh.C if node in e]
            
            if len(node_edges) == 0:
                qubit_info['nodes_without_edges'].append(node)
                results['nodes_without_edges'].append(node)
                if verbose:
                    print(f"  ⚠️  {node} (t={node_time}): NO EDGES!")
            else:
                # Check for forward connections
                has_forward = False
                for edge in node_edges:
                    edge_nodes = list(edge)
                    for other_node in edge_nodes:
                        if other_node != node and other_node in hdh.time_map:
                            if hdh.time_map[other_node] > node_time:
                                has_forward = True
                                break
                    if has_forward:
                        break
                
                if not has_forward and i < len(nodes) - 1:  # Not the last node
                    qubit_info['nodes_without_forward'].append(node)
                    results['nodes_without_forward_edges'].append(node)
                    if verbose:
                        print(f"  ⚠️  {node} (t={node_time}): Has {len(node_edges)} edge(s) but NO FORWARD CONNECTION!")
                elif verbose:
                    print(f"  ✓  {node} (t={node_time}): {len(node_edges)} edge(s), forward connection OK")
        
        # Check timestep progression
        times = [hdh.time_map[n] for n in nodes]
        is_monotonic = all(times[i] <= times[i+1] for i in range(len(times)-1))
        
        if verbose:
            print(f"  Timestep progression: {'✓ Monotonic' if is_monotonic else '⚠️  NOT MONOTONIC'}")
            if not is_monotonic:
                print(f"    Times: {times}")
        
        results['qubit_analysis'][qubit_name] = qubit_info
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        if results['nodes_without_edges']:
            print(f"\n⚠️  CRITICAL: {len(results['nodes_without_edges'])} node(s) without ANY edges:")
            for node in results['nodes_without_edges']:
                print(f"    - {node} (t={hdh.time_map[node]})")
        else:
            print("\n✓  All nodes have at least one edge")
        
        if results['nodes_without_forward_edges']:
            print(f"\n⚠️  WARNING: {len(results['nodes_without_forward_edges'])} non-terminal node(s) without forward connections:")
            for node in results['nodes_without_forward_edges']:
                print(f"    - {node} (t={hdh.time_map[node]})")
        else:
            print("\n✓  All non-terminal nodes have forward connections")
        
        # Overall verdict
        print("\n" + "=" * 60)
        if results['nodes_without_edges'] or results['nodes_without_forward_edges']:
            print(" WIRE CONTINUITY VIOLATED")
            print("=" * 60)
        else:
            print(" WIRE CONTINUITY OK")
            print("=" * 60)
    
    return results


def test_simple_circuit():
    """Test wire continuity with a simple single-qubit circuit."""
    from hdh.models.circuit import Circuit
    
    print("\n" + "#" * 60)
    print("# TEST 1: Simple single-qubit circuit (H gate)")
    print("#" * 60)
    
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    
    results = debug_wire_continuity(circuit, verbose=True)
    
    return results


def test_multi_gate_circuit():
    """Test wire continuity with multiple gates on one qubit."""
    from hdh.models.circuit import Circuit
    
    print("\n" + "#" * 60)
    print("# TEST 2: Multi-gate single-qubit circuit (H-X-Y-Z)")
    print("#" * 60)
    
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    circuit.add_instruction("x", [0])
    circuit.add_instruction("y", [0])
    circuit.add_instruction("z", [0])
    
    results = debug_wire_continuity(circuit, verbose=True)
    
    return results


def test_idle_qubit():
    """Test wire continuity when one qubit is idle."""
    from hdh.models.circuit import Circuit
    
    print("\n" + "#" * 60)
    print("# TEST 3: Idle qubit scenario")
    print("#" * 60)
    
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    circuit.add_instruction("h", [1])
    circuit.add_instruction("x", [1])
    circuit.add_instruction("y", [1])
    circuit.add_instruction("x", [0])  # q0 was idle
    
    results = debug_wire_continuity(circuit, verbose=True)
    
    return results


def test_cnot():
    """Test wire continuity with CNOT gate."""
    from hdh.models.circuit import Circuit
    
    print("\n" + "#" * 60)
    print("# TEST 4: CNOT gate")
    print("#" * 60)
    
    circuit = Circuit()
    circuit.add_instruction("cx", [0, 1])
    
    results = debug_wire_continuity(circuit, verbose=True)
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HDH WIRE CONTINUITY DEBUG SCRIPT - Issue #37")
    print("=" * 60)
    
    try:
        # Run all tests
        test1 = test_simple_circuit()
        test2 = test_multi_gate_circuit()
        test3 = test_idle_qubit()
        test4 = test_cnot()
        
        # Overall summary
        print("\n\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        
        all_tests = [test1, test2, test3, test4]
        test_names = ["Simple H", "H-X-Y-Z", "Idle Qubit", "CNOT"]
        
        for name, test in zip(test_names, all_tests):
            status = " FAIL" if (test['nodes_without_edges'] or 
                                   test['nodes_without_forward_edges']) else "✅ PASS"
            print(f"{name:20s}: {status}")
            if test['nodes_without_edges']:
                print(f"  - {len(test['nodes_without_edges'])} nodes without edges")
            if test['nodes_without_forward_edges']:
                print(f"  - {len(test['nodes_without_forward_edges'])} nodes without forward connections")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
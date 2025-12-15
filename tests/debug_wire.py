"""
Detailed diagnosis to identify exactly which nodes are problematic
"""

def detailed_single_gate_analysis():
    """Analyze a simple H gate in detail to see the node/edge structure."""
    from hdh.models.circuit import Circuit
    
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: Single H Gate on q0")
    print("=" * 70)
    
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    hdh = circuit.build_hdh()
    
    # Get all q0 nodes
    q0_nodes = sorted([n for n in hdh.S if n.startswith('q0_')],
                      key=lambda n: hdh.time_map[n])
    
    print(f"\nTotal q0 nodes: {len(q0_nodes)}")
    print(f"Nodes: {q0_nodes}")
    print(f"Times: {[hdh.time_map[n] for n in q0_nodes]}")
    
    # Analyze each node in detail
    print("\n" + "-" * 70)
    print("PER-NODE ANALYSIS:")
    print("-" * 70)
    
    for node in q0_nodes:
        node_time = hdh.time_map[node]
        edges = [e for e in hdh.C if node in e]
        
        print(f"\n{node} (t={node_time}):")
        print(f"  Number of edges: {len(edges)}")
        
        if len(edges) == 0:
            print(f"  ⚠️  NO EDGES - This node is DISCONNECTED!")
        else:
            for i, edge in enumerate(edges, 1):
                edge_nodes = list(edge)
                edge_type = hdh.tau.get(edge, '?')
                edge_name = hdh.gate_name.get(edge, '?')
                
                print(f"  Edge {i}:")
                print(f"    Type: {edge_type}, Gate: {edge_name}")
                print(f"    Nodes: {edge_nodes}")
                
                # Check times of connected nodes
                for other_node in edge_nodes:
                    if other_node != node and other_node in hdh.time_map:
                        other_time = hdh.time_map[other_node]
                        direction = "forward" if other_time > node_time else "backward" if other_time < node_time else "same"
                        print(f"      -> {other_node} (t={other_time}, {direction})")
    
    # Check for expected pattern
    print("\n" + "-" * 70)
    print("EXPECTED PATTERN CHECK:")
    print("-" * 70)
    
    print("\nFor a single H gate, we expect:")
    print("  - Initial node at t=0")
    print("  - Output node at t=1")  
    print("  - Edge connecting them with gate='h'")
    print("  - Possibly terminal nodes")
    
    print(f"\nActual node count: {len(q0_nodes)}")
    
    if len(q0_nodes) >= 2:
        first_node = q0_nodes[0]
        second_node = q0_nodes[1]
        
        print(f"\nFirst two nodes: {first_node} (t={hdh.time_map[first_node]}), "
              f"{second_node} (t={hdh.time_map[second_node]})")
        
        # Check if they're connected
        first_edges = [e for e in hdh.C if first_node in e]
        common_edges = [e for e in first_edges if second_node in e]
        
        if common_edges:
            print(f"✓ First two nodes ARE connected by {len(common_edges)} edge(s)")
            for edge in common_edges:
                print(f"  Gate: {hdh.gate_name.get(edge, '?')}")
        else:
            print(f"⚠️  First two nodes are NOT connected!")
    
    # Identify the problematic nodes
    print("\n" + "-" * 70)
    print("PROBLEMATIC NODES:")
    print("-" * 70)
    
    disconnected = []
    no_forward = []
    
    for i, node in enumerate(q0_nodes):
        edges = [e for e in hdh.C if node in e]
        node_time = hdh.time_map[node]
        
        if len(edges) == 0:
            disconnected.append(node)
            print(f"⚠️  {node} (t={node_time}): COMPLETELY DISCONNECTED")
        else:
            # Check for forward connections (if not last node)
            if i < len(q0_nodes) - 1:
                has_forward = any(
                    any(hdh.time_map.get(n, -1) > node_time for n in e if n != node)
                    for e in edges
                )
                if not has_forward:
                    no_forward.append(node)
                    print(f"⚠️  {node} (t={node_time}): Has edges but NO FORWARD connection")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY:")
    print("=" * 70)
    print(f"Disconnected nodes: {len(disconnected)}")
    print(f"Nodes without forward connections: {len(no_forward)}")
    
    if disconnected:
        print(f"\nDisconnected: {disconnected}")
        print(f"Times: {[hdh.time_map[n] for n in disconnected]}")
    
    if no_forward:
        print(f"\nNo forward connection: {no_forward}")
        print(f"Times: {[hdh.time_map[n] for n in no_forward]}")
    
    # Hypothesis about which code created these nodes
    print("\n" + "=" * 70)
    print("HYPOTHESIS:")
    print("=" * 70)
    print("The disconnected nodes are likely created by the code at")
    print("circuit.py lines 155-161 which adds nodes without edges.")
    print("\nThe nodes WITH edges are likely created by the code at")
    print("circuit.py lines 240-251 which properly adds both nodes AND edges.")


def compare_single_vs_two_qubit():
    """Compare single-qubit (failing) vs two-qubit (passing) gate structure."""
    from hdh.models.circuit import Circuit
    
    print("\n\n" + "=" * 70)
    print("COMPARISON: Single-qubit (H) vs Two-qubit (CNOT)")
    print("=" * 70)
    
    # Single qubit
    print("\n" + "-" * 70)
    print("SINGLE-QUBIT GATE (H on q0) - FAILS:")
    print("-" * 70)
    
    circuit1 = Circuit()
    circuit1.add_instruction("h", [0])
    hdh1 = circuit1.build_hdh()
    
    q0_nodes_h = [n for n in hdh1.S if n.startswith('q0_')]
    edges_h = len([e for e in hdh1.C if any(n.startswith('q0_') for n in e)])
    
    print(f"q0 nodes: {len(q0_nodes_h)}")
    print(f"q0 edges: {edges_h}")
    print(f"Ratio: {edges_h / len(q0_nodes_h) if q0_nodes_h else 0:.2f} edges per node")
    
    # Two qubit
    print("\n" + "-" * 70)
    print("TWO-QUBIT GATE (CNOT) - PASSES:")
    print("-" * 70)
    
    circuit2 = Circuit()
    circuit2.add_instruction("cx", [0, 1])
    hdh2 = circuit2.build_hdh()
    
    q0_nodes_cx = [n for n in hdh2.S if n.startswith('q0_')]
    q1_nodes_cx = [n for n in hdh2.S if n.startswith('q1_')]
    edges_cx = len(hdh2.C)
    
    print(f"q0 nodes: {len(q0_nodes_cx)}")
    print(f"q1 nodes: {len(q1_nodes_cx)}")
    print(f"Total edges: {edges_cx}")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCE:")
    print("=" * 70)
    print("Single-qubit gates create disconnected nodes")
    print("Two-qubit gates create properly connected nodes")
    print("\nThis confirms the bug is in SINGLE-QUBIT gate handling,")
    print("specifically in circuit.py around lines 155-161 and/or 240-251")



if __name__ == "__main__":
    detailed_single_gate_analysis()
    compare_single_vs_two_qubit()
    
    """
    This script shows EXACTLY what needs to be fixed in circuit.py

    Based on the debug results:
    - Single-qubit gates create 2 disconnected nodes
    - These are likely from lines 155-161 which add nodes without edges

    The fix: Add hyperedges after creating the nodes
    """

    print("=" * 70)
    print("EXACT FIX NEEDED FOR circuit.py")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("LOCATION: hdh/models/circuit.py, lines ~155-161")
    print("-" * 70)

    print("\nCURRENT CODE (BROKEN):")
    print("-" * 70)
    print("""
    mid_id   = f"{qname}_t{t1}"
    final_id = f"{qname}_t{t2}"
    post_id  = f"{qname}_t{t3}"

    hdh.add_node(mid_id,   "q", t1, node_real=cond_flag)
    hdh.add_node(final_id, "q", t2, node_real=cond_flag)
    hdh.add_node(post_id,  "q", t3, node_real=cond_flag)
    # ❌ PROBLEM: Nodes created but NO EDGES added!
    """)

    print("\nFIXED CODE (OPTION 1 - Add the missing edges):")
    print("-" * 70)
    print("""
    mid_id   = f"{qname}_t{t1}"
    final_id = f"{qname}_t{t2}"
    post_id  = f"{qname}_t{t3}"

    hdh.add_node(mid_id,   "q", t1, node_real=cond_flag)
    hdh.add_node(final_id, "q", t2, node_real=cond_flag)
    hdh.add_node(post_id,  "q", t3, node_real=cond_flag)

    # ✅ FIX: Add hyperedges to connect the nodes
    edge1 = hdh.add_hyperedge({mid_id, final_id}, "q", name=name, node_real=cond_flag)
    edge2 = hdh.add_hyperedge({final_id, post_id}, "q", name=name, node_real=cond_flag)
    edges.append(edge1)
    edges.append(edge2)
    """)

    print("\nFIXED CODE (OPTION 2 - Remove redundant code):")
    print("-" * 70)
    print("""
    # If these nodes are redundant with the ones created at lines 240-251,
    # simply COMMENT OUT or DELETE lines 155-161:

    # mid_id   = f"{qname}_t{t1}"
    # final_id = f"{qname}_t{t2}"
    # post_id  = f"{qname}_t{t3}"
    # 
    # hdh.add_node(mid_id,   "q", t1, node_real=cond_flag)
    # hdh.add_node(final_id, "q", t2, node_real=cond_flag)
    # hdh.add_node(post_id,  "q", t3, node_real=cond_flag)
    """)

    print("\n" + "=" * 70)
    print("DECISION GUIDE")
    print("=" * 70)
    print("""
    To decide between Option 1 and Option 2, you need to understand:

    1. WHAT is this code for? 
    - Is it for multi-qubit gates?
    - Is it for conditional operations?
    - Is it creating auxiliary nodes?

    2. WHY create 3 nodes (t1, t2, t3)?
    - What do mid_id, final_id, post_id represent?

    3. Does this code DUPLICATE the work of lines 240-251?
    - Lines 240-251 already handle single-qubit gates WITH edges

    ACTION ITEMS:
    1. Find the context around lines 155-161 in circuit.py
    2. Read the surrounding code and comments
    3. Determine if these nodes serve a unique purpose
    4. If YES -> Use Option 1 (add edges)
    5. If NO -> Use Option 2 (remove redundant code)
    """)

    print("\n" + "=" * 70)
    print("VERIFICATION AFTER FIX")
    print("=" * 70)
    print("""
    After applying the fix, run:

    1. The debug script:
    python debug_wire_continuity.py
    
    Expected output:
    Simple H            : ✅ PASS
    H-X-Y-Z             : ✅ PASS
    Idle Qubit          : ✅ PASS
    CNOT                : ✅ PASS

    2. The actual tests:
    pytest tests/test_constraints.py::TestWireContinuityFixed -v
    
    All tests should pass.

    3. If tests still fail, run:
    python detailed_diagnosis.py
    
    This will show exactly which nodes are still problematic.
    """)

    print("\n" + "=" * 70)
    print("ADDITIONAL INVESTIGATION NEEDED")
    print("=" * 70)
    print("""
    The comment on line 240 suggests another potential issue:

        t_in = last_gate_input_time[qubit] # BREAK?: t_in = qubit_time[qubit]

    This needs investigation:
    1. Test with BOTH variable options
    2. See which one maintains proper wire continuity
    3. The correct choice depends on how time tracking works in your system

    You may need to test both scenarios:
    - Using last_gate_input_time[qubit]
    - Using qubit_time[qubit]

    And see which one produces the correct HDH structure.
    """)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
    1. Locate circuit.py lines 155-161 in your codebase
    2. Read the surrounding context
    3. Apply either Option 1 or Option 2 fix
    4. Run debug_wire_continuity.py to verify
    5. Uncomment and run the failing tests
    6. If tests pass, close issue #37! ✅
    """)

#!/usr/bin/env python3
"""
Diagnostic for the two failing tests
"""
from hdh.models.circuit import Circuit

def test_cardinality_growth():
    """Test the cardinality growth issue"""
    print("=" * 70)
    print("TEST: Cardinality Growth")
    print("=" * 70)
    
    # Shallow circuit
    circuit1 = Circuit()
    circuit1.add_instruction("h", [0])
    hdh1 = circuit1.build_hdh()
    nodes1 = len([n for n in hdh1.S if hdh1.sigma[n] == 'q'])
    
    print(f"\n1 H gate:")
    print(f"  Q0 nodes: {nodes1}")
    q0_nodes_1 = sorted([n for n in hdh1.S if n.startswith('q0_')],
                        key=lambda n: hdh1.time_map[n])
    print(f"  Nodes: {q0_nodes_1}")
    
    # Deeper circuit
    circuit2 = Circuit()
    for i in range(10):
        circuit2.add_instruction("h", [0])
    hdh2 = circuit2.build_hdh()
    nodes2 = len([n for n in hdh2.S if hdh2.sigma[n] == 'q'])
    
    print(f"\n10 H gates:")
    print(f"  Q0 nodes: {nodes2}")
    q0_nodes_10 = sorted([n for n in hdh2.S if n.startswith('q0_')],
                         key=lambda n: hdh2.time_map[n])
    print(f"  Nodes: {q0_nodes_10}")
    
    print(f"\nComparison:")
    print(f"  nodes2 > nodes1: {nodes2 > nodes1} (should be True)")
    print(f"  nodes2 < nodes1 * 20: {nodes2 < nodes1 * 20} (should be True)")
    print(f"  Ratio: {nodes2 / nodes1:.1f}x")
    
    if nodes2 <= nodes1:
        print("\n❌ FAIL: Deeper circuit doesn't have more nodes!")
    elif nodes2 >= nodes1 * 20:
        print(f"\n❌ FAIL: Growth too high! {nodes2} >= {nodes1 * 20}")
    else:
        print("\n✓ PASS")
    
    return nodes2 > nodes1 and nodes2 < nodes1 * 20


def test_long_chain():
    """Test long chain of gates"""
    print("\n\n" + "=" * 70)
    print("TEST: Long Chain Continuity")
    print("=" * 70)
    
    circuit = Circuit()
    for _ in range(10):
        circuit.add_instruction("h", [0])
    
    hdh = circuit.build_hdh()
    
    q0_nodes = sorted([n for n in hdh.S if n.startswith('q0_')],
                      key=lambda n: hdh.time_map[n])
    
    print(f"\n10 H gates on q0:")
    print(f"  Total nodes: {len(q0_nodes)}")
    print(f"  Should have >= 10 nodes")
    
    # Check each node has edges
    print(f"\nNode analysis:")
    disconnected = []
    for i, node in enumerate(q0_nodes):
        edges = [e for e in hdh.C if node in e]
        status = "✓" if len(edges) > 0 else "❌"
        print(f"  {status} {node} (t={hdh.time_map[node]}): {len(edges)} edges")
        if len(edges) == 0:
            disconnected.append(node)
    
    # Check timestamps increase
    times = [hdh.time_map[n] for n in q0_nodes]
    monotonic = all(times[i] <= times[i+1] for i in range(len(times)-1))
    print(f"\nTimestep progression: {'✓ Monotonic' if monotonic else '❌ NOT monotonic'}")
    print(f"  Times: {times}")
    
    # Check edges
    q0_edges = [e for e in hdh.C if any(n.startswith('q0_') for n in e)]
    print(f"\nTotal q0 edges: {len(q0_edges)}")
    print(f"  Should have >= 10 edges")
    
    if len(q0_nodes) < 10:
        print(f"\n❌ FAIL: Not enough nodes ({len(q0_nodes)} < 10)")
        return False
    elif disconnected:
        print(f"\n❌ FAIL: {len(disconnected)} disconnected nodes: {disconnected}")
        return False
    elif len(q0_edges) < 10:
        print(f"\n❌ FAIL: Not enough edges ({len(q0_edges)} < 10)")
        return False
    elif not monotonic:
        print("\n❌ FAIL: Times not monotonic")
        return False
    else:
        print("\n✓ PASS")
        return True


def compare_structures():
    """Compare single gate vs chain structure"""
    print("\n\n" + "=" * 70)
    print("COMPARISON: Single H vs Chain of H")
    print("=" * 70)
    
    # Single H
    circuit1 = Circuit()
    circuit1.add_instruction("h", [0])
    hdh1 = circuit1.build_hdh()
    q0_1 = [n for n in hdh1.S if n.startswith('q0_')]
    
    # Two H gates
    circuit2 = Circuit()
    circuit2.add_instruction("h", [0])
    circuit2.add_instruction("h", [0])
    hdh2 = circuit2.build_hdh()
    q0_2 = sorted([n for n in hdh2.S if n.startswith('q0_')],
                  key=lambda n: hdh2.time_map[n])
    
    print(f"\n1 H gate: {len(q0_1)} nodes")
    print(f"2 H gates: {len(q0_2)} nodes")
    print(f"Expected pattern: each H should add ~2 nodes (input + output)")
    
    print(f"\n2 H gate details:")
    for node in q0_2:
        edges = [e for e in hdh2.C if node in e]
        print(f"  {node} (t={hdh2.time_map[node]}): {len(edges)} edges")


if __name__ == "__main__":
    cardinality_ok = test_cardinality_growth()
    chain_ok = test_long_chain()
    compare_structures()
    
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cardinality Growth: {'✓ PASS' if cardinality_ok else '❌ FAIL'}")
    print(f"Long Chain:         {'✓ PASS' if chain_ok else '❌ FAIL'}")
    print("=" * 70)
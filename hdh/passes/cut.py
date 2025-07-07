import networkx as nx
import networkx.algorithms.community as nx_comm
import metis
print(metis.__version__)
from typing import List, Set, Union, DefaultDict, Optional, Dict
from hdh.hdh import HDH
from qiskit import QuantumCircuit
import re

class AncillaAllocator:
    def __init__(self):
        self.counter = 0
    def new(self, base: str, time: int):
        name = f"{base}_anc{self.counter}_t{time}"
        self.counter += 1
        return name

def extract_qidx(n):
    # Match strings like 'q0', 'q3_t12', 'qA_anc0_t18', etc.
    m = re.search(r'q(?:[A-Za-z_]*?)(\d+)', n)
    if m:
        return int(m.group(1))
    raise ValueError(f"[ERROR] extract_qidx failed on: {n}")

def extract_cidx(n):
    # Match strings like 'c1', 'c9_t0', 'c_anc3_t12', etc.
    m = re.search(r'c(?:[A-Za-z_]*?)(\d+)', n)
    if m:
        return int(m.group(1))
    raise ValueError(f"[ERROR] extract_cidx failed on: {n}")

def get_logical_qubit(node_id: str) -> str:
    return node_id.split('_')[0]

def compute_cut(hdh: HDH, num_parts: int) -> List[Set[str]]:
    """
    Use METIS to partition HDH nodes into disjoint blocks.
    
    Returns a list of disjoint sets of node IDs.
    """
    # 1. Build undirected graph from HDH
    G = nx.Graph()
    G.add_nodes_from(hdh.S)
    
    for edge in hdh.C:
        edge_nodes = list(edge)
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                G.add_edge(edge_nodes[i], edge_nodes[j])

    # 2. Convert to METIS-compatible graph (requires contiguous integer node IDs)
    node_list = list(G.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(node_list)}
    idx_node_map = {idx: node for node, idx in node_idx_map.items()}

    metis_graph = nx.relabel_nodes(G, node_idx_map, copy=True)
    
    # 3. Call METIS
    _, parts = metis.part_graph(metis_graph, nparts=num_parts)
    
    # 4. Build partition sets
    partition = [set() for _ in range(num_parts)]
    for idx, part in enumerate(parts):
        node_id = idx_node_map[idx]
        partition[part].add(node_id)
    
    return partition

def select_comm_primitive(role, node_type, allowed):
    if role == "teledata":
        return "tp" if "tp" in allowed["quantum"] else "cat"
    elif role == "telegate":
        return "cat"
    elif role == "classical":
        return "ccom" if "ccom" in allowed["classical"] else "crep"
    raise ValueError(f"Unknown role: {role}")

def extract_cidx(n):
    m = re.search(r'c(?:[A-Za-z_]+)?(\d+)', n)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract classical bit index from: {n}")

def cut_and_rewrite_hdh(
    hdh,
    num_parts: int,
    allowed_primitives: Dict[str, Set[str]],
    insert_qiskit_circuits: bool = False,
    qiskit_primitives: Optional[Dict[str, QuantumCircuit]] = None
) -> List:

    def select_comm_primitive(role, node_type, allowed, primitive_lib=None):
        if role == "teledata":
            label = "tp" if "tp" in allowed["quantum"] else "cat"
        elif role == "telegate":
            label = "cat"
        elif role == "classical":
            label = "ccom" if "ccom" in allowed["classical"] else "crep"
        else:
            raise ValueError(f"Unknown role: {role}")

        circ = None
        if primitive_lib:
            circ = primitive_lib.get(label)
            if not isinstance(circ, QuantumCircuit):
                print(f"Warning: primitive '{label}' is not a valid Qiskit circuit. Skipping circuit insert.")
                circ = None
        return label, circ

    partitions = compute_cut(hdh, num_parts)
    node_to_part = {node: i for i, part in enumerate(partitions) for node in part}
    cut_edges = [e for e in hdh.C if len({node_to_part[n] for n in e}) > 1]
    print(f"[DEBUG] Number of cut edges: {len(cut_edges)}")

    partitioned_hdhs = [hdh.__class__() for _ in range(num_parts)]
    anc = AncillaAllocator()

    # Copy nodes
    for i, part in enumerate(partitions):
        for node in part:
            partitioned_hdhs[i].add_node(node, hdh.sigma[node], hdh.time_map[node])

    # Copy non-cut edges
    for edge in hdh.C - set(cut_edges):
        parts = {node_to_part[n] for n in edge}
        p = parts.pop()
        new_edge = partitioned_hdhs[p].add_hyperedge(
            edge,
            hdh.tau[edge],
            name=hdh.gate_name.get(edge),
            role=hdh.edge_role.get(edge)
        )
        if edge in hdh.edge_args:
            partitioned_hdhs[p].edge_args[new_edge] = hdh.edge_args[edge]
        if hasattr(hdh, "edge_metadata") and edge in hdh.edge_metadata:
            partitioned_hdhs[p].edge_metadata[new_edge] = hdh.edge_metadata[edge]

    # Handle cut edges
    for edge in cut_edges:
        parts = list({node_to_part[n] for n in edge})
        if len(parts) != 2:
            continue  # Only handle bipartition edges

        print(f"[DEBUG] Cutting edge: {edge} with name: {hdh.gate_name.get(edge)}")

        nodes = list(edge)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                if node_to_part[n1] != node_to_part[n2]:
                    src, dst = (n1, n2) if hdh.time_map[n1] <= hdh.time_map[n2] else (n2, n1)
                    t_src, t_dst = hdh.time_map[src], hdh.time_map[dst]
                    qtype = hdh.sigma[src]
                    role = hdh.edge_role.get(edge)

                    if role is None:
                        label = hdh.gate_name.get(edge, "")
                        if label in {"cx", "cz"}:
                            role = "telegate"
                        else:
                            role = "teledata"

                    primitive, qiskit_circuit = select_comm_primitive(
                        role, qtype, allowed_primitives, qiskit_primitives if insert_qiskit_circuits else None
                    )

                    print(f"[DEBUG] Assigning primitive '{primitive}' for edge {edge}, role: {role}")

                    if insert_qiskit_circuits and qiskit_circuit is None:
                        qiskit_circuit = default_primitive(primitive)

                    src_part = node_to_part[src]
                    dst_part = node_to_part[dst]
                    if src_part == dst_part:
                        continue

                    src_stub = f"{src}_send"
                    dst_stub = f"{dst}_recv"

                    partitioned_hdhs[src_part].add_node(src_stub, qtype, t_src)
                    partitioned_hdhs[dst_part].add_node(dst_stub, qtype, t_dst)

                    partitioned_hdhs[src_part].add_hyperedge({src, src_stub}, qtype)
                    partitioned_hdhs[dst_part].add_hyperedge({dst_stub, dst}, qtype)

                    comm_nodes = {src_stub, dst_stub}

                    anc_qs = []
                    anc_cs = []

                    if insert_qiskit_circuits and qiskit_circuit:
                        print(f"[DEBUG] Inserting circuit for edge {edge}: {qiskit_circuit.name}")
                        for _ in range(qiskit_circuit.num_qubits):
                            anc_q = anc.new("qA", t_src)
                            partitioned_hdhs[src_part].add_node(anc_q, 'q', t_src)
                            comm_nodes.add(anc_q)
                            anc_qs.append(anc_q)
                        for _ in range(qiskit_circuit.num_clbits):
                            anc_c = anc.new("c", t_src)
                            partitioned_hdhs[src_part].add_node(anc_c, 'c', t_src)
                            anc_cs.append(anc_c)

                    comm_edge = partitioned_hdhs[src_part].add_hyperedge(
                        comm_nodes,
                        primitive,
                        name=primitive,
                        role=role
                    )

                    if insert_qiskit_circuits and qiskit_circuit:
                        partitioned_hdhs[src_part].edge_args[comm_edge] = qiskit_circuit

                        q_order = [src_stub, dst_stub] + anc_qs
                        qubit_candidates = [n for n in q_order if n.startswith('q')]
                        if len(qubit_candidates) < qiskit_circuit.num_qubits:
                            raise ValueError(
                                f"Edge {edge} ({primitive}) expects {qiskit_circuit.num_qubits} qubits, "
                                f"but got only {len(qubit_candidates)}: {qubit_candidates}"
                            )

                        qubit_idxs = [extract_qidx(n) for n in qubit_candidates[:qiskit_circuit.num_qubits]]
                        cbit_idxs = [extract_cidx(n) for n in anc_cs[:qiskit_circuit.num_clbits]]

                        partitioned_hdhs[src_part].edge_metadata[comm_edge] = {
                            "qubits": qubit_idxs,
                            "cbits": cbit_idxs,
                            "timestep": t_src,
                            "gate": qiskit_circuit.to_instruction()
                        }

                    partitioned_hdhs[src_part].motifs[comm_edge] = {
                        "type": primitive,
                        "role": role,
                        "source": src,
                        "target": dst,
                        "qtype": qtype,
                        "time_src": t_src,
                        "time_dst": t_dst,
                        "ancilla_qubits": anc_qs,
                        "ancilla_bits": anc_cs
                    }

    return partitioned_hdhs

def cost(hdh: HDH, partition: List[Set[str]]) -> int:
    """Return number of hyperedges in HDH that span multiple partitions."""
    # Map node -> part index
    node_to_part = {}
    for part_idx, part in enumerate(partition):
        for node in part:
            node_to_part[node] = part_idx

    cut_edges = 0
    for edge in hdh.C:
        parts_in_edge = {node_to_part[n] for n in edge if n in node_to_part}
        if len(parts_in_edge) > 1:
            cut_edges += 1

    return cut_edges

def partition_sizes(partition: List[Set[str]]) -> List[int]:
    return [len(part) for part in partition]

def compute_parallelism_by_time(
    hdh: HDH,
    partition: List[Set[str]],
    mode: str = "global",
    time_step: Union[int, None] = None) -> Union[List[int], int]:
    """
    Compute parallelism over time:
    
    - If mode == "global": return list of partition counts per time step.
    - If mode == "local": return partition count at `time_step`.

    Args:
        hdh: The HDH object
        partition: List of sets of node IDs
        mode: "global" or "local"
        time_step: required if mode == "local"
    
    Returns:
        List[int] for global mode, int for local mode
    """
    node_to_part = {node: i for i, part in enumerate(partition) for node in part}

    if mode == "global":
        time_to_active_parts = DefaultDict(set)
        for node in hdh.S:
            if node in node_to_part:
                t = node[1]  # assumes node = (id, timestamp)
                time_to_active_parts[t].add(node_to_part[node])
        return [len(time_to_active_parts[t]) for t in sorted(hdh.T)]

    elif mode == "local":
        if time_step is None:
            raise ValueError("`time_step` must be specified for local mode.")
        active_parts = {
            node_to_part[node]
            for node in hdh.S
            if node in node_to_part and node[1] == time_step
        }
        return len(active_parts)

    else:
        raise ValueError("mode must be 'global' or 'local'")

def compute_cut_by_time_percent(hdh: HDH, percent: float) -> List[Set[str]]:
    """
    Cut the HDH horizontally across time at a given percentage (e.g. 0.3 = 30%).
    Returns two partitions: before and after the cut.
    """
    assert 0 <= percent <= 1, "Percent must be between 0 and 1"
    max_time = max(hdh.time_map.values())
    threshold = int(percent * max_time)

    part0 = {n for n in hdh.S if hdh.time_map[n] <= threshold}
    part1 = hdh.S - part0
    return [part0, part1]

def gates_by_partition(hdh, partitions):
    """
    Classify HDH edges as intra- or inter-partition based on provided partitions.
    Returns (intra_edges, inter_edges)
    """
    node_to_part = {}
    for i, part in enumerate(partitions):
        for node in part:
            node_to_part[node] = i

    intra = [[] for _ in partitions]
    inter = []

    for edge in hdh.C:
        parts = {node_to_part.get(n) for n in edge if n in node_to_part}
        parts.discard(None)

        if len(parts) == 1:
            intra[list(parts)[0]].append(edge)
        elif len(parts) > 1:
            inter.append(edge)

    return intra, inter

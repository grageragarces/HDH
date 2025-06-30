import networkx as nx
import networkx.algorithms.community as nx_comm
import metis
print(metis.__version__)
from typing import List, Set, Union, DefaultDict, Optional
from hdh.hdh import HDH

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

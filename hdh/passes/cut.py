"""
This code is currently under development and is subject to change.
Full integration with primitives is still pending.

Partitioning utilities:
- HDH-based (node-level, hypergraph-aware) greedy partititioning
- Telegate-based (qubit graph) METIS partitioning

Parallelism and Participation Metrics:
- participation(): Counts how many partitions have any activity at each timestep
  (useful for temporal participation overview, not true concurrency)
  
- parallelism(): Measures true concurrent work by counting τ-edges (operations)
  executing at each timestep (actual computational parallelism)
  
- fair_parallelism(): Capacity-normalized concurrency following Jean's fairness principle
  (detects imbalances in how partitions utilize their available capacity)
"""

from __future__ import annotations

# ------------------------------ Imports ------------------------------
import math
import re
import itertools
import random
from typing import List, Set, Tuple, Dict, Optional, Iterable
from collections import defaultdict, Counter
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

# ------------------------------ Regexes ------------------------------
# useful for recognising qubit and bit IDs
_Q_RE   = re.compile(r"^q(\d+)_t\d+$")
_C_RE   = re.compile(r"^c(\d+)_t\d+$")

# ------------------------------- Greedy partitioning on HDH -------------------------------

def _qubit_of(nid: str) -> Optional[int]:
    m = _Q_RE.match(nid)
    return int(m.group(1)) if m else None

def _build_hdh_incidence(hdh) -> Tuple[Dict[str, Set[frozenset]], Dict[frozenset, Set[str]], Dict[frozenset, int]]:
    """
    Returns:
      inc[v]  -> set of incident hyperedges for node v
      pins[e] -> set of node-ids in e
      w[e]    -> weight (default 1)
    """
    pins: Dict[frozenset, Set[str]] = {e: set(e) for e in hdh.C}
    inc:  Dict[str, Set[frozenset]] = defaultdict(set)
    for e, mems in pins.items():
        for v in mems:
            inc[v].add(e)
    w = {e: int(getattr(hdh, "edge_weight", {}).get(e, 1)) for e in hdh.C}
    return inc, pins, w

def _group_qnodes(hdh) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """q -> [node_ids], and node_id -> q"""
    qnodes_by_qubit: Dict[int, List[str]] = defaultdict(list)
    qubit_of: Dict[str, int] = {}
    for nid in hdh.S:
        m = _Q_RE.match(nid)
        if m:
            q = int(m.group(1))
            qnodes_by_qubit[q].append(nid)
            qubit_of[nid] = q
    return qnodes_by_qubit, qubit_of

class _HDHState:
    """
    Assign at node-level, but capacity applies to UNIQUE QUBITS/bin.
    When a qubit enters a bin, all its remaining nodes are auto-assigned to that bin.
    """
    __slots__ = ("assign","bin_nodes","bin_qubits","qubit_bin",
                 "pin_in_bin","unassigned_pins","k","cap","reserve_frac")
    def __init__(self, k:int, cap:int, edges:Iterable[frozenset], reserve_frac:float):
        self.assign: Dict[str,int] = {}                 # node -> bin
        self.bin_nodes = [0]*k                          # for stats only
        self.bin_qubits: List[Set[int]] = [set() for _ in range(k)]  # unique qubits/bin
        self.qubit_bin: Dict[int,int] = {}             # qubit -> bin
        self.pin_in_bin: Dict[frozenset, Counter] = {e: Counter() for e in edges}
        self.unassigned_pins: Dict[frozenset,int] = {e: len(e) for e in edges}
        self.k, self.cap, self.reserve_frac = k, cap, reserve_frac

    def qubit_load(self, b:int) -> int:
        return len(self.bin_qubits[b])

    def bin_capacity(self, b:int) -> int:
        used_q = self.qubit_load(b)
        hard = self.cap
        shadow = max(0, int(self.cap*(1.0 - self.reserve_frac)))
        return hard if used_q >= int(0.8*self.cap) else shadow

    def can_place_qubit(self, q: Optional[int], b:int) -> bool:
        if q is None:
            return True
        if q in self.qubit_bin:
            return self.qubit_bin[q] == b  # already anchored elsewhere? only ok if same bin
        return self.qubit_load(b) < self.bin_capacity(b)

def _delta_cost_hdh(v:str, b:int, st:_HDHState, inc, pins, w) -> int:
    d = 0
    for e in inc.get(v, ()):
        was = st.pin_in_bin[e][b]
        full_after = (st.pin_in_bin[e][b] + 1 == len(pins[e])) and (st.unassigned_pins[e] == 1)
        if was == 0 and not full_after:
            d += w[e]
        if full_after:
            d -= w[e]
    return d

def _place_hdh(v:str, b:int, st:_HDHState, inc, w, qnodes_by_qubit: Dict[int, List[str]]):
    """Place v, and if it's the first node of its qubit, auto-place all siblings to the same bin."""
    q = _qubit_of(v)
    # Respect existing qubit anchor
    if q is not None and q in st.qubit_bin and st.qubit_bin[q] != b:
        b = st.qubit_bin[q]

    def _place_one(nid:str):
        if nid in st.assign:
            return
        st.assign[nid] = b
        st.bin_nodes[b] += 1
        for e in inc.get(nid, ()):
            st.pin_in_bin[e][b] += 1
            st.unassigned_pins[e] -= 1

    _place_one(v)

    # First time we see this qubit? anchor + auto-place its other nodes
    if q is not None and q not in st.qubit_bin:
        st.qubit_bin[q] = b
        st.bin_qubits[b].add(q)
        for sib in qnodes_by_qubit.get(q, []):
            if sib != v:
                _place_one(sib)

def _total_cost_hdh(st:_HDHState, pins, w) -> int:
    cost = 0
    for e, cnt in st.pin_in_bin.items():
        nonzero = sum(1 for c in cnt.values() if c>0)
        if nonzero >= 2:
            cost += w[e]
    return cost


def _best_candidates_for_bin(items: Iterable[str],
                             b:int,
                             delta_fn,
                             state,
                             frontier_score_fn,
                             beam_k:int) -> List[Tuple[int,int,str]]: 
    # candidate picker (no capacity gating)
    """Top-K by (Δ, -frontier_score, id)."""
    cand = []
    for v in items:
        if v in state.assign:
            continue
        d = delta_fn(v, b)
        fr = frontier_score_fn(v, b)
        cand.append((d, -fr, v))
    cand.sort(key=lambda t: (t[0], t[1], t[2]))
    return cand[:beam_k]

def _first_unassigned_rep_that_fits(order, st, b):
    """Pick the first representative v (of an unanchored qubit) that fits bin b."""
    for v in order:
        if v in st.assign:
            continue
        q = _qubit_of(v)
        if st.can_place_qubit(q, b):
            return v
    return None

def compute_cut(hdh_graph, k:int, cap:int, *,
                beam_k:int=3,
                backtrack_window:int=0,
                polish_1swap_budget:int=0,   # disabled for HDH (moving whole qubits is heavier)
                restarts:int=1,
                reserve_frac:float=0.08,
                predictive_reject:bool=True,
                seed:int=0) -> Tuple[List[Set[str]], int]:
    """
    Greedy HDH partitioner (bin-fill). Capacity is on UNIQUE QUBITS/bin.
    When a qubit enters a bin, all its remaining nodes are auto-assigned to that bin.
    """
    inc, pins, w = _build_hdh_incidence(hdh_graph)
    qnodes_by_qubit, _ = _group_qnodes(hdh_graph)

    # choose one representative node per qubit for ordering
    deg_w = {}
    for v in hdh_graph.S:
        if _qubit_of(v) is not None:
            deg_w[v] = sum(w[e] for e in inc.get(v, ()))
    reps = []
    for q, lst in qnodes_by_qubit.items():
        pick = max(lst, key=lambda v: deg_w.get(v, 0))
        reps.append(pick)
    if not reps:
        return [set() for _ in range(k)], 0
    order = sorted(reps, key=lambda v: (-deg_w.get(v, 0), v))

    def run_once(rng) -> Tuple[List[Set[str]], int]:
        st = _HDHState(k, cap, edges=hdh_graph.C, reserve_frac=reserve_frac)

        def frontier_score(v:str, b:int)->int:
            sc = 0
            for e in inc.get(v, ()):
                if st.pin_in_bin[e][b] > 0:
                    sc += w[e]
            return sc

        delta = lambda v, b: _delta_cost_hdh(v, b, st, inc, pins, w)

        for b in range(k):
            # fill by unique qubits
            while st.qubit_load(b) < st.bin_capacity(b):
                # build beam
                cands = _best_candidates_for_bin(
                    items=order, b=b,
                    delta_fn=lambda v, bb=b: delta(v, bb),
                    state=st,
                    frontier_score_fn=lambda v, bb=b: frontier_score(v, bb),
                    beam_k=beam_k
                )
                # enforce qubit-capacity
                cands = [(d, fr, v) for (d, fr, v) in cands if st.can_place_qubit(_qubit_of(v), b)]

                if not cands:
                    # Fallback: seed the bin with the first unassigned rep that fits.
                    v0 = _first_unassigned_rep_that_fits(order, st, b)
                    if v0 is None:
                        break  # truly nothing fits this bin → move to next bin
                    _place_hdh(v0, b, st, inc, w, qnodes_by_qubit)
                    continue

                placed = False
                for d, _, v in cands:
                    if predictive_reject and st.qubit_load(b) >= st.bin_capacity(b) - 1:
                        touching = sum(1 for e in inc.get(v, ()) if st.pin_in_bin[e][b] > 0)
                        if touching == 0:
                            continue
                    _place_hdh(v, b, st, inc, w, qnodes_by_qubit)
                    placed = True
                    break

                if not placed:
                    # Even the beam couldn’t place (likely due to predictive reject on a near‑full bin).
                    # Try a single seed anyway to avoid empty bins.
                    v0 = _first_unassigned_rep_that_fits(order, st, b)
                    if v0 is None:
                        break
                    _place_hdh(v0, b, st, inc, w, qnodes_by_qubit)

        # ---- Final mop-up: distribute any remaining unassigned reps round‑robin
        remaining = [v for v in order if v not in st.assign]
        if remaining:
            bi = 0
            for v in remaining:
                tries = 0
                placed = False
                while tries < k and not placed:
                    if st.can_place_qubit(_qubit_of(v), bi) and st.qubit_load(bi) < st.cap:
                        _place_hdh(v, bi, st, inc, w, qnodes_by_qubit)
                        placed = True
                        break
                    bi = (bi + 1) % k
                    tries += 1
                # if not placed, all bins are truly at capacity — safe to skip

        # materialize bins
        bins_nodes: List[Set[str]] = [set() for _ in range(k)]
        for nid, bb in st.assign.items():
            bins_nodes[bb].add(nid)
        cost = _total_cost_hdh(st, pins, w)
        return bins_nodes, cost

    best_bins, best_cost = None, float("inf")
    for r in range(max(1, restarts)):
        rng = random.Random(seed + r)
        bins_nodes, cost = run_once(rng)
        if cost < best_cost:
            best_bins, best_cost = bins_nodes, cost
    return best_bins, best_cost

# ------------------------------- METIS telegate -------------------------------

def telegate_hdh(hdh: "HDH") -> nx.Graph:
    """
    Build the telegate graph of an HDH.
    Nodes = qubits (as 'q{idx}').
    Undirected edges = quantum operations between qubits (co-appearance in a quantum hyperedge).
    Edge attribute 'weight' counts multiplicity.
    """
    G = nx.Graph()

    qubits_seen = set()
    for n in hdh.S:
        m = _Q_RE.match(n)
        if m:
            qubits_seen.add(int(m.group(1)))
    for q in qubits_seen:
        G.add_node(f"q{q}")

    for e in hdh.C:
        if hasattr(hdh, "tau") and hdh.tau.get(e, None) != "q":
            continue
        qs = []
        for node in e:
            m = _Q_RE.match(node)
            if m:
                qs.append(int(m.group(1)))
        for a, b in itertools.combinations(sorted(set(qs)), 2):
            u, v = f"q{a}", f"q{b}"
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

def _bins_from_parts(parts) -> List[Set[str]]:
    return [set(map(str, p)) for p in parts]

def _sizes(bins: List[Set[str]]) -> List[int]:
    return [len(b) for b in bins]

def _over_under(bins: List[Set[str]], cap: int):
    sizes = _sizes(bins)
    over = [i for i, s in enumerate(sizes) if s > cap]
    under = [i for i, s in enumerate(sizes) if s < cap]
    return over, under

def _best_move_for_node(G: nx.Graph, node: str, src_idx: int, tgt_idx: int,
                        bins: List[Set[str]]) -> float:
    """Heuristic gain if moving `node` src->tgt. Higher is better."""
    to_src = 0
    to_tgt = 0
    for nbr, data in G[node].items():
        w = data.get("weight", 1)
        if nbr in bins[src_idx]:
            to_src += w
        if nbr in bins[tgt_idx]:
            to_tgt += w
    return to_tgt - to_src

def _repair_overflow(G: nx.Graph, bins: List[Set[str]], cap: int) -> List[Set[str]]:
    """Greedy rebalancer to enforce bin capacity."""
    while True:
        over, under = _over_under(bins, cap)
        if not over or not under:
            break
        moved_any = False
        over.sort(key=lambda i: len(bins[i]), reverse=True)
        for src in over:
            under.sort(key=lambda i: len(bins[i]))
            best_gain = None
            best_choice = None
            for node in list(bins[src]):
                for tgt in under:
                    if len(bins[tgt]) >= cap:
                        continue
                    gain = _best_move_for_node(G, node, src, tgt, bins)
                    if (best_gain is None) or (gain > best_gain):
                        best_gain = gain
                        best_choice = (node, tgt)
            if best_choice:
                node, tgt = best_choice
                bins[src].remove(node)
                bins[tgt].add(node)
                moved_any = True
                break
        if not moved_any:
            for src in over:
                for tgt in under:
                    if len(bins[tgt]) >= cap:
                        continue
                    node = next(iter(bins[src]))
                    bins[src].remove(node)
                    bins[tgt].add(node)
                    moved_any = True
                    break
                if moved_any:
                    break
            if not moved_any:
                break
    return bins

def _cut_edges_unweighted(G: nx.Graph, bins: List[Set[str]]) -> int:
    """Count edges crossing between different bins (unweighted)."""
    where = {}
    for i, b in enumerate(bins):
        for n in b:
            where[n] = i
    cut = 0
    for u, v in G.edges():
        if where.get(u) != where.get(v):
            cut += 1
    return cut

def _kl_fallback_partition(G: nx.Graph, k: int) -> List[Set[str]]:
    """Recursive bisection using Kernighan–Lin; returns list of node sets."""
    parts: List[Set[str]] = [set(G.nodes())]
    while len(parts) < k:
        parts.sort(key=len, reverse=True)
        big = parts.pop(0)
        if len(big) <= 1:
            parts.append(big)
            break
        H = G.subgraph(big).copy()
        try:
            A, B = kernighan_lin_bisection(H, weight="weight")
        except Exception:
            nodes = list(big)
            mid = len(nodes) // 2
            A, B = set(nodes[:mid]), set(nodes[mid:])
        parts.extend([set(A), set(B)])
    while len(parts) > k:
        parts.sort(key=len)
        a = parts.pop(0); b = parts.pop(0)
        parts.append(a | b)
    return parts

def metis_telegate(hdh: "HDH", partitions: int, capacities: int) -> Tuple[List[Set[str]], int, bool, str]:
    """
    Partition the telegate (qubit) graph via METIS (or KL fallback), with capacity on #qubits/bin.
    Returns: (bins_qubits, cut_cost, respects_capacity, method['metis'|'kl'])
    """
    G: nx.Graph = telegate_hdh(hdh)

    used_metis = False
    try:
        import nxmetis  # type: ignore
        used_metis = True
    except Exception:
        used_metis = False

    n = G.number_of_nodes()
    if partitions <= 0 or capacities <= 0:
        empty = [set() for _ in range(max(0, partitions))]
        return empty, 0, False, "error"
    if n == 0:
        empty = [set() for _ in range(partitions)]
        return empty, 0, True, "metis" if used_metis else "kl"
    if partitions * capacities < n:
        return [], 0, False, "metis" if used_metis else "kl"

    nx.set_node_attributes(G, {n: 1 for n in G.nodes}, name="weight")

    if used_metis:
        target = capacities / float(n)
        tpwgts = [target] * partitions
        ubvec = [1.001]
        try:
            import nxmetis
            _, parts = nxmetis.partition(
                G, partitions,
                node_weight="weight", edge_weight="weight",
                tpwgts=tpwgts, ubvec=ubvec
            )
        except TypeError:
            _, parts = nxmetis.partition(G, partitions, node_weight="weight", edge_weight="weight")
        bins = _bins_from_parts(parts)
        method = "metis"
    else:
        parts = _kl_fallback_partition(G, partitions)
        bins = _bins_from_parts(parts)
        method = "kl"

    bins = _repair_overflow(G, bins, capacities)
    cost = _cut_edges_unweighted(G, bins)
    respects = all(len(b) <= capacities for b in bins)
    return bins, cost, respects, method


# ------------------------------- Public API Functions -------------------------------

def cost(hdh_graph, partitions) -> Tuple[float, float]:
    """
    Calculate the cost of a given partitioning of the HDH graph.
    
    Args:
        hdh_graph: HDH graph object
        partitions: List of sets, where each set contains node IDs in that partition
    
    Returns:
        Tuple[float, float]: (cost_q, cost_c) - quantum and classical cut costs
            cost_q: number of quantum hyperedges that span multiple partitions
            cost_c: number of classical hyperedges that span multiple partitions
    """
    if not partitions or not hasattr(hdh_graph, 'C'):
        return 0.0, 0.0
    
    # Create mapping from node to partition index
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Count hyperedges that cross partitions (separated by type)
    cost_q = 0  # Quantum cost
    cost_c = 0  # Classical cost
    
    for edge in hdh_graph.C:
        # Get partitions of all nodes in this hyperedge
        edge_partitions = set()
        for node in edge:
            if node in node_to_partition:
                edge_partitions.add(node_to_partition[node])
        
        # If hyperedge spans multiple partitions, it contributes to cost
        if len(edge_partitions) > 1:
            # Get edge weight if available
            edge_weight = 1
            if hasattr(hdh_graph, 'edge_weight'):
                edge_weight = hdh_graph.edge_weight.get(edge, 1)
            
            # Determine if edge is quantum or classical
            edge_type = 'q'  # Default to quantum
            if hasattr(hdh_graph, 'tau'):
                edge_type = hdh_graph.tau.get(edge, 'q')
            
            if edge_type == 'q':
                cost_q += edge_weight
            else:
                cost_c += edge_weight
    
    return float(cost_q), float(cost_c)


def partition_size(partitions) -> List[int]:
    """
    Calculate the sizes (number of nodes) of each partition.
    
    Args:
        partitions: List of sets, where each set contains node IDs in that partition
    
    Returns:
        List[int]: Size of each partition
    """
    if not partitions:
        return []
    
    return [len(partition) for partition in partitions]


def participation(hdh_graph, partitions) -> Dict[str, float]:
    """
    Compute partition participation metrics based on temporal analysis.
    
    This measures how many partitions have any activity (nodes or edges) at each timestep,
    providing an overview of temporal participation but not true concurrent work.
    
    Args:
        hdh_graph: HDH graph object with temporal structure
        partitions: List of sets, where each set contains node IDs in that partition
    
    Returns:
        Dict[str, float]: Dictionary containing participation metrics
    """
    if not partitions or not hasattr(hdh_graph, 'T') or not hasattr(hdh_graph, 'time_map'):
        return {
            'max_participation': 0.0,
            'average_participation': 0.0,
            'temporal_efficiency': 0.0,
            'partition_utilization': 0.0,
            'timesteps': 0,
            'num_partitions': len(partitions) if partitions else 0
        }
    
    # Create mapping from node to partition
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Analyze participation at each time step
    timestep_participation = []
    total_active_partitions = 0
    
    for t in sorted(hdh_graph.T):
        # Find which partitions have any nodes at time t
        active_partitions = set()
        
        for node in hdh_graph.S:
            if hdh_graph.time_map.get(node) == t and node in node_to_partition:
                active_partitions.add(node_to_partition[node])
        
        participation = len(active_partitions)
        timestep_participation.append(participation)
        total_active_partitions += participation
    
    # Calculate metrics
    num_timesteps = len(hdh_graph.T) if hdh_graph.T else 1
    max_participation = max(timestep_participation) if timestep_participation else 0
    avg_participation = sum(timestep_participation) / num_timesteps if num_timesteps > 0 else 0
    
    # Temporal efficiency: how well we utilize available time steps
    total_possible_work = len(partitions) * num_timesteps
    actual_work = total_active_partitions
    temporal_efficiency = actual_work / total_possible_work if total_possible_work > 0 else 0
    
    # Partition utilization: average fraction of partitions active per timestep
    partition_utilization = avg_participation / len(partitions) if partitions else 0
    
    return {
        'max_participation': float(max_participation),
        'average_participation': float(avg_participation),
        'temporal_efficiency': float(temporal_efficiency),
        'partition_utilization': float(partition_utilization),
        'timesteps': num_timesteps,
        'num_partitions': len(partitions)
    }


def parallelism(hdh_graph, partitions) -> Dict[str, float]:
    """
    Compute true parallelism metrics by counting concurrent τ-edges (operations) per timestep.
    
    Parallelism is defined as the number of τ-edges that can execute simultaneously at a given
    timestep, representing actual concurrent computational work, not just partition activity.
    
    Args:
        hdh_graph: HDH graph object with temporal structure
        partitions: List of sets, where each set contains node IDs in that partition
    
    Returns:
        Dict[str, float]: Dictionary containing parallelism metrics
    """
    if not partitions or not hasattr(hdh_graph, 'T') or not hasattr(hdh_graph, 'time_map'):
        return {
            'max_parallelism': 0.0,
            'average_parallelism': 0.0,
            'total_operations': 0,
            'timesteps': 0,
            'num_partitions': len(partitions) if partitions else 0
        }
    
    # Create mapping from node to partition
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Map each edge to its timestep based on its constituent nodes
    edge_to_time = {}
    for edge in hdh_graph.C:
        # Get the timestep(s) of nodes in this edge
        edge_times = set()
        for node in edge:
            if node in hdh_graph.time_map:
                edge_times.add(hdh_graph.time_map[node])
        
        # Assign edge to the maximum timestep of its nodes (operational time)
        if edge_times:
            edge_to_time[edge] = max(edge_times)
    
    # Count operations (τ-edges) per timestep
    timestep_operations = []
    total_operations = 0
    
    for t in sorted(hdh_graph.T):
        # Count τ-edges executing at this timestep
        operations_at_t = 0
        
        for edge, edge_time in edge_to_time.items():
            if edge_time == t:
                # Only count edges with a type defined (operations)
                if hasattr(hdh_graph, 'tau') and edge in hdh_graph.tau:
                    operations_at_t += 1
        
        timestep_operations.append(operations_at_t)
        total_operations += operations_at_t
    
    # Calculate metrics
    num_timesteps = len(hdh_graph.T) if hdh_graph.T else 1
    max_parallelism = max(timestep_operations) if timestep_operations else 0
    avg_parallelism = sum(timestep_operations) / num_timesteps if num_timesteps > 0 else 0
    
    return {
        'max_parallelism': float(max_parallelism),
        'average_parallelism': float(avg_parallelism),
        'total_operations': int(total_operations),
        'timesteps': num_timesteps,
        'num_partitions': len(partitions)
    }


def fair_parallelism(hdh_graph, partitions, capacities: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Compute fair parallelism following Jean's fairness principle.
    
    Fair parallelism normalizes concurrency by partition capacity, measuring how evenly
    computational work is distributed across partitions relative to their capacity.
    If partitions have equal capacity and each runs the same number of operations,
    fair_parallelism equals parallelism. Imbalances reduce fair_parallelism below raw parallelism.
    
    Args:
        hdh_graph: HDH graph object with temporal structure
        partitions: List of sets, where each set contains node IDs in that partition
        capacities: Optional list of capacity values per partition (default: equal capacities)
    
    Returns:
        Dict[str, float]: Dictionary containing fair parallelism metrics
    """
    if not partitions or not hasattr(hdh_graph, 'T') or not hasattr(hdh_graph, 'time_map'):
        return {
            'max_fair_parallelism': 0.0,
            'average_fair_parallelism': 0.0,
            'fairness_ratio': 0.0,
            'total_operations': 0,
            'timesteps': 0,
            'num_partitions': len(partitions) if partitions else 0
        }
    
    # Use equal capacities if not provided
    if capacities is None:
        capacities = [1.0] * len(partitions)
    elif len(capacities) != len(partitions):
        raise ValueError(f"Number of capacities ({len(capacities)}) must match number of partitions ({len(partitions)})")
    
    # Normalize capacities to sum to 1 for fair distribution
    total_capacity = sum(capacities)
    if total_capacity == 0:
        return {
            'max_fair_parallelism': 0.0,
            'average_fair_parallelism': 0.0,
            'fairness_ratio': 0.0,
            'total_operations': 0,
            'timesteps': 0,
            'num_partitions': len(partitions)
        }
    
    normalized_capacities = [c / total_capacity for c in capacities]
    
    # Create mapping from node to partition
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Map each edge to its timestep and partition
    edge_to_time = {}
    edge_to_partition = {}
    for edge in hdh_graph.C:
        # Get the timestep(s) and partition(s) of nodes in this edge
        edge_times = set()
        edge_partitions = set()
        for node in edge:
            if node in hdh_graph.time_map:
                edge_times.add(hdh_graph.time_map[node])
            if node in node_to_partition:
                edge_partitions.add(node_to_partition[node])
        
        # Assign edge to the maximum timestep and primary partition (first one)
        if edge_times:
            edge_to_time[edge] = max(edge_times)
        if edge_partitions:
            edge_to_partition[edge] = min(edge_partitions)  # Use consistent partition assignment
    
    # Count operations per partition per timestep
    timestep_fair_parallelism = []
    total_operations = 0
    total_raw_parallelism = 0
    
    for t in sorted(hdh_graph.T):
        # Count operations per partition at this timestep
        partition_ops = [0] * len(partitions)
        
        for edge, edge_time in edge_to_time.items():
            if edge_time == t:
                # Only count edges with a type defined (operations)
                if hasattr(hdh_graph, 'tau') and edge in hdh_graph.tau:
                    if edge in edge_to_partition:
                        p = edge_to_partition[edge]
                        partition_ops[p] += 1
                        total_operations += 1
        
        # Calculate fair parallelism for this timestep
        # Fair parallelism = sum of (ops_i / capacity_i) normalized
        raw_ops = sum(partition_ops)
        total_raw_parallelism += raw_ops
        
        if raw_ops > 0:
            # Weighted by capacity: fair contribution from each partition
            fair_contribution = sum(
                (partition_ops[i] / normalized_capacities[i]) if normalized_capacities[i] > 0 else 0
                for i in range(len(partitions))
            )
            # Normalize to get fair parallelism metric
            fair_p = fair_contribution / len(partitions)
        else:
            fair_p = 0.0
        
        timestep_fair_parallelism.append(fair_p)
    
    # Calculate metrics
    num_timesteps = len(hdh_graph.T) if hdh_graph.T else 1
    max_fair_parallelism = max(timestep_fair_parallelism) if timestep_fair_parallelism else 0
    avg_fair_parallelism = sum(timestep_fair_parallelism) / num_timesteps if num_timesteps > 0 else 0
    avg_raw_parallelism = total_raw_parallelism / num_timesteps if num_timesteps > 0 else 0
    
    # Fairness ratio: how fair is the distribution (1.0 = perfectly fair)
    fairness_ratio = avg_fair_parallelism / avg_raw_parallelism if avg_raw_parallelism > 0 else 1.0
    
    return {
        'max_fair_parallelism': float(max_fair_parallelism),
        'average_fair_parallelism': float(avg_fair_parallelism),
        'fairness_ratio': float(fairness_ratio),
        'total_operations': int(total_operations),
        'timesteps': num_timesteps,
        'num_partitions': len(partitions)
    }


# Keep old name as alias for backward compatibility (deprecated)
def compute_parallelism_by_time(hdh_graph, partitions) -> Dict[str, float]:
    """
    Deprecated: Use `parallelism()` instead.
    
    This function now calls `parallelism()` for backward compatibility.
    """
    return parallelism(hdh_graph, partitions)


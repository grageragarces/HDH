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

# from __future__ import annotations

# ------------------------------ Imports ------------------------------
import math
import os
import re
import itertools
import random
import heapq
from typing import List, Set, Tuple, Dict, Optional, Iterable
from collections import defaultdict, Counter
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

# ------------------------------ KaHyPar cutter (qubit-level -> HDH node-level) ------------------------------

def kahypar_cutter(
    hdh,
    k: int,
    cap: int,
    *,
    seed: int = 0,
    config_path: Optional[str] = None,
    suppress_output: bool = True,
):
    """Partition an HDH using the kahypar Python package.

    Args:
        hdh: HDH object with .S (nodes) and .C (hyperedges)
        k: number of partitions
        cap: max unique qubits per partition
        seed: RNG seed passed to KaHyPar (if supported)
        config_path: path to KaHyPar INI config; required by KaHyPar.
        suppress_output: whether to silence KaHyPar stdout.

    Returns:
        partitions: list[set[str]] length k; each set is HDH node-ids assigned to that partition
        cut_cost: number of cut hyperedges in the original HDH (same definition as compute_cut)
    """
    try:
        import kahypar  # type: ignore
    except Exception as e:
        raise ImportError(
            "`kahypar` is not installed. Install it (e.g. `pip install kahypar==1.3.6`) "
            "and ensure you have the build requirements from the PyPI page."
        ) from e

    if k <= 0:
        raise ValueError("k must be >= 1")

    # --- Build qubit vertex set ---
    qubits: Set[int] = set()
    qubit_nodes = defaultdict(list)  # q -> [node ids]
    classical_nodes_by_idx = defaultdict(list)  # c -> [node ids]

    for nid in getattr(hdh, "S", set()):
        qm = _Q_RE.match(nid)
        if qm:
            q = int(qm.group(1))
            qubits.add(q)
            qubit_nodes[q].append(nid)
            continue
        cm = _C_RE.match(nid)
        if cm:
            c = int(cm.group(1))
            classical_nodes_by_idx[c].append(nid)

    qubit_list = sorted(qubits)
    n = len(qubit_list)
    if n == 0:
        # No qubits: nothing meaningful to partition.
        return [set() for _ in range(k)], 0

    if k > n:
        raise ValueError(f"k={k} cannot exceed number of qubits n={n} for kahypar_cutter")

    if cap <= 0:
        raise ValueError("cap must be >= 1 (capacity = max unique qubits per partition)")

    # If cap is too small to ever fit an even split, fail early.
    min_required = (n + k - 1) // k
    if cap < min_required:
        raise ValueError(
            f"cap={cap} is too small for n={n}, k={k}. "
            f"Need at least ceil(n/k)={min_required} to be feasible."
        )

    q_to_vid = {q: i for i, q in enumerate(qubit_list)}

    # --- Build undirected hyperedges on qubit-vertices ---
    # KaHyPar expects hyperedges as pins (vertex IDs). We collapse each HDH hyperedge
    # to the set of qubits it touches.
    hedge_pins: List[List[int]] = []
    hedge_weights: List[int] = []

    for e in getattr(hdh, "C", set()):
        qs = set()
        for nid in e:
            qm = _Q_RE.match(nid)
            if qm:
                qs.add(int(qm.group(1)))
        if len(qs) >= 2:
            hedge_pins.append([q_to_vid[q] for q in sorted(qs)])
            hedge_weights.append(1)

    # Degenerate case: no multi-qubit couplings
    if not hedge_pins:
        # Purely disconnected qubits: just pack sequentially.
        partitions = [set() for _ in range(k)]
        for i, q in enumerate(qubit_list):
            b = i % k
            partitions[b].update(qubit_nodes[q])
            # Attach same-index classical nodes if present
            partitions[b].update(classical_nodes_by_idx.get(q, []))
        return partitions, _compute_cut_cost(hdh, {nid: b for b, part in enumerate(partitions) for nid in part})

    # Convert pins list to CSR-style arrays required by KaHyPar.
    # hyperedge_indices: prefix-sum offsets into `hyperedges` array.
    hyperedge_indices = [0]
    hyperedges: List[int] = []
    for pins in hedge_pins:
        hyperedges.extend(pins)
        hyperedge_indices.append(len(hyperedges))

    num_hyperedges = len(hedge_pins)
    vertex_weights = [1] * n  # each qubit counts as 1 capacity unit

    # --- KaHyPar context/config ---
    if config_path is None:
        # Prefer a local config if user has one, otherwise force them to provide.
        candidate = "kahypar/config/km1_kKaHyPar_sea20.ini"
        if os.path.exists(candidate):
            config_path = candidate
        else:
            raise FileNotFoundError(
                "KaHyPar needs an INI configuration file. "
                "Pass `config_path=...` (e.g. a KaHyPar .ini file)."
            )

    # Balance: ensure max block size <= cap.
    target = n / float(k)
    epsilon = max(0.0, (cap / target) - 1.0)

    context = kahypar.Context()
    context.loadINIconfiguration(str(config_path))
    context.setK(k)
    context.setEpsilon(epsilon)

    # Some builds support setting a seed.
    if hasattr(context, "setSeed"):
        try:
            context.setSeed(int(seed))
        except Exception:
            pass

    if suppress_output and hasattr(context, "suppressOutput"):
        try:
            context.suppressOutput(True)
        except Exception:
            pass

    # --- Build and partition hypergraph ---
    # KaHyPar signature differs slightly across versions; try the common one.
    try:
        hg = kahypar.Hypergraph(
            n,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            hedge_weights,
            vertex_weights,
        )
    except TypeError:
        # Older signature: Hypergraph(num_vertices, num_hyperedges, hyperedge_indices, hyperedges, k, ...)
        hg = kahypar.Hypergraph(
            n,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            k,
            hedge_weights,
            vertex_weights,
        )

    kahypar.partition(hg, context)

    # --- Read partitioning and lift back to HDH nodes ---
    qubit_block = {qubit_list[v]: int(hg.blockID(v)) for v in range(n)}

    partitions: List[Set[str]] = [set() for _ in range(k)]
    for q, nodes in qubit_nodes.items():
        b = qubit_block[q]
        partitions[b].update(nodes)
        # Attach same-index classical nodes if present
        partitions[b].update(classical_nodes_by_idx.get(q, []))

    # Any remaining classical nodes with indices not matching qubits go to block 0
    for c_idx, nodes in classical_nodes_by_idx.items():
        if c_idx not in qubit_nodes:
            partitions[0].update(nodes)

    node_assignment = {nid: b for b, part in enumerate(partitions) for nid in part}
    cut_cost = _compute_cut_cost(hdh, node_assignment)
    return partitions, cut_cost


def kahypar_cutter_nodebalanced(
    hdh,
    k: int,
    *,
    seed: int = 0,
    config_path: Optional[str] = None,
    epsilon: float = 0.03,
    suppress_output: bool = True,
):
    """Partition an HDH using KaHyPar with **node-balanced** constraints.

    This is intentionally the "capacity-oblivious" baseline:
    - vertices = HDH nodes (q*_t* and c*_t*)
    - vertex_weights = 1 for all vertices
    - balance enforced by epsilon around equal-size blocks (in *nodes*, not logical qubits)

    This is useful for measuring how often such a baseline violates a
    *logical-qubit* capacity constraint when you evaluate it post-hoc.

    Returns:
        partitions: list[set[str]] of HDH node IDs per block
        cut_cost: cut hyperedge count in the original HDH
    """
    try:
        import kahypar  # type: ignore
    except Exception as e:
        raise ImportError(
            "`kahypar` is not installed. Install it (e.g. `pip install kahypar==1.3.6`)."
        ) from e

    if k <= 0:
        raise ValueError("k must be >= 1")

    nodes = sorted(list(getattr(hdh, "S", set())))
    n = len(nodes)
    if n == 0:
        return [set() for _ in range(k)], 0
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of HDH nodes n={n} for node-balanced KaHyPar")

    # Build node id -> vertex id
    nid_to_vid = {nid: i for i, nid in enumerate(nodes)}

    # Build undirected hyperedges as pins (vertex IDs)
    hedge_pins: List[List[int]] = []
    hedge_weights: List[int] = []
    for e in getattr(hdh, "C", set()):
        pins = [nid_to_vid[nid] for nid in e if nid in nid_to_vid]
        pins = sorted(set(pins))
        if len(pins) >= 2:
            hedge_pins.append(pins)
            hedge_weights.append(1)

    # Degenerate case: no usable hyperedges
    if not hedge_pins:
        parts = [set() for _ in range(k)]
        for i, nid in enumerate(nodes):
            parts[i % k].add(nid)
        node_assignment = {nid: b for b, part in enumerate(parts) for nid in part}
        return parts, _compute_cut_cost(hdh, node_assignment)

    hyperedge_indices = [0]
    hyperedges: List[int] = []
    for pins in hedge_pins:
        hyperedges.extend(pins)
        hyperedge_indices.append(len(hyperedges))

    num_hyperedges = len(hedge_pins)
    vertex_weights = [1] * n

    if config_path is None:
        candidate = "kahypar/config/km1_kKaHyPar_sea20.ini"
        if os.path.exists(candidate):
            config_path = candidate
        else:
            raise FileNotFoundError(
                "KaHyPar needs an INI configuration file. Pass `config_path=...` (a KaHyPar .ini file)."
            )

    context = kahypar.Context()
    context.loadINIconfiguration(str(config_path))
    context.setK(k)
    context.setEpsilon(float(epsilon))

    if hasattr(context, "setSeed"):
        try:
            context.setSeed(int(seed))
        except Exception:
            pass
    if suppress_output and hasattr(context, "suppressOutput"):
        try:
            context.suppressOutput(True)
        except Exception:
            pass

    try:
        hg = kahypar.Hypergraph(
            n,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            hedge_weights,
            vertex_weights,
        )
    except TypeError:
        hg = kahypar.Hypergraph(
            n,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            k,
            hedge_weights,
            vertex_weights,
        )

    kahypar.partition(hg, context)

    partitions: List[Set[str]] = [set() for _ in range(k)]
    for vid, nid in enumerate(nodes):
        b = int(hg.blockID(vid))
        partitions[b].add(nid)

    node_assignment = {nid: b for b, part in enumerate(partitions) for nid in part}
    cut_cost = _compute_cut_cost(hdh, node_assignment)
    return partitions, cut_cost

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
            return (self.qubit_bin[q] == b)
        return (self.qubit_load(b) < self.bin_capacity(b))

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

def _extract_qubit_id(node_id: str) -> Optional[int]:
    """Extract qubit number from node ID like 'q5_t2' -> 5"""
    m = _Q_RE.match(node_id)
    return int(m.group(1)) if m else None


def _build_node_connectivity(hdh) -> Dict[str, Set[str]]:
    """
    Build node connectivity from HDH hyperedges.
    Two nodes are connected if they appear in the same hyperedge.
    
    Returns: {node_id: {connected_node_ids}}
    """
    adjacency = defaultdict(set)
    
    for hyperedge in hdh.C:
        nodes_in_edge = list(hyperedge)
        # Connect all pairs of nodes in this hyperedge
        for i, n1 in enumerate(nodes_in_edge):
            for n2 in nodes_in_edge[i+1:]:
                adjacency[n1].add(n2)
                adjacency[n2].add(n1)
    
    return adjacency


def _get_qubit_to_nodes(hdh) -> Dict[int, List[str]]:
    """
    Map each qubit to all its temporal nodes.
    
    Returns: {qubit_id: [node_ids]}
    """
    qubit_nodes = defaultdict(list)
    
    for node in hdh.S:
        q = _extract_qubit_id(node)
        if q is not None:
            qubit_nodes[q].append(node)
    
    return qubit_nodes


def _build_temporal_incidence(hdh) -> Tuple[Dict[str, List[Tuple[frozenset, int]]], Dict[frozenset, Set[str]]]:
    """
    Build temporal incidence structure for HDH.
    
    Returns:
        inc: {node -> [(hyperedge, edge_time), ...]}
        pins: {hyperedge -> {nodes}}
    """
    pins = {e: set(e) for e in hdh.C}
    inc = defaultdict(list)
    
    for edge in hdh.C:
        # Compute edge time as max of node times
        edge_times = [hdh.time_map.get(node, 0) for node in edge]
        edge_time = max(edge_times) if edge_times else 0
        
        for node in edge:
            inc[node].append((edge, edge_time))
    
    return inc, pins


def _push_next_valid_neighbors(hdh, node: str, frontier: List[Tuple[int, int, str]], 
                                unassigned: Set[str], 
                                inc: Dict[str, List[Tuple[frozenset, int]]],
                                pins: Dict[frozenset, Set[str]], 
                                counter: List[int]):
    """
    Push unassigned neighbors of node to the frontier priority queue.
    
    Frontier is ordered by earliest valid connection time (temporal greedy).
    The counter provides unique tie-breakers for deterministic ordering.
    """
    node_time = hdh.time_map.get(node, 0)
    
    # Examine all hyperedges incident to this node
    for edge, edge_time in inc.get(node, []):
        # Only consider edges at or after the current node's time (temporal validity)
        if edge_time < node_time:
            continue
            
        # Find all unassigned neighbors in this edge
        for neighbor in pins[edge]:
            if neighbor != node and neighbor in unassigned:
                # Calculate earliest connection time for this neighbor
                # This is the maximum of: edge time, neighbor's own time
                neighbor_time = hdh.time_map.get(neighbor, 0)
                earliest_time = max(edge_time, neighbor_time)
                
                # Push to heap with unique counter for deterministic tie-breaking
                heapq.heappush(frontier, (earliest_time, counter[0], neighbor))
                counter[0] += 1


def _compute_delta_cost_simple(node: str, 
                                bin_idx: int, 
                                partitions: List[Set[str]],
                                inc: Dict[str, List[Tuple[frozenset, int]]],
                                pins: Dict[frozenset, Set[str]]) -> int:
    """
    Compute delta cost of adding node to bin_idx.
    
    Delta cost measures how many hyperedge cuts would be created (positive)
    or eliminated (negative) by adding this node to this bin.
    
    Returns:
        Negative = reduces cuts (good)
        Positive = increases cuts (bad)
        Zero = neutral
    """
    delta = 0
    
    for edge, _ in inc.get(node, []):
        # Current state: which partitions does this edge touch?
        current_parts = set()
        for n in pins[edge]:
            for p_idx, p_nodes in enumerate(partitions):
                if n in p_nodes:
                    current_parts.add(p_idx)
                    break
        
        # Future state: add node to bin_idx
        future_parts = current_parts.copy()
        future_parts.add(bin_idx)
        
        was_cut = len(current_parts) > 1
        will_be_cut = len(future_parts) > 1
        
        # If adding this node creates a new cut, that's bad
        if not was_cut and will_be_cut:
            delta += 1
        # If this completes an edge within one partition, that's good
        elif was_cut and not will_be_cut:
            delta -= 1
    
    return delta


def _select_best_from_frontier_with_rejected(frontier: List[Tuple[int, int, str]], 
                                              unassigned: Set[str],
                                              rejected: Set[str],
                                              bin_idx: int,
                                              partitions: List[Set[str]],
                                              inc: Dict[str, List[Tuple[frozenset, int]]],
                                              pins: Dict[frozenset, Set[str]],
                                              beam_k: int = 3) -> Optional[str]:
    """
    Select next node from frontier using delta cost awareness, excluding rejected nodes.
    
    Instead of just taking the earliest node, consider top beam_k candidates
    and pick the one with best (most negative) delta cost.
    
    Args:
        rejected: Set of nodes that have been rejected for this bin (e.g., due to capacity)
    
    Returns:
        Best node to add, or None if frontier is empty
    """
    if not frontier:
        return None
    
    candidates = []
    examined = []
    
    # Extract top beam_k earliest VALID (unassigned and not rejected) nodes
    while frontier and len(candidates) < beam_k:
        item = heapq.heappop(frontier)
        time, counter, node = item
        examined.append(item)
        
        if node in unassigned and node not in rejected:
            candidates.append(node)
    
    # Put back only the nodes we examined but didn't select as candidates
    for item in examined:
        _, _, node = item
        if node not in candidates and node in unassigned and node not in rejected:
            heapq.heappush(frontier, item)
    
    if not candidates:
        return None
    
    # Score each candidate by delta cost
    scored = []
    for node in candidates:
        delta = _compute_delta_cost_simple(node, bin_idx, partitions, inc, pins)
        # Prioritize: low delta cost (negative is best)
        scored.append((delta, node))
    
    scored.sort()
    best_node = scored[0][1]
    
    # Put back the candidates we didn't select
    for item in examined:
        _, _, node = item
        if node != best_node and node in unassigned and node not in rejected:
            heapq.heappush(frontier, item)
    
    return best_node


def _select_best_from_frontier(frontier: List[Tuple[int, int, str]], 
                                unassigned: Set[str],
                                bin_idx: int,
                                partitions: List[Set[str]],
                                inc: Dict[str, List[Tuple[frozenset, int]]],
                                pins: Dict[frozenset, Set[str]],
                                beam_k: int = 3) -> Optional[str]:
    """Wrapper for _select_best_from_frontier_with_rejected with empty rejected set."""
    return _select_best_from_frontier_with_rejected(frontier, unassigned, set(), 
                                                     bin_idx, partitions, inc, pins, beam_k)


def _pop_earliest_valid(frontier: List[Tuple[int, int, str]], 
                        unassigned: Set[str]) -> Optional[str]:
    """
    Pop the earliest valid (still unassigned) node from the frontier.
    
    Returns None if no valid candidates remain.
    """
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in unassigned:
            return node
    return None


def _compute_cut_cost(hdh, node_assignment: Dict[str, int]) -> int:
    """
    Count hyperedges that span multiple partitions.
    
    A hyperedge is cut if its nodes are assigned to different partitions.
    """
    cut_count = 0
    
    for hyperedge in hdh.C:
        partitions_in_edge = set()
        for node in hyperedge:
            if node in node_assignment:
                partitions_in_edge.add(node_assignment[node])
        
        # Cut if spans multiple partitions
        if len(partitions_in_edge) > 1:
            cut_count += 1
    
    return cut_count


def compute_cut(hdh_graph, k: int, cap: int, *,
                beam_k: int = 3,
                backtrack_window: int = 0,
                polish_1swap_budget: int = 0,
                restarts: int = 1,
                reserve_frac: float = 0.08,
                predictive_reject: bool = True,
                seed: int = 0) -> Tuple[List[Set[str]], int]:
    """
    Capacity-aware temporal greedy partitioner for HDH graphs.
    
    Implements the algorithm from the paper with cost-aware improvements:
    - Cost-aware greedy frontier selection using delta cost evaluation
    - Best-fit residual round-robin placement
    
    Works directly on the HDH hypergraph structure:
    - Partitions at the NODE level (nodes like "q0_t1", "q1_t2", etc.)
    - Uses temporal hyperedge connectivity from HDH.C
    - Respects capacity by counting unique QUBITS per partition
    - Allows teledata cuts (same qubit in different partitions)
    - Priority queue selects earliest-time unassigned neighbors
    - Delta cost guides selection among top-k frontier candidates
        
    Args:
        hdh_graph: HDH object with .S (nodes), .C (hyperedges), .time_map
        k: Number of partitions (QPUs)
        cap: Capacity per partition (max unique qubits, not nodes)
        beam_k: Beam width for frontier selection (default 3)
        
        The following parameters are accepted for compatibility but currently not used:
        backtrack_window, polish_1swap_budget, restarts, 
        reserve_frac, predictive_reject, seed
    
    Returns:
        partitions: List of k sets, each containing node IDs assigned to that partition
        cost: Total communication cost (number of cut hyperedges)
    """
    if not hdh_graph.S or not hdh_graph.C:
        return [set() for _ in range(k)], 0
    
    # Build temporal incidence structure
    inc, pins = _build_temporal_incidence(hdh_graph)
    
    # Initialize partitions and tracking structures
    partitions = [set() for _ in range(k)]
    unassigned = set(hdh_graph.S)
    partition_qubits = [set() for _ in range(k)]  # Track unique qubits per partition
    used = [0] * k  # Track number of unique qubits used per partition
    
    # QPU order (for now, just sequential; could be topology-aware)
    qpu_order = list(range(k))
    
    # Phase 1 & 2: Greedy bin filling with cost-aware frontier selection
    for i in range(k):
        bin_idx = qpu_order[i]
        
        if not unassigned:
            break
        
        # Select seed: lowest-index unassigned node
        seed_node = min(unassigned, key=lambda n: (hdh_graph.time_map.get(n, 0), n))
        
        # Initialize bin with seed - update partitions immediately
        partitions[bin_idx].add(seed_node)
        unassigned.remove(seed_node)
        
        # Track qubits in this bin
        seed_qubit = _extract_qubit_id(seed_node)
        if seed_qubit is not None:
            partition_qubits[bin_idx].add(seed_qubit)
            used[bin_idx] = len(partition_qubits[bin_idx])
        
        # Initialize frontier with seed's neighbors
        frontier = []  # Min-heap of (time, counter, node)
        counter = [0]  # Counter for tie-breaking
        rejected = set()  # Nodes rejected due to capacity in this bin
        _push_next_valid_neighbors(hdh_graph, seed_node, frontier, unassigned, inc, pins, counter)
        
        # Greedy temporal expansion with cost-aware selection
        while used[bin_idx] < cap:
            # Select best node from frontier using delta cost (excluding rejected)
            next_node = _select_best_from_frontier_with_rejected(frontier, unassigned, rejected,
                                                                  bin_idx, partitions, inc, pins, beam_k)
            
            if next_node is None:
                break  # No more valid neighbors
            
            # Check if adding this node would exceed capacity
            next_qubit = _extract_qubit_id(next_node)
            if next_qubit is not None:
                # Would this introduce a new qubit?
                if next_qubit not in partition_qubits[bin_idx]:
                    if used[bin_idx] + 1 > cap:
                        # Would exceed capacity, reject and try next candidate
                        rejected.add(next_node)
                        continue
            
            # Add node to partition immediately (not just to local variable)
            partitions[bin_idx].add(next_node)
            unassigned.remove(next_node)
            
            # Update qubit tracking
            if next_qubit is not None and next_qubit not in partition_qubits[bin_idx]:
                partition_qubits[bin_idx].add(next_qubit)
                used[bin_idx] = len(partition_qubits[bin_idx])
            
            # Push this node's neighbors to frontier
            _push_next_valid_neighbors(hdh_graph, next_node, frontier, unassigned, inc, pins, counter)
    
    # Phase 3: Residual best-fit placement with delta cost
    unplaceable_nodes = set()
    
    while unassigned:
        # Find the next unassigned node
        remaining = unassigned - unplaceable_nodes
        if not remaining:
            break  # All remaining nodes are unplaceable
        
        node = min(remaining, key=lambda n: (hdh_graph.time_map.get(n, 0), n))
        node_qubit = _extract_qubit_id(node)
        
        # Compute delta cost for each bin and find best fit
        best_bin = None
        best_delta = float('inf')
        
        for bin_idx in range(k):
            # Check capacity constraint
            can_add = True
            if node_qubit is not None:
                if node_qubit not in partition_qubits[bin_idx]:
                    if used[bin_idx] >= cap:
                        can_add = False
            
            if can_add:
                delta = _compute_delta_cost_simple(node, bin_idx, partitions, inc, pins)
                if delta < best_delta:
                    best_delta = delta
                    best_bin = bin_idx
        
        # Place node in best bin
        if best_bin is not None:
            partitions[best_bin].add(node)
            unassigned.remove(node)
            
            # Update qubit tracking only if this is a new qubit for this bin
            if node_qubit is not None and node_qubit not in partition_qubits[best_bin]:
                partition_qubits[best_bin].add(node_qubit)
                used[best_bin] = len(partition_qubits[best_bin])
        else:
            # Try teledata fallback: find a bin where this qubit already exists
            placed = False
            if node_qubit is not None:
                for b in range(k):
                    if node_qubit in partition_qubits[b]:
                        partitions[b].add(node)
                        unassigned.remove(node)
                        placed = True
                        break
            
            if not placed:
                # Node is truly unplaceable, mark it and try next node
                unplaceable_nodes.add(node)
    
    # Compute cost (count cut hyperedges)
    node_assignment = {}
    for partition_idx, partition_nodes in enumerate(partitions):
        for node in partition_nodes:
            node_assignment[node] = partition_idx
    
    cost = _compute_cut_cost(hdh_graph, node_assignment)
    
    return partitions, cost

# ------------------------------- Cost Evaluation Functions -------------------------------

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


def weighted_cost(cost_tuple: Tuple[float, float]) -> float:
    """
    Apply weighting to cost tuple: quantum cuts cost 10x more than classical.
    
    Weighting scheme:
        - Quantum hyperedge cut cost = 10
        - Classical hyperedge cut cost = 1
    
    Args:
        cost_tuple: (cost_q, cost_c) from cost() function
    
    Returns:
        Weighted total cost: 10 * cost_q + 1 * cost_c
    """
    cost_q, cost_c = cost_tuple
    return 10.0 * cost_q + 1.0 * cost_c


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


def partition_logical_qubit_size(partitions) -> List[int]:
    """Return the number of *unique logical qubits* used in each partition.

    Notes
    -----
    - HDH node IDs are time-expanded (e.g., ``q7_t16``), so counting nodes vastly
      overestimates resource usage.
    - In this codebase, capacity ``cap`` is defined in *logical qubits*.

    Args:
        partitions: List[set[str]]; each set contains HDH node IDs.

    Returns:
        List[int]: unique logical-qubit count per partition.
    """
    if not partitions:
        return []

    sizes: List[int] = []
    for part in partitions:
        qubits = set()
        for nid in part:
            m = _Q_RE.match(str(nid))
            if m:
                qubits.add(int(m.group(1)))
        sizes.append(len(qubits))
    return sizes


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
        _, qubit_parts = nxmetis.partition(G, partitions)
    except Exception:
        qubit_parts = _kl_fallback_partition(G, partitions)

    bins = _bins_from_parts(qubit_parts)
    bins = _repair_overflow(G, bins, capacities)

    cut_cost = _cut_edges_unweighted(G, bins)
    sizes = _sizes(bins)
    respects_capacity = all(s <= capacities for s in sizes)
    method = "metis" if used_metis else "kl"
    return bins, cut_cost, respects_capacity, method


def participation(hdh_graph, partitions) -> Dict[str, float]:
    """
    Count how many partitions have any activity at each timestep.
    
    This metric measures temporal participation, not true concurrency.
    It answers: "How many QPUs are doing *something* at each timestep?"
    but doesn't measure actual concurrent work.
    
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
    
    # Count active partitions per timestep
    timestep_participation = []
    for t in sorted(hdh_graph.T):
        active_partitions = set()
        for node, time in hdh_graph.time_map.items():
            if time == t and node in node_to_partition:
                active_partitions.add(node_to_partition[node])
        timestep_participation.append(len(active_partitions))
    
    # Calculate metrics
    num_timesteps = len(hdh_graph.T) if hdh_graph.T else 1
    max_participation = max(timestep_participation) if timestep_participation else 0
    avg_participation = sum(timestep_participation) / num_timesteps if num_timesteps > 0 else 0
    
    # Temporal efficiency: average participation as fraction of total partitions
    temporal_efficiency = avg_participation / len(partitions) if partitions else 0
    
    # Partition utilization: how many partitions are used at all
    used_partitions = set()
    for node, partition_idx in node_to_partition.items():
        used_partitions.add(partition_idx)
    partition_utilization = len(used_partitions) / len(partitions) if partitions else 0
    
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

# Keeping old name as alias for backward compatibility (deprecated)
def compute_parallelism_by_time(hdh_graph, partitions) -> Dict[str, float]:
    """
    Deprecated: Use `parallelism()` instead.
    
    This function now calls `parallelism()` for backward compatibility.
    """
    return parallelism(hdh_graph, partitions)
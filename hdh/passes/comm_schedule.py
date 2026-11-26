"""
Temporal inter-partition communication schedule extraction and traffic-profile metrics.

Extends `hdh.passes.cut` beyond scalar cut counts by exposing *when* and *between which
partitions* cross-device communication occurs across an HDH partition:

- `communication_layers()`: maps each timestep to the multiset of cross-partition pairs
  induced by cut hyperedges (accepting either node-level partitions from `compute_cut` /
  `kahypar_cutter` or wire-level bins from `metis_telegate`).
- `pack_matching_rounds()`: edge-colors a timestep's cross-partition pairs into
  vertex-disjoint simultaneous communication rounds (conflict-free concurrent links).
- `communication_metrics()`: summarizes communication depth, peak/average concurrent
  inter-partition pairs, traffic dynamism (`unique_matchings`, `matching_reuse_ratio`),
  per-pair demand counts, and resource-round requirements under a configurable per-round
  multipartite or switch concurrency capacity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

Pair = Tuple[int, int]
Matching = Tuple[Pair, ...]


def communication_layers(
    hdh,
    partitions: Sequence[Set[str]],
    *,
    quantum_only: bool = True,
) -> Dict[int, List[Pair]]:
    """Extract the temporal sequence of cross-partition communication pairs.

    Each cut hyperedge is assigned to the maximum timestep of its incident nodes
    (matching `hdh.passes.cut.parallelism`). A cut hyperedge spanning ``k >= 2``
    partitions is decomposed into a star of ``k - 1`` unordered partition pairs
    anchored at the lowest-index touched partition.

    Accepts both node-level partitions (sets of HDH node IDs, e.g. from `compute_cut`
    or `kahypar_cutter`) and wire-level bins (sets of wire labels, e.g. from
    `metis_telegate`).

    Args:
        hdh: An `HDH` instance.
        partitions: Sequence of partition sets (node IDs or wire labels).
        quantum_only: If ``True`` (default), include only quantum (``"q"``) hyperedges.

    Returns:
        Sorted dictionary ``{timestep: [(u, v), ...]}`` with ``u < v`` for every pair.
    """
    if not partitions:
        return {}

    item_to_part: Dict[str, int] = {
        item: idx for idx, part in enumerate(partitions) for item in part
    }
    wire_of = getattr(hdh, "wire_of", {})
    hyperedge_types = getattr(hdh, "hyperedge_types", {})
    time_map = getattr(hdh, "time_map", {})

    layers: Dict[int, List[Pair]] = defaultdict(list)

    for edge in hdh.hyperedges:
        if quantum_only and hyperedge_types.get(edge, "q") != "q":
            continue

        touched_set: Set[int] = set()
        for node in edge:
            if node in item_to_part:
                touched_set.add(item_to_part[node])
            elif node in wire_of and wire_of[node] in item_to_part:
                touched_set.add(item_to_part[wire_of[node]])

        if len(touched_set) < 2:
            continue

        touched = sorted(touched_set)
        edge_times = [time_map[n] for n in edge if n in time_map]
        t_edge = max(edge_times) if edge_times else 0

        anchor = touched[0]
        for other in touched[1:]:
            layers[t_edge].append((anchor, other))

    return {t: sorted(pairs) for t, pairs in sorted(layers.items())}


def pack_matching_rounds(pairs: Sequence[Pair]) -> List[Matching]:
    """Pack a multiset of partition pairs into vertex-disjoint simultaneous matching rounds.

    Each returned matching represents a set of inter-partition links that share no
    endpoint partition and can therefore execute concurrently without endpoint contention.
    Self-loops (``u == v``) are ignored.

    Args:
        pairs: Sequence of ``(u, v)`` partition pairs requested at a timestep.

    Returns:
        List of sorted matchings ``[((u1, v1), ...), ...]`` with ``u_i < v_i``.
    """
    remaining: List[Pair] = [
        (u, v) if u < v else (v, u) for u, v in pairs if u != v
    ]
    rounds: List[Matching] = []

    while remaining:
        used: Set[int] = set()
        current: List[Pair] = []
        deferred: List[Pair] = []
        for u, v in remaining:
            if u not in used and v not in used:
                current.append((u, v))
                used.add(u)
                used.add(v)
            else:
                deferred.append((u, v))
        if current:
            rounds.append(tuple(sorted(current)))
        remaining = deferred

    return rounds


def communication_metrics(
    hdh,
    partitions: Sequence[Set[str]],
    *,
    quantum_only: bool = True,
    round_capacity: Optional[int] = None,
) -> Dict[str, object]:
    """Compute temporal communication-schedule and traffic-profile metrics for an HDH partition.

    Args:
        hdh: An `HDH` instance.
        partitions: Node-level partitions or wire-level bins.
        quantum_only: If ``True`` (default), consider only quantum hyperedges.
        round_capacity: Optional maximum number of simultaneous pairs a single
            network resource round can serve (e.g. a bounded switch or multipartite
            buffer capacity ``c >= 1``). When provided, ``rounds_at_capacity`` reports
            the total resource rounds needed (sum over matchings of ``ceil(|M| / c)``).

    Returns:
        Dictionary with keys:
        - ``total_cut_pairs`` (int): Total pairwise cross-partition demands.
        - ``active_layers`` (int): Number of timesteps with at least one cut hyperedge.
        - ``matching_rounds`` (int): Total vertex-disjoint matching rounds (comm depth).
        - ``max_concurrent_pairs`` (int): Peak simultaneous disjoint pairs in one round.
        - ``avg_concurrent_pairs`` (float): Mean pairs per matching round.
        - ``unique_matchings`` (int): Distinct matching patterns observed across rounds.
        - ``matching_reuse_ratio`` (float): ``1 - unique_matchings / matching_rounds``
          (0.0 when every round is distinct; approaches 1.0 for static/repeating traffic).
        - ``pair_demand`` (dict[tuple[int, int], int]): Total demand count per QPU pair.
        - ``rounds_at_capacity`` (int, optional): Included when ``round_capacity`` is set.
    """
    if round_capacity is not None and round_capacity < 1:
        raise ValueError(f"round_capacity must be >= 1, got {round_capacity}")

    layers = communication_layers(hdh, partitions, quantum_only=quantum_only)
    all_rounds: List[Matching] = []
    pair_demand: Dict[Pair, int] = defaultdict(int)

    for pairs in layers.values():
        for pair in pairs:
            pair_demand[pair] += 1
        all_rounds.extend(pack_matching_rounds(pairs))

    total_pairs = sum(pair_demand.values())
    n_rounds = len(all_rounds)
    max_conc = max((len(m) for m in all_rounds), default=0)
    avg_conc = (total_pairs / n_rounds) if n_rounds else 0.0
    unique_m = len(set(all_rounds))
    reuse = (1.0 - unique_m / n_rounds) if n_rounds else 0.0

    result: Dict[str, object] = {
        "total_cut_pairs": total_pairs,
        "active_layers": len(layers),
        "matching_rounds": n_rounds,
        "max_concurrent_pairs": max_conc,
        "avg_concurrent_pairs": avg_conc,
        "unique_matchings": unique_m,
        "matching_reuse_ratio": reuse,
        "pair_demand": dict(sorted(pair_demand.items())),
    }

    if round_capacity is not None:
        result["rounds_at_capacity"] = sum(
            (len(m) + round_capacity - 1) // round_capacity for m in all_rounds
        )

    return result

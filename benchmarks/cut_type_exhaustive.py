"""
Exact (exhaustive) version of the teledata/telegate/combined comparison in
cut_type_comparison.py. Removes all heuristic-quality confounds by finding
the TRUE optimal cut cost under each mode's constraints, via a branch-and-
bound search with capacity pruning - same spirit as the underlying HDH
manuscript's own Section 4.2 methodology (which validates compute_cut's
heuristic quality against exhaustive search on small instances).

Only tractable for small circuits (few qubits, shallow depth) - use
cut_type_comparison.cut_by_mode's greedy for anything larger.
"""
import time
from collections import defaultdict

from hdh.hdh import HDH
from .cut_type_comparison import _build_units, _build_unit_edges, cut_by_mode


def exhaustive_optimal(hdh: HDH, k: int, cap: int, mode: str, time_limit_s: float = 60):
    """True-optimal cut cost under `mode`'s constraints, via branch-and-bound.

    Args:
        hdh: HDH to partition.
        k: number of devices.
        cap: max distinct qubits per device.
        mode: "combined", "telegate_only", or "teledata_only".
        time_limit_s: search budget; if exceeded, returns the best solution
            found so far (not necessarily optimal - check `timed_out`).

    Returns:
        (best_cost, timed_out, n_units): best_cost is None only if no
        feasible placement was found within the time budget (or none
        exists, e.g. a gate needs more qubits than any device has capacity
        for). n_units is the search space size, for diagnostics.
    """
    units, unit_qubits, node_to_unit = _build_units(hdh, mode)
    unit_edges = list(_build_unit_edges(hdh, node_to_unit))
    n = len(units)

    # process most-constrained units (most qubits) first, for better pruning
    order = sorted(range(n), key=lambda i: (-len(unit_qubits[units[i]]), i))

    # seed with the greedy result as an initial upper bound, so pruning is
    # effective from the very first branch rather than only after the first
    # complete assignment is found
    try:
        greedy_cost, _ = cut_by_mode(hdh, k, cap, mode)
        best = [greedy_cost]
    except RuntimeError:
        best = [None]

    assign = [-1] * n
    bin_qubits = [set() for _ in range(k)]

    # branch-and-bound: precompute, for each processing position, which
    # edges become fully-assigned exactly at that position (i.e. this unit
    # is the last of the edge's members to be placed) - lets us add each
    # edge's cut/not-cut contribution to a running cost exactly once,
    # incrementally, instead of rescanning all edges at every leaf.
    unit_index = {u: i for i, u in enumerate(units)}
    edges_by_units = [[unit_index[u] for u in e] for e in unit_edges]
    order_position = {u_idx: pos for pos, u_idx in enumerate(order)}
    edges_finalized_at = defaultdict(list)
    for e in edges_by_units:
        last_pos = max(order_position[i] for i in e)
        edges_finalized_at[last_pos].append(e)

    start = time.time()
    timed_out = [False]

    def backtrack(pos, running_cost):
        if timed_out[0]:
            return
        if time.time() - start > time_limit_s:
            timed_out[0] = True
            return
        # cost only ever grows as more units are placed, so a running cost
        # already >= the best complete solution found can never improve on it
        if best[0] is not None and running_cost >= best[0]:
            return
        if pos == n:
            best[0] = running_cost
            return
        u_idx = order[pos]
        u = units[u_idx]
        for b in range(k):
            new_qubits = unit_qubits[u] - bin_qubits[b]
            if len(bin_qubits[b]) + len(new_qubits) > cap:
                continue
            assign[u_idx] = b
            bin_qubits[b] |= unit_qubits[u]
            added = 0
            for e in edges_finalized_at[pos]:
                if len({assign[i] for i in e}) > 1:
                    added += 1
            backtrack(pos + 1, running_cost + added)
            bin_qubits[b] -= new_qubits
            assign[u_idx] = -1
            if timed_out[0]:
                return

    backtrack(0, 0)
    return best[0], timed_out[0], n

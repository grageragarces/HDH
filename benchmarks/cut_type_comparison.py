"""
Compares three partitioning modes on real circuits, to test how much HDH's
combined teledata+telegate expressivity helps relative to the qubit-level
formulation used by prior hypergraph partitioning approaches:

- combined:      full HDH node-level partitioning (teledata + telegate both
                  allowed) - what HDH itself provides.
- telegate_only: each qubit's whole timeline is contracted to one unit (no
                  teledata cuts possible). This matches prior qubit-level
                  hypergraph formulations, e.g. Andres-Martinez et al. 2019,
                  where a qubit is assigned to exactly one device for the
                  entire circuit.
- teledata_only: each multi-qubit gate's node group is contracted to one unit
                  (no telegate cuts possible - a gate's qubits must always be
                  co-located). A qubit's timeline can still be split between
                  gates (i.e. teleported). Not a real prior method; included
                  as a completeness baseline, not because any existing
                  library implements exactly this restriction.

The same placement algorithm (`cut_by_mode`, a simple greedy) is used for all
three modes, varying only which HDH nodes get pre-contracted into a single
atomic unit before placement - this isolates the effect of cut-type
expressivity from any algorithm-quality differences between compute_cut /
metis_telegate / kahypar_cutter, which each use a different underlying
heuristic. `cut_type_exhaustive.py` provides a true-optimal (exhaustive
search) version of the same comparison for small instances, removing even
the greedy-quality confound.

Capacity accounting matches the formal Mapping Problem definition in the
underlying HDH manuscript: capacity counts the number of distinct qubits
assigned to a device across the whole computation, not per-timestep - so a
teleported qubit costs capacity on every device it ever visits.
"""
import re
from collections import defaultdict

from hdh.hdh import HDH

_Q_RE = re.compile(r"^q(\d+)_t\d+$")


def _qubit_of(node_id: str):
    m = _Q_RE.match(node_id)
    return int(m.group(1)) if m else None


def _build_units(hdh: HDH, mode: str):
    """
    Returns:
        units: list of frozenset(node_id) - atomic assignable groups
        unit_qubits: dict unit -> set of qubit indices it touches
        node_to_unit: dict node_id -> unit
    """
    node_to_unit = {}

    if mode == "combined":
        for n in hdh.S:
            node_to_unit[n] = frozenset([n])

    elif mode == "teledata_only":
        # contract each multi-qubit gate's stage2 edge into one unit
        parent = {n: n for n in hdh.S}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in hdh.C:
            name = hdh.gate_name.get(edge, "")
            if name.endswith("_stage2"):
                nodes = list(edge)
                for n in nodes[1:]:
                    union(nodes[0], n)

        groups = defaultdict(set)
        for n in hdh.S:
            groups[find(n)].add(n)
        for rep, members in groups.items():
            u = frozenset(members)
            for n in members:
                node_to_unit[n] = u

    elif mode == "telegate_only":
        # contract each qubit's entire timeline into one unit
        groups = defaultdict(set)
        for n in hdh.S:
            q = _qubit_of(n)
            key = ("q", q) if q is not None else ("node", n)
            groups[key].add(n)
        for key, members in groups.items():
            u = frozenset(members)
            for n in members:
                node_to_unit[n] = u

    else:
        raise ValueError(mode)

    units = sorted(set(node_to_unit.values()), key=lambda u: min(u))
    unit_qubits = {}
    for u in units:
        qs = {_qubit_of(n) for n in u}
        qs.discard(None)
        unit_qubits[u] = qs

    return units, unit_qubits, node_to_unit


def _build_unit_edges(hdh: HDH, node_to_unit: dict):
    """Map original HDH hyperedges onto units; drop self-loops (edges fully
    inside one unit, e.g. a contracted stage2 edge in teledata_only mode).

    Returns a list (NOT a set/dedup) - contraction can make two distinct
    original gate occurrences collapse onto the same pair of units (e.g. two
    separate CX gates between the same qubit pair, once each qubit's whole
    timeline is one unit in telegate_only mode). Each occurrence must still
    count as its own edge for cut cost, or repeated interactions between the
    same pair get silently undercounted.
    """
    unit_edges = []
    for edge in hdh.C:
        us = frozenset(node_to_unit[n] for n in edge if n in node_to_unit)
        if len(us) > 1:
            unit_edges.append(us)
    return unit_edges


def cut_by_mode(hdh: HDH, k: int, cap: int, mode: str):
    """Greedy placement under `mode`'s constraints.

    Args:
        hdh: HDH to partition.
        k: number of devices.
        cap: max distinct qubits per device (see module docstring on how
            capacity is counted for teleported qubits).
        mode: "combined", "telegate_only", or "teledata_only".

    Returns:
        (cut_cost, bin_sizes): total cut hyperedges, and qubit count per bin.

    Raises:
        RuntimeError: if no feasible placement exists for some unit under
            the given k/cap.
    """
    units, unit_qubits, node_to_unit = _build_units(hdh, mode)
    unit_edges = _build_unit_edges(hdh, node_to_unit)

    incident = defaultdict(list)
    for e in unit_edges:
        for u in e:
            incident[u].append(e)

    bins = [set() for _ in range(k)]
    bin_qubits = [set() for _ in range(k)]

    def cost_if_placed(u, b):
        delta = 0
        for e in incident[u]:
            placed_bins = {
                bi for bi, bu in enumerate(bins) for m in e if m in bu
            }
            will_be = placed_bins | {b}
            was_cut = len(placed_bins) > 1
            now_cut = len(will_be) > 1
            if not was_cut and now_cut:
                delta += 1
            elif was_cut and not now_cut:
                delta -= 1
        return delta

    for u in units:
        candidates = []
        for b in range(k):
            new_qubits = unit_qubits[u] - bin_qubits[b]
            if len(bin_qubits[b]) + len(new_qubits) > cap:
                continue
            d = cost_if_placed(u, b)
            # tie-break: prefer bins that already hold more of this unit's
            # qubits, to avoid needlessly fragmenting a qubit's capacity
            # footprint across bins (matches compute_cut's own tie-break).
            candidates.append((d, len(new_qubits), b))
        if not candidates:
            raise RuntimeError(
                f"cut_by_mode({mode}): no feasible bin for unit touching qubits "
                f"{unit_qubits[u]} (k={k}, cap={cap})"
            )
        candidates.sort()
        best_b = candidates[0][2]
        bins[best_b].add(u)
        bin_qubits[best_b] |= unit_qubits[u]

    unit_bin = {u: b for b, us in enumerate(bins) for u in us}
    cut_cost = sum(1 for e in unit_edges if len({unit_bin[u] for u in e}) > 1)
    sizes = [len(q) for q in bin_qubits]
    return cut_cost, sizes

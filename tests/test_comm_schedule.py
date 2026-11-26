"""Tests for temporal inter-partition communication schedule extraction and metrics."""

import pytest

from hdh.hdh import HDH
from hdh.models.circuit import Circuit
from hdh.passes import (
    communication_layers,
    communication_metrics,
    compute_cut,
    pack_matching_rounds,
)
from hdh.passes.cut import metis_telegate


def _build_parallel_6qubit_hdh() -> HDH:
    """Two-layer 6-qubit circuit with 3 concurrent cross-partition CNOTs per layer."""
    c = Circuit()
    # Layer 1: (0,1), (2,3), (4,5)
    c.add_instruction("cx", [0, 1])
    c.add_instruction("cx", [2, 3])
    c.add_instruction("cx", [4, 5])
    # Layer 2: shifted matching (1,2), (3,4), (5,0)
    c.add_instruction("cx", [1, 2])
    c.add_instruction("cx", [3, 4])
    c.add_instruction("cx", [5, 0])
    return c.build_hdh()


def test_pack_matching_rounds_disjoint_and_conflicting():
    # Three disjoint pairs fit into a single concurrent round
    rounds = pack_matching_rounds([(0, 1), (2, 3), (4, 5)])
    assert rounds == [((0, 1), (2, 3), (4, 5))]

    # Star of 3 pairs sharing partition 0 requires 3 sequential rounds
    star_rounds = pack_matching_rounds([(0, 1), (0, 2), (0, 3)])
    assert len(star_rounds) == 3
    for r in star_rounds:
        assert len(r) == 1

    # Self-loops are ignored and endpoints normalized u < v
    assert pack_matching_rounds([(2, 2), (3, 1)]) == [((1, 3),)]


def test_communication_layers_node_and_wire_partitions():
    hdh = _build_parallel_6qubit_hdh()

    # Wire-level bins (as returned by metis_telegate)
    wire_bins = [{f"q{i}"} for i in range(6)]
    layers_wire = communication_layers(hdh, wire_bins)
    assert sum(len(v) for v in layers_wire.values()) == 6

    # Node-level partitions (as returned by compute_cut / kahypar_cutter)
    node_parts = [
        {nid for nid in hdh.nodes if hdh.wire_of.get(nid) == f"q{i}"}
        for i in range(6)
    ]
    layers_node = communication_layers(hdh, node_parts)
    assert layers_node == layers_wire


def test_communication_metrics_concurrency_and_capacity():
    hdh = _build_parallel_6qubit_hdh()
    wire_bins = [{f"q{i}"} for i in range(6)]

    m_full = communication_metrics(hdh, wire_bins, round_capacity=3)
    assert m_full["total_cut_pairs"] == 6
    assert m_full["max_concurrent_pairs"] == 3
    assert m_full["matching_rounds"] == 2
    assert m_full["unique_matchings"] == 2
    assert m_full["matching_reuse_ratio"] == pytest.approx(0.0)
    assert m_full["rounds_at_capacity"] == 2

    # With per-round capacity capped at 2, each 3-pair round splits into 2 resource rounds
    m_cap2 = communication_metrics(hdh, wire_bins, round_capacity=2)
    assert m_cap2["rounds_at_capacity"] == 4

    with pytest.raises(ValueError):
        communication_metrics(hdh, wire_bins, round_capacity=0)


def test_communication_metrics_works_with_hdh_partitioners():
    hdh = _build_parallel_6qubit_hdh()
    bins, _, _, _ = metis_telegate(hdh, 3, 2)
    metrics_metis = communication_metrics(hdh, bins)
    assert metrics_metis["total_cut_pairs"] >= 0
    assert metrics_metis["matching_rounds"] >= 0

    parts, _ = compute_cut(hdh, 3, 2)
    metrics_greedy = communication_metrics(hdh, parts)
    assert metrics_greedy["total_cut_pairs"] >= 0

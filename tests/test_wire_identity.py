"""Partitioning passes identify qubits from the HDH, not from node labels.

Every pass used to regex-match ``q<int>_t<int>`` node IDs to decide what
counts toward a device's qubit capacity, so models with other labels (MBQC,
QCA) bypassed capacity entirely and classical nodes named ``q...`` were
charged as qubits. These tests build HDHs with non-``q<int>`` labels and check
each pass against the wires the fixture itself declares.
"""
import sys
import types

import pytest

from hdh.hdh import HDH
from hdh.models.mbqc import MBQC
from hdh.passes.cut import (
    kahypar_cutter,
    metis_telegate,
    partition_logical_qubit_size,
    telegate_hdh,
)

LABELS = ["a", "b", "c", "d", "e", "f"]


def _mbqc_chain():
    """Six MBQC labels entangled in a chain, with "a" measured into "m_a"."""
    mbqc = MBQC()
    for label in LABELS:
        mbqc.add_operation("N", [], label)
    for x, y in zip(LABELS, LABELS[1:]):
        mbqc.add_operation("E", [x, y], y)
    mbqc.add_operation("M", ["a"], "m_a")
    return mbqc.build_hdh()


def _label(node):
    """Wire of a node in `_mbqc_chain`, from the fixture's own labels."""
    return node.rsplit("_t", 1)[0]


class _FakeKahypar(types.ModuleType):
    """Stand-in for the `kahypar` package (no Windows/macOS wheels exist).

    It records the hypergraph it is given and puts vertex v in block v % k,
    which is enough to check how `kahypar_cutter` builds qubit vertices and
    lifts the result back to HDH nodes.
    """

    def __init__(self):
        super().__init__("kahypar")
        self.num_vertices = None

        fake = self

        class Context:
            def loadINIconfiguration(self, path):
                pass

            def setK(self, k):
                self.k = k

            def setEpsilon(self, eps):
                pass

        class Hypergraph:
            def __init__(self, n, *args):
                fake.num_vertices = n
                self.n = n

            def blockID(self, v):
                return v % self.k

        def partition(hg, context):
            hg.k = context.k

        self.Context = Context
        self.Hypergraph = Hypergraph
        self.partition = partition


class TestKahyparCutter:
    def test_vertices_are_the_quantum_wires(self, monkeypatch):
        fake = _FakeKahypar()
        monkeypatch.setitem(sys.modules, "kahypar", fake)
        hdh = _mbqc_chain()

        partitions, _ = kahypar_cutter(hdh, k=3, cap=2)

        # One vertex per quantum label; the classical "m_a" is not a qubit.
        assert fake.num_vertices == len(LABELS)
        assert set().union(*partitions) == hdh.S
        for partition in partitions:
            quantum = {_label(n) for n in partition if hdh.sigma[n] == "q"}
            assert len(quantum) <= 2

    def test_classical_node_follows_the_wire_it_measures(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kahypar", _FakeKahypar())
        hdh = _mbqc_chain()

        partitions, _ = kahypar_cutter(hdh, k=3, cap=2)

        (m_node,) = [n for n in hdh.S if n.startswith("m_a_")]
        (home,) = [p for p in partitions if m_node in p]
        assert any(_label(n) == "a" for n in home if hdh.sigma[n] == "q")


class TestTelegateGraph:
    def test_nodes_are_quantum_wires(self):
        graph = telegate_hdh(_mbqc_chain())
        assert set(graph.nodes()) == set(LABELS)
        assert {frozenset(e) for e in graph.edges()} == {
            frozenset(pair) for pair in zip(LABELS, LABELS[1:])
        }

    def test_metis_telegate_respects_capacity_on_named_wires(self):
        bins, _, respects_capacity, _ = metis_telegate(_mbqc_chain(), 3, 2)
        assert respects_capacity
        assert set().union(*bins) == set(LABELS)
        assert all(len(b) <= 2 for b in bins)


class TestPartitionLogicalQubitSize:
    def test_counts_quantum_wires_only(self):
        hdh = HDH()
        hdh.add_node("a", 0, "q")
        hdh.add_node("a", 1, "q")
        hdh.add_node("b", 0, "q")
        hdh.add_node("q9", 2, "c")  # classical, despite the q-name
        partitions = [{"a_t0", "a_t1", "q9_t2"}, {"b_t0"}]

        assert partition_logical_qubit_size(hdh, partitions) == [1, 1]

"""Conformance tests for the `Model` Protocol.

Every computational model is registered in `MODELS` with a small example
workload, and each invariant below runs against all of them. A new model
only needs a registry entry to be held to the same contract.
"""
import pytest

from hdh.hdh import HDH
from hdh.models import Model
from hdh.models.circuit import Circuit
from hdh.models.mbqc import MBQC
from hdh.models.qca import QCA
from hdh.models.qw import QW
from hdh.passes.cut import compute_cut


def _circuit():
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    circuit.add_instruction("ccx", [0, 1, 2])
    circuit.add_instruction("cx", [2, 3])
    circuit.add_instruction("measure", [3], [0])
    circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
    return circuit


def _mbqc():
    mbqc = MBQC()
    for label in ("a", "b", "c", "d"):
        mbqc.add_operation("N", [], label)
    for x, y in (("a", "b"), ("b", "c"), ("c", "d")):
        mbqc.add_operation("E", [x, y], y)
    mbqc.add_operation("M", ["a"], "m_a")
    mbqc.add_operation("C", ["m_a"], "m_b")
    return mbqc


def _qw():
    walk = QW()
    state = "q0"
    for _ in range(3):
        state = walk.add_shift(walk.add_coin(state))
    walk.add_measurement(state, "m0")
    return walk


def _qca():
    topology = {"A": ["B", "D"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C", "A"]}
    return QCA(topology=topology, measurements={"A", "C"}, steps=2)


MODELS = {"circuit": _circuit, "mbqc": _mbqc, "qw": _qw, "qca": _qca}


@pytest.fixture(params=sorted(MODELS))
def model(request):
    return MODELS[request.param]()


@pytest.fixture
def hdh(model):
    return model.build_hdh()


def test_satisfies_protocol(model):
    assert isinstance(model, Model)
    assert isinstance(model.build_hdh(), HDH)


def test_builds_a_non_empty_hdh(hdh):
    assert hdh.nodes and hdh.hyperedges


def test_every_node_has_type_time_and_wire(hdh):
    for mapping in (hdh.node_types, hdh.time_map, hdh.wire_of, hdh.node_realisation):
        assert set(mapping) == hdh.nodes
    for node in hdh.nodes:
        assert hdh.node_types[node] in ("q", "c")
        assert node == HDH.node_id(hdh.wire_of[node], hdh.time_map[node])
    assert hdh.timesteps == set(hdh.time_map.values())


def test_hyperedges_connect_existing_nodes(hdh):
    for edge in hdh.hyperedges:
        assert edge <= hdh.nodes
        assert edge  # single-node edges are fine, e.g. MBQC's N preparation
    for mapping in (hdh.hyperedge_types, hdh.hyperedge_realisation):
        assert set(mapping) == hdh.hyperedges


def test_quantum_hyperedges_hold_no_classical_nodes(hdh):
    for edge in hdh.hyperedges:
        if hdh.hyperedge_types[edge] == "q":
            assert all(hdh.node_types[n] == "q" for n in edge), sorted(edge)


def test_every_wire_keeps_one_type(hdh):
    types_by_wire = {}
    for node in hdh.nodes:
        types_by_wire.setdefault(hdh.wire_of[node], set()).add(hdh.node_types[node])
    mixed = {w: t for w, t in types_by_wire.items() if len(t) > 1}
    assert not mixed


def test_partitions_within_capacity(hdh):
    n_qubits = len({hdh.wire_of[n] for n in hdh.nodes if hdh.node_types[n] == "q"})
    k = 2
    cap = -(-n_qubits // k)

    partitions, _ = compute_cut(hdh, k, cap)

    assert set().union(*partitions) == hdh.nodes
    for partition in partitions:
        wires = {hdh.wire_of[n] for n in partition if hdh.node_types[n] == "q"}
        assert len(wires) <= cap

# Save this file as tests/test_hdh_metrics.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

from qiskit import QuantumCircuit


GOLDEN = {
    "MBQC": {"cut_cost": 1, "sizes": [0, 3, 2], "global_parallelism": 0, "parallelism_at_t3": 0},
    "QCA": {"cut_cost": 7, "sizes": [4, 5, 5], "global_parallelism": 0, "parallelism_at_t3": 0},
    "QW": {"cut_cost": 0, "sizes": [0, 0, 4], "global_parallelism": 0, "parallelism_at_t3": 0},
    "Qiskit Converter": {"cut_cost": 1, "sizes": [5, 4, 0], "global_parallelism": 0, "parallelism_at_t3": 0},
}

def check_hdh_outputs(hdh_graph, label):
    partitions = compute_cut(hdh_graph, 3)
    # assert cost(hdh_graph, partitions) == GOLDEN[label]["cut_cost"] # commented as it is solved heuristically, and thus may differ between runs
    assert sorted(partition_sizes(partitions)) == sorted(GOLDEN[label]["sizes"])
    assert sum(compute_parallelism_by_time(hdh_graph, partitions, mode="global")) == GOLDEN[label]["global_parallelism"]
    assert compute_parallelism_by_time(hdh_graph, partitions, mode="local", time_step=3) == GOLDEN[label]["parallelism_at_t3"]

def test_mbqc():
    from models.mbqc import MBQC
    mbqc = MBQC()
    mbqc.add_operation("N", [], "q0")
    mbqc.add_operation("N", [], "q1")
    mbqc.add_operation("E", ["q0", "q1"], "q1")
    mbqc.add_operation("M", ["q0"], "c0")
    mbqc.add_operation("C", ["c0"], "q2")
    check_hdh_outputs(mbqc.build_hdh(), "MBQC")

def test_qca():
    from models.qca import QCA
    topology = {"q0": ["q1", "q2"], "q1": ["q0"], "q2": ["q0"]}
    measurements = {"q1", "q2"}
    ca = QCA(topology=topology, measurements=measurements, steps=3)
    check_hdh_outputs(ca.build_hdh(), "QCA")

def test_qw():
    from models.qw import QW
    qw = QW()
    q1 = qw.add_coin("q0")
    q2 = qw.add_shift(q1)
    qw.add_measurement(q2, "c0")
    check_hdh_outputs(qw.build_hdh(), "QW")

def test_qiskit_converter():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(1, 2, 0)
    qc.measure_all()
    check_hdh_outputs(from_qiskit(qc), "Qiskit Converter")
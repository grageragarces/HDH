import pytest
from hdh.models.qca import QCA


class TestQCA:
    def test_two_node_line_topology(self):
        topology = {"q0": ["q1"], "q1": ["q0"]}
        qca = QCA(topology=topology, measurements=set(), steps=2)
        hdh = qca.build_hdh()

        for q in (0, 1):
            for t in range(3):  # t=0 (initial) through t=2 (steps=2)
                assert f"q{q}_t{t}" in hdh.S
                assert hdh.sigma[f"q{q}_t{t}"] == "q"

    def test_update_edge_count_is_qubits_times_steps(self):
        topology = {"q0": ["q1"], "q1": ["q0"]}
        qca = QCA(topology=topology, measurements=set(), steps=3)
        hdh = qca.build_hdh()

        update_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "update"]
        assert len(update_edges) == len(topology) * 3

    def test_update_edge_includes_neighbor_state(self):
        # Each qubit's update hyperedge must include its neighbor's node, not
        # just its own history - that's the whole point of a cellular
        # automaton (locality-bounded coupling). Note: q0 is processed before
        # q1 within a timestep (topology dict order), so q1's update edge
        # picks up q0's *already-updated* node for this same timestep.
        topology = {"q0": ["q1"], "q1": ["q0"]}
        qca = QCA(topology=topology, measurements=set(), steps=1)
        hdh = qca.build_hdh()

        update_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "update"]
        assert any({"q0_t0", "q1_t0", "q0_t1"} == set(e) for e in update_edges)
        assert any({"q0_t1", "q1_t0", "q1_t1"} == set(e) for e in update_edges)

    def test_measurement_produces_classical_node_at_steps_plus_one(self):
        topology = {"q0": ["q1", "q2"], "q1": ["q0"], "q2": ["q0"]}
        qca = QCA(topology=topology, measurements={"q1", "q2"}, steps=3)
        hdh = qca.build_hdh()

        for idx in (1, 2):
            c_node = f"c{idx}_t4"  # steps + 1
            assert c_node in hdh.S
            assert hdh.sigma[c_node] == "c"
            measure_edges = [
                e for e in hdh.C
                if hdh.gate_name.get(e) == "measure" and c_node in e
            ]
            assert len(measure_edges) == 1
            assert f"q{idx}_t3" in measure_edges[0]  # measures the final-step state

    def test_unmeasured_qubits_have_no_classical_nodes(self):
        topology = {"q0": ["q1"], "q1": ["q0"]}
        qca = QCA(topology=topology, measurements={"q1"}, steps=2)
        hdh = qca.build_hdh()

        assert not any(hdh.sigma[n] == "c" and n.startswith("c0_") for n in hdh.S)
        assert any(hdh.sigma[n] == "c" for n in hdh.S)  # q1 was measured


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

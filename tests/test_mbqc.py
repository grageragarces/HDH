import pytest
from hdh.models.mbqc import MBQC


class TestMBQC:
    def _build_documented_example(self):
        # Matches the exact usage example from docs/models.md.
        mbqc = MBQC()
        mbqc.add_operation("N", [], "q0")
        mbqc.add_operation("N", [], "q1")
        mbqc.add_operation("E", ["q0", "q1"], "q1")
        mbqc.add_operation("M", ["q0"], "c0")
        mbqc.add_operation("C", ["c0"], "q2")
        return mbqc.build_hdh()

    def test_n_produces_a_quantum_node(self):
        hdh = self._build_documented_example()
        assert "q0_t0" in hdh.S
        assert hdh.sigma["q0_t0"] == "q"

    def test_e_entangles_input_and_output(self):
        hdh = self._build_documented_example()
        e_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "e"]
        assert len(e_edges) == 1
        # The E op reuses "q1"'s label for its output, at a new timestep.
        assert "q0_t0" in e_edges[0]
        assert hdh.tau[e_edges[0]] == "q"

    def test_m_measurement_produces_classical_output(self):
        hdh = self._build_documented_example()
        m_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "m"]
        assert len(m_edges) == 1
        c_nodes = [n for n in m_edges[0] if hdh.sigma[n] == "c"]
        assert len(c_nodes) == 1
        assert c_nodes[0].startswith("c0_")

    def test_c_correction_is_purely_classical(self):
        hdh = self._build_documented_example()
        c_edges = [e for e in hdh.C if hdh.gate_name.get(e) == "c"]
        assert len(c_edges) == 1
        assert hdh.tau[c_edges[0]] == "c"
        # Per docs/models.md: MBQC node labels don't have to follow the q_/c_
        # convention - the "C" op's output here is labelled "q2" but is
        # still a classical (sigma="c") node, since its type comes from the
        # op type, not the label.
        assert all(hdh.sigma[n] == "c" for n in c_edges[0])

    def test_operations_get_increasing_timesteps(self):
        hdh = self._build_documented_example()
        times = [hdh.time_map[n] for n in hdh.S]
        assert len(set(times)) == len(hdh.S)  # every node at a distinct tick
        assert sorted(times) == list(range(len(hdh.S)))

    def test_reusing_a_classical_label_as_quantum_input_raises(self):
        # Regression-style guard: mixing types on one label must fail loudly
        # (via HDH.add_node's type-mismatch check) rather than silently
        # corrupting the node, mirroring the same class of bug fixed for QW.
        mbqc = MBQC()
        mbqc.add_operation("N", [], "q0")
        mbqc.add_operation("M", ["q0"], "c0")
        mbqc.add_operation("E", ["c0", "q0"], "c0")  # reuses "c0" as quantum

        with pytest.raises(ValueError):
            mbqc.build_hdh()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

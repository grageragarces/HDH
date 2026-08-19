import pytest
from hdh.models.qw import QW


class TestQW:
    def test_coin_shift_measurement(self):
        w = QW()
        a = w.add_coin("q0")
        b = w.add_shift(a)
        w.add_measurement(b, "c0")

        hdh = w.build_hdh()

        c_nodes = [n for n in hdh.S if hdh.sigma[n] == "c"]
        assert len(c_nodes) == 1
        assert c_nodes[0].startswith("c0_")

    def test_measurement_output_id_matches_its_type(self):
        # Regression test for JOSS review issue #69: a classical measurement
        # output used to be able to reuse a "q"-labelled node ID (e.g.
        # "q2_t2"), silently overwriting that quantum node's type.
        w = QW()
        a = w.add_coin("q0")
        b = w.add_shift(a)
        w.add_measurement(b, "c0")

        hdh = w.build_hdh()

        for node_id in hdh.S:
            prefix = node_id[0]
            assert prefix == hdh.sigma[node_id], (
                f"node '{node_id}' has type '{hdh.sigma[node_id]}' but an "
                f"ID prefix of '{prefix}'"
            )

    def test_reusing_a_quantum_label_for_measurement_raises(self):
        # Misuse case straight from the review repro: passing an existing
        # quantum-state label as the measurement's classical output label
        # must fail loudly instead of corrupting that node's type.
        w = QW()
        a = w.add_coin("q0")
        b = w.add_shift(a)
        w.add_measurement(a, b)

        with pytest.raises(ValueError):
            w.build_hdh()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

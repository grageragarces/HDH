import os
import pytest


class TestKahyparConfigPackaging:
    def test_default_config_is_shipped_with_the_package(self):
        # Regression test for JOSS review issue #69: kahypar_cutter used to look
        # for its config file at a path relative to the caller's CWD
        # ("kahypar/config/..."), which only worked by accident from the repo
        # root and was never actually included in the installed package.
        from hdh.passes.cut import _DEFAULT_KAHYPAR_CONFIG

        assert os.path.isfile(_DEFAULT_KAHYPAR_CONFIG)
        assert _DEFAULT_KAHYPAR_CONFIG.startswith(
            os.path.dirname(os.path.dirname(__file__))
        ) or "hdh" in _DEFAULT_KAHYPAR_CONFIG

    def test_kahypar_cutter_finds_config_from_any_cwd(self, tmp_path, monkeypatch):
        kahypar = pytest.importorskip("kahypar")

        from qiskit import QuantumCircuit
        from hdh.converters.qiskit_converter import from_qiskit
        from hdh.passes.cut import kahypar_cutter

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        hdh = from_qiskit(qc)

        monkeypatch.chdir(tmp_path)  # somewhere with no local "kahypar/" dir
        partitions, cut_cost = kahypar_cutter(hdh, k=2, cap=3)

        assert set().union(*partitions) == hdh.S


class TestMetisTelegate:
    def test_falls_back_to_kl_and_reports_it_honestly(self):
        # Regression test: metis_telegate used to try `import nxmetis`, a
        # package that doesn't exist on PyPI, so the METIS code path could
        # never actually run for anyone. It's since fixed to use the real
        # `metis` package's API. Without METIS installed (this sandbox has
        # no system METIS C library), it must still degrade to the working
        # Kernighan-Lin fallback rather than raising.
        from qiskit import QuantumCircuit
        from hdh.converters.qiskit_converter import from_qiskit
        from hdh.passes.cut import metis_telegate

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        hdh = from_qiskit(qc)

        bins, cut_cost, respects_capacity, method = metis_telegate(
            hdh, partitions=2, capacities=3
        )

        assert method in ("kl", "metis")
        assert cut_cost >= 0
        assert sum(len(b) for b in bins) == hdh.get_num_qubits()


class TestOptionalConverterExports:
    """hdh.converters exposes SDK converters as attributes even when the
    corresponding optional extra isn't installed, instead of raising
    ImportError on `import hdh.converters`."""

    def test_optional_converters_are_present_or_none(self):
        import hdh.converters as converters

        for name in ("from_cirq", "from_pennylane", "to_pennylane", "from_braket"):
            assert hasattr(converters, name), f"hdh.converters is missing {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

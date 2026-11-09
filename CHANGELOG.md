# Changelog

## 1.0.0

First stable release, the version archived for the JOSS paper. It contains the
fixes below on top of 0.5.0; no 0.5.1 was published.

### Fixed

- `from_qiskit` (and so `from_qasm`) measures into the classical bit Qiskit
  names instead of the qubit's own index. Circuits whose qubit-to-clbit mapping
  is not the identity, such as a teleportation pattern with `measure(2, 0)`
  followed by a condition on clbit 0, were silently converted wrongly.
- `from_qiskit` raises `NotImplementedError` for `IfElseOp` constructs an HDH
  cannot represent (an else branch with gates, a measurement inside the body,
  nested control flow) instead of crashing or converting them incorrectly. An
  empty or barrier-only else branch is still accepted.

### Documentation

- The README QASM quickstart runs as written; `docs/passes.md` names the
  current cost functions and fixes several typos.
- The JOSS paper figures are regenerated at print resolution
  (`miscellaneous/table_figures.py`).

### Added

- `CITATION.cff` and `.zenodo.json` for archiving releases on Zenodo.

## 0.5.0

### Breaking

- `HDH.add_node` now takes the wire and time and builds the node ID itself,
  instead of taking an ID plus the type and time it already encoded:

  ```python
  # before
  hdh.add_node("q0_t1", "q", 1)
  # now
  hdh.add_node("q0", 1, "q")        # returns "q0_t1"; "q" is the default
  ```

  The wire is stored in the new `HDH.wire_of` map. Calls using the old
  signature raise a `TypeError` that shows the new form.
- `partition_logical_qubit_size(partitions)` is now
  `partition_logical_qubit_size(hdh, partitions)`, matching `cost`.

### Fixed

- Partitioners (`compute_cut`, `kahypar_cutter`, `metis_telegate`) decide what
  counts toward qubit capacity from `sigma` and `wire_of` instead of matching
  `q<int>_t<int>` node IDs. MBQC and QCA HDHs with other labels previously
  bypassed capacity entirely (#70).
- `QCA` accepts non-integer cell names (#71).
- `plot_hdh` reads wires and timesteps from the HDH, so non-circuit models
  render instead of being skipped (#5).

### Documentation

- Shorter API docstrings, each with an example that runs as a doctest in
  the test suite, covering every converter in both directions (#78, #61).
- CI installs the Cirq, PennyLane and Braket extras on Python 3.11 and 3.12,
  so their converter tests and examples run there instead of skipping.

### Added

- `NodeType`, `Realisation` and `EdgeRole` enums; `add_node` and
  `add_hyperedge` reject invalid values (#76).
- `HDH.node_id(wire, time)` returns the ID `add_node` would build.
- Readable names for the core attributes: `nodes`, `hyperedges`, `timesteps`,
  `node_types`, `hyperedge_types`, `node_realisation` and
  `hyperedge_realisation` alias `S`, `C`, `T`, `sigma`, `tau`, `upsilon` and
  `phi`, which remain as the paper's notation (#79).
- `hdh.models.Model` Protocol for computational models, with conformance
  tests every registered model must pass; see CONTRIBUTING.md (#80).

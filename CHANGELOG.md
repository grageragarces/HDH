# Changelog

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

### Added

- `NodeType`, `Realisation` and `EdgeRole` enums; `add_node` and
  `add_hyperedge` reject invalid values (#76).
- `HDH.node_id(wire, time)` returns the ID `add_node` would build.

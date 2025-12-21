# METIS Telegate Partitioner

METIS-based partitioning on the telegate (qubit interaction) graph. Converts HDH to qubit graph where edges represent gate interactions, then uses METIS library for partitioning.

**Contributor**: grageragarces
**Date**: 2025-12-20
**Library function:** `hdh.passes.cut.metis_telegate`

## Description

METIS-based partitioning on the telegate (qubit interaction) graph. Converts HDH to qubit graph where edges represent gate interactions, then uses METIS library for partitioning.

## Guarantees

- **Respects Capacity**: False
- **Deterministic**: False
- **Complete Assignment**: True

## References

- [HDH Documentation](https://grageragarces.github.io/HDH/)
- [Partitioning Guide](https://grageragarces.github.io/HDH/passes/)

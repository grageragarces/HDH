# Greedy HDH Partitioner

Greedy bin-filling algorithm operating directly on HDH hypergraph. Uses beam search to select candidates and automatically places all nodes of a qubit together.

**Contributor**: grageragarces
**Date**: 2025-12-20
**Library function:** `hdh.passes.cut.compute_cut`

## Description

Greedy bin-filling algorithm operating directly on HDH hypergraph. Uses beam search to select candidates and automatically places all nodes of a qubit together.

## Parameters

- **beam_k** (default: `3`): Beam width for candidate selection (higher = more thorough but slower)
- **reserve_frac** (default: `0.08`): Fraction of capacity to reserve until bins reach 80% (helps load balancing)
- **restarts** (default: `1`): Number of random restarts to try (best result is kept)
- **seed** (default: `0`): Random seed for reproducibility

## Guarantees

- **Respects Capacity**: True
- **Deterministic**: True
- **Complete Assignment**: True

## References

- [HDH Documentation](https://grageragarces.github.io/HDH/)
- [Partitioning Guide](https://grageragarces.github.io/HDH/passes/)

# Benchmarks

Reproducible scripts supporting the Research Impact Statement's claim about
combining teledata and telegate cuts.

- `cut_type_comparison.py` — core comparison logic: a greedy partitioner
  that can be restricted to `combined`, `telegate_only`, or `teledata_only`
  cut types by contracting different HDH nodes into atomic units before
  placement.
- `cut_type_exhaustive.py` — true-optimal (exhaustive, branch-and-bound)
  version of the same comparison, for small instances where heuristic
  quality shouldn't be a confound.
- `mqtbench_sweep.py` — runs the greedy comparison across real circuits
  from [MQT Bench](https://www.cda.cit.tum.de/mqtbench/). Requires
  `pip install mqt.bench` (not a project dependency — only needed to
  reproduce this specific benchmark).
- `switching_sweep.py` — exhaustive comparison on a parameterized circuit
  family where a qubit's interaction pattern shifts partway through the
  computation, sweeping how many times it shifts.
- `heuristic_vs_exhaustive.py` — compares the library's capacity-aware
  greedy partitioner (`hdh.passes.cut.compute_cut`) against a time-capped
  exhaustive enumeration of capacity-respecting assignments, on real MQT
  Bench circuits, in the tight-capacity regime (3 devices, network overhead
  1) used in Section 4.2 of the companion manuscript. Both sides are scored
  with the library's own cost model (`cost` + `weighted_cost`: 10 per cut
  quantum hyperedge, 1 per cut classical hyperedge), and the enumeration
  cross-checks its incremental bookkeeping against that scorer on every
  result it returns. Requires `pip install mqt.bench`.
  Run `python -m benchmarks.heuristic_vs_exhaustive --time-limit 300` to use
  the manuscript's per-circuit cap.

  A cost ratio below 1.0 does not mean the heuristic beat the optimum: it
  means the heuristic found a cheaper assignment than the capped search
  reached in the time allowed. Only rows with `exhaustive_timed_out ==
  False` have a proven optimum as the denominator.

Run either sweep directly (`python -m benchmarks.mqtbench_sweep` /
`python -m benchmarks.switching_sweep`) from the repo root; each writes a
CSV and a PNG to `benchmarks/results/` (gitignored — regenerate on demand).

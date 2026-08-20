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

Run either sweep directly (`python -m benchmarks.mqtbench_sweep` /
`python -m benchmarks.switching_sweep`) from the repo root; each writes a
CSV and a PNG to `benchmarks/results/` (gitignored — regenerate on demand).

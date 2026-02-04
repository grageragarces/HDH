# HDH Partitioning Utilities

Here is an overview of the partitioning utilities available in the HDH library, designed to distribute quantum computations across multiple devices.

---

## Partitioning HDHs for Distribution

The `hdh/passes` directory contains scripts for partitioning and manipulating HDH graphs. The primary file for partitioning is `cut.py`, which offers two main approaches: a greedy, HDH-aware method and a METIS-based method operating on a qubit graph representation (telegate).

---

## Greedy HDH Partitioner

The main partitioning function is `compute_cut`, which implements the **Capacity-Aware Greedy HDH Partitioner** (time-aware, node-level), matching the current implementation in `cut.py`. fileciteturn4file9

### Core mechanics (what changed vs the older version)

* **Node-level assignment:** individual HDH nodes like `q7_t16` are assigned, not whole logical qubits. fileciteturn4file4
* **Capacity is in unique logical qubits:** each bin tracks the set of `q` indices present; adding a node only “costs capacity” if it introduces a new logical qubit into that bin. fileciteturn4file4
* **No automatic sibling co-location:** temporal siblings of the same qubit are *not* forced into the same bin (teledata-style cuts are therefore possible). fileciteturn4file9
* **Temporal validity:** candidates are expanded through a time-respecting frontier (edges “activate” at the max time of their pins). fileciteturn4file18

### High-level algorithm (3 phases)

This is easiest to read as three phases, mirroring the code structure. fileciteturn4file4

#### Phase 1: Seed selection (per bin)

For each bin (QPU), pick the earliest unassigned node as the seed (ties broken deterministically by node id), place it, and initialise a frontier of temporally valid neighbours. fileciteturn4file4turn4file18

#### Phase 2: Temporal greedy expansion (priority frontier + beam-k scoring)

While the bin still has capacity left:

1. Maintain a **min-heap frontier** keyed by earliest valid connection time.
2. Pop up to `beam_k` earliest candidates (skipping candidates already rejected for this bin).
3. Score those candidates by **delta cut cost** (lower is better, negative means it reduces cuts).
4. Pick the best-scoring candidate; if it would exceed capacity (by introducing a new qubit when already at `cap`), reject it for this bin and try again.

This is “temporal greedy” for reach, with “beam-k” only used for *selection among the earliest few frontier nodes*, not a global beam search over all unassigned nodes. fileciteturn4file4turn4file6turn4file18

#### Phase 3: Residual best-fit placement (delta-cost)

Once the per-bin expansions are done, remaining nodes are placed using a best-fit rule:

* For the next earliest unassigned node, compute its delta cost into every bin that can accept it under capacity.
* Place it into the bin that minimises delta cost.
* If no bin can accept it under capacity, try a **teledata fallback** by placing it into any bin that already contains its logical qubit (so it does not consume additional capacity).
* If that is also impossible, mark it as unplaceable and continue.

This replaces the older “round-robin mop-up”. fileciteturn4file5

### Function signature

```python
def compute_cut(hdh_graph, k: int, cap: int, *,
                beam_k: int = 3,
                backtrack_window: int = 0,
                polish_1swap_budget: int = 0,
                restarts: int = 1,
                reserve_frac: float = 0.08,
                predictive_reject: bool = True,
                seed: int = 0) -> Tuple[List[Set[str]], int]
```

**Note:** in the current implementation, only `beam_k` is operational; the other keyword params are accepted for compatibility but not used. fileciteturn4file9

---

## METIS Telegate Partitioner

For an alternative approach to partitioning, the library provides the `metis_telegate` function, which leverages the METIS algorithm (with a fallback to the Kernighan-Lin algorithm if METIS is not available).

### Telegate Graph Construction

* **Graph transformation:** This method first converts the HDH into a "telegate" graph using the `telegate_hdh` function. In this representation:
    * **Nodes** are the qubits of the quantum circuit (labeled as `q{idx}`).
    * **Edges** represent quantum operations between qubits (i.e., their co-appearance in a quantum hyperedge).
    * **Edge weights** correspond to the multiplicity of interactions between two qubits.
* **Quantum operation filtering:** Only hyperedges marked as quantum operations (with `tau` attribute = "q") are considered when building the telegate graph.

### Partitioning Process

* **METIS partitioning:** The telegate graph is partitioned using the `nxmetis` library, which provides Python bindings to the highly efficient METIS graph partitioning tool.
    * If METIS is unavailable, the algorithm automatically falls back to the Kernighan-Lin bisection algorithm from NetworkX.
    * METIS attempts to respect capacity constraints through the `tpwgts` (target partition weights) and `ubvec` (unbalance vector) parameters.
* **Overflow repair:** Since METIS does not guarantee perfectly balanced partitions, a greedy rebalancing algorithm (`_repair_overflow`) is used to adjust the partitions and ensure that no bin exceeds its qubit capacity.
    * The repair algorithm uses a heuristic gain function (`_best_move_for_node`) to choose which qubits to move between bins.
    * It prioritizes moving qubits that minimize the increase in cut edges.


## KaHyPar hypergraph partitioner 

`cut.py` also includes a KaHyPar-based partitioner, taken from the \href{https://kahypar.org}{KaHyPar library}.

* **`kahypar_cutter` (qubit-level hypergraph):**
  * Vertices are *logical qubits*.
  * Each HDH hyperedge contributes a hyperedge over the qubits that appear in it.
  * KaHyPar then runs its multilevel hypergraph partitioning pipeline (coarsening → initial partition → refinement), configured by an INI file (e.g., `km1_kKaHyPar_sea20.ini`).
  * Capacity is expressed as a *balance constraint* via KaHyPar’s `epsilon` (derived from `cap` relative to the ideal target size `n/k`). fileciteturn4file3
  * This means the partitioner primarily “knows” about **balancing qubit counts**; it does not model HDH-specific capacity nuances (for example, heterogeneous per-QPU capacities, or time-expanded node effects), and any “capacity” notion lives inside the balance constraint.

* **`kahypar_cutter_nodebalanced` (HDH-node-level hypergraph):**
  * Vertices are *HDH nodes* (time-expanded).
  * Balance is therefore in **node count**, not in unique logical qubits.
  * As a result, it can produce partitions that look well-balanced to KaHyPar but **do not respect logical-qubit capacity** (it is intentionally a balance-only baseline for comparison). fileciteturn4file12

### Function Signature

```python
def metis_telegate(hdh: "HDH", partitions: int, capacities: int) -> Tuple[List[Set[str]], int, bool, str]
```

**Parameters:**
* `hdh`: The HDH graph to partition
* `partitions`: Number of partitions (k)
* `capacities`: Capacity per partition (in qubits)

**Returns:**
* A tuple of `(bins_qubits, cut_cost, respects_capacity, method)` where:
    * `bins_qubits` is a list of sets, each containing qubit IDs (as strings like `"q0"`, `"q1"`) in that partition
    * `cut_cost` is the number of edges crossing between partitions (unweighted)
    * `respects_capacity` is a boolean indicating whether all bins satisfy the capacity constraint
    * `method` is either `"metis"` or `"kl"` indicating which algorithm was used

---

## Cut Cost Evaluation

The quality of a partition is determined by the number of "cuts"—that is, the number of hyperedges that span across multiple bins. The library provides two functions for this purpose:

### `_total_cost_hdh`

Calculates the total cost of a partition on an HDH graph. This function:
* Iterates through all hyperedges in the HDH
* Counts a hyperedge as "cut" if its pins (nodes) are distributed across 2 or more bins
* Returns the sum of weights of all cut hyperedges
* Respects edge weights if defined in the HDH graph (defaults to 1 if not specified)

**Algorithm:** For each hyperedge, count the number of distinct bins containing at least one pin. If this count is ≥ 2, add the edge's weight to the total cost.

### `_cut_edges_unweighted`

Counts the number of edges that cross between different bins in a standard graph (used for evaluating telegate graph partitions). This function:
* Takes a NetworkX graph and a partition assignment
* Counts edges where the two endpoints are in different bins
* Returns an unweighted count (each cut edge counts as 1)

**Use case:** This is specifically used by `metis_telegate` to evaluate the quality of qubit-graph partitions.

---

## Helper Functions and Internal Components

The `cut.py` file contains helper utilities for the partitioners (note that some older helpers/classes remain in the file as legacy code paths).

### Temporal incidence + frontier utilities (greedy partitioner)

* `_build_temporal_incidence`: builds `inc[node] -> [(hyperedge, edge_time)]` and `pins[hyperedge] -> {nodes}`, where `edge_time = max(pin_times)`; used to enforce temporal validity. fileciteturn4file18
* `_push_next_valid_neighbors`: expands a node into the min-heap frontier, pushing only temporally valid neighbour candidates. fileciteturn4file18
* `_select_best_from_frontier_with_rejected`: takes the earliest `beam_k` frontier items (skipping rejected), evaluates delta cost, and returns the best candidate. fileciteturn4file6
* `_compute_delta_cost_simple`: delta in (unweighted) cut-hyperedge count if a node is added to a specific bin. fileciteturn4file18
* `_extract_qubit_id`: parses `q{idx}_t{t}` to extract the logical qubit index for capacity accounting. fileciteturn4file18

### METIS utilities

* `telegate_hdh`: converts the HDH to a qubit interaction graph (“telegate graph”). fileciteturn4file16
* `_repair_overflow`, `_best_move_for_node`, `_over_under`: post-processing used to fix capacity violations after METIS/KL. fileciteturn4file10
* `_cut_edges_unweighted`: unweighted cut-edge count for the telegate graph. fileciteturn4file10

---

## Notes on Evaluating Partitioners on Random Circuits

We would like to warn users and partitioning strategy developers that we have found partitioners to behave differently on real quantum workloads when compared to randomly generated ones. As such, we recommend not testing partitioners on randomly generated workloads unless that is specifically your goal.

**Key considerations:**
* **Circuit structure matters:** Real quantum algorithms often have characteristic patterns (e.g., layered structures, specific qubit interaction patterns) that random circuits lack.
* **Connectivity patterns:** Random circuits may not reflect the typical connectivity found in QAOA, VQE, quantum simulation, or other structured quantum algorithms.
* **Tagging in the database:** In the database, you can specify the origin of your workloads, tagging them as "random" if appropriate. This helps others understand the context of benchmark results.

When developing new partitioning strategies, we strongly encourage testing on workloads representative of your target applications rather than relying solely on random circuit benchmarks.

---

## Future Enhancements

The current `cut.py` implementation includes several parameters that are reserved for future enhancements:

* **Backtracking:** The `backtrack_window` parameter is currently unused but reserved for implementing backtracking search
* **Local search:** The `polish_1swap_budget` parameter is disabled for HDH partitioning (as moving whole qubits is computationally heavier than single-node swaps)
* **Predictive rejection:** The `predictive_reject` parameter is reserved for future heuristics

These features may be enabled in future versions of the library as the partitioning algorithms continue to evolve.

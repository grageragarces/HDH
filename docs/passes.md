# HDH Partitioning Utilities

Here is an overview of the partitioning utilities available in the HDH library, designed to distribute quantum computations across multiple devices.

---

## Partitioning HDHs for Distribution

The `hdh/passes` directory contains scripts for partitioning and manipulating HDH graphs. The primary file for partitioning is `cut.py`, which offers two main approaches: a greedy, HDH-aware method and a METIS-based method operating on a qubit graph representation (telegate).

---

## Greedy HDH Partitioner

The main partitioning function is `compute_cut`, which implements a greedy, bin-filling algorithm with hypergraph awareness. Here's how it works:

### Core Mechanics

* **Node-level assignment:** The partitioner assigns individual nodes of the HDH graph to bins.
* **Qubit-based capacity:** While the assignment is at the node level, the capacity of each bin is determined by the number of *unique qubits* it contains.
* **Automatic sibling placement:** Once a node corresponding to a particular qubit is placed in a bin, all other nodes associated with that same qubit are automatically assigned to the same bin. This ensures that all temporal instances of a qubit remain co-located.

### Algorithm Details

The algorithm operates in two main phases: a beam-search-based bin-filling phase, followed by a round-robin mop-up phase.

#### Phase 1: Main Bin-Filling (Beam Search)

**Step 1 - Ordering and Representative Selection:**

Each qubit in the quantum circuit may appear at multiple time steps (e.g., qubit 0 at time 1, time 5, time 12). The algorithm:
1. Selects ONE representative node for each qubit - specifically, the node with the highest weighted degree (most connections to other operations)
2. Sorts these representatives in descending order of their weighted degrees

**Why?** Placing highly connected qubits first gives the algorithm more context to make informed placement decisions. It's like seating the most social people at a party first - they help determine where their friends should sit.

**Step 2 - Beam Search for Candidate Selection:**

For each bin being filled, instead of greedily picking just the single best node (which might not fit due to capacity), the algorithm uses **beam search** to keep the top `beam_k` candidates.

**What is Beam Search?**

Beam search is a heuristic that balances between:
- **Greedy search** (k=1): Only considers the single best option - fast but inflexible
- **Exhaustive search** (k=∞): Considers all options - thorough but slow
- **Beam search** (k=3 default): Keeps the top k candidates - good balance

**Example with 50 unassigned qubits and beam_k=3:**

```
1. Evaluate all 50 qubits based on:
   - Delta cost: How much would placing this node increase cut cost?
   - Frontier score: How well connected is this node to nodes already in the bin?

2. Keep top 3 candidates:
   - Qubit 42: delta_cost=-5, frontier_score=100 (BEST)
   - Qubit 17: delta_cost=-4, frontier_score=95  (2nd best)
   - Qubit 23: delta_cost=-3, frontier_score=90  (3rd best)

3. Check capacity constraints for each in order:
   - Qubit 42: Would make bin have 26/25 qubits X
   - Qubit 17: Would make bin have 25/25 qubits ✓
   - Place qubit 17!
```

**Effect of changing beam_k:**
- `beam_k=1`: Fastest, but may miss good alternatives if the best choice violates capacity
- `beam_k=3` (default): Good balance - considers alternatives without much overhead
- `beam_k=10`: More thorough exploration, slower, potentially better quality for complex constraints

#### Phase 1 Capacity Management: The "Shadow Capacity" System

The algorithm uses a dynamic capacity model to encourage better load balancing across bins.

**How it works:**

With `cap=25` qubits per bin and `reserve_frac=0.08`:
- **Shadow capacity** = 25 × (1.0 - 0.08) = **23 qubits**
- **Hard capacity** = **25 qubits**
- **80% threshold** = 25 × 0.8 = **20 qubits**

| Qubits Currently in Bin | Effective Capacity Limit | Reasoning |
|-------------------------|-------------------------|-----------|
| 0-19 qubits | 23 (shadow limit) | "Save some room for later bins" |
| 20-25 qubits | 25 (hard limit) | "Now you can fill to the top" |

**Why this helps:**

Without shadow capacity, early bins might greedily fill to 100%, leaving later bins with insufficient room for well-connected qubits. The shadow capacity ensures more even distribution.

**Example scenario with 4 bins and 100 qubits:**
- **Without shadow capacity:** Bins might fill as [25, 25, 25, 25] perfectly, but if you have 102 qubits, 2 are stranded
- **With shadow capacity:** Bins fill as [23, 23, 23, 23] initially, then even out to [26, 25, 26, 25], accommodating all qubits better

#### Fallback Seeding

**When:** If all candidates in the beam violate capacity constraints, or if no good candidates exist.

**What happens:** The algorithm falls back to placing the first unassigned qubit representative that fits capacity, even if it's not optimal by the scoring criteria.

**Why necessary:** Prevents the algorithm from getting stuck when beam search finds no viable candidates. Ensures bins get initialized even in difficult cases.

#### Phase 2: Mop-up (Round-Robin Distribution)

After the main bin-filling completes, some qubits might remain unassigned. These are distributed round-robin:

```
For each remaining unassigned qubit:
  Try bins in order: 0, 1, 2, 3, 0, 1, 2, 3, ...
  Place in first bin with available capacity
  If no bins have space → qubit is not placed
```

**Example:**
```
5 qubits left: Q1, Q2, Q3, Q4, Q5
4 bins with capacity 25:
  Bin 0: 24/25 qubits
  Bin 1: 25/25 qubits (FULL)
  Bin 2: 23/25 qubits
  Bin 3: 24/25 qubits

→ Q1 placed in Bin 0 (now 25/25)
→ Q2 skips Bin 1 (full), placed in Bin 2 (now 24/25)
→ Q3 placed in Bin 3 (now 25/25)
→ Q4 placed in Bin 2 (now 25/25)
→ Q5 cannot be placed - all bins full
```

**Important:** The round-robin phase **respects the hard capacity limit**. It will NOT overfill bins. If all bins are at 100% capacity and qubits remain, those qubits are skipped (not placed).

**To avoid unplaced qubits:** Ensure `k × cap ≥ total_number_of_qubits` when calling the partitioner.

### Function Signature

```python
def compute_cut(hdh_graph, k: int, cap: int, *,
                beam_k: int = 3,
                backtrack_window: int = 0,
                polish_1swap_budget: int = 0,  # disabled for HDH
                restarts: int = 1,
                reserve_frac: float = 0.08,
                predictive_reject: bool = True,
                seed: int = 0) -> Tuple[List[Set[str]], int]
```

**Parameters:**
* `hdh_graph`: The HDH graph to partition
* `k`: Number of partitions (bins)
* `cap`: Capacity per bin (in unique qubits)
* `beam_k`: Beam width for candidate selection (default: 3)
* `backtrack_window`: Currently unused (reserved for future enhancements)
* `polish_1swap_budget`: Currently disabled for HDH partitioning
* `restarts`: Number of random restarts to try (default: 1)
* `reserve_frac`: Fraction of capacity to reserve until 80% full (default: 0.08)
* `predictive_reject`: Currently unused (reserved for future enhancements)
* `seed`: Random seed for reproducibility

**Returns:**
* A tuple of `(bins_nodes, cost)` where:
    * `bins_nodes` is a list of sets, each containing node IDs assigned to that bin
    * `cost` is the total cut cost (number of hyperedges spanning multiple bins)

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

The `cut.py` file contains several internal helper functions that support the main partitioning algorithms:

### Hypergraph Incidence Structure
* `_build_hdh_incidence`: Builds the incidence structure for efficient hypergraph queries
* `_group_qnodes`: Groups nodes by their associated qubit for whole-qubit placement

### State Management
* `_HDHState`: A class that tracks the current partition assignment, bin loads, qubit anchoring, and pin distributions during the greedy algorithm

### Cost Calculation
* `_delta_cost_hdh`: Computes the incremental change in cut cost when placing a node in a bin
* `_qubit_of`: Extracts the qubit index from a node ID using regex matching

### Candidate Selection
* `_best_candidates_for_bin`: Selects the top-k candidate nodes for placement using beam search
* `_first_unassigned_rep_that_fits`: Fallback function to seed bins when no good candidates are found

### Graph Repair
* `_repair_overflow`: Greedy rebalancer to enforce bin capacity after METIS partitioning
* `_best_move_for_node`: Heuristic to compute the gain of moving a node between bins
* `_over_under`: Identifies which bins are over or under capacity

### Partitioning Utilities
* `_kl_fallback_partition`: Implements recursive Kernighan-Lin bisection as a fallback when METIS is unavailable

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

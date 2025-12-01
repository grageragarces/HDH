# HDH Partitioning Utilities

Here is an overview of the partitioning utilities available in the HDH library, designed to distribute quantum computations across multiple devices.

---

## Partitioning HDHs for Distribution

The `hdh/passes` directory contains scripts for partitioning and manipulating HDH graphs. The primary file for partitioning is `cut.py`, which offers two main approaches: a greedy, HDH-aware method and a METIS-based method operating on a qubit graph representation (telegate).

---

## Partitioner

The main partitioning function is `compute_cut`, which implements a greedy, bin-filling algorithm. Here's how it works:

* **Node-level assignment:** The partitioner assigns individual nodes of the HDH graph to bins.
* **Qubit-based capacity:** While the assignment is at the node level, the capacity of each bin is determined by the number of *unique qubits* it contains.
* **Automatic sibling placement:** Once a node corresponding to a particular qubit is placed in a bin, all other nodes associated with that same qubit are automatically assigned to the same bin.
* **Ordering and selection:**
    * The algorithm first selects a representative node for each qubit, based on the weighted degree of the nodes.
    * These representatives are then sorted in descending order of their weighted degrees to prioritize high-connectivity qubits.
    * A beam search is used to select the best candidate nodes to place in each bin, considering the change in cut cost and a "frontier score" that measures how connected a node is to the nodes already in the bin.
* **Mop-up:** Any remaining unassigned nodes are distributed among the bins in a round-robin fashion, respecting the capacity constraints.

---

## METIS partitioners

For a different approach to partitioning, the library provides the `metis_telegate` function, which leverages the METIS algorithm (with a fallback to the Kernighan-Lin algorithm if METIS is not available).

* **Telegate graph:** This method first converts the HDH into a "telegate" graph. In this representation:
    * **Nodes** are the qubits of the quantum circuit.
    * **Edges** represent quantum operations between qubits (i.e., their co-appearance in a quantum hyperedge). The weight of each edge corresponds to the number of times the two connected qubits interact.
* **METIS partitioning:** The telegate graph is then partitioned using METIS, which is a highly efficient graph partitioning tool.
* **Overflow repair:** Since METIS does not guarantee perfectly balanced partitions, a greedy rebalancing algorithm (`_repair_overflow`) is used to adjust the partitions and ensure that no bin exceeds its qubit capacity.

---

## Cut evaluations

The quality of a partition is determined by the number of "cuts"—that is, the number of hyperedges that span across multiple bins. The library provides two functions for this purpose:

* `_total_cost_hdh`: Calculates the total cost of a partition on an HDH graph. This is the sum of the weights of all hyperedges that have nodes in more than one bin.
* `_cut_edges_unweighted`: Counts the number of edges that cross between different bins in a standard graph (used for evaluating the telegate graph partitions).
<!-- TODO: expand this -->

<!-- ---

## Primitive injections

While the `cut.py` file focuses on partitioning, the `hdh/passes/primitives.py` file is responsible for injecting the necessary "primitive" operations into the partitioned HDH. These primitives are the fundamental building blocks of distributed quantum computation and include operations like teleportation and entanglement swapping, which are essential for performing gates between qubits located in different bins. -->

---

## The partitioner leaderboard

For a detailed comparison of the performance of different partitioning strategies on various quantum circuits, please refer to the partitioner leaderboard in the repository's [database](https://github.com/grageragarces/HDH/tree/database-branch/database). This can provide valuable insights into which partitioning method is best suited for your specific needs. See the [database file in the documentation](database.md) for more details.

---

# Notes on evaluating partitioners on random circuits

We would like to warn users and partitioning strategy developpers that we have found partitioners to behave differently on real quantum workloads when compared to randomly generated ones. As such we recommend to not test partitioners on randomly generated workloads unless that is the goal. In the database you can specify the origin of your workloads, tagging it as random if it is so.
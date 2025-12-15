---
title: 'HDH: A Python package for distributed quantum computing'
tags:
  - Python
  - distributed quantum computing
  - quantum partitioning
  - hypergraphs
authors:
  - name: Maria Gragera Garces
    orcid: 0009-0000-9018-7435
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: University of Edinburgh
   index: 1
date: 15 December 2025
bibliography: paper.bib

---

# Summary

Quantum computing aims to solve computational problems that are classically hard. 
To achieve this in utility settings, quantum computers will require 
thousands if not millions of qubits. Current devices hold hundreds of qubits at most. 
It is believed that the path towards these scales will come from distribution, meaning the collaboration of various devices to complete tasks larger than their individual capacities.
The main goal behind Distributed Quantum Computing (DQC) is to allocate sub-partitions of large quantum computations across multiple devices smaller than the computation itself. Existing approaches abstract computations to hypergraphs which are then partitioned, but they lack a unified framework for comparing and developing partitioning strategies within and across computational models.


# Statement of need

`HDH` is a Python package designed for researchers to test and develop partitioning strategies 
for quantum computation. 
HDHs (Hybrid Dependency Hypergraphs) are an abstraction which transforms quantum computation, originating from any quantum computational model (including circuits, measurement-based quantum computing, quantum cellular automata, and quantum walks), to a directed hypergraph that expresses all possible partitions available within the computation.
They were originally proposed in [@Gragera:2025] as a unifying approach to quantum distribution, extending the hypergraph abstraction method for partitioning across devices originally proposed in [@Andres:2019].
Since then, various partitioning strategies have been proposed [@Clark:2023; @Escofet:2023; @Sundaram:2023], but many are tested on inconsistent hypergraph abstractions, making cross-partitioner testing and improvement impossible.
Having an easy to implement, open-source, and model-agnostic abstraction will enable the fair and consistent cross-comparison of partitioning strategies in future work. 
Furthermore, HDHs extend this capability beyond the circuit model, addressing a current blind spot in DQC research. 

`HDH` is designed to be used by both distributed quantum architecture researchers 
and compiler developers. No other libraries are dedicated to the specific advancement 
of partitioning heuristics based on directed hypergraph abstraction.
While quantum compilation frameworks like 
Qiskit [@Qiskit], Cirq [@Cirq], and PennyLane [@PennyLane] provide circuit 
optimization and device mapping, they do not offer model-agnostic abstractions 
for distributed quantum computing. Hypergraph-based approaches exist in specific 
contexts ([@Andres:2019], [@Clark:2023], [@Escofet:2023]), but lack a unified implementation that 
supports multiple computational models and provides a common platform for 
the development of partitioners.

## Model conversions

Any quantum computing model comprises a series of commands which establish qubit state 
rotations, measurements and entanglements. For instance, quantum circuits are 
comprised of a sequence of quantum gates applied to qubits. Single-qubit gates 
perform rotations on the Bloch sphere, while multi-qubit gates (such as CNOT) 
create entanglement dependencies between qubits.
Mapping a quantum workload such as a circuit to an HDH involves applying specific correspondences between model elements and hypergraph motifs. The library provides model specific classes such as the `Circuit` class that enables straightforward conversions using mapping tables:

![Circuit to HDH mapping table.\label{fig:circuit_mappings}](docs/img/circuitmappings.png)

These entanglement operations can be made non-local and thus partitioned through 
a quantum network via quantum communication primitives [@Wu:2022]. Alternatively, 
qubit states can be individually forwarded through teleportation protocols 
[@Moghadam:2017]. HDHs aim to showcase all these possible partitionings, enabling 
heuristic partitioners to exploit recurring patterns from quantum algorithm 
implementations and map workloads to quantum or hybrid networks whilst minimizing 
communication or other costs. 
This table shows how `HDHs` superseed previous abstractions in its expressivity of these options:
![Table showing HDH flexibility.\label{fig:comparison_table}](docs/img/comparison_table.png)

Mapping quantum workloads to HDHs involves applying specific correspondences 
between model elements and hypergraph motifs. The library provides model-specific 
classes such as the `Circuit` class that enables straightforward conversion:
```python
from hdh.models.circuit import Circuit

circuit = Circuit()
circuit.add_instruction("ccx", [0, 1, 2])
circuit.add_instruction("h", [3])
circuit.add_instruction("cx", [3, 4])
circuit.add_instruction("measure", [2])

hdh = circuit.build_hdh()  # Generate HDH representation
```

The resulting HDH is shown bellow as a graph representation of a hypergraph. This visualization strategy is chosen due to the unscalability of alternative hypergraph visuals %TODO reword:

![Example circuit and its HDH representation.\label{fig:circuit_example}](docs/img/hdhfromcircuit.png)


This table shows how `HDHs` superseed previous abstractions:
![Table showing HDH flexibility.\label{fig:comparison_table}](docs/img/comparison_table.png)



# The HDH database
To support reproducible evaluation and training of partitioning strategies, this library's git repository also includes a database of pre-generated HDHs as well as the partitions offered by some of the state of the art partitioners.
This resource aims to facilitate benchmarking across diverse workloads and enable the development of learning-based distribution agents.

# Acknowledgements

We acknowledge contributions from [Joseph Tedds](https://github.com/josephtedds), [Manuel Alejandro](https://github.com/manalejandro), and [Alessandro Cosentino](https://github.com/cosenal).

We thank Unitary Fund for supporting this project through their quantum microgrant program.

The work of the author is supported by the EPSRC UK Quantum Technologies Programme under grant EP/T001062/1 and VeriQloud.

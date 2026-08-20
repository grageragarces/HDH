---
title: 'HDH: A Python Library for Hypergraph-Based Distributed Quantum Computing Partitioning'
tags:
  - Python
  - distributed quantum computing
  - quantum partitioning
  - hypergraphs
authors:
  - name: Maria Gragera Garces
    orcid: 0009-0000-9018-7435
    affiliation: 1
affiliations:
 - name: University of Edinburgh
   index: 1
date: 15 December 2025
bibliography: paper.bib

---

 # Summary

Today's quantum computers are limited by how many qubits a single device can hold. Distributed quantum computing (DQC) works around this by linking multiple smaller devices together so they can jointly run computations too large for any one of them alone, which first requires deciding how to split, or partition, a computation across those devices.

`HDH` (Hybrid Dependency Hypergraphs) is a Python library that gives researchers a common representation to develop and compare partitioning strategies against. It converts a quantum computation — expressed as a circuit, a measurement-based pattern, a quantum walk, or a quantum cellular automaton — into a hypergraph that captures every way the computation could be split across devices, including hard constraints such as per-device qubit limits that prior abstractions treat as soft penalties rather than requirements. Researchers can run their own partitioning heuristics directly on this representation, or use `HDH`'s built-in capacity-aware baseline, and compare results on a consistent, model-agnostic footing. `HDH` also converts to and from popular quantum SDKs (Qiskit, Cirq, PennyLane, Amazon Braket), so partitioned results can be turned back into runnable circuits.

 # State of the Field 

Quantum computing aims to solve computational problems that are classically hard. 
To achieve this in utility settings, quantum computers will require 
thousands if not millions of qubits. Current devices hold hundreds of qubits at most. 
It is believed that the path towards these scales will come from distribution, meaning the collaboration of various devices to complete tasks larger than their individual capacities.
The main goal behind Distributed Quantum Computing (DQC) is to allocate sub-partitions of large quantum computations across multiple devices smaller than the computation itself. 
Existing approaches abstract computations to hypergraphs which are then partitioned, using [Balanced Hypergraph Partitioning]{.smallcaps} solvers such as KaHyPar [@schlag2023high].

This framing has two fundamental limitations:

1. It reduces distribution to a balanced partitioning problem that ignores a hard physical constraint: individual QPUs have fixed qubit capacities, and a valid distribution must respect these limits strictly rather than treating them as soft penalties.
2. Existing hypergraph abstractions are model-specific and encode only a subset of possible partition cuts, meaning partitioning strategies are routinely evaluated on inconsistent abstractions, making cross-comparison unreliable and hindering the systematic development of improved heuristics.

While libraries for distributed quantum computing exist 
(DISQCO [@burt2026multilevel], Qdislib [@tejedor2025orchestrating], 
Optyx [@kupper2025optyx], DC-MBQC [@xue2026dc]), these implement end-to-end 
distribution pipelines rather than exposing the underlying abstraction as a 
research tool. No existing library provides a model-agnostic hypergraph abstraction 
designed specifically to enable the development and fair comparison of partitioning 
heuristics (the role `HDH` is built to fill).

Quantum compilation frameworks like Qiskit [@Qiskit], Cirq [@Cirq], and 
PennyLane [@PennyLane] provide circuit optimization and device mapping, but they do 
not offer model-agnostic abstractions for distributed quantum computing. 
The `HDH` library is compatible with these SDKs, making it a seamless addition 
to state-of-the-art quantum software stacks rather than a replacement for them.

The `HDH` library aims to tackle both limitations above, providing a unified and accessible starting point for the research of DQC partitioners.

# Statement of need

`HDH` is a Python package designed for researchers to test and develop partitioning strategies 
for quantum workloads. 

HDHs (Hybrid Dependency Hypergraphs) are an abstraction which transforms quantum computation, originating from any quantum computational model (including circuits, measurement-based quantum computing, quantum cellular automata, and quantum walks), to a directed hypergraph that expresses all possible partitions available within the computation.
They were originally proposed in [@Gragera:2025] as a unifying approach to quantum distribution, extending the hypergraph abstraction method for partitioning across devices originally proposed in [@Andres:2019].
Various partitioning strategies have 
since been proposed building on that earlier abstraction [@Clark:2023; @Escofet:2023; @Sundaram:2023], but many are tested on inconsistent hypergraph abstractions, hindering cross-partitioner comparison and improvement.

Having an easy to implement, open-source, and model-agnostic abstraction will enable the fair and consistent cross-comparison of partitioning strategies in future work. 
Furthermore, HDHs extend this capability beyond the circuit model, addressing a current blind spot in DQC research. 

`HDH` is designed to be used by both distributed quantum architecture researchers 
and compiler developers building on existing frameworks who require a model-agnostic distribution layer.

## Software Design 

The central design decision in `HDH` was to separate the abstraction layer from 
the partitioning layer. Rather than building a monolithic tool that both constructs 
hypergraphs and partitions them, `HDH` exposes the HDH as a first-class data 
structure that any downstream partitioner can consume. This makes the library 
useful both as a standalone research tool and as a substrate for third-party heuristics.

A hypergraph-based representation was chosen deliberately over simpler graph 
alternatives, as quantum computing models frequently involve operations with more 
than two inputs or outputs (a Toffoli gate, for instance, acts on three qubits 
simultaneously), requiring multi-way correlations.

Two further design choices trade added complexity for partitioning flexibility. 
First, HDHs represent each qubit's state at every timestep as a separate node 
rather than a single node per logical qubit, and partitioning operates over 
these timestepped nodes rather than whole qubits. This allows a single qubit's 
history to be split across devices at whichever point in the computation makes 
the split cheapest, with its state teleported between them at that boundary, 
rather than committing the qubit to one device for the circuit's full duration. 
Second, a multi-qubit gate is represented not as one hyperedge but three: one 
linking each qubit's pre-gate state to an intermediate state, one spanning all 
involved qubits at that intermediate point, and one linking each qubit's 
post-gate state onward. This staging is what allows a partitioner to cut 
through a multi-qubit gate at either boundary — modelling either a non-local 
gate executed over the network or a teleportation of one qubit's state 
immediately before or after the gate — instead of being limited to the 
coarser choice of which side of a single gate-edge to place a device boundary 
on. The trade-off is a wider hypergraph in exchange for exposing substantially 
more of the partitioning search space than the single-edge-per-gate 
abstractions used by prior approaches.

The library includes a capacity-aware greedy heuristic as a built-in baseline. 
Existing DQC research typically benchmarks against KaHyPar [@schlag2023high], a 
general-purpose hypergraph partitioner not designed for quantum hardware constraints. 
While simpler than KaHyPar, the included heuristic enforces per-device qubit 
capacity, making it a more 
appropriate DQC baseline and a concrete starting point for researchers developing 
improved strategies.

Finally, `HDH` is implemented in Python to minimise the barrier to adoption. 
The quantum software community has converged on Python as its primary language, 
and compatibility with Qiskit, Cirq, and PennyLane was prioritised from the 
outset to ensure `HDH` integrates naturally into existing workflows.

## Model conversions

Any quantum computing model comprises a series of commands which establish qubit state 
rotations, measurements and entanglements. For instance, quantum circuits are 
comprised of a sequence of quantum gates applied to qubits. Single-qubit gates 
perform rotations on the Bloch sphere, while multi-qubit gates (such as CNOT) 
create entanglement dependencies between qubits.

HDHs use the following notation to describe quantum workload dependencies, 
including predicted elements that represent potential future state 
transformations based on classical measurement outcomes:

![HDH symbol legend.\label{fig:hdh_legend}](docs/img/HDHobjects.png){ width=35% }

Mapping a quantum workload such as a circuit to an HDH involves applying specific correspondences between model elements and hypergraph motifs. This library provides model-specific classes such as the `Circuit` class that enable straightforward conversions to HDHs using mapping tables:

![Circuit to HDH mapping table.\label{fig:circuit_mappings}](docs/img/circuitmappings.png){ width=35% }

In the context of DQC, entangling operations in a model can be made non-local (namely non-local gates) and thus partitioned through 
a quantum network via quantum communication primitives [@Wu:2022]. Alternatively, 
qubit states can be individually forwarded through teleportation protocols 
[@Moghadam:2017]. Because HDHs represent both cut types within a single structure, a partitioner is free to combine them within one workload — e.g. keeping a qubit local to a device via non-local gates during one phase of a computation, then teleporting it elsewhere for a later phase — rather than being restricted to whichever single strategy the chosen abstraction happens to support.

The table below shows how HDHs supersede previous abstractions in their 
expressivity of these partitioning options. Unlike prior approaches that 
represent only non-local gates or only teleportation, HDHs capture both 
strategies simultaneously, enabling partitioners to optimize across all 
available distribution methods:

![Table showing HDH expressivity.\label{fig:comparison_table}](docs/img/comparison_table.png){ width=50% }

Usage examples for these conversions are available in the [documentation](https://grageragarces.github.io/HDH/). As an example, the figure below shows the HDH produced from a six-qubit circuit combining a three-qubit Toffoli gate, single-qubit gates, entangling two-qubit gates, a classically-conditioned gate, and mid-circuit measurements, shown as a graph representation of a hypergraph since visualizing large, multi-colored hypergraphs directly becomes impractical at scale. Gates have hyperedges corresponding to the qubit state transformations they generate, as well as preceding and following hyperedges that capture pre- and post-teleportation of the involved states. HDHs differ from previous abstractions in two key ways: 

(1) nodes represent possible state transformations rather than individual qubits or operations, and 

(2) classical data flows are explicitly included (shown in orange):

![Example circuit and its HDH representation.\label{fig:circuit_example}](docs/img/hdhfromcircuit.svg){ width=80% }

## Research Impact Statement

HDH is the software implementation of a theoretical framework currently under peer review, as the library and its formal foundations were developed in tandem. 
In that manuscript, the capacity-aware greedy heuristic included in this library is evaluated against exhaustive search over feasible capacity-respecting assignments, using real quantum circuits from the MQT Bench suite [@MQTBench]. Across instances up to 10 qubits (beyond which exhaustive enumeration becomes intractable, exceeding $10^{70}$ candidate assignments), the heuristic achieves a mean cost ratio of 0.978 relative to the exhaustive-search optimum (median 1.0, with 30% of instances matching it exactly), while scaling to circuits far beyond exhaustive search's reach: an average of 70.65 seconds for 100-qubit circuits. This is direct, specific evidence that `HDH`'s built-in partitioning heuristic achieves communication costs close to optimal while remaining practical at problem sizes where exhaustive comparison is no longer possible. The manuscript is available from the author upon request for verification.

We separately evaluated whether combining teledata and telegate cuts, rather than committing to either alone, actually reduces communication cost, using a reproducible comparison script (`benchmarks/`) that restricts a common partitioner to each cut type in turn. Across real MQT Bench circuits, the combined and telegate-only (qubit-level) formulations tie in every instance tested: standard benchmark algorithms do not exhibit the specific structure this expressivity exploits. That structure is a qubit whose interaction pattern shifts partway through the computation — repeatedly interacting with one group of qubits, then a different group later. We constructed such circuits and, using exhaustive search, confirmed a substantial, provable reduction over the qubit-level formulation used by prior hypergraph approaches (e.g. a $3\times$ reduction in cut cost for 4 interaction-pattern switches). This is honest, specific evidence for a targeted claim: the benefit of combining cut types is real but structural, occurring when a workload's dependency pattern changes over time, rather than a general-purpose improvement over every workload.

Early community engagement has been encouraging. The project was presented as a poster at SIGCOMM 2025 [@Gragera:2025] (a major networking venue), has received funding through the Unitary Fund microgrant program (dedicated to supporting open source quantum software to benefit humanity) and has already seen external contributors (acknowledged bellow).
Further, we're in discussion with companies in the Distributed Quantum Computing space regarding the library's integration within their stack.

The library is designed for immediate use by two audiences: 
distributed quantum architecture researchers who need a consistent abstraction to evaluate their partitioning strategies, and 
compiler developers building on existing framework who require a model-agnostic distribution layer. 
The open-source release and documentation are intended to lower the barrier to reproducible research in DQC partitioning.

# Acknowledgements

We acknowledge contributions from [Joseph Tedds](https://github.com/josephtedds), [Manuel Alejandro](https://github.com/manalejandro), and [Alessandro Cosentino](https://github.com/cosenal).

We thank Unitary Fund for supporting this project through their quantum microgrant program.

The work of the author is supported by the EPSRC UK Quantum Technologies Programme under grant EP/T001062/1 and VeriQloud.

# AI Usage Disclosure
Claude was used during both library development and paper writing. 

In the library, Claude generated initial draft code implementations that were subsequently rewritten by the author. 
It also assisted in producing unit tests, which were validated against expected behaviour across both passing and failing scenarios. These were not always fully re-written but they were revised and thorougly tested.
Additionally, AI was used to generate inline code comments throughout the library, with the aim of improving readability for contributors and users who may check the source code.

In the paper, Claude was used to assist with wording and polish. 

All AI-generated content (both code and text) was reviewed or modified (as per the above descriptions) 
by the author before inclusion in the present version.

# References
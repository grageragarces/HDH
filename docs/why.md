# Why HDHs?

Hybrid Dependency Hypergraphs provide a model-agnostic abstraction for quantum computation.Their purpose is to enable distributed quantum computing, meaning the collaboration of various quantum computers to complete a task greater than their individual capacities.

Different quantum computers work with different computational models due to physical constraints or architectural choices.For example, photonic systems do not naturally support the circuit model because photons interact weakly, while superconducting devices commonly use circuits but can support cellular automata or quantum walks when those models map better to the task.

HDHs encode not only these models, but also the hybrid nature of quantum processing, where classical control and classical-quantum feedback loops are intrinsic.This also enables the representation of hybrid algorithms, which currently deliver the most practical gains.
HDHs sit between high-level quantum programming languages and machine-level instructions.Because they are based on quantum models, they provide a familiar representation for developers and accurately capture workload structure and weight.

These properties make the abstraction powerful, as it enables the creation of a dependency pattern from any quantum workload.These patterns can then be partitioned (or cut) and distributed to the different devices collaborating to complete the task.

The notion of partitioning computation at a hypergraph level predates quantum computing and has been explored in HPC, without much success.The difference is that quantum computation uniquely exposes explicit dependency structures in its models.This enables well-defined and meaningful partitioning, contrary to classical settings where dependencies can be implicit and costly to infer.The idea of using hypergraph partitioning for distribution has circulated in the quantum community since at least 2018.HDHs aim to serve as a baseline abstraction over which partitioning techniques can be evaluated fairly.

State-of-the-art hypergraph partitioning techniques used in this context include KaHyPar and Fiduccia–Mattheyses.However, we do not yet have a good understanding of how well these and other partitioners compare across the recurring dependency structures that arise in quantum computing.Current efforts rely on ad-hoc, independently constructed representations, which makes cross-method evaluation difficult.HDHs aim to make the construction of these patterns simple, transparent, and fast, enabling systematic comparison of existing partitioners and supporting the development of specialised techniques for distributed quantum workloads.


It is important to note that the goal of HDHs is not to establish or model the physical quantum communication layer, nor to define classical channels between devices when no quantum connection exists.Those aspects are valuable for full-stack evaluation, and future versions will include quantum communication primitives, but HDHs focus first on the abstraction of computation, not the transport layer.
Similarly, HDHs do not prescribe partitioning strategies. Their role is to provide a fair, consistent substrate on which partitioning techniques can be applied and compared.


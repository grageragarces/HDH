
# Welcome to the HDH library

HDH refers to *Hybrid Dependency Hypergraph* an abstraction developped to enable the partitioning of quantum computations in the context of Distributed Quantum Computing.
HDHs are a directed hypergraph based abstraction that is:

* **Workload agnostic**: HDHs can cover all computational models and instruction sets available accross the quantum stack,
* **Make every cutting option available**: encoding both spatial and temporal dependencies HDHs superseed both telegate and teledata abstractions, enabling partitioners to choose from every possible cut combination,
* **Hybrid**: as the name suggests HDHs represent both quantum and classical data dependencies, enabling hybrid computations and classical fowarding considerations under cut communication requirements,
* **Easy and fast to build**: the mapping algorithm from a workload to a HDH grows linearly (the specific speed may depend on the computational model mapped but overall the process is fast).


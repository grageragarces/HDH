# hdh

![HDH Logo](https://raw.githubusercontent.com/grageragarces/hdh/main/docs/img/logo.png)

**Hybrid Dependency Hypergraphs for Quantum Computation**  
[![PyPI version](https://badge.fury.io/py/hdh.svg)](https://pypi.org/project/hdh/)  · MIT Licensed · Author: Maria Gragera Garces  

**Work in Progress — Preparing for 1.0**

---

## What is HDH?

**HDH (Hybrid Dependency Hypergraph)** is an intermediate representation designed to describe quantum computations in a model-agnostic way.
It provides a unified structure that makes it easier to:

- Translate quantum programs (e.g., from Qiskit or QASM) into a common hypergraph format
- Analyze and visualize the logical and temporal dependencies within a computation
- Partition workloads across devices using tools like METIS or KaHyPar, taking into account hardware and network constraints

---

## Current Capabilities

- Qiskit circuit translation  
- OpenQASM 2.0 file parsing  
- Graph-based printing and canonical formatting  
- Partitioning with METIS using custom HDH-to-graph translation  
- Model-specific abstractions for:
  - Quantum Circuits
  - Measurement-Based Quantum Computing (MBQC)
  - Quantum Walks
  - Quantum Cellular Automata (QCA)
- Analysis tools for:
  - Cut cost estimation across partitions
  - Partition size reporting
  - Parallelism tracking by time step
  - Integration with `networkx` and `metis`

Includes test examples for:

- Circuit translation (`test_convert_from_qiskit.py`)
- QASM import (`test_convert_from_qasm.py`)
- MBQC (`mbqc_test.py`)
- Quantum Walks (`qw_test.py`)
- Quantum Cellular Automata (`qca_test.py`)
- Protocol demos (`teleportation_protocol_logo.py`)

---

## Installation

```bash
pip install hdh
```

---

## Quickstart

### From Qiskit

```python
from qiskit import QuantumCircuit
from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.visualize import plot_hdh

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

hdh = from_qiskit(qc)

plot_hdh(hdh)
```

### From QASM file

```python
from hdh.converters.convert_from_qasm import from_qasm
from hdh.visualize import plot_hdh

qasm_path = os.path.join(os.path.dirname(__file__), 'test_qasm_file.qasm')
hdh = from_qasm('file', qasm_path)

plot_hdh(hdh)
```

### Partitioning

```python
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time

num_parts = 3
partitions = compute_cut(hdh, num_parts)

print(f"\nMETIS partition into {num_parts} parts:")
for i, part in enumerate(partitions):
    print(f"Partition {i}: {sorted(part)}")
    
# plot_hdh(hdh)
cut_cost = cost(hdh, partitions)
sizes = partition_sizes(partitions)
global_parallelism = compute_parallelism_by_time(hdh, partitions, mode="global")
parallelism_at_t3 = compute_parallelism_by_time(hdh, partitions, mode="local", time_step=3)

print("\n--- QW Metrics ---")
print(f"\nCut cost: {cut_cost}")
print(f"Partition sizes: {sizes}")
print(f"Parallelism over time: {global_parallelism}")
print(f"Parallelism at time t=3: {parallelism_at_t3}")

```

---

## Example Use Cases

- Visualize quantum protocols (e.g., teleportation)  
- Analyze dependencies in quantum walk evolutions  
- Explore entanglement flow in MBQC patterns  
- Partition large circuits across heterogeneous QPUs  

---

## Coming Soon

- Compatibility with Cirq, Braket, and Pennylane  
- Full graphical UI for HDH visualization  
- Native noise-aware binning strategies  
- Better cut handling for distributed execution  

---

## Tests and Demos

All tests are under `tests/` and can be run with:

```bash
pytest
```

If you're interested in the HDH of a specific model, see in manual_tests:

- `mbqc_test.py` for MBQC circuits  
- `qca_test.py` for Cellular Automata  
- `qw_test.py` for Quantum Walks  
- `teleportation_protocol_logo.py` for a protocol-specific demo  

---

## Contributing

Pull requests welcome. Please open an issue or get in touch if you're interested in:

- SDK compatibility  
- Optimization strategies  
- Frontend tools (visualization, benchmarking)  

---

## Citation

More formal citation and paper preprint coming soon. Stay tuned for updates.

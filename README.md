# hdh

![HDH Logo](https://raw.githubusercontent.com/grageragarces/hdh/main/docs/img/logo.png)

**Hybrid Dependency Hypergraphs for Quantum Computation**  
[![PyPI version](https://badge.fury.io/py/hdh.svg)](https://pypi.org/project/hdh/)  · MIT Licensed · Version `0.0.2` · Author: Maria Gragera Garces  

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
from hdh.frontend.qiskit_loader import convert_qiskit_circuit_to_hdh

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

hdh = convert_qiskit_circuit_to_hdh(qc)
hdh.print()
```

### From QASM file

```python
from hdh.frontend.qasm_loader import convert_qasm_file_to_hdh

hdh = convert_qasm_file_to_hdh("test_qasm_file.qasm")
hdh.print()
```

### Partitioning

```python
from hdh.partitioning.metis_partition import compute_metis_partition

partition = compute_metis_partition(hdh, num_bins=2)
print(partition)
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

If you're interested in the HDH of a specific model, see:

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


![HDH Logo](https://raw.githubusercontent.com/grageragarces/hdh/miscellaneous/img/logo.png)

# Hybrid Dependency Hypergraphs for Quantum Computation

<p style="text-align:center">
  <a href="https://pypi.org/project/hdh/">
    <img src="https://badge.fury.io/py/hdh.svg" alt="PyPI version">
  </a>
  · MIT Licensed ·
  <a href="https://unitary.foundation">
    <img src="https://img.shields.io/badge/Supported%20By-UNITARY%20FOUNDATION-brightgreen.svg?style=for-the-badge" alt="Unitary Foundation">
  </a>
  · Author: Maria Gragera Garces
  <br><br>
  <em>Work in Progress - Preparing for 1.0</em>
</p>

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://your-docs-url)

Documentation can be found at: https://grageragarces.github.io/HDH/

---

## What is HDH?

**HDH (Hybrid Dependency Hypergraph)** is an intermediate directed hypergraph-based representation designed to encode the dependecies arising in any quantum workload.
It provides a unified structure that makes it easier to:

- Translate quantum programs (e.g., a circuit or a mbqc pattern) into a unified hypergraph format
- Analyze and visualize the logical and temporal dependencies within a computation
<!-- - Partition workloads across devices using tools like METIS or KaHyPar, taking into account hardware and network constraints -->

---

## Current Capabilities

- Qiskit, Braket, Cirq and Pennylane circuit mappings to HDHs
- OpenQASM 2.0 file parsing  
- Model-specific abstractions for:
  - Quantum Circuits
  - Measurement-Based Quantum Computing (MBQC)
  - Quantum Walks
  - Quantum Cellular Automata (QCA)

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
---

<!-- ## Tests and Demos

All tests are under `tests/` and can be run with:

```bash
pytest
```

If you're interested in the HDH of a specific model, see in manual_tests:

- `mbqc_test.py` for MBQC circuits  
- `qca_test.py` for Cellular Automata  
- `qw_test.py` for Quantum Walks  
- `teleportation_protocol_logo.py` for a protocol-specific demo  

--- -->

## Contributing

Pull requests welcome. Please open an issue or get in touch if you're interested in:

- SDK compatibility  
- Frontend tools (visualization, benchmarking) 

or if you've found a bug! 

---

## Citation

More formal citation and paper preprint coming soon. Stay tuned for updates.


![HDH Logo](https://raw.githubusercontent.com/grageragarces/hdh/main/miscellaneous/img/logo.png)

# Hybrid Dependency Hypergraphs for Quantum Computation

<p style="text-align:center">
  <a href="https://pypi.org/project/hdh/">
    <img src="https://badge.fury.io/py/hdh.svg" alt="PyPI version">
  </a>
  · 
  <a href="https://grageragarces.github.io/HDH/">
    <img src="https://img.shields.io/badge/docs-online-blue" alt="Docs">
  </a>
  · 
  <a href="https://unitary.foundation">
    <img src="https://img.shields.io/badge/Supported%20By-UNITARY%20FOUNDATION-brightgreen.svg?style=for-the-badge" alt="Unitary Foundation">
  </a>
  · MIT Licensed
  <br><br>
</p>

<!-- Documentation can be found at: https://grageragarces.github.io/HDH/ -->

---

## What is a HDH?

**HDH (Hybrid Dependency Hypergraph)** is an intermediate directed hypergraph-based representation designed to encode the dependencies arising in any quantum workload.
It provides a unified structure that makes it easier to:

- Translate quantum programs (e.g., a circuit or a mbqc pattern) into a unified hypergraph format
- Analyze and visualize the logical and temporal dependencies within a computation
- Partition workloads across devices, taking into account hardware and network constraints

---

## Current Capabilities

- Qiskit, Braket, Cirq and Pennylane circuit mappings to HDHs
- OpenQASM 2.0 file parsing  
- Model-specific abstractions for:
  - Quantum Circuits
  - Measurement-Based Quantum Computing (MBQC)
  - Quantum Walks
  - Quantum Cellular Automata (QCA)
- Capability to partition HDHs and evaluate partitions

---

## Installation

```bash
pip install hdh
```

Qiskit conversion works out of the box. Cirq, PennyLane, Amazon Braket, and
the KaHyPar/METIS partitioners are optional and installed via extras:

```bash
pip install hdh[cirq]        # Cirq conversion (needs Python >=3.11)
pip install hdh[pennylane]   # PennyLane conversion (needs Python >=3.11)
pip install hdh[braket]      # Amazon Braket conversion (needs Python >=3.11)
pip install hdh[kahypar]     # KaHyPar-based partitioning
pip install hdh[metis]       # METIS-based partitioning (metis_telegate)
pip install hdh[all]         # everything above (needs Python >=3.11)
```

Tested against Cirq 1.7, PennyLane 0.45, and amazon-braket-sdk 1.125 — all
three now require Python >=3.11 upstream, so those extras aren't installable
on Python 3.10.
`hdh[metis]` installs the Python binding only — it talks to a
system-installed METIS C library via ctypes, so METIS itself must already be
available on your machine (e.g. via your OS package manager or built from
source). Without it, `metis_telegate` automatically falls back to a
Kernighan-Lin partition and reports which method it used.

---
## Quickstart

### From Qiskit

```python
from qiskit import QuantumCircuit
from hdh.converters import from_qiskit
from hdh.visualize import plot_hdh

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

hdh = from_qiskit(qc)

plot_hdh(hdh)
```

### From QASM file

```python
from hdh.converters import from_qasm
from hdh.visualize import plot_hdh

qasm_path = os.path.join(os.path.dirname(__file__), 'test_qasm_file.qasm')
hdh = from_qasm('file', qasm_path)

plot_hdh(hdh)
```
---

## Tests and Demos

All tests are under `tests/` and can be run with:

```bash
pytest
```

---

## Contributing

Pull requests welcome. Please open an issue or get in touch if you're interested in:

- SDK compatibility  
- Frontend tools (visualization, benchmarking) 

or if you've found a bug! 

---

## Citation

More formal citation and paper preprint coming soon. Stay tuned for updates.

# HDH Graphs

This directory contains HDH (Hybrid Dependency Hypergraph) representations of workloads.

## Structure

```
HDHs/
└── <Model>/
    └── <Origin>/
        ├── pkl/              # Pickled HDH objects
        │   └── *.pkl
        └── text/             # CSV exports
            ├── *__nodes.csv
            ├── *__edges.csv
            └── *__edge_members.csv
```

## Converting Workloads to HDHs

Use the HDH library's conversion functions:

```python
from hdh.converters import from_qasm

hdh_graph = from_qasm('file', 'path/to/circuit.qasm')
```

See the [HDH documentation](https://grageragarces.github.io/HDH/) for details.

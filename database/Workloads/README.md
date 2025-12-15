# Workloads

This directory contains the original quantum workload files.

## Structure

```
Workloads/
└── <Model>/          # e.g., Circuits, MBQC, QW, QCA
    └── <Origin>/     # e.g., MQTBench, Custom
        └── *.qasm    # Workload files
```

## Adding Workloads

1. Create directory: `Workloads/<Model>/<Origin>/`
2. Add your workload files (QASM, etc.)
3. Convert to HDHs (see HDHs README)
4. Run partitioning methods
5. Add results using `scripts/add_method_results.py`

# The HDH Database

To support reproducible evaluation and training of partitioning strategies, this library includes a database of pre-generated HDHs.  
We aim for this resource to facilitate benchmarking across diverse workloads and enables the development of learning-based distribution agents.

***Our goal is to extend the database with performance metrics of partitioning techniques for each workload.
This will allow the community to build a data-driven understanding of which hypergraph partitioning methods perform best under different conditions.
We encourage users to contribute results from their own partitioning methods when working with this database.
Instructions for how to upload results can be found below.***

## Database Location and Structure

The database is available in the `main` branch [of the repository](https://github.com/grageragarces/HDH). 

**Important**: The database exists only in the repository and is **not included in the pip package or wheels**. Users who want to use the database for benchmarking should clone the repository from GitHub. The database files are excluded from the PyPI distribution to keep the package size small.

The database is organized into two main directories:

* **`Database/`**: Contains the actual database files (HDHs and workloads)
* **`Database_generator/`**: Contains scripts, converters, and utilities for generating and extending the database

---

## Database Directory Structure

```
Database/
├── Workloads/          # Raw workload commands
│   └── <Model>/
│       └── <Origin>/
└── HDHs/               # Corresponding Hybrid Dependency Hypergraphs
    └── <Model>/
        └── <Origin>/
            ├── pkl/            # Pickled HDH objects
            ├── text/           # Human-readable CSV files
            ├── images/         # (reserved for future visualizations)
└── Partitions/     # Partitioning method results
            ├── leaderboards/    # Pre-computed comparison view
            ├── methods/         # Method configurations and documentation
            ├── results/         # Main results database (`partitions.csv` - .gz compressed)
            ├── scripts          # Helper scripts for database operations
```


where:

* **Model** = computational model (e.g., Circuit, MBQC, QW, QCA)  
* **Origin** = source of the workload (e.g., benchmark suite, custom circuit)

---

## The Mapping Problem %HERE

Partitioning an HDH is fundamentally a **mapping problem**.  

We want to assign ** qubits** (and therefore all corresponding HDH **state nodes across time**) onto a fixed number of QPUs, connected by some underlying network topology.  
The goal is to distribute the workload while minimizing inter-QPU communication.

More formally:

* **Inputs**:  
  * an HDH workload with time-labelled state nodes and dependency hyperedges  
  * a QPU network graph, where nodes have finite **qubit capacities** and edges represent physical connectivity  

* **Output**:  
  * an assignment of qubits into `k` bins (QPUs), defining where computation is executed  

A valid mapping typically needs to satisfy:

* **Capacity constraints**: each bin can host at most `capacity` unique qubits  
* **Communication minimization**: hyperedges spanning multiple bins induce cuts (and thus communication)  
* **Temporal structure preservation**: assignments should respect the time-ordered dependency structure of the HDH  

---

### When capacity constraints cannot be guaranteed

Some partitioning techniques cannot ensure that strict QPU capacities are always respected.  

For example, general-purpose hypergraph partitioners (such as **KaHyPar** or **hMETIS**) are designed to produce *balanced* partitions, but they cannot guarantee that a partition will remain below an exact hardware qubit limit in all cases.  

In these settings, the partition result should be treated as *not capacity-safe*, and we record:

* `respects_capacity = False`

This directly affects downstream processing:  

when a partition does not reliably satisfy capacity constraints, we may disable automatic completion of missing assignments, meaning:

* `respects_capacity = False`

so that the database reflects the *raw mapping outcome* rather than an artificially forced valid embedding.

---

### Communication cost model 

Cut hyperedges induce inter-QPU communication, and different communication types carry different costs.  

In the current implementation we use:

* **quantum communication cost = 10**  
* **classical communication cost = 1**

These weights define the objective optimized by our greedy cut heuristic and are configurable depending on the target hardware assumptions.


---


## File Formats

### Workloads Directory
* QASM files representing the quantum workloads

### HDHs Directory
* **`.pkl`**: Python-pickled `HDH` objects
* **`.csv`**: Human-readable HDH structure:
  * `__nodes.csv`: node_id, type, time, realisation  
  * `__edges.csv`: edge_index, type, metadata  
  * `__edge_members.csv`: edge-node incidence list  

---

## Partitions Directory

* **`partitions.csv.gz`**: Main partitioning performance database

---

## CSV Schema (Matches run_partition_tests.py)

The partition results file is written directly by `run_partition_tests.py`.

The column schema is fixed by the internal `fieldnames = [...]` list:

#### Mandatory Columns

Each row represents:

**(workload, k, capacity, method, config)**

The database columns are:

* **`workload_file`**
* **`model`**
* **`origin`**
* **`n_qubits`**
* **`k_partitions`**
* **`capacity`**

* **`method_name`**
* **`method_version`**
* **`config_hash`**

* **`bins`** (JSON list of partitions)
* **`cost`**
* **`respects_capacity`**
* **`method_metadata`**

* **`time_seconds`**
* **`memory_mb`**

* **`date_run`**
* **`library_version`**
* **`contributor`**
* **`notes`**

These match exactly the schema produced by:

```python
fieldnames = [
    'workload_file', 'model', 'origin', 'n_qubits', 'k_partitions', 'capacity',
    'method_name', 'method_version', 'config_hash',
    'bins', 'cost', 'respects_capacity', 'method_metadata',
    'time_seconds', 'memory_mb',
    'date_run', 'library_version', 'contributor', 'notes'
]
```
### Example CSV

```csv
workload_file,model,origin,n_qubits,k_partitions,capacity,method_name,method_version,config_hash,bins,cost,respects_capacity,method_metadata,time_seconds,memory_mb,date_run,library_version,contributor,notes
qft_8.qasm,Circuits,MQTBench,8,2,4,metis_telegate,2026-02-02,default,"[["q0","q1","q2","q3"],["q4","q5","q6","q7"]]",12,true,"{"metis_method":"metis","failed":false}",0.031200,28.40,2026-02-02,0.0.0,alice_researcher,
qft_8.qasm,Circuits,MQTBench,8,2,4,greedy_hdh,2026-02-02,9f12a3c4,"[["q0_t0","q1_t0"],["q2_t0","q3_t0"]]",18,true,"{"beam_k":3,"reserve_frac":0.08,"restarts":1,"seed":0}",0.084500,31.10,2026-02-02,0.0.0,alice_researcher,"example row; node-level bins"
```

Notes:
* `config_hash` is produced automatically from the JSON passed via `--config`.  
* Some methods output **qubit-level bins** (e.g., `metis_telegate`), while others output **node-level bins** (e.g., `greedy_hdh`).  
  The stored `bins` column reflects whatever the method returns; use `method_metadata` to record interpretation if needed.
### Standard Partitioning Methods

Here are the standard partitioning methods currently supported by `run_partition_tests.py` (and therefore represented in the results database).

#### Cut (HDH temporal greedy partitioning) — `compute_cut`

This is the **greedy heuristic proposed in the paper**.  
Implementation: `hdh.passes.cut.compute_cut` (called by `run_partition_tests.py` as `run_greedy_hdh`).

Key properties:

* **Works directly on the HDH hypergraph** (uses HDH nodes and HDH hyperedges)
* **Partitions at node level** (bins contain node IDs like `q0_t1`, `q3_t7`, etc.)
* **Capacity is enforced on unique qubits per bin**, not on node count  
  (each node `q<i>_t<j>` contributes to qubit `i`)
* **Allows teledata-style cuts**  
  (different time-nodes of the same qubit may appear in different bins, which corresponds to teleporting state across QPUs)

Algorithm sketch (as implemented):

1. **Temporal incidence build**: construct an incidence structure from HDH hyperedges that preserves time adjacency.  
2. **Greedy fill for each bin** (up to `k` bins):
   * pick the earliest-time unassigned node as a **seed**  
   * expand a **frontier** of temporally adjacent unassigned neighbors (min-heap by time)  
   * repeatedly select the best next node from the frontier using **delta cut-cost** evaluation  
     (among the top `beam_k` frontier candidates)
   * if adding a node would introduce a new qubit and exceed `cap`, **reject** it for this bin and try the next candidate
3. **Residual placement**: for any remaining nodes, attempt a best-fit placement using delta cost across bins.  
   Nodes that cannot be placed without violating capacity are treated as **unplaceable** (this is where `respects_capacity` may become false).

**Cost**: the returned `cost` is the number of HDH hyperedges that span more than one bin (unweighted cut count).  
Recommended metadata: store parameters like `beam_k`, and (if applicable) a `respects_capacity` flag in `method_metadata`.

#### **METIS (telegate graph)**

Builds a **qubit interaction graph** (nodes = qubits; edges represent cross-qubit interaction pressure derived from two-qubit gates).  
It then runs **METIS** via `nxmetis` when available, and falls back to a **Kernighan–Lin** style bisection otherwise.

**Capacity**: enforced on *qubits per bin*, with a post-check stored as `respects_capacity`.

**Cost**: the returned `cost` is the **unweighted cut size** of the telegate graph (number of graph edges crossing partitions).  
`method_metadata["metis_method"]` records whether `metis` or the fallback was used.

#### **KaHyPar (hypergraph baseline)**

Runs a KaHyPar-based hypergraph partitioner (when available) on a qubit-level abstraction.

**Capacity**: enforced on *qubits per bin* (post-check stored as `respects_capacity=false` if violated).

**Cost**: whatever objective value the KaHyPar configuration reports.  
Record the solver configuration path and important options in `method_metadata` for reproducibility.

### Adding Your Results


#### Step 1: Update `Partitions/results/partitions.csv`

The main results file is:

```
Database/Partitions/results/partitions.csv
```

The recommended way to populate it is to use `Database_generator/run_partition_tests.py`, which will:
* discover HDHs under `Database/HDHs/<Model>/<Origin>/pkl/`
* run one or more methods for each `k`
* compute a `config_hash` automatically
* append (or overwrite) matching rows in the CSV

If you prefer to add a row manually, treat the CSV as **append-only long-format** (one row per method run).  
At minimum, include the mandatory columns listed above, and serialize `bins` and `method_metadata` as JSON strings.


#### Step 2: Document Your Method 

Create or update `Database/Partitions/README.md`:

```markdown
# Partitioning Methods for <Origin>

## Your Method Name

**Contributor**: your_github_username
**Date**: YYYY-MM-DD

### Description
Brief description of your partitioning algorithm...

### Parameters
- Parameter 1: description
- Parameter 2: description

### Implementation Details
Input your code or/and detailed explanation...

### Performance Characteristics
- Time complexity: O(...)
- Space complexity: O(...)
- Works best for: ...
```

#### Step 3: Capacity Constraints and Failure States

`capacity` is stored **explicitly per row** and is defined as the maximum number of *qubits* allowed in each bin.

In `run_partition_tests.py`, the default capacity is:

* `capacity = ceil(n_qubits / k_partitions)`, optionally scaled by an `overhead` factor.

Each row records whether the returned partition respects the constraint via:

* **`respects_capacity`**: `true/false`

If your method can fail in other ways (timeout, solver crash, infeasible config), record that in:

* **`notes`** (human-readable), and/or  
* **`method_metadata`** (structured JSON)

This keeps the core schema stable while still preserving failure details for later analysis.


#### Step 4: Generate leaderboards / “best” views

The raw results file is long-format and does **not** store a `best` column.  
Instead, “best method” views should be computed by aggregating `Partitions/results/partitions.csv`, typically grouping by:

* `(workload_file, model, origin, k_partitions, capacity)`  

and selecting the minimum `cost` among rows with `respects_capacity=true`.

If your repository includes `generate_leaderboards.py`, use that script to regenerate the pre-computed leaderboard views under:

```
Database/Partitions/leaderboards/
```

(If you implement your own leaderboard logic, keep the rules explicit: which `cost` definition you compare, how you handle `respects_capacity=false`, and how ties are broken.)


#### Step 5: Re-compress your csv file
Make sure to re-compress the csv file as `partitions.csv.gz` before you attempt to push (it won't otherwise due to size).
To do this, you can just run:

```
gzip partitions.csv
```

<!-- Maybe I should make a compression file to? - TODO in issues -->

#### Step 6: Submit PR 

Submit a PR to the `main` branch with:
* Updated `partitions.csv.gz`
* Updated or new `README.md` in the Partitions folder
* Your GitHub username in the `contributor` column

---

## Database Usage Guidelines

### For Benchmarking

```python
import pickle
import pandas as pd
from pathlib import Path

# Load all HDHs from an origin
origin_path = Path("Database/HDHs/Circuits/MQTBench/pkl")
hdhs = {}
for pkl_file in origin_path.glob("*.pkl"):
    with open(pkl_file, "rb") as f:
        hdhs[pkl_file.stem] = pickle.load(f)

# Load partitioning results
results_df = pd.read_csv("Database/HDHs/Circuits/MQTBench/Partitions/partitions.csv")

# Benchmark your method against existing results
for name, hdh in hdhs.items():
    your_bins, _, _, _ = your_method(hdh, k=3)
    your_cost, _ = cost(hdh, your_bins)
    
    # Compare with best existing method
    existing_best = results_df[results_df['file'] == f"{name}.qasm"]['best'].values[0]
    existing_cost = results_df[results_df['file'] == f"{name}.qasm"][f"{existing_best}_cost"].values[0]
    
    print(f"{name}: Your method: {your_cost}, Best existing: {existing_cost}")
```

### Citation and Acknowledgment

When using this database in publications, note that it was generated thanks to:
* The HDH library project
* The Munich Quantum Benchmarking Dataset (for MQTBench workloads)
* Individual contributors whose partitioning results you use

### Data License

The database is provided under the same license as the HDH library (MIT License).
Individual workloads may have their own licenses - check the origin-specific README files.

---

## Contributing

We welcome contributions! When adding to the database:

1. Use clear, descriptive commit messages
2. Document your methods thoroughly in README files 
3. Include your GitHub username as contributor
4. Verify your data loads correctly before submitting
5. Update this documentation if adding new features

For questions or discussions, please open an issue on the main repository.

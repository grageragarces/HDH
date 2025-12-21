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

### Database Directory Structure

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
            ├── configs/         # Test configurations and settings
            ├── leaderboards/    # Pre-computed comparison view
            ├── methods/         # Method configurations and documentation
            ├── results/         # Main results database (`partitions.csv` - .gz compressed)
            ├── scripts          # Helper scripts for database operations
```

where:
* **Model** = computational model (e.g., Circuit, MBQC, QW, QCA)
* **Origin** = source of the workload (e.g., benchmark suite, custom circuit, artificial circuit)

### Database_generator Directory

The `Database_generator/` folder contains:
* Utilities for batch processing (`run_partition_tests.py`)
* A file to decompress the `partitions.csv.gz` compressed database (`decompress_data.py`)
* Leadeboard generators (`generate_leaderboards.py`)
* A data querying file which can be used to "ask" specific questions (`query_results.py`). By calling things like:
```python
  # Compare two methods
  python query_results.py --database-root ./Database \\
    compare --method-a greedy_hdh --method-b metis_telegate
```
All the possible query commands can be found in the documents comments. They include finding the best method (meaning partitioner strategy) for a workload, method comparison, method statistics, etc.


The database currently contains:

* HDHs derived from the [Munich Quantum Benchmarking Dataset](https://www.cda.cit.tum.de/mqtbench/)

## File Formats

### Workloads Directory
* QASM files representing the quantum workloads

### HDHs Directory
* **`.pkl`**: Python-pickled `HDH` objects for programmatic use
* **`.txt`** / **`.csv`**: Human-readable text files with annotated metadata
  * `__edge_members.csv`: Edge-node relationships (edge_index, node_id)
  * `__edges.csv`: Edge information (edge_index, type, realisation, gate_name, role, edge_args, edge_metadata)
  * `__nodes.csv`: Node information (node_id, type, time, realisation)

### Partitions Directory
* **`partitions.csv.gz`**: Partitioning results (aka the dabase)

## Partitioning Performance Metrics

Thanks to the recent additions in PR #24, the library now provides comprehensive metrics for evaluating partitioning quality. These metrics can be computed and added to the database to build a performance baseline.

### Available Metrics (from `hdh.passes`)

#### 1. **`cost(hdh_graph, partitions)`** → `Tuple[float, float]`
Returns `(cost_q, cost_c)` - the quantum and classical cut costs:
* `cost_q`: Number of quantum hyperedges that span multiple partitions
* `cost_c`: Number of classical hyperedges that span multiple partitions

This is the **primary metric** for comparing partitioning methods.

#### 2. **`partition_size(partitions)`** → `List[int]`
Returns the size (number of nodes) of each partition.
Useful for checking balance constraints.

#### 3. **`participation(hdh_graph, partitions)`** → `Dict[str, float]`
Measures temporal participation (which partitions have activity at each timestep).
**Note**: This measures presence, not true computational parallelism.

Returns:
* `max_participation`: Peak number of active partitions
* `average_participation`: Mean active partitions per timestep
* `temporal_efficiency`: How well time is utilized
* `partition_utilization`: Average fraction of partitions active
* `timesteps`: Total timesteps
* `num_partitions`: Number of partitions

#### 4. **`parallelism(hdh_graph, partitions)`** → `Dict[str, float]`
Measures **true parallelism** by counting concurrent τ-edges (operations) per timestep.
This represents actual computational work that can execute simultaneously.

Returns:
* `max_parallelism`: Peak concurrent operations
* `average_parallelism`: Mean operations per timestep
* `total_operations`: Total operation count
* `timesteps`: Total timesteps
* `num_partitions`: Number of partitions

#### 5. **`fair_parallelism(hdh_graph, partitions, capacities)`** → `Dict[str, float]`
Implements **Jean's fairness principle** - normalizes parallelism by partition capacity to detect workload imbalances.

Returns:
* `max_fair_parallelism`: Peak fair parallelism
* `average_fair_parallelism`: Mean fair parallelism
* `fairness_ratio`: Distribution fairness (1.0 = perfectly fair)
* `total_operations`: Total operation count
* `timesteps`: Total timesteps
* `num_partitions`: Number of partitions

### Usage Example

```python
from hdh.passes import (
    cost, partition_size, 
    participation, parallelism, fair_parallelism
)

# After running your partitioning method
bins, _, _, _ = your_partitioning_method(hdh_graph, k=3)

# Evaluate the partition
cost_q, cost_c = cost(hdh_graph, bins)
sizes = partition_size(bins)
participation_metrics = participation(hdh_graph, bins)
parallelism_metrics = parallelism(hdh_graph, bins)
fair_metrics = fair_parallelism(hdh_graph, bins, capacities=[10, 10, 10])

print(f"Quantum cut cost: {cost_q}")
print(f"Classical cut cost: {cost_c}")
print(f"Partition sizes: {sizes}")
print(f"Average parallelism: {parallelism_metrics['average_parallelism']}")
print(f"Fairness ratio: {fair_metrics['fairness_ratio']}")
```

## Extending the Dataset

We encourage users to:

* Add new workloads (QASM or [other supported formats](models.md))
* Generate corresponding HDHs
* Run partitioning methods and contribute results
* Propose and document new metrics

Pull requests that expand the benchmark set or enrich metadata are very welcome!

There are two ways to contribute:

---

### 1) Add New Workloads + HDHs

#### Step 1: Place Workloads
Put your workload origin files under:  
```
Database/Workloads/<Model>/<Origin>/
```

This could be anything from a QASM file to circuit generation code.

If the HDH is not generated from functions within the library, we request you add a `README.md` to your origin folder explaining how the HDHs were generated.

Example:  
```
Database/Workloads/Circuits/MQTBench/qft_8.qasm
```

#### Step 2: Run the Converter
Convert the files (QASM strings, Qiskit circuits, etc.) to HDHs.

The converter will create:
```
Database/HDHs/<Model>/<Origin>/pkl/<filename>.pkl
Database/HDHs/<Model>/<Origin>/text/<filename>__nodes.csv
Database/HDHs/<Model>/<Origin>/text/<filename>__edges.csv
Database/HDHs/<Model>/<Origin>/text/<filename>__edge_members.csv
```

##### Converter Script (QASM → HDH → {pkl,csv})

The converter script is available in `Database_generator/` folder. Requirements: tqdm, the HDH library available on PYTHONPATH, and your QASM converter (`hdh.converters.from_qasm`).

```python
#!/usr/bin/env python3
import sys
import os
import csv
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

# Repo import path (adjust as needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters import from_qasm

BASE_DIR = Path(__file__).resolve().parent

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_pkl(hdh_graph, out_base: Path):
    p = out_base.with_suffix(".pkl")
    with open(p, "wb") as f:
        pickle.dump(hdh_graph, f)
    return p

def save_nodes_csv(hdh_graph, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "type", "time", "realisation"])
        for nid in sorted(hdh_graph.S):
            w.writerow([
                nid,
                hdh_graph.sigma.get(nid, ""),
                getattr(hdh_graph, "time_map", {}).get(nid, ""),
                hdh_graph.upsilon.get(nid, "")
            ])

def save_edges_csvs(hdh_graph, edges_path: Path, members_path: Path):
    edges_sorted = sorted(hdh_graph.C, key=lambda e: tuple(sorted(e)))
    
    # edges table
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "edge_index", "type", "realisation", "gate_name",
            "role", "edge_args", "edge_metadata"
        ])
        for idx, e in enumerate(edges_sorted):
            w.writerow([
                idx,
                hdh_graph.tau.get(e, ""),
                hdh_graph.phi.get(e, ""),
                getattr(hdh_graph, "gate_name", {}).get(e, ""),
                getattr(hdh_graph, "edge_role", {}).get(e, ""),
                json.dumps(getattr(hdh_graph, "edge_args", {}).get(e, None)) if e in getattr(hdh_graph, "edge_args", {}) else "",
                json.dumps(getattr(hdh_graph, "edge_metadata", {}).get(e, None)) if e in getattr(hdh_graph, "edge_metadata", {}) else ""
            ])
    
    # edge_members table
    with open(members_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_index", "node_id"])
        for idx, e in enumerate(edges_sorted):
            for nid in sorted(e):
                w.writerow([idx, nid])

def main():
    ap = argparse.ArgumentParser(description="Convert QASM workloads to HDH artifacts")
    ap.add_argument("--model", default="Circuits", help="Model folder")
    ap.add_argument("--origin", default="MQTBench", help="Origin folder")
    ap.add_argument("--limit", type=int, default=None, help="Max files to convert")
    ap.add_argument("--src-root", default=None, help="Override source root")
    args = ap.parse_args()

    SRC_DIR = Path(args.src_root) if args.src_root else BASE_DIR / "Database" / "Workloads" / args.model / args.origin
    DST_ROOT = BASE_DIR / "Database" / "HDHs" / args.model / args.origin
    PKL_ROOT = DST_ROOT / "pkl"
    TXT_ROOT = DST_ROOT / "text"
    IMG_ROOT = DST_ROOT / "images"

    if not SRC_DIR.exists():
        print(f"[error] Source directory not found: {SRC_DIR}")
        sys.exit(1)

    for d in (PKL_ROOT, TXT_ROOT, IMG_ROOT):
        ensure_dir(d)

    qasm_files = sorted(SRC_DIR.rglob("*.qasm"))
    if not qasm_files:
        print(f"[info] No .qasm files found under {SRC_DIR}")
        return

    if args.limit is not None:
        qasm_files = qasm_files[:args.limit]

    ok = fail = 0
    with tqdm(total=len(qasm_files), desc="Converting QASM → HDH", unit="file") as pbar:
        for qf in qasm_files:
            rel = qf.relative_to(SRC_DIR)
            stem = rel.stem
            pkl_dir = PKL_ROOT / rel.parent
            txt_dir = TXT_ROOT / rel.parent
            for d in (pkl_dir, txt_dir):
                ensure_dir(d)
            pbar.set_postfix_str(str(rel))
            try:
                hdh_graph = from_qasm("file", str(qf))
                save_pkl(hdh_graph, pkl_dir / stem)
                save_nodes_csv(hdh_graph, txt_dir / f"{stem}__nodes.csv")
                save_edges_csvs(hdh_graph, txt_dir / f"{stem}__edges.csv", txt_dir / f"{stem}__edge_members.csv")
                ok += 1
            except Exception as e:
                tqdm.write(f"[fail] {qf}: {e}")
                fail += 1
            finally:
                pbar.update(1)
    
    print(f"[done] Converted: {ok} | Failed: {fail}")

if __name__ == "__main__":
    main()
```

#### Step 3: Verify & Inspect
Please open at least one of the `text/*.csv` files and load at least one of the `pkl/*.pkl` objects in Python to verify everything works.

```python
import pickle
import pandas as pd

# Load pickled HDH
with open("Database/HDHs/Circuits/MQTBench/pkl/qft_8.pkl", "rb") as f:
    hdh = pickle.load(f)

# Load CSV files
nodes_df = pd.read_csv("Database/HDHs/Circuits/MQTBench/text/qft_8__nodes.csv")
edges_df = pd.read_csv("Database/HDHs/Circuits/MQTBench/text/qft_8__edges.csv")
members_df = pd.read_csv("Database/HDHs/Circuits/MQTBench/text/qft_8__edge_members.csv")
```

#### Step 4: Submit a PR
If all went smoothly, submit a PR with your workloads and HDHs back to the `main` branch.

**Note**: Database files might be too large to directly upload. Use [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage):

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

**Windows:**
```bash
winget install Git.GitLFS
git lfs install
```

From repo root:
```bash
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Adding <Origin> HDHs to database"
git push -u origin main
```

---

## 2) Add Partitioning Method Results

If you want to share **partitioning method results**, add them to:
```
Database/HDHs/<Model>/<Origin>/Partitions/partitions.csv.gz
```

### CSV Format for Partitioning Results

The `partitions.csv.gz` file tracks performance metrics across different partitioning methods.
The `.gz` decorator comes from compression (necessary to mantain this database on GitHub).

#### Mandatory Columns #TOUPDATE

* **`file`**: Name of the origin file
* **`n_qubits`**: Number of qubits in workload
* **`k_partitions`**: Number of partitions (e.g., 2 if cut once)
* **`<method>_bins`**: Sets of qubits per partition (JSON format)
* **`<method>_cost`**: Quantum communication cost (number of quantum hyperedges cut)
* **`best`**: Name of the method with the lowest cost

#### Optional Columns (Method-Specific) #TOUPDATE

* **`<method>_cost_q`**: Quantum cut cost (if separating q/c)
* **`<method>_cost_c`**: Classical cut cost
* **`<method>_partition_sizes`**: List of partition sizes
* **`<method>_avg_parallelism`**: Average parallelism metric
* **`<method>_fairness_ratio`**: Fairness ratio from fair_parallelism
* **`<method>_fails`**: Boolean indicating if method failed capacity constraints
* **`<method>_method`**: Sub-method used (e.g., for METIS: 'kl', 'recursive')
* **`contributor`**: GitHub username of the person who added this result

### Example CSV #TOUPDATE

```csv
file,n_qubits,k_partitions,greedy_bins,greedy_cost,metis_bins,metis_cost,metis_fails,metis_method,greedytg_bins,greedytg_cost,best,contributor
ae_indep_qiskit_10.qasm,10,2,"[[""q0"",""q1"",""q2"",""q3"",""q8""],[""q4"",""q5"",""q6"",""q7"",""q9""]]",30,"[[""q1"",""q3"",""q5"",""q6"",""q7""],[""q0"",""q2"",""q4"",""q8"",""q9""]]",25,False,kl,"[[""q0"",""q1"",""q2"",""q3"",""q9""],[""q4"",""q5"",""q6"",""q7"",""q8""]]",30,metis,alice_researcher
ae_indep_qiskit_10.qasm,10,3,"[[""q0"",""q1"",""q2"",""q8""],[""q3"",""q4"",""q6"",""q7""],[""q5"",""q9""]]",40,"[[""q3"",""q5"",""q6"",""q7""],[""q0"",""q2""],[""q1"",""q4"",""q8"",""q9""]]",32,False,kl,"[[""q0"",""q1"",""q2"",""q9""],[""q3"",""q4"",""q5"",""q6""],[""q7"",""q8""]]",38,metis,alice_researcher
```

### Standard Partitioning Methods

Here are the standard partitioning methods currently in the database:

#### **Greedy (HDH)**
Partitions directly on the HDH hypergraph where each hyperedge captures one operation's dependency set.
We fill bins sequentially: order qubits by heuristic (e.g., incident cut weight, then degree), and place each into the earliest bin that (i) respects the logical-qubit capacity and (ii) gives the smallest marginal cut increase.
If nothing fits, open the next bin up to k.

**Cost**: Sum of weights of hyperedges spanning >1 bin (default weight 1 per op; domain weights optional).

#### **METIS (Telegate graph)** #TOUPDATE: true? I don't think so this is graph based
Converts the workload into a telegate qubit-interaction graph (nodes = logical qubits; edge weights = interaction pressure indicating a non-local gate would require a "telegate" communication if cut).
Uses the [METIS library](https://pypi.org/project/metis/) to compute a k-way partition with balance constraints and minimal cut on this graph.
Partitions are then re-evaluated on the HDH cost for apples-to-apples comparison.

**Cost**: Re-evaluated on HDH hypergraph cut metric.

#### **Greedy-TG (Telegate graph)** #TOUPDATE - I don't think this is true anymore
Same fill-first policy as Greedy (HDH), but decisions are made on the telegate graph.
Nodes are qubits; edge weights reflect how costly it is to separate two qubits (expected telegate load).
Each qubit goes to the earliest feasible bin that minimizes marginal cut on the telegate graph.

**Cost**: Re-evaluated on HDH hypergraph cut metric.

### Adding Your Results

#### Step 1: Update partitions.csv #TOUPDATE

Add your results to the existing CSV file:

```python
import pandas as pd

# Load existing results
df = pd.read_csv("Database/HDHs/Circuits/MQTBench/Partitions/partitions.csv")

# Add your new column(s) if they don't exist
# Update or append your row
# Recalculate 'best' column

df.to_csv("Database/HDHs/Circuits/MQTBench/Partitions/partitions.csv", index=False)
```

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

#### Step 3: Capacity Constraints and Failure States #TOUPDATE - maybe this should be within 1? or within the readme: specific oh its not assured , maybe both?

**Capacity** = Total qubits ÷ number of partitions (rounded up)

If your partitioner cannot respect this capacity:
* Log a failure status in a `<method>_fails` column
* Set to `True` if capacity was violated
* Exclude from `best` evaluation if failed

**Important**: Document in the Partitions README whether your method:
* Always respects capacity constraints
* May violate constraints (and how failures are handled)
* Requires specific capacity settings

#### Step 4: Recalculate Best #TOUPDATE - this will basically jsut be running 'generate_leaderboards.py'

The `best` column should identify the method with the **lowest quantum cost** among methods that:
1. Did not fail capacity constraints (where `<method>_fails` is False or not present)
2. Successfully completed partitioning

```python
# Example: Recalculate best
cost_columns = [col for col in df.columns if col.endswith('_cost') and not col.endswith('_cost_c')]
methods = [col.replace('_cost', '') for col in cost_columns]

def get_best_method(row):
    valid_methods = []
    for method in methods:
        fails_col = f'{method}_fails'
        cost_col = f'{method}_cost'
        
        # Check if method failed
        if fails_col in row and row[fails_col] == True:
            continue
        
        # Check if cost is valid
        if pd.notna(row[cost_col]):
            valid_methods.append((method, row[cost_col]))
    
    if not valid_methods:
        return None
    
    # Return method with minimum cost
    return min(valid_methods, key=lambda x: x[1])[0]

df['best'] = df.apply(get_best_method, axis=1)
```

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

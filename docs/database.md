# The HDH Database

To support reproducible evaluation and training of partitioning strategies, this library includes a database of pre-generated HDHs.  
We aim for this resource to facilitate benchmarking across diverse workloads and enables the development of learning-based distribution agents.

***Our goal is to extend the database with performance metrics of partitioning techniques for each workload.
This will allow the community to build a data-driven understanding of which hypergraph partitioning methods perform best under different conditions.
We encourage users to contribute results from their own partitioning methods when working with this database.
Instructions for how to upload results can be found below.***

The database currently contains:

* HDHs derived from the [Munich Quantum Benchmarking Dataset](https://www.cda.cit.tum.de/mqtbench/).

The database is organized into two mirrored top-level folders:

* **Workloads**: contains the raw workload commands (currently QASM files).  
* **HDHs**: contains the corresponding Hybrid Dependency Hypergraphs for each workload.

Both folders share the same internal structure: ```Model/Origin_of_Workload/```.

where:

* Model = computational model (e.g., Circuit, MBQC, QW, QCA).  
* Origin_of_Workload = source of the workload (e.g., benchmark suite, custom circuit).  

## File formats

* Workloads/  
    * QASM files representing the quantum workloads.

* HDHs/  
    * `.pkl`: Python*pickled `HDH` objects for programmatic use.  
    * `.txt`: human*readable text files with annotated metadata.

## HDH text metadata

Each `.txt` file includes metadata lines before the hypergraph specification:

* *Model type*: which computational model the HDH was generated from.  
* *Workload origin*; reference to the source workload.  
* *Hybrid status*: whether the HDH contains both quantum and classical nodes.  
* *Node count*: total number of nodes in the hypergraph.  
* *Connectivity degree*: average connectivity of the hypergraph.  
* *Disconnected subgraphs*: number of disconnected components.

## Extending the dataset

We encourage users to:

* Add new workloads (QASM or [other supported formats](models.md)).  
* Generate corresponding HDHs.  
* Propose and document new metrics (e.g., depth, cut size, entanglement width).  

Pull requests that expand the benchmark set or enrich metadata are very welcome!

### How to add to this database

There are two ways to contribute:


#### 1) Add new workloads + HDHs

##### 1) **Place workloads**  
Put your workload origin files under:  
```Workloads/<Model>/<Origin>/```
This could be anything from a qasm file to circuit generation code.
If the HDH is not generated from functions within the library, we request you add a ```README.md``` to your origin folder explaining how the HDHs were generated.

Example:  
```Workloads/Circuits/MQTBench/qft_8.qasm```

##### 2) **Run the converter**  
Convert the files (qasm strings, qiskit circuits, ...) to HDHs.
```
HDHs/<Model>/<Origin>/pkl/.pkl
HDHs/<Model>/<Origin>/text/__nodes.csv
HDHs/<Model>/<Origin>/text/__edges.csv
HDHs/<Model>/<Origin>/text/__edge_members.csv
```

The example script below converts QASM → HDH and writes them as expected (it can be adapted for other models):  


###### Converter script (QASM → HDH → {pkl,csv})

Requirements: tqdm, the HDH library available on PYTHONPATH, and your QASM converter (hdh.converters.qasm.from_qasm).
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

# Repo import path (one level up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.qasm import from_qasm

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
        w.writerow(["node_id", "type", "time", "realisation"])  # a|p
        for nid in sorted(hdh_graph.S):
            w.writerow([
                nid,
                hdh_graph.sigma.get(nid, ""),
                getattr(hdh_graph, "time_map", {}).get(nid, ""),
                hdh_graph.upsilon.get(nid, "")
            ])

def save_edges_csvs(hdh_graph, edges_path: Path, members_path: Path):
    # Stable ordering to assign edge_index deterministically
    edges_sorted = sorted(hdh_graph.C, key=lambda e: tuple(sorted(e)))
    # edges table (one row per edge)
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "edge_index",
            "type",          # q|c
            "realisation",   # a|p
            "gate_name",
            "role",          # teledata|telegate|''
            "edge_args",     # JSON
            "edge_metadata"  # JSON
        ])
        for idx, e in enumerate(edges_sorted):
            w.writerow([
                idx,
                hdh_graph.tau.get(e, ""),
                hdh_graph.phi.get(e, ""),
                getattr(hdh_graph, "gate_name", {}).get(e, ""),
                getattr(hdh_graph, "edge_role", {}).get(e, ""),
                json.dumps(getattr(hdh_graph, "edge_args", {}).get(e, None), ensure_ascii=False) if e in getattr(hdh_graph, "edge_args", {}) else "",
                json.dumps(getattr(hdh_graph, "edge_metadata", {}).get(e, None), ensure_ascii=False) if e in getattr(hdh_graph, "edge_metadata", {}) else ""
            ])
    # edge_members table (one row per (edge,node))
    with open(members_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_index", "node_id"])
        for idx, e in enumerate(edges_sorted):
            for nid in sorted(e):
                w.writerow([idx, nid])

def main():
    ap = argparse.ArgumentParser(description="Convert QASM workloads to HDH artifacts")
    ap.add_argument("--model", default="Circuits", help="Model folder under Workloads/ and HDHs/")
    ap.add_argument("--origin", default="MQTBench", help="Origin folder under Workloads/<Model>/ and HDHs/<Model>/")
    ap.add_argument("--limit", type=int, default=None, help="Max number of files to convert (useful for large datasets)")
    ap.add_argument("--src-root", default=None, help="Override source root. Default: Workloads/<Model>/<Origin>")
    args = ap.parse_args()

    SRC_DIR = Path(args.src_root) if args.src_root else BASE_DIR / "Workloads" / args.model / args.origin
    DST_ROOT = BASE_DIR / "HDHs" / args.model / args.origin
    PKL_ROOT = DST_ROOT / "pkl"
    TXT_ROOT = DST_ROOT / "text"
    IMG_ROOT = DST_ROOT / "images"  # reserved for future visual exports

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
        qasm_files = qasm_files[: args.limit]

    ok = fail = 0
    with tqdm(total=len(qasm_files),
              desc="Converting QASM → HDH",
              unit="file",
              dynamic_ncols=True,
              mininterval=0.2) as pbar:
        for qf in qasm_files:
            rel = qf.relative_to(SRC_DIR)
            stem = rel.stem
            pkl_dir = PKL_ROOT / rel.parent
            txt_dir = TXT_ROOT / rel.parent
            for d in (pkl_dir, txt_dir):
                ensure_dir(d)
            pbar.set_postfix_str(str(rel))
            pbar.refresh()
            try:
                hdh_graph = from_qasm("file", str(qf))
                # pkl
                save_pkl(hdh_graph, pkl_dir / stem)
                # text (CSV)
                save_nodes_csv(hdh_graph, (txt_dir / f"{stem}__nodes.csv"))
                save_edges_csvs(
                    hdh_graph,
                    edges_path=(txt_dir / f"{stem}__edges.csv"),
                    members_path=(txt_dir / f"{stem}__edge_members.csv"),
                )
                ok += 1
            except Exception as e:
                tqdm.write(f"[fail] {qf}: {e}")
                fail += 1
            finally:
                pbar.update(1)
    print(f"[done] Converted: {ok} | Failed: {fail} | Total processed: {len(qasm_files)}")

if __name__ == "__main__":
    main()
```

##### 3) **Verify & inspect**  
Please open at least one of the `text/*.csv` files (human-readable) and load at least one of the `pkl/*.pkl` objects in Python, to check everything works!

##### 4) **Submit a PR**  
If all went smoothly and you're happy to, submit a PR with your workloads, HDHs, and metrics back to the repository so we can keep growing our testing database.  
If you are going to also submit partitioning method results we recommend to wait and do it all in one!

Note that the datafiles might be a bit too big to directly upload.
In that is the case try doing so with [LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage):

macOS:
```
brew install git-lfs
git lfs install
```
Debian/Ubuntu:
```
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```
Windows:
```
winget install Git.GitLFS
git lfs install
```

From repo root (make sure to change ```<Origin>``` for your Origin name):
```
git lfs track "*.csv"
git add .gitattributes
git commit -m "Adding <Origin> HDHs to database"
git push -u origin main
```

You may need to add your origin
```
git remote add origin https://github.com/<you>/<repo>.git
```

Once you've got it commited to your fork you can push it through a pull request as per usual.

#### 2) Add partitioning method results

If you want to share **partitioning method results**, you can do so by adding adding the partitioning method metadata to the ```HDHs/<Model>/<Origin>/Partitions/partitions_all.csv``` file.

These files may look like: 

```csv
file,n_qubits,k_partitions,greedy_bins,greedy_cost,metis_bins,metis_cost,metis_fails,metis_method,greedytg_bins,greedytg_bins_cost,best
ae_indep_qiskit_10.qasm,10,2,"[[""q0"",""q1"",""q2"",""q3"",""q8""],[""q4"",""q5"",""q6"",""q7"",""q9""]]",30,"[[""q1"",""q3"",""q5"",""q6"",""q7""],[""q0"",""q2"",""q4"",""q8"",""q9""]]",25,False,kl,"[[""q0"",""q1"",""q2"",""q3"",""q9""],[""q4"",""q5"",""q6"",""q7"",""q8""]]",30,metis_tl
ae_indep_qiskit_10.qasm,10,3,"[[""q0"",""q1"",""q2"",""q8""],[""q3"",""q4"",""q6"",""q7""],[""q5"",""q9""]]",40,"[[""q3"",""q5"",""q6"",""q7""],[""q0"",""q2""],[""q1"",""q4"",""q8"",""q9""]]",32,False,kl,"[[""q0"",""q1"",""q2"",""q9""],[""q3"",""q4"",""q5"",""q6""],[""q7"",""q8""]]",38,metis_tl
ae_indep_qiskit_10.qasm,10,4,"[[""q0"",""q1"",""q8""],[""q2"",""q3"",""q7""],[""q4"",""q5"",""q6""],[""q9""]]",45,"[[""q5"",""q7"",""q9""],[""q2"",""q8""],[""q0"",""q4""],[""q1"",""q3"",""q6""]]",37,False,kl,"[[""q0"",""q1"",""q9""],[""q2"",""q3"",""q4""],[""q5"",""q6"",""q7""],[""q8""]]",43,metis_tl
```

First the metadata:

* ```file```: name of the origin file
* ```n_qubits```: number of qubits in workload
* ```k_partitions```: number of partitions made by the method. For instance if you cut your workload once you only create 2 partitions

Then partitioners results can be added. 
In this example we can see 3 partitioning strategies: greedy, metis and greedytg.
They correspond to:

* **Greedy (HDH)** :
Partitions directly on the HDH hypergraph where each hyperedge captures one operation’s dependency set.
We fill bins sequentially: order qubits by a heuristic (e.g., incident cut weight, then degree), and place each into the earliest bin that (i) respects the logical-qubit capacity and (ii) gives the smallest marginal cut increase.
If nothing fits, open the next bin up to k.
Cost = sum of weights of hyperedges spanning >1 bin (default weight 1 per op; domain weights optional).

* **METIS (Telegate graph)**:
Converts the workload into a telegate qubit-interaction graph (nodes = logical qubits; edge weights = interaction pressure indicating a non-local gate would require a “telegate” communication if cut).
Uses the [METIS library](https://pypi.org/project/metis/) to compute a k-way partition with balance constraints and minimal cut on this graph.
Partitions are then re-evaluated on the HDH cost for apples-to-apples comparison.
We typically set edge weights from interaction counts; you can also up-weight edges representing expensive non-local primitives to steer METIS away from cutting them.

* **Greedy-TG (Telegate graph)**:
Same fill-first policy as Greedy (HDH), but decisions are made on the telegate graph.
Nodes are qubits; edge weights reflect how costly it is to separate two qubits (i.e., expected telegate load).
Each qubit goes to the earliest feasible bin that minimizes marginal cut on the telegate graph, ensuring a fair, representation-matched comparison with the HDH greedy approach.

All this information regarding the method used and its origin must be saved in 
```HDHs/<Model>/<Origin>/Partitions/README.md``` 
, otherwise the data will not be merged.

As you can see in the example depending on the strategy you can have more or less saved data. Mandatory columns include:

* quantum communication cost achieved (```cost```) = number of quantum partitioned hyperedges
* the sets of qubits per partitions (```bins```)

Additionally, ```best``` must be re-calculated.
Best corresponds to the name of the method with the lowest cost.

In this example, additional metadata includes the sub-method used within the METIS partitioner (it can default to various sub-methods if the original fails), as well as a failure state for METIS.
The failure state is very important for methods that do not assure the ability to respect a given capacity.
Capacity (i.e., the maximum number of qubits allowed in one partition) should be set to the total number of qubits divided by the number of partitions (rounded up to the next integer).
If the partitioner cannot respect this capacity, the potential failure status should be logged, and if true the method should not be considered in the best evaluation.
An explanation on whether these types of additional logs are necessary must be added to any commit adding new data to the database.
If they are needed, an explanation of what is added is also required.
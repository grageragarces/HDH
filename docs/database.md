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

1) **Place workloads**  
Put your QASM files under:  
```Workloads/<Model>/<Origin>/```
Example:  
```Workloads/Circuits/MQTBench/qft_8.qasm```

2) **Run the converter**  
The script in the section below converts QASM → HDH and writes (can be adapted for other models):  
```
HDHs/<Model>/<Origin>/pkl/.pkl
HDHs/<Model>/<Origin>/text/__nodes.csv
HDHs/<Model>/<Origin>/text/__edges.csv
HDHs/<Model>/<Origin>/text/__edge_members.csv
```

3) **Verify & inspect**  
Open the `text/*.csv` files (human-readable) and load the `pkl/*.pkl` objects in Python for programmatic checks.

4) **(Optional) Add metrics**  
Compute any extra metrics (e.g., partition cut size, modularity, width).  
Save them next to the HDH text files as `*__metrics.json`.

5) **Submit a PR**  
Contribute your workloads, HDHs, and metrics back to the repository.  


#### 2) Add partitioning method results

If you are not adding new workloads but want to share **partitioning method results**, you can do so by uploading a CSV file with suffix:  
HDHs/<Model>/<Origin>/text/<workload_stem>__cut.csv

Each row corresponds to one partitioning run. The file should contain the following columns:

| Field           | Description |
|-----------------|-------------|
| `method`        | Name of the partitioning method (e.g., `metis_kway`, `greedy_binfill`). |
| `qcost`         | Number of **quantum hyperedges cut**. |
| `ccost`         | Number of **classical hyperedges cut**. |
| `k`             | Number of partitions. |
| `partitions`    | Serialized representation of partition sets (e.g., JSON list of node-sets or indices). |
| `parallelism`   | Parallelism value, evaluated as in the HDH paper. |
| `sizes`         | Sizes of partitions (list or JSON array). |

**Example `__cut.csv`:**

```csv
method,qcost,ccost,k,partitions,parallelism,sizes
metis_kway,12,3,4,"[{q0_t0,q1_t0},{q2_t0},{q3_t0,q4_t0},{q5_t0}]",0.67,"[2,1,2,1]"
greedy_binfill,15,4,4,"[{q0_t0,q2_t0},{q1_t0,q3_t0},{q4_t0},{q5_t0}]",0.61,"[2,2,1,1]"
Converter script (QASM → HDH → {pkl,csv})
Requirements: tqdm, the HDH library available on PYTHONPATH, and your QASM converter (hdh.converters.qasm.from_qasm).
```

##### Converter script (QASM → HDH → {pkl,csv})

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
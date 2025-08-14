# HDH Circuit Database Structure

This repository stores **Hybrid Dependency Hypergraph (HDH)** representations of quantum circuits, generated from source workloads (e.g., QASM files).  
The database is organized into **two top-level directories**:

## 1. `Workloads/`
Contains the original circuit definitions, grouped by dataset or source model.  
- **Subdirectories = models/origins** (e.g., `Circuits/MQTBench`).  
- Within each subdirectory are the raw workload files: the initial dataset is made up of almost 2000 qasm circuit implementations of quantum algorithms taken from [MQTBench](https://www.cda.cit.tum.de/mqtbench/).  
- Additional datasets can be added under this structure.

## 2. `HDHs/`
Contains the generated HDH representations of the workloads.  
The subdirectory structure mirrors `Workloads/` to preserve dataset/model organization.  
Inside each dataset folder:
- **`pkl/`** — Binary `.pkl` pickles containing the serialized `hdh_graph` object.  
- **`text/`** — CSV representations:  
  - `__nodes.csv`: list of nodes with their properties:
    - `node_id`, `type`, `time`, `realisation`
  - `__edges.csv`: list of edges with their properties:
    - `edge_index`, `type`, `realisation`, `gate_name`, `role`, `edge_args`, `edge_metadata`
  - `__edge_members.csv`: mapping of `edge_index` to `node_id` for edge membership.

## Data Generation
`database_gen.py` is an example script that:
1. Reads workload circuits from `Workloads/`.  
2. Converts them to HDH format via `from_qasm()`.  
3. Saves both **binary** (`.pkl`) and **CSV** (`text/`) formats under the mirrored `HDHs/` directory.  

The example script processes the **first 200 MQTBench circuits** and can be adapted to add more data from any QASM dataset.


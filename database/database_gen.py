#!/usr/bin/env python3
import sys
import os
import csv
import json
import pickle
from pathlib import Path
from tqdm import tqdm

# Repo import path (one level up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdh.converters.qasm import from_qasm

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "Workloads" / "Circuits" / "MQTBench"
DST_ROOT = BASE_DIR / "HDHs" / "Circuits" / "MQTBench"
PKL_ROOT = DST_ROOT / "pkl"
TXT_ROOT = DST_ROOT / "text"

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
                hdh_graph.time_map.get(nid, ""),
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
                hdh_graph.gate_name.get(e, ""),
                hdh_graph.edge_role.get(e, ""),
                json.dumps(hdh_graph.edge_args.get(e, None), ensure_ascii=False) if e in hdh_graph.edge_args else "",
                json.dumps(hdh_graph.edge_metadata.get(e, None), ensure_ascii=False) if e in hdh_graph.edge_metadata else ""
            ])
    # edge_members table (one row per (edge,node))
    with open(members_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_index", "node_id"])
        for idx, e in enumerate(edges_sorted):
            for nid in sorted(e):
                w.writerow([idx, nid])

def rel_targets(qasm_file: Path):
    rel = qasm_file.relative_to(SRC_DIR)
    pkl_dir = PKL_ROOT / rel.parent
    txt_dir = TXT_ROOT / rel.parent
    img_dir = IMG_ROOT / rel.parent
    return rel, pkl_dir, (txt_dir, rel.stem), img_dir

def convert_all():
    if not SRC_DIR.exists():
        print(f"[error] Source directory not found: {SRC_DIR}")
        sys.exit(1)

    for d in (PKL_ROOT, TXT_ROOT, IMG_ROOT):
        ensure_dir(d)

    qasm_files = sorted(SRC_DIR.rglob("*.qasm"))
    if not qasm_files:
        print(f"[info] No .qasm files found under {SRC_DIR}")
        return

    target = min(200, len(qasm_files))  # 100% at 200 or fewer
    ok = fail = 0

    with tqdm(total=target,
              desc="Converting QASM → HDH (100% at 200 files or done)",
              unit="file",
              dynamic_ncols=True,
              mininterval=0.2) as pbar:

        for idx, qf in enumerate(qasm_files, start=1):
            rel, pkl_dir, (txt_dir, stem), img_dir = rel_targets(qf)
            for d in (pkl_dir, txt_dir, img_dir):
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
                if idx <= target:
                    pbar.update(1)

    print(f"[done] Converted: {ok} | Failed: {fail} | Total found: {len(qasm_files)}")

if __name__ == "__main__":
    convert_all()

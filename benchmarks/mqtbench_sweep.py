"""
Runs the three-mode comparison (see cut_type_comparison.py) across real
circuits from the MQT Bench suite, reporting whatever the results actually
are - this is a broad honesty check, not a demonstration. See
switching_sweep.py for an exact result on the specific structural condition
(a qubit's interaction pattern shifting over time) where combined mode is
known to help; this script instead asks "how much does that show up across
typical real algorithms," without cherry-picking for the answer.

Requires mqt.bench (`pip install mqt.bench`) - not a project dependency,
only needed to reproduce this specific benchmark.

Usage: python -m benchmarks.mqtbench_sweep
"""
import csv
import pathlib

from hdh.converters.qiskit_converter import from_qiskit
from .cut_type_comparison import cut_by_mode

OUT_DIR = pathlib.Path(__file__).parent / "results"

CIRCUITS = ["qft", "ghz", "graphstate", "qpeexact", "wstate", "qftentangled"]
SIZES = [6, 8]
SETTINGS = [(2, 3), (2, 4), (3, 3)]  # (k, cap)


def run():
    from mqt.bench import get_benchmark, BenchmarkLevel

    rows = []
    for name in CIRCUITS:
        for n in SIZES:
            qc = get_benchmark(name, BenchmarkLevel.INDEP, circuit_size=n)
            hdh = from_qiskit(qc)
            for k, cap in SETTINGS:
                row = {"circuit": name, "n_qubits": n, "k": k, "cap": cap}
                for mode in ("combined", "telegate_only", "teledata_only"):
                    try:
                        cost, _ = cut_by_mode(hdh, k, cap, mode)
                        row[mode] = cost
                    except RuntimeError:
                        row[mode] = None  # infeasible under this k/cap
                rows.append(row)
                print(
                    f"{name:12s} n={n} k={k} cap={cap}  "
                    f"combined={row['combined']} telegate_only={row['telegate_only']} "
                    f"teledata_only={row['teledata_only']}"
                )
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path):
    import matplotlib.pyplot as plt
    import numpy as np

    feasible = [r for r in rows if r["combined"] is not None and r["telegate_only"] is not None]
    labels = [f"{r['circuit']}\nn={r['n_qubits']},k={r['k']},cap={r['cap']}" for r in feasible]
    x = np.arange(len(feasible))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(feasible) * 0.6), 5))
    ax.bar(x - width, [r["combined"] for r in feasible], width, label="combined (HDH)")
    ax.bar(x, [r["telegate_only"] for r in feasible], width, label="telegate-only (prior work)")
    teledata_vals = [r["teledata_only"] if r["teledata_only"] is not None else 0 for r in feasible]
    ax.bar(x + width, teledata_vals, width, label="teledata-only")

    ax.set_ylabel("Cut cost (greedy)")
    ax.set_title("HDH cut-type comparison across real MQT Bench circuits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    rows = run()
    save_csv(rows, OUT_DIR / "mqtbench_sweep.csv")
    plot(rows, OUT_DIR / "mqtbench_sweep.png")

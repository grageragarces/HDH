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

Each row also records the number of possible placements per mode: the atomic
groups of HDH nodes the placer assigns. combined places every node on its
own; telegate_only contracts each qubit's whole timeline into one unit, and
teledata_only each multi-qubit gate's nodes, with every other node (e.g.
classical ones) left as its own unit. This is the size of the search space
each formulation exposes.

`summarize` turns the rows into the exact statistics quoted in the paper;
they are printed and written to results/mqtbench_summary.csv.

Usage: python -m benchmarks.mqtbench_sweep
"""
import csv
import pathlib
import statistics

from hdh.converters.qiskit_converter import from_qiskit
from .cut_type_comparison import _build_units, cut_by_mode

MODES = ("combined", "telegate_only", "teledata_only")

OUT_DIR = pathlib.Path(__file__).parent / "results"

CIRCUITS = ["qft", "ghz", "graphstate", "qpeexact", "wstate", "qftentangled"]
SIZES = [6, 8]
SETTINGS = [(2, 3), (2, 4), (3, 3)]  # (k, cap)

# MQT Bench's graphstate builds a random regular graph unless seeded, so an
# unseeded run gives different circuits (and cut costs) every time.
GRAPH_SEED = 0


def _benchmark_kwargs(name):
    return {"seed": GRAPH_SEED} if name == "graphstate" else {}


def run():
    from mqt.bench import get_benchmark, BenchmarkLevel

    rows = []
    for name in CIRCUITS:
        for n in SIZES:
            qc = get_benchmark(name, BenchmarkLevel.INDEP, circuit_size=n,
                               **_benchmark_kwargs(name))
            hdh = from_qiskit(qc)
            units = {mode: len(_build_units(hdh, mode)[0]) for mode in MODES}
            for k, cap in SETTINGS:
                row = {"circuit": name, "n_qubits": n, "k": k, "cap": cap}
                for mode in MODES:
                    try:
                        cost, _ = cut_by_mode(hdh, k, cap, mode)
                        row[mode] = cost
                    except RuntimeError:
                        row[mode] = None  # infeasible under this k/cap
                for mode in MODES:
                    row[f"units_{mode}"] = units[mode]
                rows.append(row)
                print(
                    f"{name:12s} n={n} k={k} cap={cap}  "
                    f"combined={row['combined']} telegate_only={row['telegate_only']} "
                    f"teledata_only={row['teledata_only']}"
                )
    return rows


def _describe(values):
    """n, mean, median, min and max of `values`, or Nones if empty."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(rows):
    """Exact summary statistics over the sweep, as (statistic, value) rows.

    Cut costs are compared only on instances where both modes being compared
    are feasible; possible placements are per circuit, so each circuit is
    counted once rather than once per (k, cap) setting.
    """
    out = [("instances", len(rows))]
    for mode in MODES:
        feasible = [r[mode] for r in rows if r[mode] is not None]
        out.append((f"{mode}_feasible", len(feasible)))
        for stat, value in _describe(feasible).items():
            if stat != "n":
                out.append((f"{mode}_cut_cost_{stat}", value))

    circuits = {(r["circuit"], r["n_qubits"]): r for r in rows}.values()
    for mode in MODES:
        for stat, value in _describe([r[f"units_{mode}"] for r in circuits]).items():
            out.append((f"{mode}_units_{stat}", value))
    ratios = [r["units_combined"] / r["units_telegate_only"] for r in circuits]
    for stat, value in _describe(ratios).items():
        if stat != "n":
            out.append((f"units_combined_over_telegate_only_{stat}",
                        None if value is None else round(value, 2)))

    both = [r for r in rows if r["combined"] is not None and r["telegate_only"] is not None]
    out += [
        ("combined_vs_telegate_only_compared", len(both)),
        ("combined_lower", sum(r["combined"] < r["telegate_only"] for r in both)),
        ("combined_equal", sum(r["combined"] == r["telegate_only"] for r in both)),
        ("combined_higher", sum(r["combined"] > r["telegate_only"] for r in both)),
    ]

    # Does combined match the best single-mode formulation available?
    with_single = [r for r in rows if r["combined"] is not None and
                   any(r[m] is not None for m in ("telegate_only", "teledata_only"))]
    best_single = [min(r[m] for m in ("telegate_only", "teledata_only") if r[m] is not None)
                   for r in with_single]
    out += [
        ("combined_vs_best_single_mode_compared", len(with_single)),
        ("combined_matches_or_beats_best_single_mode",
         sum(r["combined"] <= b for r, b in zip(with_single, best_single))),
        ("combined_worse_than_best_single_mode",
         sum(r["combined"] > b for r, b in zip(with_single, best_single))),
    ]
    return out


def save_summary(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        writer.writerows(summary)


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
    summary = summarize(rows)
    save_summary(summary, OUT_DIR / "mqtbench_summary.csv")
    print("\nSummary:")
    for statistic, value in summary:
        print(f"  {statistic}: {value}")
    plot(rows, OUT_DIR / "mqtbench_sweep.png")

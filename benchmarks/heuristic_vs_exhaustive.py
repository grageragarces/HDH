"""
Reproduces the heuristic-quality claim in the paper's Research Impact
Statement: how close the library's capacity-aware greedy partitioner
(`hdh.passes.cut.compute_cut`) gets to the best assignment a *time-capped*
exhaustive enumeration can find, on real algorithmic circuits from MQT Bench.

This is the in-repo counterpart to Section 4.2 of the companion manuscript.
Methodology follows it deliberately:

- k = 3 fully-connected devices, capacity set for network overhead 1
  (total network capacity == circuit width), the most constrained feasible
  regime.
- Cost is the library's own model: 10 per cut quantum hyperedge, 1 per cut
  classical hyperedge (`hdh.passes.cut.cost` + `weighted_cost`), so the
  heuristic and the enumeration are scored by exactly the same function.
- The enumeration is branch-and-bound over node-level assignments with
  capacity and cost pruning, under a per-circuit time cap.

The cap matters for interpretation. The full space of capacity-respecting
node assignments exceeds 10^70 candidates past six qubits, so for most
instances the enumeration does NOT complete: it returns the best assignment
it reached, flagged `exhaustive_timed_out`. A cost ratio below 1.0 therefore
does not mean the heuristic beat the optimum -- it means the heuristic found
a cheaper assignment than the capped search reached in the time allowed.
Rows where the search completed (`exhaustive_timed_out == False`) are the
only ones where the denominator is a proven optimum.

Requires mqt.bench (`pip install mqt.bench`) - not a project dependency,
only needed to reproduce this specific benchmark.

Usage:
    python -m benchmarks.heuristic_vs_exhaustive
    python -m benchmarks.heuristic_vs_exhaustive --time-limit 300   # paper's cap
"""
import argparse
import csv
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict

from hdh.converters.qiskit_converter import from_qiskit
from hdh.passes.cut import compute_cut, cost, weighted_cost

OUT_DIR = pathlib.Path(__file__).parent / "results"

_Q_RE = re.compile(r"^q(\d+)_t\d+$")

CIRCUITS = ["ghz", "graphstate", "wstate", "qft", "qftentangled", "qpeexact"]
SIZES = [3, 4, 5, 6, 7, 8, 9, 10]
K_DEVICES = 3


def _qubit_of(node_id: str):
    """Qubit index a node belongs to, or None for classical nodes (which do
    not consume device capacity - same rule the greedy partitioner uses)."""
    m = _Q_RE.match(node_id)
    return int(m.group(1)) if m else None


def _edge_weight(hdh, edge) -> int:
    """Cut cost of one hyperedge: quantum dependencies are 10x classical."""
    return 10 if hdh.tau.get(edge, "q") == "q" else 1


def heuristic_cost(hdh, k: int, cap: int):
    """Weighted cut cost of the library's capacity-aware greedy partitioner."""
    partitions, _ = compute_cut(hdh, k, cap)
    return weighted_cost(cost(hdh, partitions))


def exhaustive_min_cost(hdh, k: int, cap: int, time_limit_s: float):
    """Best weighted cut cost found by branch-and-bound over node assignments.

    Deliberately NOT seeded with the greedy result: seeding would bound the
    search by the heuristic and make a ratio below 1.0 impossible, hiding
    exactly the cases the paper reports.

    Returns:
        (best_cost, timed_out, best_partitions). best_cost is None if no
        feasible complete assignment was reached (e.g. a gate spans more
        qubits than any single device's capacity).
    """
    nodes = sorted(hdh.S, key=lambda n: (hdh.time_map.get(n, 0), n))
    n = len(nodes)
    pos_of = {node: i for i, node in enumerate(nodes)}
    node_qubit = {node: _qubit_of(node) for node in nodes}

    # An edge's cut/not-cut status is decided once its last node is placed,
    # so each edge contributes to the running cost exactly once.
    edges_finalized_at = defaultdict(list)
    for edge in hdh.C:
        members = [m for m in edge if m in pos_of]
        if not members:
            continue
        edges_finalized_at[max(pos_of[m] for m in members)].append(
            ([pos_of[m] for m in members], _edge_weight(hdh, edge))
        )

    assign = [-1] * n
    bin_qubits = [set() for _ in range(k)]
    best = [None]
    best_assign = [None]
    start = time.time()
    timed_out = [False]

    def backtrack(pos, running_cost):
        if timed_out[0]:
            return
        if time.time() - start > time_limit_s:
            timed_out[0] = True
            return
        # cut cost only grows as more nodes are placed
        if best[0] is not None and running_cost >= best[0]:
            return
        if pos == n:
            best[0] = running_cost
            best_assign[0] = list(assign)
            return

        node = nodes[pos]
        q = node_qubit[node]
        seen_empty = False
        for b in range(k):
            if not bin_qubits[b] and q is not None:
                # devices are interchangeable: only ever open the
                # lowest-indexed empty one, to kill k! symmetric branches
                if seen_empty:
                    continue
                seen_empty = True
            is_new_qubit = q is not None and q not in bin_qubits[b]
            if is_new_qubit and len(bin_qubits[b]) + 1 > cap:
                continue

            assign[pos] = b
            if is_new_qubit:
                bin_qubits[b].add(q)
            added = 0
            for members, w in edges_finalized_at[pos]:
                if len({assign[i] for i in members}) > 1:
                    added += w
            backtrack(pos + 1, running_cost + added)
            if is_new_qubit:
                bin_qubits[b].discard(q)
            assign[pos] = -1
            if timed_out[0]:
                return

    backtrack(0, 0)
    if best_assign[0] is None:
        return None, timed_out[0], None
    partitions = [set() for _ in range(k)]
    for i, b in enumerate(best_assign[0]):
        partitions[b].add(nodes[i])
    # Cross-check the incremental cost against the library's own scorer, so a
    # bug in this script's bookkeeping can't quietly manufacture a result.
    scored = weighted_cost(cost(hdh, partitions))
    assert scored == best[0], (
        f"internal cost mismatch: incremental={best[0]} library={scored}"
    )
    return best[0], timed_out[0], partitions


def run(sizes=SIZES, circuits=CIRCUITS, k=K_DEVICES, time_limit_s=60.0):
    from mqt.bench import get_benchmark, BenchmarkLevel

    rows = []
    for name in circuits:
        for n_qubits in sizes:
            # network overhead 1: total capacity across k devices == circuit width
            cap = math.ceil(n_qubits / k)
            try:
                qc = get_benchmark(name, BenchmarkLevel.INDEP, circuit_size=n_qubits)
            except Exception as exc:  # size unsupported for this benchmark
                print(f"{name:13s} n={n_qubits}: skipped ({exc})")
                continue
            hdh = from_qiskit(qc)

            try:
                h_cost = heuristic_cost(hdh, k, cap)
            except RuntimeError:
                h_cost = None  # greedy found no feasible placement

            e_cost, e_timed_out, _ = exhaustive_min_cost(hdh, k, cap, time_limit_s)
            ratio = (h_cost / e_cost) if (h_cost is not None and e_cost) else None

            rows.append({
                "circuit": name,
                "n_qubits": n_qubits,
                "k": k,
                "cap": cap,
                "n_nodes": len(hdh.S),
                "heuristic_cost": h_cost,
                "exhaustive_cost": e_cost,
                "exhaustive_timed_out": e_timed_out,
                "cost_ratio": ratio,
            })
            print(
                f"{name:13s} n={n_qubits:2d} cap={cap}  heuristic={h_cost}  "
                f"capped_exhaustive={e_cost}{' (timed out)' if e_timed_out else ''}  "
                f"ratio={ratio if ratio is None else round(ratio, 3)}"
            )
    return rows


def summarize(rows):
    ratios = [r["cost_ratio"] for r in rows if r["cost_ratio"] is not None]
    if not ratios:
        return {}
    completed = [r for r in rows if not r["exhaustive_timed_out"] and r["cost_ratio"] is not None]
    return {
        "instances": len(ratios),
        "mean_cost_ratio": round(statistics.mean(ratios), 4),
        "median_cost_ratio": round(statistics.median(ratios), 4),
        "pct_matching_exactly": round(
            100 * sum(1 for r in ratios if abs(r - 1.0) < 1e-9) / len(ratios), 1
        ),
        "instances_search_completed": len(completed),
        "mean_cost_ratio_completed_only": (
            round(statistics.mean([r["cost_ratio"] for r in completed]), 4)
            if completed else None
        ),
    }


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["cost_ratio"] is not None]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    proven = [r for r in pts if not r["exhaustive_timed_out"]]
    capped = [r for r in pts if r["exhaustive_timed_out"]]
    if proven:
        ax.scatter([r["n_qubits"] for r in proven], [r["cost_ratio"] for r in proven],
                   marker="o", label="search completed (true optimum)")
    if capped:
        ax.scatter([r["n_qubits"] for r in capped], [r["cost_ratio"] for r in capped],
                   marker="x", label="search time-capped (bound only)")
    ax.axhline(1.0, linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Cost ratio (heuristic / exhaustive)")
    ax.set_title("Greedy heuristic vs time-capped exhaustive enumeration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="per-circuit enumeration cap in seconds "
                             "(the companion manuscript uses 300)")
    parser.add_argument("--max-qubits", type=int, default=10,
                        help="largest circuit width to attempt")
    args = parser.parse_args()

    sizes = [n for n in SIZES if n <= args.max_qubits]
    rows = run(sizes=sizes, time_limit_s=args.time_limit)
    if not rows:
        print("No instances ran.")
        return
    save_csv(rows, OUT_DIR / "heuristic_vs_exhaustive.csv")
    save_plot(rows, OUT_DIR / "heuristic_vs_exhaustive.png")
    print("\nSummary:")
    for key, value in summarize(rows).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

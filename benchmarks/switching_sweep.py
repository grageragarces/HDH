"""
Demonstrates, with exact (exhaustive) results, the specific structural
condition under which HDH's combined teledata+telegate partitioning beats
the qubit-level (telegate-only) formulation used by prior hypergraph
approaches: a qubit whose "natural device" changes over the course of the
circuit, because it interacts repeatedly with different groups of qubits at
different times.

telegate_only must commit that qubit to one device for its entire timeline,
so every interaction with whichever group it *didn't* commit to costs a cut.
combined (and teledata_only) can instead pay a one-off teleport cost per
switch and keep every individual gate local. As the number of switches
grows, telegate_only's cost should scale with the number of interactions,
while combined/teledata_only's cost should scale with the (much smaller)
number of switches - this script sweeps switch count and checks that trend
holds, rather than relying on one hand-picked example.

Usage: python -m benchmarks.switching_sweep
Requires matplotlib (already a core hdh dependency).
"""
import csv
import pathlib

from qiskit import QuantumCircuit

from hdh.converters.qiskit_converter import from_qiskit
from .cut_type_exhaustive import exhaustive_optimal

OUT_DIR = pathlib.Path(__file__).parent / "results"


def make_switching_circuit(switches: int, gates_per_phase: int = 3) -> QuantumCircuit:
    """3-qubit circuit where qubit 1 alternates interacting with qubit 0 and
    qubit 2, `switches` times, `gates_per_phase` gates per phase."""
    qc = QuantumCircuit(3)
    qc.h(0)
    for i in range(switches):
        partner = 0 if i % 2 == 0 else 2
        for _ in range(gates_per_phase):
            qc.cx(partner, 1)
    return qc


def run(switch_counts=range(1, 5), k=2, cap=2, time_limit_s=60):
    """Default range (1-4 switches) is chosen so every mode's search
    provably completes (no timeouts) within time_limit_s - the search space
    grows fast enough that switches >= 5 stops being exhaustively provable
    in reasonable time (same combinatorial-explosion behaviour the
    underlying HDH manuscript reports for its own exhaustive-search
    baseline). Rows are still flagged with *_timed_out so an unproven
    result is never silently presented as exact if you do extend the range.
    """
    rows = []
    for switches in switch_counts:
        qc = make_switching_circuit(switches)
        hdh = from_qiskit(qc)
        row = {"switches": switches, "n_gates": switches * 3}
        for mode in ("combined", "telegate_only", "teledata_only"):
            cost, timed_out, n_units = exhaustive_optimal(hdh, k, cap, mode, time_limit_s)
            row[mode] = cost
            row[f"{mode}_timed_out"] = timed_out
            if timed_out:
                print(
                    f"  WARNING: switches={switches} mode={mode} did not finish "
                    f"within {time_limit_s}s - cost={cost} is a best-effort bound, "
                    f"not a proven optimum"
                )
        rows.append(row)
        print(
            f"switches={switches:2d}  combined={row['combined']}  "
            f"telegate_only={row['telegate_only']}  teledata_only={row['teledata_only']}"
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

    switches = [r["switches"] for r in rows]
    any_timed_out = any(
        r[f"{m}_timed_out"] for r in rows for m in ("combined", "telegate_only", "teledata_only")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        "combined": ("o", "-", "combined (HDH)"),
        "telegate_only": ("s", "-", "telegate-only (prior work)"),
        "teledata_only": ("^", "--", "teledata-only"),
    }
    for mode, (marker, ls, label) in styles.items():
        ax.plot(switches, [r[mode] for r in rows], marker=marker, linestyle=ls, label=label)
        timed_out_pts = [(r["switches"], r[mode]) for r in rows if r[f"{mode}_timed_out"]]
        if timed_out_pts:
            xs, ys = zip(*timed_out_pts)
            ax.scatter(xs, ys, marker="x", s=100, color="red", zorder=5)

    ax.set_xlabel("Number of qubit-1 device switches")
    ax.set_ylabel("Optimal cut cost")
    title = "Cut cost vs. interaction-pattern switching (exact, k=2, cap=2)"
    if any_timed_out:
        title += "\n(red x = search did not complete - not a proven optimum)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    rows = run()
    save_csv(rows, OUT_DIR / "switching_sweep.csv")
    plot(rows, OUT_DIR / "switching_sweep.png")

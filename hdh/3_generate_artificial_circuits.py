"""
Step 3: Generate Artificial / Pathological Circuits
=====================================================
Generates four families of artificial quantum circuits designed to stress-test
partitioner evaluation by skewing expected communication costs.

Families (all saved under Database/HDHs/Circuit/Artificial/pkl/):

  1. bloated_*   — Real MQTBench circuits with injected identity gate-pairs
                   (e.g. CNOT–CNOT, H–H on random qubit pairs).  Inflates
                   the interaction graph without changing circuit semantics.
                   → Makes partitioners look *worse* than on real workloads.

  2. star_*      — One control qubit applies CNOT to every other qubit
                   (repeated for several layers).  Pathologically high
                   communication: almost any cut touches the hub.
                   → Inflates communication cost estimates.

  3. brickwall_* — Alternating layers of CNOT on adjacent qubit pairs only
                   (even-odd pattern).  Highly local interactions; any
                   sequential-qubit partition is nearly communication-free.
                   → Deflates communication cost estimates.

  4. fullconn_*  — All-pairs CNOT layers (complete interaction graph).
                   Maximum possible density; every partition is equally bad.
                   → Inflates cost AND removes any meaningful signal between
                     partitioner strategies.

RUN:
    python 3_generate_artificial_circuits.py --database-root ../database
    python 3_generate_artificial_circuits.py --database-root ../database \\
        --mqtbench-pkl-dir ../database/HDHs/Circuit/MQTBench/pkl \\
        --min-qubits 3 --max-qubits 12 --reps 3 \\
        --bloat-ratios 0.25 0.5 1.0 2.0
"""

import argparse
import pickle
import random
import warnings
from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate, HGate
except ImportError:
    raise ImportError("Qiskit is required: pip install qiskit")

try:
    from hdh.converters import from_qasm
except ImportError:
    raise ImportError("hdh library is required.")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def circuit_to_hdh(qc: QuantumCircuit):
    """Convert QuantumCircuit → HDH via QASM string."""
    try:
        qasm_str = qc.qasm()
    except Exception:
        try:
            from qiskit import qasm2
            import io
            buf = io.StringIO()
            qasm2.dump(qc, buf)
            qasm_str = buf.getvalue()
        except Exception as e:
            raise RuntimeError(f"Could not export circuit to QASM: {e}")
    return from_qasm("string", qasm_str)


def save_hdh(hdh_graph, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(hdh_graph, f)
    return out_path


# ---------------------------------------------------------------------------
# Family 1: Bloated circuits
# ---------------------------------------------------------------------------

def make_bloated_circuit(
    source_qc: QuantumCircuit,
    bloat_ratio: float,
    rng: random.Random,
) -> QuantumCircuit:
    """
    Insert identity gate-pairs into a copy of source_qc.

    bloat_ratio: fraction of original gate count to add as extra identity pairs.
    E.g. bloat_ratio=0.5 adds 0.5 * original_depth identity pairs.

    Identity pairs used: H-H, X-X, CNOT-CNOT (on random qubit pairs).
    These are semantically transparent but create spurious hyperedges.
    """
    n = source_qc.num_qubits
    if n < 1:
        return source_qc.copy()

    qc = source_qc.copy()
    original_size = sum(1 for instr in qc.data if instr.operation.name not in ('barrier', 'measure'))
    n_injections = max(1, int(original_size * bloat_ratio))

    for _ in range(n_injections):
        choice = rng.randint(0, 2)
        if choice == 0:
            # H-H on a random qubit
            q = rng.randint(0, n - 1)
            qc.h(q)
            qc.h(q)
        elif choice == 1:
            # X-X on a random qubit
            q = rng.randint(0, n - 1)
            qc.x(q)
            qc.x(q)
        else:
            # CNOT-CNOT on a random qubit pair
            if n >= 2:
                q1, q2 = rng.sample(range(n), 2)
                qc.cx(q1, q2)
                qc.cx(q1, q2)
            else:
                q = 0
                qc.h(q)
                qc.h(q)

    return qc


def generate_bloated(
    mqtbench_pkl_dir: Path,
    out_dir: Path,
    bloat_ratios: List[float],
    seed: int = 42,
    skip_existing: bool = True,
) -> None:
    pkl_files = sorted(p for p in mqtbench_pkl_dir.glob("*.pkl")
                       if not p.name.startswith("random_"))

    if not pkl_files:
        print(f"  [bloated] No non-random pkl files found in {mqtbench_pkl_dir}")
        return

    # We need to reconstruct QuantumCircuit from HDH pkl.
    # Since HDH pkl doesn't store the original circuit, we look for matching
    # QASM files in the sibling Workloads/ tree.  If not available, skip.
    # The workloads are expected at:
    #   Database/Workloads/Circuit/MQTBench/<stem>.qasm
    workloads_dir = mqtbench_pkl_dir.parent.parent.parent.parent.parent / "Workloads" / "Circuit" / "MQTBench"

    rng = random.Random(seed)
    ok = fail = skip = 0

    total = len(pkl_files) * len(bloat_ratios)
    with tqdm(total=total, desc="  bloated", unit="circuit") as pbar:
        for pkl_path in pkl_files:
            stem = pkl_path.stem

            # Find matching QASM
            qasm_path = workloads_dir / f"{stem}.qasm"
            if not qasm_path.exists():
                tqdm.write(f"    [SKIP bloated] No QASM found for {stem}")
                skip += len(bloat_ratios)
                pbar.update(len(bloat_ratios))
                continue

            try:
                source_qc = QuantumCircuit.from_qasm_file(str(qasm_path))
                source_qc.remove_final_measurements(inplace=True)
            except Exception as e:
                tqdm.write(f"    [FAIL bloated] Could not load {stem}: {e}")
                fail += len(bloat_ratios)
                pbar.update(len(bloat_ratios))
                continue

            for ratio in bloat_ratios:
                ratio_str = str(ratio).replace(".", "p")
                name = f"bloated_{stem}_r{ratio_str}"
                out_path = out_dir / f"{name}.pkl"
                pbar.update(1)

                if skip_existing and out_path.exists():
                    skip += 1
                    continue

                try:
                    bloated_qc = make_bloated_circuit(source_qc, ratio, rng)
                    hdh_graph = circuit_to_hdh(bloated_qc)
                    save_hdh(hdh_graph, out_dir, name)
                    ok += 1
                except Exception as e:
                    tqdm.write(f"    [FAIL bloated] {name}: {e}")
                    fail += 1

    print(f"  [bloated] {ok} generated, {skip} skipped, {fail} failed")


# ---------------------------------------------------------------------------
# Family 2: Star topology circuits
# ---------------------------------------------------------------------------

def make_star_circuit(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """
    Hub qubit (index 0) applies CNOT to every other qubit, repeated n_layers times.
    This creates a highly centralised interaction graph — worst-case for cuts.
    """
    assert n_qubits >= 2, "Star circuit needs at least 2 qubits"
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for target in range(1, n_qubits):
            qc.cx(0, target)
    return qc


def generate_star(
    out_dir: Path,
    qubit_range: range,
    layers_list: List[int],
    skip_existing: bool = True,
) -> None:
    configs = [(n, l) for n in qubit_range for l in layers_list]
    ok = fail = skip = 0

    for n_qubits, n_layers in tqdm(configs, desc="  star", unit="circuit"):
        name = f"star_{n_qubits}q_{n_layers}layers"
        out_path = out_dir / f"{name}.pkl"

        if skip_existing and out_path.exists():
            skip += 1
            continue

        try:
            qc = make_star_circuit(n_qubits, n_layers)
            hdh_graph = circuit_to_hdh(qc)
            save_hdh(hdh_graph, out_dir, name)
            ok += 1
        except Exception as e:
            tqdm.write(f"  [FAIL star] {name}: {e}")
            fail += 1

    print(f"  [star] {ok} generated, {skip} skipped, {fail} failed")


# ---------------------------------------------------------------------------
# Family 3: Brick-wall (nearest-neighbour) circuits
# ---------------------------------------------------------------------------

def make_brickwall_circuit(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """
    Alternating layers of CNOT on adjacent pairs:
      Layer 0 (even): (0,1), (2,3), (4,5), ...
      Layer 1 (odd):  (1,2), (3,4), (5,6), ...

    Interactions are strictly local — trivially partitioned by contiguous qubit blocks.
    """
    assert n_qubits >= 2, "Brick-wall circuit needs at least 2 qubits"
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            qc.cx(q, q + 1)
    return qc


def generate_brickwall(
    out_dir: Path,
    qubit_range: range,
    layers_list: List[int],
    skip_existing: bool = True,
) -> None:
    configs = [(n, l) for n in qubit_range for l in layers_list]
    ok = fail = skip = 0

    for n_qubits, n_layers in tqdm(configs, desc="  brickwall", unit="circuit"):
        name = f"brickwall_{n_qubits}q_{n_layers}layers"
        out_path = out_dir / f"{name}.pkl"

        if skip_existing and out_path.exists():
            skip += 1
            continue

        try:
            qc = make_brickwall_circuit(n_qubits, n_layers)
            hdh_graph = circuit_to_hdh(qc)
            save_hdh(hdh_graph, out_dir, name)
            ok += 1
        except Exception as e:
            tqdm.write(f"  [FAIL brickwall] {name}: {e}")
            fail += 1

    print(f"  [brickwall] {ok} generated, {skip} skipped, {fail} failed")


# ---------------------------------------------------------------------------
# Family 4: Fully-connected (all-pairs) circuits
# ---------------------------------------------------------------------------

def make_fullconn_circuit(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """
    Every ordered pair (i, j) with i ≠ j gets a CNOT, repeated n_layers times.
    Maximum interaction density — partitioner cannot find any 'good' cut.
    """
    assert n_qubits >= 2, "Full-connected circuit needs at least 2 qubits"
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qc.cx(i, j)
    return qc


def generate_fullconn(
    out_dir: Path,
    qubit_range: range,
    layers_list: List[int],
    skip_existing: bool = True,
) -> None:
    configs = [(n, l) for n in qubit_range for l in layers_list]
    ok = fail = skip = 0

    for n_qubits, n_layers in tqdm(configs, desc="  fullconn", unit="circuit"):
        name = f"fullconn_{n_qubits}q_{n_layers}layers"
        out_path = out_dir / f"{name}.pkl"

        if skip_existing and out_path.exists():
            skip += 1
            continue

        try:
            qc = make_fullconn_circuit(n_qubits, n_layers)
            hdh_graph = circuit_to_hdh(qc)
            save_hdh(hdh_graph, out_dir, name)
            ok += 1
        except Exception as e:
            tqdm.write(f"  [FAIL fullconn] {name}: {e}")
            fail += 1

    print(f"  [fullconn] {ok} generated, {skip} skipped, {fail} failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial/pathological quantum circuits for partitioner bias analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Circuit families:
  bloated   Real MQTBench circuits with injected identity gate-pairs (inflates cost)
  star      Hub qubit connected to all others (inflates cost, highly centralised)
  brickwall Adjacent-only CNOT layers (deflates cost, trivially local)
  fullconn  All-pairs CNOT (maximum density, removes discriminative signal)

Examples:
  python 3_generate_artificial_circuits.py --database-root ./Database
  python 3_generate_artificial_circuits.py --database-root ./Database \\
      --families star brickwall fullconn \\
      --min-qubits 2 --max-qubits 10 --layers 1 3 5
        """,
    )
    parser.add_argument("--database-root", type=Path, required=True,
                        help="Root directory of the database (contains HDHs/)")
    parser.add_argument("--mqtbench-pkl-dir", type=Path, default=None,
                        help="Override path to MQTBench pkl dir (for bloated family)")

    parser.add_argument("--families", nargs="+",
                        choices=["bloated", "star", "brickwall", "fullconn"],
                        default=["bloated", "star", "brickwall", "fullconn"],
                        help="Which circuit families to generate (default: all)")

    # Qubit/layer range for star, brickwall, fullconn
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=12)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Layer counts for star/brickwall/fullconn (default: 1 3 5 10)")

    # Bloat-specific
    parser.add_argument("--bloat-ratios", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0],
                        help="Bloat ratios for bloated family (default: 0.25 0.5 1.0 2.0)")
    parser.add_argument("--bloat-seed", type=int, default=42)

    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate even if pkl already exists")

    args = parser.parse_args()

    mqtbench_pkl_dir = (
        args.mqtbench_pkl_dir
        or args.database_root / "HDHs" / "Circuit" / "MQTBench" / "pkl"
    )
    out_dir = args.database_root / "HDHs" / "Circuit" / "Artificial" / "pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    qubit_range = range(args.min_qubits, args.max_qubits + 1)
    skip_existing = not args.no_skip

    print("="*60)
    print("Artificial Circuit Generator")
    print("="*60)
    print(f"  Families   : {args.families}")
    print(f"  Qubits     : {args.min_qubits} – {args.max_qubits}")
    print(f"  Layers     : {args.layers}")
    print(f"  Bloat ratios: {args.bloat_ratios}")
    print(f"  Output     : {out_dir}")
    print()

    if "bloated" in args.families:
        print("[1/4] Generating BLOATED circuits ...")
        generate_bloated(
            mqtbench_pkl_dir=mqtbench_pkl_dir,
            out_dir=out_dir,
            bloat_ratios=args.bloat_ratios,
            seed=args.bloat_seed,
            skip_existing=skip_existing,
        )

    if "star" in args.families:
        print("[2/4] Generating STAR circuits ...")
        generate_star(
            out_dir=out_dir,
            qubit_range=qubit_range,
            layers_list=args.layers,
            skip_existing=skip_existing,
        )

    if "brickwall" in args.families:
        print("[3/4] Generating BRICKWALL circuits ...")
        generate_brickwall(
            out_dir=out_dir,
            qubit_range=qubit_range,
            layers_list=args.layers,
            skip_existing=skip_existing,
        )

    if "fullconn" in args.families:
        print("[4/4] Generating FULLY-CONNECTED circuits ...")
        generate_fullconn(
            out_dir=out_dir,
            qubit_range=qubit_range,
            layers_list=args.layers,
            skip_existing=skip_existing,
        )

    print()
    print("="*60)
    print(f"All done. Circuits saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

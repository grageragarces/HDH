"""
Step 2: Generate Qiskit Random Circuits
========================================
Uses qiskit.circuit.random.random_circuit to generate random quantum circuits
across a range of qubit counts and depths, converts each to HDH, and saves
as pkl files in Database/HDHs/Circuit/QiskitRandom/pkl/.

Each circuit is named:  qiskit_random_{n_qubits}q_{depth}d_{seed}.pkl

RUN:
    python 2_generate_qiskit_random.py --database-root ../database
    python 2_generate_qiskit_random.py --database-root ../database \\
        --min-qubits 2 --max-qubits 12 --depths 5 10 20 --seeds-per-config 5
"""

import argparse
import pickle
import warnings
from pathlib import Path

from tqdm import tqdm

try:
    from qiskit.circuit.random import random_circuit
except ImportError:
    raise ImportError("Qiskit is required: pip install qiskit")

try:
    from hdh.converters import from_qasm
except ImportError:
    raise ImportError("hdh library is required.")


def circuit_to_hdh(qc):
    """Convert a Qiskit QuantumCircuit to HDH via QASM string."""
    try:
        qasm_str = qc.qasm()
    except Exception:
        # Fallback: use qasm2 exporter if available
        try:
            from qiskit import qasm2
            import io
            buf = io.StringIO()
            qasm2.dump(qc, buf)
            qasm_str = buf.getvalue()
        except Exception as e:
            raise RuntimeError(f"Could not export circuit to QASM: {e}")
    return from_qasm("string", qasm_str)


def generate_qiskit_random(
    database_root: Path,
    min_qubits: int = 2,
    max_qubits: int = 15,
    depths: list = None,
    seeds_per_config: int = 3,
    max_operand_size: int = 2,    # gates acting on at most 2 qubits (realistic)
    conditional: bool = False,
    reset: bool = False,
    measure: bool = False,
    skip_existing: bool = True,
) -> None:
    if depths is None:
        depths = [5, 10, 20, 40]

    out_dir = database_root / "HDHs" / "Circuit" / "QiskitRandom" / "pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (n, d, s)
        for n in range(min_qubits, max_qubits + 1)
        for d in depths
        for s in range(seeds_per_config)
    ]

    print(f"QiskitRandom generation")
    print(f"  Qubits   : {min_qubits} – {max_qubits}")
    print(f"  Depths   : {depths}")
    print(f"  Seeds    : {seeds_per_config} per config  (seed = config index)")
    print(f"  Total    : {len(configs)} circuits")
    print(f"  Output   : {out_dir}")
    print()

    ok = 0
    fail = 0
    skip = 0

    for n_qubits, depth, seed in tqdm(configs, desc="Generating", unit="circuit"):
        name = f"qiskit_random_{n_qubits}q_{depth}d_{seed}"
        out_path = out_dir / f"{name}.pkl"

        if skip_existing and out_path.exists():
            skip += 1
            continue

        try:
            qc = random_circuit(
                num_qubits=n_qubits,
                depth=depth,
                max_operands=max_operand_size,
                conditional=conditional,
                reset=reset,
                measure=measure,
                seed=seed,
            )

            # Remove measurements if any slipped through — cleaner for partitioning
            qc.remove_final_measurements(inplace=True)

            hdh_graph = circuit_to_hdh(qc)

            with open(out_path, "wb") as f:
                pickle.dump(hdh_graph, f)

            ok += 1

        except Exception as e:
            tqdm.write(f"  [FAIL] {name}: {e}")
            fail += 1

    print()
    print(f"Done — {ok} generated, {skip} skipped (already existed), {fail} failed.")
    print(f"Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qiskit random circuits and save as HDH pkl files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database-root", type=Path, required=True,
                        help="Root directory of the database (contains HDHs/)")
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=15)
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help="Circuit depths to generate (default: 5 10 20 40)")
    parser.add_argument("--seeds-per-config", type=int, default=3,
                        help="Number of random seeds per (qubits, depth) config (default: 3)")
    parser.add_argument("--max-operand-size", type=int, default=2,
                        help="Maximum gate operand size / arity (default: 2)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate even if pkl already exists")
    args = parser.parse_args()

    generate_qiskit_random(
        database_root=args.database_root,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        depths=args.depths,
        seeds_per_config=args.seeds_per_config,
        max_operand_size=args.max_operand_size,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()

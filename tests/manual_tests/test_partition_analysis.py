import os
from qiskit import QuantumCircuit
from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.passes.cut import compute_cut, compute_cut_by_time_percent, gates_by_partition

def load_qiskit_circuit(qasm_path):
    return QuantumCircuit.from_qasm_file(qasm_path)

def run_tests(qasm_path, num_parts=3, cut_percent=0.3):
    print(f"\n[INFO] Loading: {qasm_path}")
    qc = load_qiskit_circuit(qasm_path)
    hdh = from_qiskit(qc)

    print(f"[✓] Qubits: {hdh.get_num_qubits()}")
    print(f"[✓] Total hyperedges: {len(hdh.C)}")

    print("\n--- METIS PARTITION ---")
    partitions = compute_cut(hdh, num_parts)
    intra, inter = gates_by_partition(hdh, partitions)

    for i, part in enumerate(intra):
        print(f"\nPartition {i}:")
        for edge in part:
            print(f"  gate: {hdh.gate_name[edge]} on {edge}")

    print("\n[✓] Inter-partition gates:")
    for edge in inter:
        print(f"  {hdh.gate_name.get(edge, '?')} between {edge}")

    print(f"\n[✓] Total inter-partition gates: {len(inter)}")

    print("\n--- TIME POSITION CUT ---")
    time_partitions = compute_cut_by_time_percent(hdh, percent=cut_percent)
    print(f"[✓] Time-cut at {cut_percent*100:.1f}% => Sizes: {[len(p) for p in time_partitions]}")

    time_intra, time_inter = gates_by_partition(hdh, time_partitions)
    print(f"[✓] Inter-partition gates (time-based): {len(time_inter)}")
    print("  Types:", {hdh.gate_name.get(e, '?') for e in time_inter})

if __name__ == "__main__":
    qasm_file = "ae_indep_qiskit_20.qasm"  
    assert os.path.isfile(qasm_file), f"QASM file not found: {qasm_file}"
    run_tests(qasm_file)

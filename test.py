from qiskit import QuantumCircuit
from hdh import HDH
from hdh.visualize import plot_hdh
from hdh.passes import cut_and_rewrite_hdh
from hdh.converters import from_qiskit, to_qiskit
from hdh.passes.primitives import teledata, telegate
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import os
import matplotlib.pyplot as plt

def save_circuit_png(qc: QuantumCircuit, filename: str):
    fig = qc.draw(output="mpl", fold=-1)  # no line folding
    fig.savefig(filename)
    plt.close(fig)
    
def build_sample_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(6)
    for i in range(6):
        qc.h(i)
        qc.rx(0.5, i)
    for i in range(5):
        qc.cx(i, i + 1)
    qc.measure_all() 
    return qc

def main():
    os.makedirs("output", exist_ok=True)

    # Step 1: Build a 6-qubit Qiskit circuit
    qc = build_sample_circuit()
    # print("Original Circuit:")
    # print(qc.draw())

    # Save original circuit as PNG
    save_circuit_png(qc, "output/original_qiskit.png")

    # Step 2: Convert to HDH
    hdh = from_qiskit(qc)

    # Step 3: Save original HDH graph
    plot_hdh(hdh, "output/original_hdh.png")

    # Step 4: Partition and insert primitives
    num_parts = 2

    allowed_primitives = {
        "quantum": {"tp", "cat"},
        "classical": {"ccom"}
    }

    qiskit_primitives = {
        "tp": teledata(),
        "cat": telegate()
    }
    partitioned = cut_and_rewrite_hdh(hdh, num_parts, allowed_primitives, insert_qiskit_circuits=True, qiskit_primitives=qiskit_primitives)

    # Step 5: Save partitioned HDHs
    for i, part in enumerate(partitioned):
        plot_hdh(part, f"output/cut_partition_{i}.png")

    # Step 6: Rebuild full Qiskit circuit and save it
    for i, part in enumerate(partitioned):
        # print(f"\nPartition {i} Circuit:")
        qci = to_qiskit(part)
        # print(qci.draw())
        save_circuit_png(qci, f"output/partition_{i}_qiskit.png")

if __name__ == "__main__":
    main()

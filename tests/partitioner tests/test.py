import sys
sys.path.append('grageragarces/hdh/HDH-non-database')
import pandas as pd
import random
import networkx as nx
from tqdm import tqdm
from hdh.models.circuit import Circuit
from hdh.models.mbqc import MBQC
from hdh.models.qw import QW
from hdh.models.qca import QCA
from hdh.passes.cut import compute_cut, metis_telegate

def generate_random_circuit(num_qubits, num_instructions):
    """Generates a random circuit workload."""
    circuit = Circuit()
    for _ in range(num_instructions):
        gate = random.choice(["h", "cx", "ccx", "z", "measure"])
        if gate == "measure":
            qubits = [random.randint(0, num_qubits - 1)]
            circuit.add_instruction(gate, qubits)
        elif gate in ["h", "z"]:
            qubits = [random.randint(0, num_qubits - 1)]
            circuit.add_instruction(gate, qubits)
        elif gate == "cx":
            qubits = random.sample(range(num_qubits), 2)
            circuit.add_instruction(gate, qubits)
        elif gate == "ccx":
            if num_qubits >= 3:
                qubits = random.sample(range(num_qubits), 3)
                circuit.add_instruction(gate, qubits)
    return circuit

def generate_random_mbqc(num_operations):
    """Generates a random MBQC workload."""
    mbqc = MBQC()
    nodes = [f"q{i}" for i in range(num_operations)]
    for i in range(num_operations):
        op_type = random.choice(["N", "E", "M", "C"])
        a = random.sample(nodes, random.randint(0, min(i, 3)))
        b = nodes[i]
        mbqc.add_operation(op_type, a, b)
    return mbqc

def generate_random_qw(num_steps):
    """Generates a random QW workload."""
    qw = QW()
    last_qubit = "q0"
    for _ in range(num_steps):
        op_type = random.choice(["coin", "shift", "measure"])
        if op_type == "coin":
            last_qubit = qw.add_coin(last_qubit)
        elif op_type == "shift":
            last_qubit = qw.add_shift(last_qubit)
        elif op_type == "measure":
            qw.add_measurement(last_qubit, f"c{random.randint(0, num_steps)}")
    return qw

def generate_random_qca(num_nodes, num_steps):
    """Generates a random QCA workload."""
    topology = {f"q{i}": [] for i in range(num_nodes)}
    for i in range(num_nodes):
        neighbors = random.sample(list(topology.keys()), random.randint(1, num_nodes -1))
        topology[f"q{i}"] = [n for n in neighbors if n != f"q{i}"]
    measurements = random.sample(list(topology.keys()), random.randint(1, num_nodes))
    return QCA(topology, measurements, num_steps)


results = []
# Run the simulation 99 times
for _ in tqdm(range(900), desc="Overall Progress"):
    # Generate and partition workloads for each model
    for model_type in ["Circuit", "MBQC", "QW", "QCA"]:
        if model_type == "Circuit":
            workload = generate_random_circuit(num_qubits=5, num_instructions=10)
        elif model_type == "MBQC":
            workload = generate_random_mbqc(num_operations=10)
        elif model_type == "QW":
            workload = generate_random_qw(num_steps=10)
        elif model_type == "QCA":
            workload = generate_random_qca(num_nodes=5, num_steps=3)

        hdh_graph = workload.build_hdh()
        num_qubits = len(set(int(n[1:].split('_')[0]) for n in hdh_graph.S if n.startswith('q')))
        capacity = num_qubits // 2 + 1 if num_qubits > 0 else 1

        # Partition with compute_cut
        partitions_greedy, cost_greedy = compute_cut(hdh_graph, k=2, cap=capacity)
        results.append({
            "model_type": model_type,
            "partitioner": "compute_cut",
            "partitions": partitions_greedy,
            "cost": cost_greedy
        })

        # Partition with metis_telegate
        try:
            partitions_metis, cost_metis, _, _ = metis_telegate(hdh_graph, partitions=2, capacities=capacity)
            results.append({
                "model_type": model_type,
                "partitioner": "metis_telegate",
                "partitions": partitions_metis,
                "cost": cost_metis
            })
        except ImportError:
            print("nxmetis is not installed. Skipping metis_telegate partitioner.")


# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv("partitions.csv", index=False)

print("Partitions saved to partitions.csv")
from qiskit import QuantumCircuit
from typing import Set
from .. import HDH

def from_qiskit_circuit(circuit: QuantumCircuit) -> HDH:
    hdh = HDH()
    qubit_time = {}  # lazy init
    clbit_time = {clbit: 0 for clbit in circuit.clbits}

    CONTROL_GATES = {
        "cx": [0],
        "ccx": [0, 1],
    }

    for inst_index, instruction in enumerate(circuit.data):
        gate = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        gate_name = gate.name.lower()

        control_indices = CONTROL_GATES.get(gate_name, list(range(len(qargs) - 1)))
        modifies_flags = [i not in control_indices for i in range(len(qargs))]

        for i, qubit in enumerate(qargs):
            if qubit not in qubit_time:
                assigned = [qubit_time[q] for q in qubit_time]
                latest_so_far = max(assigned) if assigned else 0
                qubit_time[qubit] = latest_so_far

        active_times = [qubit_time[q] for q in qargs]
        time_step = max(active_times) + 1

        in_nodes: Set[str] = set()
        out_nodes: Set[str] = set()

        for i, qubit in enumerate(qargs):
            qname = f"q{circuit.find_bit(qubit).index}"
            modifies = modifies_flags[i]

            t_in = qubit_time[qubit]
            in_id = f"{qname}_t{t_in}"
            hdh.add_node(in_id, "q", t_in)
            in_nodes.add(in_id)

            if modifies or gate_name == "reset":
                t_out = time_step
                out_id = f"{qname}_t{t_out}"
                hdh.add_node(out_id, "q", t_out)
                out_nodes.add(out_id)
                qubit_time[qubit] = t_out

        if gate_name == "measure":
            for i, qubit in enumerate(qargs):
                qname = f"q{circuit.find_bit(qubit).index}"
                t_in = qubit_time[qubit]
                in_id = f"{qname}_t{t_in}"
                hdh.add_node(in_id, "q", t_in)
                in_nodes.add(in_id)

            for clbit in cargs:
                cname = f"c{circuit.find_bit(clbit).index}"
                t_q = max([qubit_time[q] for q in qargs]) + 1
                out_id = f"{cname}_t{t_q}"
                hdh.add_node(out_id, "c", t_q)
                out_nodes.add(out_id)
                clbit_time[clbit] = t_q + 1

        if hasattr(gate, 'condition') and gate.condition:
            cond_reg, cond_val = gate.condition
            for clbit in cond_reg:
                cname = f"c{circuit.find_bit(clbit).index}"
                t_c = clbit_time[clbit]
                in_id = f"{cname}_t{t_c}"
                hdh.add_node(in_id, "c", t_c)
                in_nodes.add(in_id)

        edge_nodes = in_nodes.union(out_nodes)
        edge_type = "c" if gate_name == "measure" else "q"
        hdh.add_hyperedge(edge_nodes, edge_type)

    return hdh
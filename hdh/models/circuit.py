from typing import List, Tuple, Optional, Set, Dict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Circuit model

class Circuit:
    def __init__(self):
        self.instructions: List[
            Tuple[str, List[int], List[int], List[bool]]
        ] = []  # (name, qubits, clbits, modifies_flags)
        self.cond_instructions: List[
            Tuple[int, int, int, str]
        ] = []  # (meas_qubit, target_qubit, outcome, gate_name)

    def add_instruction(
        self,
        name: str,
        qubits: List[int],
        clbits: Optional[List[int]] = None,
        modifies_flags: Optional[List[bool]] = None
    ):
        name = name.lower()

        if name == "measure":
            if clbits is not None:
                raise ValueError("Do not specify classical bits for 'measure'. They are assumed to match qubit indices.")
            clbits = qubits.copy()
            modifies_flags = [True] * len(qubits)
        else:
            clbits = clbits or []
            modifies_flags = modifies_flags or [True] * len(qubits)

        self.instructions.append((name, qubits, clbits, modifies_flags))

    def add_conditional_gate(
        self,
        meas_qubit: int,
        target_qubit: int,
        # outcome: int,
        gate_name: str
    ):
        if not isinstance(gate_name, str):
            raise ValueError("gate_name must be a string (e.g., 'z')")
        gate_name = gate_name.lower()
        self.cond_instructions.append((meas_qubit, target_qubit, 
                                    #    outcome, 
                                       gate_name))
        self.add_instruction("measure", [meas_qubit])

    def build_hdh(self, hdh_cls=HDH) -> HDH:
        hdh = hdh_cls()
        qubit_time: Dict[int, int] = {}
        clbit_time: Dict[int, int] = {}

        for name, qargs, cargs, modifies_flags in self.instructions:
            if name in {"barrier", "snapshot", "delay", "label"}:
                continue

            for q in qargs:
                if q not in qubit_time:
                    qubit_time[q] = max(qubit_time.values(), default=0)

            active_times = [qubit_time[q] for q in qargs]
            time_step = max(active_times) + 1 if active_times else 0

            in_nodes: Set[str] = set()
            out_nodes: Set[str] = set()

            for i, qubit in enumerate(qargs):
                t_in = qubit_time[qubit]
                qname = f"q{qubit}"
                in_id = f"{qname}_t{t_in}"
                hdh.add_node(in_id, "q", t_in)
                in_nodes.add(in_id)

                if modifies_flags[i] and name != "measure":
                    t_out = time_step
                    out_id = f"{qname}_t{t_out}"
                    hdh.add_node(out_id, "q", t_out)
                    out_nodes.add(out_id)
                    qubit_time[qubit] = t_out

            if name == "measure":
                for i, qubit in enumerate(qargs):
                    qname = f"q{qubit}"
                    in_id = f"{qname}_t{qubit_time[qubit]}"
                    in_nodes.add(in_id)

                    relevant_edges = [
                        edge for edge in hdh.C
                        if any(n.startswith(qname) for n in edge)
                    ]
                    latest_q_time = max(
                        hdh.time_map[n]
                        for edge in relevant_edges
                        for n in edge if n.startswith(qname)
                    ) if relevant_edges else qubit_time[qubit]

                    clbit = cargs[i]
                    cname = f"c{clbit}"
                    t_out = latest_q_time + 1
                    out_id = f"{cname}_t{t_out}"
                    hdh.add_node(out_id, "c", t_out)
                    out_nodes.add(out_id)
                    clbit_time[clbit] = t_out + 1

            for clbit in cargs:
                if name != "measure":
                    t = clbit_time.get(clbit, 0)
                    cname = f"c{clbit}"
                    out_id = f"{cname}_t{t + 1}"
                    hdh.add_node(out_id, "c", t + 1)
                    out_nodes.add(out_id)
                    clbit_time[clbit] = t + 2

            all_nodes = in_nodes | out_nodes
            if all(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            elif any(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            else:
                edge_type = "q"

            edges = []
            for in_node in in_nodes:
                edge_nodes = {in_node} | out_nodes
                edge = hdh.add_hyperedge(edge_nodes, edge_type, name=name)
                edges.append(edge)

            q_with_time = [(q, qubit_time[q]) for q in qargs]
            c_with_time = [(c, clbit_time.get(c, 0)) for c in cargs]
            for edge in edges:
                hdh.edge_args[edge] = (q_with_time, c_with_time, modifies_flags)

        # Conditional gates - currently enable mid circuit measurements
        for meas_q, target_q, gate_name in self.cond_instructions:
            """
            Currently not setup up for multiqubit gates
            something wrong with next step - not advacing correctly post the operation
            """
            if gate_name in {"cx", "ccx"}:
                raise ValueError(f"Multiqubit conditional gates are unsupported at the moment.If this you would like them to be supported please open an issue on GitHub: {gate_name}")
            # Resolve classical time (and create classical node)
            cname = f"c{meas_q}"
            if meas_q not in clbit_time:
                clbit_time[meas_q] = max(clbit_time.values(), default=0)
            c_in_time = clbit_time[meas_q]
            c_out_time = c_in_time + 1
            c_in_node = f"{cname}_t{c_in_time}"
            c_out_node = f"{cname}_t{c_out_time}"
            hdh.add_node(c_in_node, "c", c_in_time)
            hdh.add_node(c_out_node, "c", c_out_time)
            clbit_time[meas_q] = c_out_time + 1

            # Resolve quantum time (and create quantum node)
            qname = f"q{target_q}"
            if target_q not in qubit_time:
                qubit_time[target_q] = max(qubit_time.values(), default=0)
            if c_out_time <= qubit_time[target_q]:
                q_in_time = qubit_time[target_q] - 1
                q_out_time = qubit_time[target_q]
            else:
                q_in_time = c_out_time - 1
                q_out_time = c_out_time
            q_in_node = f"{qname}_t{q_in_time}"
            q_out_node = f"{qname}_t{q_out_time}"
            hdh.add_node(q_out_node, "q", q_out_time)
            qubit_time[target_q] = q_out_time
            
            # Add classical control edge
            c_in_node = f"{cname}_t{c_in_time-1}" # do not create a new node
            hdh.add_hyperedge({q_in_node, c_in_node}, "c", name=f"{gate_name}_if_{meas_q}")
            print(c_in_node, c_in_time, c_out_node, c_out_time, q_in_node, q_in_time, q_out_node, q_out_time)
            
            # Add quantum gate edge
            hdh.add_hyperedge({q_in_node, q_out_node}, "q", name=gate_name)
            print(c_in_node, c_in_time, c_out_node, c_out_time, q_in_node, q_in_time, q_out_node, q_out_time)
            
            qubit_time[target_q] = q_out_time + 1
            clbit_time[meas_q] = c_out_time + 1

        return hdh

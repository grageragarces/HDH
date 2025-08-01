from typing import List, Tuple, Optional, Set, Dict, Literal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

class Circuit:
    def __init__(self):
        self.instructions: List[
            Tuple[str, List[int], List[int], List[bool], Literal["a", "p"]]
        ] = []  # (name, qubits, bits, modifies_flags, cond_flag)

    def add_instruction(
        self,
        name: str,
        qubits: List[int],
        bits: Optional[List[int]] = None,
        modifies_flags: Optional[List[bool]] = None,
        cond_flag: Literal["a", "p"] = "a"
    ):
        name = name.lower()

        if name == "measure":
            if bits is not None:
                raise ValueError("Do not specify classical bits for 'measure'. They are assumed to match qubit indices.")
            bits = qubits.copy()
            modifies_flags = [True] * len(qubits)
        else:
            bits = bits or []
            modifies_flags = modifies_flags or [True] * len(qubits)

        self.instructions.append((name, qubits, bits, modifies_flags, cond_flag))

    def build_hdh(self, hdh_cls=HDH) -> HDH:
        hdh = hdh_cls()
        qubit_time: Dict[int, int] = {}
        bit_time: Dict[int, int] = {}

        for name, qargs, cargs, modifies_flags, cond_flag in self.instructions:
            if name in {"barrier", "snapshot", "delay", "label"}:
                continue
            
            # Conditional gate handling
            if cond_flag == "p":
                if len(qargs) < 2:
                    raise ValueError("Conditional gates must be of the form: [control_clbit, target_qubit]. These may be one and the same.")

                ctrl_idx = qargs[0]
                target_qubits = qargs[1:]
                cname = f"c{ctrl_idx}"
                c_time = bit_time.get(ctrl_idx, 1) - 1  # use the time of the last created classical output
                ctrl_node = f"{cname}_t{c_time}"

                for tq in target_qubits:
                    qname = f"q{tq}"
                    t_out = c_time + 1

                    q_out = f"{qname}_t{t_out}"

                    hdh.add_node(q_out, "q", t_out)

                    hdh.add_hyperedge({ctrl_node, q_out}, "c", name=name, node_real="p")

                    # Advance times
                    qubit_time[tq] = t_out
                    bit_time[ctrl_idx] = max(bit_time.get(ctrl_idx, 0), t_out)

            # Non-conditional (actualized) gate
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
                hdh.add_node(in_id, "q", t_in, node_real=cond_flag)
                in_nodes.add(in_id)

                if modifies_flags[i] and name != "measure":
                    t_out = time_step
                    out_id = f"{qname}_t{t_out}"
                    hdh.add_node(out_id, "q", t_out, node_real=cond_flag)
                    out_nodes.add(out_id)
                    qubit_time[qubit] = t_out

            if name == "measure":
                for i, qubit in enumerate(qargs):
                    qname = f"q{qubit}"
                    in_id = f"{qname}_t{qubit_time[qubit]}"
                    in_nodes.add(in_id)

                    latest_q_time = qubit_time[qubit]

                    bit = cargs[i]
                    cname = f"c{bit}"
                    t_out = latest_q_time + 1
                    out_id = f"{cname}_t{t_out}"
                    hdh.add_node(out_id, "c", t_out, node_real=cond_flag)
                    out_nodes.add(out_id)
                    bit_time[bit] = t_out + 1

            for bit in cargs:
                if name != "measure":
                    t = bit_time.get(bit, 0)
                    cname = f"c{bit}"
                    out_id = f"{cname}_t{t + 1}"
                    hdh.add_node(out_id, "c", t + 1, node_real=cond_flag)
                    out_nodes.add(out_id)
                    bit_time[bit] = t + 1

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
                edge = hdh.add_hyperedge(edge_nodes, edge_type, name=name, node_real=cond_flag)
                edges.append(edge)

            q_with_time = [(q, qubit_time[q]) for q in qargs]
            c_with_time = [(c, bit_time.get(c, 0)) for c in cargs]
            for edge in edges:
                hdh.edge_args[edge] = (q_with_time, c_with_time, modifies_flags)

        return hdh

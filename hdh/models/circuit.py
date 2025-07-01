from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
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

    def add_instruction(
        self,
        name: str,
        qubits: List[int],
        clbits: Optional[List[int]] = None,
        modifies_flags: Optional[List[bool]] = None
    ):
        clbits = clbits or []
        modifies_flags = modifies_flags or [True] * len(qubits)
        self.instructions.append((name.lower(), qubits, clbits, modifies_flags))

    def build_hdh(self, hdh_cls=HDH) -> HDH:
        hdh = hdh_cls()
        qubit_time: Dict[int, int] = {}
        clbit_time: Dict[int, int] = {}

        for name, qargs, cargs, modifies_flags in self.instructions:
            if name in {"barrier", "snapshot", "delay", "label"}:
                continue

            # Init times
            for q in qargs:
                if q not in qubit_time:
                    qubit_time[q] = max(qubit_time.values(), default=0)

            active_times = [qubit_time[q] for q in qargs]
            time_step = max(active_times) + 1 if active_times else 0

            in_nodes: Set[str] = set()
            out_nodes: Set[str] = set()

            # Qubit inputs and outputs
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

            # Measurement handling — FIXED
            if name == "measure":
                for i, qubit in enumerate(qargs):
                    qname = f"q{qubit}"
                    in_id = f"{qname}_t{qubit_time[qubit]}"
                    in_nodes.add(in_id)

                    # Look up latest time this qubit was involved in any edge
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

            # Classical outputs (non-measure)
            for clbit in cargs:
                if name != "measure":
                    t = clbit_time.get(clbit, 0)
                    cname = f"c{clbit}"
                    out_id = f"{cname}_t{t + 1}"
                    hdh.add_node(out_id, "c", t + 1)
                    out_nodes.add(out_id)
                    clbit_time[clbit] = t + 2

            # Classify edge type
            edge_nodes = in_nodes.union(out_nodes)
            if all(n.startswith("c") for n in edge_nodes):
                edge_type = "c"
            elif any(n.startswith("c") for n in edge_nodes):
                edge_type = "c"
            else:
                edge_type = "q"

            hdh.add_hyperedge(edge_nodes, edge_type, name=name)
            edge = hdh.add_hyperedge(edge_nodes, edge_type, name=name)
            q_with_time = [(q, qubit_time[q]) for q in qargs]
            c_with_time = [(c, clbit_time.get(c, 0)) for c in cargs]
            hdh.edge_args[edge] = (q_with_time, c_with_time, modifies_flags)

        return hdh

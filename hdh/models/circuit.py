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
            modifies_flags = [True] * len(qubits)
        else:
            bits = bits or []
            modifies_flags = modifies_flags or [True] * len(qubits)

        self.instructions.append((name, qubits, bits, modifies_flags, cond_flag))

    def build_hdh(self, hdh_cls=HDH) -> HDH:
        hdh = hdh_cls()
        qubit_time: Dict[int, int] = {}
        bit_time: Dict[int, int] = {}
        last_gate_input_time: Dict[int, int] = {}

        for name, qargs, cargs, modifies_flags, cond_flag in self.instructions:
            # --- Canonicalize inputs ---
            qargs = list(qargs or [])
            if name == "measure":
                cargs = list(cargs) if cargs is not None else qargs.copy()  # 1:1 map
                if len(cargs) != len(qargs):
                    raise ValueError("measure: len(bits) must equal len(qubits)")
                modifies_flags = [True] * len(qargs)
            else:
                cargs = list(cargs or [])
                if modifies_flags is None:
                    modifies_flags = [True] * len(qargs)
                elif len(modifies_flags) != len(qargs):
                    raise ValueError("len(modifies_flags) must equal len(qubits)")
            
            #print(f"\n=== Adding instruction: {name} on qubits {qargs} ===")
            # for q in qargs:
                #print(f"  [before] qubit_time[{q}] = {qubit_time.get(q)}")
                
            # Measurements
            if name == "measure":
                for i, qubit in enumerate(qargs):
                    # Use current qubit time (default 0), do NOT advance it here
                    t_in = qubit_time.get(qubit, 0)
                    q_in = f"q{qubit}_t{t_in}"
                    hdh.add_node(q_in, "q", t_in, node_real=cond_flag)

                    bit = cargs[i]
                    t_out = t_in + 1              # classical result at next tick
                    c_out = f"c{bit}_t{t_out}"
                    hdh.add_node(c_out, "c", t_out, node_real=cond_flag)

                    hdh.add_hyperedge({q_in, c_out}, "c", name="measure", node_real=cond_flag)

                    # Next-free convention for this bit stream
                    bit_time[bit] = t_out + 1

                    # Important: do NOT set qubit_time[qubit] = t_in + k
                    # The quantum wire collapses; keep its last quantum tick unchanged.
                continue
            
            # Conditional gate handling
            if name != "measure" and cond_flag == "p" and cargs:
                # Supports 1 classical control; extend to many if you like
                ctrl = cargs[0]

                # Ensure times exist
                for q in qargs:
                    if q not in qubit_time:
                        qubit_time[q] = max(qubit_time.values(), default=0)
                        last_gate_input_time[q] = qubit_time[q]

                # Classical node must already exist (e.g., produced by a prior measure)
                # bit_time points to "next free" slot; the latest existing node is at t = bit_time-1
                c_latest = bit_time.get(ctrl, 1) - 1
                cnode = f"c{ctrl}_t{c_latest}"
                hdh.add_node(cnode, "c", c_latest, node_real=cond_flag)

                edges = []
                for tq in qargs:
                    # gate happens at next tick after both inputs are ready
                    t_in_q = qubit_time[tq]
                    t_gate = max(t_in_q, c_latest) + 1
                    qname = f"q{tq}"
                    qout = f"{qname}_t{t_gate}"

                    # ensure the quantum output node exists at gate time
                    hdh.add_node(qout, "q", t_gate, node_real=cond_flag)

                    # add classical hyperedge feeding the quantum node
                    e = hdh.add_hyperedge({cnode, qout}, "c", name=name, node_real=cond_flag)
                    edges.append(e)

                    # advance quantum time
                    last_gate_input_time[tq] = t_in_q
                    qubit_time[tq] = t_gate

                # store edge_args for reconstruction/debug
                q_with_time = [(q, qubit_time[q]) for q in qargs]
                c_with_time = [(ctrl, c_latest + 1)]  # next-free convention; adjust if you track exact
                for e in edges:
                    hdh.edge_args[e] = (q_with_time, c_with_time, modifies_flags or [True] * len(qargs))

                continue

            #Actualized gate (non-conditional)
            for q in qargs:
                if q not in qubit_time:
                    qubit_time[q] = max(qubit_time.values(), default=0)
                    last_gate_input_time[q] = qubit_time[q]  # initial input time

            active_times = [qubit_time[q] for q in qargs]
            time_step = max(active_times) + 1 if active_times else 0

            in_nodes: List[str] = []
            out_nodes: List[str] = []

            intermediate_nodes: List[str] = []
            final_nodes: List[str] = []
            post_nodes: List[str] = []
            
            #DEBUG
            #print("  [after]", {q: qubit_time[q] for q in qargs})
            #print("  [after]", {q: qubit_time[q] for q in qargs})

            multi_gate = (name != "measure" and len(qargs) > 1)
            common_start = max((qubit_time.get(q, 0) for q in qargs), default=0) if multi_gate else None

            for i, qubit in enumerate(qargs):
                t_in = qubit_time[qubit]
                qname = f"q{qubit}"
                in_id = f"{qname}_t{t_in}"
                hdh.add_node(in_id, "q", t_in, node_real=cond_flag)
                #print(f"    [+] Node added: {in_id} (type q, time {t_in})")
                #print(f"    [+] Node added: {in_id} (type q, time {t_in})")
                in_nodes.append(in_id)
                #print(f"    [qubit {qubit}] modifies_flag = {modifies_flags[i]}")
                #print(f"    [qubit {qubit}] modifies_flag = {modifies_flags[i]}")

                # choose timeline
                if multi_gate:
                    t1 = common_start + 1
                    t2 = common_start + 2
                    t3 = common_start + 3
                else:
                    t1 = t_in + 1
                    t2 = t1 + 1
                    t3 = t2 + 1

                # create mid/final/post nodes for BOTH cases
                mid_id   = f"{qname}_t{t1}"
                final_id = f"{qname}_t{t2}"
                post_id  = f"{qname}_t{t3}"

                hdh.add_node(mid_id,   "q", t1, node_real=cond_flag)
                hdh.add_node(final_id, "q", t2, node_real=cond_flag)
                hdh.add_node(post_id,  "q", t3, node_real=cond_flag)

                intermediate_nodes.append(mid_id)
                final_nodes.append(final_id)
                post_nodes.append(post_id)

                last_gate_input_time[qubit] = t_in
                qubit_time[qubit] = t3

            edges = []
            if len(qargs) > 1:
                # Stage 1: input → intermediate (1:1)
                for in_node, mid_node in zip(in_nodes, intermediate_nodes):
                    e = hdh.add_hyperedge({in_node, mid_node}, "q", name=f"{name}_stage1", node_real=cond_flag)
                    #print(f"    [~] stage1 {in_node} → {mid_node}")
                    #print(f"    [~] stage1 {in_node} → {mid_node}")
                    edges.append(e)

                # Stage 2: full multiqubit edge from intermediate → final
                e2 = hdh.add_hyperedge(set(intermediate_nodes) | set(final_nodes), "q", name=f"{name}_stage2", node_real=cond_flag)
                #print(f"    [~] stage2 |intermediate|={len(intermediate_nodes)} → |final|={len(final_nodes)}")
                #print(f"    [~] stage2 |intermediate|={len(intermediate_nodes)} → |final|={len(final_nodes)}")
                edges.append(e2)

                # Stage 3: final → post (1:1)
                for f_node, p_node in zip(final_nodes, post_nodes):
                    e = hdh.add_hyperedge({f_node, p_node}, "q", name=f"{name}_stage3", node_real=cond_flag)
                    #print(f"    [~] stage3 {f_node} → {p_node}")
                    #print(f"    [~] stage3 {f_node} → {p_node}")
                    edges.append(e)

            if name == "measure":
                for i, qubit in enumerate(qargs):
                    t_in = qubit_time.get(qubit, 0)
                    q_in = f"q{qubit}_t{t_in}"
                    hdh.add_node(q_in, "q", t_in, node_real=cond_flag)

                    bit = cargs[i]
                    t_out = t_in + 1
                    c_out = f"c{bit}_t{t_out}"
                    hdh.add_node(c_out, "c", t_out, node_real=cond_flag)

                    hdh.add_hyperedge({q_in, c_out}, "c", name="measure", node_real=cond_flag)
                    bit_time[bit] = t_out + 1
                continue

            if name != "measure":
                for bit in cargs:
                    t = bit_time.get(bit, 0)
                    cname = f"c{bit}"
                    out_id = f"{cname}_t{t + 1}"
                    hdh.add_node(out_id, "c", t + 1, node_real=cond_flag)
                    out_nodes.append(out_id)
                    bit_time[bit] = t + 1

            all_nodes = set(in_nodes) | set(out_nodes)
            if all(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            elif any(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            else:
                edge_type = "q"

            edges = []

            if len(qargs) > 1:
                # Multi-qubit gate 
                # Stage 1: input → intermediate (1:1)
                for in_node, mid_node in zip(in_nodes, intermediate_nodes):
                    edge = hdh.add_hyperedge({in_node, mid_node}, "q", name=f"{name}_stage1", node_real=cond_flag)
                    # DEBUG
                    #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage1")
                    #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage1")
                    edges.append(edge)

                # Stage 2: full multiqubit edge from intermediate → final
                edge2 = hdh.add_hyperedge(set(intermediate_nodes) | set(final_nodes), "q", name=f"{name}_stage2", node_real=cond_flag)
                # DEBUG
                #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage2")
                #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage2")
                edges.append(edge2)

                # Stage 3: final → post (1:1 again)
                for final_node, post_node in zip(final_nodes, post_nodes):
                    edge = hdh.add_hyperedge({final_node, post_node}, "q", name=f"{name}_stage3", node_real=cond_flag)
                    # DEBUG
                    #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage1")
                    #print(f"    [~] Hyperedge added over: {in_node} → {mid_node}, label: {name}_stage1")
                    edges.append(edge)
                                                    
            else:
                # Single-qubit gate
                for i, qubit in enumerate(qargs):
                    
                    if modifies_flags[i] and name != "measure":
                        t_in = last_gate_input_time[qubit]
                        t_out = t_in + 1
                        qname = f"q{qubit}"
                        in_id = f"{qname}_t{t_in}"
                        out_id = f"{qname}_t{t_out}"
                        # DEBUG
                        #print(f"[{name}] Q{qubit} t_in = {t_in}, expected from qubit_time = {qubit_time[qubit]}")
                        #print(f"[{name}] Q{qubit} t_in = {t_in}, expected from qubit_time = {qubit_time[qubit]}")
                        hdh.add_node(out_id, "q", t_out, node_real=cond_flag)
                        # DEBUG
                        #print(f"    [+] Node added: {in_id} (type q, time {t_in})")
                        #print(f"    [+] Node added: {in_id} (type q, time {t_in})")
                        edge = hdh.add_hyperedge({in_id, out_id}, "q", name=name, node_real=cond_flag)
                        # DEBUG
                        #print(f"    [~] Hyperedge added over: {in_id} → {out_id}, label: {name}_stage1")
                        #print(f"    [~] Hyperedge added over: {in_id} → {out_id}, label: {name}_stage1")
                        edges.append(edge)
                        # Update time
                        qubit_time[qubit] = t_out
                        last_gate_input_time[qubit] = t_in

            q_with_time = [(q, qubit_time[q]) for q in qargs]
            c_with_time = [(c, bit_time.get(c, 0)) for c in cargs]
            for edge in edges:
                hdh.edge_args[edge] = (q_with_time, c_with_time, modifies_flags)

        return hdh

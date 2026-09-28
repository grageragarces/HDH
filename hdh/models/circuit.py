from typing import List, Tuple, Optional, Set, Dict, Literal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

class Circuit:
    """Gate-model circuit builder: record gates in order, then call `build_hdh`.

    Example:
        >>> c = Circuit()
        >>> c.add_instruction("h", [0])
        >>> c.add_instruction("cx", [0, 1])
        >>> c.add_instruction("measure", [1])
        >>> hdh = c.build_hdh()
        >>> hdh.get_num_qubits()
        2
        >>> sorted({hdh.wire_of[n] for n in hdh.nodes})
        ['c1', 'q0', 'q1']
    """

    def __init__(self):
        self.instructions: List[
            Tuple[str, List[int], List[int], List[bool], Literal["a", "p"], Optional[List[float]]]
        ] = []  # (name, qubits, bits, modifies_flags, cond_flag, params)

    def add_instruction(
        self,
        name: str,
        qubits: List[int],
        bits: Optional[List[int]] = None,
        modifies_flags: Optional[List[bool]] = None,
        cond_flag: Literal["a", "p"] = "a",
        params: Optional[List[float]] = None,
    ):
        """Append a gate or measurement.

        Args:
            name: Gate name, e.g. ``"h"``, ``"cx"``, ``"rx"`` or ``"measure"``.
            qubits: Qubit indices, in order.
            bits: Classical bits. A measurement defaults to ``bits = qubits``.
            modifies_flags: Per qubit, whether the gate changes its state.
                Defaults to all True.
            cond_flag: ``"p"`` if the gate only happens when a classical
                condition holds (prefer `add_conditional_gate`), else ``"a"``.
            params: Gate parameters such as ``[theta]`` for ``"rx"``; kept on the
                HDH so converters can restore them.

        Example:
            >>> c = Circuit()
            >>> c.add_instruction("rx", [0], params=[0.5])
            >>> hdh = c.build_hdh()
            >>> list(hdh.gate_params.values())
            [[0.5]]
        """
        name = name.lower()

        if name == "measure":
            modifies_flags = [True] * len(qubits)
        else:
            bits = bits or []
            modifies_flags = modifies_flags or [True] * len(qubits)

        self.instructions.append((name, qubits, bits, modifies_flags, cond_flag, params))

    def add_conditional_gate(
        self,
        classical_bit: int,
        target_qubit: int,
        gate_name: str,
        additional_qubits: Optional[List[int]] = None,
        modifies_flags: Optional[List[bool]] = None,
        params: Optional[List[float]] = None,
    ):
        """Append a gate that only applies when `classical_bit` is 1.

        The bit must already hold a value, typically from an earlier
        ``"measure"``. The gate's output states are marked predicted (``"p"``).

        Args:
            classical_bit: The controlling classical bit.
            target_qubit: Qubit the gate acts on.
            gate_name: Gate name, as in `add_instruction`.
            additional_qubits: Further qubits, for a multi-qubit conditional gate.
            modifies_flags: As in `add_instruction`.
            params: As in `add_instruction`.

        Example:
            >>> c = Circuit()
            >>> c.add_instruction("h", [0])
            >>> c.add_instruction("measure", [0])
            >>> c.add_conditional_gate(0, 1, "x")
            >>> hdh = c.build_hdh()
            >>> sorted(n for n in hdh.nodes if hdh.node_realisation[n] == "p")
            ['q1_t3']
        """
        gate_name = gate_name.lower()

        # Build the qubit list
        if additional_qubits is None:
            qubits = [target_qubit]
        else:
            qubits = [target_qubit] + additional_qubits

        # Set default modifies_flags
        if modifies_flags is None:
            modifies_flags = [True] * len(qubits)

        # Add the instruction with cond_flag="p" (positive condition)
        # and the classical bit in the bits list
        self.add_instruction(
            name=gate_name,
            qubits=qubits,
            bits=[classical_bit],
            modifies_flags=modifies_flags,
            cond_flag="p",
            params=params,
        )

    def build_hdh(self, hdh_cls=HDH) -> HDH:
        """Translate the recorded instructions into an HDH.

        Qubit ``i`` and bit ``j`` become wires ``"q<i>"`` and ``"c<j>"``. A
        multi-qubit gate spans three timesteps as three hyperedge layers
        (``<name>_stage1``, ``_stage2``, ``_stage3``), so a partitioner can cut
        before, through or after it.

        Args:
            hdh_cls: HDH class to instantiate, for subclasses.
        """
        hdh = hdh_cls()
        qubit_time: Dict[int, int] = {}
        bit_time: Dict[int, int] = {}
        last_gate_input_time: Dict[int, int] = {}

        for name, qargs, cargs, modifies_flags, cond_flag, params in self.instructions:
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
            
            # Measurements
            if name == "measure":
                for i, qubit in enumerate(qargs):
                    # Use current qubit time (default 0), do NOT advance it here
                    t_in = qubit_time.get(qubit, 0)
                    q_in = f"q{qubit}_t{t_in}"
                    
                    # Check if node already exists - preserve its potential status
                    if q_in not in hdh.nodes:
                        hdh.add_node(f"q{qubit}", t_in, "q", node_real="a")  # Default to actual

                    bit = cargs[i]
                    t_out = t_in + 1              # classical result at next tick
                    c_out = f"c{bit}_t{t_out}"
                    # Classical output is always actual - measurement is unconditional
                    hdh.add_node(f"c{bit}", t_out, "c", node_real=cond_flag)

                    # Measurement hyperedge is always actual - the operation itself is unconditional
                    # (even if measuring a potential quantum state)
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
                        qubit_time[q] = 0  # ← Initialize at t=0
                        last_gate_input_time[q] = 0  # ← Initialize at t=0

                # Classical node must already exist (e.g., produced by a prior measure)
                # bit_time points to "next free" slot; the latest existing node is at t = bit_time-1
                c_latest = bit_time.get(ctrl, 1) - 1
                cnode = f"c{ctrl}_t{c_latest}"
                hdh.add_node(f"c{ctrl}", c_latest, "c", node_real="a")  # Classical node is actual

                edges = []
                for tq in qargs:
                    # gate happens at next tick after both inputs are ready
                    t_in_q = qubit_time[tq]
                    t_gate = max(t_in_q, c_latest) + 1
                    qname = f"q{tq}"
                    
                    # Create input quantum node (actual state before conditional)
                    qin = f"{qname}_t{t_in_q}"
                    hdh.add_node(qname, t_in_q, "q", node_real="a")
                    
                    # Create output quantum node (potential state after conditional)
                    qout = f"{qname}_t{t_gate}"
                    hdh.add_node(qname, t_gate, "q", node_real=cond_flag)

                    # Add quantum hyperedge for wire continuity (potential)
                    q_edge = hdh.add_hyperedge({qin, qout}, "q", name=name, node_real=cond_flag)
                    edges.append(q_edge)
                    
                    # Add classical hyperedge for conditional dependency (potential)
                    c_edge = hdh.add_hyperedge({cnode, qout}, "c", name=name, node_real=cond_flag)
                    edges.append(c_edge)

                    # advance time
                    last_gate_input_time[tq] = t_in_q
                    qubit_time[tq] = t_gate

                # store edge_args for reconstruction/debug
                q_with_time = [(q, qubit_time[q]) for q in qargs]
                c_with_time = [(ctrl, c_latest + 1)]  # next-free convention; adjust if you track exact
                for e in edges:
                    hdh.edge_args[e] = (q_with_time, c_with_time, modifies_flags or [True] * len(qargs))
                    if params:
                        hdh.gate_params[e] = params

                continue

            #Actualized gate (non-conditional)
            for q in qargs:
                if q not in qubit_time:
                    qubit_time[q] = 0  # ← Initialize at t=0
                    last_gate_input_time[q] = 0  # ← Initialize at t=0

            active_times = [qubit_time[q] for q in qargs]
            time_step = max(active_times) + 1 if active_times else 0

            in_nodes: List[str] = []
            out_nodes: List[str] = []

            intermediate_nodes: List[str] = []
            final_nodes: List[str] = []
            post_nodes: List[str] = []

            multi_gate = (name != "measure" and len(qargs) > 1)
            common_start = max((qubit_time.get(q, 0) for q in qargs), default=0) if multi_gate else None

            for i, qubit in enumerate(qargs):
                t_in = qubit_time[qubit]
                qname = f"q{qubit}"
                in_id = f"{qname}_t{t_in}"
                hdh.add_node(qname, t_in, "q", node_real=cond_flag)
                in_nodes.append(in_id)

                # choose timeline
                if multi_gate:
                    t1 = common_start + 1
                    t2 = common_start + 2
                    t3 = common_start + 3
                    
                    # FIX ISSUE #37: Create intermediate nodes INSIDE loop for each qubit
                    mid_id   = f"{qname}_t{t1}"
                    final_id = f"{qname}_t{t2}"
                    post_id  = f"{qname}_t{t3}"

                    hdh.add_node(qname, t1, "q", node_real=cond_flag)
                    hdh.add_node(qname, t2, "q", node_real=cond_flag)
                    hdh.add_node(qname, t3, "q", node_real=cond_flag)

                    intermediate_nodes.append(mid_id)
                    final_nodes.append(final_id)
                    post_nodes.append(post_id)
                    
                    last_gate_input_time[qubit] = t_in
                    qubit_time[qubit] = t3
                else:
                    # Single-qubit gates: don't create nodes here
                    # created by the single-qubit handler below
                    t1 = t_in + 1
                    t2 = t1 + 1
                    t3 = t2 + 1

            edges = []
            if len(qargs) > 1:
                # Multi-qubit gate
                # Stage 1: input → intermediate (1:1)
                for in_node, mid_node in zip(in_nodes, intermediate_nodes):
                    e = hdh.add_hyperedge({in_node, mid_node}, "q", name=f"{name}_stage1", node_real=cond_flag)
                    edges.append(e)

                # Stage 2: full multiqubit edge from intermediate → final
                e2 = hdh.add_hyperedge(set(intermediate_nodes) | set(final_nodes), "q", name=f"{name}_stage2", node_real=cond_flag)
                edges.append(e2)

                # Stage 3: final → post (1:1)
                for f_node, p_node in zip(final_nodes, post_nodes):
                    e = hdh.add_hyperedge({f_node, p_node}, "q", name=f"{name}_stage3", node_real=cond_flag)
                    edges.append(e)

            if name == "measure":
                for i, qubit in enumerate(qargs):
                    t_in = qubit_time.get(qubit, 0)
                    q_in = f"q{qubit}_t{t_in}"
                    hdh.add_node(f"q{qubit}", t_in, "q", node_real=cond_flag)

                    bit = cargs[i]
                    t_out = t_in + 1
                    c_out = f"c{bit}_t{t_out}"
                    hdh.add_node(f"c{bit}", t_out, "c", node_real=cond_flag)

                    hdh.add_hyperedge({q_in, c_out}, "c", name="measure", node_real=cond_flag)
                    bit_time[bit] = t_out + 1
                continue

            if name != "measure":
                for bit in cargs:
                    t = bit_time.get(bit, 0)
                    cname = f"c{bit}"
                    out_id = f"{cname}_t{t + 1}"
                    hdh.add_node(cname, t + 1, "c", node_real=cond_flag)
                    out_nodes.append(out_id)
                    bit_time[bit] = t + 1

            all_nodes = set(in_nodes) | set(out_nodes)
            if all(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            elif any(n.startswith("c") for n in all_nodes):
                edge_type = "c"
            else:
                edge_type = "q"

            if len(qargs) == 1:
                # Single-qubit gate
                for i, qubit in enumerate(qargs):
                    if modifies_flags[i] and name != "measure":
                        # Use current qubit_time
                        t_in = qubit_time[qubit]  
                        t_out = t_in + 1
                        qname = f"q{qubit}"
                        in_id = f"{qname}_t{t_in}"
                        out_id = f"{qname}_t{t_out}"
                        hdh.add_node(qname, t_out, "q", node_real=cond_flag)
                        edge = hdh.add_hyperedge({in_id, out_id}, "q", name=name, node_real=cond_flag)
                        edges.append(edge)
                        # Update time for next gate
                        qubit_time[qubit] = t_out
                        last_gate_input_time[qubit] = t_in

            q_with_time = [(q, qubit_time[q]) for q in qargs]
            c_with_time = [(c, bit_time.get(c, 0)) for c in cargs]
            for edge in edges:
                hdh.edge_args[edge] = (q_with_time, c_with_time, modifies_flags)
                if params:
                    hdh.gate_params[edge] = params

        return hdh
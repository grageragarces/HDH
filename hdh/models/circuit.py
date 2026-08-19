from typing import List, Tuple, Optional, Set, Dict, Literal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

class Circuit:
    """Gate-model quantum circuit builder that compiles down to an HDH.

    Instructions are recorded in order via ``add_instruction`` (and, for
    classically-conditioned gates, ``add_conditional_gate``), then translated
    into an HDH's nodes and hyperedges by ``build_hdh``. This mirrors how you'd
    build a circuit in Qiskit/Cirq/etc., but stores gates as a flat list rather
    than executing them immediately.

    Attributes:
        instructions: Recorded gates, one tuple per call to ``add_instruction``:
            ``(name, qubits, bits, modifies_flags, cond_flag, params)``. Built
            up by ``add_instruction``/``add_conditional_gate`` and consumed by
            ``build_hdh`` — not usually read or written directly.
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
        """Append one gate or measurement to the circuit.

        Args:
            name: Gate name (e.g. ``"h"``, ``"cx"``, ``"rx"``, ``"measure"``).
                Case-insensitive; stored lower-cased.
            qubits: Qubit indices the instruction acts on, in order.
            bits: Classical bit indices involved. For ``"measure"``, defaults
                to a 1:1 mapping with `qubits` if omitted; ignored for
                unconditional gates unless explicitly provided.
            modifies_flags: Per-qubit flag marking whether that qubit's state
                is actually changed by this instruction. Defaults to all
                `True`. Rarely needed directly — prefer `add_conditional_gate`
                for classically-controlled gates.
            cond_flag: `"a"` (actualized) for an unconditional instruction, or
                `"p"` (potential) for one whose effect depends on a classical
                condition not yet known at build time.
            params: Rotation angles / gate parameters (e.g. ``[theta]`` for an
                `rx` gate), if the gate is parametric. Stored alongside the
                instruction and later attached to the corresponding HDH
                hyperedge via `HDH.gate_params` so they survive a round trip
                through `build_hdh` and back to a circuit representation.
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
        """Append a gate whose application is conditioned on a classical bit.

        Convenience wrapper around `add_instruction` for the common
        single-classical-control case (e.g. a mid-circuit-measurement
        feed-forward gate): it sets `cond_flag="p"` and puts `classical_bit`
        in the instruction's `bits`, so the resulting HDH marks the gate's
        output as a *potential* (not yet actualized) state until that
        classical value is known. `classical_bit` must already have a value
        by the time this instruction executes — typically produced by an
        earlier `add_instruction("measure", ...)` call.

        Args:
            classical_bit: Index of the classical bit the gate is conditioned on.
            target_qubit: Primary qubit the gate acts on.
            gate_name: Gate name, as in `add_instruction`.
            additional_qubits: Extra qubits for a multi-qubit conditional gate,
                applied after `target_qubit`.
            modifies_flags: As in `add_instruction`; defaults to all `True`.
            params: Rotation angles / gate parameters, as in `add_instruction`.
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

        Each qubit/bit gets one node per timestep it's touched, named
        ``q{idx}_t{time}`` / ``c{idx}_t{time}``. Single-qubit gates add one
        hyperedge connecting a qubit's input and output node at consecutive
        timesteps.

        Multi-qubit gates are deliberately spread across *three* timesteps
        per gate rather than one, via three hyperedges suffixed
        ``_stage1``/``_stage2``/``_stage3``: stage 1 and 3 are per-qubit wire
        continuity (input->intermediate, final->post), and stage 2 is the
        single hyperedge spanning every involved qubit's intermediate and
        final nodes. This is intentional — it's what lets the HDH represent
        pre- and post-gate teleportation as separate cuttable edges rather
        than only a single all-or-nothing gate boundary — so a multi-qubit
        gate's apparent "depth" in `time_map` is 3 ticks even though it's one
        logical operation.

        Args:
            hdh_cls: HDH class to instantiate (override for a subclass).

        Returns:
            HDH: the built hypergraph, with `S`/`C`/`sigma`/`tau`/`time_map`
            and friends populated. `edge_args` and `gate_params` are also
            populated per gate, letting `hdh.converters.qiskit_converter.to_qiskit`
            (and similar) reconstruct a circuit from it.
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
                    if q_in not in hdh.S:
                        hdh.add_node(q_in, "q", t_in, node_real="a")  # Default to actual

                    bit = cargs[i]
                    t_out = t_in + 1              # classical result at next tick
                    c_out = f"c{bit}_t{t_out}"
                    # Classical output is always actual - measurement is unconditional
                    hdh.add_node(c_out, "c", t_out, node_real=cond_flag)

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
                hdh.add_node(cnode, "c", c_latest, node_real="a")  # Classical node is actual

                edges = []
                for tq in qargs:
                    # gate happens at next tick after both inputs are ready
                    t_in_q = qubit_time[tq]
                    t_gate = max(t_in_q, c_latest) + 1
                    qname = f"q{tq}"
                    
                    # Create input quantum node (actual state before conditional)
                    qin = f"{qname}_t{t_in_q}"
                    hdh.add_node(qin, "q", t_in_q, node_real="a")
                    
                    # Create output quantum node (potential state after conditional)
                    qout = f"{qname}_t{t_gate}"
                    hdh.add_node(qout, "q", t_gate, node_real=cond_flag)

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
                hdh.add_node(in_id, "q", t_in, node_real=cond_flag)
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

                    hdh.add_node(mid_id,   "q", t1, node_real=cond_flag)
                    hdh.add_node(final_id, "q", t2, node_real=cond_flag)
                    hdh.add_node(post_id,  "q", t3, node_real=cond_flag)

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
                        hdh.add_node(out_id, "q", t_out, node_real=cond_flag)
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
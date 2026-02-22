"""
hdh_to_qiskit.py
================
Reversal pass: HDH → Qiskit QuantumCircuit.

This is the inverse of the  from_qiskit → Circuit.build_hdh()  pipeline and
is intended for post-partitioning analysis — e.g. inspecting sub-circuits
produced by kahypar_cutter, measuring gate counts, or feeding results into
other Qiskit tooling.

Pipeline position
-----------------
    QuantumCircuit  ──from_qiskit──►  HDH
                                       │  (partition / analysis)
    QuantumCircuit  ◄──hdh_to_qiskit── HDH

Why a new file instead of patching qiskit_converter.to_qiskit
--------------------------------------------------------------
The existing to_qiskit() relies on edge_metadata (never populated by
build_hdh) and expects edge_args to hold Qiskit Instruction objects, when
it actually holds (q_with_time, c_with_time, modifies_flags) tuples.
Rather than patching the old function in-place (which risks breaking callers
that depend on its current signature), this module provides a clean,
well-documented replacement.

Known limitations
-----------------
• Gate parameters (θ for rx/ry/rz etc.) are not preserved in the HDH
  representation — add_instruction() and build_hdh() discard them at
  ingestion time.  Parametric gates are reconstructed with θ=0 and a
  warning is emitted.
• IfElseOp false-bodies (cond_flag="n") are not yet handled; they will
  be skipped with a warning.
• Multi-qubit gates above 3 qubits (CCX is the highest in the default
  gate map) will be skipped with a warning unless you extend GATE_FACTORY.
"""

from __future__ import annotations

import re
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    HGate, IGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    RXGate, RYGate, RZGate, U1Gate, U2Gate, U3Gate,
    CXGate, CYGate, CZGate, CHGate,
    SwapGate, ISwapGate,
    CCXGate, CSGate, CSwapGate,
    PhaseGate, CPhaseGate,
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# ──────────────────────────────────────────────────────────────────────────────
#  Gate factory
#  Maps lowercase gate name (as stored in HDH gate_name) → callable that
#  returns a Qiskit Gate instance.
#  Parametric gates receive params=[] when none were stored; θ defaults to 0.
# ──────────────────────────────────────────────────────────────────────────────

def _make_gate(name: str, params: list):
    """
    Construct a Qiskit gate from its name and (possibly empty) parameter list.
    Returns None if the gate name is not recognised.
    """
    p = params  # shorthand

    _FIXED = {
        # ── 1-qubit ──────────────────────────────────────────────────────────
        "h":    HGate(),
        "i":    IGate(),
        "id":   IGate(),
        "x":    XGate(),
        "y":    YGate(),
        "z":    ZGate(),
        "s":    SGate(),
        "sdg":  SdgGate(),
        "t":    TGate(),
        "tdg":  TdgGate(),
        "sx":   SXGate(),
        "sxdg": SXdgGate(),
        # ── 2-qubit ──────────────────────────────────────────────────────────
        "cx":    CXGate(),
        "cnot":  CXGate(),
        "cy":    CYGate(),
        "cz":    CZGate(),
        "ch":    CHGate(),
        "cs":    CSGate(),
        "swap":  SwapGate(),
        "iswap": ISwapGate(),
        # ── 3-qubit ──────────────────────────────────────────────────────────
        "ccx":   CCXGate(),
        "toffoli": CCXGate(),
        "cswap": CSwapGate(),
        "fredkin": CSwapGate(),
    }

    if name in _FIXED:
        return _FIXED[name]

    # ── Parametric ────────────────────────────────────────────────────────────
    theta = p[0] if p else 0.0
    phi   = p[1] if len(p) > 1 else 0.0
    lam   = p[2] if len(p) > 2 else 0.0

    _PARAMETRIC = {
        "rx":     RXGate(theta),
        "ry":     RYGate(theta),
        "rz":     RZGate(theta),
        "p":      PhaseGate(theta),
        "phase":  PhaseGate(theta),
        "u1":     U1Gate(lam),
        "u2":     U2Gate(phi, lam),
        "u3":     U3Gate(theta, phi, lam),
        "u":      U3Gate(theta, phi, lam),
        "cp":     CPhaseGate(theta),
        "cphase": CPhaseGate(theta),
    }

    if name in _PARAMETRIC:
        if not p:
            print(
                f"[WARNING] Gate '{name}' is parametric but no parameters were "
                f"stored in the HDH (they are discarded at ingestion time). "
                f"Defaulting to θ=φ=λ=0.  Reconstruct manually if the exact "
                f"angle matters."
            )
        return _PARAMETRIC[name]

    return None  # unknown


# ──────────────────────────────────────────────────────────────────────────────
#  Node-ID helpers
# ──────────────────────────────────────────────────────────────────────────────

_Q_RE = re.compile(r'^q(\d+)_t\d+$')
_C_RE = re.compile(r'^c(\d+)_t\d+$')


def _parse_qubit(node_id: str) -> Optional[int]:
    m = _Q_RE.match(node_id)
    return int(m.group(1)) if m else None


def _parse_cbit(node_id: str) -> Optional[int]:
    m = _C_RE.match(node_id)
    return int(m.group(1)) if m else None


# ──────────────────────────────────────────────────────────────────────────────
#  Main reconstruction function
# ──────────────────────────────────────────────────────────────────────────────

def hdh_to_qiskit(hdh: HDH) -> QuantumCircuit:
    """
    Reconstruct a Qiskit QuantumCircuit from an HDH object.

    Works with the HDH produced by  Circuit.build_hdh()  (and therefore by
    from_qiskit / from_qasm).  All qubit/bit indices and gate ordering are
    recovered from  edge_args  and  gate_name  stored during ingestion.

    Edge filtering strategy
    -----------------------
    build_hdh() stores multi-qubit gates as three layers of hyperedges:
      _stage1  — individual input→intermediate wires  (skip)
      _stage2  — the actual multi-qubit operation      (keep)
      _stage3  — individual final→post wires           (skip)
    Conditional gates produce both a q-edge (the gate) and a c-edge (the
    classical dependency wire); only the q-edge is kept.
    Measure edges have no edge_args, so their qubit/bit indices are parsed
    directly from the node IDs in the hyperedge.

    Parameters
    ----------
    hdh : HDH
        An HDH object previously built by Circuit.build_hdh().

    Returns
    -------
    QuantumCircuit
        A flat Qiskit circuit replaying the operations in temporal order.
        Parametric angles default to 0 when not stored (see module docstring).
    """

    # ── 1. Determine circuit dimensions from node IDs ─────────────────────────
    qubit_indices: set[int] = set()
    bit_indices:   set[int] = set()

    for node_id in hdh.S:
        q = _parse_qubit(node_id)
        if q is not None:
            qubit_indices.add(q)
            continue
        c = _parse_cbit(node_id)
        if c is not None:
            bit_indices.add(c)

    n_qubits = max(qubit_indices) + 1 if qubit_indices else 0
    n_bits   = max(bit_indices)   + 1 if bit_indices   else 0

    if n_qubits == 0:
        return QuantumCircuit()

    qc = QuantumCircuit(n_qubits, n_bits) if n_bits > 0 else QuantumCircuit(n_qubits)

    # ── 2. Collect one canonical record per gate from the hyperedge set ────────
    #
    # Record format: (sort_time, gate_name, q_indices, c_indices, is_conditional)
    #
    records: list[tuple] = []

    for edge in hdh.C:
        raw_name  = hdh.gate_name.get(edge, "")
        edge_type = hdh.tau.get(edge, "q")

        # ── Measure: no edge_args — parse directly from node IDs ──────────────
        if raw_name == "measure":
            q_nodes = sorted(
                (n for n in edge if hdh.sigma.get(n) == "q"),
                key=lambda n: hdh.time_map.get(n, 0),
            )
            c_nodes = sorted(
                (n for n in edge if hdh.sigma.get(n) == "c"),
                key=lambda n: hdh.time_map.get(n, 0),
            )
            if not q_nodes or not c_nodes:
                continue
            q_idx = _parse_qubit(q_nodes[0])
            c_idx = _parse_cbit(c_nodes[0])
            if q_idx is None or c_idx is None:
                continue
            sort_time = hdh.time_map.get(c_nodes[0], 0)
            records.append((sort_time, "measure", [q_idx], [c_idx], False))
            continue

        # ── Skip wiring stubs (stage1 / stage3 of multi-qubit gates) ─────────
        if raw_name.endswith("_stage1") or raw_name.endswith("_stage3"):
            continue

        # ── Skip classical dependency wires for conditional gates ─────────────
        #    These are "c"-type edges that share the same edge_args as the
        #    corresponding q-edge but represent the classical→quantum dependency,
        #    not the gate operation itself.
        if edge_type == "c":
            continue

        # ── Everything else needs edge_args ──────────────────────────────────
        args = hdh.edge_args.get(edge)
        if args is None:
            continue

        q_with_time, c_with_time, _ = args

        # Strip "_stage2" suffix to recover the user-facing gate name
        actual_name = raw_name[:-7] if raw_name.endswith("_stage2") else raw_name

        q_indices  = [q for q, _ in q_with_time]
        c_indices  = [c for c, _ in c_with_time]

        # Sort by the minimum post-gate qubit time (all stage2 qubits share
        # common_start+2, single-qubit gates use t_in+1 — both work for ordering)
        sort_time  = min((t for _, t in q_with_time), default=0)

        # Potential (p) realization means the edge is conditional
        is_cond = hdh.phi.get(edge) == "p"

        records.append((sort_time, actual_name, q_indices, c_indices, is_cond))

    # ── 3. Sort and de-duplicate ───────────────────────────────────────────────
    #
    # De-duplication is necessary because each qubit of a multi-qubit gate
    # independently stores edge_args pointing to the same (time, name, qubits)
    # tuple — the stage2 edge covers all qubits at once, but we guard anyway.
    records.sort(key=lambda r: r[0])

    seen: set[tuple] = set()
    unique_records: list[tuple] = []
    for rec in records:
        sort_time, name, q_indices, c_indices, is_cond = rec
        dedup_key = (sort_time, name, tuple(q_indices))
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_records.append(rec)

    # ── 4. Emit Qiskit operations ──────────────────────────────────────────────
    for sort_time, name, q_indices, c_indices, is_cond in unique_records:

        # Measure
        if name == "measure":
            for qi, ci in zip(q_indices, c_indices):
                qc.measure(qi, ci)
            continue

        # Build gate object (params are not preserved — defaults to 0)
        gate = _make_gate(name, params=[])
        if gate is None:
            print(
                f"[WARNING] Unrecognised gate '{name}' at t={sort_time} "
                f"on qubits {q_indices} — skipped.  "
                f"Extend hdh_to_qiskit.GATE_FACTORY to support it."
            )
            continue

        # Validate qubit indices are in range
        if any(qi >= n_qubits or qi < 0 for qi in q_indices):
            print(
                f"[WARNING] Gate '{name}' at t={sort_time} references "
                f"out-of-range qubits {q_indices} (circuit has {n_qubits}) — skipped."
            )
            continue

        # Validate expected qubit count
        if gate.num_qubits != len(q_indices):
            print(
                f"[WARNING] Gate '{name}' expects {gate.num_qubits} qubits "
                f"but HDH recorded {len(q_indices)} ({q_indices}) — skipped."
            )
            continue

        # Conditional (if_test) gate
        if is_cond and c_indices:
            ctrl_bit = c_indices[0]
            if ctrl_bit >= n_bits:
                print(
                    f"[WARNING] Conditional gate '{name}' references "
                    f"out-of-range classical bit {ctrl_bit} — applied unconditionally."
                )
                qc.append(gate, q_indices)
            else:
                with qc.if_test((qc.clbits[ctrl_bit], 1)):
                    qc.append(gate, q_indices)
        else:
            qc.append(gate, q_indices)

    return qc


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience: reconstruct one circuit per partition
# ──────────────────────────────────────────────────────────────────────────────

def partitions_to_qiskit(
    hdh: HDH,
    partitions: list[set[str]],
) -> list[QuantumCircuit]:
    """
    Given an HDH and a list of node-set partitions (as returned by
    kahypar_cutter or any other cut.py partitioner), build one Qiskit
    QuantumCircuit per partition.

    Each sub-circuit contains only the gates whose *all* participating
    qubits lie inside that partition.  Cross-partition gates are skipped
    (they would become teledata/telegate primitives in a full implementation).

    Parameters
    ----------
    hdh        : HDH produced by build_hdh()
    partitions : list of node-ID sets, one set per QPU partition

    Returns
    -------
    list of QuantumCircuit, indexed by partition number
    """
    sub_circuits = []

    for part_idx, node_set in enumerate(partitions):
        # Build a minimal HDH view restricted to this partition's nodes
        sub_hdh = _project_hdh(hdh, node_set)
        qc = hdh_to_qiskit(sub_hdh)
        sub_circuits.append(qc)

    return sub_circuits


def _project_hdh(hdh: HDH, node_set: set[str]) -> HDH:
    """
    Return a new HDH containing only the nodes in node_set and the edges
    whose *entire* node set is contained within node_set.

    This is used by partitions_to_qiskit to produce a local-only sub-HDH
    for each partition (cross-partition edges are excluded).
    """
    from hdh.hdh import HDH as _HDH  # local import to avoid circularity

    sub = _HDH()

    # Copy nodes
    for nid in node_set:
        if nid not in hdh.S:
            continue
        sub.add_node(
            nid,
            hdh.sigma[nid],
            hdh.time_map[nid],
            node_real=hdh.upsilon.get(nid, "a"),
        )

    # Copy edges whose every node is inside the partition
    for edge in hdh.C:
        if not edge.issubset(node_set):
            continue  # cross-partition — excluded
        name = hdh.gate_name.get(edge)
        role = hdh.edge_role.get(edge)
        sub.add_hyperedge(
            set(edge),
            hdh.tau[edge],
            name=name,
            node_real=hdh.phi.get(edge, "a"),
            role=role,
        )
        if edge in hdh.edge_args:
            sub.edge_args[edge] = hdh.edge_args[edge]

    return sub

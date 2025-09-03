from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH
from hdh.models.circuit import Circuit

import braket._sdk as braket
from braket.circuits import Circuit as BraketCircuit
from braket.circuits import Instruction as BraketInstruction

# ----- helpers -----

def _collect_qubits(bk: BraketCircuit) -> List[Any]:
    """Return a stable, sorted list of qubit labels used by the circuit."""
    qs = set()
    for instr in bk.instructions:
        for q in instr.target:
            qs.add(q)
    # Braket qubits are usually ints; sort by repr to be safe with Qubit objects
    return sorted(qs, key=lambda x: (str(type(x)), int(x) if isinstance(x, int) else str(x)))

def _qindex_map(bk: BraketCircuit) -> Dict[Any, int]:
    """Map Braket qubit labels to contiguous indices 0..n-1."""
    ordered = _collect_qubits(bk)
    return {q: i for i, q in enumerate(ordered)}

def _name_of(instr: BraketInstruction) -> str:
    """Lowercase operator name (e.g., 'x', 'cx', 'measure')."""
    op = instr.operator
    # gates.Gate has .name; noise.Noise has .name; Measure is a gate too
    n = getattr(op, "name", None)
    if n is None:
        # fallback to class name
        n = op.__class__.__name__
    return n.lower()

def _is_measure(instr: BraketInstruction) -> bool:
    return _name_of(instr) in {"measure", "m"}

def _is_ignorable(instr: BraketInstruction) -> bool:
    # No explicit barrier in Braket; keep hook for future skips
    return False

# ----- main -----

def from_braket(bk: BraketCircuit) -> HDH:
    """
    Convert an AWS Braket Circuit to HDH via hdh.models.circuit.Circuit.

    Supported:
    - Standard gates (all mapped by name)
    - Measurements (Measure)
    - Noise ops are treated as gates by name

    Not supported (raises NotImplementedError if encountered):
    - Classical conditionals / control flow blocks
    """
    qmap = _qindex_map(bk)
    circuit = Circuit()

    for instr in bk.instructions:
        if _is_ignorable(instr):
            continue

        name = _name_of(instr)
        q_indices = [qmap[q] for q in instr.target]

        # Basic check for classical control (Braket may wrap ops with conditions in newer APIs)
        if hasattr(instr, "condition") and instr.condition is not None:
            # Only implement if you have a bit-index mapping. Braket conditions are not 1-bit Clbits.
            raise NotImplementedError("Classical conditionals in Braket are not supported yet.")

        if _is_measure(instr):
            # Measurement is per qubit target
            circuit.add_instruction("measure", q_indices, None)
            continue

        # Treat any other op (gate or noise) as a modifying quantum op
        modifies_flags = [True] * len(q_indices)
        circuit.add_instruction(name, q_indices, [], modifies_flags)

    return circuit.build_hdh()

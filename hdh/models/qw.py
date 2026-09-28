from typing import List, Tuple, Set, Dict
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Quantum Walks (QW) model
class QW:
    """Discrete-time quantum walk builder: coin and shift steps, then measurement.

    `add_coin` and `add_shift` return the label of the new walker state, to
    pass to the next step.

    Example:
        >>> w = QW()
        >>> state = w.add_shift(w.add_coin("q0"))
        >>> w.add_measurement(state, "m0")
        >>> sorted(w.build_hdh().nodes)
        ['m0_t3', 'q0_t0', 'q1_t1', 'q2_t2']
    """

    def __init__(self, hdh_cls=HDH):
        self.steps = []  # (type, a, b)
        self.hdh_cls = hdh_cls
        self.qubit_counter = 0  # For auto-generating digit-only qubit IDs

    def _new_qubit_id(self):
        """Generate the next auto-incrementing walker-state label."""
        self.qubit_counter += 1
        return f"q{self.qubit_counter}"

    def add_coin(self, a: str):
        """Apply a coin operation to walker state `a`; returns the new state label."""
        a_prime = self._new_qubit_id()
        self.steps.append(("K", a, a_prime))
        return a_prime

    def add_shift(self, a_prime: str):
        """Apply a shift operation to walker state `a_prime`; returns the new state label."""
        b = self._new_qubit_id()
        self.steps.append(("R", a_prime, b))
        return b

    def add_measurement(self, a: str, b: str):
        """Measure walker state `a` into classical label `b`, which must not already
        be used for a quantum state.
        """
        self.steps.append(("M", a, b))

    def build_hdh(self) -> HDH:
        """Translate the recorded coin/shift/measurement steps into an HDH."""
        hdh = self.hdh_cls()
        time_map: Dict[str, int] = {}
        
        for step_index, (op_type, a, b) in enumerate(self.steps):
            in_time = time_map.get(a, 0)
            out_time = in_time + 1

            in_type = "q"
            out_type = "q" if op_type in {"K", "R"} else "c"
            edge_type = "q" if op_type in {"K", "R"} else "c"

            in_id = hdh.add_node(a, in_time, in_type)
            out_id = hdh.add_node(b, out_time, out_type)
            hdh.add_hyperedge({in_id, out_id}, edge_type, name=op_type.lower())

            time_map[b] = out_time  # set output time
        return hdh

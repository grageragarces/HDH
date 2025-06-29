from typing import List, Tuple, Set, Dict
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Quantum Walks (QW) model
class QW:
    def __init__(self, hdh_cls=HDH):
        self.steps = []  # (type, a, b)
        self.hdh_cls = hdh_cls
        self.qubit_counter = 0  # For auto-generating digit-only qubit IDs

    def _new_qubit_id(self):
        self.qubit_counter += 1
        return f"q{self.qubit_counter}"

    def add_coin(self, a: str):
        a_prime = self._new_qubit_id()
        self.steps.append(("K", a, a_prime))
        return a_prime

    def add_shift(self, a_prime: str):
        b = self._new_qubit_id()
        self.steps.append(("R", a_prime, b))
        return b

    def add_measurement(self, a: str, b: str):
        self.steps.append(("M", a, b))

    def build_hdh(self) -> HDH:
        hdh = self.hdh_cls()
        time_map: Dict[str, int] = {}
        
        for step_index, (op_type, a, b) in enumerate(self.steps):
            in_time = time_map.get(a, 0)
            out_time = in_time + 1

            in_id = f"{a}_t{in_time}"
            out_id = f"{b}_t{out_time}"

            in_type = "q"
            out_type = "q" if op_type in {"K", "R"} else "c"
            edge_type = "q" if op_type in {"K", "R"} else "c"

            hdh.add_node(in_id, in_type, in_time)
            hdh.add_node(out_id, out_type, out_time)
            hdh.add_hyperedge({in_id, out_id}, edge_type, name=op_type.lower())

            time_map[b] = out_time  # set output time
        return hdh

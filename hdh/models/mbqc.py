from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.hdh import HDH

# Measurement Based Quantum Computing (MBQC) model 

class MBQC:
    def __init__(self, hdh_cls=HDH):
        self.pattern = []  # (op_type, A, b)
        self.hdh_cls = hdh_cls

    def add_operation(self, op_type: str, A: List[str], b: str):
        self.pattern.append((op_type.upper(), A, b))

    def build_hdh(self) -> HDH:
        hdh = self.hdh_cls()
        time_map = {}
        current_time = 0

        for op_type, A, b in self.pattern:
            in_nodes = set()
            out_nodes = set()
            all_nodes = A + [b]

            # Assign time steps
            op_time = current_time
            current_time += 1

            for x in A:
                t = time_map.get(x, 0)
                hdh.add_node(f"{x}_t{t}", self._node_type(op_type, input=True), t)
                in_nodes.add(f"{x}_t{t}")

            hdh.add_node(f"{b}_t{op_time}", self._node_type(op_type, input=False), op_time)
            out_nodes.add(f"{b}_t{op_time}")
            time_map[b] = op_time

            edge_nodes = in_nodes | out_nodes
            hdh.add_hyperedge(edge_nodes, self._edge_type(op_type), name=op_type.lower())

        return hdh

    def _node_type(self, op_type, input=False):
        if op_type == "N":
            return "c" if input else "q"
        if op_type == "E":
            return "q"
        if op_type == "M":
            return "q" if input else "c"
        if op_type == "C":
            return "c"

    def _edge_type(self, op_type):
        return "q" if op_type == "E" else "c"

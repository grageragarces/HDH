from collections import defaultdict
from typing import Dict, Set, Tuple, List, Literal, Union, Optional  

NodeType = Literal["q", "c"]  # quantum or classical
EdgeType = Literal["q", "c"]
NodeID = str
TimeStep = int

class HDH:
    def __init__(self):
        self.S: Set[NodeID] = set()
        self.C: Set[frozenset] = set()
        self.T: Set[TimeStep] = set()
        self.sigma: Dict[NodeID, NodeType] = {}  # node types
        self.tau: Dict[frozenset, EdgeType] = {}  # hyperedge types
        self.time_map: Dict[NodeID, TimeStep] = {}  # f: S -> T
        self.gate_name: Dict[frozenset, str] = {}  # maps hyperedge → gate name string
        self.edge_args: Dict[frozenset, Tuple[List[int], List[int], List[bool]]] = {} #mapping for nackwards translations

    def add_node(self, node_id: NodeID, node_type: NodeType, time: TimeStep):
        self.S.add(node_id)
        self.sigma[node_id] = node_type
        self.time_map[node_id] = time
        self.T.add(time)

    def add_hyperedge(self, node_ids: Set[NodeID], edge_type: EdgeType, name: Optional[str] = None):
        edge = frozenset(node_ids)
        self.C.add(edge)
        self.tau[edge] = edge_type
        if name:
            self.gate_name[edge] = name.lower()  # ensures 'CX' becomes 'cx', etc.
        return edge

    def get_ancestry(self, node: NodeID) -> Set[NodeID]:
        """Return nodes with paths ending at `node` and earlier time steps."""
        return {
            s for s in self.S
            if self.time_map[s] <= self.time_map[node] and self._path_exists(s, node)
        }

    def get_lineage(self, node: NodeID) -> Set[NodeID]:
        """Return nodes reachable from `node` with later time steps."""
        return {
            s for s in self.S
            if self.time_map[s] >= self.time_map[node] and self._path_exists(node, s)
        }

    def _path_exists(self, start: NodeID, end: NodeID) -> bool:
        """DFS to find a time-respecting path from `start` to `end`."""
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            visited.add(current)
            neighbors = {
                neighbor
                for edge in self.C if current in edge
                for neighbor in edge
                if neighbor != current and self.time_map[neighbor] > self.time_map[current]
            }
            stack.extend(neighbors - visited)
        return False

    def get_num_qubits(self) -> int:
        """Returns the total number of distinct qubit indices in the circuit."""
        qubit_indices = {
            int(node_id.split("_")[0][1:])
            for node_id in self.S
            if self.sigma[node_id] == 'q'
        }
        return len(qubit_indices)

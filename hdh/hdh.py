from collections import defaultdict
from typing import Dict, Set, Tuple, List, Literal, Union, Optional  

NodeType = Literal["q", "c"]  # quantum or classical
EdgeType = Literal["q", "c"]
NodeReal = Literal["a", "p"]  # actualized or predicted
EdgeReal = Literal["a", "p"]
NodeID = str
# need an edge ID? - to map back?
TimeStep = int

class HDH:
    def __init__(self):
        self.S: Set[NodeID] = set()
        self.C: Set[frozenset] = set()
        self.T: Set[TimeStep] = set()
        self.sigma: Dict[NodeID, NodeType] = {}  # node types 
        self.tau: Dict[frozenset, EdgeType] = {}  # hyperedge types
        self.upsilon: Dict[NodeID, NodeReal] = {} # node realization a,p
        self.phi: Dict[frozenset, EdgeReal] = {} # hyperedge realization 
        self.time_map: Dict[NodeID, TimeStep] = {}  # f: S -> T
        self.gate_name: Dict[frozenset, str] = {}  # maps hyperedge → gate name string
        self.edge_args: Dict[frozenset, Tuple[List[int], List[int], List[bool]]] = {} #mapping for nackwards translations
        self.edge_role: Dict[frozenset, Literal["teledata", "telegate"]] = {}  # tracks nature edges -> for primitive implementation
        self.motifs = {}  
        self.edge_metadata: Dict[frozenset, Dict] = {}

    def add_node(self, node_id: NodeID, node_type: NodeType, time: TimeStep, node_real: NodeReal = "a"):
        self.S.add(node_id)
        self.sigma[node_id] = node_type
        self.time_map[node_id] = time
        self.T.add(time)
        self.upsilon[node_id] = node_real

    def add_hyperedge(self, node_ids: Set[NodeID], edge_type: EdgeType, name: Optional[str] = None, node_real: EdgeReal = "a", role: Optional[Literal["teledata", "telegate"]] = None):
        edge = frozenset(node_ids)
        self.C.add(edge)
        self.tau[edge] = edge_type
        self.phi[edge] = node_real
        if name:
            self.gate_name[edge] = name.lower()
        if role:
            self.edge_role[edge] = role
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
        qubit_indices = set()
        for node_id in self.S:
            if self.sigma[node_id] == 'q':
                try:
                    base = node_id.split('_')[0]  # e.g. "q4"
                    idx = int(base[1:])  # skip 'q'
                    qubit_indices.add(idx)
                except:
                    continue
        return max(qubit_indices) + 1 if qubit_indices else 0

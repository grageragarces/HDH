from collections import defaultdict
from enum import Enum
from typing import Dict, FrozenSet, Set, Tuple, List, Union, Optional


class _StrEnum(str, Enum):
    """A str-valued Enum that prints as its value, e.g. ``str(NodeType.QUANTUM) == "q"``."""

    def __str__(self) -> str:
        return self.value


class NodeType(_StrEnum):
    """Whether a node (or hyperedge) carries quantum or classical information.

    Members compare equal to their values, so ``"q"`` and
    ``NodeType.QUANTUM`` are interchangeable wherever a type is expected.
    """
    QUANTUM = "q"
    CLASSICAL = "c"


class Realisation(_StrEnum):
    """Whether a node (or hyperedge) always happens, or only if a classical
    condition holds."""
    ACTUAL = "a"
    PREDICTED = "p"


class EdgeRole(_StrEnum):
    """Distribution primitive assigned to a cut hyperedge."""
    TELEDATA = "teledata"
    TELEGATE = "telegate"


EdgeType = NodeType
NodeReal = Realisation
EdgeReal = Realisation
NodeID = str
TimeStep = int
Hyperedge = FrozenSet[NodeID]


def _coerce(enum_cls, value, what: str) -> str:
    """Validate `value` against `enum_cls` and return its plain string value.

    Storing plain strings (not members) keeps `sigma`, `tau` etc. identical to
    what they held before the Enums existed.
    """
    try:
        return enum_cls(value).value
    except ValueError:
        allowed = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(f"Invalid {what} {value!r}; expected one of {allowed}.") from None

class HDH:
    """A Hybrid Dependency Hypergraph: the model-agnostic representation this
    library is built around.

    An HDH represents a quantum workload (from any computational model —
    circuits, MBQC patterns, quantum walks, QCA) as a directed hypergraph of
    node *states* connected by hyperedges that model operations. It's usually
    not constructed directly; instead, build one of `hdh.models.circuit.Circuit`,
    `hdh.models.mbqc.MBQC`, `hdh.models.qw.QW`, or `hdh.models.qca.QCA` and call
    its `build_hdh()`, or convert an existing circuit with
    `hdh.converters.qiskit_converter.from_qiskit` (or the Cirq/PennyLane/Braket
    equivalents).

    Node IDs are strings of the form ``q{index}_t{timestep}`` (quantum) or
    ``c{index}_t{timestep}`` (classical) — the prefix must always match the
    node's `sigma` type, and hyperedges connect a set of such nodes to
    represent one operation's effect on the states it touches.

    Attributes:
        S: All node IDs in the hypergraph.
        C: All hyperedges, each a `frozenset` of node IDs.
        T: All distinct timesteps that appear in `time_map`.
        sigma: Node ID -> `"q"` (quantum) or `"c"` (classical).
        tau: Hyperedge -> `"q"` or `"c"`, mirroring `sigma` for edges.
        upsilon: Node ID -> `"a"` (actualized) or `"p"` (potential/predicted,
            e.g. a state that only exists if a classical condition holds).
        phi: Hyperedge -> `"a"` or `"p"`, mirroring `upsilon` for edges.
        time_map: Node ID -> the timestep it occurs at.
        gate_name: Hyperedge -> the gate/operation name that produced it
            (e.g. ``"h"``, ``"cx_stage2"``, ``"measure"``).
        gate_params: Hyperedge -> rotation angles / gate parameters, for
            hyperedges from a parametric gate that had params recorded.
        edge_args: Hyperedge -> `(qubits_with_time, bits_with_time,
            modifies_flags)`, used by converters to reconstruct a circuit
            representation from the HDH.
        edge_role: Hyperedge -> `"teledata"` or `"telegate"`, for hyperedges
            that have been assigned a distribution primitive.
        edge_metadata: Free-form per-hyperedge metadata.
        motifs: Reserved for motif-matching passes; unused by the core API.
    """

    def __init__(self):
        self.S: Set[NodeID] = set()
        self.C: Set[Hyperedge] = set()
        self.T: Set[TimeStep] = set()
        self.sigma: Dict[NodeID, str] = {}  # node types, a NodeType value
        self.tau: Dict[Hyperedge, str] = {}  # hyperedge types, a NodeType value
        self.upsilon: Dict[NodeID, str] = {} # node realization, a Realisation value
        self.phi: Dict[Hyperedge, str] = {} # hyperedge realization, a Realisation value
        self.time_map: Dict[NodeID, TimeStep] = {}  # f: S -> T
        self.gate_name: Dict[Hyperedge, str] = {}  # maps hyperedge → gate name string
        self.gate_params: Dict[Hyperedge, List[float]] = {}  # maps hyperedge → rotation params, if any
        self.edge_args: Dict[Hyperedge, Tuple[List[int], List[int], List[bool]]] = {} #mapping for nackwards translations
        self.edge_role: Dict[Hyperedge, str] = {}  # an EdgeRole value -> for primitive implementation
        self.motifs = {}
        self.edge_metadata: Dict[Hyperedge, Dict] = {}

    def add_node(self, node_id: NodeID, node_type: NodeType, time: TimeStep, node_real: NodeReal = "a"):
        """Add a node, or no-op if an identical node already exists.

        Args:
            node_id: Node ID, e.g. ``"q0_t0"`` or ``"c1_t2"``. The leading
                letter must match `node_type` ("q"/"c") — see `sigma`.
            node_type: A `NodeType`, or its value `"q"` / `"c"`.
            time: Timestep this node occurs at.
            node_real: A `Realisation`, or its value `"a"` / `"p"`.

        Raises:
            ValueError: If `node_type` or `node_real` is not a valid value,
                or if `node_id` already exists with a *different*
                `node_type`. Re-adding the same ID with the same type is
                fine (e.g. a later gate referencing an already-created
                input node) and simply leaves the existing node untouched.
        """
        node_type = _coerce(NodeType, node_type, "node type")
        node_real = _coerce(Realisation, node_real, "node realisation")
        existing_type = self.sigma.get(node_id)
        if existing_type is not None and existing_type != node_type:
            raise ValueError(
                f"Node '{node_id}' already exists with type '{existing_type}'; "
                f"cannot redefine it as type '{node_type}'. This usually means two "
                f"different logical values were mapped to the same node ID."
            )
        self.S.add(node_id)
        self.sigma[node_id] = node_type
        self.time_map[node_id] = time
        self.T.add(time)
        self.upsilon[node_id] = node_real

    def add_hyperedge(self, node_ids: Set[NodeID], edge_type: EdgeType, name: Optional[str] = None, node_real: EdgeReal = "a", role: Optional[EdgeRole] = None) -> Hyperedge:
        """Add a hyperedge connecting `node_ids`, representing one operation.

        Args:
            node_ids: The nodes this operation touches (its inputs and
                outputs together, since HDH edges are undirected).
            edge_type: A `NodeType`, or its value `"q"` / `"c"` — see `tau`.
            name: Operation name, e.g. ``"h"``, ``"cx_stage2"``, ``"measure"``.
                Stored lower-cased in `gate_name`; omit for an unnamed edge.
            node_real: A `Realisation`, or its value `"a"` / `"p"` — see `phi`.
            role: Distribution primitive this edge has been assigned, if any
                — an `EdgeRole`, or `"teledata"` / `"telegate"`. Usually set
                later by a partitioning pass, not at construction time.

        Returns:
            Hyperedge: the edge, as added to `C` — use this as the key into
            `tau`/`phi`/`gate_name`/`edge_args`/`gate_params`/etc.

        Raises:
            ValueError: If `edge_type`, `node_real` or `role` is not a valid value.
        """
        edge_type = _coerce(NodeType, edge_type, "edge type")
        node_real = _coerce(Realisation, node_real, "edge realisation")
        if role:
            role = _coerce(EdgeRole, role, "edge role")
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
        """Return the number of logical qubits, inferred from node IDs.

        Computed as `max(qubit_index) + 1` over every quantum node's index
        (e.g. ``"q4_t2"`` -> qubit 4), not a stored count — so it reflects
        the highest qubit index actually used, not necessarily how many
        distinct qubits are involved (a circuit that only uses qubit 0 and
        qubit 4 still reports 5).

        Returns:
            int: number of qubits, or 0 if there are no quantum nodes.
        """
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

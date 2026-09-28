from collections import defaultdict
import numbers
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

    Each node is the state of one *wire* (a qubit, a classical bit, or a
    model-specific label) at one timestep, with ID ``"<wire>_t<time>"``
    built by `add_node`. Hyperedges connect a set of such nodes to represent
    one operation's effect on the states it touches.

    Each core attribute has a readable name and a formal one matching the
    notation of the HDH paper; both refer to the same object.

    Attributes:
        nodes (S): All node IDs in the hypergraph.
        hyperedges (C): All hyperedges, each a frozenset of node IDs.
        timesteps (T): All distinct timesteps that appear in `time_map`.
        node_types (sigma): Node ID -> `"q"` (quantum) or `"c"` (classical).
        hyperedge_types (tau): Hyperedge -> `"q"` or `"c"`.
        node_realisation (upsilon): Node ID -> `"a"` (actualized) or `"p"`
            (predicted: only exists if a classical condition holds).
        hyperedge_realisation (phi): Hyperedge -> `"a"` or `"p"`.
        time_map: Node ID -> the timestep it occurs at.
        wire_of: Node ID -> the wire (qubit, bit, or model label) it is a
            state of. Partitioners count capacity per quantum wire.
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
        self.wire_of: Dict[NodeID, str] = {}  # node -> the wire it is a state of
        self.gate_name: Dict[Hyperedge, str] = {}  # maps hyperedge → gate name string
        self.gate_params: Dict[Hyperedge, List[float]] = {}  # maps hyperedge → rotation params, if any
        self.edge_args: Dict[Hyperedge, Tuple[List[int], List[int], List[bool]]] = {} #mapping for nackwards translations
        self.edge_role: Dict[Hyperedge, str] = {}  # an EdgeRole value -> for primitive implementation
        self.motifs = {}
        self.edge_metadata: Dict[Hyperedge, Dict] = {}

    # Readable names for the formal attributes above (same objects, not copies).

    @property
    def nodes(self) -> Set[NodeID]:
        """All node IDs (`S` in the paper's notation)."""
        return self.S

    @property
    def hyperedges(self) -> Set[Hyperedge]:
        """All hyperedges (`C`)."""
        return self.C

    @property
    def timesteps(self) -> Set[TimeStep]:
        """All distinct timesteps (`T`)."""
        return self.T

    @property
    def node_types(self) -> Dict[NodeID, str]:
        """Node ID -> `"q"` or `"c"` (`sigma`)."""
        return self.sigma

    @property
    def hyperedge_types(self) -> Dict[Hyperedge, str]:
        """Hyperedge -> `"q"` or `"c"` (`tau`)."""
        return self.tau

    @property
    def node_realisation(self) -> Dict[NodeID, str]:
        """Node ID -> `"a"` or `"p"` (`upsilon`)."""
        return self.upsilon

    @property
    def hyperedge_realisation(self) -> Dict[Hyperedge, str]:
        """Hyperedge -> `"a"` or `"p"` (`phi`)."""
        return self.phi

    @staticmethod
    def node_id(wire: str, time: TimeStep) -> NodeID:
        """The ID `add_node` gives the state of `wire` at `time`: ``"<wire>_t<time>"``."""
        return f"{wire}_t{time}"

    def add_node(self, wire: str, time: TimeStep, node_type: NodeType = NodeType.QUANTUM,
                 node_real: NodeReal = Realisation.ACTUAL) -> NodeID:
        """Add the state of `wire` at `time` and return its node ID.

        The ID is built for you (see `node_id`), and the wire is recorded in
        `wire_of`, so nothing about the node is passed twice. Adding the same
        wire and time again is a no-op apart from updating `node_real`.

        Example:
            >>> hdh = HDH()
            >>> hdh.add_node("q0", 1)
            'q0_t1'
            >>> hdh.add_node("c0", 2, "c")
            'c0_t2'

        Args:
            wire: The qubit, classical bit or other carrier this state belongs
                to, e.g. ``"q0"``, ``"c1"`` or an MBQC label like ``"a"``.
                Partitioners count capacity per quantum wire.
            time: Timestep this state occurs at.
            node_type: A `NodeType`, or its value `"q"` / `"c"`.
            node_real: A `Realisation`, or its value `"a"` / `"p"`.

        Returns:
            NodeID: the node's ID, ``"<wire>_t<time>"``.

        Raises:
            TypeError: If called with the pre-0.5 signature
                ``add_node(node_id, node_type, time)``.
            ValueError: If `node_type` or `node_real` is not a valid value, or
                the node already exists with a different `node_type`.
        """
        if not isinstance(time, numbers.Integral) or isinstance(time, bool):
            raise TypeError(
                f"add_node(wire, time, node_type) expects an integer time, got {time!r}. "
                f"Since hdh 0.5 the node ID is built from the wire and time: "
                f"write add_node('q0', 1, 'q') instead of add_node('q0_t1', 'q', 1)."
            )
        if not isinstance(wire, str) or not wire:
            raise TypeError(f"wire must be a non-empty string, got {wire!r}")
        time = int(time)
        node_type = _coerce(NodeType, node_type, "node type")
        node_real = _coerce(Realisation, node_real, "node realisation")
        node_id = self.node_id(wire, time)
        existing_type = self.sigma.get(node_id)
        if existing_type is not None and existing_type != node_type:
            raise ValueError(
                f"Node '{node_id}' already exists with type '{existing_type}'; "
                f"cannot redefine it as type '{node_type}'. This usually means two "
                f"different logical values were mapped to the same wire label."
            )
        self.S.add(node_id)
        self.sigma[node_id] = node_type
        self.wire_of[node_id] = wire
        self.time_map[node_id] = time
        self.T.add(time)
        self.upsilon[node_id] = node_real
        return node_id

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
        """Return the number of logical qubits (quantum wires).

        Circuit-style wires ``q<index>`` report `max(index) + 1`, so a circuit
        using only qubits 0 and 4 reports 5, matching the register size a
        converter needs. If any quantum wire is named otherwise (MBQC or QCA
        labels), the count of distinct quantum wires is returned instead.

        Returns:
            int: number of qubits, or 0 if there are no quantum nodes.
        """
        wires = {self.wire_of[n] for n in self.S if self.sigma[n] == NodeType.QUANTUM}
        indices = [int(w[1:]) for w in wires if w[:1] == "q" and w[1:].isdigit()]
        if len(indices) != len(wires):
            return len(wires)
        return max(indices) + 1 if indices else 0

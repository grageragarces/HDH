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
    """A Hybrid Dependency Hypergraph: a quantum workload as states (nodes)
    joined by the operations between them (hyperedges).

    Usually built by a model (`Circuit`, `MBQC`, `QW`, `QCA`) or a
    converter such as `from_qiskit`. Each node is the state of one wire at one
    timestep, with ID ``"<wire>_t<time>"``. Core attributes have a readable
    name and a formal one matching the HDH paper's notation; both refer to the
    same object.

    Example:
        >>> hdh = HDH()
        >>> a = hdh.add_node("q0", 0)
        >>> b = hdh.add_node("q0", 1)
        >>> edge = hdh.add_hyperedge({a, b}, "q", name="h")
        >>> sorted(hdh.nodes)
        ['q0_t0', 'q0_t1']
        >>> hdh.hyperedge_types[edge], hdh.gate_name[edge]
        ('q', 'h')

    Attributes:
        nodes (S): All node IDs.
        hyperedges (C): All hyperedges, each a frozenset of node IDs.
        timesteps (T): All timesteps in use.
        node_types (sigma): Node -> ``"q"`` or ``"c"``.
        hyperedge_types (tau): Hyperedge -> ``"q"`` or ``"c"``.
        node_realisation (upsilon): Node -> ``"a"`` (always happens) or
            ``"p"`` (only if a classical condition holds).
        hyperedge_realisation (phi): Hyperedge -> ``"a"`` or ``"p"``.
        time_map: Node -> its timestep.
        wire_of: Node -> the wire it is a state of. Partitioners count
            capacity per quantum wire.
        gate_name: Hyperedge -> the operation that produced it, e.g.
            ``"cx_stage2"``.
        gate_params: Hyperedge -> gate parameters, when recorded.
        edge_args: Hyperedge -> data the converters use to rebuild a circuit.
        edge_role: Hyperedge -> ``"teledata"`` or ``"telegate"``, once assigned.
        edge_metadata: Free-form per-hyperedge metadata.
        motifs: Reserved; unused by the core API.
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

        The ID is ``"<wire>_t<time>"`` (see `node_id`) and the wire is recorded
        in `wire_of`. Re-adding an existing state is a no-op apart from updating
        `node_real`.

        Args:
            wire: Qubit, bit or model label, e.g. ``"q0"``, ``"c1"`` or ``"a"``.
            time: Timestep of this state.
            node_type: ``"q"`` (default) or ``"c"``, or a `NodeType`.
            node_real: ``"a"`` (default) or ``"p"``, or a `Realisation`.

        Returns:
            The node ID.

        Raises:
            TypeError: If called with the pre-0.5 signature
                ``add_node(node_id, node_type, time)``.
            ValueError: On an invalid type, or if the state already exists with
                another type.

        Example:
            >>> hdh = HDH()
            >>> hdh.add_node("q0", 1)
            'q0_t1'
            >>> hdh.add_node("c0", 2, "c")
            'c0_t2'
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
        """Connect `node_ids` with a hyperedge representing one operation.

        Args:
            node_ids: The states the operation touches, inputs and outputs together.
            edge_type: ``"q"`` or ``"c"``, or a `NodeType`.
            name: Operation name, stored lower-cased in `gate_name`.
            node_real: ``"a"`` (default) or ``"p"``, or a `Realisation`.
            role: ``"teledata"`` or ``"telegate"``, usually assigned later by a
                partitioning pass.

        Returns:
            The hyperedge (a frozenset), used as the key into `hyperedge_types`,
            `gate_name` and the other per-edge maps.

        Raises:
            ValueError: If `edge_type`, `node_real` or `role` is invalid.

        Example:
            >>> hdh = HDH()
            >>> q, c = hdh.add_node("q0", 0), hdh.add_node("c0", 1, "c")
            >>> edge = hdh.add_hyperedge({q, c}, "c", name="measure")
            >>> sorted(edge)
            ['c0_t1', 'q0_t0']
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

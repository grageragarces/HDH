from typing import Protocol, runtime_checkable

from hdh.hdh import HDH


@runtime_checkable
class Model(Protocol):
    """What a computational model must provide to be used with HDH.

    A model is any object that can translate the computation it describes
    into an `HDH`. `Circuit`, `MBQC`, `QW` and `QCA` all satisfy this
    Protocol; a new model only needs a `build_hdh` method, no base class.
    `tests/test_model_protocol.py` checks every registered model against the
    invariants below, so a new model should be added to its registry.

    The HDH returned by `build_hdh` must:

    - create every node with `HDH.add_node`, so each has a type, a time and
      a wire (qubit, bit, or model label);
    - keep each wire a single type (a label used for a classical result is
      never reused as a quantum state);
    - only connect nodes that exist;
    - keep quantum hyperedges free of classical nodes.

    Examples:
        >>> from hdh.models.circuit import Circuit
        >>> isinstance(Circuit(), Model)
        True
    """

    def build_hdh(self) -> HDH:
        """Translate the model's recorded operations into an HDH."""
        ...

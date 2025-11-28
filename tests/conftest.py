import pytest
from hdh.models.circuit import Circuit
from hdh.hdh import HDH

@pytest.fixture
def empty_circuit():
    """Fixture for empty circuit"""
    return Circuit()

@pytest.fixture
def bell_state_circuit():
    """Fixture for Bell state circuit"""
    circuit = Circuit()
    circuit.add_instruction("h", [0])
    circuit.add_instruction("cx", [0, 1])
    return circuit

@pytest.fixture
def empty_hdh():
    """Fixture for empty HDH"""
    return HDH()

@pytest.fixture
def simple_hdh():
    """Fixture for simple HDH with a few nodes"""
    hdh = HDH()
    hdh.add_node("q0_t0", "q", 0)
    hdh.add_node("q0_t1", "q", 1)
    hdh.add_hyperedge({"q0_t0", "q0_t1"}, "q", name="h")
    return hdh
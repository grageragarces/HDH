from .qiskit_converter import from_qiskit, to_qiskit, partitions_to_qiskit
from .qasm_converter import from_qasm

# These SDKs are optional extras (see pyproject.toml's [tool.poetry.extras]);
# fall back to None so `hdh.converters.from_cirq` etc. are always attributes,
# just unusable until the corresponding extra is installed.
try:
    from .cirq_converter import from_cirq
except ImportError:
    from_cirq = None

try:
    from .pennylane_converter import from_pennylane, to_pennylane
except ImportError:
    from_pennylane = None
    to_pennylane = None

try:
    from .braket_converter import from_braket
except ImportError:
    from_braket = None
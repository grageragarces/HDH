from qiskit import QuantumCircuit
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.visualize import plot_hdh

# Teleportation protocol between A & B
qc = QuantumCircuit(3, 2)

# Prepa EPR pair
qc.h(1)
qc.cx(1, 2)

# Bell measurement on A
qc.cx(0, 1)
qc.h(0)

# Measurement A
qc.measure(0, 0)
qc.measure(1, 1)

# B correction based on A
qc.cx(1, 2)
qc.cz(0, 2)

hdh_graph = from_qiskit(qc)
plot_hdh(hdh_graph)

from qiskit import QuantumCircuit
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hdh.converters.convert_from_qiskit import from_qiskit
from hdh.visualize import plot_hdh

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.ccx(1, 2, 0)
qc.measure_all()

hdh_graph = from_qiskit(qc)
plot_hdh(hdh_graph)
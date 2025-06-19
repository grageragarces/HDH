from qiskit import QuantumCircuit
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdh.converters.convert_from_qasm import from_qasm
from hdh.visualize import plot_hdh

qasm_path = os.path.join(os.path.dirname(__file__), 'test_qasm_file.qasm')
hdh_graph = from_qasm('file', qasm_path)

plot_hdh(hdh_graph)

from qiskit import QuantumCircuit
from hdh.convert_from_qiskit import circuit_to_hdh
from hdh.visualize import plot_hdh

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

hdh_graph = circuit_to_hdh(qc)
plot_hdh(hdh_graph)

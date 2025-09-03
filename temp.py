from qiskit import QuantumCircuit

from hdh import plot_hdh
from hdh.converters import from_qiskit

# Quantum Circuit (workload)
qc = QuantumCircuit(5,3)
qc.ccx(0, 1, 2)
qc.h(3)
qc.cx(3, 4)
with qc.if_test(1):
    qc.z(4)
qc.measure(2,2)
qc.cx(0,3)
qc.measure(4,4)
qc.draw("circuit.png")

# Convert to HDH
hdh = from_qiskit(qc)
plot_hdh(hdh, "circuit_hdh.png") #Plot

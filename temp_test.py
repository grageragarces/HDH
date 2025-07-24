import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdh
from hdh.models.circuit import Circuit
from hdh.converters.qiskit import from_qiskit
from hdh.visualize import plot_hdh
from hdh.passes.cut import compute_cut, cost, partition_sizes, compute_parallelism_by_time
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import circuit_drawer

def circuit_test():

    circuit = Circuit()
    
    circuit.add_instruction("ccx", [0, 1, 2]) # ccx(q0, q1, q2)
    circuit.add_instruction("cx", [2, 1]) # cx(q2, q1)
    circuit.add_instruction("h", [3]) # h(q3)
    circuit.add_instruction("cx", [3, 4]) # cx(q3, q4)
    circuit.add_instruction("cx", [0, 3]) # cx(q0, q3)
    
    hdh = circuit.build_hdh()
    fig = plot_hdh(hdh)
    
    return hdh


def qiskit_test():
    q = QuantumRegister(5)
    c = ClassicalRegister(4)  # enough bits for classical outcomes
    qc = QuantumCircuit(q, c)

    qc.ccx(q[0], q[1], q[2])
    qc.cx(q[2], q[1])
    qc.h(q[3])
    qc.cx(q[3], q[4])

    # Mid-circuit measurement of qubit 1, store in classical bit 0
    # qc.measure(q[1], c[0])

    # Inject result of measurement into Z gate on qubit 4
    # qc.z(q[4]).c_if(c, 1 << 0)  # apply Z if c[0] == 1

    # qc.measure(q[2], c[1])
    qc.cx(q[0], q[3])
    # qc.measure(q[4], c[2])
    # qc.measure(q[5], c[3])

    image_path = "circuit_for_hdh.png"
    circuit_drawer(qc, output="latex", filename=image_path)

    hdh_graph = from_qiskit(qc)
    fig = plot_hdh(hdh_graph)
    
    return hdh_graph

if __name__ == "__main__":
    if circuit_test() == qiskit_test():
        print("HDH graphs match!")
    else:
        print("HDH graphs do not match!")
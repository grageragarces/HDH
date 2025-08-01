import sys
import os

from hdh.models import circuit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdh
from hdh.models.circuit import Circuit
from hdh.converters.qiskit import from_qiskit
from hdh.visualize import plot_hdh
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import circuit_drawer
from qiskit.circuit.controlflow import IfElseOp

def circuit_test():

    circuit = Circuit()
    
    circuit.add_instruction("ccx", [0, 1, 2]) # ccx(q0, q1, q2)
    circuit.add_instruction("cx", [2, 1]) # cx(q2, q1)
    circuit.add_instruction("h", [3]) # h(q3)
    circuit.add_instruction("cx", [3, 4]) # cx(q3, q4)
    # Mid-circuit measurement
    circuit.add_instruction("h", [1]) 
    circuit.add_conditional_gate(1, 4, "z")
    # TODO: c1_t3 should be injecting into q4_t5 but instead it is going for q4_t4
    # TODO: measurement is coming from q4_t4 instead of q4_t5, that includes misplaced node
    circuit.add_instruction("measure", [2]) 
    circuit.add_instruction("cx", [0, 3]) # cx(q0, q3)
    # circuit.add_instruction("measure", [4]) 
    circuit.add_instruction("measure", [5]) 
    
    hdh = circuit.build_hdh()
    fig = plot_hdh(hdh)
    
    return hdh

def qiskit_test():
    q = QuantumRegister(6)
    c = ClassicalRegister(4)  # enough bits for classical outcomes
    qc = QuantumCircuit(q, c)

    qc.ccx(q[0], q[1], q[2])
    qc.cx(q[2], q[1])
    qc.h(q[3])
    qc.cx(q[3], q[4])

    # Mid-circuit measurement 
    with qc.if_test((c[1], 1)):
        qc.z(q[4])
        
    qc.measure(q[2], c[1])
    qc.cx(q[0], q[3])
    qc.measure(q[4], c[2])
    qc.measure(q[5], c[3])

    image_path = "circuit_for_hdh.png"
    circuit_drawer(qc, output="mpl", filename=image_path)

    # hdh_graph = from_qiskit(qc)
    # plot_hdh(hdh_graph)
    
    return 1

def test():

    circuit = Circuit()
    
    circuit.add_instruction("h", [3])
    circuit.add_instruction("ccx", [0, 1, 2]) 
    circuit.add_instruction("cx", [2, 1])  
    circuit.add_instruction("cx", [3, 4])
    circuit.add_instruction("measure", [4])
    circuit.add_instruction("z", [4,4], cond_flag="p")


    hdh = circuit.build_hdh()
    fig = plot_hdh(hdh)
    
    return hdh

if __name__ == "__main__":
    # circuit_test() 
    # qiskit_test()
    test()
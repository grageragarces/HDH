"""
This set of functions has been use to create the figures presented accross HDH papers and posters.
Published versions may sligthly differ from the library's final product due to patches and updates.
The structures are in no way special, they simply often showcase many of the building blocks of HDHs and thus serve as good examples.
They have often also been used to study bugs thus why some comments may seem out of place sometimes & some functions may be from deprecated old versions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from typing import List, Tuple, Optional, Set, Dict
import matplotlib.pyplot as plt

import hdh
from hdh import HDH
from hdh.models.circuit import Circuit
from hdh.visualize import plot_hdh
from hdh.converters import from_qiskit


import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import circuit_drawer
from qiskit.circuit.controlflow import IfElseOp
from qiskit.qasm3 import dumps
from qiskit.circuit.library import ZGate

import matplotlib.pyplot as plt

# import pennylane as qml
# from pennylane.tape import OperationRecorder

import warnings

# import cirq

# import braket._sdk as braket
# from braket.circuits import Circuit 
# from converters.braket import from_braket

def circuit_test():
    qc = QuantumCircuit(2,2)

    # Non-Clifford portion
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)  # T gate is non-Clifford
    qc.cx(0, 1)

    # # Clifford subcircuit
    # qc.h(1)
    # qc.s(1)
    # qc.cx(0, 1)
    qc.measure(0,0)
    qc.measure(1,1)

    # qc.draw('mpl')
    hdh = from_qiskit(qc)
    hdh.add_node("c0_t10", "c",10,"p")
    hdh.add_node("c1_t10", "c",10,"p")
    hdh.add_hyperedge(["c0_t9", "c0_t10"], "c",)    
    hdh.add_hyperedge(["c1_t9", "c1_t10"], "c")    
    hdh.add_hyperedge(["c0_t9", "c1_t10"], "c","p")  
    hdh.add_hyperedge(["c1_t9", "c0_t10"], "c","p") 

    fig = plot_hdh(hdh)
    
def circuit_test_h():

    qc = QuantumCircuit(2,2)

    # Non-Clifford portion
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)  # T gate is non-Clifford
    qc.cx(0, 1)
    qc.measure(0,0)
    qc.measure(1,1)
    # daw pipeline to classical computer 

    qc.draw('mpl')
    
def hdh_circuit_test():
    
    qc = Circuit()

    qc.add_instruction("h", [0]) 
    qc.add_instruction("cx", [0, 1])
    qc.add_instruction("t", [0])  # T gate is non-Clifford
    qc.add_instruction("cx", [0, 1])
    
    # Clifford subcircuit
    qc.add_instruction("h", [1])
    qc.add_instruction("s", [1])
    qc.add_instruction("cx", [0, 1])
    qc.add_instruction("measure", [0])
    qc.add_instruction("measure", [1])


    hdh = qc.build_hdh()
    fig = plot_hdh(hdh)
    
    return 1

def hdh_circuit_test_h():
    
    qc = Circuit()

    qc.add_instruction("h", [0]) 
    qc.add_instruction("cx", [0, 1])
    qc.add_instruction("t", [0])  # T gate is non-Clifford
    qc.add_instruction("i", [0])
    qc.add_instruction("i", [1])
    qc.add_instruction("cx", [0, 1])
    qc.add_instruction("measure", [0])
    qc.add_instruction("measure", [1])
        
    hdh = qc.build_hdh()
    
    hdh.add_node("c0_t7","c",7, "p")
    hdh.add_node("c1_t7","c",7, "p")
    hdh.add_hyperedge(["c0_t7", "c0_t6","c1_t7", "c1_t6"], "c")

    fig = plot_hdh(hdh)
    
    return 1

def cat_test():

    hdh = HDH()
    
    # swap
    hdh.add_node("q1_t0","q",0)
    hdh.add_node("q3_t0","q",0)
    hdh.add_node("q1_t1","q",1)
    hdh.add_node("q3_t1","q",1)
    hdh.add_node("q1_t2","q",2)
    hdh.add_node("q3_t2","q",2)
    hdh.add_node("q1_t3","q",3)
    hdh.add_node("q3_t3","q",3)
    hdh.add_hyperedge(["q1_t0", "q1_t1"], "q")
    hdh.add_hyperedge(["q3_t0", "q3_t1"], "q")
    hdh.add_hyperedge(["q1_t1", "q3_t1", "q1_t2", "q3_t2"], "q")    
    hdh.add_hyperedge(["q1_t2", "q1_t3"], "q")
    hdh.add_hyperedge(["q3_t2", "q3_t3"], "q")
    
    # # cnot
    hdh.add_node("q0_t4","q",4)
    hdh.add_node("q0_t3","q",3)
    hdh.add_node("q0_t2","q",2)
    hdh.add_node("q1_t4","q",4)
    hdh.add_node("q0_t5","q",5)
    hdh.add_node("q1_t5","q",5)
    hdh.add_hyperedge(["q0_t2", "q0_t3"], "q")
    hdh.add_hyperedge(["q1_t3", "q1_t4", "q0_t3", "q0_t4"], "q")
    hdh.add_hyperedge(["q0_t4", "q0_t5"], "q")
    hdh.add_hyperedge(["q1_t4", "q1_t5"], "q")

    # meas
    hdh.add_node("c1_t6","c",6)   
    hdh.add_node("q3_t7","q",7)   
    hdh.add_node("q2_t7","q",7)
    hdh.add_hyperedge(["c1_t6", "q1_t5"], "c")
    hdh.add_hyperedge(["c1_t6", "q3_t7"], "c")
    hdh.add_hyperedge(["c1_t6", "q2_t7"], "c")
    
    # # target cnot
    hdh.add_node("q3_t8","q",8)
    hdh.add_node("q4_t8","q",8)
    hdh.add_node("q3_t9","q",9)
    hdh.add_node("q4_t9","q",9)
    hdh.add_node("q4_t7","q",7)
    hdh.add_node("q4_t10","q",10)
    hdh.add_node("q3_t10","q",10)
    hdh.add_hyperedge(["q3_t8", "q4_t8","q3_t9", "q4_t9"], "q")
    hdh.add_hyperedge(["q3_t8", "q3_t7"], "q")
    hdh.add_hyperedge(["q4_t8", "q4_t7"], "q")
    hdh.add_hyperedge(["q4_t9", "q4_t10"], "q")
    hdh.add_hyperedge(["q3_t9", "q3_t10"], "q")
    
    # # h
    hdh.add_node("q3_t11","q",11)
    hdh.add_hyperedge(["q3_t10","q3_t11"], "q")
    
    # # meas
    hdh.add_node("q0_t13","q",13)
    hdh.add_node("c3_t12","c",12)
    hdh.add_hyperedge(["c3_t12", "q3_t11"], "c")
    hdh.add_hyperedge(["c3_t12", "q0_t13"], "c")
    hdh.add_hyperedge(["q0_t5", "q0_t13"], "q")
    
    fig = plot_hdh(hdh)
    
    return hdh

def cat_circuit_test():

    circuit = Circuit()
    
    circuit.add_instruction("cx", [0, 1]) 
    
    hdh = circuit.build_hdh()
    fig = plot_hdh(hdh)
    
    return hdh

# ==== hdhex start ====

def circuit_test_ex():

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

def circex():

    circuit = Circuit()
    
    circuit.add_instruction("h", [3])
    circuit.add_instruction("ccx", [0, 1, 2]) 
    circuit.add_instruction("cx", [2, 1])  
    circuit.add_instruction("cx", [3, 4])
    circuit.add_instruction("measure", [4])
    circuit.add_instruction("z", [4,4], cond_flag="p")
    circuit.add_instruction("measure", [2])
    circuit.add_instruction("measure", [5])


    hdh = circuit.build_hdh()
    fig = plot_hdh(hdh)
    
    return hdh

# ==== hdhex end ====

def test():
    qc = Circuit()
    
    qc.add_instruction("h", [0])
    qc.add_instruction("cx", [0, 1])
    qc.add_instruction("measure", [0],[2])
    qc.add_instruction("measure", [1],[1])

    hdh = qc.build_hdh()
    fig = plot_hdh(hdh)
    
def qiskit_test_cond():
    qr = QuantumRegister(1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    qc.measure(0, 0)

    with qc.if_test((cr, 1)):
        qc.x(0)
    
    # print(dumps(qc))
    hdh = from_qiskit(qc)
    fig = plot_hdh(hdh)

def test_cond():
    qc = Circuit()
    
    qc.add_instruction("h", [0])
    qc.add_instruction("cx", [0, 1], cond_flag="p")
    qc.add_instruction("measure", [0],[2])
    
    hdh = qc.build_hdh()
    fig = plot_hdh(hdh)
    
# dev = qml.device("default.qubit", wires=2, shots=None)

# @qml.qnode(dev)
# def qml_circuit():
#     qml.Hadamard(0)
#     qml.RX(10, wires=1)
#     qml.CNOT([0, 1])
#     return qml.probs(wires=[0, 1])

def test_penny():
    # Record the underlying Python function WITHOUT executing the device
    with OperationRecorder() as rec:
        qml_circuit.func()   # call the wrapped function directly

    hdh = from_pennylane(rec) # your converter accepts OperationRecorder
    plot_hdh(hdh)
    
def test_pennylane_with_terminal_measurements():
    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev2)
    def circ_mid():
        qml.Hadamard(0)
        m = qml.measure(0)  # MidMeasureMP

        def then_branch():
            qml.X(1)

        # Use function form when condition involves a measurement value
        qml.cond(m == 1, then_branch)

        return qml.probs(wires=[0, 1])  # terminal measurement

    with OperationRecorder() as rec, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        circ_mid.func()

    hdh = from_pennylane(rec)
    plot_hdh(hdh) 
    
def test_cirq():
    q0, q1 = cirq_converter.LineQubit.range(2)
    qc = cirq_converter.Circuit(cirq_converter.H(q0), cirq_converter.CX(q0, q1), cirq_converter.measure(q0, q1, key='b'))
    hdh = from_cirq(qc)
    plot_hdh(hdh)
    
def test_braket():
    qc = Circuit()
    qc.h(0)
    qc.cnot(control=0, target=1)
    qc.measure(1)
    hdh = from_braket(qc)
    plot_hdh(hdh)
    
# Fixed figures draft ---

def circuit_test_alt(): #figure 5; circuit example
    qc = QuantumCircuit(6,6)

    qc.ccx(0, 1, 2)  
    qc.h(3)
    qc.cx(2,1)
    qc.cx(3,4)
    qc.h(5)
    qc.measure(5, 5)   
    
    with qc.if_test((qc.clbits[5], 1)):  
        qc.x(4)  
    
    qc.cx(0,3)
    
    qc.measure(2,2)
    qc.measure(4,4)
    
    qc.draw('mpl', filename='qiskit_circuit.svg')

    
    hdh = from_qiskit(qc)
    
    fig = plot_hdh(hdh)

def easycnotex():
    circuit = Circuit()
    circuit.add_instruction("cx", [1, 2])
    circuit.add_instruction("cx", [0, 1])
    circuit.add_instruction("cx", [2, 4])
    circuit.add_instruction("cx", [1, 3])
    circuit.add_instruction("cx", [2, 3])
    circuit.add_instruction("cx", [2, 4])

    hdh = circuit.build_hdh() 
    fig = plot_hdh(hdh) 
    
# End fixed figures draft ---

if __name__ == "__main__":
    circuit_test_alt()

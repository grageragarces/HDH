import hdh
from hdh.hdh import HDH
from hdh.visualize import plot_hdh

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

# target cnot
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

# h gate
hdh.add_node("q3_t11","q",11)
hdh.add_hyperedge(["q3_t10","q3_t11"], "q")

# meas
hdh.add_node("q0_t13","q",13)
hdh.add_node("c3_t12","c",12)
hdh.add_hyperedge(["c3_t12", "q3_t11"], "c")
hdh.add_hyperedge(["c3_t12", "q0_t13"], "c")
hdh.add_hyperedge(["q0_t5", "q0_t13"], "q")

fig = plot_hdh(hdh) # Visualize HDH

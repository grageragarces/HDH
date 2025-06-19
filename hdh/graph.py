
class HDHNode:
    def __init__(self, id, type, time):
        self.id = id
        self.type = type  # e.g., 'qubit', 'classical', 'measurement'
        self.time = time

    def __repr__(self):
        return f"HDHNode(id={self.id}, type={self.type}, time={self.time})"

class HDHEdge:
    def __init__(self, id, operation, inputs, outputs):
        self.id = id
        self.operation = operation  # e.g., 'H', 'CX', 'M', etc.
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f"HDHEdge(id={self.id}, op={self.operation}, in={self.inputs}, out={self.outputs})"

class HDHGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)

    def __repr__(self):
        return f"HDHGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

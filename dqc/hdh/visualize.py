import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from . import HDH
import re

def plot_hdh(hdh, save_path=None):
    nodes = list(hdh.S)
    edges = [tuple(e) for e in hdh.C]

    node_positions = {}
    node_timesteps = {}
    node_qubits = {}
    qubit_labels = set()
    timesteps = set()

    for node in nodes:
        if node.startswith("q") or node.startswith("c"):
            match = re.match(r"[qc](\d+)_t(\d+)", node)
            if match:
                index, timestep = map(int, match.groups())
                node_positions[node] = (timestep, index)
                node_timesteps[node] = timestep
                node_qubits[node] = index
                qubit_labels.add(index)
                timesteps.add(timestep)
        else:
            print(f"Skipping node due to unrecognized format: {node}")

    if not node_positions:
        print("No valid nodes found with q{index}_t{step} or c{index}_t{step} format.")
        return

    qubit_ticks = sorted(qubit_labels)
    timestep_ticks = sorted(timesteps)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Qubit/Clbit Index")
    ax.set_xticks(timestep_ticks)
    ax.set_yticks(qubit_ticks)
    ax.set_ylim(min(qubit_ticks) - 1, max(qubit_ticks) + 1)

    involved_nodes = set()
    for edge in edges:
        involved_nodes.update(edge)

    for node in involved_nodes:
        if node in node_positions:
            x, y = node_positions[node]
            node_type = hdh.sigma.get(node, "q")
            color = {
                "q": "skyblue",
                "ctrl": "orange",
                "c": "lightgreen"
            }.get(node_type, "gray")
            ax.plot(x, y, 'o', markersize=10, color=color)
            ax.text(x, y + 0.15, node, ha='center')

    seen_pairs = set()
    for edge in edges:
        edge_nodes = [n for n in edge if n in node_positions]
        edge_type = hdh.tau.get(edge, "q")
        color = "green" if edge_type == "c" else "gray"
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                n1, n2 = edge_nodes[i], edge_nodes[j]
                t1, t2 = node_timesteps[n1], node_timesteps[n2]
                q1, q2 = node_qubits[n1], node_qubits[n2]

                if t1 == t2:
                    continue

                type1 = hdh.sigma.get(n1, "q")
                type2 = hdh.sigma.get(n2, "q")
                if type1 == "ctrl" and type2 == "ctrl":
                    continue

                if t1 > t2:
                    n1, n2 = n2, n1
                    t1, t2 = t2, t1

                pair = (n1, n2)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                x0, y0 = node_positions[n1]
                x1, y1 = node_positions[n2]
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_hdh_networkx(hdh: HDH):
    G = nx.DiGraph()

    for node in hdh.S:
        node_type = hdh.sigma[node]
        color = {
            "q": "skyblue",
            "ctrl": "orange",
            "c": "lightgreen"
        }.get(node_type, "gray")
        G.add_node(node, color=color)

    for edge in hdh.C:
        edge_type = hdh.tau[edge]
        edge_nodes = list(edge)
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                G.add_edge(edge_nodes[i], edge_nodes[j], type=edge_type)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray")
    plt.show()
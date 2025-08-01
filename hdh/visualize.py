import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .hdh import HDH

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
                node_timesteps[node] = timestep
                node_qubits[node] = index
                qubit_labels.add(index)
                timesteps.add(timestep)
        else:
            print(f"Skipping node due to unrecognized format: {node}")

    if not qubit_labels:
        print("No valid nodes found with q{{index}}_t{{step}} or c{{index}}_t{{step}} format.")
        return

    max_index = max(qubit_labels)

    for node in nodes:
        if node in node_timesteps and node in node_qubits:
            timestep = node_timesteps[node]
            flipped_index = max_index - node_qubits[node]
            node_positions[node] = (timestep, flipped_index)

    qubit_ticks = sorted(qubit_labels)
    flipped_ticks = [max_index - i for i in qubit_ticks]
    timestep_ticks = sorted(timesteps)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Qubit/Clbit Index")
    ax.set_xticks(timestep_ticks)
    ax.set_yticks(flipped_ticks)
    ax.set_yticklabels(qubit_ticks)
    ax.set_ylim(min(flipped_ticks) - 1, max(flipped_ticks) + 1)

    involved_nodes = set()
    for edge in edges:
        involved_nodes.update(edge)

    for node in involved_nodes:
        if node in node_positions:
            x, y = node_positions[node]
            node_type = hdh.sigma.get(node, "q")
            is_predicted = hdh.upsilon.get(node, "a") == "p"
            color = {
                "q": "black",
                "ctrl": "black",
                "c": "orange"
            }.get(node_type, "black")

            face_color = "white" if is_predicted else color
            ax.plot(x, y, 'o', markersize=10,
                    markerfacecolor=face_color,
                    markeredgecolor=color,
                    markeredgewidth=2,
                    linestyle='--' if is_predicted else '-')
            ax.text(x, y + 0.15, node, ha='center')

    seen_pairs = set()
    for edge in edges:
        edge_nodes = [n for n in edge if n in node_positions]

        edge_type = hdh.tau.get(frozenset(edge))
        if edge_type is None:
            node_types = [hdh.sigma.get(n, "q") for n in edge]
            edge_type = "c" if all(t == "c" for t in node_types) else "q"

        color = "orange" if edge_type == "c" else "black"
        is_predicted = hdh.phi.get(frozenset(edge), "a") == "p"
        line_style = '--' if is_predicted else '-'

        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                n1, n2 = edge_nodes[i], edge_nodes[j]
                t1, t2 = node_timesteps[n1], node_timesteps[n2]

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

                if edge_type == "c":
                    dx = x1 - x0
                    dy = y1 - y0
                    dist = np.hypot(dx, dy)
                    if dist == 0:
                        continue
                    t = np.linspace(0, 1, 200)
                    x_line = x0 + dx * t
                    y_line = y0 + dy * t
                    nx_vec = -dy / dist
                    ny_vec = dx / dist
                    displacement = 0.08 * np.sin(6 * 2 * np.pi * t)
                    x_vals = x_line + displacement * nx_vec
                    y_vals = y_line + displacement * ny_vec
                    ax.plot(x_vals, y_vals, color=color, linewidth=2, linestyle=line_style)
                else:
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, linestyle=line_style)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".png", ".jpg"]:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

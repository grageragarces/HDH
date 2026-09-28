import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .hdh import HDH

def plot_hdh(hdh, save_path="hdh_plot.svg"):
    """Plot an HDH with time on the x-axis and one row per wire.

    Circuit wires ``q<i>`` and ``c<i>`` share row ``i``; any other wire gets a
    row of its own.

    Args:
        hdh: The HDH to plot.
        save_path: File to save to (the extension sets the format), or
            ``None`` to show the plot instead.

    Example:
        >>> from hdh.models.circuit import Circuit
        >>> c = Circuit()
        >>> c.add_instruction("cx", [0, 1])
        >>> plot_hdh(c.build_hdh(), save_path="cx.svg")  # doctest: +SKIP
    """
    nodes = list(hdh.nodes)
    edges = [tuple(e) for e in hdh.hyperedges]

    if not nodes:
        print("Nothing to plot: the HDH has no nodes.")
        return

    # One row per wire, read from the HDH rather than parsed from node IDs.
    # Circuit wires q<i> and c<i> share row i (a qubit and the bit it is
    # measured into); any other wire, e.g. an MBQC label, gets its own row.
    def _row_key(wire):
        m = re.fullmatch(r"[qc](\d+)", wire)
        return int(m.group(1)) if m else wire

    keys = {_row_key(hdh.wire_of[n]) for n in nodes}
    numbered = sorted(k for k in keys if isinstance(k, int))
    named = sorted(k for k in keys if not isinstance(k, int))
    row_order = numbered + named
    row_of = {key: len(row_order) - 1 - i for i, key in enumerate(row_order)}  # first row on top

    node_positions = {
        n: (hdh.time_map[n], row_of[_row_key(hdh.wire_of[n])]) for n in nodes
    }
    timestep_ticks = sorted({hdh.time_map[n] for n in nodes})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Timestep",fontsize=16)
    ax.set_ylabel("Qubit/Clbit Index",fontsize=16)
    ax.set_xticks(timestep_ticks)
    ax.set_yticks([row_of[k] for k in row_order])
    ax.set_yticklabels([str(k) for k in row_order])
    ax.set_ylim(-1, len(row_order))

    involved_nodes = set()
    for edge in edges:
        involved_nodes.update(edge)

    for node in involved_nodes:
        if node in node_positions:
            x, y = node_positions[node]
            node_type = hdh.node_types.get(node, "q")
            is_predicted = hdh.node_realisation.get(node, "a") == "p"
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

        edge_type = hdh.hyperedge_types.get(frozenset(edge))
        if edge_type is None:
            node_types = [hdh.node_types.get(n, "q") for n in edge]
            edge_type = "c" if all(t == "c" for t in node_types) else "q"

        color = "orange" if edge_type == "c" else "black"
        is_predicted = hdh.hyperedge_realisation.get(frozenset(edge), "a") == "p"
        line_style = '--' if is_predicted else '-'

        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                n1, n2 = edge_nodes[i], edge_nodes[j]
                t1, t2 = hdh.time_map[n1], hdh.time_map[n2]

                if t1 == t2:
                    continue

                type1 = hdh.node_types.get(n1, "q")
                type2 = hdh.node_types.get(n2, "q")
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
                    # Calculate perpendicular unit vector
                    nx_vec = -dy / dist
                    ny_vec = dx / dist
                    # Offset distance for double line
                    offset = 0.05
                    # Draw two parallel lines
                    ax.plot([x0 + offset * nx_vec, x1 + offset * nx_vec], 
                            [y0 + offset * ny_vec, y1 + offset * ny_vec], 
                            color=color, linewidth=1, linestyle=line_style)
                    ax.plot([x0 - offset * nx_vec, x1 - offset * nx_vec], 
                            [y0 - offset * ny_vec, y1 - offset * ny_vec], 
                            color=color, linewidth=1, linestyle=line_style)
                else:
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, linestyle=line_style)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".png", ".jpg"]:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
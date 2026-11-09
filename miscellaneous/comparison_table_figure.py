"""
Redraws docs/img/comparison_table.{svg,png}: which cut types each abstraction can express.

The earlier PNG was a ~770 px raster and blurred when zoomed; this version is
vector, so it stays sharp at any size. Coordinates are in "2x screenshot
pixels" measured from that original, y pointing down.

Run from the repository root:  python miscellaneous/comparison_table_figure.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
W, H = 1546, 1628
PT = 72 / 200  # points per drawing unit (figure is 200 units per inch)

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

GREEN_BOX, YELLOW_BOX = "#c9edc0", "#fdefb2"
GREEN_DOT, YELLOW_DOT = "#2f4a1c", "#5b4a12"
RED, BLUE = "#e8202a", "#0b2fa0"
TEXT = "#2b2b2b"

fig = plt.figure(figsize=(W / 200, H / 200))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def line(xs, ys, lw_units=3, color="black", **kw):
    ax.plot(xs, ys, color=color, lw=lw_units * PT, solid_capstyle="butt", **kw)


def text(x, y, s, size=14, ha="center", color=TEXT, **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va="center", color=color, **kw)


def box(x0, y0, x1, y1, color):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color, ec="none", zorder=0))


def node(x, y, color="black", r=11):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=3))


def cross(cx, cy, w=182, h=138):
    """Two tapered, slightly bowed red strokes, like the original hand-drawn cross."""
    t = np.linspace(0, 1, 60)
    for (x0, y0, x1, y1, bow) in [
        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 10),
        (cx - w / 2, cy + h / 2, cx + w / 2, cy - h / 2, -10),
    ]:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        n = np.hypot(dx, dy)
        px, py = -dy / n, dx / n  # unit normal
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * (mx + bow * px) + t**2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * (my + bow * py) + t**2 * y1
        wid = 12 * np.sin(np.pi * t) ** 0.35 + 0.5
        ax.fill(
            np.concatenate([bx + px * wid, (bx - px * wid)[::-1]]),
            np.concatenate([by + py * wid, (by - py * wid)[::-1]]),
            color=RED, ec="none", zorder=4,
        )


def circuit(y0, y1, x_wire0, x_wire1, x_c1, x_c2, qx, cut=None):
    """Two-qubit circuit: CNOT(q0->q1) then CNOT(q1->q0), Qiskit-style, with an optional red cut."""
    for y in (y0, y1):
        line([x_wire0, x_wire1], [y, y], 3, "#222222")
    text(qx, y0, "$q_0$", 13)
    text(qx, y1, "$q_1$", 13)
    for xc, yc, yt in [(x_c1, y0, y1), (x_c2, y1, y0)]:  # control row, target row
        line([xc, xc], [yc, yt], 3.5, BLUE, zorder=2)
        node(xc, yc, BLUE, 9)
        ax.add_patch(Circle((xc, yt), 24, fc=BLUE, ec="none", zorder=3))
        line([xc - 15, xc + 15], [yt, yt], 3.5, "white", zorder=4)
        line([xc, xc], [yt - 15, yt + 15], 3.5, "white", zorder=4)
    if cut:
        (cx0, cy0, cx1, cy1) = cut
        line([cx0, cx1], [cy0, cy1], 6, "#ff3030", linestyle=(0, (2.2, 1.6)), zorder=5)


def hdh(xs, y_top, y_bot, label_dy=36, node_colors=None):
    """Two-wire HDH of the circuit above: wires plus the pre/post-gate crossing edges."""
    for y in (y_top, y_bot):
        line([xs[0], xs[3]], [y, y], 2, "#333333")
    line([xs[1], xs[2]], [y_top, y_bot], 2, "#333333")
    line([xs[1], xs[2]], [y_bot, y_top], 2, "#333333")
    for i, x in enumerate(xs):
        for y, q in ((y_top, "q1"), (y_bot, "q2")):
            c = "black" if node_colors is None else node_colors[(q, i)]
            node(x, y, c)
            text(x, y + (-label_dy if y == y_top else label_dy), f"{q}_t{i}", 11)


# ---- grid and headers --------------------------------------------------------
for x, y0 in ((368, 30), (683, 30), (1018, 42)):
    line([x, x], [y0, 1605], 8)
for y, x0, x1 in ((130, 28, 1522), (392, 28, 1522), (762, 35, 1546)):
    line([x0, x1], [y, y], 8)
text(523, 78, "Telegate", 20)
text(850, 78, "Teledata", 20)
text(1235, 78, "HDH", 20)
text(57, 470, "Gate cut", 26, ha="left", color="black")
text(62, 822, "Wire cut", 26, ha="left", color="black")

# ---- row 1: the circuit with no cut -----------------------------------------
circuit(216, 313, 122, 325, 173, 268, 85)
text(495, 228, "q1", 14); node(495, 281, "black"); node(604, 281, "black")
line([495, 604], [281, 281], 2, "#333333"); text(604, 327, "q2", 14)
text(780, 228, "CNOT 1", 14); node(795, 281, "black"); node(905, 281, "black")
line([795, 905], [281, 281], 2, "#333333"); text(907, 327, "CNOT 2", 14)
hdh([1118, 1215, 1325, 1433], 215, 302)

# ---- row 2: gate cut --------------------------------------------------------
circuit(574, 668, 138, 342, 190, 287, 102, cut=(126, 620, 238, 620))
box(415, 514, 547, 621, GREEN_BOX); box(547, 563, 676, 669, YELLOW_BOX)
text(489, 536, "q1", 14); text(597, 635, "q2", 14)
line([489, 599], [588, 588], 2, "#4a4a2a"); node(489, 588, GREEN_DOT); node(599, 588, YELLOW_DOT)
cross(855, 585)
box(1065, 478, 1480, 603, GREEN_BOX); box(1065, 603, 1480, 730, YELLOW_BOX)
hdh([1110, 1208, 1317, 1426], 550, 637, node_colors={
    (q, i): (GREEN_DOT if q == "q1" else YELLOW_DOT) for q in ("q1", "q2") for i in range(4)})

# ---- row 3: wire cuts (three places) ----------------------------------------
B = 760
circuit(B + 202, B + 297, 110, 315, 163, 260, 75, cut=(206, B + 160, 206, B + 326))
circuit(B + 408, B + 503, 110, 315, 163, 260, 75, cut=(117, B + 372, 117, B + 540))
circuit(B + 658, B + 753, 118, 322, 170, 265, 82, cut=(298, B + 628, 298, B + 794))
for cy in (B + 245, B + 448, B + 695):
    cross(518 if cy != B + 695 else 538, cy)
for cy in (B + 460, B + 695):
    cross(867, cy)
box(722, B + 150, 856, B + 258, GREEN_BOX); box(856, B + 215, 990, B + 322, YELLOW_BOX)
text(788, B + 181, "CNOT 1", 14); text(915, B + 281, "CNOT 2", 14)
line([802, 911], [B + 233, B + 233], 2, "#4a4a2a"); node(802, B + 233, GREEN_DOT); node(911, B + 233, YELLOW_DOT)


def hdh_cut(xs, y_top, y_bot, g0, g1, split_x, y_box0, y_box1, green_cols):
    """HDH with a vertical cut: columns in `green_cols` are in the first device (green)."""
    x0, x1 = g0, g1
    box(x0, y_box0, split_x, y_box1, GREEN_BOX)
    box(split_x + 6, y_box0, x1, y_box1, YELLOW_BOX)
    hdh(xs, y_top, y_bot, node_colors={
        (q, i): (GREEN_DOT if i in green_cols else YELLOW_DOT) for q in ("q1", "q2") for i in range(4)})


hdh_cut([1113, 1212, 1320, 1430], B + 132, B + 219, 1056, 1488, 1272, B + 70, B + 291, (0, 1))
hdh_cut([1113, 1212, 1320, 1430], B + 391, B + 478, 1056, 1488, 1166, B + 328, B + 549, (0,))
hdh_cut([1124, 1221, 1330, 1440], B + 658, B + 744, 1067, 1497, 1387, B + 597, B + 816, (0, 1, 2))

for ext in ("svg", "png"):
    fig.savefig(os.path.join(OUT, f"comparison_table.{ext}"), dpi=300, facecolor="white")

"""
Regenerates the two small table figures used in paper.md at print resolution:

    docs/img/HDHobjects.{png,svg}         symbol legend
    docs/img/circuitmappings.{png,svg}    circuit operation -> HDH motif table

The earlier versions were ~300 px wide screenshots and blurred when zoomed.
Run from the repository root:  python miscellaneous/table_figures.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DPI = 600
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

ORANGE = "#ff7f0e"
GREY = "#555555"
BLACK = "#000000"

plt.rcParams["svg.fonttype"] = "path"  # text becomes outlines, so SVGs render without the font
plt.rcParams["font.family"] = "STIX Two Text"
plt.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "Times New Roman"]


def canvas(width, height):
    """Axes in 'original screenshot pixel' units, origin top-left, y pointing down."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    """Write a 600 dpi PNG (used by the docs) and an SVG (used by the paper)."""
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), facecolor="white")
    plt.close(fig)


def dot(ax, x, y, color=BLACK, ms=4.2):
    ax.plot([x], [y], "o", color=color, ms=ms, mew=0, zorder=3)


def legend_figure():
    s = 1 / 4  # the layout below was measured on a 4x enlargement
    fig, ax = canvas(295, 180)
    text = dict(ha="center", va="center", fontsize=11, color=BLACK)

    ax.plot([363 * s] * 2, [30 * s, 700 * s], color=GREY, lw=0.9)
    ax.plot([38 * s, 1107 * s], [125 * s] * 2, color=GREY, lw=0.9)
    ax.text(200 * s, 72 * s, "Symbol", **text)
    ax.text(735 * s, 72 * s, "Meaning", **text)

    rows = [
        (170, "Classical node"),
        (268, "Quantum node"),
        (365, "Classical hyperedge"),
        (462, "Quantum hyperedge"),
        (550, "Predicted node"),
        (650, "Predicted hyperedge"),
    ]
    for y, label in rows:
        ax.text(735 * s, y * s, label, **text)

    dot(ax, 200 * s, 172 * s, ORANGE, 4.6)
    dot(ax, 200 * s, 268 * s, BLACK, 4.6)
    for y in (358, 373):
        ax.plot([138 * s, 262 * s], [y * s] * 2, color=ORANGE, lw=0.9, solid_capstyle="butt")
    ax.plot([163 * s, 235 * s], [475 * s] * 2, color=GREY, lw=1.0, solid_capstyle="butt")
    ax.add_patch(Circle((200 * s, 543 * s), 37 * s, fill=False, ec="#888888", lw=0.7, ls=(0, (4, 3))))
    for x0, x1 in ((150, 175), (202, 228)):
        ax.plot([x0 * s, x1 * s], [672 * s] * 2, color="#888888", lw=1.0, solid_capstyle="butt")

    save(fig, "HDHobjects")


def mapping_figure():
    s = 1 / 6  # measured on a 6x enlargement, cropped 30 px from the top
    fig, ax = canvas(270, 188)

    def X(v):
        return v * s

    def Y(v):
        return v * s + 30

    text = dict(ha="left", va="center", fontsize=9.5, color=BLACK)
    head = dict(ha="center", va="center", fontsize=9.5, color=BLACK)

    ax.plot([X(735)] * 2, [5, 188], color=GREY, lw=0.9)
    ax.plot([2, 268], [Y(38)] * 2, color=GREY, lw=0.9)
    ax.text(X(420), 21, "Motif", **head)
    ax.text(195, 21, "Operation", **head)

    cols = [X(255), X(420), X(585)]
    wire0 = X(100)

    def wires(ys):
        for y in ys:
            ax.plot([wire0, cols[2]], [y] * 2, color=BLACK, lw=0.8, zorder=1)
            for x in cols:
                dot(ax, x, y)
        # every pre-gate state links to every post-gate state
        for ya in ys:
            for yb in ys:
                if ya != yb:
                    ax.plot([cols[0], cols[1]], [ya, yb], color=BLACK, lw=0.7, zorder=1)

    # qubit initialisation
    dot(ax, X(345), Y(130))
    # single qubit gate
    ax.plot([cols[0], cols[1]], [Y(258)] * 2, color=GREY, lw=0.9)
    dot(ax, cols[1], Y(258))
    # two- and three-qubit gates
    wires([Y(315), Y(445)])
    wires([Y(500), Y(635), Y(765)])
    # measurement
    for dy in (-1.1, 1.1):
        ax.plot([cols[0], cols[1]], [Y(892) + dy] * 2, color=ORANGE, lw=0.8, solid_capstyle="butt")
    dot(ax, cols[1], Y(892), ORANGE)

    x_text = X(795)
    for y, label in [
        (Y(130) - 1, "Qubit initialisation"),
        (Y(258) - 1, "Single qubit gate"),
        (Y(380), "Two qubit gate"),
        (Y(635), "Three qubit gate"),
        (Y(892), "Measurement"),
    ]:
        ax.text(x_text, y, label, **text)

    save(fig, "circuitmappings")


if __name__ == "__main__":
    legend_figure()
    mapping_figure()

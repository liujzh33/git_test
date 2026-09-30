from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "site" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

INK = "#16242C"
SLATE = "#5B6B73"
FILM = "#25577A"
BONE = "#B8913F"
RULE = "#D5DBDC"
PLATE = "#0E1418"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
})

VERT_NAMES = {i: n for i, n in enumerate(
    ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
     "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
     "L1", "L2", "L3", "L4", "L5", "L6"], start=1)}
VERT_NAMES[28] = "T13"


def _ramp(c0, c1, n):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return [tuple(a + (b - a) * t) for t in np.linspace(0, 1, n)]


# Region-coded vertebra colours: cervical blues, thoracic teals/greens, lumbar ochres.
_CERV = _ramp("#8FC3E8", "#2F6FA8", 7)
_THOR = _ramp("#9ED9C3", "#2E8A6E", 12)
_LUMB = _ramp("#F2CF7E", "#C0782B", 6)
VERT_COLORS = {i + 1: c for i, c in enumerate(_CERV + _THOR + _LUMB)}
VERT_COLORS[28] = VERT_COLORS[19]


def label_cmap(max_label=28):
    colors = [(0, 0, 0, 0)] + [(*VERT_COLORS.get(i, (0.6, 0.6, 0.6)), 1.0) for i in range(1, max_label + 1)]
    return ListedColormap(colors)


def region_of(lbl):
    if lbl <= 7:
        return "颈椎"
    if lbl <= 19 or lbl == 28:
        return "胸椎"
    return "腰椎"


def window(img, level, width):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


def plate_axes(ax):
    ax.set_facecolor(PLATE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, px_per_mm, mm=50, x0=0.06, y0=0.05, color="white"):
    """Draw a horizontal scale bar of `mm` millimetres in axes-fraction placement."""
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    w = abs(xl[1] - xl[0])
    h = abs(yl[1] - yl[0])
    xs = min(xl) + x0 * w
    ys = (max(yl) - y0 * h) if yl[0] > yl[1] else (min(yl) + y0 * h)
    ax.plot([xs, xs + mm * px_per_mm], [ys, ys], color=color, lw=2.2, solid_capstyle="butt")
    ax.text(xs + mm * px_per_mm / 2, ys, f"{mm} mm", color=color, ha="center",
            va="bottom" if yl[0] > yl[1] else "bottom", fontsize=7.5,
            transform=ax.transData, clip_on=False)


def orient_marks(ax, top, bottom, left, right, color="#C9D3D8"):
    kw = dict(color=color, fontsize=8, transform=ax.transAxes, fontweight="bold")
    ax.text(0.5, 0.985, top, ha="center", va="top", **kw)
    ax.text(0.5, 0.015, bottom, ha="center", va="bottom", **kw)
    ax.text(0.015, 0.5, left, ha="left", va="center", **kw)
    ax.text(0.985, 0.5, right, ha="right", va="center", **kw)

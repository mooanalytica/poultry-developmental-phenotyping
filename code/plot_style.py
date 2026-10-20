"""Shared figure style taken from the original plotting code (code/figures.py and 28aug/code/p19_notitle_figs.py)."""

import matplotlib.pyplot as plt

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM
FONT_FAMILY = "DejaVu Sans"
MIN_PT = 7.0
ORIG_BASE_PT = 10.0


def fs(orig_pt, g=1.0):
    """Font size at final size: the original size scaled by the figure's scale factor g (final width / original width), never below MIN_PT."""
    return max(MIN_PT, round(orig_pt * g, 2))


BLOCKS = ["morph", "yolo", "flow", "audio_bio", "audio_equip", "time/flags"]
NAME = {
    "morph": "Morphometry",
    "yolo": "Spatial behaviour",
    "flow": "Optical flow",
    "audio_bio": "Bioacoustics",
    "audio_equip": "Acoustic level",
    "time/flags": "Time & data quality",
}
COL = {
    "morph": "#1b4965",
    "yolo": "#3d7ea6",
    "flow": "#8bbdd9",
    "audio_bio": "#d1495b",
    "audio_equip": "#edae49",
    "time/flags": "#bcbcbc",
}
ROOMS = ["room2", "room6", "room7"]
ROOM_COL = {"room2": "#1b4965", "room6": "#d1495b", "room7": "#2a9d6f"}
ROOM_LABEL = {r: r.replace("room", "Room ") for r in ROOMS}
NAVY, LIGHT, RED, AMBER, STEEL = (
    COL["morph"],
    COL["flow"],
    COL["audio_bio"],
    COL["audio_equip"],
    COL["yolo"],
)
BCOL = {
    "backward extrapolation": "#1b4965",
    "internal gap": "#8bbdd9",
    "forward extrapolation": "#d1495b",
}
NEG_CMAP = {"leak": "#c1121f", "ok": "#1b4965", "ctrl": "#9a9a9a"}
MODEL_CLASS_COL = [
    "#1b4965",
    "#3d7ea6",
    "#d1495b",
]
GREY_TEXT, GRID_GREY = "#333333", "#bbbbbb"

POOLED_COL = "#555555"


def rc(g=1.0):
    return {
        "font.family": FONT_FAMILY,
        "font.size": fs(ORIG_BASE_PT, g),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": max(0.3, 0.6 * g),
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "poultry-phase3",
        "axes.labelsize": fs(ORIG_BASE_PT, g),
        "xtick.labelsize": fs(ORIG_BASE_PT, g),
        "ytick.labelsize": fs(ORIG_BASE_PT, g),
        "axes.titlesize": fs(10, g),
        "legend.fontsize": fs(10, g),
        "axes.linewidth": max(0.5, 0.8 * g),
        "xtick.major.width": max(0.4, 0.8 * g),
        "ytick.major.width": max(0.4, 0.8 * g),
        "xtick.major.size": max(2.0, 3.5 * g),
        "ytick.major.size": max(2.0, 3.5 * g),
    }


def apply(g=1.0):
    plt.rcParams.update(rc(g))


def new_figure(orig_w_in, orig_h_in, width_mm, nrows=1, ncols=1, **kw):
    """Figure at width_mm with the ORIGINAL aspect ratio; applies the style scaled by g = final width / original width. Returns (fig, axes, g)."""
    w = width_mm * MM
    h = w * orig_h_in / orig_w_in
    g = w / orig_w_in
    apply(g)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, axes, g


def lw(orig, g):
    return max(0.6, orig * g)


def ms(orig, g):
    return max(2.5, orig * g)


def sa(orig_area, g):
    return max(1.2, orig_area * g * g)

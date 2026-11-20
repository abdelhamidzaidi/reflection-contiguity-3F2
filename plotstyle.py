"""
plotstyle.py
============

Common Matplotlib settings for the figures of the manuscript: serif fonts
(STIX, close to the Palatino text font of the MDPI template), a width of
13.5 cm for full-width figures, and vector PDF output in figures.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
WIDTH_IN = 13.5 / 2.54          # full text width of the MDPI template, in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    """Save `fig` as figures/<name>.pdf and return the path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    return os.path.normpath(path)

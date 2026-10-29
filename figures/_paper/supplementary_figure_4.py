"""Supplementary Fig. 4: computational scaling.

a, mean-feature extraction; b, graph construction; c, GraphMAE training; d, peak memory.
Values are read from source_data/supplementary_figure_4.xlsx.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

OK = dict(orange="#E69F00", bluish_green="#009E73", blue="#0072B2", reddish_purple="#CC79A7")
GRAPH_RULE = {"grid8": ("8-neighbour grid", OK["blue"]), "knn10": ("k-NN, k = 10", OK["bluish_green"]),
              "knn40": ("k-NN, k = 40", OK["orange"]), "radius4patch": ("Radius, 4 patch widths", OK["reddish_purple"])}
PSIZE_MARK = {5: "o", 10: "s", 20: "^"}

FS_LAB, FS_TICK, FS_LEG, FS_LET = 6.5, 6.0, 6.0, 8.0
MM = 1 / 25.4


def use_style():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": FS_LAB,
        "axes.labelsize": FS_LAB, "axes.titlesize": FS_LAB, "legend.fontsize": FS_LEG,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.linewidth": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.0, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left", "axes.unicode_minus": True,
    })


class Page:
    """Fixed-size page in millimetres; positions measured from the top-left corner."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = float(w_mm), float(h_mm)
        self.fig = plt.figure(figsize=(self.W * MM, self.H * MM))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def letter(self, x, y, s):
        return self.fig.text(x / self.W, 1 - y / self.H, s, ha="left", va="top", fontsize=FS_LET, fontweight="bold")


def pts(ax, df, xcol, ycol, scol=None, colour=None, filled=True, scale=1.0, ms=3.0):
    """Mean and s.d. per configuration; colour shows the graph rule, marker the patch size."""
    for _, r in df.iterrows():
        col = colour or GRAPH_RULE[r.graph][1]
        yv = r[ycol] * scale; ye = (r[scol] * scale) if scol else None
        ax.errorbar(r[xcol], yv, yerr=ye, fmt=PSIZE_MARK[int(r.patch_size)], ms=ms, mfc=col if filled else "white", mec=col,
                    mew=0.6, ecolor=col, elinewidth=0.5, capsize=0.8, zorder=3 if filled else 4)


def plot(source_data_dir: Path, **kw) -> plt.Figure:
    """Draw Supplementary Fig. 4 (131 x 97 mm)."""
    book = Path(source_data_dir) / "supplementary_figure_4.xlsx"
    pub = pd.read_excel(book, sheet_name="a_d_public")
    gine = pd.read_excel(book, sheet_name="c_d_application")
    use_style()
    GiB = 2.0 ** 30
    pg = Page(131, 97.0)
    X = {0: 13.0, 1: 78.5}; PW = 50.0
    Y = {0: 13.0, 1: 54.0}; PH = 27.0

    def ax_at(r, c):
        return pg.ax(X[c], Y[r], PW, PH)

    ax = ax_at(0, 0)
    g8 = pub[pub.graph == "grid8"]
    pts(ax, g8, "nodes", "mean_feature_extraction_seconds_mean", "mean_feature_extraction_seconds_std")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Candidate patch nodes"); ax.set_ylabel("Wall time (s)")
    ax.set_title("Mean-feature extraction, ten channels", fontsize=FS_LAB, pad=3)
    pg.letter(1.0, Y[0] - 4.5, "a")

    ax = ax_at(0, 1)
    pts(ax, pub, "directed_edges", "graph_construction_seconds_mean", "graph_construction_seconds_std")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Directed graph edges"); ax.set_ylabel("Wall time (s)")
    ax.set_title("Graph construction (public API)", fontsize=FS_LAB, pad=3)
    pg.letter(66.5, Y[0] - 4.5, "b")

    ax = ax_at(1, 0)
    pts(ax, pub, "directed_edges", "graphmae_training_seconds_mean", "graphmae_training_seconds_std")
    pts(ax, gine, "directed_edges", "gine_training_seconds_mean", "gine_training_seconds_std", colour="black")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Directed graph edges"); ax.set_ylabel("Wall time (s)")
    ax.set_title("GraphMAE training", fontsize=FS_LAB, pad=3)
    ax.set_ylim(0.2, 300)
    pg.letter(1.0, Y[1] - 4.5, "c")

    ax = ax_at(1, 1)
    pts(ax, pub, "directed_edges", "sampled_peak_rss_bytes_mean", "sampled_peak_rss_bytes_std", filled=False, scale=1 / GiB)
    pts(ax, pub, "directed_edges", "cuda_pipeline_peak_allocated_bytes_mean", "cuda_pipeline_peak_allocated_bytes_std", scale=1 / GiB)
    pts(ax, gine, "directed_edges", "cuda_training_peak_allocated_bytes_mean", "cuda_training_peak_allocated_bytes_std",
        colour="black", scale=1 / GiB)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Directed graph edges"); ax.set_ylabel("Peak memory (GiB)")
    ax.set_title("Peak memory", fontsize=FS_LAB, pad=3)
    pg.letter(66.5, Y[1] - 4.5, "d")

    lax = pg.ax(2, 0.5, 127, 5.5); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.0, mfc=c, mec=c, label=n) for n, c in GRAPH_RULE.values()],
               ncol=4, loc="center", fontsize=FS_LEG, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    lax = pg.ax(2, Y[1] + PH + 8.0, 127, 8.0); lax.set_axis_off()
    p5, p10, p20 = [Line2D([], [], ls="", marker=PSIZE_MARK[p], ms=3.0, mfc="#555555", mec="#555555", label=f"{p}-px patches")
                    for p in (5, 10, 20)]
    gi = Line2D([], [], ls="", marker="o", ms=3.0, mfc="black", mec="black", label="Application GINE (c, d)")
    rss = Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec="#555555", mew=0.6, label="Process RSS (d)")
    cuda = Line2D([], [], ls="", marker="o", ms=3.0, mfc="#555555", mec="#555555", label="CUDA allocated (d)")
    lax.legend(handles=[p5, gi, p10, rss, p20, cuda], ncol=3, loc="center", fontsize=FS_LEG, handletextpad=0.3,
               columnspacing=1.2, labelspacing=0.35, borderaxespad=0)
    return pg.fig

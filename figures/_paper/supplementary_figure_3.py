"""Supplementary Fig. 3: lung ECM clusters, cell-ECM association and graphs of all six ROIs."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ECM_PAL = {0: "#785EF0", 1: "#2A9D8F", 2: "#E9A820"}
EDGE = {"cc": "#1f4e9b", "ee": "#2d7a2d", "ce": "#c64a8c"}
CELL_PAL = {
    "AEC": "#d62728", "ATI": "#3cb44b", "ATII": "#4363d8", "Alveolar Fibroblast": "#f58231",
    "Alveolar Macrophages": "#911eb4", "B Cells": "#42d4f4", "CD11b Immune": "#f032e6",
    "Endothelial": "#bfef45", "Fibroblast": "#fabed4", "Neutrophils": "#469990",
    "Peribronchial Fibroblasts": "#dcbeff", "Smooth Muscle": "#9A6324",
}
NICHE = {0: "alveolar", 1: "bronchovascular wall", 2: "adventitial"}
LAYER = {"cc": "cell-cell", "ee": "ECM-ECM", "ce": "cell-ECM"}
MID, GUIDE = "#7F7F7F", "#BFBFBF"
FS_LAB, FS_TICK, FS_LEG, FS_NOTE, FS_LET = 6.5, 6.0, 6.0, 5.5, 8.0
MM = 1 / 25.4
MINUS = "\u2212"
PATCH = 10
W, H = 180.0, 160.0


def use_style():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": FS_LAB,
        "axes.labelsize": FS_LAB, "axes.titlesize": FS_LAB, "legend.fontsize": FS_LEG,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.linewidth": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.0, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left", "axes.unicode_minus": True,
        "hatch.linewidth": 0.5,
    })


class Page:
    """Fixed-size page in millimetres; positions measured from the top-left corner."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = float(w_mm), float(h_mm)
        self.fig = plt.figure(figsize=(self.W * MM, self.H * MM))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_LAB)
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def letter(self, x, y, s, title=None):
        self.text(x, y, s, ha="left", va="top", fontsize=FS_LET, fontweight="bold")
        if title:
            self.text(x + 3.0, y + 0.35, title, ha="left", va="top", fontsize=FS_LAB)


def fmt_signed(v, digits=1):
    return f"{v:+.{digits}f}".replace("-", MINUS)


def frame(ax, lw=0.5):
    """Thin black box around a spatial panel (no ticks)."""
    ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_edgecolor("black")


def panel_a(pg, scan):
    """Silhouette and cross-seed ARI against the number of K-means components."""
    pg.letter(1.0, 1.0, "a", "Number of K-means components")
    ax = pg.ax(12.5, 9.0, 40.0, 30.0); scan = scan.sort_values("total_components")
    K = scan.total_components.to_numpy(int); sil = scan.silhouette.to_numpy(float); ari = scan.cross_seed_ari.to_numpy(float)
    ax.plot(K, sil, color="black", marker="o", ms=2.6, lw=0.9, zorder=3); ksel = int(K[np.argmax(sil)])
    ax.plot([ksel], [sil.max()], ls="", marker="o", ms=6.5, mfc="none", mec="black", mew=0.7, zorder=4)
    ax.annotate(f"Selected: {ksel}\n(background + 3 signal)", xy=(ksel, sil.max()), xytext=(2.95, 0.338), fontsize=FS_NOTE, ha="left", va="top",
                linespacing=1.05, arrowprops=dict(arrowstyle="-", lw=0.5, color=MID, shrinkA=0, shrinkB=3.2))
    ax.set_xticks(K); ax.set_xlim(2.6, 7.4); ax.set_ylim(0.28, 0.50); ax.set_yticks([0.3, 0.4, 0.5])
    ax.set_xlabel("Total K-means components (one background)"); ax.set_ylabel("Signal-patch silhouette")
    ax2 = ax.twinx(); ax2.plot(K, ari, color=MID, marker="s", ms=2.2, lw=0.8, ls=(0, (3, 1.5)), zorder=2)
    ax2.set_ylim(0.984, 1.0005); ax2.set_yticks([0.985, 0.99, 0.995, 1.0]); ax2.set_yticklabels(["0.985", "0.990", "0.995", "1.000"])
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(MID); ax2.tick_params(axis="y", colors=MID); ax2.set_ylabel("Cross-seed ARI", color=MID)


def panel_b(pg, rules):
    """AEC association with each ECM cluster under random edge rules and the Fig. 4 rule."""
    pg.letter(66.0, 1.0, "b", "AEC association across cell–ECM edge rules")
    ax = pg.ax(79.0, 9.0, 36.0, 30.0)
    ref = rules[rules.is_reference]; rnd = rules[~rules.is_reference]
    n_rules = rnd.recipe_index.nunique(); rng = np.random.default_rng(20261028)
    for k in range(3):
        v = rnd[rnd.cluster == f"ECM {k}"].log2_obs_exp.replace(-np.inf, np.nan).dropna().to_numpy()
        ax.scatter(k + rng.uniform(-0.16, 0.16, v.size), v, s=3.2, color=ECM_PAL[k], alpha=0.75, lw=0, zorder=2)
        rv = float(ref[ref.cluster == f"ECM {k}"].log2_obs_exp.iloc[0])
        ax.plot([k], [rv], ls="", marker="*", ms=6.5, mfc="black", mec="white", mew=0.3, zorder=5)
    ax.axhline(0, color="black", lw=0.6, zorder=1); ax.set_xticks(range(3)); ax.set_xticklabels([f"ECM {k}" for k in range(3)]); ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-6, 4.3); ax.set_yticks([-6, -4, -2, 0, 2, 4]); ax.set_ylabel("AEC log2(observed / expected)")
    lax = pg.ax(67.0, 42.0, 52.0, 5.0); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.2, mfc=MID, mec="none", label=f"{n_rules} random edge rules"),
                        Line2D([], [], ls="", marker="*", ms=5.5, mfc="black", mec="white", mew=0.3, label="Fig. 4 rule")],
               ncol=2, loc="center", fontsize=FS_NOTE, handletextpad=0.25, columnspacing=1.0, borderaxespad=0)


def panel_c(pg, assoc):
    """Cell type x ECM cluster log2(observed / expected) values."""
    piv = assoc.pivot(index="cell_type", columns="cluster", values="log2_obs_exp")[[f"ECM {k}" for k in range(3)]].replace([-np.inf], np.nan)
    order = piv.sort_values("ECM 1", ascending=False).index.tolist(); piv = piv.loc[order]; nrow = len(order)
    pg.letter(124.0, 1.0, "c", "Cell type × ECM cluster association")
    bx, bw, y0, rowh = 152.0, 26.0, 9.5, 2.75
    ax = pg.ax(bx, y0, bw, nrow * rowh); im = ax.imshow(piv.to_numpy(float), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto", interpolation="nearest")
    for i in range(nrow):
        for j in range(3):
            val = piv.iloc[i, j]
            if not np.isfinite(val):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor=GUIDE, hatch="//////", lw=0)); continue
            ax.text(j, i, fmt_signed(val), ha="center", va="center", fontsize=FS_NOTE, color="white" if abs(val) > 2.4 else "black")
    for x in (0.5, 1.5):
        ax.axvline(x, color="white", lw=0.6)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(order, fontsize=FS_NOTE); ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"ECM {k}" for k in range(3)], fontsize=FS_TICK, fontweight="bold"); ax.xaxis.tick_top()
    for t_, k in zip(ax.get_xticklabels(), range(3)):
        t_.set_color(ECM_PAL[k])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = pg.ax(bx, y0 + nrow * rowh + 2.2, bw, 1.5); cb = pg.fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-4, -2, 0, 2, 4])
    cb.ax.tick_params(labelsize=FS_NOTE, length=1.5, pad=0.8); cb.outline.set_linewidth(0.4); cb.set_label("log2(observed / expected)", fontsize=FS_NOTE, labelpad=1.0)


def draw_graph(ax, cells, patches, edges, layers):
    """Draw the cell, ECM or joint graph of one ROI."""
    pxy = patches[["x", "y"]].to_numpy(float); cxy = cells[["x", "y"]].to_numpy(float)
    xy = pd.concat([cells, patches]).set_index("node")[["x", "y"]]
    if "ee" in layers:
        sig = patches[patches.label != "background"]
        ax.scatter(sig.x, sig.y, s=0.9, marker="s", c=[ECM_PAL[int(l[-1])] for l in sig.label], linewidths=0, zorder=2,
                   alpha=(0.55 if ("cc" in layers or "ce" in layers) else 0.8))
    for k in layers:
        e = edges[edges.layer == LAYER[k]]
        seg = np.stack([xy.loc[e.source].to_numpy(float), xy.loc[e.target].to_numpy(float)], axis=1)
        ax.add_collection(LineCollection(seg, colors=EDGE[k], linewidths={"cc": 0.25, "ee": 0.08, "ce": 0.15}[k],
                                         alpha={"cc": 0.9, "ee": 0.35, "ce": 0.9}[k], zorder={"cc": 1.5, "ee": 2.5, "ce": 2.6}[k]))
    if "cc" in layers or "ce" in layers:
        ax.scatter(cxy[:, 0], cxy[:, 1], s=0.45, c=[CELL_PAL[t] for t in cells.label], linewidths=0, zorder=3)
    ax.set_xlim(pxy[:, 0].min() - PATCH, pxy[:, 0].max() + PATCH); ax.set_ylim(pxy[:, 1].max() + PATCH, pxy[:, 1].min() - PATCH)
    ax.set_aspect("equal"); frame(ax, lw=0.4)


def panels_d_f(pg, nodes, edges):
    """Cell (d), ECM (e) and joint (f) graphs of all six ROIs, with legends."""
    gy0 = 58.0; gap = 1.4; x0 = 13.0; tw = (W - x0 - 1.0 - 5 * gap) / 6; th = tw; rowgap = 2.0
    rows = [("d", "Cell graph", ["cc"]), ("e", "ECM graph", ["ee"]), ("f", "Joint graph", ["cc", "ee", "ce"])]
    for j, r in enumerate(sorted(nodes.roi.unique())):
        n_ = nodes[nodes.roi == r]; e_ = edges[edges.roi == r]
        cells = n_[n_.kind == "cell"]; patches = n_[n_.kind == "ECM patch"]
        xx = x0 + j * (tw + gap)
        pg.text(xx + tw / 2, gy0 - 1.0, f"ROI {r[-3:]} · {cells.mouse.iloc[0]}", ha="center", va="bottom", fontsize=FS_NOTE)
        for i, (_, _, lay) in enumerate(rows):
            axg = pg.ax(xx, gy0 + i * (th + rowgap), tw, th); draw_graph(axg, cells, patches, e_, lay)
            if j == 5 and i == 2:
                xb = patches.x.max() - 110; yb = patches.y.max() - 12
                axg.plot([xb, xb + 100], [yb, yb], color="k", lw=1.5, solid_capstyle="butt", zorder=6)
    for i, (L_, nm, _) in enumerate(rows):
        yy = gy0 + i * (th + rowgap); pg.letter(1.0, yy - 0.5, L_)
        pg.text(9.0, yy + th / 2, nm, rotation=90, ha="center", va="center", fontsize=FS_NOTE)
    ly = gy0 + 3 * (th + rowgap) + 0.5
    lax = pg.ax(10, ly, 168, 9); lax.set_axis_off()
    hc = [Line2D([], [], marker="o", ls="", color=CELL_PAL[t], ms=3, label=t) for t in sorted(CELL_PAL)]
    he = [Line2D([], [], marker="s", ls="", color=ECM_PAL[k], ms=3.5, label=f"ECM {k} ({NICHE[k]})") for k in range(3)]
    hd = [Line2D([], [], color=EDGE[k], lw=1.0, label=n) for k, n in (("cc", "Cell–cell"), ("ee", "ECM–ECM"), ("ce", "Cell–ECM"))]
    l1 = lax.legend(handles=hc, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6, fontsize=FS_NOTE, handletextpad=0.2, columnspacing=0.8); lax.add_artist(l1)
    lax.legend(handles=he + hd, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=6, fontsize=FS_NOTE, handletextpad=0.3, columnspacing=0.9)


def plot(source_data_dir: Path, **kw) -> plt.Figure:
    """Draw Supplementary Fig. 3 from source_data/supplementary_figure_3.xlsx."""
    book = Path(source_data_dir) / "supplementary_figure_3.xlsx"
    sheets = pd.read_excel(book, sheet_name=["a_components", "b_edge_rules", "c_association", "d_f_nodes", "d_f_edges"])
    use_style(); pg = Page(W, H)
    panel_a(pg, sheets["a_components"])
    panel_b(pg, sheets["b_edge_rules"])
    panel_c(pg, sheets["c_association"])
    panels_d_f(pg, sheets["d_f_nodes"], sheets["d_f_edges"])
    return pg.fig

"""Fig. 4: cell-matrix association and recovery of ECM labels from cell context in healthy mouse lung (panels a-j).

All panels are drawn from ``source_data/fig4.xlsx``. Panel g shows the fibronectin channel of ROI 001 from the raw
image (Zenodo 10.5281/zenodo.21538382, loaded with ``mantpy.datasets.balbc_pbs_lung``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
from scipy import ndimage

RC = {"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
      "legend.fontsize": 7, "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False,
      "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
      "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.style": "normal", "figure.dpi": 150}
MIN_FONT = 7.5  # smallest text size on the final page (pt)
MM = 1 / 25.4
PATCH = 10      # ECM patch side (pixels)

ECM_PAL = {0: "#785EF0", 1: "#2A9D8F", 2: "#E9A820"}
EDGE = {"cell-cell": "#1f4e9b", "ECM-ECM": "#2d7a2d", "cell-ECM": "#c64a8c"}
NICHE = {0: "alveolar", 1: "bronchovascular wall", 2: "adventitial"}
CELL_PAL = {"AEC": "#d62728", "ATI": "#3cb44b", "ATII": "#4363d8", "Alveolar Fibroblast": "#f58231",
            "Alveolar Macrophages": "#911eb4", "B Cells": "#42d4f4", "CD11b Immune": "#f032e6", "Endothelial": "#bfef45",
            "Fibroblast": "#fabed4", "Neutrophils": "#469990", "Peribronchial Fibroblasts": "#dcbeff", "Smooth Muscle": "#9A6324"}
CTX_COL = {"cell context": "#D55E00", "cell + corrupt ECM": "#E69F00", "corrupt ECM": "#785EF0"}
CTX_LABEL = {"cell context": "Cell\ncontext", "cell + corrupt ECM": "Cell +\ncorrupt ECM", "corrupt ECM": "Corrupt\nECM"}
MOUSE_MARK = {"mouse 1": "o", "mouse 2": "^"}


def load_fibronectin(cache_dir=None) -> np.ndarray:
    """Fibronectin channel of lung ROI 001 from Mantpy's lung loader (downloaded on first use)."""
    import mantpy as mt
    data = mt.datasets.balbc_pbs_lung(cache_dir=cache_dir, progressbar=False)
    panel = data.panel.copy()
    panel.columns = [str(c).strip() for c in panel.columns]
    kept = panel[panel["keep"] == 1].reset_index(drop=True)
    channel = int(np.flatnonzero(kept["name"].astype(str).str.strip() == "Fibronectin")[0])
    return np.asarray(data.images["roi_001"])[channel]


# ------------------------------------------------------------------ helpers
def title_letter(ax, s: str, gap_mm: float = 1.2) -> None:
    """Panel letter on the baseline of the axes' left title, gap_mm to the left of the axes."""
    tr = ax.transAxes + ax.titleOffsetTrans + ScaledTranslation(-gap_mm * MM, 0, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, s, transform=tr, fontsize=10, fontweight="bold", ha="right", va="baseline")


def scalebar(ax, length: float = 100, label: str = "100 µm") -> None:
    """Scale bar inset from the lower-right corner, label above the bar, on a white backing."""
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    thick = 1.5 * abs(np.diff(ax.get_xlim())[0]) / w_pt
    sb = AnchoredSizeBar(ax.transData, length, label, loc="lower right", pad=0.3, borderpad=0.7, sep=1.5, frameon=True,
                         size_vertical=thick, color="black", label_top=True, fontproperties=fm.FontProperties(size=6))
    sb.patch.set_facecolor("white"); sb.patch.set_edgecolor("none"); sb.patch.set_alpha(0.85)
    ax.add_artist(sb)


def frame(ax, lw: float = 0.5) -> None:
    ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_edgecolor("black")


def spatial_limits(ax, pxy) -> None:
    ax.set_xlim(pxy[:, 0].min() - PATCH, pxy[:, 0].max() + PATCH); ax.set_ylim(pxy[:, 1].max() + PATCH, pxy[:, 1].min() - PATCH)
    ax.set_aspect("equal"); frame(ax)


# ------------------------------------------------------------------ panels
def plot_graph(ax, nodes: pd.DataFrame, edges: pd.DataFrame, layers, title: str) -> None:
    """a-c: one or more graph layers of ROI 001 (cells as dots, ECM patches as squares)."""
    cells, patches = nodes[nodes.kind == "cell"], nodes[nodes.kind == "ECM patch"]
    xy = nodes.set_index("node")[["x", "y"]]
    if "ECM-ECM" in layers:
        sig = patches[patches.label != "background"]
        ax.scatter(sig.x, sig.y, s=(PATCH * 0.55) ** 2 * 0.18, marker="s", c=[ECM_PAL[int(k[-1])] for k in sig.label],
                   linewidths=0, zorder=2, alpha=0.55 if ("cell-cell" in layers or "cell-ECM" in layers) else 0.75)
    for k in layers:
        e = edges[edges.layer == k]
        seg = np.stack([xy.loc[e.source].to_numpy(), xy.loc[e.target].to_numpy()], axis=1)
        ax.add_collection(LineCollection(seg, colors=EDGE[k], linewidths={"cell-cell": 0.45, "ECM-ECM": 0.15, "cell-ECM": 0.35}[k],
                                         alpha={"cell-cell": 0.9, "ECM-ECM": 0.35, "cell-ECM": 0.9}[k],
                                         zorder={"cell-cell": 1.5, "ECM-ECM": 2.5, "cell-ECM": 2.6}[k]))
    if "cell-cell" in layers or "cell-ECM" in layers:
        ax.scatter(cells.x, cells.y, s=2.4, c=[CELL_PAL[t] for t in cells.label], edgecolors="black", linewidths=0.2, zorder=3)
    spatial_limits(ax, patches[["x", "y"]].to_numpy(float))
    ax.set_title(title, loc="left")
    if "cell-ECM" in layers:
        scalebar(ax)


def plot_graph_legend(ax) -> None:
    """Legend strip for a-c: cell types, ECM clusters and edge types."""
    ax.set_axis_off()
    cells = [Line2D([], [], marker="o", ls="", color=CELL_PAL[t], ms=4, label=t) for t in sorted(CELL_PAL)]
    ecm = [Line2D([], [], marker="s", ls="", color=ECM_PAL[k], ms=5, label=f"ECM {k} ({NICHE[k]})") for k in range(3)]
    edges = [Line2D([], [], color=EDGE[k], lw=1.2, label=n) for k, n in (("cell-cell", "Cell–cell"), ("ECM-ECM", "ECM–ECM"),
                                                                         ("cell-ECM", "Cell–ECM"))]
    first = ax.legend(handles=cells, loc="lower center", bbox_to_anchor=(0.5, 0.45), ncol=6, handletextpad=0.3, columnspacing=0.9, fontsize=6)
    ax.add_artist(first)
    ax.legend(handles=ecm + edges, loc="upper center", bbox_to_anchor=(0.5, 0.40), ncol=6, handletextpad=0.3, columnspacing=0.9, fontsize=6)


def plot_profiles(ax, prof: pd.DataFrame) -> None:
    """d: ECM-cluster marker profiles (mean pooled z-score per patch)."""
    cent = prof.to_numpy()
    vmax = np.abs(cent).max()
    im = ax.imshow(cent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(-0.5, cent.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, cent.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.4); ax.tick_params(which="minor", length=0); ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_edgecolor("black")
    for i in range(cent.shape[0]):
        for j in range(cent.shape[1]):
            ax.text(j, i, f"{cent[i, j]:+.1f}", ha="center", va="center", fontsize=6, color="white" if abs(cent[i, j]) > 0.55 * vmax else "black")
    ax.set_yticks(range(len(prof))); ax.set_yticklabels([f"ECM {k}" for k in range(len(prof))])
    for t, k in zip(ax.get_yticklabels(), range(len(prof))):
        t.set_color(ECM_PAL[k]); t.set_fontweight("bold")
    ax.set_xticks(range(cent.shape[1])); ax.set_xticklabels(prof.columns, rotation=0)
    ax.set_title("ECM-cluster marker profiles (mean pooled z-score per patch)", loc="left")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("black")


def plot_association(ax, per: pd.DataFrame) -> None:
    """e: AEC association by ECM cluster; bars pooled over ROIs, points per ROI by mouse."""
    floor = -7.4
    for k in range(3):
        d = per[per.cluster == f"ECM {k}"].sort_values("roi")
        pooled = d["log2_obs_exp_pooled"].iloc[0]
        ax.bar(k, pooled, color=ECM_PAL[k], width=0.62, edgecolor="black", linewidth=0.4, zorder=1)
        for x, (m, v) in zip(k + np.linspace(-0.18, 0.18, len(d)), d[["mouse", "log2_obs_exp_roi"]].itertuples(index=False)):
            if np.isfinite(v):
                ax.scatter(x, v, s=11, marker=MOUSE_MARK[m], facecolors="white", edgecolors="black", linewidths=0.6, zorder=3)
            else:
                ax.scatter(x, floor, s=11, marker=MOUSE_MARK[m], facecolors="none", edgecolors="#777777", linewidths=0.6, zorder=3)
                ax.text(x + 0.16, floor, "0 obs.", ha="left", va="center", fontsize=5, color="#555555")
        ax.text(k, 0.18 if pooled > 0 else -0.18, f"{pooled:+.1f}", ha="center", va="bottom" if pooled > 0 else "top", fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"ECM {k}" for k in range(3)])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(floor - 0.7, 4.2)
    ax.set_ylabel("log2(observed / expected)"); ax.set_title("AEC association by ECM cluster", loc="left")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=3.5, label="Mouse 1"),
                       Line2D([], [], ls="", marker="^", mfc="white", mec="black", ms=3.5, label="Mouse 2")],
              loc="upper right", fontsize=6, handletextpad=0.2, borderaxespad=0.2, frameon=False)


def plot_contrast(ax, per: pd.DataFrame) -> None:
    """f: ECM 1 marker difference, patches with minus without an AEC edge; bars mean over ROIs, points per ROI."""
    pooled = per.groupby("marker")["difference_mean_over_rois"].first().sort_values()
    for i, (mk, v) in enumerate(pooled.items()):
        ax.barh(i, v, color=ECM_PAL[1] if v > 0 else "#bdbdbd", edgecolor="black", linewidth=0.4, height=0.66, zorder=1)
        d = per[per.marker == mk].sort_values("roi")
        for y, (m, c) in zip(i + np.linspace(-0.2, 0.2, len(d)), d[["mouse", "difference_roi"]].itertuples(index=False)):
            ax.scatter(c, y, s=8, marker=MOUSE_MARK[m], facecolors="white", edgecolors="black", linewidths=0.5, zorder=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(pooled))); ax.set_yticklabels(pooled.index)
    ax.set_title("AEC-associated ECM 1 markers", loc="left")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=3.5, label="Mouse 1"),
                       Line2D([], [], ls="", marker="^", mfc="white", mec="black", ms=3.5, label="Mouse 2")],
              loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.1, frameon=False)


def plot_overlay(ax, image: np.ndarray, patches: pd.DataFrame, aec_xy: np.ndarray) -> None:
    """g: fibronectin (grey, 3 x 3 median filter) with fibronectin inside AEC-associated ECM 1 patches (green) and AECs (red)."""
    img = ndimage.median_filter(image, size=3, mode="nearest")
    mask = np.zeros(img.shape, bool)
    for x, y in patches[["x", "y"]].to_numpy():
        x0, y0 = int(x - PATCH / 2), int(y - PATCH / 2)
        mask[y0:y0 + PATCH, x0:x0 + PATCH] = True
    hi_all = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1
    ax.imshow(np.clip(img / hi_all, 0, 1), cmap="Greys", vmin=0, vmax=1.8, interpolation="nearest")
    sel = img[mask]
    hi = np.percentile(sel[sel > 0], 99.0) if (sel > 0).any() else 1
    ax.imshow(np.ma.masked_where(~mask, np.clip(img / hi, 0, 1)), cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.scatter(aec_xy[:, 0], aec_xy[:, 1], s=1.5, c="#d62728", linewidths=0)
    frame(ax); ax.set_title("FN, AEC-associated ECM 1", loc="left")
    scalebar(ax)


def plot_labels(ax, t: pd.DataFrame, blocks: pd.DataFrame, column: str, title: str) -> None:
    """h-i: ECM labels of ROI 004, corrupted ('observed') or after reconstruction ('cleaned')."""
    pxy, lab = t[["x", "y"]].to_numpy(float), t[column].to_numpy(int)
    sig = lab >= 0
    ax.scatter(pxy[sig, 0], pxy[sig, 1], s=(PATCH * 0.55) ** 2 * 0.18 * 0.65, marker="s", c=[ECM_PAL[int(k)] for k in lab[sig]], linewidths=0)
    for b in blocks.itertuples():
        ax.add_patch(Rectangle((b.centre_x - b.side / 2, b.centre_y - b.side / 2), b.side, b.side, fill=False, ec="k", lw=0.8))
    if column == "observed":
        iso = t["is_artifact"].to_numpy(bool) & ~t["in_block"].to_numpy(bool)
        ax.scatter(pxy[iso, 0], pxy[iso, 1], s=4, facecolors="none", edgecolors="k", linewidths=0.35)
    spatial_limits(ax, pxy)
    ax.set_title(title, loc="left")
    if column == "cleaned":
        scalebar(ax)


def plot_detection(ax, per: pd.DataFrame) -> None:
    """j: raw (open) and prior-adjusted (filled) detection ROC-AUC per ROI of the held-out mouse; dashed line = label prior."""
    prior = per[per.context == "label prior"]["roc_auc_raw"].mean()
    ctxs = ["cell context", "cell + corrupt ECM", "corrupt ECM"]
    for i, ctx in enumerate(ctxs):
        d = per[per.context == ctx].sort_values("roi"); col = CTX_COL[ctx]
        for off, key, filled in ((-0.17, "roc_auc_raw", False), (0.17, "roc_auc_prior_adjusted", True)):
            v = d[key].to_numpy()
            for x, y, m in zip(i + off + np.linspace(-0.06, 0.06, len(v)), v, d["mouse"]):
                ax.scatter(x, y, s=12, marker=MOUSE_MARK[m], facecolors=col if filled else "white", edgecolors=col, linewidths=0.8, zorder=3)
            ax.hlines(v.mean(), i + off - 0.11, i + off + 0.11, color=col, lw=1.8, zorder=4)
            if filled and ctx == "cell context":
                ax.text(i + off + 0.13, v.mean(), f"{v.mean():.2f}", ha="left", va="center", fontsize=6, color=col)
            elif filled:
                ax.text(i + off, v.min() - 0.03, f"{v.mean():.2f}", ha="center", va="top", fontsize=6, color=col)
            else:
                ax.text(i + off, max(v.max(), v.mean()) + 0.025, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=6, color=col)
    ax.axhline(prior, ls="--", color="#666666", lw=0.8, zorder=1)
    ax.text(0.40, prior - 0.008, f"Label prior ({prior:.2f})", ha="left", va="top", fontsize=6, color="#555555")
    ax.axhline(0.5, ls=":", color="grey", lw=0.8, zorder=1)
    ax.text(2.45, 0.492, "Chance", ha="right", va="top", fontsize=6, color="#777777")
    ax.set_ylim(0.40, 1.06); ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([CTX_LABEL[c] for c in ctxs])
    ax.set_ylabel("Corrupted-label detection\nROC-AUC (held-out mouse)"); ax.set_title("Detecting corrupted labels", loc="left")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec="#444444", ms=4, label="Raw score"),
                       Line2D([], [], ls="", marker="o", mfc="#444444", mec="#444444", ms=4, label="Prior-adjusted")],
              loc="lower left", bbox_to_anchor=(0.0, 0.17), fontsize=6, handletextpad=0.2, borderaxespad=0.2, frameon=False)
    ax.tick_params(length=2.5, width=0.6, pad=1.5)
    ax.yaxis.labelpad = 2.0
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


# ------------------------------------------------------------------ figure
def plot(source_data_dir: Path, fibronectin: np.ndarray | None = None, cache_dir=None) -> mpl.figure.Figure:
    """Draw Fig. 4 (180 x 225 mm) from ``source_data_dir/fig4.xlsx``.

    fibronectin: fibronectin channel of lung ROI 001 (H x W) for panel g; loaded with ``load_fibronectin(cache_dir)``
    when omitted.
    """
    book = Path(source_data_dir) / "fig4.xlsx"
    sheet = lambda name: pd.read_excel(book, sheet_name=name)
    fibronectin = load_fibronectin(cache_dir) if fibronectin is None else fibronectin
    nodes, edges = sheet("a_c_nodes"), sheet("a_c_edges")
    labels, blocks = sheet("h_i_labels"), sheet("h_blocks")
    aec_xy = nodes[(nodes.kind == "cell") & (nodes.label == "AEC")][["x", "y"]].to_numpy(float)
    mpl.rcParams.update(RC)

    fig = plt.figure(figsize=(180 * MM, 225 * MM))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.28, 0.2, 0.55, 0.95, 1.15], hspace=0.36, wspace=0.35,
                          left=0.115, right=0.98, top=0.965, bottom=0.045)
    top = gs[0, :].subgridspec(1, 3, wspace=0.12)
    axa, axb, axc = (fig.add_subplot(top[0, i]) for i in range(3))
    plot_graph(axa, nodes, edges, ["cell-cell"], "Cell graph")
    plot_graph(axb, nodes, edges, ["ECM-ECM"], "ECM graph")
    plot_graph(axc, nodes, edges, ["cell-cell", "ECM-ECM", "cell-ECM"], "Joint cell–ECM graph")
    plot_graph_legend(fig.add_subplot(gs[1, :]))
    axd = fig.add_subplot(gs[2, :]); plot_profiles(axd, sheet("d_profiles").set_index("cluster"))
    axe, axf, axg = (fig.add_subplot(gs[3, i]) for i in range(3))
    plot_association(axe, sheet("e_association"))
    plot_contrast(axf, sheet("f_contrast"))
    plot_overlay(axg, fibronectin, sheet("g_patches"), aec_xy)
    row4 = gs[4, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.28)
    pair = row4[0, 0].subgridspec(1, 2, wspace=0.06)
    axh, axi, axj = fig.add_subplot(pair[0, 0]), fig.add_subplot(pair[0, 1]), fig.add_subplot(row4[0, 1])
    plot_labels(axh, labels, blocks, "observed", "Corrupted input")
    plot_labels(axi, labels, blocks, "cleaned", "Cell-context reconstruction")
    plot_detection(axj, sheet("j_per_roi"))
    for ax, s in zip((axa, axb, axc, axd, axe, axf, axg, axh, axi, axj), "abcdefghij"):
        title_letter(ax, s)

    fig.canvas.draw()
    for artist in fig.findobj(match=mpl.text.Text):
        if artist.get_text() and artist.get_fontsize() < MIN_FONT:
            artist.set_fontsize(MIN_FONT)
    return fig

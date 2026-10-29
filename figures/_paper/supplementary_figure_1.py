"""Supplementary Fig. 1: label-free choice of K, CNN architecture and neighbourhood settings,
and partitions of every method in the two human small intestine images."""
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

WORKBOOK = "supplementary_figure_1.xlsx"
PIXEL_UM = 0.37744

OK = dict(black="#000000", orange="#E69F00", sky_blue="#56B4E9", bluish_green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", reddish_purple="#CC79A7")
CLASSES = ["Muscularis propria", "Muscularis mucosae", "Submucosa", "Crypts", "Villus"]
CLS_COL = [OK["blue"], OK["sky_blue"], OK["orange"], OK["bluish_green"], OK["reddish_purple"]]
REP = dict(pixel="#CFCFCF", patch="#8C8C8C", graph="#7B3294")
GRID, MID, GUIDE = "#E8E8E8", "#7F7F7F", "#BFBFBF"
PICK = OK["vermillion"]
RED = "#C8102E"

FS_LAB, FS_TICK, FS_LEG, FS_NOTE, FS_LET = 6.5, 6.0, 6.0, 5.5, 8.0
MM = 1 / 25.4

W, H = 131.0, 110.0
ROW0, RH, RG = 7.5, 15.5, 8.5
XA, XB, XC, AW = 17.0, 58.0, 105.0, 25.5

MROWS = ["GraphMAE", "CellCharter", "UTAG", "BANKSY"]
READOUT_MARK = {"avgpool1": "o", "avgpool2": "s", "maxpool2": "^"}
MAP_COLS = [("Manual\nannotation", "annotation", None), ("Pixel\nK-means", "pixel_kmeans", "pixel"),
            ("Multi-\nOtsu", "multi_otsu", "patch"), ("Patch-mean\nK-means", "patch_kmeans", "patch"),
            ("CNN\nK-means", "cnn_kmeans", "patch"), ("BANKSY", "banksy", "graph"), ("UTAG", "utag", "graph"),
            ("Cell-\nCharter", "cellcharter", "graph"), ("GraphMAE\n(GIN)", "graphmae", "graph")]


def use_style():
    """Set the paper's matplotlib style."""
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

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_LAB)
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def letter(self, x, y, s):
        return self.text(x, y, s, ha="left", va="top", fontsize=FS_LET, fontweight="bold")


def frame_map(ax, lw=0.6):
    """Remove ticks and draw a thin black frame around a map."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(lw)


def scale_bar(ax, shape, um=1000.0):
    """1-mm bar near the top-right corner of a 16-pixel display map."""
    bar = um / PIXEL_UM / 16; x1 = shape[1] * 0.96; y = shape[0] * 0.06
    ax.plot([x1 - bar, x1], [y, y], color="black", lw=0.8, solid_capstyle="butt")
    ax.text(x1 - bar / 2, y + shape[0] * 0.025, "1 mm", ha="center", va="top", fontsize=5.0)


def panel_k(ax, k, bottom):
    """a: seed stability, silhouette and ARI against the annotation across K."""
    K = k.k.to_numpy()
    ax.plot(K, k.seed_stability, color=OK["blue"], marker="o", ms=1.6, lw=0.7, zorder=3)
    ax.plot(K, k.silhouette, color=OK["orange"], marker="s", ms=1.5, lw=0.7, ls=(0, (3, 1.5)), zorder=3)
    ax.errorbar(K, k.ari_mean, yerr=k.ari_sd, color=OK["bluish_green"], marker="^", ms=1.8, lw=0.7,
                elinewidth=0.45, capsize=0.8, zorder=3)
    known = int(k.loc[k.annotated_k, "k"].iloc[0]); pick = int(k.loc[k.label_free_k, "k"].iloc[0])
    ax.axvline(known, color=RED, lw=0.9, zorder=1)
    ax.axvline(pick, color="black", lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
    ax.set_xlim(1.6, 8.4); ax.set_ylim(0, 1.03); ax.set_xticks(range(2, 9, 2))
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Score")
    if bottom:
        ax.set_xlabel("Number of clusters, K")


def panel_cnn(ax, dd, bottom):
    """b: CNN K-means seed stability of the twelve candidate architectures."""
    for r in dd.itertuples():
        if r.selected:
            ax.scatter([r.architecture], [r.stability], marker=READOUT_MARK[r.readout], s=11, c=PICK,
                       edgecolors="black", lw=0.45, zorder=5)
        else:
            ax.scatter([r.architecture], [r.stability], marker=READOUT_MARK[r.readout], s=6, c=MID, lw=0, zorder=3)
    for xv in (2.5, 8.5):
        ax.axvline(xv, color=GRID, lw=0.5, zorder=1)
    ax.axvline(5.5, color=GUIDE, lw=0.6, zorder=1)
    ax.set_xlim(-0.7, 11.7); ax.set_ylim(0.35, 1.0); ax.set_yticks([0.4, 0.7, 1.0]); ax.set_yticklabels(["0.4", "0.7", "1"])
    ax.set_xticks([1, 4, 7, 10]); ax.set_xticklabels(["w16", "w32", "w16", "w32"]); ax.tick_params(axis="x", length=0, pad=1.0)
    if bottom:
        ax.text(2.5, -0.22, "2 blocks", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_NOTE)
        ax.text(8.5, -0.22, "3 blocks", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_NOTE)
    ax.set_ylabel("CNN K-means\nseed stability", linespacing=1.0)


def panel_settings(ax, S, bottom):
    """c: seed stability of each spatial method's neighbourhood settings."""
    for j, m in enumerate(MROWS):
        g = S[(S.method == m) & S.stability.notna()]
        o = g[~g.selected]; ax.scatter(o.stability, np.full(len(o), j), s=4, c=MID, lw=0, zorder=3)
        q = g[g.selected]; ax.scatter(q.stability, [j], marker="D", s=10, c=PICK, edgecolors="black", lw=0.45, zorder=5)
    ax.set_ylim(len(MROWS) - 0.4, -0.6); ax.set_xlim(0.0, 1.03); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks(range(len(MROWS))); ax.set_yticklabels(MROWS, fontsize=FS_NOTE); ax.tick_params(axis="y", length=0)
    if bottom:
        ax.set_xlabel("Seed stability")


def legends_abc(pg, yL):
    """Legends below panels a, b and c."""
    lax = pg.ax(XA - 6, yL, 40, 6.0); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], color=OK["blue"], marker="o", ms=1.8, lw=0.7, label="Seed stability"),
                        Line2D([], [], color=OK["orange"], marker="s", ms=1.7, lw=0.7, ls=(0, (3, 1.5)), label="Silhouette"),
                        Line2D([], [], color=OK["bluish_green"], marker="^", ms=2.0, lw=0.7, label="ARI vs annotation"),
                        Line2D([], [], color=RED, lw=0.9, label="Annotated K"),
                        Line2D([], [], color="black", lw=0.8, ls=(0, (2.2, 1.6)), label="Label-free K")],
               ncol=2, loc="upper left", fontsize=FS_NOTE, handlelength=1.6, handletextpad=0.3, columnspacing=0.6,
               borderaxespad=0, labelspacing=0.25)
    lax = pg.ax(XB - 7, yL, 40, 6.0); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], ls="", marker=m, ms=2.4, mfc=MID, mec="none", label=l) for m, l in
                        (("o", "Avg pool 1×1"), ("s", "Avg pool 2×2"), ("^", "Max pool 2×2"))] +
               [Line2D([], [], ls="", marker="o", ms=2.8, mfc=PICK, mec="black", mew=0.4, label="Selected")],
               ncol=2, loc="upper left", fontsize=FS_NOTE, handletextpad=0.15, columnspacing=0.6, borderaxespad=0,
               labelspacing=0.25)
    lax = pg.ax(XC - 2, yL, 28, 6.0); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.0, mfc=MID, mec="none", label="Setting"),
                        Line2D([], [], ls="", marker="D", ms=2.6, mfc=PICK, mec="black", mew=0.4, label="Selected")],
               ncol=2, loc="upper left", fontsize=FS_NOTE, handletextpad=0.2, columnspacing=0.8, borderaxespad=0)


def display_maps(cells, shape):
    """Integer maps per method: 0 background, 1 not analysed / not annotated, 2+ annotation class."""
    yy, xx = cells.y_cell.to_numpy(int), cells.x_cell.to_numpy(int)
    maps = {}
    for _, key, _ in MAP_COLS:
        m = np.zeros(shape, np.int16)
        m[yy, xx] = [CLASSES.index(s) + 2 if s in CLASSES else 1 for s in cells[key].to_numpy()]
        maps[key] = m
    return maps


def panel_maps(pg, maps, shapes, yf):
    """d: partitions of every method, matched to the annotation classes for display."""
    mx0, gap = 8.0, 0.7; mw = (W - mx0 - 0.5 - 8 * gap) / 9
    cmap = ListedColormap(["white", GRID] + CLS_COL)
    pg.letter(1.0, yf - 8.6, "d"); pg.text(4.6, yf - 8.4, "Partitions of every method", ha="left", va="top", fontsize=FS_LEG, color=MID)
    y = yf
    for n in (1, 2):
        mh = mw * shapes[n][0] / shapes[n][1]
        for j, (title, key, lev) in enumerate(MAP_COLS):
            x = mx0 + j * (mw + gap); ax = pg.ax(x, y, mw, mh)
            ax.imshow(maps[n][key], cmap=cmap, vmin=-0.5, vmax=cmap.N - 0.5, interpolation="nearest"); frame_map(ax, lw=0.5)
            if n == 1:
                pg.text(x + mw / 2, y - 1.4, title, ha="center", va="bottom", fontsize=FS_NOTE, linespacing=1.0)
                if lev:
                    ax.add_patch(Rectangle((0, 1), 1, 0.03, transform=ax.transAxes, color=REP[lev], clip_on=False))
            if j == 0 and n == 1:
                scale_bar(ax, shapes[n])
        pg.text(mx0 - 0.6, y + mh / 2, f"Human small\nintestine {n}", ha="right", va="center", rotation=90,
                fontsize=FS_NOTE, linespacing=1.0, multialignment="center")
        y += mh + 1.0
    lax = pg.ax(mx0, y + 0.6, W - mx0 - 0.5, 3.0); lax.set_axis_off()
    lax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=nm) for c, nm in zip(CLS_COL, CLASSES)],
               ncol=5, loc="center", fontsize=FS_NOTE, handlelength=1.0, handleheight=0.8, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0)


def plot(source_data_dir: pathlib.Path, **kw) -> mpl.figure.Figure:
    """Draw Supplementary Fig. 1 from its source-data workbook (extra keyword arguments are ignored)."""
    wb = pathlib.Path(source_data_dir) / WORKBOOK
    K = pd.read_excel(wb, sheet_name="a_k")
    C = pd.read_excel(wb, sheet_name="b_cnn")
    S = pd.read_excel(wb, sheet_name="c_settings")
    sh = pd.read_excel(wb, sheet_name="d_map_shape").set_index("image")
    names = {n: f"Human small intestine {n}" for n in (1, 2)}
    shapes = {n: (int(sh.loc[names[n], "n_rows"]), int(sh.loc[names[n], "n_cols"])) for n in (1, 2)}
    maps = {n: display_maps(pd.read_excel(wb, sheet_name=f"d_maps_image{n}"), shapes[n]) for n in (1, 2)}

    use_style()
    pg = Page(W, H)
    rows_y = [ROW0, ROW0 + RH + RG]
    for i, n in enumerate((1, 2)):
        y, bottom = rows_y[i], i == 1
        pg.text(7.6, y + RH / 2, f"Human small\nintestine {n}", ha="right", va="center", rotation=90, fontsize=FS_NOTE,
                linespacing=1.0, multialignment="center")
        panel_k(pg.ax(XA, y, AW, RH), K[K.image == names[n]].sort_values("k"), bottom)
        panel_cnn(pg.ax(XB, y, AW, RH), C[C.image == names[n]].sort_values("architecture"), bottom)
        panel_settings(pg.ax(XC, y, AW, RH), S[S.image == names[n]].sort_values("setting"), bottom)
    for x, L, title in ((1.0, "a", "Number of clusters"), (45.0, "b", "CNN architecture"), (86.0, "c", "Neighbourhood settings")):
        pg.letter(x, 1.0, L); pg.text(x + 3.6, 1.2, title, ha="left", va="top", fontsize=FS_LEG, color=MID)
    yL = rows_y[1] + RH + 10.5
    legends_abc(pg, yL)
    panel_maps(pg, maps, shapes, yL + 19.0)
    return pg.fig

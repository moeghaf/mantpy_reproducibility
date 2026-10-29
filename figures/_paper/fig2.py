"""Fig. 2: collagen IV graphs in two human small-intestine images."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CLASSES = ["Muscularis propria", "Muscularis mucosae", "Submucosa", "Crypts", "Villus"]
CLS_COL = ["#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
REP = dict(pixel="#CFCFCF", patch="#8C8C8C", graph="#7B3294", noedge="#C2A5CF")
REP_EDGE = dict(pixel="#9E9E9E", patch="#6E6E6E", graph="#7B3294")
WHITE, GRID, CONTEXT, POINTS, NEUTRAL_MID, ZERO_LINE = "#FFFFFF", "#E8E8E8", "#E4E4E4", "#262626", "#9E9E9E", "#BDBDBD"
TAG = {"PyG": ("#FEE8D2", "#F47C04"), "AnnData": ("#E3EEFA", "#1464BC"), "NumPy": ("white", "#7F7F7F")}
PIXEL_UM = 0.37744             # native pixel size; maps use 16 x 16-pixel display cells
IMAGES = ("Human small intestine 1", "Human small intestine 2")
CONDITIONS = {"CNN K-means": "patch", "GraphMAE, shuffled edges": "shuffled",
              "GraphMAE, no edges": "noedge", "GraphMAE, full graph": "graph"}

W = 7.087                      # layout units; W maps to the 131-mm figure width
ML, MR = 0.14, 6.95
LW, GX = 0.48, 0.06            # left edge of the map grid and gap between maps
MW = (MR - LW - 4 * GX) / 5
XS = [LW + i * (MW + GX) for i in range(5)]
TOP, TT, GAB, LEG, CG, CH, FOOT = 0.06, 0.17, 0.28, 0.22, 0.42, 1.40, 0.05
LEGH = 0.34                    # height of the panel-d legend above the c/d axes
LETTER = dict(ha="left", va="top", fontsize=8, fontweight="bold")
HEADS = [("Collagen IV", None), ("Manual annotation", None),
         ("Pixel K-means", "pixel"), ("CNN K-means", "patch"), ("GraphMAE", "graph")]
PAGE_X0, PAGE_Y0 = 0.1356, -4.056   # top-left corner of the saved (tight) page on the canvas, in pt
DX0 = 5.00                     # left edge of the panel-d axis


def use_style() -> None:
    """Reset Matplotlib and apply the figure settings."""
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.style": "normal",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.grid": False, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelweight": "normal", "lines.linewidth": 1.2,
        "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def frame_map(ax, lw=0.6):
    """Thin black outline around a map, no ticks."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(lw)


def load(source_data_dir: Path) -> dict:
    """Read the panel tables of fig2.xlsx and rebuild the display grids."""
    book = Path(source_data_dir) / "fig2.xlsx"
    sheet = lambda name: pd.read_excel(book, sheet_name=name)
    grid = sheet("grid").set_index("image")
    code = {name: i for i, name in enumerate(CLASSES)}
    images = []
    for letter, image in zip("ab", IMAGES):
        cells = sheet(f"{letter}_maps")
        shape = (int(grid.loc[image, "rows"]), int(grid.loc[image, "columns"]))
        y, x = cells.y_cell.to_numpy(int), cells.x_cell.to_numpy(int)

        def raster(column):
            values = np.full(shape, -1, np.int16)
            values[y, x] = cells[column].map(code).fillna(-1).to_numpy(np.int16)
            return values

        coll = np.zeros(shape, np.float32)            # background cells: no signal (drawn white)
        coll[y, x] = cells.collagen_iv_mean.to_numpy(np.float32)
        support = np.zeros(shape, bool); support[y, x] = True
        ref = raster("annotation")
        ari = sheet(f"{letter}_ari").set_index("method")
        score = lambda m: (ari.loc[m, "ari_mean"], None if pd.isna(ari.loc[m, "ari_sd"]) else ari.loc[m, "ari_sd"])
        images.append(dict(shape=shape, coll=coll, tis=raster("graphmae") >= 0, background_support=support, ref=ref,
                           maps=dict(pixel=raster("pixel_kmeans"), patch=raster("cnn_kmeans"), graph=raster("graphmae")),
                           domains=sorted(int(c) for c in np.unique(ref) if c >= 0),
                           ari=dict(pixel=score("Pixel K-means"), patch=score("CNN K-means"), graph=score("GraphMAE"))))
    controls = sheet("d_seeds").sort_values(["image", "seed"], kind="stable")
    controls = controls.assign(cond=controls.condition.map(CONDITIONS))
    return dict(I=images[0], I2=images[1], c_ari=sheet("c_ari"), controls=controls)


class Canvas:
    def __init__(self, d):
        use_style()
        self.hI = MW * d["I"]["shape"][0] / d["I"]["shape"][1]
        self.hI2 = MW * d["I2"]["shape"][0] / d["I2"]["shape"][1]
        self.yA = TOP + TT                            # top of the panel-a maps
        self.yB = self.yA + self.hI + GAB             # top of the panel-b maps
        self.yl = self.yB + self.hI2 + 0.27           # layer legend for both images
        self.yC = self.yl + LEG + CG
        self.yN = self.yC + CH + 0.29
        self.wE = (MR - LW - 0.18 - 7 * 0.05) / 9     # panel e: nine maps, two groups
        self.yD = self.yN + 0.13 + 0.12 + 0.37        # top of the panel-e maps
        self.hE = max(self.wE * D["shape"][0] / D["shape"][1] for D in (d["I"], d["I2"]))
        self.H = self.yD + self.hE + FOOT
        self.fig = plt.figure(figsize=(131 / 25.4, self.H * 131 / 180))
        self.Wpt, self.Hpt = 131 / 25.4 * 72, self.H * 131 / 180 * 72

    def ax_in(self, x, yt, w, h):
        return self.fig.add_axes([x / W, 1 - (yt + h) / self.H, w / W, h / self.H])

    def T(self, x, yt, text, **kwargs):
        self.fig.text(x / W, 1 - yt / self.H, text, **kwargs)

    def ax_pt(self, x0, y0, x1, y1):
        """Axes from page coordinates in pt (origin at the top-left corner of the saved page)."""
        return self.fig.add_axes([(x0 + PAGE_X0) / self.Wpt, 1 - (y1 + PAGE_Y0) / self.Hpt,
                                  (x1 - x0) / self.Wpt, (y1 - y0) / self.Hpt])

    def T_pt(self, x, y, text, **kwargs):
        kwargs.setdefault("fontsize", 6)
        self.fig.text((x + PAGE_X0) / self.Wpt, 1 - (y + PAGE_Y0) / self.Hpt, text, **kwargs)


def _row(cv, D, y, h, letter, image, titles):
    cmap = ListedColormap([WHITE, GRID] + CLS_COL)
    for index, (title, key) in enumerate(HEADS):
        ax = cv.ax_in(XS[index], y, MW, h)
        if index == 0:
            lo, hi = np.percentile(D["coll"][D["tis"]], [1, 99.5])
            ax.imshow(D["coll"], cmap="Greys", vmin=lo, vmax=hi, interpolation="nearest")
        else:
            values = D["ref"] if index == 1 else D["maps"][key]
            display = np.where(D["background_support"], 1, 0).astype(np.int16)
            display[values >= 0] = values[values >= 0] + 2
            ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=cmap.N - 0.5, interpolation="nearest")
            if index > 1:
                mean, sd = D["ari"][key]
                text = f"ARI {mean:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")
                cv.T(XS[index] + MW / 2, y + h + 0.03, text.replace("-", "−"), ha="center", va="top", fontsize=6,
                     fontweight="bold" if key == "graph" else "normal")
                ax.add_patch(plt.Rectangle((0, 1), 1, 0.018, transform=ax.transAxes,
                                           color=REP[key], clip_on=False))
        frame_map(ax)
        if titles:
            cv.T(XS[index] + MW / 2, y - 0.045, title, ha="center", va="bottom", fontsize=6.3)
        if index == 0:
            shape = D["shape"]
            bar = 1000 / PIXEL_UM / 16
            ax.plot([shape[1] * 0.05, shape[1] * 0.05 + bar], [shape[0] * 0.97] * 2,
                    color="black", lw=1.2, solid_capstyle="butt")
            ax.text(shape[1] * 0.05 + bar / 2, shape[0] * 0.94, "1 mm",
                    ha="center", va="bottom", fontsize=6)
    cv.T(ML, y - (0.17 if titles else 0.03), letter, **LETTER)
    cv.T(LW - 0.06, y + h / 2, f"Human small\nintestine {image}", ha="right", va="center", fontsize=6.3,
         rotation=90, linespacing=1.15, multialignment="center")


def _layer_legend(cv, y, domains):
    for index, cluster in enumerate(domains):
        x = XS[index]
        ax = cv.ax_in(x, y, 0.07, 0.065)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=CLS_COL[cluster]))
        ax.set_axis_off()
        cv.T(x + 0.10, y + 0.033, CLASSES[cluster], ha="left", va="center", fontsize=6)


def panel_a(cv, d):
    _row(cv, d["I"], cv.yA, cv.hI, "a", 1, titles=True)


def panel_b(cv, d):
    _row(cv, d["I2"], cv.yB, cv.hI2, "b", 2, titles=False)
    _layer_legend(cv, cv.yl, sorted(set(d["I"]["domains"]) | set(d["I2"]["domains"])))


def panel_c(cv, d):
    """Covered-pixel ARI per image: tuned methods (mean ± s.d.), default settings and single-run baselines."""
    table = d["c_ari"]
    rows = table[table.image == IMAGES[0]][["method", "level", "input", "setting"]].to_numpy()
    n = len(rows); pitch = 9.07; row0 = 207.1
    top, bot = row0 - 0.55 * pitch, row0 + (n - 1) * pitch + 0.45 * pitch
    for j, (x0, x1) in enumerate(((95.2, 151.0), (166.8, 222.6))):
        ax = cv.ax_pt(x0, top, x1, bot)
        values = table[table.image == IMAGES[j]].set_index("method")
        for i, (method, level, _, setting) in enumerate(rows):
            col, edge = REP[level], REP_EDGE[level]
            r = values.loc[method]
            mu = r.ari_mean
            ax.plot([0, mu], [i, i], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
            if setting == "tuned":
                ax.errorbar(mu, i, xerr=r.ari_sd, fmt="s", ms=3.0, mfc=col, mec=edge, mew=0.5, ecolor=POINTS,
                            elinewidth=0.5, capsize=1.0, zorder=5)
                if pd.notna(r.ari_default):
                    ax.scatter([r.ari_default], [i], s=11, facecolors="white", edgecolors=edge, lw=0.65, zorder=4)
            else:
                ax.scatter([mu], [i], s=10, c=col, edgecolors=edge, lw=0.5, zorder=4)
        ax.set_ylim(n - 0.45, -0.55); ax.set_yticks([]); ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=6)
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=1.5)
        ax.set_title(f"Human small\nintestine {j + 1}", fontsize=6, loc="center", pad=3, linespacing=1.1)
    for i, (method, _, tag, _) in enumerate(rows):
        y = row0 + i * pitch; fc, ec = TAG[tag]
        cv.T_pt(93.1, y, tag, ha="right", va="center", bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.25", fc=fc, ec=ec, lw=0.5))
        cv.T_pt(65.9, y, method, ha="right", va="center")
    cv.T_pt((95.2 + 222.6) / 2, 290.0, "ARI vs manual annotation", ha="center", va="top", fontsize=6.3)
    cv.T_pt(7.2, 183.4, "c", ha="left", va="top", fontsize=8, fontweight="bold")
    lax = cv.ax_pt(16.0, 185.5, 100, 201); lax.set_axis_off(); lax.set_xlim(0, 1); lax.set_ylim(3, 0)
    for k, (mk, fc, ec, lab) in enumerate([("s", REP["graph"], REP_EDGE["graph"], "Tuned, 12 settings (3 seeds)"),
                                           ("o", "white", REP_EDGE["graph"], "Default settings"),
                                           ("o", REP["patch"], REP_EDGE["patch"], "Single run")]):
        lax.scatter([0.04], [k + 0.5], marker=mk, s=10 if mk == "o" else 9, facecolors=fc, edgecolors=ec, lw=0.6, clip_on=False)
        lax.text(0.10, k + 0.5, lab, ha="left", va="center", fontsize=5.5)


def panel_d(cv, d):
    ax = cv.ax_in(DX0, cv.yC, MR - DX0, CH)
    conditions = [("patch", "CNN K-means"), ("shuffled", "Shuffled edges"),
                  ("noedge", "No edges"), ("graph", "Full graph")]
    offsets = np.linspace(-0.30, 0.30, 4)
    for index, image in enumerate(IMAGES):
        for offset, (condition, _) in zip(offsets, conditions):
            values = d["controls"].query("image == @image and cond == @condition").ari.to_numpy()
            x = index + offset
            ax.bar(x, values.mean(), width=0.17,
                   color="white" if condition == "shuffled" else REP[condition],
                   edgecolor=REP["graph"] if condition == "shuffled" else "none",
                   linewidth=0.6, zorder=2)
            ax.scatter(x + np.linspace(-0.035, 0.035, len(values)), values, s=4,
                       color=POINTS, linewidths=0, zorder=3)
    ax.axhline(0, color=ZERO_LINE, lw=0.6)
    ax.set_xticks([0, 1], ["Human small\nintestine 1", "Human small\nintestine 2"], fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.2, 1)
    ax.set_ylabel("ARI vs manual annotation", fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    handles = [Patch(facecolor="white" if condition == "shuffled" else REP[condition],
                     edgecolor=REP["graph"] if condition == "shuffled" else "none",
                     lw=0.6, label=label) for condition, label in conditions]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1),
              ncol=2, fontsize=6, frameon=False, handlelength=1, columnspacing=0.8)
    cv.T(DX0 - 0.60, cv.yC - LEGH, "d", **LETTER)


def panel_e(cv, d):
    width, x = cv.wE, LW
    head = cv.yD - 0.23
    short = {"Muscularis propria": "M. propria", "Muscularis mucosae": "M. mucosae"}
    for D, label in zip((d["I"], d["I2"]), IMAGES):
        graph, x0 = D["maps"]["graph"], x
        for cluster in D["domains"]:
            height = width * D["shape"][0] / D["shape"][1]
            ax = cv.ax_in(x, cv.yD, width, height)
            values = np.where(D["tis"], 1, 0).astype(np.int16)
            values[graph == cluster] = 2
            ax.imshow(values, cmap=ListedColormap([WHITE, CONTEXT, CLS_COL[cluster]]),
                      vmin=-0.5, vmax=2.5, interpolation="nearest")
            frame_map(ax)
            name = short.get(CLASSES[cluster], CLASSES[cluster])
            cv.T(x + width / 2, cv.yD - 0.02, name, ha="center", va="bottom", fontsize=6)
            x += width + 0.05
        x1 = x - 0.05
        cv.T((x0 + x1) / 2, head, label, ha="center", va="bottom", fontsize=6.3)
        cv.fig.add_artist(Line2D([x0 / W, x1 / W], [1 - (head + 0.03) / cv.H] * 2, transform=cv.fig.transFigure,
                                 color=NEUTRAL_MID, lw=0.5))
        x += 0.13
    cv.T(ML, head - 0.14, "e", **LETTER)


def plot(source_data_dir: Path, **kw) -> plt.Figure:
    """Draw Fig. 2 from source_data/fig2.xlsx (save with the default tight bounding box)."""
    d = load(source_data_dir)
    canvas = Canvas(d)
    for panel in (panel_a, panel_b, panel_c, panel_d, panel_e):
        panel(canvas, d)
    return canvas.fig

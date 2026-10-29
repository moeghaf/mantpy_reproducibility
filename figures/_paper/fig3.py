"""Fig. 3: ECM composition and organisation in naive and Schistosoma mansoni-infected mouse liver (panels a-i).

All panels are drawn from ``source_data/fig3.xlsx``; the marker composites of panel d are made from the raw images
(Zenodo 10.5281/zenodo.21538382, loaded with ``mantpy.datasets.schistosoma_ecm``) using the display scale in the workbook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import gaussian_filter

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.style": "normal",
    "font.size": 7.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0, "legend.fontsize": 6.0,
    "xtick.labelsize": 5.0, "ytick.labelsize": 5.0, "axes.titlelocation": "left",
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
    "xtick.major.size": 3.0, "xtick.major.width": 0.6, "ytick.major.size": 3.0, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "patch.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200.0, "savefig.dpi": 300.0, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
}
SIZES = (9, 8, 7.4)  # font sizes in pt: axis labels and titles, legends, tick labels and annotations
MIN_FONT = 7.5       # smallest text size on the final page (pt)

PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # ECM 0-5
C5 = PAL[5]
INF, NAI = "#333333", "#8C8C8C"
BAND = {"Naive": ("#5B7C99", "white"), "Infected": ("#A6463A", "white"), "KO": ("#4D4D4D", "white"), "WT": ("#D0D0D0", "#111111")}
MK = ["Fibronectin", "Pan-laminin", "CollagenIV", "VersicanV0V1", "CollagenI", "HeparanSulfate"]
MCOL = {"Fibronectin": "#e41a1c", "Pan-laminin": "#ffd400", "CollagenIV": "#ff33cc", "VersicanV0V1": "#a040ff",
        "CollagenI": "#1f77ff", "HeparanSulfate": "#2ecc40"}
MNAME = {"Fibronectin": "Fibronectin", "Pan-laminin": "Pan-laminin", "CollagenIV": "Collagen IV", "VersicanV0V1": "Versican V0/V1",
         "CollagenI": "Collagen I", "HeparanSulfate": "Heparan sulfate"}
MKL = ["FN", "Laminin", "Col IV", "Versican", "Col I", "HS"]
IMP = LinearSegmentedColormap.from_list("ind", ["#F2EFF7", "#9E9AC8", "#54278F", "#2A0A4A"])  # importance (IG) maps
IMP.set_bad("#FFFFFF")
DEG = plt.get_cmap("coolwarm")  # ECM 5 node degree
DMAX = 45
FRAME_LW, FRAME_COL = 0.5, "#000000"
SEP_LW, SEP_COL = 0.4, "#000000"

GROUPS = ["Naive KO", "Naive WT", "Infected KO", "Infected WT"]
REP = {"Naive KO": "roi_017", "Naive WT": "roi_021", "Infected KO": "roi_004", "Infected WT": "roi_009"}

WF, HF = 7.09, 8.52                    # page (inches)
MW, GAP, X0 = 1.2, 0.05, 0.42          # map tile size, gap and left edge (inches)
def colx(j): return X0 + j * (MW + GAP) + (0.08 if j >= 2 else 0.0)
y3 = 0.52; H3 = MW; rowy2 = y3 + H3 + 0.24; rowy1 = rowy2 + MW + 0.05; rowy0 = rowy1 + MW + 0.05; mtop = rowy0 + MW
yG = mtop + 0.03; yC = yG + 0.17; y1 = yC + 0.66; h1 = 1.45; top3 = rowy2 - 0.05
LX = colx(3) + MW + 0.14
BOUT = ("black", 0.4)
BANDS4 = [[(-0.4, 0.4, "KO", True), (0.6, 1.4, "WT", True), (1.6, 2.4, "KO", True), (2.6, 3.4, "WT", True)],
          [(-0.4, 1.4, "Naive", False), (1.6, 3.4, "Infected", False)]]


def frame(ax, lw: float = FRAME_LW, col: str = FRAME_COL) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_edgecolor(col)


def cell_grid(ax, n_rows: int, n_cols: int, lw: float = SEP_LW, col: str = SEP_COL) -> None:
    ax.hlines(np.arange(0.5, n_rows - 0.5), -0.5, n_cols - 0.5, colors=col, linewidths=lw, zorder=2)
    ax.vlines(np.arange(0.5, n_cols - 0.5), -0.5, n_rows - 0.5, colors=col, linewidths=lw, zorder=2)


class Canvas:
    """A page of fixed size (inches) on which panels are placed at absolute positions."""

    def __init__(self, width: float, height: float, sizes=SIZES):
        mpl.rcdefaults()
        mpl.rcParams.update(RC)
        self.WF, self.HF = float(width), float(height)
        self.SB, self.SA, self.ST = sizes
        mpl.rcParams.update({"font.size": self.SB, "axes.labelsize": self.SB, "axes.titlesize": self.SB, "legend.fontsize": self.SA,
                             "xtick.labelsize": self.ST, "ytick.labelsize": self.ST})
        self.fig = plt.figure(figsize=(self.WF, self.HF))

    def A(self, x, y, w, h):
        """Axes at (x, y) with size (w, h), all in inches from the bottom-left corner."""
        return self.fig.add_axes([x / self.WF, y / self.HF, w / self.WF, h / self.HF])

    def letter(self, x, y, label):
        self.fig.text(x / self.WF, y / self.HF, label, fontsize=self.SB + 2, fontweight="bold", ha="left", va="top")

    def title(self, x, y, text):
        self.fig.text(x / self.WF, y / self.HF, text, fontsize=self.SB, ha="left", va="top")

    def fband(self, x0, x1, y, h, key, text=None, fs=None):
        """Condition/genotype band in figure coordinates."""
        c_, tc = BAND[key]
        self.fig.patches.append(Rectangle((x0 / self.WF, y / self.HF), (x1 - x0) / self.WF, h / self.HF, transform=self.fig.transFigure,
                                          facecolor=c_, zorder=-1, edgecolor=BOUT[0], linewidth=BOUT[1]))
        self.fig.text((x0 + x1) / 2 / self.WF, (y + h / 2) / self.HF, text or key, ha="center", va="center", color=tc, fontsize=fs or self.SA,
                      fontweight="bold")

    def axbands(self, ax, levels, h_in, bh=0.13, gap=0.03, first=0.04):
        """Bands below an axis; levels = list of lists of (x_lo, x_hi, key, italic) in data x, top level first."""
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        y = -first / h_in
        for lev in levels:
            for xl, xh, key, it in lev:
                c_, tc = BAND[key]
                ax.add_patch(Rectangle((xl, y - bh / h_in), xh - xl, bh / h_in, transform=tr, facecolor=c_, clip_on=False,
                                       edgecolor=BOUT[0], linewidth=BOUT[1]))
                ax.text((xl + xh) / 2, y - bh / h_in / 2, key, transform=tr, ha="center", va="center", color=tc, fontsize=self.ST,
                        fontweight="bold", fontstyle="italic" if it else "normal")
            y -= (bh + gap) / h_in


def composite(image: np.ndarray, display: pd.DataFrame) -> np.ndarray:
    """Six-marker RGB composite of one raw ROI image (6, H, W): arcsinh / pooled 99.5th percentile, then one display scale for all ROIs."""
    to_rgb = lambda h: np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    scale = display.set_index("marker")
    rgb = np.zeros(image.shape[1:] + (3,))
    for c, m in enumerate(MK):
        bound = np.float32(scale.loc[m, "transform_bound_99p5"])
        t, hi = float(scale.loc[m, "display_threshold_otsu"]), float(scale.loc[m, "display_ceiling"])
        im = np.clip(np.arcsinh(np.asarray(image[c], dtype=np.float32) / 1.0) / bound, 0.0, 1.0).astype(float)
        rgb += np.where(im > t, np.clip((im - t) / max(hi - t, 1e-6), 0, 1), 0.0)[..., None] * to_rgb(MCOL[m])
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def load_images() -> dict:
    """Raw images of the four ROIs shown, in the order of ``MK``, from Mantpy's liver loader (downloaded on first use)."""
    import mantpy as mt
    liver = mt.datasets.schistosoma_ecm(progressbar=False, verbose=False)
    order = [list(liver.ecm_names).index(m) for m in MK]
    return {roi: np.asarray(liver.images[roi], dtype=np.float32)[order] for roi in REP.values()}


def grid_index(s: pd.DataFrame):
    """Patch-grid row/column of each patch centre (10-px patches, centres at 5, 15, ...)."""
    return ((s.y.to_numpy() - 5) // 10).astype(int), ((s.x.to_numpy() - 5) // 10).astype(int)


def cluster_tile(d, roi):
    """ECM cluster map of one ROI (white = background patches)."""
    s = d["patches"][d["patches"].roi == roi]; gy, gx = grid_index(s)
    cl = np.full((100, 100), -1); cl[gy, gx] = s.cluster.to_numpy(); rgb = np.ones((100, 100, 3))
    for k in range(6): rgb[cl == k] = mpl.colors.to_rgb(PAL[k])
    return rgb, (cl == 5).astype(float)


def ig_tile(d, roi):
    """Absolute IG towards 'infected', scaled to the infected-ROI 99th percentile."""
    s = d["patches"][d["patches"].roi == roi]; gy, gx = grid_index(s)
    ig = np.full((100, 100), np.nan); ig[gy, gx] = s.abs_ig_scaled.to_numpy()
    return ig


def _tile(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, -0.5); frame(ax)


def panel_a(cv, d):
    gm = d["groups"]; ax = cv.A(0.45, y1, 1.3, h1); bottom = np.zeros(4)
    for k in range(6):
        v = gm[f"ECM{k}"].to_numpy(); ax.bar(np.arange(4), v, bottom=bottom, color=PAL[k], width=0.72, edgecolor=SEP_COL, linewidth=SEP_LW); bottom += v
    for j, g in enumerate(GROUPS):
        if g.startswith("Infected"): ax.text(j, 1.012, f"{100 * gm.loc[g, 'ECM5']:.1f}%", ha="center", va="bottom", fontsize=cv.ST, color=C5)
    ax.set_xlim(-0.55, 3.55); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("Fraction of ECM patches"); ax.set_xticks([])
    cv.axbands(ax, BANDS4, h1)
    ax.legend(handles=[Patch(facecolor=PAL[k], edgecolor=SEP_COL, linewidth=SEP_LW, label=f"ECM {k}") for k in range(5, -1, -1)], loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=cv.ST, handlelength=1.0, handletextpad=0.4, labelspacing=0.35)
    cv.letter(0.02, HF - 0.06, "a"); cv.title(0.25, HF - 0.08, "ECM composition")


def panel_b(cv, d):
    Z = d["profiles"].set_index("cluster")[[f"z_mean_{m}" for m in MK]].to_numpy()
    axb = cv.A(2.72, y1 + 0.12, 1.55, h1 - 0.12)
    im = axb.imshow(Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -2.3, 2.3), aspect="auto")
    for i in range(6):
        for j in range(6): axb.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=cv.ST, color="white" if abs(Z[i, j]) > 1.4 else "#222222")
    axb.set_xticks(range(6)); axb.set_xticklabels(MKL, rotation=45, ha="right", rotation_mode="anchor"); axb.set_yticks(range(6)); axb.set_yticklabels([f"ECM {k}" for k in range(6)])
    for k, t in enumerate(axb.get_yticklabels()): t.set_color(PAL[k]); t.set_fontweight("bold")
    axb.tick_params(length=0); frame(axb); cell_grid(axb, 6, 6)
    cb = cv.fig.colorbar(im, cax=cv.A(4.32, y1 + 0.12, 0.07, h1 - 0.12)); cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=cv.ST, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("z-score across clusters", fontsize=cv.ST, labelpad=2)
    cv.letter(2.27, HF - 0.06, "b"); cv.title(2.5, HF - 0.08, "Marker profiles of ECM clusters")


def pair_matrix(pairs: pd.DataFrame, column: str) -> np.ndarray:
    D = np.full((6, 6), np.nan)
    for _, p in pairs.iterrows():
        D[int(p.cluster_a), int(p.cluster_b)] = D[int(p.cluster_b), int(p.cluster_a)] = p[column]
    return D


def panel_c(cv, d):
    D = pair_matrix(d["pairs"], "ko_minus_wt")
    axc = cv.A(5.13, y1 + 0.12, 1.42, h1 - 0.12); vm = 4.5
    imc = axc.imshow(D, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vm, vm), aspect="auto")
    for i in range(6):
        for j in range(6): axc.text(j, i, f"{D[i, j]:+.1f}", ha="center", va="center", fontsize=cv.ST, color="white" if abs(D[i, j]) > 2.6 else "#222222")
    axc.set_xticks(range(6)); axc.set_xticklabels([f"ECM {k}" for k in range(6)], rotation=45, ha="right", rotation_mode="anchor")
    axc.set_yticks(range(6)); axc.set_yticklabels([f"ECM {k}" for k in range(6)])
    for k in range(6):
        for t in (axc.get_xticklabels()[k], axc.get_yticklabels()[k]): t.set_color(PAL[k]); t.set_fontweight("bold")
    axc.tick_params(length=0); frame(axc); cell_grid(axc, 6, 6)
    cbc = cv.fig.colorbar(imc, cax=cv.A(6.6, y1 + 0.12, 0.07, h1 - 0.12)); cbc.set_ticks([-4, -2, 0, 2, 4]); cbc.ax.tick_params(labelsize=cv.ST, length=2)
    cbc.outline.set_linewidth(0.4); cbc.set_label("Δ log2(obs/exp), KO − WT", fontsize=cv.ST, labelpad=2)
    cv.letter(4.7, HF - 0.06, "c"); cv.title(4.93, HF - 0.08, "Co-occurrence, infected KO − WT")


def panel_d(cv, d):
    for j0, key in ((0, "Naive"), (2, "Infected")): cv.fband(colx(j0), colx(j0 + 1) + MW, yC, 0.15, key)
    for j, g in enumerate(GROUPS):
        roi = REP[g]; geno = g.split(" ")[1]
        cv.fband(colx(j), colx(j) + MW, yG, 0.13, geno, text=f"{geno}  ·  {roi.replace('roi_', 'ROI ')}", fs=cv.ST)
        a0 = cv.A(colx(j), rowy0, MW, MW); a0.imshow(d["composites"][roi], interpolation="antialiased"); a0.set_xticks([]); a0.set_yticks([]); frame(a0)
        if j == 0:
            a0.plot([110, 210], [950, 950], color="white", lw=1.4, solid_capstyle="butt"); a0.text(160, 925, "100 µm", color="white", fontsize=cv.ST, ha="center", va="bottom")   # 1 px = 1 µm
            a0.set_ylabel("Six ECM markers", fontsize=cv.SB, labelpad=3)
    cv.fig.legend(handles=[Patch(facecolor=MCOL[m], label=MNAME[m]) for m in MK], loc="center left", bbox_to_anchor=(LX / WF, (rowy0 + MW / 2) / HF), frameon=False,
                  fontsize=cv.ST, handlelength=1.0, labelspacing=0.35, title="ECM markers", title_fontsize=cv.ST, alignment="left")
    cv.letter(0.02, yC + 0.4, "d")


def panel_e(cv, d):
    for j, g in enumerate(GROUPS):
        rgb, _ = cluster_tile(d, REP[g]); a1 = cv.A(colx(j), rowy1, MW, MW); a1.imshow(rgb, interpolation="nearest"); _tile(a1)
        if j == 0: a1.set_ylabel("ECM clusters", fontsize=cv.SB, labelpad=3)
    cv.fig.legend(handles=[Patch(facecolor=PAL[k], label=f"ECM {k}") for k in range(6)] + [Patch(facecolor="white", edgecolor="#999999", lw=0.4, label="No ECM signal")],
                  loc="center left", bbox_to_anchor=(LX / WF, (rowy1 + MW / 2) / HF), frameon=False, fontsize=cv.ST, handlelength=1.0, labelspacing=0.35,
                  title="ECM clusters", title_fontsize=cv.ST, alignment="left")
    cv.letter(0.02, rowy1 + MW + 0.02, "e")


def panel_f(cv, d):
    loc = d["localisation"]
    for j, g in enumerate(GROUPS):
        roi = REP[g]; ig = ig_tile(d, roi); _, m5 = cluster_tile(d, roi)
        a2 = cv.A(colx(j), rowy2, MW, MW); a2.imshow(np.ma.masked_invalid(ig), cmap=IMP, vmin=0, vmax=1, interpolation="nearest")
        if m5.sum(): a2.contour(gaussian_filter(m5, 0.7), levels=[0.25], colors=[C5], linewidths=0.6)
        rr = loc[(loc.roi == roi) & (loc.cluster == 5)].pearson_r
        lab = f"r = {float(rr.iloc[0]):.2f}" if len(rr) and np.isfinite(rr.iloc[0]) else "No ECM 5"
        a2.text(0.03, 0.04, lab, transform=a2.transAxes, fontsize=cv.ST, ha="left", va="bottom", color="#222222", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        _tile(a2)
        if j == 0: a2.set_ylabel("Absolute IG", fontsize=cv.SB, labelpad=3)
    cbi = cv.fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=IMP), cax=cv.A(LX + 0.02, rowy2 + 0.35, 0.08, 0.6)); cbi.set_ticks([0, 1])
    cbi.set_ticklabels(["0", "≥ 99th pct"]); cbi.ax.tick_params(labelsize=cv.ST, length=2); cbi.outline.set_linewidth(0.4)
    cv.fig.text(LX / WF, (rowy2 + 1.02) / HF, "Absolute IG\ntowards ‘infected’", fontsize=cv.ST, ha="left", va="bottom")
    cv.fig.legend(handles=[Line2D([], [], color=C5, lw=1.0, label="ECM 5 region")], loc="center left", bbox_to_anchor=(LX / WF, (rowy2 + 0.17) / HF), frameon=False,
                  fontsize=cv.ST, handlelength=1.4)
    cv.letter(0.02, rowy2 + MW + 0.02, "f")


def panel_g(cv, d):
    """Held-out P(infected) per mouse: GNN ensemble (circles) and composition-only LR (diamonds)."""
    mice = d["mice"].set_index("mouse"); gnn, lr = mice.p_infected_gnn, mice.p_infected_lr
    axg = cv.A(0.55, y3, 0.96, H3); xs = {"Naive KO": 0, "Naive WT": 1, "Infected KO": 2, "Infected WT": 3}
    groups = mice.group; truth = mice.infected.astype(bool)
    seen = {}
    for m in gnn.index:
        g = groups[m]; x = xs[g] + (-0.14 if seen.get(g, 0) == 0 else 0.14); seen[g] = seen.get(g, 0) + 1
        f_ = bool(truth[m])
        axg.scatter(x, gnn[m], s=14, facecolor=INF if f_ else "white", edgecolor=INF if f_ else NAI, lw=0.8, zorder=3)
        axg.scatter(x + 0.09, lr[m], s=9, marker="D", facecolor="#BFBFBF" if f_ else "white", edgecolor="#7F7F7F", lw=0.6, zorder=2)
    axg.axhline(0.5, color="#888888", lw=0.6, ls="--", zorder=1); axg.set_xlim(-0.5, 3.5); axg.set_ylim(0, 1.24); axg.set_yticks([0, 0.5, 1]); axg.set_xticks([])
    axg.spines["left"].set_bounds(0, 1)
    axg.set_ylabel("Held-out\nP(infected), per mouse")
    cv.axbands(axg, [[(-0.45, 0.45, "KO", True), (0.55, 1.45, "WT", True), (1.55, 2.45, "KO", True), (2.55, 3.45, "WT", True)],
                     [(-0.47, 1.47, "Naive", False), (1.53, 3.47, "Infected", False)]], H3)
    axg.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INF, mec=INF, ms=3.2, label="GNN"),
                        Line2D([], [], marker="D", ls="", mfc="#BFBFBF", mec="#7F7F7F", ms=2.4, label="LR")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=cv.ST, handletextpad=0.2, borderaxespad=0.1, columnspacing=0.8)
    cv.letter(0.02, top3, "g")


def panel_h(cv, d):
    loc = d["localisation"]; axh = cv.A(1.98, y3, 0.86, H3); li = loc[loc.group.str.startswith("Infected")]
    for k in range(6):
        v = li[li.cluster == k]
        for gname, mk in (("Infected KO", "o"), ("Infected WT", "s")):
            vv = v[v.group == gname].pearson_r.dropna().to_numpy(); jit = np.linspace(-0.12, 0.12, len(vv)) if len(vv) > 1 else np.zeros(len(vv))
            axh.scatter(k + jit + (-0.06 if mk == "o" else 0.06), vv, s=6, marker=mk, color=PAL[k], lw=0, alpha=0.9, zorder=3)
        med = v.pearson_r.median(); axh.plot([k - 0.3, k + 0.3], [med, med], color="#222222", lw=1.0, zorder=4)
    axh.axhline(0, color="#AAAAAA", lw=0.5); axh.set_xlim(-0.6, 5.6); axh.set_ylim(-0.6, 1.05); axh.set_yticks([-0.5, 0, 0.5, 1])
    axh.set_xticks(range(6)); axh.set_xticklabels([str(k) for k in range(6)])
    for k in range(6): axh.get_xticklabels()[k].set_color(PAL[k]); axh.get_xticklabels()[k].set_fontweight("bold")
    axh.set_xlabel("ECM cluster", labelpad=2); axh.set_ylabel("Pearson r", labelpad=1)
    axh.legend(handles=[Line2D([], [], marker="o", ls="", color="#666666", ms=3.0, label="KO"), Line2D([], [], marker="s", ls="", color="#666666", ms=3.0, label="WT")],
               loc="upper left", frameon=False, fontsize=cv.ST, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1, title="ROI", title_fontsize=cv.ST, alignment="left")
    cv.letter(1.52, top3, "h")


def degree_map(d, roi):
    """ECM 5 subgraph map of one ROI: node degree (40 px) for subgraphs of >= 5 patches, grey for smaller fragments."""
    s = d["patches"][d["patches"].roi == roi]; sn = d["subgraph_nodes"][d["subgraph_nodes"].roi == roi]
    img = np.ones((100, 100, 3)); gy, gx = grid_index(s); img[gy, gx] = mpl.colors.to_rgb("#EEEEEE")
    gy5, gx5 = grid_index(sn); keep = sn.retained.to_numpy(bool); deg = sn.node_degree_40px.to_numpy(float)
    img[gy5[~keep], gx5[~keep]] = mpl.colors.to_rgb("#BDBDBD"); img[gy5[keep], gx5[keep]] = DEG(np.clip(deg[keep] / DMAX, 0, 1))[:, :3]
    km = np.zeros((100, 100)); km[gy5[keep], gx5[keep]] = 1
    return img, km


def panel_i(cv, d):
    for j, g in ((2, "Infected KO"), (3, "Infected WT")):
        img, km = degree_map(d, REP[g])
        ar = cv.A(colx(j), y3, MW, MW); ar.imshow(img, interpolation="nearest"); ar.set_xticks([]); ar.set_yticks([]); ar.set_xlim(-0.5, 99.5); ar.set_ylim(99.5, -0.5)
        if km.sum(): ar.contour(gaussian_filter(km, 0.6), levels=[0.25], colors=["#333333"], linewidths=0.4)
        frame(ar)
    cbd = cv.fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, DMAX), cmap=DEG), cax=cv.A(colx(2), y3 - 0.13, 2 * MW + GAP, 0.05), orientation="horizontal")
    cbd.set_ticks([0, 10, 20, 30, 40]); cbd.ax.tick_params(labelsize=cv.ST, length=2); cbd.outline.set_linewidth(0.4)
    cbd.set_label("ECM 5 subgraph node degree", fontsize=cv.ST, labelpad=2.5)
    sg = d["subgraphs"]; axi = cv.A(6.12, y3, 0.78, H3); mice = ["mouse_01", "mouse_02", "mouse_03", "mouse_04"]
    for j, m in enumerate(mice):
        v = sg[sg.mouse == m].mean_degree.to_numpy(); jit = np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else np.zeros(len(v))
        axi.scatter(j + jit, v, s=10, c=DEG(np.clip(v / DMAX, 0, 1)), lw=0.35, edgecolor="#333333", zorder=3)
        axi.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color="#222222", lw=1.1, zorder=4)
    axi.set_xlim(-0.6, 3.6); axi.set_ylim(0, 42); axi.set_yticks([0, 10, 20, 30, 40]); axi.set_xticks(range(4)); axi.set_xticklabels(["1", "2", "3", "4"])
    axi.tick_params(axis="x", pad=1.5); axi.set_ylabel("Mean node degree\nper subgraph")
    cv.axbands(axi, [[(-0.45, 1.45, "KO", True), (1.55, 3.45, "WT", True)]], H3, first=0.17)
    cv.letter(2.93, top3, "i")


def plot(source_data_dir: Path, images: dict | None = None) -> mpl.figure.Figure:
    """Draw Fig. 3 from ``source_data_dir/fig3.xlsx``.

    images: raw images {roi: array (6, H, W), markers in the order of ``MK``} of ROIs 017, 021, 004 and 009 for panel d;
    loaded with ``mantpy.datasets.schistosoma_ecm`` when omitted.
    """
    book = Path(source_data_dir) / "fig3.xlsx"
    sheet = lambda name: pd.read_excel(book, sheet_name=name)
    images = load_images() if images is None else images
    display = sheet("d_display")
    d = {"groups": sheet("a_groups").set_index("group"), "profiles": sheet("b_profiles"), "pairs": sheet("c_cooccurrence"),
         "composites": {roi: composite(images[roi], display) for roi in REP.values()},
         "patches": sheet("e_f_patches"), "mice": sheet("g_mice"), "localisation": sheet("h_localisation"),
         "subgraphs": sheet("i_subgraphs"), "subgraph_nodes": sheet("i_maps")}
    cv = Canvas(WF, HF)
    for panel in (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g, panel_h, panel_i):
        panel(cv, d)
    cv.fig.canvas.draw()
    for artist in cv.fig.findobj(match=mpl.text.Text):
        if artist.get_text() and artist.get_fontsize() < MIN_FONT:
            artist.set_fontsize(MIN_FONT)
    return cv.fig

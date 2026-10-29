"""Supplementary Fig. 2: liver cluster selection, per-mouse composition and co-occurrence, and all 24 ROIs.

a  seed stability of the candidate encoders across the number of clusters K
b  ECM composition of each mouse
c  co-occurrence of ECM clusters in each infected mouse
d-f  six-marker composite and ECM clusters of every ROI (composites drawn from the raw images)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # ECM 0-5
BAND = {"Naive": ("#5B7C99", "white"), "Infected": ("#A6463A", "white"), "KO": ("#4D4D4D", "white"), "WT": ("#D0D0D0", "#111111")}
MARKERS = [("Fibronectin", "#e41a1c"), ("Pan-laminin", "#ffd400"), ("Collagen IV", "#ff33cc"), ("Versican V0/V1", "#a040ff"),
           ("Collagen I", "#1f77ff"), ("Heparan sulfate", "#2ecc40")]
LIGHT, MID, SHADE = "#BBBBBB", "#7F7F7F", "#EFEFEF"
FS_LAB, FS_TICK, FS_LEG, FS_NOTE, FS_LET = 6.5, 6.0, 6.0, 5.5, 8.0
MM = 1 / 25.4
PT = 25.4 / 72
INF4 = ["mouse_01", "mouse_02", "mouse_03", "mouse_04"]
GEN = {"mouse_01": "KO", "mouse_02": "KO", "mouse_03": "WT", "mouse_04": "WT", "mouse_05": "KO", "mouse_06": "KO", "mouse_07": "WT", "mouse_08": "WT"}
PAGE_W, PAGE_H = 131.0, 517.067 * PT
LETTERS = [("a", 1.4, 8.0), ("b", 173.4, 7.5), ("c", 1.4, 120.6), ("d", 1.4, 234.3), ("e", 1.4, 331.2), ("f", 1.4, 428.2)]  # pt, baseline


def use_style():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": FS_LAB,
        "axes.labelsize": FS_LAB, "axes.titlesize": FS_LAB, "legend.fontsize": FS_LEG,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.linewidth": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.0, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left", "axes.unicode_minus": True,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    })


class Page:
    """Fixed-size page in millimetres; positions from the top-left corner, shifted by the current origin."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = float(w_mm), float(h_mm)
        self.fig = plt.figure(figsize=(self.W * MM, self.H * MM))
        self.ox, self.oy = 0.0, 0.0

    def ax(self, x, y, w, h, **kw):
        x, y = x + self.ox, y + self.oy
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_LAB)
        return self.fig.text((x + self.ox) / self.W, 1 - (y + self.oy) / self.H, s, **kw)

    def band(self, x0, x1, y, h, key, text=None, fs=FS_NOTE):
        c_, tc = BAND[key]
        self.fig.patches.append(Rectangle(((x0 + self.ox) / self.W, 1 - (y + self.oy + h) / self.H), (x1 - x0) / self.W, h / self.H,
                                          transform=self.fig.transFigure, facecolor=c_, edgecolor="none", zorder=-1))
        self.text((x0 + x1) / 2, y + h / 2, text or key, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold")


def frame_map(ax, lw=0.5):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(lw)


def full_matrix(df, col):
    M = np.full((6, 6), np.nan)
    for _, r in df.iterrows():
        a, b = int(r.cluster_a), int(r.cluster_b)
        M[a, b] = M[b, a] = r[col]
    return M


def cluster_grid(pat, roi):
    g = np.full((100, 100), 6, dtype=np.int64)
    p = pat[pat.roi == roi]
    g[((p.y.to_numpy() - 5) // 10).astype(int), ((p.x.to_numpy() - 5) // 10).astype(int)] = p.cluster.to_numpy()
    return g


def composites(images, scale):
    """Six-marker RGB composites with one display scale for all ROIs (arcsinh / pooled bound, then threshold and ceiling)."""
    to_rgb = lambda h: np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    out = {}
    for r, raw in images.items():
        raw = np.asarray(raw, dtype=np.float32)
        rgb = np.zeros(raw.shape[1:] + (3,))
        for c, (_, colour) in enumerate(MARKERS):
            bound, t, hi = scale[c]
            im = np.clip(np.arcsinh(raw[c] / 1.0) / np.float32(bound), 0.0, 1.0).astype(float)
            rgb += np.where(im > t, np.clip((im - t) / max(hi - t, 1e-6), 0, 1), 0.0)[..., None] * to_rgb(colour)
        out[r] = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return out


def panel_a(pg, k):
    """Seed stability of the candidate encoder-K pairs and the K rule."""
    ax = pg.ax(62.5, 7.5, 32.0, 23.5)
    ax.axvspan(5.6, 6.4, color=SHADE, zorder=0, lw=0)
    ax.axhline(0.80, color=MID, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for _, r in k.iterrows():
        ax.plot([r.K - 0.14, r.K + 0.14], [r.stability_3_seeds, r.stability_5_seeds], color=LIGHT, lw=0.5, zorder=1)
        ax.scatter(r.K - 0.14, r.stability_3_seeds, s=7, facecolor="white", edgecolor="#555555", lw=0.5, zorder=2)
        ax.scatter(r.K + 0.14, r.stability_5_seeds, s=20 if r.selected else 8,
                   color="black" if r.selected else ("#555555" if r.reaches_0_80 else LIGHT), lw=0, zorder=3)
    ax.text(6.0, 0.99, "K = 6", ha="center", va="top", fontsize=FS_NOTE, fontweight="bold")
    ax.text(12.55, 0.807, "0.80", ha="right", va="bottom", fontsize=FS_NOTE, color=MID)
    ax.set_xlim(2.5, 12.6); ax.set_xticks(range(3, 13)); ax.set_ylim(0.5, 1.0); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xlabel("Number of clusters, K"); ax.set_ylabel("Seed stability (ARI)")
    lax = pg.ax(58.0, 0.6, 38.0, 5.4); lax.set_axis_off()
    lax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=0.5, ms=2.8, label="3 seeds"),
                        Line2D([], [], marker="o", ls="", mfc="#555555", mec="none", ms=2.8, label="5 seeds, eligible"),
                        Line2D([], [], marker="o", ls="", mfc=LIGHT, mec="none", ms=2.8, label="5 seeds, < 0.80"),
                        Line2D([], [], marker="o", ls="", mfc="black", mec="none", ms=4.0, label="Selected")],
               ncol=2, loc="center left", fontsize=FS_NOTE, handletextpad=0.2, columnspacing=0.8, borderaxespad=0, labelspacing=0.25)
    pg.fig.text(68.6 * PT / pg.W, 1 - 26.5 * PT / pg.H, "largest K \u2265 0.80", ha="left", va="baseline", fontsize=FS_NOTE, color=(0.25, 0.25, 0.25))


def panel_b(pg, mice):
    """Fraction of each mouse's ECM patches per cluster."""
    order = ["mouse_05", "mouse_06", "mouse_07", "mouse_08", "mouse_01", "mouse_02", "mouse_03", "mouse_04"]
    xb = [0, 1, 2.3, 3.3, 4.9, 5.9, 7.2, 8.2]
    ax = pg.ax(10.0, 4.0, 43.0, 20.0)
    for x, m in zip(xb, order):
        bottom = 0.0
        for k in range(6):
            v = float(mice.loc[m, f"ECM{k}"])
            ax.bar(x, v, bottom=bottom, width=0.8, color=PAL[k], edgecolor="black", lw=0.3)
            bottom += v
    ax.set_xlim(-0.6, 8.8); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("Fraction of\nECM patches", linespacing=1.1)
    ax.set_xticks(xb); ax.set_xticklabels([m[-2:] for m in order], fontsize=FS_NOTE); ax.tick_params(axis="x", length=0, pad=1.2)
    ax.text(-0.75, -0.085, "Mouse", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=FS_NOTE, color=MID)
    x2mm = lambda xv: 10.0 + 43.0 * (xv + 0.6) / 9.4
    for lo, hi, key in ((0, 1, "KO"), (2.3, 3.3, "WT"), (4.9, 5.9, "KO"), (7.2, 8.2, "WT")):
        pg.band(x2mm(lo - 0.45), x2mm(hi + 0.45), 27.0, 2.3, key)
    for lo, hi, key in ((0, 3.3, "Naive"), (4.9, 8.2, "Infected")):
        pg.band(x2mm(lo - 0.45), x2mm(hi + 0.45), 29.7, 2.3, key)


def panel_c(pg, co):
    """log2(observed/expected) co-occurrence in each infected mouse."""
    yd, md_ = 39.5, 26.0
    xm = [9.0, 38.2, 67.4, 96.6]; norm = TwoSlopeNorm(0, -5.0, 5.0)
    for j, m in enumerate(INF4):
        M = full_matrix(co, f"log2_obs_exp_mouse_{m[-2:]}")
        ax = pg.ax(xm[j], yd, md_, md_)
        ax.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest")
        for a in range(6):
            for b in range(6):
                ax.text(b, a, f"{M[a, b]:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=FS_NOTE,
                        color="white" if abs(M[a, b]) > 2.6 else "#222222")
        ax.hlines(np.arange(0.5, 5.5), -0.5, 5.5, colors="black", lw=0.3); ax.vlines(np.arange(0.5, 5.5), -0.5, 5.5, colors="black", lw=0.3)
        frame_map(ax, 0.5)
        ax.set_xticks(range(6)); ax.set_xticklabels([str(k) for k in range(6)], fontsize=FS_NOTE); ax.tick_params(axis="x", length=0, pad=1.2)
        for k, tl in enumerate(ax.get_xticklabels()):
            tl.set_color(PAL[k]); tl.set_fontweight("bold")
        if j == 0:
            ax.set_yticks(range(6)); ax.set_yticklabels([str(k) for k in range(6)], fontsize=FS_NOTE); ax.tick_params(axis="y", length=0, pad=1.2)
            for k, tl in enumerate(ax.get_yticklabels()):
                tl.set_color(PAL[k]); tl.set_fontweight("bold")
        pg.band(xm[j], xm[j] + md_, yd - 3.0, 2.4, GEN[m], f"Infected {GEN[m]} \u00b7 mouse {m[-2:]}")
    pg.text(4.6, yd + md_ / 2, "ECM cluster", rotation=90, ha="center", va="center", fontsize=FS_NOTE)
    cax = pg.ax(124.2, yd + 2.0, 1.3, md_ - 4.0)
    cb = pg.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax); cb.set_ticks([-4, -2, 0, 2, 4])
    cb.ax.tick_params(labelsize=FS_NOTE, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("log$_2$(observed/expected)", fontsize=FS_NOTE, labelpad=1.5)


def legend_markers_clusters(pg):
    lax = pg.ax(1.0, 68.6, 130.0, 6.2); lax.set_axis_off()
    hm = [Patch(facecolor=c, edgecolor="none", label=n) for n, c in MARKERS]
    hc = [Patch(facecolor=PAL[k], edgecolor="none", label=f"ECM {k}") for k in range(6)] + [Patch(facecolor="white", edgecolor="#9A9A9A", lw=0.4, label="No ECM signal")]
    l1 = lax.legend(handles=hm, ncol=6, loc="upper left", bbox_to_anchor=(0.105, 1.0), fontsize=FS_NOTE, handlelength=0.9, handleheight=0.8,
                    handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    lax.add_artist(l1)
    lax.legend(handles=hc, ncol=7, loc="lower left", bbox_to_anchor=(0.105, 0.0), fontsize=FS_NOTE, handlelength=0.9, handleheight=0.8,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    lax.text(0.0, 0.78, "ECM markers", transform=lax.transAxes, ha="left", va="center", fontsize=FS_NOTE, fontweight="bold")
    lax.text(0.0, 0.2, "ECM clusters", transform=lax.transAxes, ha="left", va="center", fontsize=FS_NOTE, fontweight="bold")


def panels_d_f(pg, pat, comp):
    """All 24 ROIs: six-marker composite (top row) and ECM clusters (bottom row), by condition, mouse and genotype."""
    cmap7 = ListedColormap(PAL + ["#FFFFFF"])
    T, gin, gbt = 13.6, 0.4, 1.3
    blocks = [("Naive", [("mouse_05", ["roi_017", "roi_018"]), ("mouse_06", ["roi_019", "roi_020"]), ("mouse_07", ["roi_021", "roi_022"]), ("mouse_08", ["roi_023", "roi_024"])]),
              ("Infected", [("mouse_01", ["roi_001", "roi_002", "roi_003", "roi_004"]), ("mouse_02", ["roi_005", "roi_006", "roi_007", "roi_008"])]),
              ("Infected", [("mouse_03", ["roi_009", "roi_010", "roi_011", "roi_012"]), ("mouse_04", ["roi_013", "roi_014", "roi_015", "roi_016"])])]
    y = 76.6
    for cond, mice in blocks:
        nroi = sum(len(r) for _, r in mice)
        width = nroi * T + (nroi - len(mice)) * gin + (len(mice) - 1) * gbt
        x = 5.5 + (125.5 - width) / 2
        pg.band(x, x + width, y, 2.2, cond)
        yt = y + 2.5 + 2.2 + 0.3
        for mouse, rlist in mice:
            wmouse = len(rlist) * T + (len(rlist) - 1) * gin
            pg.band(x, x + wmouse, y + 2.5, 2.2, GEN[mouse], f"{GEN[mouse]} \u00b7 mouse {mouse[-2:]}")
            for roi in rlist:
                axm = pg.ax(x, yt, T, T); arr = comp[roi]
                axm.imshow(arr, interpolation="antialiased"); frame_map(axm)
                axm.text(0.04, 0.96, roi[-3:], transform=axm.transAxes, ha="left", va="top", fontsize=FS_NOTE, color="white")
                if roi == rlist[0] and mouse == mice[0][0]:
                    axm.plot([0.72 * arr.shape[1], 0.92 * arr.shape[1]], [0.9 * arr.shape[0]] * 2, color="white", lw=0.9, solid_capstyle="butt")
                axc = pg.ax(x, yt + T + gin, T, T)
                axc.imshow(cluster_grid(pat, roi), cmap=cmap7, vmin=-0.5, vmax=6.5, interpolation="nearest"); frame_map(axc)
                x += T + gin
            x += gbt - gin
        lab_x = 5.5 + (125.5 - width) / 2 - 1.2
        pg.text(lab_x, yt + T / 2, "Markers", rotation=90, ha="center", va="center", fontsize=FS_NOTE)
        pg.text(lab_x, yt + T + gin + T / 2, "Clusters", rotation=90, ha="center", va="center", fontsize=FS_NOTE)
        y = yt + 2 * T + gin + 1.6


def plot(source_data_dir: Path, images=None, cache_dir=None) -> mpl.figure.Figure:
    """Draw Supplementary Fig. 2 from source_data/supplementary_figure_2.xlsx and the composite scale in fig3.xlsx.

    images: {roi: (6, H, W) raw marker image} in the marker order of MARKERS; if None, the liver dataset is
    loaded with mantpy (mt.datasets.schistosoma_ecm, cached in cache_dir).
    """
    source_data_dir = Path(source_data_dir)
    wb = source_data_dir / "supplementary_figure_2.xlsx"
    k = pd.read_excel(wb, sheet_name="a_k")
    mice = pd.read_excel(wb, sheet_name="b_mice").set_index("mouse")
    co = pd.read_excel(wb, sheet_name="c_cooccurrence")
    pat = pd.read_excel(wb, sheet_name="d_f_patches")
    disp = pd.read_excel(source_data_dir / "fig3.xlsx", sheet_name="d_display")
    scale = disp[["transform_bound_99p5", "display_threshold_otsu", "display_ceiling"]].to_numpy(float)
    if images is None:
        import mantpy as mt
        images = mt.datasets.schistosoma_ecm(cache_dir=cache_dir, progressbar=False, verbose=False).images
    comp = composites({r: images[r] for r in sorted(pat.roi.unique())}, scale)

    use_style()
    pg = Page(PAGE_W, PAGE_H)
    pg.ox, pg.oy = -148.5 * PT, 0.0
    panel_a(pg, k)
    pg.ox, pg.oy = 172.0 * PT, 0.0
    panel_b(pg, mice)
    pg.ox, pg.oy = 0.0, 12.5 * PT
    panel_c(pg, co)
    legend_markers_clusters(pg)
    panels_d_f(pg, pat, comp)
    for s, x, y in LETTERS:
        pg.fig.text(x * PT / pg.W, 1 - y * PT / pg.H, s, ha="left", va="baseline", fontsize=FS_LET, fontweight="bold")
    return pg.fig

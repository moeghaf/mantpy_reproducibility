"""Redraw the figures of the Mantpy paper from the tables in source_data/.

    from paper_figures import plot_figure2
    fig = plot_figure2()
    fig.savefig("fig2.pdf")

Each function returns a matplotlib Figure drawn with the code that produced the
published figure. Panels showing raw images (Fig. 3d, Fig. 4g and
Supplementary Fig. 2d-f) read the Zenodo images through mantpy.
"""
from __future__ import annotations

import importlib
from pathlib import Path

SOURCE_DATA = Path(__file__).resolve().parent.parent / "source_data"

FIGURES = {
    "figure2": "fig2",
    "figure3": "fig3",
    "figure4": "fig4",
    "supplementary_figure1": "supplementary_figure_1",
    "supplementary_figure2": "supplementary_figure_2",
    "supplementary_figure3": "supplementary_figure_3",
    "supplementary_figure4": "supplementary_figure_4",
}


def _plot(module: str, source_data=None, **kw):
    mod = importlib.import_module(f"_paper.{module}")
    return mod.plot(Path(source_data) if source_data else SOURCE_DATA, **kw)


def plot_figure2(source_data=None, **kw):
    """Fig. 2: collagen IV graphs in two human small-intestine images."""
    return _plot("fig2", source_data, **kw)


def plot_figure3(source_data=None, **kw):
    """Fig. 3: ECM composition and organisation in naive and infected mouse liver."""
    return _plot("fig3", source_data, **kw)


def plot_figure4(source_data=None, **kw):
    """Fig. 4: cell-matrix association and label recovery in healthy mouse lung."""
    return _plot("fig4", source_data, **kw)


def plot_supplementary_figure1(source_data=None, **kw):
    """Supplementary Fig. 1: label-free choices and partitions in the intestine images."""
    return _plot("supplementary_figure_1", source_data, **kw)


def plot_supplementary_figure2(source_data=None, **kw):
    """Supplementary Fig. 2: liver cluster selection, per-mouse summaries and all ROIs."""
    return _plot("supplementary_figure_2", source_data, **kw)


def plot_supplementary_figure3(source_data=None, **kw):
    """Supplementary Fig. 3: lung ECM clusters, cell-ECM association and all ROI graphs."""
    return _plot("supplementary_figure_3", source_data, **kw)


def plot_supplementary_figure4(source_data=None, **kw):
    """Supplementary Fig. 4: computational scaling."""
    return _plot("supplementary_figure_4", source_data, **kw)


def plot_all(folder="outputs", source_data=None):
    """Draw every figure and save it as PDF in folder."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for name, module in FIGURES.items():
        _plot(module, source_data).savefig(out / f"{module}.pdf")
    return sorted(out.glob("*.pdf"))

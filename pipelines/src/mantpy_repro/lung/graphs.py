"""Cell-cell, ECM-ECM and cell-ECM graphs of one ROI, built with Mantpy from a :class:`GraphRecipe`.

Mantpy supplies the topology and the Euclidean edge lengths. The settings that make the
searched recipes deterministic are fixed here: SciPy's nearest-neighbour tie order, and a
seeded jitter of 1e-6 pixels before every Delaunay triangulation (the patch lattice is
regular). In the ECM-ECM and cell-ECM layers a Delaunay edge is kept up to twice the recipe's
radius when the recipe has one; the cell-cell Delaunay layer has no length limit.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .config import GraphRecipe

DELAUNAY_JITTER, DELAUNAY_SEED = 1e-6, 0


@dataclass
class Graphs:
    """Sparse layers of one ROI. ``*_w`` hold edge weights and ``*_d`` edge lengths in pixels.
    The cell-ECM matrices have one row per cell and one column per patch."""

    cc_w: sp.csr_matrix
    cc_d: sp.csr_matrix
    ee_w: sp.csr_matrix
    ee_d: sp.csr_matrix
    ce_w: sp.csr_matrix
    ce_d: sp.csr_matrix
    recipe: GraphRecipe

    def n_edges(self) -> dict[str, int]:
        return {"cc": int(sp.triu(self.cc_d, 1).nnz), "ee": int(sp.triu(self.ee_d, 1).nnz), "ce": int(self.ce_d.nnz)}


def edge_weights(lengths: sp.csr_matrix, rule: str) -> sp.csr_matrix:
    """Weights from edge lengths: 1, 1/d, 1/d^2, or a Gaussian whose width is the median length."""
    weights = lengths.copy().astype(np.float64)
    if rule == "none":
        weights.data[:] = 1.0
    elif rule == "inv_d":
        weights.data = 1.0 / weights.data
    elif rule == "inv_d2":
        weights.data = 1.0 / (weights.data ** 2)
    elif rule == "gaussian_median":
        if weights.nnz:
            sigma = float(np.median(weights.data))
            weights.data = np.exp(-(weights.data ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"unknown edge_weight {rule!r}")
    return weights


def mantpy_arguments(recipe: GraphRecipe, patch_size: float) -> dict:
    """Keyword arguments of ``mantpy.gr.ensure_cell_ecm_graph`` for a recipe."""
    kw: dict = {}
    if recipe.cc_rule == "knn":
        kw.update(cell_edge_method="knn", cell_k=int(recipe.cc_k), cell_Dmax=recipe.cc_radius_px)
    elif recipe.cc_rule == "radius":
        kw.update(cell_edge_method="radius", cell_r=float(recipe.cc_radius_px))
    else:
        kw.update(cell_edge_method="delaunay")
    if recipe.ee_rule in ("grid8", "grid4"):
        kw.update(ecm_edge_method="grid", ecm_grid_connectivity=8 if recipe.ee_rule == "grid8" else 4)
    elif recipe.ee_rule == "knn":
        kw.update(ecm_edge_method="knn", ecm_k=int(recipe.ee_k), ecm_Dmax=recipe.ee_radius_px)
    elif recipe.ee_rule == "radius":
        kw.update(ecm_edge_method="radius", ecm_r=float(recipe.ee_radius_px))
    else:
        kw.update(ecm_edge_method="delaunay",
                  ecm_delaunay_max_length=2.0 * recipe.ee_radius_px if recipe.ee_radius_px else None)
    if recipe.ce_rule == "knn":
        kw.update(cell_ecm_edge_method="knn", cell_ecm_k=int(recipe.ce_k), cell_ecm_Dmax=recipe.ce_dmax_px)
    elif recipe.ce_rule == "radius":
        kw.update(cell_ecm_edge_method="radius",
                  cell_ecm_r=float(recipe.ce_dmax_px) if recipe.ce_dmax_px is not None else 15.0)
    elif recipe.ce_rule == "delaunay":
        kw.update(cell_ecm_edge_method="delaunay",
                  cell_ecm_delaunay_max_length=2.0 * recipe.ce_dmax_px if recipe.ce_dmax_px else None)
    else:
        kw.update(cell_ecm_edge_method="containing_ring", cell_ecm_patch_size=float(patch_size))
    methods = (kw["cell_edge_method"], kw["ecm_edge_method"], kw["cell_ecm_edge_method"])
    if "knn" in methods:
        kw["knn_backend"] = "scipy"
    if "delaunay" in methods:
        kw.update(delaunay_jitter=DELAUNAY_JITTER, delaunay_seed=DELAUNAY_SEED)
    return kw


def _edge_lists(cell_xy, patch_xy, background, patch_size, recipe: GraphRecipe):
    """Run Mantpy and return the edges of each layer as rows (i, j, length) in cell / patch indices."""
    import mantpy as mt
    import pandas as pd
    from anndata import AnnData

    keep = np.ones(len(patch_xy), bool) if recipe.ee_include_background_nodes else ~np.asarray(background, bool)
    index = np.flatnonzero(keep)
    carrier = AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(len(cell_xy))]))
    carrier.obsm["spatial"] = cell_xy
    carrier.uns["ecm_patches"] = pd.DataFrame({"x": patch_xy[index, 0], "y": patch_xy[index, 1],
                                               "ecm_cluster": np.zeros(index.size, int)})
    mt.gr.ensure_cell_ecm_graph(carrier, **mantpy_arguments(recipe, patch_size))

    def node(name):
        kind, number = str(name).split("_")
        return kind, int(number)

    cc = [(node(u)[1], node(v)[1], float(d["distance"])) for u, v, d in carrier.uns["cell_graph_nx"].edges(data=True)]
    ee = [(int(index[node(u)[1]]), int(index[node(v)[1]]), float(d["distance"]))
          for u, v, d in carrier.uns["ecm_graph"].edges(data=True)]
    ce = []
    for u, v, d in carrier.uns["cell_ecm_graph"].edges(data=True):
        if d.get("edge_type") != "cell-ecm":
            continue
        (kind_u, i_u), (_, i_v) = node(u), node(v)
        cell, patch = (i_u, i_v) if kind_u == "cell" else (i_v, i_u)
        ce.append((cell, int(index[patch]), float(d["distance"])))
    return tuple(np.asarray(edges, dtype=float).reshape(-1, 3) for edges in (cc, ee, ce))


def _symmetric(edges: np.ndarray, n: int) -> sp.csr_matrix:
    i, j, d = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]
    return sp.coo_matrix((np.concatenate([d, d]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n)).tocsr()


def build_graphs(cell_xy, patch_xy, background, patch_size, recipe: GraphRecipe) -> Graphs:
    """Build the three layers of one ROI.

    With ``ee_include_background_nodes=False`` background patches are left out of both the
    ECM-ECM and the cell-ECM layer. A cell at the exact centre of its patch keeps a length of
    1e-6 pixels under the containing-ring rule so that inverse-distance weights stay finite.
    """
    cell_xy, patch_xy = np.asarray(cell_xy, float), np.asarray(patch_xy, float)
    cc, ee, ce = _edge_lists(cell_xy, patch_xy, background, patch_size, recipe)
    n_cells, n_patches = len(cell_xy), len(patch_xy)
    cc_d, ee_d = _symmetric(cc, n_cells), _symmetric(ee, n_patches)
    lengths = ce[:, 2].copy()
    if recipe.ce_rule == "containing_ring":
        lengths = np.maximum(lengths, 1e-6)
    ce_d = sp.coo_matrix((lengths, (ce[:, 0].astype(int), ce[:, 1].astype(int))), shape=(n_cells, n_patches)).tocsr()
    rule = recipe.edge_weight
    return Graphs(cc_w=edge_weights(cc_d, rule), cc_d=cc_d, ee_w=edge_weights(ee_d, rule), ee_d=ee_d,
                  ce_w=edge_weights(ce_d, rule), ce_d=ce_d, recipe=recipe)


class GraphCache:
    """Recently used graphs of one cohort, keyed by ROI and recipe. Graphs depend on which
    patches are background, so every partition needs its own cache."""

    def __init__(self, maxsize: int = 96):
        self._graphs: OrderedDict[tuple[str, str], Graphs] = OrderedDict()
        self.maxsize = maxsize

    def get(self, roi, recipe: GraphRecipe) -> Graphs:
        key = (roi.roi, recipe.digest())
        graphs = self._graphs.get(key)
        if graphs is None:
            graphs = build_graphs(roi.cell_xy, roi.patch_xy, roi.background, roi.patch_size, recipe)
            self._graphs[key] = graphs
            if len(self._graphs) > self.maxsize:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        return graphs


def joint_graph(roi, graphs: Graphs):
    """The layers of one ROI as a Mantpy joint graph that keeps the cell and patch identities."""
    import anndata as ad
    import mantpy as mt
    import pandas as pd

    cells = ad.AnnData(obs=pd.DataFrame({"cell_type": roi.cell_type}, index=roi.cell_id.astype(str)))
    patches = ad.AnnData(obs=pd.DataFrame({"ecm_cluster": roi.label}, index=roi.patch_id.astype(str)))
    cells.obsm["spatial"], patches.obsm["spatial"] = roi.cell_xy, roi.patch_xy
    joint = mt.gr.joint_graph_from_connectivities(
        cells, patches, cell_connectivities=graphs.cc_w, ecm_connectivities=graphs.ee_w,
        cell_ecm_connectivities=graphs.ce_w, cell_distances=graphs.cc_d,
        ecm_distances=graphs.ee_d, cell_ecm_distances=graphs.ce_d,
    )
    joint.uns["cell_ecm_graph"] = mt.gr.to_networkx(joint)
    return joint

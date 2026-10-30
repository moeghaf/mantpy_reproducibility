"""Reviewed adapter: public Mantpy topology and true Euclidean distances.

Uses the SciPy kNN tie order and deterministic Delaunay jitter required by the
frozen lattice recipes. No historical distance-multiplicity mode is available
in the current analysis; its diagnostic implementation is archived separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

import mantpy as mt
from .graphs import GraphRecipe, Graphs, _weights

JITTER, SEED = 1e-6, 0

def recipe_to_kwargs(rec: GraphRecipe, patch_size: float) -> dict:
    kw: dict = {}
    if rec.cc_rule == "knn":
        kw.update(cell_edge_method="knn", cell_k=int(rec.cc_k), cell_Dmax=rec.cc_radius_px)
    elif rec.cc_rule == "radius":
        kw.update(cell_edge_method="radius", cell_r=float(rec.cc_radius_px))
    elif rec.cc_rule == "delaunay":
        kw.update(cell_edge_method="delaunay")
    else:
        raise ValueError(rec.cc_rule)
    if rec.ee_rule in ("grid8", "grid4"):
        kw.update(ecm_edge_method="grid", ecm_grid_connectivity=8 if rec.ee_rule == "grid8" else 4)
    elif rec.ee_rule == "knn":
        kw.update(ecm_edge_method="knn", ecm_k=int(rec.ee_k), ecm_Dmax=rec.ee_radius_px)
    elif rec.ee_rule == "radius":
        kw.update(ecm_edge_method="radius", ecm_r=float(rec.ee_radius_px))
    elif rec.ee_rule == "delaunay":
        kw.update(ecm_edge_method="delaunay", ecm_delaunay_max_length=2.0 * rec.ee_radius_px if rec.ee_radius_px else None)
    else:
        raise ValueError(rec.ee_rule)
    if rec.ce_rule == "knn":
        kw.update(cell_ecm_edge_method="knn", cell_ecm_k=int(rec.ce_k), cell_ecm_Dmax=rec.ce_dmax_px)
    elif rec.ce_rule == "radius":
        kw.update(cell_ecm_edge_method="radius", cell_ecm_r=float(rec.ce_dmax_px) if rec.ce_dmax_px is not None else 15.0)
    elif rec.ce_rule == "delaunay":
        kw.update(cell_ecm_edge_method="delaunay", cell_ecm_delaunay_max_length=2.0 * rec.ce_dmax_px if rec.ce_dmax_px else None)
    elif rec.ce_rule == "containing_ring":
        kw.update(cell_ecm_edge_method="containing_ring", cell_ecm_patch_size=float(patch_size))
    else:
        raise ValueError(rec.ce_rule)
    methods = (kw["cell_edge_method"], kw["ecm_edge_method"], kw["cell_ecm_edge_method"])
    if "knn" in methods:
        kw["knn_backend"] = "scipy"
    if "delaunay" in methods:
        kw.update(delaunay_jitter=JITTER, delaunay_seed=SEED)
    return kw

def mantpy_edge_lists(cell_xy, patch_xy, background, patch_size, rec: GraphRecipe):
    """Run Mantpy; return (cc, ee, ce) edge arrays (i, j, d) in the full cell / patch index space."""
    cell_xy = np.asarray(cell_xy, float); patch_xy = np.asarray(patch_xy, float)
    node_mask = np.ones(len(patch_xy), bool) if rec.ee_include_background_nodes else ~np.asarray(background, bool)
    idx = np.flatnonzero(node_mask)
    a = AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(len(cell_xy))]))
    a.obsm["spatial"] = cell_xy
    a.uns["ecm_patches"] = pd.DataFrame({"x": patch_xy[idx, 0], "y": patch_xy[idx, 1], "ecm_cluster": np.zeros(idx.size, int)})
    mt.gr.ensure_cell_ecm_graph(a, **recipe_to_kwargs(rec, patch_size))

    def node(name):
        kind, k = str(name).split("_")
        return kind, int(k)

    cc = [(node(u)[1], node(v)[1], float(d["distance"])) for u, v, d in a.uns["cell_graph_nx"].edges(data=True)]
    ee = [(int(idx[node(u)[1]]), int(idx[node(v)[1]]), float(d["distance"])) for u, v, d in a.uns["ecm_graph"].edges(data=True)]
    ce = []
    for u, v, d in a.uns["cell_ecm_graph"].edges(data=True):
        if d.get("edge_type") != "cell-ecm":
            continue
        (ku, iu), (kv, iv) = node(u), node(v)
        ci, ei = (iu, iv) if ku == "cell" else (iv, iu)
        ce.append((ci, int(idx[ei]), float(d["distance"])))
    as_arr = lambda e: np.asarray(e, dtype=float).reshape(-1, 3)
    return as_arr(cc), as_arr(ee), as_arr(ce), idx

def _sym(e: np.ndarray, n: int) -> sp.csr_matrix:
    i, j, d = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 2]
    return sp.coo_matrix((np.concatenate([d, d]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n)).tocsr()

def build_graphs_mantpy(cell_xy, patch_xy, patch_grid, background, patch_size, recipe: GraphRecipe) -> Graphs:
    cell_xy = np.asarray(cell_xy, float); patch_xy = np.asarray(patch_xy, float)
    cc, ee, ce, idx = mantpy_edge_lists(cell_xy, patch_xy, background, patch_size, recipe)
    return graphs_from_edges(cc, ee, ce, idx, cell_xy, patch_xy, recipe)

def graphs_from_edges(cc, ee, ce, idx, cell_xy, patch_xy, recipe: GraphRecipe) -> Graphs:
    n_c, n_p = len(cell_xy), len(patch_xy)
    cc_d, ee_d = _sym(cc, n_c), _sym(ee, n_p)
    d_ce = ce[:, 2].copy()
    if recipe.ce_rule == "containing_ring":
        d_ce = np.maximum(d_ce, 1e-6)  # analysis convention (graphs.py): a cell at a patch centre keeps a finite length
    ce_d = sp.coo_matrix((d_ce, (ce[:, 0].astype(int), ce[:, 1].astype(int))), shape=(n_c, n_p)).tocsr()
    return Graphs(cc_w=_weights(cc_d, recipe.edge_weight), cc_d=cc_d, ee_w=_weights(ee_d, recipe.edge_weight), ee_d=ee_d,
                  ce_w=_weights(ce_d, recipe.edge_weight), ce_d=ce_d, recipe=recipe)

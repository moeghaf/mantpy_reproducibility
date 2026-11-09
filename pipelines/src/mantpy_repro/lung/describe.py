"""Graphs and cell-ECM association (Fig. 4a-g, Supplementary Fig. 3b-f).

``graphs``       the graph of Fig. 4 built as in the paper, with Squidpy for the cell graph,
                 and its edge counts per ROI.
``association``  for every cell type and ECM cluster, observed cell-ECM edges against the
                 expected count (the cell type's edges x the cluster's share of all cell-ECM
                 edges in that ROI), summed over ROIs; and the marker profile of ECM 1 patches
                 with an airway-epithelial-cell edge minus those without.
``edge_rules``   the airway-epithelial association under 48 seeded random cell-ECM edge rules.

Edges to background patches are never counted.
"""

from __future__ import annotations

import numpy as np

from .config import FOCAL_CELL_TYPE, SPACE, WALL_CLUSTER, GraphRecipe
from .core import Context, environment, log, pack
from .graphs import GraphCache, build_graphs, joint_graph

EDGE_RULE_SEED = 0


# ----------------------------------------------------------------------------------- graph counts
def graph_counts(cohort, cache: GraphCache):
    """Edge counts of the Fig. 4 graph in every ROI.

    The first three counts come from Mantpy's cohort builder with a Squidpy cell graph (cells
    within 15 pixels, eight-connected patch grid, five nearest patches within 15 pixels) and
    include background patches as nodes. The ``signal`` counts keep only edges whose patches
    are signal patches, as drawn in the paper's figures.
    """
    import anndata as ad
    import mantpy as mt
    import pandas as pd
    import scipy.sparse as sp

    carriers = {}
    for name, roi in cohort.rois.items():
        cells = ad.AnnData(obs=pd.DataFrame({"cell_type": roi.cell_type}, index=roi.cell_id.astype(str)))
        cells.obsm["spatial"] = roi.cell_xy
        cells.uns["ecm_patches"] = pd.DataFrame({"x": roi.patch_xy[:, 0], "y": roi.patch_xy[:, 1],
                                                 "ecm_cluster": roi.label})
        carriers[name] = cells
    built = mt.gr.build_cell_ecm_graphs(carriers, cell_graph="squidpy", cell_radius=(0, 15), ecm_edge_method="grid",
                                        ecm_grid_connectivity=8, cell_ecm_edge_method="knn", cell_ecm_k=5,
                                        cell_ecm_Dmax=15)
    rows = []
    for summary, (name, roi) in zip(built.summaries, cohort.rois.items()):
        graphs = cache.get(roi, GraphRecipe())
        signal = roi.label >= 0
        ecm = sp.triu(graphs.ee_d, 1).tocoo()
        rows.append({"roi": name, "mouse": roi.mouse, "cells": summary.n_cells, "patches": summary.n_ecm,
                     "signal_patches": int(signal.sum()), "cell_cell_edges": summary.cell_cell_edges,
                     "ecm_ecm_edges": summary.ecm_ecm_edges, "cell_ecm_edges": summary.cell_ecm_edges,
                     "ecm_ecm_edges_signal": int((signal[ecm.row] & signal[ecm.col]).sum()),
                     "cell_ecm_edges_signal": int(signal[graphs.ce_d.tocoo().col].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ association
def _clean(table, index: list[str], levels: list):
    """Keep every declared category, with zero counts where a category has no edges, and
    report log2(observed/expected) as -inf when either count is zero."""
    import pandas as pd

    table = table.set_index(index).reindex(pd.MultiIndex.from_product(levels, names=index)).reset_index()
    table[["n_obs", "n_exp"]] = table[["n_obs", "n_exp"]].fillna(0)
    table.loc[(table.n_obs <= 0) | (table.n_exp <= 0), "log2_enr"] = -np.inf
    return table


def association(cohort, cache: GraphCache, recipe: GraphRecipe):
    """Observed and expected cell-ECM edges of every cell type and cluster: pooled and per ROI."""
    import mantpy as mt

    carriers = {name: joint_graph(roi, cache.get(roi, recipe)) for name, roi in cohort.rois.items()}
    result = mt.tl.cell_ecm_enrichment_matrix(carriers, K_ecm=cohort.n_labels, cell_types=cohort.cell_types,
                                              n_perm=0, eps=0, return_details=True)
    clusters = list(range(cohort.n_labels))
    per_roi = result.per_roi.rename(columns={"sample_id": "roi"})
    if "roi" not in per_roi and "sample" in per_roi:
        per_roi = per_roi.rename(columns={"sample": "roi"})
    pooled = _clean(result.pooled.copy(), ["cell_type", "cluster"], [cohort.cell_types, clusters])
    per_roi = _clean(per_roi, ["roi", "cell_type", "cluster"], [cohort.roi_names, cohort.cell_types, clusters])
    per_roi["mouse"] = per_roi["roi"].map({name: roi.mouse for name, roi in cohort.rois.items()})
    return (pooled[["cell_type", "cluster", "n_obs", "n_exp", "log2_enr"]],
            per_roi[["roi", "mouse", "cell_type", "cluster", "n_obs", "n_exp", "log2_enr"]])


def marker_contrast(cohort, cache: GraphCache, recipe: GraphRecipe, cell_type: str = FOCAL_CELL_TYPE,
                    cluster: int = WALL_CLUSTER):
    """Per ROI: mean standardised marker intensity of the cluster's patches with an edge to a
    cell of ``cell_type`` minus that of the cluster's patches without one. Each patch counts once."""
    import mantpy as mt
    import pandas as pd
    import scipy.sparse as sp

    rows, contacts = {}, {}
    for name, roi in cohort.rois.items():
        contacts[name] = mt.tl.contact_marker_contrast(
            cache.get(roi, recipe).ce_w, roi.patch_feat, cell_labels=roi.cell_type, cell_type=cell_type,
            patch_ids=roi.patch_id, weighting="unique_patch", accumulation_dtype="input").contact_mask.to_numpy()
        rows[name] = mt.tl.contact_marker_contrast(
            sp.csr_matrix(contacts[name].astype(float)[None, :]), roi.patch_feat, patch_labels=roi.label,
            cluster=cluster, patch_ids=roi.patch_id, weighting="unique_patch",
            accumulation_dtype="input").contrast.to_numpy()
    return pd.DataFrame.from_dict(rows, orient="index", columns=cohort.ecm_names).rename_axis("roi"), contacts


# ------------------------------------------------------------------------------------- edge rules
def edge_rules(n_rules: int, seed: int = EDGE_RULE_SEED) -> list[GraphRecipe]:
    """The Fig. 4 graph followed by ``n_rules`` random cell-ECM edge rules from the search space.

    For each rule the generator draws, in this order, the rule, its distance cap, the number
    of neighbours and whether background patches are nodes. Edges are unweighted.
    """
    rng = np.random.default_rng(seed)
    recipes = [GraphRecipe()]
    for _ in range(n_rules):
        rule = rng.choice(SPACE["ce_rule"])
        cap = rng.choice(list(SPACE["ce_dmax_px"]))
        cap = None if cap is None else float(cap)
        neighbours = int(rng.integers(SPACE["ce_k"][0], SPACE["ce_k"][1] + 1))
        background = bool(rng.choice(SPACE["ee_include_background_nodes"]))
        recipes.append(GraphRecipe(ce_rule=str(rule), ce_k=neighbours, ce_dmax_px=cap,
                                   ee_include_background_nodes=background, edge_weight="none"))
    return recipes


def edge_rule_sensitivity(cohort, recipes: list[GraphRecipe], cell_type: str = FOCAL_CELL_TYPE):
    """Pooled association of ``cell_type`` with every cluster under each edge rule."""
    import mantpy as mt
    import pandas as pd

    rows = []
    for index, recipe in enumerate(recipes):
        carriers, n_edges = {}, 0
        for name, roi in cohort.rois.items():
            graphs = build_graphs(roi.cell_xy, roi.patch_xy, roi.background, roi.patch_size, recipe)
            carriers[name] = joint_graph(roi, graphs)
            n_edges += int((roi.label[graphs.ce_d.tocoo().col] >= 0).sum())
        result = mt.tl.cell_ecm_enrichment(carriers, cell_type=cell_type, K_ecm=cohort.n_labels, n_perm=0, eps=0,
                                           return_details=True)
        for entry in result.pooled.itertuples(index=False):
            rows.append({"recipe_index": index, "is_reference": index == 0, **recipe.to_dict(),
                         "cluster": int(entry.cluster), "n_obs": int(entry.n_obs), "n_exp": float(entry.n_exp),
                         "log2_enr": float(entry.log2_enr) if entry.n_obs > 0 and entry.n_exp > 0 else -np.inf,
                         "n_cell_ecm_edges": n_edges})
    table = pd.DataFrame(rows)
    table["ce_dmax_px"] = table["ce_dmax_px"].astype(float)      # None (no cap) is stored as NaN
    return table


# ------------------------------------------------------------------------------------------ stage
def run(ctx: Context) -> None:
    """Compute the three parts; each is one resumable unit."""
    cohort, cache = ctx.cohort(), ctx.graphs()
    identity = {"partition": ctx.partition, "labels": cohort.label_digest()}
    reference = GraphRecipe()

    unit = ctx.store.unit("describe", ctx.partition, "graphs", request={**identity, "graph": reference.to_dict(),
                                                                       "cell_graph": "squidpy"})
    ctx.store.compute(unit, lambda: (pack(graph_counts(cohort, cache), "counts__"), {"environment": environment()}),
                      label="graph counts", log=log)

    def fit_association():
        pooled, per_roi = association(cohort, cache, reference)
        contrast, contacts = marker_contrast(cohort, cache, reference)
        arrays = {**pack(pooled, "pooled__"), **pack(per_roi, "per_roi__"),
                  **pack(contrast.reset_index(), "contrast__"),
                  "contact": np.concatenate([contacts[name] for name in cohort.roi_names])}
        return arrays, {"environment": environment()}

    unit = ctx.store.unit("describe", ctx.partition, "association",
                          request={**identity, "graph": reference.to_dict(), "cell_type": FOCAL_CELL_TYPE,
                                   "cluster": WALL_CLUSTER})
    ctx.store.compute(unit, fit_association, label="association", log=log)

    recipes = edge_rules(ctx.budget.n_edge_rules)
    unit = ctx.store.unit("describe", ctx.partition, "edge_rules",
                          request={**identity, "cell_type": FOCAL_CELL_TYPE, "seed": EDGE_RULE_SEED,
                                   "recipes": [recipe.to_dict() for recipe in recipes]})
    ctx.store.compute(unit, lambda: (pack(edge_rule_sensitivity(cohort, recipes), "rules__"),
                                     {"environment": environment()}), label="edge rules", log=log)

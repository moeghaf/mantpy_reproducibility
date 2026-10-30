"""Cell-type x ECM-cluster association on the fixed reference graph (within-ROI label permutations)
and the graph-rule sensitivity landscape (effect sizes only)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .data import Cohort
from .graphs import GraphRecipe, build_graphs


def _edge_lists(cohort: Cohort, recipe: GraphRecipe):
    per = []
    for name, roi in cohort.rois.items():
        g = build_graphs(roi.cell_xy, roi.patch_xy, roi.patch_grid, roi.background, roi.patch_size, recipe)
        coo = g.ce_d.tocoo()
        cl = roi.label[coo.col]
        keep = cl >= 0
        per.append({"roi": name, "ct": roi.cell_type_idx.copy(), "e_cell": coo.row[keep], "e_clu": cl[keep]})
    return per


def association_matrix(cohort: Cohort, recipe: GraphRecipe, n_perm: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Degree-adjusted signed residuals, with expected counts recomputed per permutation.

    Two-sided empirical P is twice the smaller one-sided tail, capped at one.
    The +1 correction is applied to each tail; the minimum P is 2/(n_perm+1).
    Uniform within-ROI relabelling does not preserve cell-type autocorrelation.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be positive.")
    T, K = len(cohort.cell_types), cohort.n_labels
    per = _edge_lists(cohort, recipe)
    rng = np.random.default_rng(seed)
    obs = np.zeros((T, K)); exp = np.zeros((T, K))
    for r in per:
        if r["e_clu"].size == 0:
            continue
        np.add.at(obs, (r["ct"][r["e_cell"]], r["e_clu"]), 1)
        freq = np.bincount(r["e_clu"], minlength=K) / r["e_clu"].size
        n_edges_by_type = np.bincount(r["ct"][r["e_cell"]], minlength=T)
        exp += n_edges_by_type[:, None] * freq[None, :]
    null_residual = np.zeros((n_perm, T, K))
    for p in range(n_perm):
        for r in per:
            if r["e_clu"].size == 0:
                continue
            ct_perm = r["ct"][rng.permutation(r["ct"].size)]
            counts = np.bincount(ct_perm[r["e_cell"]] * K + r["e_clu"], minlength=T * K).reshape(T, K)
            expected = counts.sum(1)[:, None] * counts.sum(0)[None, :] / counts.sum()
            null_residual[p] += counts - expected
    eps = 1e-9
    log2 = np.log2((obs + eps) / (exp + eps))
    p_two = empirical_two_tail(obs - exp, null_residual)
    rows = []
    for t in range(T):
        for k in range(K):
            rows.append({"cell_type": cohort.cell_types[t], "cluster": k, "n_obs": int(obs[t, k]), "n_exp": float(exp[t, k]),
                         "log2_enr": float(log2[t, k]) if obs[t, k] > 0 else float("-inf"), "p_value": float(p_two[t, k])})
    df = pd.DataFrame(rows)
    df["p_fdr"] = bh(df["p_value"].to_numpy())
    df["p_floor"] = 2.0 / (n_perm + 1)
    return df


def empirical_two_tail(observed: np.ndarray, null: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Twice the smaller empirical signed-residual tail, including ties.

    A signed-tail comparison also works for a restricted spatial null whose
    residual distribution is not centred on zero.
    """
    if null.shape[0] < 1 or null.shape[1:] != observed.shape:
        raise ValueError("Null samples must have the observed residual shape.")
    lower = (1 + (null <= observed[None] + atol).sum(0)) / (len(null) + 1)
    upper = (1 + (null >= observed[None] - atol).sum(0)) / (len(null) + 1)
    return np.minimum(1.0, 2 * np.minimum(lower, upper))


def bh(p: np.ndarray) -> np.ndarray:
    n = p.size; order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1.0); return out


def per_roi_effects(cohort: Cohort, recipe: GraphRecipe, cell_type: str) -> pd.DataFrame:
    t = cohort.cell_types.index(cell_type); K = cohort.n_labels
    rows = []
    for r in _edge_lists(cohort, recipe):
        if r["e_clu"].size == 0:
            continue
        mask = r["ct"][r["e_cell"]] == t
        obs = np.bincount(r["e_clu"][mask], minlength=K); freq = np.bincount(r["e_clu"], minlength=K) / r["e_clu"].size
        exp = mask.sum() * freq
        for k in range(K):
            rows.append({"roi": r["roi"], "mouse": cohort.rois[r["roi"]].mouse, "cluster": k, "n_obs": int(obs[k]), "n_exp": float(exp[k]),
                         "log2_enr": float(np.log2(obs[k] / exp[k])) if obs[k] > 0 and exp[k] > 0 else float("-inf")})
    return pd.DataFrame(rows)


def sensitivity_landscape(cohort: Cohort, space: dict, cell_type: str, n_recipes: int = 48, seed: int = 0) -> pd.DataFrame:
    """Descriptive effect sizes for the focal cell type under random graph recipes from the search space."""
    G = space["bo_axes"]["graph"]; rng = np.random.default_rng(seed)
    recipes = [GraphRecipe.from_dict(space["fixed_reference_config"])]
    for _ in range(n_recipes):
        rule = rng.choice(G["ce_rule"])
        dm = rng.choice([v for v in G["ce_dmax_px"]]); dm = None if dm is None else float(dm)
        recipes.append(GraphRecipe(ce_rule=str(rule), ce_k=int(rng.integers(G["ce_k"][0], G["ce_k"][1] + 1)), ce_dmax_px=dm,
                                   ee_include_background_nodes=bool(rng.choice(G["ee_include_background_nodes"])), edge_weight="none"))
    t = cohort.cell_types.index(cell_type); K = cohort.n_labels
    rows = []
    for i, rec in enumerate(recipes):
        obs = np.zeros(K); exp = np.zeros(K); n_edges = 0
        for r in _edge_lists(cohort, rec):
            if r["e_clu"].size == 0:
                continue
            mask = r["ct"][r["e_cell"]] == t
            obs += np.bincount(r["e_clu"][mask], minlength=K); exp += mask.sum() * np.bincount(r["e_clu"], minlength=K) / r["e_clu"].size
            n_edges += r["e_clu"].size
        for k in range(K):
            rows.append({"recipe_index": i, "is_reference": i == 0, **rec.to_dict(), "cluster": k, "n_obs": int(obs[k]), "n_exp": float(exp[k]),
                         "log2_enr": float(np.log2(obs[k] / exp[k])) if obs[k] > 0 and exp[k] > 0 else float("-inf"), "n_cell_ecm_edges": int(n_edges)})
    return pd.DataFrame(rows)


def run_association(cohort: Cohort, space: dict, protocol: dict, out_dir: str, cell_type: str = "AEC", log=print) -> None:
    d = os.path.join(out_dir, "association"); os.makedirs(d, exist_ok=True)
    rec = GraphRecipe.from_dict(space["fixed_reference_config"])
    n_perm = int(protocol["association"]["permutations"])
    mat = association_matrix(cohort, rec, n_perm=n_perm, seed=0)
    mat.to_csv(os.path.join(d, "celltype_cluster_matrix.csv"), index=False)
    per_roi_effects(cohort, rec, cell_type).to_csv(os.path.join(d, f"{cell_type}_per_roi.csv"), index=False)
    sensitivity_landscape(cohort, space, cell_type).to_csv(os.path.join(d, f"{cell_type}_graph_sensitivity.csv"), index=False)
    json.dump({"recipe": rec.to_dict(), "n_perm": n_perm, "p_floor": 2 / (n_perm + 1), "bh_scope": "full matrix", "cell_type": cell_type,
               "expected_recomputed_per_permutation": True, "tail": "twice the smaller signed-residual tail"},
              open(os.path.join(d, "association_meta.json"), "w"), indent=1)
    log(f"[association] done: {len(mat)} entries, {int((mat['p_value'] <= 2 / (n_perm + 1) + 1e-12).sum())} at floor")

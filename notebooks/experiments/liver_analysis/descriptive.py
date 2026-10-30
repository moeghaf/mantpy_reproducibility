"""Descriptive statistics of the liver ECM clusters (Fig. 3a-c, i; Supplementary Figs 6 i-j and 6).

Definitions (as in the paper's analysis):
- mouse fraction of a cluster = its patches / ECM signal patches, both summed over the mouse's ROIs; group values are the
  mean of the two mice (equal weight per mouse);
- infection-associated cluster (amendment 3): the largest log2((mean infected fraction + 1e-3) / (mean naive fraction + 1e-3))
  among clusters that make up >= 1 % of the ECM patches of every infected mouse;
- marker profile = per-cluster mean of each feature, z-scored across the six clusters (ddof 0);
- co-occurrence: Delaunay graph over all patch centres (Mantpy), ECM-ECM edges only; every undirected a-b edge adds 1 to
  O[a, b] and 1 to O[b, a]; per mouse, O and the cluster counts n are summed over its ROIs; F = log2((O + 1e-9) / E) with
  E = sum(O) p p^T, p = n / sum(n); the KO - WT contrast is the mean F of the two KO mice minus that of the two WT mice;
- ECM 5 subgraphs: ECM 5 patches joined within 40 px (Mantpy radius reconnection), >= 5 patches; mean degree = 2 E / N.
"""
from __future__ import annotations

import anndata
import mantpy as mt
import numpy as np
import pandas as pd

from .data import K, MARKERS, FEAT_COLS

SPECIFICITY_EPS = 1e-3
SPECIFICITY_PRESENCE = 0.01
SUBGRAPH_RADIUS, SUBGRAPH_MIN = 40.0, 5
SUBGRAPH_GRID = [(r, m) for r in (30.0, 40.0, 50.0) for m in (3, 5, 10)]
KO_MICE, WT_MICE = ["mouse_01", "mouse_02"], ["mouse_03", "mouse_04"]


# ---------------------------------------------------------------- composition and marker profiles
def mouse_fractions(nodes: pd.DataFrame) -> pd.DataFrame:
    """Rows = mice (with group); columns 0..K-1 = fraction of the mouse's ECM patches in each cluster."""
    sig = nodes[nodes.cluster >= 0]
    cnt = sig.groupby(["mouse", "cluster"]).size().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
    frac = cnt.div(cnt.sum(1), axis=0)
    frac.insert(0, "group", sig.groupby("mouse").group.first().reindex(frac.index))
    return frac


def composition_per_mouse(nodes: pd.DataFrame) -> pd.DataFrame:
    return mouse_fractions(nodes).reset_index().rename(columns={k: f"ECM{k}" for k in range(K)})


def infection_cluster(nodes: pd.DataFrame) -> dict:
    """Amendment-3 rule (uses only the per-mouse cluster fractions; no classifier output)."""
    fr = mouse_fractions(nodes)
    inf = fr[fr.group.str.startswith("Infected")][list(range(K))]
    nai = fr[fr.group.str.startswith("Naive")][list(range(K))]
    diff = inf.mean() - nai.mean()
    log2r = np.log2((inf.mean() + SPECIFICITY_EPS) / (nai.mean() + SPECIFICITY_EPS))
    present = (inf >= SPECIFICITY_PRESENCE).all()
    spec_rank = log2r[present].sort_values(ascending=False, kind="mergesort")
    diff_rank = diff.sort_values(ascending=False, kind="mergesort")
    primary = int((diff_rank if len(spec_rank) == 0 else spec_rank).index[0])
    return dict(cluster=primary, difference_rule_cluster=int(diff_rank.index[0]),
                log2_ratio={int(k): float(v) for k, v in log2r.items()},
                present_in_every_infected_mouse={int(k): bool(v) for k, v in present.items()},
                mean_fraction_infected=inf.mean().to_dict(), mean_fraction_naive=nai.mean().to_dict())


def marker_profiles(nodes: pd.DataFrame) -> pd.DataFrame:
    sig = nodes[nodes.cluster >= 0]
    # Accumulate the stored float32 features in float64, as in the original source table.
    features = sig[["cluster", *FEAT_COLS]].astype({c: "float64" for c in FEAT_COLS})
    means = features.groupby("cluster")[FEAT_COLS].mean().reindex(range(K))
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1.0)
    names = [f"mean_{m}" for m in MARKERS] + [f"sd_{m}" for m in MARKERS]
    out = pd.DataFrame({"cluster": range(K), "n_patches": sig.groupby("cluster").size().reindex(range(K)).to_numpy()})
    for c, nm in zip(FEAT_COLS, names):
        out[f"raw_{nm}"] = means[c].to_numpy()
    for c, nm in zip(FEAT_COLS, names):
        out[f"z_{nm}"] = z[c].to_numpy()
    return out


# ---------------------------------------------------------------- graphs (Mantpy) and co-occurrence
def roi_graph(sub: pd.DataFrame, edge_method: str = "delaunay", grid_connectivity: int = 8) -> anndata.AnnData:
    """Mantpy ECM graph over all patch centres of one ROI (background patches carry cluster -1)."""
    ad = anndata.AnnData(np.zeros((0, len(MARKERS)), dtype=np.float32))
    ad.uns["ecm_patches"] = pd.DataFrame({"x": sub.x.to_numpy(float), "y": sub.y.to_numpy(float), "ecm_cluster": sub.cluster.to_numpy()})
    mt.gr.build_ecm_graph(ad, edge_method=edge_method, grid_connectivity=grid_connectivity, edge_weight="uniform")
    return ad


def edge_array(ad: anndata.AnnData) -> np.ndarray:
    """Undirected edges as (E, 2) row indices into the ROI's patch table (Mantpy names nodes 'ecm_<row>')."""
    e = np.array([(int(str(u).split("_")[-1]), int(str(v).split("_")[-1])) for u, v in ad.uns["ecm_graph"].edges()], dtype=np.int64)
    e.sort(1)
    return e


def ordered_counts(edges: np.ndarray, cl: np.ndarray):
    a, b = cl[edges[:, 0]], cl[edges[:, 1]]
    keep = (a >= 0) & (b >= 0)
    O = np.zeros((K, K))
    np.add.at(O, (a[keep], b[keep]), 1.0)
    np.add.at(O, (b[keep], a[keep]), 1.0)
    n = np.bincount(cl[cl >= 0], minlength=K).astype(float)
    return O, n


def score(O: np.ndarray, n: np.ndarray):
    p = n / n.sum() if n.sum() else np.zeros_like(n)
    E = O.sum() * np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.log2((O + 1e-9) / E)
    F[E == 0] = np.nan
    return F, E


def mantpy_score(adatas: list, O: np.ndarray, n: np.ndarray):
    """Public Mantpy enrichment, checked against independent ordered endpoint counts.

    Keep absent-cluster cells missing in this study's descriptive tables. Mantpy
    returns zero for those cells; the explicit E == 0 mask does not affect any
    estimable enrichment or contrast.
    """
    F = mt.tl.cluster_cooccurrence(adatas, n_clusters=K, eps=1e-9)
    independent, E = score(O, n)
    present = E > 0
    difference = float(np.max(np.abs(F[present] - independent[present]))) if present.any() else 0.0
    if not np.allclose(F[present], independent[present], rtol=0, atol=1e-12):
        raise AssertionError(f"Mantpy co-occurrence differs from ordered endpoint counts: {difference:g}")
    F[E == 0] = np.nan
    return F, E, difference


def cooccurrence(nodes: pd.DataFrame, edge_method: str = "delaunay", grid_connectivity: int = 8, infection_cluster_id: int = 5,
                 graphs: dict | None = None) -> dict:
    """Co-occurrence per ROI, per infected mouse and the KO - WT contrasts. `graphs` (optional) caches the Mantpy graphs by ROI."""
    rois = sorted(nodes.roi.unique())
    meta = nodes.groupby("roi")[["mouse", "group"]].first()
    roi_O, roi_n, roi_ad, inv, per_roi, validation = {}, {}, {}, [], [], []
    for roi in rois:
        sub = nodes[nodes.roi == roi].reset_index(drop=True)
        ad = graphs[roi] if graphs is not None and roi in graphs else roi_graph(sub, edge_method, grid_connectivity)
        if graphs is not None:
            graphs[roi] = ad
        roi_ad[roi] = ad
        e = edge_array(ad)
        cl = sub.cluster.to_numpy()
        roi_O[roi], roi_n[roi] = ordered_counts(e, cl)
        ss = (cl[e[:, 0]] >= 0) & (cl[e[:, 1]] >= 0)
        inv.append(dict(roi=roi, mouse=meta.loc[roi, "mouse"], group=meta.loc[roi, "group"], n_nodes=len(sub), n_signal=int((cl >= 0).sum()),
                        n_edges_undirected=len(e), n_edges_signal_signal=int(ss.sum())))
        F, E, difference = mantpy_score([ad], roi_O[roi], roi_n[roi])
        validation.append(dict(level="ROI", unit=roi, max_abs_difference=difference,
                               estimable_cells=int((E > 0).sum()), absent_cells=int((E == 0).sum())))
        for a in range(K):
            for b in range(K):
                per_roi.append(dict(roi=roi, mouse=meta.loc[roi, "mouse"], group=meta.loc[roi, "group"], cluster_a=a, cluster_b=b,
                                    O=roi_O[roi][a, b], E=E[a, b], F=F[a, b]))
    inf_mice = KO_MICE + WT_MICE
    mO = {m: sum(roi_O[r] for r in rois if meta.loc[r, "mouse"] == m) for m in inf_mice}
    mn = {m: sum(roi_n[r] for r in rois if meta.loc[r, "mouse"] == m) for m in inf_mice}
    mF = {}
    for m in inf_mice:
        mF[m], E, difference = mantpy_score([roi_ad[r] for r in rois if meta.loc[r, "mouse"] == m], mO[m], mn[m])
        validation.append(dict(level="mouse", unit=m, max_abs_difference=difference,
                               estimable_cells=int((E > 0).sum()), absent_cells=int((E == 0).sum())))
    pool = {}
    for g, ms in (("KO", KO_MICE), ("WT", WT_MICE)):
        ads = [roi_ad[r] for r in rois if meta.loc[r, "mouse"] in ms]
        pool[g], E, difference = mantpy_score(ads, sum(mO[m] for m in ms), sum(mn[m] for m in ms))
        validation.append(dict(level="genotype pool", unit=g, max_abs_difference=difference,
                               estimable_cells=int((E > 0).sum()), absent_cells=int((E == 0).sum())))
    pairs = []
    for a in range(K):
        for b in range(a, K):
            pairs.append(dict(cluster_a=a, cluster_b=b,
                              delta_equal_mouse=np.mean([mF[m][a, b] for m in KO_MICE]) - np.mean([mF[m][a, b] for m in WT_MICE]),
                              delta_pooled_tissue=pool["KO"][a, b] - pool["WT"][a, b],
                              **{f"F_{m}": mF[m][a, b] for m in inf_mice}, **{f"O_{m}": mO[m][a, b] for m in inf_mice},
                              involves_infection_cluster=bool(infection_cluster_id in (a, b))))
    return dict(pairs=pd.DataFrame(pairs), per_roi=pd.DataFrame(per_roi), inventory=pd.DataFrame(inv),
                validation=pd.DataFrame(validation))


def pair_matrix(pairs: pd.DataFrame, column: str) -> np.ndarray:
    D = np.full((K, K), np.nan)
    for _, p in pairs.iterrows():
        D[int(p.cluster_a), int(p.cluster_b)] = D[int(p.cluster_b), int(p.cluster_a)] = p[column]
    return D


# ---------------------------------------------------------------- ECM 5 subgraphs (Mantpy radius reconnection)
def subgraphs(nodes: pd.DataFrame, graphs: dict, cluster: int = 5, settings=None) -> dict:
    """Components of `cluster` per ROI for each (radius, min_nodes) setting, from Mantpy's radius reconnection.

    Returns {"table": subgraphs at 40 px / >= 5 patches, "sensitivity": per-mouse summary over the settings grid,
             "nodes": node-level membership and degree (40 px) for maps}.
    """
    settings = settings or SUBGRAPH_GRID
    meta = nodes.groupby("roi")[["mouse", "group"]].first()
    comps = {}  # (roi, radius) -> list of Mantpy component records
    for roi in sorted(nodes.roi.unique()):
        has = bool((nodes[nodes.roi == roi].cluster == cluster).any())
        for r in sorted({r for r, _ in settings} | {SUBGRAPH_RADIUS}):
            comps[(roi, r)] = mt.gr.extract_components_radius_reconnect(graphs[roi], target_cluster=cluster, radius=r, min_nodes=1) if has else []
    rows, node_rows = [], []
    for roi in sorted(nodes.roi.unique()):
        kept = sorted([c for c in comps[(roi, SUBGRAPH_RADIUS)] if c["n_nodes"] >= SUBGRAPH_MIN], key=lambda c: -c["n_nodes"])
        for k, c in enumerate(kept, start=1):
            rows.append(dict(roi=roi, mouse=meta.loc[roi, "mouse"], group=meta.loc[roi, "group"], component=k, n_nodes=int(c["n_nodes"]),
                             n_edges=int(c["n_edges"]), mean_degree=2 * c["n_edges"] / c["n_nodes"]))
        for j, c in enumerate(comps[(roi, SUBGRAPH_RADIUS)]):
            # Mantpy numbers the subgraph's nodes by their position among all target-cluster nodes; coords follow sorted(nodes)
            sub = c["subgraph"]; labels = sorted(sub.nodes)
            for i, (x, y) in enumerate(np.asarray(c["coords"], float)):
                node_rows.append(dict(roi=roi, x=float(x), y=float(y), component_index=j, component_size=int(c["n_nodes"]),
                                      retained=bool(c["n_nodes"] >= SUBGRAPH_MIN), node_degree_40px=int(sub.degree(labels[i]))))
    sens = []
    for r, mnod in settings:
        for mouse, g in meta.groupby("mouse").group.first().items():
            cc = [c for roi in meta.index[meta.mouse == mouse] for c in comps[(roi, r)] if c["n_nodes"] >= mnod]
            sens.append(dict(radius_px=r, min_nodes=mnod, mouse=mouse, group=g, n_components=len(cc),
                             median_n_nodes=float(np.median([c["n_nodes"] for c in cc])) if cc else np.nan,
                             median_mean_degree=float(np.median([2 * c["n_edges"] / c["n_nodes"] for c in cc])) if cc else np.nan,
                             total_infection_cluster_nodes_retained=int(sum(c["n_nodes"] for c in cc))))
    cols = ["roi", "mouse", "group", "component", "n_nodes", "n_edges", "mean_degree"]
    return dict(table=pd.DataFrame(rows, columns=cols), sensitivity=pd.DataFrame(sens), nodes=pd.DataFrame(node_rows))


# ---------------------------------------------------------------- display composites
def otsu(v, nbins=256) -> float:
    hist, edges = np.histogram(v, bins=nbins)
    c = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist); w2 = w1[-1] - w1
    m1 = np.cumsum(hist * c) / np.maximum(w1, 1)
    m2 = (np.cumsum((hist * c)[::-1])[::-1]) / np.maximum(w2, 1)
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[:-1]) ** 2
    return float(c[np.argmax(var)])


def composites(norm: dict, colours: dict) -> tuple[dict, dict]:
    """Six-marker RGB composites with one display scale for the cohort: per marker, an Otsu threshold and a 99.5th-percentile
    ceiling over the pooled non-zero pixels of all ROIs; colours are added."""
    rois = sorted(norm)
    thr = {}
    for c, m in enumerate(MARKERS):
        pos = np.concatenate([norm[r][c][norm[r][c] > 0] for r in rois])
        thr[m] = (otsu(pos), float(np.percentile(pos, 99.5)))
        del pos
    to_rgb = lambda h: np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    out = {}
    for r in rois:
        rgb = np.zeros(norm[r].shape[1:] + (3,))
        for c, m in enumerate(MARKERS):
            t, hi = thr[m]
            im = norm[r][c].astype(float)
            rgb += np.where(im > t, np.clip((im - t) / max(hi - t, 1e-6), 0, 1), 0.0)[..., None] * to_rgb(colours[m])
        out[r] = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return out, thr


# ---------------------------------------------------------------- explainer summary
def ig_localisation(att: pd.DataFrame, infection_cluster_id: int = 5) -> pd.DataFrame:
    """Per ROI and cluster: Pearson r between |importance| (IG towards 'infected') and cluster membership."""
    loc = []
    for roi, sub in att.groupby("roi", sort=True):
        for k in range(K):
            m = (sub.cluster == k).to_numpy(float)
            r = float(np.corrcoef(sub.ig_infected, m)[0, 1]) if 2 <= m.sum() < len(m) and sub.ig_infected.std() > 0 else np.nan
            loc.append(dict(roi=roi, mouse=sub.mouse.iloc[0], group=sub.group.iloc[0], cluster=k, is_infection_cluster=k == infection_cluster_id,
                            n_cluster_nodes=int(m.sum()), n_nodes=len(m), pearson_r_abs_ig_vs_membership=r))
    return pd.DataFrame(loc)

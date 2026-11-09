"""Descriptive statistics of the frozen ECM clusters (Fig. 3a-c, h, i; Supplementary Fig. 2b, c).

Definitions of the paper:

- mouse fraction of a cluster = its patches / ECM signal patches, both pooled over the mouse's ROIs;
  group values are the mean of the two mice;
- infection-associated cluster = the largest log2((mean infected fraction + 1e-3) / (mean naive
  fraction + 1e-3)) among clusters making up at least 1% of the ECM patches of every infected mouse;
- graph = edges of the Delaunay triangulation of all patch centres of an ROI; co-occurrence counts
  the edges joining two ECM signal patches. Every a-b edge adds 1 to O[a, b] and 1 to O[b, a]
  (O = L^T A L); O and the cluster sizes n are summed within a mouse; the score is
  log2((O + 1e-9) / E) with E = sum(O) p p^T and p = n / sum(n);
- KO - WT = mean score of the two infected KO mice minus that of the two infected WT mice;
- subgraphs = patches of the infection-associated cluster joined within 40 pixels, connected
  components of at least five patches; mean degree = 2 x edges / patches;
- localisation of the explanation = Pearson correlation, within a held-out ROI, between patch
  importance and membership of a cluster.
"""

from __future__ import annotations

import numpy as np

from .config import MARKERS, SPECIFICITY_EPS, SPECIFICITY_PRESENCE, SUBGRAPH_MIN_NODES, SUBGRAPH_RADIUS


# ------------------------------------------------------------------------------------ composition
def mouse_fractions(nodes, k: int):
    """Rows = mice (with their group); columns 0..K-1 = fraction of the mouse's ECM signal patches."""
    signal = nodes[nodes.cluster >= 0]
    counts = signal.groupby(["mouse", "cluster"]).size().unstack(fill_value=0).reindex(columns=range(k), fill_value=0)
    fractions = counts.div(counts.sum(1), axis=0)
    fractions.insert(0, "group", signal.groupby("mouse").group.first().reindex(fractions.index))
    return fractions


def infection_cluster(nodes, k: int) -> dict:
    """The infection-associated cluster, from per-mouse fractions only (no classifier output)."""
    fractions = mouse_fractions(nodes, k)
    infected = fractions[fractions.group.str.startswith("Infected")][list(range(k))]
    naive = fractions[fractions.group.str.startswith("Naive")][list(range(k))]
    difference = (infected.mean() - naive.mean()).sort_values(ascending=False, kind="mergesort")
    log2_ratio = np.log2((infected.mean() + SPECIFICITY_EPS) / (naive.mean() + SPECIFICITY_EPS))
    present = (infected >= SPECIFICITY_PRESENCE).all()
    ranked = log2_ratio[present].sort_values(ascending=False, kind="mergesort")
    # No cluster present in every infected mouse: fall back to the largest infected - naive difference.
    chosen = int((ranked if len(ranked) else difference).index[0])
    return {"cluster": chosen, "log2_ratio": {int(c): float(v) for c, v in log2_ratio.items()},
            "present_in_every_infected_mouse": {int(c): bool(v) for c, v in present.items()}}


def marker_profiles(nodes, features: np.ndarray, k: int):
    """Mean of each marker per cluster over ECM signal patches, raw and z-scored across the clusters."""
    import pandas as pd

    keep = nodes.cluster.to_numpy() >= 0
    table = pd.DataFrame(features[keep, :len(MARKERS)].astype(np.float64), columns=MARKERS)
    table["cluster"] = nodes.cluster.to_numpy()[keep]
    means = table.groupby("cluster")[MARKERS].mean().reindex(range(k))
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1.0)
    out = pd.DataFrame({"cluster": range(k), "n_patches": table.groupby("cluster").size().reindex(range(k)).to_numpy()})
    for marker in MARKERS:
        out[f"raw_mean_{marker}"] = means[marker].to_numpy()
    for marker in MARKERS:
        out[f"z_mean_{marker}"] = z[marker].to_numpy()
    return out


# ------------------------------------------------------------------------- graphs, co-occurrence
def delaunay_edges(xy: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E x 2, i < j) of the Delaunay triangulation (SciPy's default Qhull options)."""
    from scipy.spatial import Delaunay

    if len(xy) < 3:
        return np.zeros((0, 2), dtype=np.int64)
    simplices = Delaunay(xy).simplices
    edges = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    edges.sort(1)
    return np.unique(edges, axis=0).astype(np.int64)


def ordered_counts(edges: np.ndarray, cluster: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """O (K x K ordered-endpoint counts over signal-signal edges) and n (signal patches per cluster)."""
    a, b = cluster[edges[:, 0]], cluster[edges[:, 1]]
    keep = (a >= 0) & (b >= 0)
    a, b = a[keep], b[keep]
    observed = np.zeros((k, k))
    np.add.at(observed, (a, b), 1.0)
    np.add.at(observed, (b, a), 1.0)
    return observed, np.bincount(cluster[cluster >= 0], minlength=k).astype(float)


def cooccurrence_score(observed: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log2((O + 1e-9) / E) and E; the score is missing where a cluster is absent (E = 0)."""
    p = sizes / sizes.sum()
    expected = observed.sum() * np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.log2((observed + 1e-9) / expected)
    score[expected == 0] = np.nan
    return score, expected


def cooccurrence(nodes, k: int):
    """Co-occurrence of every cluster pair in each infected mouse and the KO - WT differences.

    Returns ``(pairs, per_roi)``: one row per unordered cluster pair, and one row per ROI and
    ordered pair.
    """
    import pandas as pd

    meta = nodes.groupby("roi")[["mouse", "group"]].first()
    counts, per_roi = {}, []
    for roi, sub in nodes.groupby("roi", sort=True):
        cluster = sub.cluster.to_numpy()
        counts[roi] = ordered_counts(delaunay_edges(sub[["x", "y"]].to_numpy(float)), cluster, k)
        score, expected = cooccurrence_score(*counts[roi])
        per_roi += [{"roi": roi, "mouse": meta.loc[roi, "mouse"], "group": meta.loc[roi, "group"], "cluster_a": a,
                     "cluster_b": b, "observed": counts[roi][0][a, b], "expected": expected[a, b], "log2_obs_exp": score[a, b]}
                    for a in range(k) for b in range(k)]
    genotype = {g: sorted(meta[meta.group == f"Infected_{g}"].mouse.unique()) for g in ("KO", "WT")}
    mice = genotype["KO"] + genotype["WT"]
    pooled = {mouse: (sum(counts[roi][0] for roi in meta.index[meta.mouse == mouse]),
                      sum(counts[roi][1] for roi in meta.index[meta.mouse == mouse])) for mouse in mice}
    score = {mouse: cooccurrence_score(*pooled[mouse])[0] for mouse in mice}
    pairs = [{"cluster_a": a, "cluster_b": b,
              "ko_minus_wt": np.mean([score[m][a, b] for m in genotype["KO"]]) - np.mean([score[m][a, b] for m in genotype["WT"]]),
              **{f"log2_obs_exp_{m}": score[m][a, b] for m in mice},
              **{f"observed_{m}": pooled[m][0][a, b] for m in mice}}
             for a in range(k) for b in range(a, k)]
    return pd.DataFrame(pairs), pd.DataFrame(per_roi)


# -------------------------------------------------------------------------------------- subgraphs
def components(xy: np.ndarray, radius: float = SUBGRAPH_RADIUS, min_nodes: int = SUBGRAPH_MIN_NODES) -> list[dict]:
    """Connected components of the points joined within ``radius``, keeping those of at least ``min_nodes``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(xy) == 0:
        return []
    pairs = cKDTree(xy).query_pairs(radius + 1e-9, output_type="ndarray")
    adjacency = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    n_components, label = connected_components(adjacency, directed=False)
    out = []
    for component in range(n_components):
        members = np.flatnonzero(label == component)
        if len(members) < min_nodes:
            continue
        n_edges = int((np.isin(pairs[:, 0], members) & np.isin(pairs[:, 1], members)).sum())
        out.append({"n_nodes": int(len(members)), "n_edges": n_edges, "mean_degree": 2 * n_edges / len(members)})
    return out


def subgraphs(nodes, cluster: int):
    """Subgraphs of ``cluster`` in every ROI, largest first within an ROI."""
    import pandas as pd

    meta = nodes.groupby("roi")[["mouse", "group"]].first()
    rows = []
    for roi in sorted(meta.index):
        xy = nodes.loc[(nodes.roi == roi) & (nodes.cluster == cluster), ["x", "y"]].to_numpy(float)
        for number, component in enumerate(sorted(components(xy), key=lambda item: -item["n_nodes"]), start=1):
            rows.append({"roi": roi, "mouse": meta.loc[roi, "mouse"], "group": meta.loc[roi, "group"], "component": number,
                         **component})
    return pd.DataFrame(rows, columns=["roi", "mouse", "group", "component", "n_nodes", "n_edges", "mean_degree"])


# ------------------------------------------------------------------------------------ explanation
def ig_localisation(attribution, k: int, cluster: int):
    """Per ROI and cluster: Pearson r between patch importance and cluster membership.

    The correlation is defined when the cluster has at least two patches in the ROI and does not
    fill it.
    """
    import pandas as pd

    rows = []
    for roi, sub in attribution.groupby("roi", sort=True):
        importance = sub.importance.to_numpy()
        for c in range(k):
            member = (sub.cluster == c).to_numpy(float)
            defined = 2 <= member.sum() < len(member) and importance.std() > 0
            rows.append({"roi": roi, "mouse": sub.mouse.iloc[0], "group": sub.group.iloc[0], "cluster": c,
                         "is_infection_cluster": c == cluster, "n_cluster_nodes": int(member.sum()), "n_nodes": len(member),
                         "pearson_r": float(np.corrcoef(importance, member)[0, 1]) if defined else np.nan})
    return pd.DataFrame(rows)

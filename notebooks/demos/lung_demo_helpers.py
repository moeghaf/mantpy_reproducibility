"""Small plotting and cell-context helpers for the mouse-lung tutorial."""
import mantpy as mt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score


def patch_grid(patches, values, patch_size=10):
    """Place row-aligned values on the original, complete patch lattice."""
    x = (patches["x"].to_numpy() / patch_size).astype(int)
    y = (patches["y"].to_numpy() / patch_size).astype(int)
    grid = np.full((y.max() + 1, x.max() + 1), np.nan)
    grid[y, x] = values
    return grid


def plot_patch_map(patches, title, *, labels=None, palette=None, patch_size=10):
    labels = patches["ecm_cluster"].to_numpy(int) if labels is None else np.asarray(labels)
    colours = [palette[k] for k in range(3)] if palette else ["#56B4E9", "#D55E00", "#009E73"]
    shape = patch_grid(patches, labels, patch_size).shape
    _, ax = plt.subplots(figsize=(6.5, 6.5))
    mt.pl.patch_map(patches, labels, patch_size=patch_size, ax=ax, title=title,
                    categorical=True, palette=dict(enumerate(colours)), vmin=-.5, vmax=2.5,
                    extent=(0, shape[1]*patch_size, shape[0]*patch_size, 0), background="white")
    ax.set(xlabel="x (µm)", ylabel="y (µm)")
    ax.legend([Patch(facecolor=c) for c in colours], ["ECM 0", "ECM 1", "ECM 2"],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()


def cell_ecm_edges(cells):
    """Cell row, patch row and physical length for each public joint-graph edge."""
    edges = []
    graph = cells.uns["cell_ecm_graph"]
    for u, v, attributes in graph.edges(data=True):
        if attributes.get("edge_type") != "cell-ecm":
            continue
        cell, patch = (u, v) if str(u).startswith("cell_") else (v, u)
        edges.append((int(str(cell).split("_")[1]), int(str(patch).split("_")[1]),
                      float(attributes["distance"])))
    return np.asarray(edges, dtype=float).reshape(-1, 3)


def aec_contact_mask(cells):
    """Each patch is counted once even if it has multiple AEC edges."""
    import scipy.sparse as sp
    edges = cell_ecm_edges(cells)
    patches = cells.uns["ecm_patches"]
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0].astype(int), edges[:, 1].astype(int))),
        shape=(cells.n_obs, len(patches)),
    ).tocsr()
    columns = [column for column in patches if column.startswith("feat_")]
    return mt.tl.contact_marker_contrast(
        adjacency, patches[columns], cell_labels=cells.obs["cell_type"].astype(str).to_numpy(),
        cell_type="AEC", weighting="unique_patch", accumulation_dtype="input",
    ).contact_mask.to_numpy()


def cell_context_features(cells, cell_types, *, edge_weight="gaussian_median"):
    """Original weighted cell context through the public label aggregation API."""
    import scipy.sparse as sp
    edges = cell_ecm_edges(cells)
    distance = edges[:, 2]
    if edge_weight == "gaussian_median":
        sigma = max(float(np.median(distance)), 1e-12) if len(distance) else 1e-12
        weights = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    elif edge_weight == "none":
        weights = np.ones(len(distance))
    else:
        raise ValueError(f"Unsupported edge weight: {edge_weight}")
    adjacency = sp.coo_matrix((weights, (edges[:, 1].astype(int), edges[:, 0].astype(int))),
                              shape=(len(cells.uns["ecm_patches"]), cells.n_obs)).tocsr()
    return mt.tl.neighbourhood_features(
        adjacency, cells.obs["cell_type"].astype(str).to_numpy(), categories=cell_types,
        normalization="counts", exclude_self=False,
    ).to_numpy()


def mixed_corruption(xy, pristine, *, seed=0, fraction=0.1, block_sides=(50, 100, 150),
                     min_block_patches=20):
    """Frozen mixed protocol: half square blocks, half isolated replacements."""
    xy = np.asarray(xy, dtype=float)
    pristine = np.asarray(pristine, dtype=int)
    rng = np.random.default_rng(seed)
    eligible = pristine >= 0
    budget = int(round(fraction * eligible.sum()))
    block_budget = budget // 2
    observed = pristine.copy()
    changed = np.zeros(len(pristine), dtype=bool)
    blocks = []
    attempts = 0
    while changed.sum() < block_budget and attempts < 200:
        attempts += 1
        side = float(rng.choice(block_sides))
        half = side / 2
        indices = np.flatnonzero(eligible & ~changed)
        tree = cKDTree(xy[indices])
        for _ in range(60):
            anchor = xy[rng.choice(indices)]
            centre = anchor + rng.uniform(-half, half, 2)
            local = tree.query_ball_point(centre, r=half * np.sqrt(2))
            inside = indices[np.asarray(local, dtype=int)]
            inside = inside[(np.abs(xy[inside] - centre) <= half).all(axis=1)]
            if len(inside) < min_block_patches:
                continue
            replacement = int(rng.integers(0, 3))
            flip = inside[pristine[inside] != replacement]
            if len(flip):
                break
        else:
            continue
        remaining = block_budget - int(changed.sum())
        if len(flip) > remaining:
            flip = rng.choice(flip, remaining, replace=False)
        observed[flip] = replacement
        changed[flip] = True
        blocks.append({"centre": centre.tolist(), "side": side, "n_flipped": len(flip)})
    if changed.sum() != block_budget:
        raise ValueError("Could not place the prescribed block-corruption budget.")
    isolated = rng.choice(np.flatnonzero(eligible & ~changed),
                          budget - block_budget, replace=False)
    for i in isolated:
        observed[i] = int(rng.choice(np.arange(3)[np.arange(3) != pristine[i]]))
        changed[i] = True
    assert changed.sum() == budget
    return observed, changed, blocks


def reconstruction_metrics(pristine, observed, cleaned, changed, anomaly):
    """Single-realisation metrics, evaluated only on signal patches."""
    signal = pristine >= 0
    uncorrupted = signal & ~changed
    return {
        "signal_patches": int(signal.sum()),
        "replaced_patches": int(changed.sum()),
        "detection_ROC_AUC": roc_auc_score(changed[signal], anomaly[signal]),
        "replaced_label_recovery": float((cleaned[changed] == pristine[changed]).mean()),
        "uncorrupted_error": float((cleaned[uncorrupted] != pristine[uncorrupted]).mean()),
        "accuracy_before": float((observed[signal] == pristine[signal]).mean()),
        "accuracy_after": float((cleaned[signal] == pristine[signal]).mean()),
    }

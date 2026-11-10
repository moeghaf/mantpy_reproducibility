"""Paper-specific pieces of the lung Figure 4 notebook.

The notebook shows every analysis step as a plain Mantpy call. This file holds what is specific to
reproducing the paper, and each function takes everything it uses as an argument:

- the paper's values, read from the source-data workbook into tables with explicit keys
  (``read_paper_values``), and the one-to-one matching of this run's clusters to the paper's
  (``match_clusters``);
- comparisons that are joined on those keys (``side_by_side``);
- the selected settings of Supplementary Table 6, translated into Mantpy arguments (``graph_options``,
  ``feature_options``), and the loop that applies the calls shown in the notebook to every held-out
  mouse, context, ROI and realisation (``evaluate_setting``, ``score_label_prior``);
- the parts of the panels that Mantpy does not draw (``draw_contacted_patches``, ``mark_corruption``,
  ``draw_detection``).
"""

from __future__ import annotations

import mantpy as mt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------------------------------
# The paper's values
# ---------------------------------------------------------------------------------------------------

# Marker names of the workbook -> marker names of the dataset.
MARKER_NAMES = {
    "HA": "HABP", "Col VI": "Col-VI", "HS": "HepSul", "Col IV": "Col-IV", "CS": "Chon-Sul",
    "Fibrinogen": "Fibrinogen", "Fbg": "Fibrinogen", "Col III": "Col-III", "Fibronectin": "Fibronectin",
    "FN": "Fibronectin", "Laminin γ1": "lamg1", "Lam γ1": "lamg1", "Col I": "Col-I",
}
CONTEXT_NAMES = {"cell context": "cell", "corrupt ECM": "ecm", "cell + corrupt ECM": "joint", "label prior": "prior"}
LABEL_NUMBERS = {"background": -1, "ECM 0": 0, "ECM 1": 1, "ECM 2": 2}

SIGNAL_PATCHES = 11_435   # ECM signal patches in the six ROIs, as stated in the paper's text

# Values that the paper states in its text and that are not in the workbook (cell context, mouse held out).
TEXT_VALUES = {
    "replaced labels corrected, mean": 0.70,
    "replaced labels corrected, lowest ROI": 0.52,
    "replaced labels corrected, highest ROI": 0.78,
    "uncorrupted labels made wrong": 0.021,
    "label accuracy before": 0.90,
    "label accuracy after": 0.95,
    "corrected, ECM 0 (alveolar)": 0.80,
    "corrected, ECM 1 (bronchovascular wall)": 0.48,
    "corrected, ECM 2 (adventitial)": 0.33,
}


def _cluster_number(names):
    return names.map(LABEL_NUMBERS).astype(int)


def _patch_index(ids):
    """``roi_001:17`` -> ``17``, the index of the patch in the patch table of its ROI."""
    return pd.Index(ids.str.split(":").str[1], name="patch")


def read_paper_values(path):
    """Read the paper's Figure 4 values into one table per quantity, each with explicit keys.

    Clusters are keyed by their number (-1 for background), ROIs by their name, markers by the names of
    the dataset, patches by their index in the patch table of their ROI, contexts by ``cell``, ``ecm``,
    ``joint`` and ``prior``.
    """
    sheets = pd.read_excel(path, sheet_name=None)

    nodes, edges = sheets["a_c_nodes"], sheets["a_c_edges"]
    patches = nodes[nodes["kind"].eq("ECM patch")]
    graph = edges["layer"].value_counts().rename(index="{} edges".format)
    graph["cells"] = int(nodes["kind"].eq("cell").sum())

    profiles = sheets["d_profiles"].set_index("cluster").rename(columns=MARKER_NAMES)
    profiles.index = _cluster_number(profiles.index.to_series()).to_numpy()

    association = sheets["e_association"].assign(cluster=lambda t: _cluster_number(t["cluster"]))
    contrast = sheets["f_contrast"].assign(marker=lambda t: t["marker"].map(MARKER_NAMES))
    if contrast["marker"].isna().any() or profiles.columns.isna().any():
        raise ValueError("The workbook has a marker name that this file does not know.")

    example = sheets["h_i_labels"]
    detection = sheets["j_per_roi"].assign(context=lambda t: t["context"].map(CONTEXT_NAMES))
    realisations = sheets["j_realisations"].assign(context=lambda t: t["context"].map(CONTEXT_NAMES))
    columns = ["roc_auc_raw", "roc_auc_prior_adjusted"]
    return {
        "graph": graph.rename_axis(None),
        "graph_patch_labels": pd.Series(_cluster_number(patches["label"]).to_numpy(),
                                        index=_patch_index(patches["node"]), name="cluster"),
        "profiles": profiles.rename_axis("cluster"),
        "association": association.pivot(index="roi", columns="cluster", values="log2_obs_exp_roi"),
        "association_edges": association.pivot(index="roi", columns="cluster", values="n_obs"),
        "association_pooled": association.groupby("cluster")["log2_obs_exp_pooled"].first(),
        "contrast": contrast.pivot(index="roi", columns="marker", values="difference_roi").rename_axis(columns=None),
        "contrast_mean": contrast.groupby("marker")["difference_mean_over_rois"].first().rename_axis(None),
        "contacted_patches": _patch_index(sheets["g_patches"]["patch_id"]),
        "example_labels": example.drop(columns=["roi", "patch_id"]).set_index(_patch_index(example["patch_id"])),
        "example_blocks": sheets["h_blocks"],
        "detection": detection.set_index(["context", "roi"])[columns],
        "detection_realisations": realisations.set_index(["context", "roi", "realisation"])[columns],
        "text": pd.Series(TEXT_VALUES),
        "signal_patches": SIGNAL_PATCHES,
    }


def match_clusters(profiles, paper_profiles):
    """Pair every cluster of this run with one cluster of the paper, by their marker profiles.

    Both tables have one row per cluster and one column per marker. The pairing is one-to-one and
    maximises the summed Pearson correlation of the paired profiles, so two clusters of this run can
    never be compared with the same paper cluster. The last column shows how close the nearest
    alternative was.
    """
    if profiles.shape != paper_profiles.shape or list(profiles.columns) != list(paper_profiles.columns):
        raise ValueError("The two profile tables need the same clusters-by-markers shape and marker order.")
    n = len(profiles)
    similarity = np.corrcoef(profiles.to_numpy(float), paper_profiles.to_numpy(float))[:n, n:]
    rows, columns = linear_sum_assignment(-similarity)
    others = similarity.copy()
    others[rows, columns] = -np.inf
    return pd.DataFrame(
        {"paper cluster": paper_profiles.index[columns],
         "profile correlation": similarity[rows, columns],
         "highest correlation with another paper cluster": others.max(axis=1)[rows]},
        index=pd.Index(profiles.index[rows], name="cluster"),
    )


def side_by_side(this_run, paper):
    """This run's values next to the paper's, joined on their index.

    Both sides need unique keys and the same set of keys (and, for tables, the same columns). A missing
    or repeated key raises an error instead of leaving a gap or a wrong pairing. For tables, every
    column is shown twice, once for this run and once for the paper.
    """
    for name, values in (("this run", this_run), ("the paper", paper)):
        if not values.index.is_unique:
            raise ValueError(f"The keys of {name} are not unique.")
    unmatched = set(this_run.index) ^ set(paper.index)
    if unmatched:
        raise ValueError(f"Keys present on one side only: {sorted(unmatched, key=str)}")
    paper = paper.loc[this_run.index]
    if isinstance(this_run, pd.Series):
        return pd.DataFrame({"this run": this_run, "paper": paper})
    if set(this_run.columns) != set(paper.columns):
        raise ValueError("The two tables have different columns.")
    both = pd.concat({"this run": this_run, "paper": paper[this_run.columns]}, axis=1).swaplevel(axis=1)
    return both[[(column, side) for column in this_run.columns for side in ("this run", "paper")]]


# ---------------------------------------------------------------------------------------------------
# The selected settings (Supplementary Table 6) as Mantpy arguments
# ---------------------------------------------------------------------------------------------------

EDGE_WEIGHTS = {"none": "uniform", "inv_d": "inverse", "inv_d2": "inverse_square", "gaussian_median": "gaussian"}
NORMALISATIONS = {"count": "counts", "fraction": "proportions", "log1p_count": "log1p_counts"}


def graph_options(graph):
    """Arguments of ``mt.gr.build_cell_ecm_graphs`` for the ``graph`` part of one selected setting."""
    radius, reach = graph["ee_radius_px"], graph["ce_dmax_px"]
    options = dict(cell_graph="mantpy")
    options.update({
        "knn": dict(cell_edge_method="knn", cell_k=graph["cc_k"], cell_Dmax=graph["cc_radius_px"]),
        "radius": dict(cell_edge_method="radius", cell_radius=graph["cc_radius_px"]),
        "delaunay": dict(cell_edge_method="delaunay"),
    }[graph["cc_rule"]])
    options.update({
        "grid8": dict(ecm_edge_method="grid", ecm_grid_connectivity=8),
        "knn": dict(ecm_edge_method="knn", ecm_k=graph["ee_k"], ecm_Dmax=radius),
        "radius": dict(ecm_edge_method="radius", ecm_r=radius),
        "delaunay": dict(ecm_edge_method="delaunay", ecm_delaunay_max_length=2 * radius),
    }[graph["ee_rule"]])
    options.update({
        "knn": dict(cell_ecm_edge_method="knn", cell_ecm_k=graph["ce_k"], cell_ecm_Dmax=reach),
        "radius": dict(cell_ecm_edge_method="radius", cell_ecm_r=reach or 15.0),
        "delaunay": dict(cell_ecm_edge_method="delaunay", cell_ecm_delaunay_max_length=2 * reach if reach else None),
    }[graph["ce_rule"]])
    rules = (graph["cc_rule"], graph["ee_rule"], graph["ce_rule"])
    if "knn" in rules:
        options.update(knn_backend="scipy")
    if "delaunay" in rules:
        options.update(delaunay_jitter=1e-6, delaunay_seed=0)
    return options


def feature_options(setting, cell_types, n_clusters):
    """Arguments of ``mt.tl.context_features`` for one selected setting."""
    return dict(
        context=setting["context"], hops=setting["features"]["hops"],
        normalization=NORMALISATIONS[setting["features"]["composition_norm"]],
        edge_weight=EDGE_WEIGHTS[setting["graph"]["edge_weight"]],
        cell_categories=list(cell_types), ecm_categories=range(n_clusters),
    )


def build_case_graphs(cells, graph):
    """Build the graph of one selected setting on copies of the ROIs, and return the copies.

    The graphs already stored in ``cells`` stay as they are. The copies do not carry the Squidpy cell
    graph of the reference graph, because Mantpy builds the cell graph of the setting in its place. When
    the setting uses signal patches only as ECM nodes, the copies hold the signal rows of the patch
    table.
    """
    case = {}
    for roi, adata in cells.items():
        copy = adata.copy()
        for key in ("spatial_connectivities", "spatial_distances"):
            copy.obsp.pop(key, None)
        if not graph["ee_include_background_nodes"]:
            table = copy.uns["ecm_patches"]
            copy.uns["ecm_patches"] = table.loc[table["ecm_cluster"].to_numpy(int) >= 0]
        case[roi] = copy
    mt.gr.build_cell_ecm_graphs(case, **graph_options(graph))
    return case


def context_table(adata, index, labels, options):
    """Context features of every patch in ``index``, computed from ``labels`` (one per patch of ``index``).

    Patches that are not nodes of the graph of ``adata`` (background patches, in a graph over the signal
    patches) get zeros. A patch's own label is never used.
    """
    in_graph = index.isin(adata.uns["ecm_patches"].index)
    table = mt.tl.context_features(adata, ecm_labels=np.asarray(labels)[in_graph], **options)
    return table.reindex(index, fill_value=0.0).to_numpy()


def fit_classifier(X, y, learner, *, seed=0):
    """Fit the multinomial logistic regression of one selected setting; returns a function giving probabilities."""
    scaler = StandardScaler().fit(X) if learner["standardize"] else None
    penalty = learner["penalty"]
    if penalty == "none":
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=seed)
    elif penalty == "l2":
        model = LogisticRegression(penalty="l2", C=learner["C"], solver="lbfgs", max_iter=2000, random_state=seed)
    else:   # "l1" or "elasticnet"
        model = LogisticRegression(penalty="elasticnet", C=learner["C"], solver="saga", max_iter=200, tol=1e-3,
                                   l1_ratio=1.0 if penalty == "l1" else learner["l1_ratio"], random_state=seed)
    model.fit(scaler.transform(X) if scaler else X, y)

    def predict(X):
        probabilities = model.predict_proba(scaler.transform(X) if scaler else X)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    return predict


# ---------------------------------------------------------------------------------------------------
# The same calls for every held-out mouse, context, ROI and realisation
# ---------------------------------------------------------------------------------------------------

def _patch_tables(cells):
    tables = {roi: adata.uns["ecm_patches"] for roi, adata in cells.items()}
    labels = {roi: table["ecm_cluster"].to_numpy(int) for roi, table in tables.items()}
    return tables, labels


def evaluate_setting(cells, corruptions, setting, *, held_out, mouse, cell_types, n_clusters=3, seed=0):
    """Fit one selected setting on the other mouse and score every corrupted ROI of the held-out mouse.

    ``corruptions`` maps ``(roi, realisation)`` to the result of ``mt.tl.corrupt_labels``. The steps are
    those shown in the notebook for one case: build the graph of the setting, compute the context
    features, fit the classifier on the uncorrupted ROIs of the other mouse, score the observed labels
    and relabel in the selected number of passes. Returns the scores (one row per ROI and realisation),
    the per-cluster recovery, and the reconstructed labels keyed like ``corruptions``.
    """
    tables, labels = _patch_tables(cells)
    categories = range(n_clusters)
    train = [roi for roi in cells if mouse[roi] != held_out]
    test = [roi for roi in cells if mouse[roi] == held_out]
    case = build_case_graphs(cells, setting["graph"])
    options = feature_options(setting, cell_types, n_clusters)

    def features(roi, current):
        return context_table(case[roi], tables[roi].index, current, options)

    y_train = np.concatenate([labels[roi][labels[roi] >= 0] for roi in train])
    X_train = np.vstack([features(roi, labels[roi])[labels[roi] >= 0] for roi in train])
    predict = fit_classifier(X_train, y_train, setting["learner"], seed=seed)
    prior = np.bincount(y_train, minlength=n_clusters) / len(y_train)       # training label frequencies
    temperature = 1.0
    if setting["calibration"] == "temperature":
        temperature = mt.tl.fit_temperature(y_train, predict(X_train), categories=categories)

    clean = setting["cleaning"]
    rows, class_rows, reconstructed = [], [], {}
    for (roi, realisation), corruption in corruptions.items():
        if roi not in test:
            continue
        observed = np.asarray(corruption.labels)
        first = predict(features(roi, observed))
        scores = {
            "raw": mt.tl.label_anomaly_scores(observed, first, categories=categories, temperature=temperature),
            "prior_adjusted": mt.tl.label_anomaly_scores(observed, first, categories=categories,
                                                         temperature=temperature, prior=prior),
        }
        result = mt.tl.reconstruct_labels_iteratively(
            observed, lambda current, roi=roi: predict(features(roi, current)),
            threshold=clean["threshold"], max_passes=clean["passes"], categories=categories,
            score=clean["anomaly_stat"], min_confidence=clean["min_pred_conf"],
            coordinates=tables[roi][["x", "y"]].to_numpy(float), spatial_radius=clean["gate_radius_px"],
            min_neighbours=clean["gate_min_neighbours"] if clean["gate_radius_px"] > 0 else 0,
            majority_smoothing=clean["majority_smoothing"], temperature=temperature,
        )
        scored = mt.tl.score_label_reconstruction(
            labels[roi], observed, result.labels, corrupted=np.asarray(corruption.is_corrupted),
            scores=scores, categories=categories,
        )
        keys = dict(context=setting["context"], held_out=held_out, roi=roi, realisation=realisation)
        rows.append({**keys, "passes": result.n_passes, "temperature": temperature, **scored.metrics})
        class_rows.append(scored.per_class.reset_index().assign(**keys))
        reconstructed[(roi, realisation)] = np.asarray(result.labels)
    return pd.DataFrame(rows), pd.concat(class_rows, ignore_index=True), reconstructed


def score_label_prior(cells, corruptions, *, held_out, mouse, n_clusters=3):
    """Detection of the replaced labels from label frequency alone (the label-prior baseline).

    Every patch gets the same probabilities, the label frequencies of the other mouse, so the anomaly
    score only says how rare the observed label is.
    """
    _, labels = _patch_tables(cells)
    y_train = np.concatenate([labels[roi][labels[roi] >= 0] for roi in cells if mouse[roi] != held_out])
    counts = np.bincount(y_train, minlength=n_clusters) + 1e-9
    rows = []
    for (roi, realisation), corruption in corruptions.items():
        if mouse[roi] != held_out:
            continue
        observed = np.asarray(corruption.labels)
        flat = np.tile(counts / counts.sum(), (len(observed), 1))
        raw = mt.tl.label_anomaly_scores(observed, flat, categories=range(n_clusters))
        metrics = mt.tl.score_label_reconstruction(labels[roi], observed, observed, scores={"raw": raw},
                                                   categories=range(n_clusters)).metrics
        rows.append(dict(context="prior", held_out=held_out, roi=roi, realisation=realisation,
                         roc_auc_raw=metrics["roc_auc_raw"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Parts of the panels that Mantpy does not draw
# ---------------------------------------------------------------------------------------------------

def draw_contacted_patches(ax, image, patch_xy, cell_xy, *, patch_size, title=None):
    """One channel in grey, the given patches tinted green, and the given cells as red dots (panel g).

    ``image`` is the raw channel; it is median-filtered and scaled to its 99th percentile for display
    only. ``patch_xy`` holds the patch centres and ``cell_xy`` the cell centroids, in pixels.
    """
    image = ndimage.median_filter(np.asarray(image, dtype=float), size=3, mode="nearest")
    scaled = np.clip(image / np.percentile(image[image > 0], 99), 0, 1)
    inside = np.zeros(image.shape, dtype=bool)
    half = patch_size / 2
    for x, y in np.asarray(patch_xy, dtype=float):
        inside[int(y - half):int(y + half), int(x - half):int(x + half)] = True
    rgb = np.repeat(scaled[:, :, None], 3, axis=2)
    rgb[inside] = np.stack([0.15 * scaled[inside], 0.35 + 0.65 * scaled[inside], 0.15 * scaled[inside]], axis=1)
    ax.imshow(rgb)
    cell_xy = np.asarray(cell_xy, dtype=float)
    ax.scatter(cell_xy[:, 0], cell_xy[:, 1], s=4, c="#d62728", linewidths=0)
    ax.set_title(title)
    ax.axis("off")
    return ax


def mark_corruption(ax, corruption, *, isolated=True):
    """Outline the replaced blocks on a patch map and, with ``isolated``, ring the isolated replaced patches."""
    limits = ax.get_xlim(), ax.get_ylim()
    for block in corruption.blocks.itertuples():
        ax.add_patch(Rectangle((block.centre_x - block.side / 2, block.centre_y - block.side / 2),
                               block.side, block.side, fill=False, edgecolor="black", lw=1))
    if isolated:
        alone = np.asarray(corruption.is_corrupted) & ~np.asarray(corruption.in_block())
        xy = np.asarray(corruption.coordinates, dtype=float)[alone]
        ax.scatter(xy[:, 0], xy[:, 1], s=16, facecolors="none", edgecolors="black", linewidths=0.6)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    return ax


def draw_detection(ax, per_roi, groups, *, contexts, names, prior_level, markers, color="C0"):
    """ROC-AUC of every ROI for each context: open markers raw, filled markers prior-adjusted (panel j).

    ``per_roi`` is indexed by context and ROI and has the columns ``roc_auc_raw`` and
    ``roc_auc_prior_adjusted``; ``groups`` maps each ROI to its mouse and ``markers`` each mouse to a
    marker. The bars are the means over the ROIs; the dashed line is the label-prior baseline.
    """
    for i, context in enumerate(contexts):
        data = per_roi.loc[context]
        spread = np.linspace(-0.08, 0.08, len(data))
        for offset, column, filled in ((-0.17, "roc_auc_raw", False), (0.17, "roc_auc_prior_adjusted", True)):
            for j, roi in enumerate(data.index):
                ax.scatter(i + offset + spread[j], data.loc[roi, column], s=26, marker=markers[groups[roi]],
                           facecolors=color if filled else "white", edgecolors=color, zorder=3)
            ax.hlines(data[column].mean(), i + offset - 0.13, i + offset + 0.13, color="black", lw=1.5)
            ax.text(i + offset, data[column].max() + 0.02, f"{data[column].mean():.2f}", ha="center", fontsize=8)
    ax.axhline(prior_level, color="grey", ls="--", lw=0.8)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xticks(range(len(contexts)), [names[context] for context in contexts])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("ROC-AUC, replaced labels")
    return ax

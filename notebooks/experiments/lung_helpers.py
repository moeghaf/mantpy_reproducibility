"""Small data, checkpoint and plotting helpers for the full lung experiment."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import balanced_accuracy_score, f1_score

from lung_analysis.cleaning import apply_temperature
from lung_analysis.data import Cohort, ROIData
from lung_analysis.graphs import GraphRecipe
from lung_analysis.metrics import detection_metrics
from lung_analysis.pipeline import Config, features_for, fit_model, score_roi


HERE = Path(__file__).resolve().parent
METRICS = (
    "roc_auc", "prior_adjusted_roc_auc", "pr_auc", "prior_adjusted_pr_auc",
    "accuracy_before", "accuracy_after", "recovery", "pristine_relabel_rate",
    "correction_precision", "macro_f1_after", "balanced_accuracy_after",
)
CONTEXTS = ("cell", "joint", "ecm", "prior")
COLORS = {"cell": "#D55E00", "joint": "#E69F00", "ecm": "#785EF0", "prior": "#888888"}


def load_selections():
    return json.loads((HERE / "lung_selected.json").read_text(encoding="utf-8"))


def selection_table(selections, scheme="mouse"):
    rows = []
    for held, contexts in selections[scheme].items():
        for context, saved in contexts.items():
            cfg = saved["config"]
            rows.append({
                "held_out": held, "context": context,
                "cell–ECM": cfg["graph"]["ce_rule"],
                "ECM–ECM": cfg["graph"]["ee_rule"],
                "weight": cfg["graph"]["edge_weight"],
                "features": cfg["features"]["composition_norm"],
                "hops": cfg["features"]["hops"],
                "penalty": cfg["learner"]["penalty"],
                "C": cfg["learner"]["C"],
                "threshold": cfg["cleaning"]["threshold"],
                "passes": cfg["cleaning"]["passes"],
                "calibration": cfg["calibration"],
                "original_inner_score": saved["inner_selection_score"],
            })
    return pd.DataFrame(rows)


def retain_primary_patch_support(ecm):
    """Retain the original extractor's strict per-ROI zero-quantile eligibility.

    Archived Mantpy applied feature-row mean > quantile(feature-row means, 0).
    Current Mantpy treats zero quantile as no filtering. Apply the archived rule
    to the freshly normalised mean features before pooled K-means.
    """
    records = []
    for name, adata in ecm.items():
        patches = adata.uns["ecm_patches"]
        columns = [column for column in patches if column.startswith("feat_")]
        row_means = patches[columns].to_numpy(dtype=np.float32).mean(axis=1)
        threshold = np.quantile(row_means, 0.0)
        keep = row_means > threshold
        omitted = patches.loc[~keep, ["x", "y"]].to_dict("records")
        retained = patches.loc[keep].reset_index(drop=True)
        retained.index = retained.index.astype(str)
        adata.uns["ecm_patches"] = retained
        record = {
            "roi": name, "native_patches": len(patches),
            "retained_patches": int(keep.sum()), "excluded_patches": int((~keep).sum()),
            "threshold": float(threshold), "excluded_coordinates": omitted,
            "criterion": "mean standardized patch features > per-ROI minimum",
        }
        adata.uns["primary_patch_eligibility"] = record
        records.append(record)
    return records


def cohort_from_mantpy(bundle, ecm, patch_size=10):
    """Preserve ROI order: evaluation seed = realisation seed * 1000 + ROI index."""
    mouse_by_roi = bundle.metadata.set_index("sample_id")["Mouse"].astype(str)
    cell_types = sorted(set().union(*[
        set(a.obs["cell_type"].astype(str)) for a in bundle.cohort_cells.values()
    ]))
    rois = {}
    for name, cells in bundle.cohort_cells.items():
        patches = ecm[name].uns["ecm_patches"]
        labels = patches["ecm_cluster"].to_numpy(int)
        cell_labels = cells.obs["cell_type"].astype(str).to_numpy()
        feat = [c for c in patches if c.startswith("feat_")]
        x, y = patches["x"].to_numpy(), patches["y"].to_numpy()
        rois[name] = ROIData(
            roi=name, mouse=mouse_by_roi[name], patch_size=patch_size,
            patch_id=np.array([f"{name}:{i}" for i in range(len(patches))]),
            patch_xy=patches[["x", "y"]].to_numpy(float),
            patch_grid=np.column_stack([
                np.floor(y / patch_size).astype(int),
                np.floor(x / patch_size).astype(int),
            ]),
            patch_feat=patches[feat].to_numpy(float), label=labels,
            cell_id=cells.obs_names.to_numpy(),
            cell_xy=np.asarray(cells.obsm["spatial"], dtype=float),
            cell_type=cell_labels,
            cell_type_idx=np.array([cell_types.index(c) for c in cell_labels]),
        )
    return Cohort("primary", rois, cell_types, 3, {
        "partition": "fresh selected K=4 Mantpy fit",
        "roi_order": list(rois),
        "ecm_names": list(bundle.ecm_names),
    })


def source_digest():
    h = hashlib.sha256()
    paths = [HERE / "lung_helpers.py", HERE / "lung_selected.json"]
    paths += sorted((HERE / "lung_analysis").rglob("*.py"))
    for path in paths:
        h.update(path.relative_to(HERE).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def cohort_digest(cohort):
    h = hashlib.sha256()
    h.update(json.dumps(cohort.cell_types).encode())
    for name, roi in cohort.rois.items():
        h.update((name + roi.mouse).encode())
        for a in (roi.patch_xy, roi.patch_grid, roi.patch_feat, roi.label,
                  roi.cell_xy, roi.cell_type_idx):
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def versions():
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "mantpy", "numpy", "pandas", "scipy", "scikit-learn", "anndata",
        )},
    }


def run_identity(cohort):
    return {
        "cohort_sha256": cohort_digest(cohort),
        "analysis_source_sha256": source_digest(),
        "versions": versions(),
    }


def identity_key(identity, extra=None):
    return hashlib.sha256(json.dumps(
        {"identity": identity, "extra": extra}, sort_keys=True,
    ).encode()).hexdigest()[:20]


def load_checkpoint(path, *, identity, config, required=()):
    """Load only a complete, checksummed checkpoint for the requested fit.

    Missing checksums, interrupted writes and mismatched identities are cache
    misses. The same open file is hashed and deserialized, avoiding a replacement
    between verification and loading.
    """
    path = Path(path)
    checksum = path.with_name(path.name + ".sha256")
    try:
        expected = checksum.read_text(encoding="ascii").strip()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected != digest:
                return None
            stream.seek(0)
            result = joblib.load(stream)
        if not isinstance(result, dict):
            return None
        if result.get("identity") != identity or result.get("config") != config:
            return None
        if any(key not in result for key in required):
            return None
        return result
    except Exception:
        return None


def save_checkpoint(result, path):
    """Atomically publish a flushed checkpoint and its SHA256 companion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = path.with_name(path.name + ".sha256")
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            data_path = Path(stream.name)
            temporary.append(data_path)
            joblib.dump(result, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=path.parent,
            prefix=f".{checksum.name}.", suffix=".tmp", delete=False,
        ) as stream:
            checksum_path = Path(stream.name)
            temporary.append(checksum_path)
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(data_path, path)
        os.replace(checksum_path, checksum)
    finally:
        for temporary_path in temporary:
            temporary_path.unlink(missing_ok=True)


def fit_fold(cohort, selections, scheme, held_out, context, cache, checkpoint_dir):
    """Fit or resume exactly one outer-fold model; cached identities include inputs/code."""
    if scheme == "mouse":
        test = cohort.mice()[held_out]
    elif scheme == "roi":
        test = [held_out]
    else:
        raise ValueError(scheme)
    train = [r for r in cohort.roi_names if r not in test]
    assert set(train).isdisjoint(test)
    if scheme == "mouse":
        assert all(cohort.rois[r].mouse != held_out for r in train)
    if context == "prior":
        cfg = Config(task="forward", context="prior",
                     graph=GraphRecipe.from_dict(selections["reference_graph"]))
        study = f"label_prior__{scheme}__{held_out}"
    else:
        saved = selections[scheme][held_out][context]
        cfg, study = Config.from_dict(saved["config"]), saved["study"]
        assert cfg.task == "forward" and not cfg.features.intensity_block
    identity = run_identity(cohort)
    key = identity_key(identity, {"scheme": scheme, "held_out": held_out,
                                 "context": context, "config": cfg.to_dict(),
                                 "train": train, "test": test, "fit_seed": 0})
    folder = Path(checkpoint_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{scheme}_{held_out}_{context}_{key}.joblib"
    started = time.perf_counter()
    result = load_checkpoint(
        path, identity=identity, config=cfg.to_dict(),
        required=("model", "cleaning", "prior", "fit_seconds", "train_rois",
                  "test_rois", "study", "scheme", "held_out", "context",
                  "fit_seed", "selected_penalty", "resolved_penalty"),
    )
    if result is not None:
        result["resumed"] = True
    else:
        model, cleaning = fit_model(cohort, train, cfg, cache, seed=0)
        labels = np.concatenate([cohort.rois[r].label for r in train])
        labels = labels[labels >= 0]
        counts = np.bincount(labels, minlength=cohort.n_labels)
        prior = counts / counts.sum()
        assert np.all(prior > 0)
        result = {
            "model": model, "cleaning": cleaning, "prior": prior,
            "identity": identity, "config": cfg.to_dict(),
            "train_rois": train, "test_rois": test, "study": study,
            "scheme": scheme, "held_out": held_out, "context": context,
            "fit_seed": 0, "resumed": False,
            "selected_penalty": cfg.learner.penalty if context != "prior" else None,
            "resolved_penalty": (
                model._clf.get_params().get("penalty")
                if context != "prior" and cfg.learner.learner == "lr" else None
            ),
        }
        result["fit_seconds"] = time.perf_counter() - started
        save_checkpoint(result, path)
    result["load_or_fit_seconds"] = time.perf_counter() - started
    return result


def score_fold(fitted, cohort, realisations, cache, kinds=("mixed",)):
    """Apply one fresh fit to every held-out ROI and evaluation realisation."""
    cfg = Config.from_dict(fitted["config"])
    model, cleaning = fitted["model"], fitted["cleaning"]
    rows, classes, examples = [], [], {}
    for (kind, seed), real in realisations.items():
        if kind not in kinds:
            continue
        for roi_name in fitted["test_rois"]:
            roi, corruption = cohort.rois[roi_name], real[roi_name]
            out = score_roi(model, cleaning, cohort, roi_name, cfg, cache, corruption)
            # Use first-pass probabilities, before any ECM-context cleaning pass.
            X = features_for(roi, cohort, cfg, cache.get(roi, cfg.graph),
                             labels=corruption.corrupted)
            first = apply_temperature(model.predict_proba(X), cleaning.temperature)
            signal = out["signal"]
            idx = np.flatnonzero(signal)
            adjusted = np.zeros(len(signal))
            adjusted[idx] = out["anomaly"][idx] + np.log(
                fitted["prior"][out["observed"][idx]]
            )
            expected_raw = -np.log(np.maximum(first[idx, out["observed"][idx]], 1e-12))
            np.testing.assert_allclose(out["anomaly"][idx], expected_raw,
                                       atol=1e-12, rtol=1e-12)
            am = detection_metrics(adjusted, out["is_artifact"], signal)
            if fitted["context"] == "prior":
                # The label prior is the adjustment denominator, not a context predictor.
                # Report its raw baseline only; microscopic smoothing residuals have no
                # meaningful prior-adjusted ranking.
                am = {"roc_auc": np.nan, "pr_auc": np.nan}
            base = {
                "scheme": fitted["scheme"], "held_out": fitted["held_out"],
                "context": cfg.context, "roi": roi_name, "mouse": roi.mouse,
                "kind": kind, "seed": seed, "temperature": cleaning.temperature,
            }
            y, pred = out["pristine"][signal], out["cleaned"][signal]
            rows.append({
                **base,
                **{k: out[k] for k in (
                    "roc_auc", "pr_auc", "accuracy_before", "accuracy_after",
                    "recovery", "pristine_relabel_rate", "correction_precision",
                    "n_relabelled", "n_artifact", "n_signal",
                )},
                "prior_adjusted_roc_auc": am["roc_auc"],
                "prior_adjusted_pr_auc": am["pr_auc"],
                "macro_f1_after": f1_score(y, pred, labels=np.arange(cohort.n_labels),
                                          average="macro", zero_division=0),
                "balanced_accuracy_after": balanced_accuracy_score(y, pred),
            })
            for cluster in range(cohort.n_labels):
                m = signal & out["is_artifact"] & (out["pristine"] == cluster)
                classes.append({**base, "cluster": cluster,
                                "n_corrupted": int(m.sum()),
                                "n_restored": int((out["cleaned"][m] == cluster).sum())})
            if kind == "mixed" and seed == 0:
                examples[roi_name] = {k: out[k] for k in (
                    "pristine", "observed", "cleaned", "is_artifact", "anomaly",
                )}
    return pd.DataFrame(rows), pd.DataFrame(classes), examples


def fold_record(fitted):
    """Serializable record of the split and the fitted calibration/penalty."""
    return {k: fitted[k] for k in (
        "study", "scheme", "held_out", "context", "train_rois", "test_rois",
        "fit_seed", "fit_seconds", "load_or_fit_seconds", "resumed",
        "selected_penalty", "resolved_penalty", "identity", "config",
    )} | {"fitted_temperature": fitted["cleaning"].temperature,
         "training_prior": fitted["prior"].tolist()}


def summarise_scores(scores):
    """Average seeds within ROI, then give held-out ROIs equal weight."""
    per_roi = scores.groupby(
        ["scheme", "context", "kind", "roi", "mouse"], sort=False,
    )[list(METRICS)].mean().reset_index()
    summary = per_roi.groupby(
        ["scheme", "context", "kind"], sort=False,
    )[list(METRICS)].agg(["mean", "std", "min", "max"])
    return per_roi, summary


def summarise_class_recovery(classes):
    """Pool corruption/restoration counts over seeds within ROI, then average ROIs."""
    per_roi = classes.groupby(
        ["scheme", "context", "kind", "roi", "cluster"], sort=False,
    )[["n_corrupted", "n_restored"]].sum().reset_index()
    per_roi["recovery"] = per_roi["n_restored"] / per_roi["n_corrupted"].replace(0, np.nan)
    summary = per_roi.groupby(
        ["scheme", "context", "kind", "cluster"], sort=False,
    )["recovery"].agg(["mean", "std", "min", "max"]).reset_index()
    return per_roi, summary


def descriptive_association(cohort, graph_cache, recipe):
    """Observed/expected cell–ECM edge counts, pooled after computing expected per ROI."""
    T, K = len(cohort.cell_types), cohort.n_labels
    observed, expected = np.zeros((T, K)), np.zeros((T, K))
    per_roi = []
    for name, roi in cohort.rois.items():
        edge = graph_cache.get(roi, recipe).ce_d.tocoo()
        cluster = roi.label[edge.col]
        keep = cluster >= 0
        cell, cluster = edge.row[keep], cluster[keep]
        obs = np.bincount(
            roi.cell_type_idx[cell] * K + cluster, minlength=T * K,
        ).reshape(T, K)
        exp = obs.sum(1)[:, None] * obs.sum(0)[None, :] / max(obs.sum(), 1)
        observed += obs
        expected += exp
        for t, cell_type in enumerate(cohort.cell_types):
            for k in range(K):
                per_roi.append({
                    "roi": name, "mouse": roi.mouse, "cell_type": cell_type,
                    "cluster": k, "n_obs": obs[t, k], "n_exp": exp[t, k],
                    "log2_enr": float(np.log2(obs[t, k] / exp[t, k]))
                    if obs[t, k] > 0 and exp[t, k] > 0 else -np.inf,
                })
    rows = []
    for t, cell_type in enumerate(cohort.cell_types):
        for k in range(K):
            rows.append({
                "cell_type": cell_type, "cluster": k,
                "n_obs": int(observed[t, k]), "n_exp": expected[t, k],
                "log2_enr": float(np.log2(observed[t, k] / expected[t, k]))
                if observed[t, k] > 0 and expected[t, k] > 0 else -np.inf,
            })
    return pd.DataFrame(rows), pd.DataFrame(per_roi)


def aec_contacts(cohort, cache, recipe):
    contacts = {}
    for name, roi in cohort.rois.items():
        ce = cache.get(roi, recipe).ce_d.tocoo()
        hit = np.zeros(len(roi.label), bool)
        hit[ce.col[roi.cell_type[ce.row] == "AEC"]] = True
        contacts[name] = hit
    return contacts


def marker_contrasts(cohort, contacts, wall_cluster=1):
    rows = {}
    for name, roi in cohort.rois.items():
        wall = roi.label == wall_cluster
        rows[name] = (
            roi.patch_feat[wall & contacts[name]].mean(axis=0)
            - roi.patch_feat[wall & ~contacts[name]].mean(axis=0)
        )
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=cohort.meta["ecm_names"],
    ).rename_axis("roi")


def graph_sensitivity(cohort, selections, output_dir):
    """Recompute the 48 recorded random recipes plus reference; resume identical inputs."""
    from lung_analysis.association import sensitivity_landscape

    identity = run_identity(cohort)
    space = {"fixed_reference_config": selections["reference_graph"],
             "bo_axes": {"graph": selections["graph_sensitivity_axes"]}}
    key = identity_key(identity, {"space": space, "recipes": 48, "seed": 0})
    path = Path(output_dir) / "checkpoints" / f"graph_sensitivity_{key}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    table = sensitivity_landscape(cohort, space, "AEC", n_recipes=48, seed=0)
    assert table.recipe_index.nunique() == 49
    table.to_csv(path, index=False)
    return table


def plot_map(roi, labels, title, palette):
    xy, ps = roi.patch_xy, roi.patch_size
    x, y = np.floor(xy[:, 0] / ps).astype(int), np.floor(xy[:, 1] / ps).astype(int)
    grid = np.full((y.max() + 1, x.max() + 1), np.nan)
    grid[y, x] = np.where(labels >= 0, labels, np.nan)
    cmap = ListedColormap([palette[k] for k in range(3)])
    cmap.set_bad("white")
    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest",
              extent=(0, grid.shape[1] * ps, grid.shape[0] * ps, 0))
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=title)
    plt.tight_layout()
    plt.show()


def plot_detection(per_roi, *, scheme="mouse", kind="mixed"):
    data = per_roi.query("scheme == @scheme and kind == @kind")
    _, ax = plt.subplots(figsize=(7, 4))
    for i, context in enumerate(CONTEXTS):
        values = data.loc[data.context.eq(context)]
        offset = np.linspace(-0.10, 0.10, len(values))
        for shift, metric, marker, fill in (
            (-0.16, "roc_auc", "o", "none"),
            (0.16, "prior_adjusted_roc_auc", "s", COLORS[context]),
        ):
            scores = values[metric].to_numpy()
            if not np.isfinite(scores).any():
                continue
            ax.scatter(i + shift + offset, scores, marker=marker,
                       edgecolors=COLORS[context], facecolors=fill, s=36)
            ax.hlines(scores.mean(), i + shift - 0.13, i + shift + 0.13,
                      color=COLORS[context], linewidth=2)
    ax.set(xticks=range(4), xticklabels=CONTEXTS,
           ylabel="ROC-AUC · each point is a held-out ROI",
           ylim=(0.0, 1.025), title=f"{scheme}-held-out · {kind} corruption")
    ax.scatter([], [], facecolors="none", edgecolors="black", label="Raw", s=36)
    ax.scatter([], [], marker="s", color="black", label="Prior adjusted", s=36)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def export_partition(cohort, output):
    frames = []
    for name, roi in cohort.rois.items():
        frame = pd.DataFrame({
            "roi": name, "mouse": roi.mouse, "patch_id": roi.patch_id,
            "x": roi.patch_xy[:, 0], "y": roi.patch_xy[:, 1], "label": roi.label,
        })
        for i, marker in enumerate(cohort.meta["ecm_names"]):
            frame[marker] = roi.patch_feat[:, i]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)

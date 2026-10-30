"""Replay training-fold target selections and paired physical corruptions.

The original control fits its ECM foreground and cluster centres within each
outer training fold. The saved signal K replaces the original K search here.
Physical proposals and metrics retain the original control's definitions.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score

import lung_helpers as H
from .benchmark import Corruption
from .data import Cohort, ROIData
from .metrics import detection_metrics
from .pipeline import Config, GraphCache, score_roi

NORMALIZATIONS = ("training_cohort_z", "per_roi_z")


def raw_statistics(bundle, patch_size=10):
    """Fresh native patch means and full-image pixel moments from the public images."""
    panel = bundle.panel.loc[bundle.panel["keep"].eq(1)]
    planes = [i for i, value in enumerate(panel["ecm"]) if int(value) == 1]
    raw = {}
    for name in bundle.roi_names:
        img = np.asarray(bundle.images[name])[planes].astype(np.float64)
        h = img.shape[1] // patch_size * patch_size
        w = img.shape[2] // patch_size * patch_size
        means = img[:, :h, :w].reshape(
            len(planes), h // patch_size, patch_size, w // patch_size, patch_size,
        ).mean((2, 4))
        raw[name] = {
            "means": means.transpose(1, 2, 0).reshape(-1, len(planes)),
            "pixel_sum": img.reshape(len(planes), -1).sum(1),
            "pixel_sumsq": (img * img).reshape(len(planes), -1).sum(1),
            "n_pixels": img.shape[1] * img.shape[2],
            "grid_shape": (h // patch_size, w // patch_size),
        }
        assert raw[name]["means"].shape == (4554, 10)
        assert np.isfinite(raw[name]["means"]).all()
        assert raw[name]["means"].min() >= 0
    return raw


def geometry_from_bundle(bundle, raw, patch_size=10):
    """Read cells and reconstruct every native tile without reading pooled ECM labels."""
    mouse = bundle.metadata.set_index("sample_id")["Mouse"].astype(str)
    cell_types = sorted(set().union(*[
        set(a.obs["cell_type"].astype(str)) for a in bundle.cohort_cells.values()
    ]))
    rois = {}
    for name, cells in bundle.cohort_cells.items():
        indices = np.arange(len(raw[name]["means"]))
        grid = np.column_stack(np.divmod(indices, raw[name]["grid_shape"][1]))
        xy = patch_size * grid[:, ::-1] + patch_size / 2
        ct = cells.obs["cell_type"].astype(str).to_numpy()
        rois[name] = ROIData(
            roi=name, mouse=mouse[name], patch_size=patch_size,
            patch_id=np.array([f"{name}:native_grid_{i}" for i in indices]),
            patch_xy=xy, patch_grid=grid,
            patch_feat=np.zeros_like(raw[name]["means"]),
            label=np.full(len(indices), -1, dtype=int),
            cell_id=cells.obs_names.to_numpy(),
            cell_xy=np.asarray(cells.obsm["spatial"], dtype=float),
            cell_type=ct,
            cell_type_idx=np.array([cell_types.index(c) for c in ct]),
        )
    return Cohort(
        "target_refit_geometry", rois, cell_types, 0,
        {"ecm_names": list(bundle.ecm_names), "full_native_grid": True},
    )


def raw_digest(raw):
    h = hashlib.sha256()
    for name, record in raw.items():
        h.update(name.encode())
        for key in ("means", "pixel_sum", "pixel_sumsq", "n_pixels"):
            array = np.ascontiguousarray(record[key])
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def pixel_moments(raw, names):
    count = sum(int(raw[name]["n_pixels"]) for name in names)
    mean = sum(raw[name]["pixel_sum"] for name in names) / count
    second = sum(raw[name]["pixel_sumsq"] for name in names) / count
    scale = np.sqrt(np.maximum(second - mean * mean, 0))
    scale[scale == 0] = 1.0
    assert np.isfinite(mean).all() and np.all(scale > 0)
    return mean, scale


def refit_targets(geometry, raw, selections, normalization, held_out, checkpoint_dir):
    """Refit selected-K centres on the five training ROIs; resume identical fits."""
    spec = selections["target_refit_control"]
    saved = spec["target_definition"][normalization][held_out]
    train = [name for name in geometry.roi_names if name != held_out]
    assert train == saved["training_rois"] and held_out not in train
    identity = H.run_identity(geometry) | {"raw_sha256": raw_digest(raw)}
    config = {
        "normalization": normalization, "training_rois": train,
        "signal_k": saved["selected_signal_k"], "seed": spec["seed"],
        "kmeans": spec["kmeans"],
        "original_definition_sha256": saved["original_definition_sha256"],
    }
    key = H.identity_key(identity, config)
    folder = Path(checkpoint_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"targets_{normalization}_{held_out}_{key}.joblib"
    started = time.perf_counter()
    fitted = H.load_checkpoint(
        path, identity=identity, config=config, required=("labels", "record"),
    )
    if fitted is not None:
        fitted["resumed"] = True
    else:
        moments = {
            name: pixel_moments(
                raw, train if normalization == "training_cohort_z" else [name],
            ) for name in geometry.roi_names
        }
        for name, (mean, scale) in moments.items():
            # Verifies the raw input/plane order against the original sufficient statistics.
            archived = saved["original_moments"][name]
            np.testing.assert_allclose(mean, archived["mean"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(scale, archived["scale"], rtol=1e-12, atol=1e-12)
        z = {
            name: (raw[name]["means"] - mean) / scale
            for name, (mean, scale) in moments.items()
        }
        training = np.concatenate([
            z[name][raw[name]["means"].sum(axis=1) > 0] for name in train
        ])
        model = KMeans(
            n_clusters=config["signal_k"] + 1, random_state=config["seed"],
            **config["kmeans"],
        ).fit(training)
        background = int(np.argmin(model.cluster_centers_.mean(axis=1)))
        components = sorted(
            [i for i in range(model.n_clusters) if i != background],
            key=lambda i: (float(model.cluster_centers_[i].mean()), i),
        )
        mapping = np.full(model.n_clusters, -1, dtype=int)
        mapping[components] = np.arange(len(components))
        labels = {}
        for name in geometry.roi_names:
            labels[name] = mapping[model.predict(z[name])]
            labels[name][raw[name]["means"].sum(axis=1) == 0] = -1
        centres = model.cluster_centers_
        original = np.asarray(saved["original_centres"])
        distance = np.linalg.norm(centres[:, None] - original[None, :], axis=2)
        a, b = linear_sum_assignment(distance)
        fitted = {
            "identity": identity, "config": config, "labels": labels,
            "resumed": False,
            "record": {
                **config, "held_out_roi": held_out, "background_component": background,
                "component_mapping": mapping.tolist(),
                "cluster_centres": centres.tolist(),
                "moments": {
                    name: {"mean": mean.tolist(), "scale": scale.tolist()}
                    for name, (mean, scale) in moments.items()
                },
                "pixel_moments_match_original": True,
                "maximum_matched_centroid_distance_from_original": float(distance[a, b].max()),
                "globally_pooled_target_labels_read": False,
                "held_out_roi_used_in_target_fit": False,
                "held_out_pixel_moments_used": normalization == "per_roi_z",
                "native_geometry_nodes": {
                    name: len(roi.label) for name, roi in geometry.rois.items()
                },
                "fit_seconds": time.perf_counter() - started,
            },
        }
        H.save_checkpoint(fitted, path)
    rois = {
        name: replace(roi, label=np.array(fitted["labels"][name], copy=True))
        for name, roi in geometry.rois.items()
    }
    cohort = Cohort(
        f"{normalization}__without_{held_out}", rois, geometry.cell_types,
        config["signal_k"],
        geometry.meta | {"target_training_rois": train, "normalization": normalization},
    )
    fitted["record"]["resumed"] = fitted["resumed"]
    fitted["record"]["load_or_fit_seconds"] = time.perf_counter() - started
    # Each target partition has a different foreground mask, hence its own graph cache.
    return cohort, GraphCache(32), fitted["record"]


def physical_proposal(xy, common, kind, seed, roi):
    """Original paired proposal: locations do not depend on replacement-label acceptance."""
    assert kind in {"mixed", "block", "isolated"}
    rng = np.random.default_rng(int(seed) + 100003 * int(roi[-3:]))
    eligible = np.flatnonzero(common)
    budget = int(round(0.1 * len(eligible)))
    block_budget = budget if kind == "block" else budget // 2 if kind == "mixed" else 0
    isolated_budget = budget - block_budget
    used = np.zeros(len(common), bool)
    blocks, count = [], 0
    for _ in range(200):
        if count >= block_budget:
            break
        pool = np.flatnonzero(common & ~used)
        if not len(pool):
            break
        side = float(rng.choice([50.0, 100.0, 150.0]))
        centre = xy[rng.choice(pool)] + rng.uniform(-side / 2, side / 2, size=2)
        inside = pool[np.all(np.abs(xy[pool] - centre) <= side / 2, axis=1)]
        if len(inside) < 20:
            continue
        if len(inside) > block_budget - count:
            inside = rng.choice(inside, size=block_budget - count, replace=False)
        used[inside] = True
        count += len(inside)
        blocks.append({
            "indices": inside, "centre": centre.tolist(), "side": side,
            "uniform": float(rng.random()),
        })
    pool = np.flatnonzero(common & ~used)
    isolated = rng.choice(pool, size=min(isolated_budget, len(pool)), replace=False)
    uniforms = rng.random(len(isolated))
    proposed = used.copy()
    proposed[isolated] = True
    return {
        "blocks": blocks, "isolated": isolated, "uniforms": uniforms,
        "proposed": proposed, "common": common.copy(),
        "requested_nodes": budget, "proposed_nodes": int(proposed.sum()),
        "unplaced_block_nodes": block_budget - count,
    }


def apply_proposal(pristine, n_labels, proposal, kind, seed):
    observed, records = pristine.copy(), []
    for block in proposal["blocks"]:
        replacement = min(n_labels - 1, int(block["uniform"] * n_labels))
        indices = block["indices"]
        observed[indices] = replacement
        records.append({
            "centre": block["centre"], "side": block["side"],
            "new_label": replacement, "n_proposed": len(indices),
            "n_flipped": int(np.sum(pristine[indices] != replacement)),
        })
    indices = proposal["isolated"]
    if len(indices):
        other = np.minimum(n_labels - 2, (proposal["uniforms"] * (n_labels - 1)).astype(int))
        replacement = other + (other >= pristine[indices])
        assert np.all(replacement != pristine[indices])
        observed[indices] = replacement
    changed = (observed != pristine) & (pristine >= 0)
    assert not np.any(changed & ~proposal["common"])
    return Corruption(changed, observed, pristine.copy(), records, seed, kind)


def paired_corruptions(cohorts, held_out, kinds=("mixed", "block", "isolated"), seeds=range(5)):
    """Use identical physical proposals and uniform variates in both normalization arms."""
    first, second = [cohorts[name].rois[held_out] for name in NORMALIZATIONS]
    assert np.array_equal(first.patch_id, second.patch_id)
    common = (first.label >= 0) & (second.label >= 0)
    realisations = {}
    for kind in kinds:
        for seed in seeds:
            proposal = physical_proposal(first.patch_xy, common, kind, seed, held_out)
            paired = {
                name: apply_proposal(
                    cohort.rois[held_out].label, cohort.n_labels, proposal, kind, seed,
                ) for name, cohort in cohorts.items()
            }
            realisations[(kind, seed)] = (common, proposal, paired)
    return realisations


def extra_metrics(result, training_prior, n_labels, common):
    """Native and common-foreground scores, retaining fixed-K macro F1 and class counts."""
    signal = result["signal"]
    rows = np.flatnonzero(signal)
    prior_score = np.zeros(len(signal))
    prior_score[rows] = -np.log(np.maximum(training_prior[result["observed"][rows]], 1e-12))
    adjusted = result["anomaly"] - prior_score
    answer = {
        "n_common_signal": int(common.sum()),
        "realized_corruption_fraction": float(result["is_artifact"].sum() / max(1, signal.sum())),
    }
    for suffix, mask in [("full_signal", signal), ("common_signal", common)]:
        if not mask.any():
            continue
        answer[f"accuracy_before_{suffix}"] = float(np.mean(
            result["observed"][mask] == result["pristine"][mask]))
        answer[f"accuracy_after_{suffix}"] = float(np.mean(
            result["cleaned"][mask] == result["pristine"][mask]))
        for when, key in (("before", "observed"), ("after", "cleaned")):
            answer[f"macro_f1_{when}_{suffix}"] = float(f1_score(
                result["pristine"][mask], result[key][mask],
                labels=np.arange(n_labels), average="macro", zero_division=0,
            ))
        answer[f"balanced_accuracy_after_{suffix}"] = float(balanced_accuracy_score(
            result["pristine"][mask], result["cleaned"][mask],
        ))
        for name, score in (("prior_adjusted", adjusted), ("label_prior", prior_score)):
            measured = detection_metrics(score, result["is_artifact"], mask)
            answer[f"{name}_roc_auc_{suffix}"] = measured["roc_auc"]
            answer[f"{name}_pr_auc_{suffix}"] = measured["pr_auc"]
        answer[f"context_roc_auc_{suffix}"] = detection_metrics(
            result["anomaly"], result["is_artifact"], mask)["roc_auc"]
    return answer, adjusted, prior_score


def score_target_fold(fitted, cohort, realisations, cache, normalization):
    cfg = Config.from_dict(fitted["config"])
    held_out = fitted["held_out"]
    rows, classes, example = [], [], None
    for (kind, seed), (common, proposal, paired) in realisations.items():
        result = score_roi(
            fitted["model"], fitted["cleaning"], cohort, held_out,
            cfg, cache, paired[normalization],
        )
        extra, adjusted, prior_score = extra_metrics(
            result, fitted["prior"], cohort.n_labels, common,
        )
        row = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        row.update(
            extra, normalization=normalization, context=cfg.context,
            outer_roi=held_out, kind=kind, seed=seed, signal_k=cohort.n_labels,
            target_coverage=float(result["signal"].mean()),
            proposal_nodes=proposal["proposed_nodes"],
            requested_proposal_nodes=proposal["requested_nodes"],
            unplaced_block_nodes=proposal["unplaced_block_nodes"],
        )
        rows.append(row)
        for cluster in range(cohort.n_labels):
            corrupted = result["is_artifact"] & (result["pristine"] == cluster)
            classes.append({
                "normalization": normalization, "context": cfg.context,
                "outer_roi": held_out, "kind": kind, "seed": seed,
                "cluster": cluster, "n_corrupted": int(corrupted.sum()),
                "n_restored": int((result["cleaned"][corrupted] == cluster).sum()),
            })
        if kind == "mixed" and seed == 0:
            example = {
                **{key: result[key] for key in (
                    "pristine", "observed", "cleaned", "is_artifact", "signal", "anomaly",
                )},
                "common_signal": common, "physical_proposed": proposal["proposed"],
                "prior_adjusted_anomaly": adjusted, "label_prior_anomaly": prior_score,
                "patch_ids": cohort.rois[held_out].patch_id,
            }
    return pd.DataFrame(rows), pd.DataFrame(classes), example


def summarise_target_scores(scores):
    """Average seeds first, then average the six held-out ROIs equally."""
    metrics = [
        "roc_auc", "recovery", "accuracy_after", "signal_k", "target_coverage",
        "realized_corruption_fraction", "prior_adjusted_roc_auc_full_signal",
        "context_roc_auc_common_signal", "prior_adjusted_roc_auc_common_signal",
        "macro_f1_after_common_signal",
    ]
    per_roi = scores.groupby(
        ["normalization", "context", "outer_roi", "kind"], sort=False,
    )[metrics].mean().reset_index()
    summary = per_roi.groupby(
        ["normalization", "context", "kind"], sort=False,
    )[metrics].mean()
    return per_roi, summary

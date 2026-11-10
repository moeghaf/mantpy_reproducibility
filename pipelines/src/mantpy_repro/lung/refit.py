"""Control: refit the ECM foreground and clusters within every training set (single ROIs held out).

In the main analysis the ECM partition is fitted once on all six ROIs. This control repeats
the single-ROI hold-out with the partition fitted on the training ROIs only, on every native
10 x 10-pixel tile, for two ways of scaling the markers:

``training_cohort_z``  mean and s.d. of the raw pixels pooled over the training ROIs;
``per_roi_z``          each image scaled by its own pixel mean and s.d.

A target definition is fitted on a training set: K-means for two to six signal clusters plus
background (three seeds), keeping the smallest K within 1e-4 of the best mean silhouette of
the signal tiles (computed on a seeded sample of at most 5,000 tiles), with every signal
cluster holding at least 1% of the signal tiles in every seed. The seed-0 centres then label
all six ROIs. In the nested search every inner fold uses the definition fitted without the
outer and the inner held-out ROI.

Both scaling arms are corrupted at the same positions on their common foreground (see
``corruption.physical_proposal``).
"""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np

from ..runtime import calculation_threads
from .config import (CONTEXTS, KINDS, NORMALIZATIONS, PRIMARY_KIND, TUNING_SEEDS, Config, inner_folds, outer_folds,
                     sampler_seed, startup_trials)
from .core import FIT_THREADS, Context, environment, log, pack
from .corruption import apply_proposal, physical_proposal
from .data import Cohort
from .denoise import detection_metrics, fit_model, score_roi, training_prior
from .graphs import GraphCache
from .search import optimise, sample_config

SIGNAL_K_RANGE = (2, 3, 4, 5, 6)
K_SELECTION_SEEDS = (0, 1, 2)
KMEANS = {"n_init": 10, "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd"}
KMEANS_THREADS = 4               # the control's K-means ran with four threads
MIN_CLUSTER_FRACTION = 0.01
SILHOUETTE_TIE = 1e-4


def train_key(train: list[str]) -> str:
    return "train_" + "_".join(name[-3:] for name in sorted(train))


def pixel_moments(raw: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and s.d. of every marker over all pixels of the named ROIs."""
    count = sum(int(raw[name]["n_pixels"]) for name in names)
    mean = sum(raw[name]["pixel_sum"] for name in names) / count
    second = sum(raw[name]["pixel_sumsq"] for name in names) / count
    scale = np.sqrt(np.maximum(second - mean * mean, 0))
    scale[scale == 0] = 1.0
    return mean, scale


def define_targets(raw: dict, train: list[str], normalization: str, signal_k: int | None = None):
    """Fit foreground and clusters on the training ROIs and label the tiles of every ROI.

    ``signal_k=None`` selects the number of signal clusters as described in the module
    docstring; a number fits that K directly (the seed-0 fit of the selection, used to replay
    the paper's choice). Clusters are ordered by ascending mean centre; the lowest-mean
    component and tiles with no raw signal are background (-1).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, not {normalization!r}")
    train = sorted(train)
    moments = {name: pixel_moments(raw, train if normalization == "training_cohort_z" else [name]) for name in raw}
    z = {name: (raw[name]["means"] - moments[name][0]) / moments[name][1] for name in raw}
    training = np.concatenate([z[name][raw[name]["means"].sum(axis=1) > 0] for name in train])

    def fit(k: int, seed: int):
        model = KMeans(n_clusters=k + 1, random_state=seed, **KMEANS).fit(training)
        return model, int(np.argmin(model.cluster_centers_.mean(axis=1)))

    candidates = []
    if signal_k is None:
        fits = {}
        for k in SIGNAL_K_RANGE:
            scores, eligible = [], True
            for seed in K_SELECTION_SEEDS:
                model, background = fit(k, seed)
                signal = model.labels_ != background
                counts = np.bincount(model.labels_[signal], minlength=k + 1)
                fractions = counts[np.arange(k + 1) != background] / max(1, signal.sum())
                ok = (len(np.unique(model.labels_[signal])) == k and fractions.min() >= MIN_CLUSTER_FRACTION
                      and signal.sum() > k)
                if ok:
                    score = float(silhouette_score(training[signal], model.labels_[signal], metric="euclidean",
                                                   sample_size=min(5000, int(signal.sum())), random_state=0))
                    ok = bool(np.isfinite(score))
                    scores.append(score)
                eligible = eligible and ok
                if seed == 0:
                    fits[k] = (model, background)
            candidates.append({"signal_k": k, "eligible": eligible,
                               "mean_silhouette": float(np.mean(scores)) if eligible else None})
        passed = [row for row in candidates if row["eligible"]]
        if not passed:
            raise RuntimeError("No number of clusters passed the cluster-size rule on this training set.")
        best = max(row["mean_silhouette"] for row in passed)
        signal_k = min(row["signal_k"] for row in passed if row["mean_silhouette"] >= best - SILHOUETTE_TIE)
        model, background = fits[signal_k]
    else:
        model, background = fit(signal_k, 0)

    components = sorted((c for c in range(model.n_clusters) if c != background),
                        key=lambda c: (float(model.cluster_centers_[c].mean()), c))
    mapping = np.full(model.n_clusters, -1, dtype=int)
    mapping[components] = np.arange(len(components))
    labels = {}
    for name in raw:
        labels[name] = mapping[model.predict(z[name])]
        labels[name][raw[name]["means"].sum(axis=1) == 0] = -1
    record = {"training_rois": train, "normalization": normalization, "signal_k": int(signal_k),
              "candidates": candidates, "centres": model.cluster_centers_.tolist(),
              "moments": {name: {"mean": moments[name][0].tolist(), "scale": moments[name][1].tolist()}
                          for name in raw}}
    return labels, record


def _agreement_with_paper(record: dict, saved: dict) -> dict:
    """How far the refitted scaling and centres are from the paper's record of the same fit."""
    from scipy.optimize import linear_sum_assignment

    mismatch = ValueError("Raw marker statistics differ from the paper's record: check the dataset version.")
    if any(np.shape(record["moments"][name][key]) != np.shape(saved["moments"][name][key])
           for name in saved["moments"] for key in ("mean", "scale")):
        raise mismatch
    moments = max(float(np.max(np.abs(np.asarray(record["moments"][name][key]) - np.asarray(saved["moments"][name][key]))
                               / np.maximum(np.abs(np.asarray(saved["moments"][name][key])), 1e-12)))
                  for name in saved["moments"] for key in ("mean", "scale"))
    if moments > 1e-6:
        raise mismatch
    distance = np.linalg.norm(np.asarray(record["centres"])[:, None] - np.asarray(saved["centres"])[None, :], axis=2)
    rows, columns = linear_sum_assignment(distance)
    return {"max_relative_moment_difference": moments, "max_centre_distance_from_paper": float(distance[rows, columns].max())}


class Targets:
    """Target definitions of one run, their cohorts and the paired corruptions built on them.

    With ``selection="paper"`` the number of clusters of each outer training set is the
    paper's choice; with ``"search"`` it is selected on the training set.
    """

    def __init__(self, ctx: Context, selection: str, paper: dict | None = None):
        self.ctx, self.selection, self.paper = ctx, selection, paper
        self.geometry = ctx.prepared().geometry()
        self.raw = ctx.prepared().raw()
        self._bundles: dict[tuple, tuple] = {}
        self._realisations: dict[tuple, tuple] = {}

    def get(self, normalization: str, train: list[str]) -> tuple[Cohort, GraphCache, dict]:
        """Cohort labelled by the definition fitted on ``train``, its graph cache and the fit record."""
        key = (normalization, tuple(sorted(train)))
        if key not in self._bundles:
            saved, signal_k = None, None
            if self.selection == "paper":
                held_out = [name for name in self.geometry.roi_names if name not in train]
                saved = self.paper["target_definition"][normalization][held_out[0]] if len(held_out) == 1 else None
                if saved is None or sorted(saved["training_rois"]) != sorted(train):
                    raise KeyError(f"The paper's selection has no target definition for {train_key(train)}.")
                signal_k = int(saved["selected_signal_k"])
            request = {"normalization": normalization, "train": sorted(train), "kmeans": KMEANS, "seed": 0,
                       "signal_k": signal_k if signal_k is not None else "selected",
                       "signal_k_range": list(SIGNAL_K_RANGE), "selection_seeds": list(K_SELECTION_SEEDS),
                       "threads": KMEANS_THREADS}
            unit = self.ctx.store.unit("refit", "targets", normalization, train_key(train),
                                       "selected_k" if signal_k is None else f"k{signal_k}", request=request)
            names = self.geometry.roi_names

            def fit():
                with calculation_threads(KMEANS_THREADS):
                    labels, record = define_targets(self.raw, train, normalization, signal_k)
                if saved is not None:
                    record["agreement_with_paper"] = _agreement_with_paper(record, saved)
                arrays = {"label": np.concatenate([labels[name] for name in names]),
                          "label_roi": np.concatenate([np.full(len(labels[name]), i) for i, name in enumerate(names)]),
                          "centres": np.asarray(record.pop("centres"))}
                return arrays, {"record": record, "environment": environment()}

            arrays = self.ctx.store.compute(unit, fit, label=f"targets {normalization} {train_key(train)}", log=log)
            record = unit.metadata()["metadata"]["record"]
            rois = {name: replace(roi, label=arrays["label"][arrays["label_roi"] == i].astype(int))
                    for i, (name, roi) in enumerate(self.geometry.rois.items())}
            cohort = Cohort(f"{normalization}__{train_key(train)}", rois, self.geometry.cell_types,
                            int(record["signal_k"]), self.geometry.ecm_names)
            # The background differs between definitions, so graphs are never shared between them.
            self._bundles[key] = (cohort, GraphCache(32), record)
        return self._bundles[key]

    def corruptions(self, train: list[str], roi: str, kind: str, seed: int):
        """Common foreground, proposal and the corruption of ``roi`` in each scaling arm."""
        key = (tuple(sorted(train)), roi, kind, seed)
        if key not in self._realisations:
            cohorts = [self.get(normalization, train)[0] for normalization in NORMALIZATIONS]
            common = np.logical_and.reduce([cohort.rois[roi].label >= 0 for cohort in cohorts])
            proposal = physical_proposal(cohorts[0].rois[roi].patch_xy, common, kind, seed, roi)
            paired = {normalization: apply_proposal(cohort.rois[roi].label, cohort.n_labels, proposal, kind, seed)
                      for normalization, cohort in zip(NORMALIZATIONS, cohorts)}
            self._realisations[key] = (common, proposal, paired)
        return self._realisations[key]


# Tasks
def refit_tasks(mice: dict[str, list[str]], roi_folds: int | None = None) -> list[dict]:
    """Every study (or final fit) of the control: scaling arm x held-out ROI x context."""
    folds = outer_folds(mice, "roi")
    if roi_folds is not None:
        folds = folds[:roi_folds]
    return [{"experiment": "refit", "normalization": normalization, "fold": fold["name"], "context": context}
            for normalization in NORMALIZATIONS for fold in folds for context in CONTEXTS]


# Search
def study_unit(ctx: Context, normalization: str, fold_name: str, context: str):
    folds = outer_folds(ctx.prepared().geometry().mice(), "roi")
    index = [fold["name"] for fold in folds].index(fold_name)
    n_trials = ctx.budget.trials["refit"]
    request = {"experiment": "refit", "normalization": normalization, "held_out": fold_name,
               "train": folds[index]["train"], "context": context, "n_trials": n_trials,
               "sampler": "TPE multivariate group", "sampler_seed": sampler_seed("refit", index, context),
               "n_startup_trials": startup_trials("refit", n_trials), "tuning_seeds": list(TUNING_SEEDS),
               "corruption": PRIMARY_KIND, "targets": "fitted within every training set"}
    return ctx.store.unit("refit", "search", normalization, fold_name, context, request=request), folds[index]


def run_study(ctx: Context, task: dict, targets: Targets) -> dict[str, np.ndarray]:
    """Nested search of one scaling arm, held-out ROI and context (150 trials)."""
    normalization, context = task["normalization"], task["context"]
    unit, fold = study_unit(ctx, normalization, task["fold"], context)

    def fit():
        train = fold["train"]
        outer_cohort, outer_cache, _ = targets.get(normalization, train)

        def objective(trial):
            config = sample_config(trial, context)
            scores = []
            for inner in inner_folds(train):
                cohort, cache, _ = targets.get(normalization, inner["train"])
                model, cleaning = fit_model(cohort, inner["train"], config, cache, seed=0)
                held_out = inner["test"][0]
                for seed in TUNING_SEEDS:
                    paired = targets.corruptions(inner["train"], held_out, PRIMARY_KIND, seed)[2]
                    result = score_roi(model, cleaning, cohort, held_out, config, cache, paired[normalization])
                    scores.append(result["accuracy_after"])
            trial.set_user_attr("config", json.dumps(config.to_dict()))
            trial.set_user_attr("edges", sum(outer_cache.get(outer_cohort.rois[train[0]], config.graph).n_edges().values()))
            return float(np.mean(scores))

        with calculation_threads(FIT_THREADS):
            trials = optimise(objective, n_trials=unit.request["n_trials"], seed=unit.request["sampler_seed"],
                              n_startup_trials=unit.request["n_startup_trials"],
                              catch=(ValueError, np.linalg.LinAlgError, FloatingPointError))
        return trials, {"environment": environment()}

    return ctx.store.compute(unit, fit, label=f"study refit {normalization} {task['fold']} {context}", log=log)


# Final fits
def extra_metrics(result: dict, prior: np.ndarray, n_labels: int, common: np.ndarray) -> dict:
    """Scores on the arm's own foreground and on the common foreground of both arms.

    The ``label_prior`` score is -log of the training frequency of the observed label; the
    prior-adjusted score is the anomaly score minus it.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    signal = result["signal"]
    rows = np.flatnonzero(signal)
    prior_score = np.zeros(len(signal))
    prior_score[rows] = -np.log(np.maximum(prior[result["observed"][rows]], 1e-12))
    adjusted = result["anomaly"] - prior_score
    out = {"n_common_signal": int(common.sum()),
           "realized_corruption_fraction": float(result["is_artifact"].sum() / max(1, signal.sum()))}
    for suffix, mask in (("full_signal", signal), ("common_signal", common)):
        if not mask.any():
            continue
        pristine = result["pristine"][mask]
        out[f"accuracy_before_{suffix}"] = float(np.mean(result["observed"][mask] == pristine))
        out[f"accuracy_after_{suffix}"] = float(np.mean(result["cleaned"][mask] == pristine))
        for when, key in (("before", "observed"), ("after", "cleaned")):
            out[f"macro_f1_{when}_{suffix}"] = float(f1_score(pristine, result[key][mask], labels=np.arange(n_labels),
                                                              average="macro", zero_division=0))
        out[f"balanced_accuracy_after_{suffix}"] = float(balanced_accuracy_score(pristine, result["cleaned"][mask]))
        for name, score in (("prior_adjusted", adjusted), ("label_prior", prior_score)):
            measured = detection_metrics(score, result["is_artifact"], mask)
            out[f"{name}_roc_auc_{suffix}"] = measured["roc_auc"]
            out[f"{name}_pr_auc_{suffix}"] = measured["pr_auc"]
        out[f"context_roc_auc_{suffix}"] = detection_metrics(result["anomaly"], result["is_artifact"], mask)["roc_auc"]
    return out


def run_final(ctx: Context, task: dict, config: Config, selection: str, targets: Targets) -> dict[str, np.ndarray]:
    """Refit the selected configuration on the five training ROIs and score the held-out ROI
    on every corruption type and evaluation realisation."""
    import pandas as pd

    normalization, held_out, context = task["normalization"], task["fold"], task["context"]
    train = [name for name in targets.geometry.roi_names if name != held_out]
    seeds = ctx.budget.evaluation_seeds
    request = {"experiment": "refit", "selection": selection, "normalization": normalization, "held_out": held_out,
               "train": train, "context": context, "config": config.to_dict(), "kinds": list(KINDS),
               "evaluation_seeds": list(seeds), "fit_seed": 0}
    unit = ctx.store.unit("refit", "final", selection, normalization, held_out, context, request=request)

    def fit():
        cohort, cache, record = targets.get(normalization, train)
        with calculation_threads(FIT_THREADS):
            model, cleaning = fit_model(cohort, train, config, cache, seed=0)
            prior = training_prior(cohort, train)
            rows, classes = [], []
            for kind in KINDS:
                for seed in seeds:
                    common, proposal, paired = targets.corruptions(train, held_out, kind, seed)
                    result = score_roi(model, cleaning, cohort, held_out, config, cache, paired[normalization])
                    base = {"normalization": normalization, "context": context, "outer_roi": held_out, "kind": kind,
                            "seed": seed}
                    rows.append({**base, **{k: v for k, v in result.items() if not isinstance(v, (np.ndarray, str))},
                                 **extra_metrics(result, prior, cohort.n_labels, common),
                                 "signal_k": cohort.n_labels, "target_coverage": float(result["signal"].mean()),
                                 "proposal_nodes": proposal["proposed_nodes"],
                                 "requested_proposal_nodes": proposal["requested_nodes"],
                                 "unplaced_block_nodes": proposal["unplaced_block_nodes"]})
                    for cluster in range(cohort.n_labels):
                        replaced = result["is_artifact"] & (result["pristine"] == cluster)
                        classes.append({**base, "cluster": cluster, "n_corrupted": int(replaced.sum()),
                                        "n_restored": int((result["cleaned"][replaced] == cluster).sum())})
        arrays = {**pack(pd.DataFrame(rows), "scores__"), **pack(pd.DataFrame(classes), "classes__")}
        metadata = {"temperature": cleaning.temperature, "training_prior": prior.tolist(),
                    "signal_k": cohort.n_labels, "agreement_with_paper": record.get("agreement_with_paper"),
                    "environment": environment()}
        return arrays, metadata

    return ctx.store.compute(unit, fit, label=f"final refit {normalization} {held_out} {context}", log=log)

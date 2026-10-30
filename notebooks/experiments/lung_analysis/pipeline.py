"""Fit / score logic shared by tuning and evaluation."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

from .benchmark import Corruption
from .cleaning import CleaningRecipe, clean_once, anomaly_scores, apply_temperature, fit_temperature
from .data import Cohort, ROIData
from .features import FeatureRecipe, ecm_target_features, cell_target_features
from .graphs import GraphRecipe, Graphs, build_graphs
from .learners.tabular import LearnerRecipe, TabularModel
from .metrics import detection_metrics, reconstruction_metrics


@dataclass(frozen=True)
class Config:
    task: str                      # forward (ECM targets) | reverse (cell targets)
    context: str                   # cell | ecm | joint | prior
    graph: GraphRecipe = field(default_factory=GraphRecipe)
    features: FeatureRecipe = field(default_factory=FeatureRecipe)
    learner: LearnerRecipe = field(default_factory=LearnerRecipe)
    cleaning: CleaningRecipe = field(default_factory=CleaningRecipe)
    calibration: str = "none"      # none | temperature

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "context": self.context, "graph": self.graph.to_dict(), "features": self.features.to_dict(),
                "learner": self.learner.to_dict(), "cleaning": self.cleaning.to_dict(), "calibration": self.calibration}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return cls(task=d["task"], context=d["context"], graph=GraphRecipe.from_dict(d["graph"]), features=FeatureRecipe(**d["features"]),
                   learner=LearnerRecipe(**d["learner"]), cleaning=CleaningRecipe(**d["cleaning"]), calibration=d.get("calibration", "none"))


class GNNModel:
    """Adapter giving the GraphSAGE fit the per-ROI prediction interface used by score_roi."""
    is_gnn = True

    def __init__(self, fit: dict[str, Any]):
        self.fit = fit

    def predict_for_roi(self, cohort: Cohort, roi: ROIData, cache: "GraphCache", labels=None, cell_types=None) -> np.ndarray:
        from .learners.gnn import predict_gnn
        return predict_gnn(self.fit, cohort, roi, cache, labels=labels, cell_types=cell_types)


class GraphCache:
    """Per-process LRU of graphs keyed by (roi, recipe digest); graphs never depend on labels."""
    def __init__(self, maxsize: int = 64):
        self._d: OrderedDict[tuple[str, str], Graphs] = OrderedDict(); self.maxsize = maxsize

    def get(self, roi: ROIData, recipe: GraphRecipe) -> Graphs:
        key = (roi.roi, recipe.digest())
        g = self._d.get(key)
        if g is None:
            g = build_graphs(roi.cell_xy, roi.patch_xy, roi.patch_grid, roi.background, roi.patch_size, recipe)
            self._d[key] = g
            if len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(key)
        return g


def target_arrays(roi: ROIData, cfg: Config, labels: np.ndarray | None = None, cell_types: np.ndarray | None = None):
    """Return (observed labels of the targets, signal mask, xy of targets, n_classes) for the task."""
    if cfg.task == "forward":
        obs = roi.label if labels is None else labels
        return obs, obs >= 0, roi.patch_xy
    obs = roi.cell_type_idx if cell_types is None else cell_types
    return obs, obs >= 0, roi.cell_xy


def features_for(roi: ROIData, cohort: Cohort, cfg: Config, g: Graphs, labels: np.ndarray | None = None,
                 cell_types: np.ndarray | None = None) -> np.ndarray:
    lab = roi.label if labels is None else labels
    ct = roi.cell_type_idx if cell_types is None else cell_types
    if cfg.context == "prior":
        n = roi.label.size if cfg.task == "forward" else roi.cell_type_idx.size
        return np.zeros((n, 1))
    if cfg.task == "forward":
        return ecm_target_features(g, ct, len(cohort.cell_types), lab, cohort.n_labels, roi.patch_feat, cfg.context, cfg.features)
    return cell_target_features(g, ct, len(cohort.cell_types), lab, cohort.n_labels, roi.patch_feat, cfg.context, cfg.features)


def n_classes(cohort: Cohort, cfg: Config) -> int:
    return cohort.n_labels if cfg.task == "forward" else len(cohort.cell_types)


def fit_model(cohort: Cohort, train_rois: list[str], cfg: Config, cache: GraphCache, seed: int = 0) -> tuple[TabularModel, CleaningRecipe]:
    """Fit on pristine training ROIs. Returns the model and the cleaning recipe with its temperature set."""
    Xs, ys = [], []
    for name in train_rois:
        roi = cohort.rois[name]
        g = cache.get(roi, cfg.graph)
        X = features_for(roi, cohort, cfg, g)
        obs, signal, _ = target_arrays(roi, cfg)
        Xs.append(X[signal]); ys.append(obs[signal])
    X = np.vstack(Xs); y = np.concatenate(ys)
    rec = cfg.learner if cfg.context != "prior" else LearnerRecipe(learner="prior")
    T = 1.0
    if rec.learner == "sage":
        from .learners.gnn import GNNRecipe, fit_gnn
        fit = fit_gnn(cohort, train_rois, cfg.graph, GNNRecipe(**rec.gnn), cfg.task, cfg.context, cache, seed=seed)
        model = GNNModel(fit)
    else:
        model = TabularModel(rec, n_classes(cohort, cfg), seed=seed).fit(X, y)
        if cfg.calibration == "temperature" and cfg.context != "prior":
            T = fit_temperature(model.predict_proba(X), y)
    cleaning = CleaningRecipe(**{**cfg.cleaning.to_dict(), "temperature": T})
    return model, cleaning


def score_roi(model: TabularModel, cleaning: CleaningRecipe, cohort: Cohort, roi_name: str, cfg: Config, cache: GraphCache,
              corruption: Corruption | None) -> dict[str, Any]:
    """Score one ROI under an (optional) corruption. Cleaning passes recompute features when the
    observed labels feed the features (ecm/joint contexts)."""
    roi = cohort.rois[roi_name]
    g = cache.get(roi, cfg.graph)
    pristine, signal, xy = target_arrays(roi, cfg)
    observed = pristine.copy() if corruption is None else corruption.corrupted.copy()
    is_art = np.zeros(pristine.size, bool) if corruption is None else corruption.is_artifact
    current = observed.copy()
    anomaly_first = None; relabelled_any = np.zeros(pristine.size, bool)
    for p in range(max(1, cleaning.passes)):
        if getattr(model, "is_gnn", False):
            proba = model.predict_for_roi(cohort, roi, cache, labels=current if cfg.task == "forward" else None,
                                          cell_types=current if cfg.task == "reverse" else None)
        else:
            if cfg.task == "forward":
                X = features_for(roi, cohort, cfg, g, labels=current)
            else:
                X = features_for(roi, cohort, cfg, g, cell_types=current)
            proba = model.predict_proba(X)
        if cfg.context == "prior":
            # detection only: the prior never relabels (argmax is always the majority class)
            anom = anomaly_scores(apply_temperature(proba, cleaning.temperature), observed, signal, "raw_neglogp")
            anomaly_first = anom; new = current; cand = np.zeros_like(signal)
        else:
            new, anom, cand = clean_once(proba, current, signal, xy, cleaning)
        if anomaly_first is None:
            anomaly_first = anomaly_scores(apply_temperature(proba, cleaning.temperature), observed, signal, "raw_neglogp")
        relabelled_any |= cand
        if not cand.any():
            current = new
            break
        current = new
    det = detection_metrics(anomaly_first, is_art, signal)
    rec = reconstruction_metrics(pristine, observed, current, is_art, signal)
    return {"roi": roi_name, **det, **rec, "anomaly": anomaly_first, "cleaned": current, "observed": observed, "pristine": pristine,
            "is_artifact": is_art, "signal": signal, "proba": proba}

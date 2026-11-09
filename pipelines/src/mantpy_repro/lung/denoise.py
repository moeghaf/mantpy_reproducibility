"""Predict each patch's ECM cluster from its graph context, detect replaced labels and clean them.

Features describe the neighbours of a patch and never its own label: the cell-type
composition of its connected cells (cell context), the cluster composition of its
neighbouring patches (ECM context) or both (joint). A multinomial logistic regression is
fitted on the pristine signal patches of the training ROIs. On a corrupted ROI the anomaly
score of a patch is -log P(observed label | context); a patch is relabelled to the predicted
cluster when its cleaning statistic passes the threshold, the prediction differs from the
observed label and its probability reaches the minimum confidence.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np

from .config import PRIOR, CleaningRecipe, Config, FeatureRecipe, LearnerRecipe
from .corruption import Corruption
from .graphs import GraphCache, Graphs

_EPS = 1e-12
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


# --------------------------------------------------------------------------------------- features
def _composition(adjacency, labels, n_categories, targets, sources, recipe: FeatureRecipe, via=None):
    import mantpy as mt

    normalization = {"count": "counts", "fraction": "proportions", "log1p_count": "log1p_counts"}[recipe.composition_norm]
    return mt.tl.neighbourhood_features(
        adjacency, labels, categories=range(n_categories), target_ids=targets, source_ids=sources,
        normalization=normalization, hops=2 if via is not None else 1, via=via, exclude_self=True,
    ).to_numpy()


def context_features(graphs: Graphs, cell_type_idx, n_types: int, labels, n_labels: int, context: str,
                     recipe: FeatureRecipe) -> np.ndarray:
    """Context features of every patch of one ROI (one row per patch).

    Cell context: weighted cell-type composition of the cells joined to the patch and, with
    two hops, of the cells joined to those cells. ECM context: cluster composition of the
    neighbouring patches and, with two hops, of patches two ECM-ECM steps away and of patches
    reached through a shared cell. Paths that return to the patch itself are removed, and
    background patches (label -1) add to no cluster.
    """
    cells = np.asarray([f"cell:{i}" for i in range(len(cell_type_idx))])
    patches = np.asarray([f"ecm:{i}" for i in range(len(labels))])
    blocks = []
    if context in ("cell", "joint"):
        blocks.append(_composition(graphs.ce_w.T, cell_type_idx, n_types, patches, cells, recipe))
        if recipe.hops == 2:
            blocks.append(_composition(graphs.ce_w.T, cell_type_idx, n_types, patches, cells, recipe, graphs.cc_w))
    if context in ("ecm", "joint"):
        blocks.append(_composition(graphs.ee_w, labels, n_labels, patches, patches, recipe))
        if recipe.hops == 2:
            blocks.append(_composition(graphs.ee_w, labels, n_labels, patches, patches, recipe, graphs.ee_w))
            blocks.append(_composition(graphs.ce_w.T, labels, n_labels, patches, patches, recipe, graphs.ce_w))
    if not blocks:
        raise ValueError(f"no features for context {context!r}")
    return np.hstack(blocks)


def features_for(roi, cohort, config: Config, graphs: Graphs, labels: np.ndarray | None = None) -> np.ndarray:
    """Features of one ROI under ``config``; ``labels`` replaces the ROI's labels (corrupted or cleaned)."""
    if config.context == PRIOR:
        return np.zeros((roi.label.size, 1))
    observed = roi.label if labels is None else labels
    return context_features(graphs, roi.cell_type_idx, len(cohort.cell_types), observed, cohort.n_labels,
                            config.context, config.features)


# ---------------------------------------------------------------------------------------- learner
class PriorModel:
    """Label-prior baseline: the training cluster frequencies, whatever the context."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PriorModel":
        prior = np.bincount(y, minlength=self.n_classes).astype(float) + 1e-9
        self.prior = prior / prior.sum()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.prior, (X.shape[0], 1))


class LogisticModel:
    """Multinomial logistic regression with optional standardisation fitted on the training rows.

    ``none`` and ``l2`` use L-BFGS (2,000 iterations). ``l1`` and ``elasticnet`` use SAGA,
    stopped after 200 iterations with tolerance 1e-3 as in the paper's runs. Probabilities are
    floored at 1e-9 so a class absent from the training rows keeps a finite score.
    """

    def __init__(self, recipe: LearnerRecipe, n_classes: int, seed: int = 0):
        self.recipe, self.n_classes, self.seed = recipe, n_classes, seed
        self._scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        recipe = self.recipe
        if recipe.standardize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        # scikit-learn 1.8 removed the ``penalty`` argument; older versions need it.
        named = "penalty" in inspect.signature(LogisticRegression).parameters
        penalty = {"penalty": None if recipe.penalty == "none" else recipe.penalty} if named else {}
        shared = dict(class_weight=recipe.class_weight, random_state=self.seed, **penalty)
        if recipe.penalty == "none":
            self._model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, **shared)
        elif recipe.penalty == "l2":
            self._model = LogisticRegression(C=recipe.C, solver="lbfgs", max_iter=2000, **shared)
        else:
            l1_ratio = 1.0 if recipe.penalty == "l1" else float(recipe.l1_ratio)
            self._model = LogisticRegression(C=recipe.C, l1_ratio=l1_ratio, solver="saga", max_iter=200, tol=1e-3,
                                             **shared)
        with warnings.catch_warnings():
            # SAGA stops at its iteration limit by design, and scikit-learn comments on the
            # penalty arguments differently in each version; neither is an error here.
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        out = np.full((X.shape[0], self.n_classes), 1e-9)
        predicted = self._model.predict_proba(X)
        for column, label in enumerate(self._model.classes_):
            out[:, int(label)] = predicted[:, column]
        return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------- cleaning
def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return proba
    logp = np.log(np.maximum(proba, _EPS)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    scaled = np.exp(logp)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Temperature from a fixed grid with the lowest negative log-likelihood on the training rows."""
    best, best_nll = 1.0, np.inf
    for temperature in TEMPERATURES:
        scaled = apply_temperature(proba, temperature)
        nll = -np.mean(np.log(np.maximum(scaled[np.arange(y.size), y], _EPS)))
        if nll < best_nll:
            best, best_nll = temperature, nll
    return float(best)


def raw_anomaly(proba: np.ndarray, observed: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """-log P(observed label | context) of every signal patch; background patches score 0."""
    p_observed = np.ones(observed.size)
    rows = np.flatnonzero(signal)
    p_observed[rows] = proba[rows, observed[rows]]
    return np.where(signal, -np.log(np.maximum(p_observed, _EPS)), 0.0)


def clean_once(proba, observed, signal, xy, recipe: CleaningRecipe):
    """One cleaning pass. Returns the new labels, the cleaning statistic and the relabelled patches."""
    import mantpy as mt

    result = mt.tl.reconstruct_labels(
        observed, proba, threshold=recipe.threshold, categories=np.arange(proba.shape[1]), signal=signal,
        score=recipe.anomaly_stat, min_confidence=recipe.min_pred_conf, coordinates=xy,
        spatial_radius=recipe.gate_radius_px,
        min_neighbours=recipe.gate_min_neighbours if recipe.gate_radius_px > 0 else 0,
        majority_smoothing=recipe.majority_smoothing, temperature=recipe.temperature, eps=_EPS,
    )
    return result.labels, result.anomaly_scores, result.relabelled


# ---------------------------------------------------------------------------------------- metrics
def detection_metrics(anomaly: np.ndarray, is_artifact: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """ROC-AUC and average precision of the anomaly score for finding replaced labels within ``mask``."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth, score = is_artifact[mask].astype(int), anomaly[mask]
    if truth.min() == truth.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(truth, score)), "pr_auc": float(average_precision_score(truth, score))}


def reconstruction_metrics(pristine, observed, cleaned, is_artifact, signal) -> dict[str, float]:
    """Label accuracy before and after cleaning, the fraction of replaced labels restored
    (``recovery``), the fraction of uncorrupted patches relabelled (``pristine_relabel_rate``)
    and the fraction of relabelled patches that became correct (``correction_precision``)."""
    changed = (cleaned != observed) & signal
    replaced = is_artifact & signal
    untouched = signal & ~is_artifact
    return {
        "accuracy_before": float(np.mean(observed[signal] == pristine[signal])),
        "accuracy_after": float(np.mean(cleaned[signal] == pristine[signal])),
        "recovery": float(np.mean(cleaned[replaced] == pristine[replaced])) if replaced.any() else float("nan"),
        "correction_precision": float(np.mean(cleaned[changed] == pristine[changed])) if changed.any() else float("nan"),
        "pristine_relabel_rate": float(np.mean(changed[untouched])) if untouched.any() else float("nan"),
        "n_relabelled": int(changed.sum()), "n_artifact": int(replaced.sum()), "n_signal": int(signal.sum()),
    }


# ----------------------------------------------------------------------------------- fit and score
def fit_model(cohort, train_rois: list[str], config: Config, cache: GraphCache, seed: int = 0):
    """Fit on the pristine signal patches of the training ROIs.

    Returns the model and the cleaning recipe with its temperature set (fitted on the same
    training rows when the configuration asks for temperature calibration, otherwise 1).
    """
    features, targets = [], []
    for name in train_rois:
        roi = cohort.rois[name]
        X = features_for(roi, cohort, config, cache.get(roi, config.graph))
        signal = roi.label >= 0
        features.append(X[signal])
        targets.append(roi.label[signal])
    X, y = np.vstack(features), np.concatenate(targets)
    temperature = 1.0
    if config.context == PRIOR:
        model = PriorModel(cohort.n_labels).fit(X, y)
    else:
        model = LogisticModel(config.learner, cohort.n_labels, seed=seed).fit(X, y)
        if config.calibration == "temperature":
            temperature = fit_temperature(model.predict_proba(X), y)
    return model, CleaningRecipe(**{**config.cleaning.to_dict(), "temperature": temperature})


def score_roi(model, cleaning: CleaningRecipe, cohort, roi_name: str, config: Config, cache: GraphCache,
              corruption: Corruption) -> dict:
    """Detect and clean the replaced labels of one corrupted ROI.

    Cleaning runs for ``cleaning.passes`` rounds; features are recomputed from the current
    labels each round and the loop stops when nothing changes. The detection score is always
    the first-pass -log P(observed label | context). The label prior never relabels.
    """
    roi = cohort.rois[roi_name]
    graphs = cache.get(roi, config.graph)
    pristine, signal = roi.label, roi.label >= 0
    observed = corruption.corrupted.copy()
    current = observed.copy()
    anomaly = None
    for _ in range(max(1, cleaning.passes)):
        proba = model.predict_proba(features_for(roi, cohort, config, graphs, labels=current))
        if anomaly is None:
            anomaly = raw_anomaly(apply_temperature(proba, cleaning.temperature), observed, signal)
        if config.context == PRIOR:
            break
        current, _, relabelled = clean_once(proba, current, signal, roi.patch_xy, cleaning)
        if not relabelled.any():
            break
    return {"roi": roi_name, **detection_metrics(anomaly, corruption.is_artifact, signal),
            **reconstruction_metrics(pristine, observed, current, corruption.is_artifact, signal),
            "anomaly": anomaly, "cleaned": current, "observed": observed, "pristine": pristine,
            "is_artifact": corruption.is_artifact, "signal": signal}


def training_prior(cohort, train_rois: list[str]) -> np.ndarray:
    """Cluster frequencies of the pristine signal patches of the training ROIs."""
    labels = np.concatenate([cohort.rois[name].label for name in train_rois])
    counts = np.bincount(labels[labels >= 0], minlength=cohort.n_labels)
    return counts / counts.sum()

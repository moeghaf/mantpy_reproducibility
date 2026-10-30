"""Detection and reconstruction metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def detection_metrics(anomaly: np.ndarray, is_artifact: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    m = signal
    y = is_artifact[m].astype(int); s = anomaly[m]
    if y.min() == y.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def reconstruction_metrics(pristine: np.ndarray, observed: np.ndarray, cleaned: np.ndarray, is_artifact: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    m = signal
    before = float(np.mean(observed[m] == pristine[m]))
    after = float(np.mean(cleaned[m] == pristine[m]))
    changed = (cleaned != observed) & m
    art = is_artifact & m
    recovered = float(np.mean(cleaned[art] == pristine[art])) if art.any() else float("nan")
    precision = float(np.mean(cleaned[changed] == pristine[changed])) if changed.any() else float("nan")
    pristine_relabel = float(np.mean(changed[m & ~is_artifact])) if (m & ~is_artifact).any() else float("nan")
    return {"accuracy_before": before, "accuracy_after": after, "recovery": recovered, "correction_precision": precision,
            "pristine_relabel_rate": pristine_relabel, "n_relabelled": int(changed.sum()), "n_artifact": int(art.sum()), "n_signal": int(m.sum())}

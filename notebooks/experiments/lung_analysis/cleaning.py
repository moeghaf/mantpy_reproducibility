"""Anomaly scoring and the spatial cleaning rule family."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import mantpy as mt

_EPS = 1e-12


@dataclass(frozen=True)
class CleaningRecipe:
    anomaly_stat: str = "minmax_neglogp"   # minmax_neglogp | raw_neglogp | one_minus_p | margin
    threshold: float = 0.30
    min_pred_conf: float = 0.0
    gate_radius_px: float = 14.0           # 0 = gate off
    gate_min_neighbours: int = 2
    majority_smoothing: bool = False
    passes: int = 1
    temperature: float = 1.0               # probability calibration (1 = none)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    if T == 1.0:
        return proba
    logp = np.log(np.maximum(proba, _EPS)) / T
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)) -> float:
    """Pick the temperature minimising NLL on (proba, y) — fitted on training rows only."""
    best, best_nll = 1.0, np.inf
    for T in grid:
        p = apply_temperature(proba, T)
        nll = -np.mean(np.log(np.maximum(p[np.arange(y.size), y], _EPS)))
        if nll < best_nll:
            best, best_nll = T, nll
    return float(best)


def anomaly_scores(proba: np.ndarray, observed: np.ndarray, signal: np.ndarray, stat: str) -> np.ndarray:
    """Higher = more anomalous. Background rows get 0."""
    n = observed.size
    p_obs = np.full(n, 1.0)
    rows = np.flatnonzero(signal)
    p_obs[rows] = proba[rows, observed[rows]]
    neglog = -np.log(np.maximum(p_obs, _EPS))
    if stat == "raw_neglogp":
        a = neglog
    elif stat == "minmax_neglogp":
        a = np.zeros(n)
        if rows.size:
            lo, hi = neglog[rows].min(), neglog[rows].max()
            a[rows] = (neglog[rows] - lo) / (hi - lo) if hi > lo else 0.0
    elif stat == "one_minus_p":
        a = 1.0 - p_obs
    elif stat == "margin":
        a = proba.max(axis=1) - p_obs
    else:
        raise ValueError(stat)
    a = np.where(signal, a, 0.0)
    return a


def clean_once(proba, observed, signal, xy, rec: CleaningRecipe):
    """One recorded cleaning pass using the public probability reconstruction API."""
    result = mt.tl.reconstruct_labels(
        observed, proba, threshold=rec.threshold, categories=np.arange(proba.shape[1]),
        signal=signal, score=rec.anomaly_stat, min_confidence=rec.min_pred_conf,
        coordinates=xy, spatial_radius=rec.gate_radius_px,
        min_neighbours=rec.gate_min_neighbours if rec.gate_radius_px > 0 else 0,
        majority_smoothing=rec.majority_smoothing, temperature=rec.temperature, eps=_EPS,
    )
    return result.labels, result.anomaly_scores, result.relabelled

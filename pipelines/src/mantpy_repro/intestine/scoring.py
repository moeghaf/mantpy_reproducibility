"""Agreement with the manual annotation, counted over annotated pixels inside analysed patches."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def ari_from_contingency(table: np.ndarray) -> float:
    """Adjusted Rand index from a contingency table of counts."""
    table = np.asarray(table, dtype=np.int64)
    n = table.sum()
    if n < 2:
        return float("nan")

    def pairs(values):
        return (values.astype(np.float64) * (values - 1) / 2).sum()

    observed, rows, columns = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * columns / (n * (n - 1) / 2)
    denominator = (rows + columns) / 2 - expected
    return 1.0 if denominator == 0 else float((observed - expected) / denominator)


def macro_f1_from_contingency(table: np.ndarray) -> float:
    """Macro-F1 after one-to-one matching of clusters (columns) to classes (rows).

    The last column holds annotated pixels outside the analysed patches; it is never matched.
    """
    matrix = np.asarray(table[:, :-1], dtype=float)
    rows, columns = linear_sum_assignment(-matrix)
    score = np.zeros(table.shape[0])
    for row, column in zip(rows, columns):
        denominator = table[row].sum() + matrix[:, column].sum()
        score[row] = 2 * matrix[row, column] / denominator if denominator else 0
    return float(score.mean())


def pairwise_ari(first: np.ndarray, second: np.ndarray, weights: np.ndarray | None = None) -> float:
    """ARI between two labelings of the same items, optionally weighting each item."""
    first = np.searchsorted(np.unique(first), first)
    second = np.searchsorted(np.unique(second), second)
    table = np.zeros((first.max() + 1, second.max() + 1), dtype=np.int64)
    np.add.at(table, (first, second), 1 if weights is None else np.asarray(weights, dtype=np.int64))
    return ari_from_contingency(table)


class PixelScorer:
    """Score patch or pixel partitions against a pixel annotation on identical support.

    ``annotation`` is an integer image: 0 = not annotated, 1..k = annotated class.
    Patch partitions are expanded over the pixels of each patch, so every method is
    scored on the same annotated pixels.
    """

    def __init__(self, grid_y: np.ndarray, grid_x: np.ndarray, patch_size: int, annotation: np.ndarray, k: int):
        self.annotation, self.k, self.patch_size = annotation, int(k), int(patch_size)
        self.coordinates = np.stack([np.asarray(grid_y, dtype=int), np.asarray(grid_x, dtype=int)], axis=1)
        size = self.patch_size
        self.total = np.bincount(annotation.ravel(), minlength=k + 1)[1:k + 1]
        self.counts = np.stack([
            np.bincount(annotation[y * size:(y + 1) * size, x * size:(x + 1) * size].ravel(), minlength=k + 1)[1:k + 1]
            for y, x in self.coordinates
        ])
        self.covered = self.counts.sum(0)
        self.coverage = float(self.covered.sum() / self.total.sum())

    def contingency(self, labels: np.ndarray, *, pixel: bool = False, full: bool = False) -> np.ndarray:
        labels = np.asarray(labels)
        size = self.patch_size
        matrix = np.zeros((self.k, self.k + 1), dtype=np.int64)
        if pixel:
            if labels.shape != self.annotation.shape:
                raise ValueError("Pixel predictions must have the shape of the image.")
            n_clusters = int(labels.max()) + 1
            matrix = np.zeros((self.k, max(n_clusters, self.k) + 1), dtype=np.int64)
            for y, x in self.coordinates:
                truth = self.annotation[y * size:(y + 1) * size, x * size:(x + 1) * size].ravel()
                predicted = labels[y * size:(y + 1) * size, x * size:(x + 1) * size].ravel()
                mask = truth > 0
                width = matrix.shape[1] - 1
                matrix[:, :width] += np.bincount(
                    (truth[mask].astype(np.int64) - 1) * width + predicted[mask], minlength=self.k * width,
                ).reshape(self.k, width)
        else:
            if labels.shape != (len(self.coordinates),) or np.any(labels < 0):
                raise ValueError("Every analysed patch needs one non-negative cluster label.")
            labels = np.searchsorted(np.unique(labels), labels)
            n_clusters = int(labels.max()) + 1
            matrix = np.zeros((self.k, max(n_clusters, self.k) + 1), dtype=np.int64)
            for cluster in range(n_clusters):
                matrix[:, cluster] = self.counts[labels == cluster].sum(0)
        if full:
            matrix[:, -1] = self.total - self.covered
        return matrix

    def score(self, labels: np.ndarray, *, pixel: bool = False) -> dict:
        common = self.contingency(labels, pixel=pixel)
        full = self.contingency(labels, pixel=pixel, full=True)
        return {
            "ARI": ari_from_contingency(common),
            "macro_F1": macro_f1_from_contingency(common),
            "full_area_ARI": ari_from_contingency(full),
            "full_area_macro_F1": macro_f1_from_contingency(full),
            "coverage": self.coverage,
            "annotated_pixels": int(self.total.sum()),
            "covered_annotated_pixels": int(self.covered.sum()),
        }

"""Tabular learners (multinomial LR, HistGradientBoosting) behind one predict_proba interface."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import inspect

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class LearnerRecipe:
    learner: str = "lr"                  # lr | hgb | prior
    C: float = 1.0
    penalty: str = "l2"                  # l2 | l1 | elasticnet | none
    l1_ratio: float = 0.5
    class_weight: str | None = None      # None | balanced
    poly_degree: int = 1
    standardize: bool = False
    hgb: dict[str, Any] = field(default_factory=dict)
    gnn: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TabularModel:
    def __init__(self, rec: LearnerRecipe, n_classes: int, seed: int = 0):
        self.rec, self.n_classes, self.seed = rec, n_classes, seed
        self._scaler = None; self._poly = None; self._clf = None; self._prior = None

    def _transform(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if self.rec.poly_degree > 1:
            if fit:
                self._poly = PolynomialFeatures(degree=self.rec.poly_degree, include_bias=False)
                X = self._poly.fit_transform(X)
            else:
                X = self._poly.transform(X)
        if self.rec.standardize:
            if fit:
                self._scaler = StandardScaler(); X = self._scaler.fit_transform(X)
            else:
                X = self._scaler.transform(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularModel":
        self.classes_ = np.arange(self.n_classes)
        if self.rec.learner == "prior":
            self._prior = np.bincount(y, minlength=self.n_classes).astype(float) + 1e-9
            self._prior /= self._prior.sum()
            return self
        X = self._transform(X, fit=True)
        if self.rec.learner == "lr":
            # Preserve the selected penalty on both sklearn 1.7 and newer APIs.
            pen = self.rec.penalty
            penalty_kw = {"penalty": None if pen == "none" else pen} if "penalty" in inspect.signature(LogisticRegression).parameters else {}
            if pen == "none":
                self._clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, class_weight=self.rec.class_weight, random_state=self.seed, **penalty_kw)
            elif pen == "l2":
                self._clf = LogisticRegression(C=self.rec.C, solver="lbfgs", max_iter=2000, class_weight=self.rec.class_weight, random_state=self.seed, **penalty_kw)
            else:
                l1r = 1.0 if pen == "l1" else float(self.rec.l1_ratio)
                # saga is slow to converge on weakly regularised problems: cap iterations and loosen tolerance (pre-registered)
                self._clf = LogisticRegression(C=self.rec.C, l1_ratio=l1r, solver="saga", max_iter=200, tol=1e-3,
                                               class_weight=self.rec.class_weight, random_state=self.seed, **penalty_kw)
        elif self.rec.learner == "hgb":
            h = dict(self.rec.hgb)
            self._clf = HistGradientBoostingClassifier(learning_rate=h.get("learning_rate", 0.1), max_depth=h.get("max_depth", None),
                                                       max_leaf_nodes=h.get("max_leaf_nodes", 31), max_iter=h.get("max_iter", 200),
                                                       l2_regularization=h.get("l2_regularization", 0.0), min_samples_leaf=h.get("min_samples_leaf", 20),
                                                       class_weight=self.rec.class_weight, random_state=self.seed, early_stopping=False)
        else:
            raise ValueError(self.rec.learner)
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.full((n, self.n_classes), 1e-9)
        if self.rec.learner == "prior":
            return np.tile(self._prior, (n, 1))
        X = self._transform(X, fit=False)
        p = self._clf.predict_proba(X)
        for j, c in enumerate(self._clf.classes_):
            out[:, int(c)] = p[:, j]
        out /= out.sum(axis=1, keepdims=True)
        return out

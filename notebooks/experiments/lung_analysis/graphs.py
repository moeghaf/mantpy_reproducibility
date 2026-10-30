"""Lung study graph schema and weights, with topology built by Mantpy.

The selected study recipes retain their original definitions, including twice
the stored search-radius parameter as the Delaunay edge cap. Euclidean lengths
now come from Mantpy; the historical duplicate-triangle length summation is
retained only in study_code/ for audit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

@dataclass(frozen=True)
class GraphRecipe:
    cc_rule: str = "radius"            # knn | radius | delaunay
    cc_k: int = 5
    cc_radius_px: float = 15.0         # radius rule, or Dmax for knn (None = no cap)
    ee_rule: str = "grid8"             # grid8 | grid4 | knn | radius | delaunay
    ee_k: int = 8
    ee_radius_px: float = 15.0
    ee_include_background_nodes: bool = True
    ce_rule: str = "knn"               # knn | radius | delaunay | containing_ring
    ce_k: int = 5
    ce_dmax_px: float | None = 15.0
    edge_weight: str = "none"          # none | inv_d | inv_d2 | gaussian_median

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphRecipe":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

@dataclass
class Graphs:
    """Sparse layers. ``*_w`` carry weights, ``*_d`` carry Euclidean edge lengths (px)."""
    cc_w: sp.csr_matrix
    cc_d: sp.csr_matrix
    ee_w: sp.csr_matrix
    ee_d: sp.csr_matrix
    ce_w: sp.csr_matrix   # shape (n_cells, n_patches)
    ce_d: sp.csr_matrix
    recipe: GraphRecipe

    def n_edges(self) -> dict[str, int]:
        return {"cc": int(sp.triu(self.cc_d, 1).nnz), "ee": int(sp.triu(self.ee_d, 1).nnz), "ce": int(self.ce_d.nnz)}

def _weights(d: sp.csr_matrix, rule: str) -> sp.csr_matrix:
    w = d.copy().astype(np.float64)
    if rule == "none":
        w.data[:] = 1.0
    elif rule == "inv_d":
        w.data = 1.0 / w.data
    elif rule == "inv_d2":
        w.data = 1.0 / (w.data ** 2)
    elif rule == "gaussian_median":
        if w.nnz:
            sigma = float(np.median(w.data))
            w.data = np.exp(-(w.data ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"unknown edge_weight {rule!r}")
    return w

def build_graphs(cell_xy, patch_xy, patch_grid, background, patch_size, recipe: GraphRecipe) -> Graphs:
    """Build the study's three graph layers using public Mantpy graph rules."""
    from .mantpy_graphs import build_graphs_mantpy
    return build_graphs_mantpy(cell_xy, patch_xy, patch_grid, background, patch_size, recipe)

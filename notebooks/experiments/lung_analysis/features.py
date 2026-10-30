"""Neighbour-composition features for ECM-patch targets (forward task) and cell targets (reverse task).

Guards: a target node's own observed label never enters its features; two-hop blocks remove
the i->j->i self paths explicitly (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import scipy.sparse as sp

from .graphs import Graphs


@dataclass(frozen=True)
class FeatureRecipe:
    composition_norm: str = "fraction"   # fraction | count | log1p_count
    hops: int = 1                        # 1 | 2
    intensity_block: bool = False        # neighbour mean ECM marker intensities
    standardize: bool = False            # StandardScaler fitted on training rows (handled by the learner wrapper)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def one_hot(idx: np.ndarray, n: int) -> sp.csr_matrix:
    """Row i has a 1 in column idx[i]; rows with idx < 0 are all-zero."""
    idx = np.asarray(idx, int)
    rows = np.flatnonzero(idx >= 0)
    return sp.csr_matrix((np.ones(rows.size), (rows, idx[rows])), shape=(idx.size, n))


def _norm(counts: np.ndarray, rule: str) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if rule == "count":
        return counts
    if rule == "log1p_count":
        return np.log1p(counts)
    if rule == "fraction":
        s = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, s, out=np.zeros_like(counts), where=s > 0)
    raise ValueError(rule)


def _two_hop(A: sp.csr_matrix, OH: sp.csr_matrix) -> np.ndarray:
    """A @ (A @ OH) minus the self contribution sum_j w_ij w_ji * OH_i (A symmetric -> w_ij^2)."""
    first = A @ OH
    second = A @ first
    self_w = np.asarray(A.multiply(A).sum(axis=1)).ravel()
    second = second - sp.diags(self_w) @ OH
    return np.asarray(second.todense()) if sp.issparse(second) else np.asarray(second)


def _bipartite_two_hop(B: sp.csr_matrix, OH: sp.csr_matrix) -> np.ndarray:
    """(B B^T) @ OH minus self paths; B is (targets x others)."""
    first = B.T @ OH   # (others x labels)
    second = B @ first
    self_w = np.asarray(B.multiply(B).sum(axis=1)).ravel()
    second = second - sp.diags(self_w) @ OH
    return np.asarray(second.todense()) if sp.issparse(second) else np.asarray(second)


def _dense(m) -> np.ndarray:
    return np.asarray(m.todense()) if sp.issparse(m) else np.asarray(m)


def ecm_target_features(g: Graphs, cell_type_idx: np.ndarray, n_types: int, labels: np.ndarray, n_labels: int,
                        patch_feat: np.ndarray | None, context: str, fr: FeatureRecipe) -> np.ndarray:
    """Features for every patch (rows = patches). ``labels`` are the *observed* (possibly corrupted)
    patch labels with -1 for background; they enter only through neighbours."""
    OH_c = one_hot(cell_type_idx, n_types)          # cells x types
    OH_l = one_hot(labels, n_labels)                 # patches x labels (background rows zero)
    blocks: list[np.ndarray] = []
    if context in ("cell", "joint"):
        blocks.append(_norm(_dense(g.ce_w.T @ OH_c), fr.composition_norm))
        if fr.hops == 2:
            blocks.append(_norm(_dense(g.ce_w.T @ (g.cc_w @ OH_c)), fr.composition_norm))
    if context in ("ecm", "joint"):
        blocks.append(_norm(_dense(g.ee_w @ OH_l), fr.composition_norm))
        if fr.hops == 2:
            blocks.append(_norm(_two_hop(g.ee_w, OH_l), fr.composition_norm))
            blocks.append(_norm(_bipartite_two_hop(g.ce_w.T.tocsr(), OH_l), fr.composition_norm))
    if fr.intensity_block and patch_feat is not None:
        A = g.ee_w
        deg = np.asarray(A.sum(axis=1)).ravel()
        mean_nb = np.divide(_dense(A @ patch_feat), deg[:, None], out=np.zeros((A.shape[0], patch_feat.shape[1])), where=deg[:, None] > 0)
        blocks.append(mean_nb)
    if not blocks:
        raise ValueError(f"no feature blocks for context {context!r}")
    return np.hstack(blocks)


def cell_target_features(g: Graphs, cell_type_idx: np.ndarray, n_types: int, labels: np.ndarray, n_labels: int,
                         patch_feat: np.ndarray | None, context: str, fr: FeatureRecipe) -> np.ndarray:
    """Features for every cell (rows = cells). ``cell_type_idx`` are the observed (possibly corrupted)
    cell types; a cell's own type enters only through other cells' neighbourhoods."""
    OH_c = one_hot(cell_type_idx, n_types)
    OH_l = one_hot(labels, n_labels)
    blocks: list[np.ndarray] = []
    if context in ("ecm", "joint"):
        blocks.append(_norm(_dense(g.ce_w @ OH_l), fr.composition_norm))
        if fr.hops == 2:
            blocks.append(_norm(_dense(g.ce_w @ (g.ee_w @ OH_l)), fr.composition_norm))
    if context in ("cell", "joint"):
        blocks.append(_norm(_dense(g.cc_w @ OH_c), fr.composition_norm))
        if fr.hops == 2:
            blocks.append(_norm(_two_hop(g.cc_w, OH_c), fr.composition_norm))
            blocks.append(_norm(_bipartite_two_hop(g.ce_w, OH_c), fr.composition_norm))
    if fr.intensity_block and patch_feat is not None:
        B = g.ce_w
        deg = np.asarray(B.sum(axis=1)).ravel()
        blocks.append(np.divide(_dense(B @ patch_feat), deg[:, None], out=np.zeros((B.shape[0], patch_feat.shape[1])), where=deg[:, None] > 0))
    if not blocks:
        raise ValueError(f"no feature blocks for context {context!r}")
    return np.hstack(blocks)

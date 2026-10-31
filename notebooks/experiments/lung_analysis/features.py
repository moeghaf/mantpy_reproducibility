"""Study block recipes using Mantpy's public neighbourhood feature engine.

The selected weights, normalization, hop blocks and observed-label inputs stay
explicit; the package handles label aggregation and focal return-path removal.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import mantpy as mt
import numpy as np
from .graphs import Graphs


@dataclass(frozen=True)
class FeatureRecipe:
    composition_norm: str = "fraction"   # fraction | count | log1p_count
    hops: int = 1                        # 1 | 2
    intensity_block: bool = False        # neighbour mean ECM marker intensities
    standardize: bool = False            # StandardScaler fitted on training rows (handled by the learner wrapper)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _composition(adjacency, labels, categories, target_ids, source_ids, recipe, via=None):
    normalization = {"count": "counts", "fraction": "proportions",
                     "log1p_count": "log1p_counts"}[recipe.composition_norm]
    return mt.tl.neighbourhood_features(
        adjacency, labels, categories=categories, target_ids=target_ids,
        source_ids=source_ids, normalization=normalization,
        hops=2 if via is not None else 1, via=via, exclude_self=True,
    ).to_numpy()


def ecm_target_features(g: Graphs, cell_type_idx, n_types, labels, n_labels,
                        patch_feat, context, fr: FeatureRecipe):
    """Original forward context blocks; background labels contribute no category."""
    cells = np.asarray([f"cell:{i}" for i in range(len(cell_type_idx))])
    patches = np.asarray([f"ecm:{i}" for i in range(len(labels))])
    blocks = []
    if context in ("cell", "joint"):
        blocks.append(_composition(g.ce_w.T, cell_type_idx, range(n_types), patches, cells, fr))
        if fr.hops == 2:
            blocks.append(_composition(g.ce_w.T, cell_type_idx, range(n_types), patches, cells, fr, g.cc_w))
    if context in ("ecm", "joint"):
        blocks.append(_composition(g.ee_w, labels, range(n_labels), patches, patches, fr))
        if fr.hops == 2:
            blocks.append(_composition(g.ee_w, labels, range(n_labels), patches, patches, fr, g.ee_w))
            blocks.append(_composition(g.ce_w.T, labels, range(n_labels), patches, patches, fr, g.ce_w))
    if fr.intensity_block and patch_feat is not None:
        blocks.append(mt.tl.neighbourhood_features(
            g.ee_w, None, values=patch_feat, normalization="mean",
            target_ids=patches, source_ids=patches, exclude_self=True,
        ).to_numpy())
    if not blocks:
        raise ValueError(f"no feature blocks for context {context!r}")
    return np.hstack(blocks)


def cell_target_features(g: Graphs, cell_type_idx, n_types, labels, n_labels,
                         patch_feat, context, fr: FeatureRecipe):
    """Original reverse context blocks with stable compartment-prefixed IDs."""
    cells = np.asarray([f"cell:{i}" for i in range(len(cell_type_idx))])
    patches = np.asarray([f"ecm:{i}" for i in range(len(labels))])
    blocks = []
    if context in ("ecm", "joint"):
        blocks.append(_composition(g.ce_w, labels, range(n_labels), cells, patches, fr))
        if fr.hops == 2:
            blocks.append(_composition(g.ce_w, labels, range(n_labels), cells, patches, fr, g.ee_w))
    if context in ("cell", "joint"):
        blocks.append(_composition(g.cc_w, cell_type_idx, range(n_types), cells, cells, fr))
        if fr.hops == 2:
            blocks.append(_composition(g.cc_w, cell_type_idx, range(n_types), cells, cells, fr, g.cc_w))
            blocks.append(_composition(g.ce_w, cell_type_idx, range(n_types), cells, cells, fr, g.ce_w.T))
    if fr.intensity_block and patch_feat is not None:
        blocks.append(mt.tl.neighbourhood_features(
            g.ce_w, None, values=patch_feat, normalization="mean",
            target_ids=cells, source_ids=patches, exclude_self=True,
        ).to_numpy())
    if not blocks:
        raise ValueError(f"no feature blocks for context {context!r}")
    return np.hstack(blocks)

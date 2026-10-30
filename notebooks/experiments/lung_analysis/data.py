"""Prepared-data model: one ROIData per ROI, loaded from the tidy CSVs written by prepare.py."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COARSE_GROUPS = {
    "AEC": "epithelial", "ATI": "epithelial", "ATII": "epithelial",
    "Endothelial": "endothelial",
    "Alveolar Fibroblast": "mesenchymal", "Fibroblast": "mesenchymal", "Peribronchial Fibroblasts": "mesenchymal", "Smooth Muscle": "mesenchymal",
    "Alveolar Macrophages": "immune", "B Cells": "immune", "CD11b Immune": "immune", "Neutrophils": "immune",
}


@dataclass
class ROIData:
    roi: str
    mouse: str
    patch_size: int
    patch_id: np.ndarray
    patch_xy: np.ndarray
    patch_grid: np.ndarray
    patch_feat: np.ndarray
    label: np.ndarray            # -1 background
    cell_id: np.ndarray
    cell_xy: np.ndarray
    cell_type: np.ndarray        # str
    cell_type_idx: np.ndarray    # int into Cohort.cell_types

    @property
    def background(self) -> np.ndarray:
        return self.label < 0


@dataclass
class Cohort:
    setting: str
    rois: dict[str, ROIData]
    cell_types: list[str]
    n_labels: int
    meta: dict = field(default_factory=dict)

    @property
    def roi_names(self) -> list[str]:
        return list(self.rois)

    def mice(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.rois.values():
            out.setdefault(r.mouse, []).append(r.roi)
        return out

    def with_granularity(self, granularity: str) -> "Cohort":
        if granularity == "fine12":
            return self
        groups = sorted(set(COARSE_GROUPS.values()))
        rois = {}
        for k, r in self.rois.items():
            ct = np.array([COARSE_GROUPS.get(t, "other") for t in r.cell_type])
            idx = np.array([groups.index(g) if g in groups else -1 for g in ct])
            rois[k] = ROIData(**{**r.__dict__, "cell_type": ct, "cell_type_idx": idx})
        return Cohort(f"{self.setting}__{granularity}", rois, groups, self.n_labels, dict(self.meta, granularity=granularity))


def load_cohort(prepared_dir: str, setting: str) -> Cohort:
    d = os.path.join(prepared_dir, setting)
    patches = pd.read_csv(os.path.join(d, "patches.csv"))
    cells = pd.read_csv(os.path.join(d, "cells.csv"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    cell_types = sorted(cells["cell_type"].astype(str).unique().tolist())
    feat_cols = [c for c in patches.columns if c.startswith("feat_")]
    rois: dict[str, ROIData] = {}
    for roi, p in patches.groupby("roi", sort=True):
        c = cells[cells["roi"] == roi]
        ct = c["cell_type"].astype(str).to_numpy()
        rois[roi] = ROIData(
            roi=roi, mouse=str(p["mouse"].iloc[0]), patch_size=int(meta["patch_size"]),
            patch_id=p["patch_id"].to_numpy(), patch_xy=p[["x", "y"]].to_numpy(float),
            patch_grid=p[["grid_y", "grid_x"]].to_numpy(int), patch_feat=p[feat_cols].to_numpy(float),
            label=p["label"].to_numpy(int), cell_id=c["cell_id"].to_numpy(), cell_xy=c[["x", "y"]].to_numpy(float),
            cell_type=ct, cell_type_idx=np.array([cell_types.index(t) for t in ct], int))
    return Cohort(setting=setting, rois=rois, cell_types=cell_types, n_labels=int(meta["n_labels"]), meta=meta)

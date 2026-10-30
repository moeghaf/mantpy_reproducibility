"""Generate, store and load benchmark realisations (one Corruption per ROI) per setting."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .benchmark import Corruption, corrupt_labels
from .data import Cohort, COARSE_GROUPS

FORWARD_KINDS = ("mixed", "block", "isolated")
REVERSE_KINDS = ("mixed", "within_group")


def _cell_group_map(cohort: Cohort) -> np.ndarray:
    groups = sorted(set(COARSE_GROUPS.values()))
    return np.array([groups.index(COARSE_GROUPS.get(t, "other")) if COARSE_GROUPS.get(t, "other") in groups else -1 for t in cohort.cell_types])


def make_realisation(cohort: Cohort, task: str, kind: str, seed: int, protocol: dict) -> dict[str, Corruption]:
    b = protocol["benchmark"]
    out = {}
    for i, (name, roi) in enumerate(cohort.rois.items()):
        s = seed * 1000 + i
        if task == "forward":
            out[name] = corrupt_labels(roi.patch_xy, roi.label, cohort.n_labels, kind, s, budget_fraction=b["budget_fraction"],
                                       block_sides=b["block_side_px"], min_units=b["min_signal_patches_per_block"])
        else:
            groups = _cell_group_map(cohort) if kind == "within_group" else None
            out[name] = corrupt_labels(roi.cell_xy, roi.cell_type_idx, len(cohort.cell_types), kind, s,
                                       budget_fraction=b["reverse_task"]["budget_fraction"], block_sides=b["block_side_px"],
                                       min_units=b["min_signal_patches_per_block"], groups=groups)
    return out


def realisation_path(bench_dir: str, setting: str, task: str, kind: str, seed: int) -> str:
    return os.path.join(bench_dir, setting, f"{task}_{kind}_seed{seed}.csv")


def save_realisation(real: dict[str, Corruption], cohort: Cohort, task: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = []
    for name, c in real.items():
        roi = cohort.rois[name]
        ids = roi.patch_id if task == "forward" else roi.cell_id
        frames.append(pd.DataFrame({"roi": name, "unit_id": ids, "pristine": c.pristine, "corrupted": c.corrupted, "is_artifact": c.is_artifact}))
    pd.concat(frames).to_csv(path, index=False)
    blocks = {name: c.blocks for name, c in real.items()}
    json.dump(blocks, open(path.replace(".csv", "_blocks.json"), "w"), indent=1)


def load_realisation(cohort: Cohort, task: str, path: str, kind: str, seed: int) -> dict[str, Corruption]:
    df = pd.read_csv(path)
    out = {}
    for name, roi in cohort.rois.items():
        d = df[df["roi"] == name]
        ids = roi.patch_id if task == "forward" else roi.cell_id
        d = d.set_index("unit_id").loc[ids]
        out[name] = Corruption(is_artifact=d["is_artifact"].to_numpy(bool), corrupted=d["corrupted"].to_numpy(int),
                               pristine=d["pristine"].to_numpy(int), blocks=[], seed=seed, kind=kind)
    return out


def ensure_realisations(cohort: Cohort, bench_dir: str, protocol: dict) -> None:
    b = protocol["benchmark"]
    seeds = list(b["evaluation_seeds"]) + list(b["tuning_seeds"])
    for task, kinds in (("forward", FORWARD_KINDS), ("reverse", REVERSE_KINDS)):
        for kind in kinds:
            for seed in seeds:
                p = realisation_path(bench_dir, cohort.setting, task, kind, seed)
                if not os.path.exists(p):
                    save_realisation(make_realisation(cohort, task, kind, seed, protocol), cohort, task, p)


def get_realisation(cohort: Cohort, bench_dir: str, task: str, kind: str, seed: int) -> dict[str, Corruption]:
    return load_realisation(cohort, task, realisation_path(bench_dir, cohort.setting, task, kind, seed), kind, seed)

"""The six lung ROIs: cells, 10 x 10-pixel ECM patches and the ECM partition.

``extract`` reads the public dataset once and returns plain arrays, which the ``prepare``
stage stores. Every later stage builds its cohorts from those arrays (:class:`Prepared`), so
cluster jobs need neither the dataset nor internet access.

Two partitions of the patches are available. ``fresh`` is recomputed here by the paper's
procedure: K-means with three to seven components scored by the silhouette of the signal
patches, the selected fit, and removal of the lowest-signal component as background.
``paper`` is the partition of the paper's runs (11,435 signal patches), packaged in
``paper_partition.json`` from the published source data of Supplementary Fig. 3. K-means with
several OpenMP threads does not repeat with a fixed seed, so the fresh partition is fitted
with one thread, and ``paper`` is the default so that the results do not depend on the
installed K-means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from ..runtime import calculation_threads
from .config import PATCH_SIZE

DATASET_DOI = "10.5281/zenodo.21538382"
SIGNAL_K_RANGE = (2, 3, 4, 5, 6)     # three to seven K-means components, one of them background
PARTITION_SEED = 0
PARTITION_THREADS = 1                # the paper's partition was fitted with one thread
PARTITIONS = ("paper", "fresh")
NOT_A_PATCH = -2                     # tile outside the patch table of a partition


@dataclass
class ROIData:
    """One ROI. ``label`` holds the ECM cluster of each patch, -1 for background."""

    roi: str
    mouse: str
    patch_size: int
    patch_id: np.ndarray
    patch_xy: np.ndarray          # patch centres (x, y) in pixels
    patch_feat: np.ndarray        # mean standardised intensity of each ECM marker
    label: np.ndarray
    cell_id: np.ndarray
    cell_xy: np.ndarray
    cell_type: np.ndarray
    cell_type_idx: np.ndarray     # index into Cohort.cell_types

    @property
    def background(self) -> np.ndarray:
        return self.label < 0


@dataclass
class Cohort:
    """All ROIs under one partition. The ROI order fixes the corruption seeds."""

    setting: str
    rois: dict[str, ROIData]
    cell_types: list[str]
    n_labels: int
    ecm_names: list[str] = field(default_factory=list)

    @property
    def roi_names(self) -> list[str]:
        return list(self.rois)

    def mice(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for roi in self.rois.values():
            groups.setdefault(roi.mouse, []).append(roi.roi)
        return groups

    def label_digest(self) -> str:
        """Identity of the partition, stored with every result computed on it."""
        digest = hashlib.sha256()
        for name, roi in self.rois.items():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(roi.label, dtype=np.int64).tobytes())
        return digest.hexdigest()[:16]


# Extraction
def load_bundle(cache_dir: str | None = None):
    """The public dataset (downloaded and cached on first use)."""
    import mantpy as mt

    return mt.datasets.balbc_pbs_lung(cache_dir=cache_dir, progressbar=False)


def retain_patch_support(ecm: dict) -> None:
    """Drop, in every ROI, the patches whose mean standardised feature equals the ROI minimum.

    The paper's runs used Mantpy 1.0.1, whose patch extraction at ``background_quantile=0``
    kept only patches strictly above the ROI's lowest mean feature (computed in float32).
    Mantpy 1.1 and later keep every patch, so the rule is applied here before clustering.
    """
    for adata in ecm.values():
        patches = adata.uns["ecm_patches"]
        columns = [column for column in patches if column.startswith("feat_")]
        row_means = patches[columns].to_numpy(dtype=np.float32).mean(axis=1)
        retained = patches.loc[row_means > np.quantile(row_means, 0.0)].reset_index(drop=True)
        retained.index = retained.index.astype(str)
        adata.uns["ecm_patches"] = retained


def raw_statistics(bundle, patch_size: int = PATCH_SIZE) -> dict[str, dict]:
    """Unnormalised patch means and whole-image pixel moments of the ECM markers of every ROI.

    The refit control standardises markers within training sets, so it starts from these raw
    statistics instead of the pooled standardised features.
    """
    panel = bundle.panel.loc[bundle.panel["keep"].eq(1)]
    planes = [i for i, value in enumerate(panel["ecm"]) if int(value) == 1]
    raw = {}
    for name in bundle.roi_names:
        image = np.asarray(bundle.images[name])[planes].astype(np.float64)
        rows, columns = image.shape[1] // patch_size, image.shape[2] // patch_size
        means = image[:, :rows * patch_size, :columns * patch_size].reshape(
            len(planes), rows, patch_size, columns, patch_size).mean((2, 4))
        raw[name] = {
            "means": means.transpose(1, 2, 0).reshape(-1, len(planes)),
            "pixel_sum": image.reshape(len(planes), -1).sum(1),
            "pixel_sumsq": (image * image).reshape(len(planes), -1).sum(1),
            "n_pixels": image.shape[1] * image.shape[2],
            "grid_shape": (rows, columns),
        }
        if not np.isfinite(raw[name]["means"]).all() or raw[name]["means"].min() < 0:
            raise ValueError(f"{name}: raw ECM intensities must be finite and non-negative.")
    return raw


def extract(bundle, patch_size: int = PATCH_SIZE) -> tuple[dict[str, np.ndarray], dict]:
    """Patches, component scan, fresh partition, cells and raw statistics as arrays.

    Tiles are stored for the whole native grid (4,554 per ROI) with their index
    ``grid_y * columns + grid_x``; ``tile_fresh_label`` is -2 for a tile outside the fresh patch
    table, -1 for background and the cluster otherwise (clusters ordered by size).
    """
    import mantpy as mt

    names = list(bundle.roi_names)
    ecm = mt.pp.ecm_patches_from_images(bundle.images, bundle.panel, normalize="pooled_zscore",
                                        patch_size=patch_size, features=["mean"], ecm_K=None,
                                        background_quantile=0.0)
    raw = raw_statistics(bundle, patch_size)
    tiles = {}
    for name in names:
        patches = ecm[name].uns["ecm_patches"]
        columns = [column for column in patches if column.startswith("feat_")]
        x, y = patches["x"].to_numpy(float), patches["y"].to_numpy(float)
        native = np.floor(y / patch_size).astype(int) * raw[name]["grid_shape"][1] + np.floor(x / patch_size).astype(int)
        tiles[name] = {"native": native, "xy": np.column_stack([x, y]), "feat": patches[columns].to_numpy(float)}

    retain_patch_support(ecm)
    with calculation_threads(PARTITION_THREADS):
        scan = mt.pp.select_ecm_cluster_count(ecm, signal_k_range=range(SIGNAL_K_RANGE[0], SIGNAL_K_RANGE[-1] + 1),
                                              random_state=PARTITION_SEED, n_seeds=3)
        clustering = mt.pp.cluster_ecm_patches(ecm, n_clusters=scan.selected_total_k, random_state=PARTITION_SEED)
    mt.pp.remove_background_patches(ecm, clustering, background="lowest_mean", order_signal_by="size")
    for name in names:
        patches = ecm[name].uns["ecm_patches"]
        kept = (np.floor(patches["y"].to_numpy(float) / patch_size).astype(int) * raw[name]["grid_shape"][1]
                + np.floor(patches["x"].to_numpy(float) / patch_size).astype(int))
        labels = np.full(len(tiles[name]["native"]), NOT_A_PATCH, dtype=np.int16)
        position = {int(value): i for i, value in enumerate(tiles[name]["native"])}
        labels[[position[int(value)] for value in kept]] = patches["ecm_cluster"].to_numpy(int)
        tiles[name]["fresh"] = labels

    mouse = bundle.metadata.set_index("sample_id")["Mouse"].astype(str)
    cell_types = sorted(set().union(*[set(cells.obs["cell_type"].astype(str)) for cells in bundle.cohort_cells.values()]))
    cells = {name: bundle.cohort_cells[name] for name in names}
    candidates = scan.candidate_scan
    arrays = {
        "roi_names": np.asarray(names), "mouse": np.asarray([mouse[name] for name in names]),
        "cell_types": np.asarray(cell_types), "ecm_names": np.asarray([str(item) for item in bundle.ecm_names]),
        "patch_size": np.asarray(patch_size),
        "grid_shape": np.asarray([raw[name]["grid_shape"] for name in names]),
        "tile_roi": np.concatenate([np.full(len(tiles[name]["native"]), i) for i, name in enumerate(names)]),
        "tile_native": np.concatenate([tiles[name]["native"] for name in names]),
        "tile_xy": np.concatenate([tiles[name]["xy"] for name in names]),
        "tile_feat": np.concatenate([tiles[name]["feat"] for name in names]),
        "tile_fresh_label": np.concatenate([tiles[name]["fresh"] for name in names]),
        "cell_roi": np.concatenate([np.full(cells[name].n_obs, i) for i, name in enumerate(names)]),
        "cell_id": np.concatenate([cells[name].obs_names.to_numpy().astype(str) for name in names]),
        "cell_xy": np.concatenate([np.asarray(cells[name].obsm["spatial"], dtype=float) for name in names]),
        "cell_type": np.concatenate([cells[name].obs["cell_type"].to_numpy().astype(str) for name in names]),
        "raw_roi": np.concatenate([np.full(len(raw[name]["means"]), i) for i, name in enumerate(names)]),
        "raw_means": np.concatenate([raw[name]["means"] for name in names]),
        "pixel_sum": np.stack([raw[name]["pixel_sum"] for name in names]),
        "pixel_sumsq": np.stack([raw[name]["pixel_sumsq"] for name in names]),
        "n_pixels": np.asarray([raw[name]["n_pixels"] for name in names]),
        "scan_signal_components": candidates["signal_k"].to_numpy(int),
        "scan_silhouette": candidates["silhouette"].to_numpy(float),
        "scan_cross_seed_ari": candidates["mean_cross_seed_ari"].to_numpy(float),
        "scan_background_patches": candidates["background_patches"].to_numpy(int),
        "selected_total_components": np.asarray(int(scan.selected_total_k)),
    }
    return arrays, {"dataset": DATASET_DOI, "mantpy": getattr(mt, "__version__", None)}


# Paper partition
def paper_partition() -> dict:
    """The paper's partition: for each ROI one character per native tile, in row-major order.

    ``x`` = tile outside the patch table, ``b`` = background, ``0``-``2`` = ECM cluster.
    """
    return json.loads(Path(__file__).with_name("paper_partition.json").read_text(encoding="utf-8"))


def decode_partition(text: str) -> np.ndarray:
    code = {"x": NOT_A_PATCH, "b": -1}
    return np.asarray([code[char] if char in code else int(char) for char in text], dtype=np.int16)


# Cohorts
class Prepared:
    """Cohorts built from the arrays stored by the ``prepare`` stage."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.roi_names = [str(name) for name in arrays["roi_names"]]
        self.cell_types = [str(name) for name in arrays["cell_types"]]
        self.ecm_names = [str(name) for name in arrays["ecm_names"]]
        self.patch_size = int(arrays["patch_size"])

    def _cells(self, index: int) -> dict:
        rows = self.arrays["cell_roi"] == index
        names = self.arrays["cell_type"][rows].astype(str)
        return {"cell_id": self.arrays["cell_id"][rows].astype(str), "cell_xy": self.arrays["cell_xy"][rows],
                "cell_type": names, "cell_type_idx": np.array([self.cell_types.index(name) for name in names])}

    def has_paper_grid(self) -> bool:
        """Whether these are the ROIs and tile grids that the paper's partition was computed on."""
        record = paper_partition()
        return (set(self.roi_names) == set(record["rois"])
                and all(tuple(shape) == tuple(record["grid_shape"]) for shape in self.arrays["grid_shape"].tolist()))

    def tile_labels(self, partition: str) -> dict[str, np.ndarray]:
        """Label of every stored tile of every ROI under ``partition`` (-2 = not a patch)."""
        a = self.arrays
        if partition == "fresh":
            return {name: a["tile_fresh_label"][a["tile_roi"] == i] for i, name in enumerate(self.roi_names)}
        if partition != "paper":
            raise ValueError(f"partition must be one of {PARTITIONS}, not {partition!r}")
        if not self.has_paper_grid():
            raise ValueError("The paper's partition applies only to the six ROIs of the lung dataset.")
        record = paper_partition()["rois"]
        labels = {}
        for i, name in enumerate(self.roi_names):
            native = a["tile_native"][a["tile_roi"] == i]
            decoded = decode_partition(record[name])
            if not np.isin(np.flatnonzero(decoded != NOT_A_PATCH), native).all() or native.max() >= len(decoded):
                raise ValueError(f"{name}: the extracted tiles do not match the paper's patch table.")
            labels[name] = decoded[native]
        return labels

    def cohort(self, partition: str) -> Cohort:
        """The six ROIs with the patches and labels of ``partition`` (patches in row-major order)."""
        a = self.arrays
        labels = self.tile_labels(partition)
        rois = {}
        for i, name in enumerate(self.roi_names):
            rows = np.flatnonzero(a["tile_roi"] == i)
            order = np.argsort(a["tile_native"][rows], kind="stable")
            rows, ordered = rows[order], labels[name][order]
            keep = ordered != NOT_A_PATCH
            rois[name] = ROIData(
                roi=name, mouse=str(a["mouse"][i]), patch_size=self.patch_size,
                patch_id=np.array([f"{name}:{j}" for j in range(int(keep.sum()))]),
                patch_xy=a["tile_xy"][rows][keep], patch_feat=a["tile_feat"][rows][keep],
                label=ordered[keep].astype(int), **self._cells(i))
        n_labels = int(max(roi.label.max() for roi in rois.values())) + 1
        return Cohort(partition, rois, self.cell_types, n_labels, self.ecm_names)

    def raw(self) -> dict[str, dict]:
        """Raw statistics of every ROI for the refit control."""
        a = self.arrays
        return {name: {"means": a["raw_means"][a["raw_roi"] == i], "pixel_sum": a["pixel_sum"][i],
                       "pixel_sumsq": a["pixel_sumsq"][i], "n_pixels": int(a["n_pixels"][i]),
                       "grid_shape": tuple(int(v) for v in a["grid_shape"][i])}
                for i, name in enumerate(self.roi_names)}

    def geometry(self) -> Cohort:
        """Every native tile of every ROI with no labels: the starting point of the refit control."""
        a = self.arrays
        raw = self.raw()
        rois = {}
        for i, name in enumerate(self.roi_names):
            indices = np.arange(len(raw[name]["means"]))
            grid = np.column_stack(np.divmod(indices, raw[name]["grid_shape"][1]))
            rois[name] = ROIData(
                roi=name, mouse=str(a["mouse"][i]), patch_size=self.patch_size,
                patch_id=np.array([f"{name}:native_grid_{j}" for j in indices]),
                patch_xy=self.patch_size * grid[:, ::-1] + self.patch_size / 2,
                patch_feat=np.zeros_like(raw[name]["means"]), label=np.full(len(indices), -1, dtype=int),
                **self._cells(i))
        return Cohort("refit_geometry", rois, self.cell_types, 0, self.ecm_names)

    def component_scan(self):
        """Silhouette of the signal patches for three to seven K-means components."""
        import pandas as pd

        a = self.arrays
        return pd.DataFrame({
            "signal_components": a["scan_signal_components"], "total_components": a["scan_signal_components"] + 1,
            "silhouette": a["scan_silhouette"], "cross_seed_ari": a["scan_cross_seed_ari"],
            "background_patches": a["scan_background_patches"]})

    def summary(self) -> dict:
        """Patch counts of both partitions and how many patches they label differently."""
        a = self.arrays
        fresh = a["tile_fresh_label"]
        out = {"rois": len(self.roi_names), "cells": int(len(a["cell_roi"])),
               "selected_total_components": int(a["selected_total_components"]),
               "fresh": {"patches": int((fresh != NOT_A_PATCH).sum()), "signal": int((fresh >= 0).sum())}}
        if self.has_paper_grid():
            paper = np.concatenate([self.tile_labels("paper")[name] for name in self.roi_names])
            out["paper"] = {"patches": int((paper != NOT_A_PATCH).sum()), "signal": int((paper >= 0).sum())}
            out["patches_labelled_differently"] = int((paper != fresh).sum())
        return out

"""Shared state of a run: results directory, prepared data, encoder fits and the frozen partition.

One encoder fit = one encoder configuration x training seed. It is clustered at every K in 3..12
and stored as one resumable unit, so the search, the selection and the final fits share it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
import time

import numpy as np

from ..runtime import environment, resolve_device
from ..store import Store
from . import models
from .config import (CLASSES, K_VALUES, SPREAD_MIN_ROIS, encoder_config, encoder_id, min_patches_per_roi, paper_partition,
                     paper_selection, validate)
from .data import Prepared

SMOKE_CROP = 200             # --smoke analyses a central 200-pixel square of every ROI with tiny training budgets


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def array_digest(*arrays: np.ndarray) -> str:
    """Short identity of integer arrays (used to tie stored fits to the patches and labels they used)."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype=np.int64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:12]


def relabel_by_size(raw: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Number clusters 0..K-1 by decreasing size (ties: the lower original label first)."""
    values, counts = np.unique(raw, return_counts=True)
    order = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    mapping = {int(value): index for index, (value, _) in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in raw], dtype=np.int64), mapping


def spread_deficit(labels: np.ndarray, roi: np.ndarray, k: int, patch_size: int) -> int:
    """Spread guard of one partition: 0 when every cluster has enough patches in at least three ROIs.

    "Enough" is ``min_patches_per_roi(patch_size)``. The value is 3 minus the smallest number of
    such ROIs over the clusters, floored at 0.
    """
    counts = np.zeros((int(k), int(roi.max()) + 1), dtype=np.int64)
    np.add.at(counts, (np.asarray(labels, dtype=np.int64), np.asarray(roi, dtype=np.int64)), 1)
    rois_per_cluster = (counts >= min_patches_per_roi(patch_size)).sum(axis=1)
    return int(max(0, SPREAD_MIN_ROIS - int(rois_per_cluster.min())))


def stability(anchor_labels: list[np.ndarray]) -> tuple[float, list[float]]:
    """Seed stability: mean pairwise adjusted Rand index on the anchors covered in every seed."""
    from sklearn.metrics import adjusted_rand_score

    covered = np.all([labels >= 0 for labels in anchor_labels], axis=0)
    pairs = [float(adjusted_rand_score(a[covered], b[covered])) for a, b in combinations(anchor_labels, 2)]
    return float(np.mean(pairs)), pairs


@dataclass
class Partition:
    """The frozen ECM clusters: every patch of the selected size with its cluster (-1 = background)."""

    name: str                # "paper" (deposited labels) or "refit" (seed-0 encoder of this results directory)
    patch_size: int
    k: int
    nodes: object            # DataFrame: roi, mouse, condition, genotype, group, x, y, cluster
    features: np.ndarray     # (n_patches, 12) marker means and standard deviations
    identity: str

    @property
    def folder(self) -> str:
        return f"{self.name}_{self.identity}"

    @property
    def n_signal(self) -> int:
        return int((self.nodes.cluster.to_numpy() >= 0).sum())


class Context:
    """Results directory, device and thread settings, prepared data and cached graphs."""

    def __init__(self, results: str | Path, *, device: str = "auto", cache_dir: str | None = None, threads: int = 16,
                 smoke: bool = False):
        self.store = Store(results)
        self.device = resolve_device(device)
        self.threads = int(threads)
        self.smoke = smoke
        mode = self.store.root / "MODE"
        expected = "smoke" if smoke else "full"
        if mode.exists() and mode.read_text().strip() != expected:
            raise RuntimeError(f"{self.store.root} holds {mode.read_text().strip()} results; "
                               f"use another --results directory for a {expected} run.")
        mode.write_text(expected)
        self.prepared = Prepared(self.store.root / "prepare", cache_dir, crop=SMOKE_CROP if smoke else None)
        self._graphs: dict[tuple[str, str], list] = {}

    # -- budgets ----------------------------------------------------------------------------------
    def encoder_budget(self, config: dict) -> dict:
        """Encoder settings actually trained: the configuration's, or a few epochs for ``--smoke``."""
        return config if not self.smoke else {**config, "cnn_epochs": max(2, config["cnn_epochs"] // 100)}

    def classifier_budget(self, config: dict) -> dict:
        """Classifier settings actually trained: the configuration's, or a few epochs for ``--smoke``."""
        return config if not self.smoke else {**config, "epochs": max(5, config["epochs"] // 20)}

    # -- signal patches ---------------------------------------------------------------------------
    def signal_source(self, selection: str) -> str:
        """Which ECM signal patches a selection analyses.

        The paper's selection uses the paper's own signal patches (deposited with its partition),
        because the two-component background split changes in a few patches with the library
        versions and the number of threads. A search, and ``--smoke``, use the split recomputed
        in this results directory.
        """
        return "paper" if selection == "paper" and not self.smoke else "recomputed"

    def signal_index(self, size: int, source: str) -> np.ndarray:
        if source == "paper":
            if size != paper_selection()["config"]["patch_size"]:
                raise ValueError("The paper's signal patches exist at its selected patch size only.")
            index, _ = paper_partition()
            if index.max() >= len(self.prepared.table(size).roi):
                raise ValueError("The paper's signal patches need the complete 24-ROI lattice.")
            return index
        return self.prepared.table(size).signal_index

    def coverage(self, size: int) -> float:
        """Coverage of a patch size, an objective of the search (recomputed background split)."""
        return self.prepared.coverage(size)

    # -- encoder fits -----------------------------------------------------------------------------
    def encoder_unit(self, config: dict, seed: int, source: str = "recomputed"):
        encoder = self.encoder_budget(encoder_config(config))
        size = encoder["patch_size"]
        index = self.signal_index(size, source)
        request = {"encoder": encoder, "seed": int(seed), "k_values": list(K_VALUES),
                   "signal": {"source": source, "n": int(len(index)), "digest": array_digest(index)}}
        return self.store.unit("encoders", source, f"p{size:02d}", encoder_id(config), f"seed_{seed}", request=request)

    def encoder_fit(self, config: dict, seed: int, source: str = "recomputed") -> dict[str, np.ndarray]:
        """Train (or load) one encoder and seed. Returns ``labels`` (one row per K in 3..12, one column
        per signal patch) and ``spread_deficit`` (one value per K)."""
        encoder = self.encoder_budget(encoder_config(config))
        size = encoder["patch_size"]
        index = self.signal_index(size, source)

        def fit():
            embedding, info = models.fit_cnn(self.prepared.blocks(size, index), encoder, seed, self.device)
            roi = self.prepared.table(size).roi[index]
            labels = np.stack([models.cluster(embedding, k, seed) for k in K_VALUES])
            deficits = np.asarray([spread_deficit(labels[i], roi, k, size) for i, k in enumerate(K_VALUES)], dtype=np.int64)
            return {"labels": labels, "spread_deficit": deficits, "losses": info["losses"]}, {
                "optimizer_updates": info["optimizer_updates"], "device": self.device, "environment": environment()}

        return self.store.compute(self.encoder_unit(config, seed, source), fit,
                                  label=f"encoder {encoder_id(config)} p{size} seed {seed}", log=log)

    def profile(self, config: dict, seeds, source: str = "recomputed") -> dict[int, dict]:
        """Per K: seed stability over ``seeds`` and the spread-guard deficit (largest over the seeds)."""
        size = config["patch_size"]
        index = self.signal_index(size, source)
        fits = [self.encoder_fit(config, seed, source) for seed in seeds]
        out = {}
        for i, k in enumerate(K_VALUES):
            value, pairs = stability([self.prepared.anchor_labels(fit["labels"][i], size, index) for fit in fits])
            out[k] = {"stability": value, "pairwise_ari": pairs,
                      "spread_deficit": int(max(int(fit["spread_deficit"][i]) for fit in fits))}
        return out

    def mean_fit_seconds(self, config: dict, seeds, source: str = "recomputed") -> float:
        return float(np.mean([self.encoder_unit(config, seed, source).metadata()["seconds"] for seed in seeds]))

    # -- frozen partition -------------------------------------------------------------------------
    def partition(self, config: dict, source: str, name: str) -> Partition:
        """The partition analysed downstream.

        ``"paper"`` is the partition deposited with the paper. ``"refit"`` is the seed-0 fit of the
        selected encoder in this results directory at the selected K, with clusters numbered by
        decreasing size (the way the paper froze its partition from the search's seed-0 fit).
        """
        config = validate(config)
        size, k = config["patch_size"], config["n_clusters"]
        index = self.signal_index(size, source)
        if name == "paper":
            if source != "paper":
                raise ValueError("The paper's partition requires --selection paper (and no --smoke).")
            _, labels = paper_partition()
        elif name == "refit":
            labels, _ = relabel_by_size(self.encoder_fit(config, 0, source)["labels"][K_VALUES.index(k)])
        else:
            raise ValueError("partition must be 'paper' or 'refit'.")
        table, meta = self.prepared.table(size), self.prepared.meta
        cluster = np.full(len(table.roi), -1, dtype=np.int64)
        cluster[index] = labels
        nodes = meta[["roi", "mouse", "condition", "genotype", "group"]].iloc[table.roi].reset_index(drop=True)
        nodes["x"], nodes["y"], nodes["cluster"] = table.x, table.y, cluster
        return Partition(name=name, patch_size=size, k=k, nodes=nodes, features=table.features,
                         identity=array_digest(np.asarray([size, k]), index, labels))

    # -- classifier graphs ------------------------------------------------------------------------
    def graphs(self, partition: Partition, graph_type: str) -> list:
        """Per-ROI graphs of the ECM signal patches (sorted ROI order), built once per process."""
        from .classify import build_graphs

        key = (partition.folder, graph_type)
        if key not in self._graphs:
            self._graphs[key] = build_graphs(self, partition, graph_type)
        return self._graphs[key]

    def roi_table(self):
        """One row per ROI in sorted order: roi, mouse, condition, genotype, group, infected (0 or 1)."""
        meta = self.prepared.meta[["roi", "mouse", "condition", "genotype", "group"]].copy()
        meta["infected"] = (meta.condition == CLASSES[1]).astype(int)
        return meta

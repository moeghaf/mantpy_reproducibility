"""One fit = one image x configuration x seed. Cached CNN features, resumable results."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time

import numpy as np

from ..runtime import environment, resolve_device
from ..store import Store, claim_mode
from . import models
from .config import IMAGES, cnn_id, config_id, has_baseline_cnn, validate
from .data import Prepared
from .scoring import pairwise_ari

MIN_FRACTION = 0.01          # every cluster must hold at least 1% of the common anchors
MAX_DIRECTED_EDGES = 2.5e6   # nsga mode skips larger graphs (memory and time); paper mode fits every packaged configuration
SMOKE_CROP = 4096            # --smoke analyses a central 4,096-pixel square with tiny training budgets


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class Context:
    """Shared state of a pipeline run: results directory, device and prepared images."""

    def __init__(self, results: str | Path, *, device: str = "auto", cache_dir: str | None = None,
                 cnn_init: str = "paper", smoke: bool = False):
        self.store = Store(results)
        self.device = resolve_device(device)
        self.cache_dir = cache_dir
        self.cnn_init = cnn_init
        self.smoke = smoke
        claim_mode(self.store.root, smoke)
        self._prepared: dict[str, Prepared] = {}

    def prepared(self, image: str) -> Prepared:
        if image not in IMAGES:
            raise KeyError(f"Unknown image {image!r}; expected one of {IMAGES}.")
        if image not in self._prepared:
            self._prepared[image] = Prepared(image, self.store.root / "prepare" / image, self.cache_dir,
                                             crop=SMOKE_CROP if self.smoke else None)
        return self._prepared[image]

    def budget(self, config: dict) -> dict:
        """Training budgets actually used: the configuration's, or tiny ones for ``--smoke``."""
        if not self.smoke:
            return config
        return {**config, "cnn_epochs": max(2, config["cnn_epochs"] // 50), "gm_updates": max(5, config["gm_updates"] // 20)}

    def init_for(self, config: dict, seed: int) -> str:
        """Which random start the CNN takes (see :func:`models.fit_cnn`).

        ``paper`` follows the paper's runs. Their features of the baseline CNN for seeds 0-3 came
        from a network that was created once. Every other CNN was created by the search code,
        which created a default network first and so drew the starting weights twice.
        """
        if self.cnn_init == "paper":
            return "once" if has_baseline_cnn(config) and seed <= 3 else "rebuilt"
        return self.cnn_init

    # Cached CNN features
    def cnn_features(self, image: str, config: dict, seed: int) -> np.ndarray:
        prepared = self.prepared(image)
        size, init = config["patch_size"], self.init_for(config, seed)
        request = {"image": image, "cnn": {key: config[key] for key in sorted(config) if key.startswith("cnn_")},
                   "patch_size": size, "seed": seed, "init": init, "thresholds": prepared.limits}
        unit = self.store.unit("cnn", image, f"p{size}", cnn_id(config), f"seed_{seed}_{init}", request=request)

        def fit():
            data = prepared.patches(size)
            windows = np.asarray(data.obsm["image_patches"], dtype=np.float32)
            if config["cnn_window"] != 1:
                gy, gx = prepared.grid(size)
                windows = models.context_windows(prepared.scaled(), gy, gx, size, config["cnn_window"])
            features, info = models.fit_cnn(windows, self.budget(config), seed, self.device, init)
            return {"features": features, "losses": info["losses"]}, {
                "optimizer_updates": info["optimizer_updates"], "environment": environment()}

        return self.store.compute(unit, fit, label=f"cnn {image} {cnn_id(config)} seed {seed}", log=log)["features"]

    # One complete fit
    def run_unit(self, image: str, config: dict, seed: int, *, control: str | None = None,
                 encoder: str | None = None, keep_embedding: bool = False) -> dict[str, np.ndarray]:
        """Fit (or load) one configuration and seed. Returns patch labels and common-anchor labels.

        ``control`` is ``None``, ``"shuffled_edges"`` or ``"no_edges"``. ``encoder`` overrides the
        GraphMAE encoder on the same graph and features (used for the GIN ablation).
        """
        config = validate(config)
        prepared = self.prepared(image)
        size, k = config["patch_size"], prepared.k
        variant = control or (f"encoder_{encoder}" if encoder else "full")
        request = {"image": image, "config": config, "seed": seed, "variant": variant,
                   "init": self.init_for(config, seed), "thresholds": prepared.limits}
        unit = self.store.unit("runs", image, config_id(config), variant, f"seed_{seed}", request=request)

        def fit():
            features = self.cnn_features(image, config, seed)
            # A copy: prepared.patches() returns the same object to every fit of this process, and
            # the lines below remove its tiles and add the graph.
            data = prepared.patches(size).copy()
            del data.obsm["image_patches"]
            data.obsm["X_cnn"] = features
            edges = models.build_graph(data, config)
            if control:
                models.apply_control(data, control, seed)
            started = time.perf_counter()
            embedding, info = models.fit_graphmae(data, self.budget(config), seed, self.device, encoder)
            graph_seconds = time.perf_counter() - started
            arrays = {"labels_graphmae": models.cluster(embedding, k), "labels_cnn": models.cluster(features, k),
                      "losses": info["losses"]}
            support = prepared.support
            for name in ("graphmae", "cnn"):
                arrays[f"common_{name}"] = support.project(arrays[f"labels_{name}"], size).astype(np.int16)
            if keep_embedding:
                arrays["embedding"] = embedding
            fractions = np.bincount(arrays["common_graphmae"], minlength=k) / len(arrays["common_graphmae"])
            return arrays, {"directed_edges": edges, "graph_seconds": graph_seconds,
                            "min_common_fraction": float(fractions.min()), "environment": environment()}

        label = f"run {image} {config_id(config)} {variant} seed {seed}"
        return self.store.compute(unit, fit, label=label, log=log,
                                  require=("embedding",) if keep_embedding else ())

    def unit_metadata(self, image: str, config: dict, seed: int, variant: str = "full") -> dict:
        folder = self.store.root / "runs" / image / config_id(validate(config)) / variant / f"seed_{seed}"
        from ..store import read_json

        return read_json(folder / "completion.json")


def stability(common_labels: list[np.ndarray]) -> tuple[float, list[float]]:
    """Seed stability: mean pairwise ARI of the labels on the common anchors."""
    pairs = [pairwise_ari(a, b) for a, b in combinations(common_labels, 2)]
    return (float(np.mean(pairs)) if pairs else float("nan")), pairs


def min_fraction(common_labels: list[np.ndarray], k: int) -> float:
    """Smallest cluster share on the common anchors over all seeds (0 if a cluster is empty)."""
    return float(min((np.bincount(labels, minlength=k) / len(labels)).min() for labels in common_labels))

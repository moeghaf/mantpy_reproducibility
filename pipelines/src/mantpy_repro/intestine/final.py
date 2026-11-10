"""Final fits of the selected configuration.

They give the scores of Fig. 2a, b and d, the intensity baselines of Fig. 2c and the seed
stability and silhouette of Supplementary Fig. 1a. The maps of Fig. 2 are drawn in the notebook.

For each image: five training seeds of the selected configuration, the two graph controls
(shuffled edges, no edges), the GIN ablation, the intensity baselines, CNN Ward clustering and
the choice of the number of clusters by seed stability.
"""

from __future__ import annotations

import numpy as np

from ..runtime import calculation_threads
from .config import config_id
from .core import Context, log
from .freeze import selected_config

FINAL_SEEDS = (0, 1, 2, 3, 4)
CONTROLS = ("shuffled_edges", "no_edges")
K_VALUES = tuple(range(2, 9))
K_SEEDS = (0, 1, 2)
BASELINE_THREADS = 8      # the paper's K-means baselines used eight threads (centres depend slightly on it)


def final_tasks(image: str) -> list[dict]:
    """Every independent task of the final stage (a GraphMAE fit, the baselines or the K sweep),
    for running as a cluster job array."""
    tasks = []
    for seed in FINAL_SEEDS:
        tasks.append({"image": image, "kind": "full", "seed": seed})
        tasks += [{"image": image, "kind": control, "seed": seed} for control in CONTROLS]
        tasks.append({"image": image, "kind": "gin", "seed": seed})
    tasks.append({"image": image, "kind": "baselines", "seed": 0})
    tasks.append({"image": image, "kind": "k_sweep", "seed": 0})
    return tasks


def run_task(ctx: Context, task: dict, selection: str = "paper") -> None:
    image, kind, seed = task["image"], task["kind"], task["seed"]
    config, _ = selected_config(ctx, image, selection)
    if kind == "full":
        ctx.run_unit(image, config, seed, keep_embedding=True)
    elif kind in CONTROLS:
        ctx.run_unit(image, config, seed, control=kind)
    elif kind == "gin":
        if config["edge_use"] == "gin":
            log(f"{image}: the selected configuration already uses GIN; skipping the ablation")
        else:
            ctx.run_unit(image, config, seed, encoder="gin")
    elif kind == "baselines":
        baselines(ctx, image, config)
    elif kind == "k_sweep":
        k_sweep(ctx, image, config)
    else:
        raise ValueError(kind)


# Baselines
def pixel_kmeans(raw: np.ndarray, k: int, clip: float, n_sample: int = 2_000_000, seed: int = 0) -> np.ndarray:
    """K-means on pixel intensities, fitted on a fixed random sample of pixels (background included).
    Every pixel is assigned to its nearest centre; labels are ordered by centre intensity."""
    from sklearn.cluster import KMeans

    height, width = raw.shape
    index = np.sort(np.random.default_rng(seed).choice(height * width, min(n_sample, height * width), replace=False))
    sample = (np.minimum(raw.ravel()[index].astype(np.float32), clip) / clip).reshape(-1, 1)
    with calculation_threads(BASELINE_THREADS):
        centres = KMeans(n_clusters=k, random_state=seed).fit(sample).cluster_centers_.ravel()
    order = np.argsort(centres)
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    labels = np.empty((height, width), dtype=np.uint8)
    for start in range(0, height, 512):
        block = np.minimum(raw[start:start + 512].astype(np.float32), clip) / clip
        labels[start:start + 512] = rank[np.argmin(np.abs(block[..., None] - centres[None, None, :]), axis=-1)]
    return labels


def baselines(ctx: Context, image: str, config: dict) -> dict[str, np.ndarray]:
    """Pixel K-means, patch-mean K-means, Multi-Otsu and CNN Ward (single runs, seed 0)."""
    prepared = ctx.prepared(image)
    size, k = config["patch_size"], prepared.k
    unit = ctx.store.unit("baselines", image, config_id(config), request={
        "image": image, "config": config, "k": k, "init": ctx.init_for(config, 0), "thresholds": prepared.limits})

    def fit():
        from skimage.filters import threshold_multiotsu
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.preprocessing import StandardScaler

        raw = prepared.loaded.raw
        pixels = pixel_kmeans(raw, k, prepared.limits["clip"])
        scorer = prepared.scorer(size)
        pixel_table = scorer.contingency(pixels, pixel=True)
        pixel_full = scorer.contingency(pixels, pixel=True, full=True)
        tiles = np.asarray(prepared.patches(size).obsm["image_patches"], dtype=np.float32)
        means = tiles.mean(axis=(1, 2, 3))
        with calculation_threads(BASELINE_THREADS):
            patch_mean = KMeans(n_clusters=k, random_state=0).fit_predict(means.reshape(-1, 1))
        otsu = np.digitize(means, threshold_multiotsu(means, classes=k))
        features = ctx.cnn_features(image, config, 0)
        scaled = StandardScaler().fit_transform(np.asarray(features, dtype=np.float32)).astype(np.float32)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        gy, gx = prepared.grid(size)
        majority = np.stack([
            np.bincount(pixels[y * size:(y + 1) * size, x * size:(x + 1) * size].ravel(), minlength=k)
            for y, x in zip(gy, gx)]).argmax(1)
        # The report reads neither the contingency over all pixels nor the patch-majority labels
        # of pixel K-means. They are saved for inspection.
        return {"pixel_contingency": pixel_table, "pixel_contingency_full": pixel_full,
                "pixel_kmeans_patch_majority": majority.astype(np.int16),
                "patch_mean_kmeans": patch_mean.astype(np.int16), "multi_otsu": otsu.astype(np.int16),
                "cnn_ward": ward.astype(np.int16)}, {}

    return ctx.store.compute(unit, fit, label=f"baselines {image}", log=log)


# Number of clusters
def k_sweep(ctx: Context, image: str, config: dict) -> dict[str, np.ndarray]:
    """Recluster the saved GraphMAE embeddings and CNN features of seeds 0-2 at K = 2..8.

    For each K: seed stability (mean pairwise ARI over the three seeds, all patches) and the mean
    silhouette on 2,000 sampled patches. The label-free choice is the most stable K, ties within
    0.01 going to the smallest K.
    """
    from itertools import combinations

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from .models import KMEANS

    prepared = ctx.prepared(image)
    unit = ctx.store.unit("k_sweep", image, config_id(config), request={
        "image": image, "config": config, "k_values": list(K_VALUES), "seeds": list(K_SEEDS),
        "thresholds": prepared.limits})

    def fit():
        embeddings = {
            "graphmae": [ctx.run_unit(image, config, seed, keep_embedding=True)["embedding"] for seed in K_SEEDS],
            "cnn": [ctx.cnn_features(image, config, seed) for seed in K_SEEDS],
        }
        arrays = {}
        settings = {key: value for key, value in KMEANS.items() if key != "random_state"}
        for name, matrices in embeddings.items():
            sample = np.random.default_rng(0).choice(len(matrices[0]), size=min(2000, len(matrices[0])), replace=False)
            labels = np.empty((len(K_VALUES), len(K_SEEDS), len(matrices[0])), dtype=np.int16)
            silhouettes = np.empty((len(K_VALUES), len(K_SEEDS)))
            for i, k in enumerate(K_VALUES):
                for seed, values in zip(K_SEEDS, matrices):
                    with calculation_threads(3):
                        labels[i, seed] = KMeans(n_clusters=k, random_state=seed, **settings).fit_predict(values)
                        silhouettes[i, seed] = silhouette_score(values[sample], labels[i, seed][sample])
            arrays[f"{name}_labels"] = labels
            arrays[f"{name}_silhouette"] = silhouettes.mean(1)
            arrays[f"{name}_stability"] = np.array([
                np.mean([adjusted_rand_score(labels[i, a], labels[i, b]) for a, b in combinations(range(len(K_SEEDS)), 2)])
                for i in range(len(K_VALUES))])
        arrays["k_values"] = np.asarray(K_VALUES)
        return arrays, {}

    return ctx.store.compute(unit, fit, label=f"k sweep {image}", log=log)


def label_free_k(stability: np.ndarray, tie: float = 0.01) -> int:
    """Most stable K; ties within ``tie`` go to the smallest K."""
    stability = np.asarray(stability)
    return int(np.asarray(K_VALUES)[stability >= stability.max() - tie].min())

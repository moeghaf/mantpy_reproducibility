"""Paper-specific pieces of the intestine Figure 2 notebook.

The notebook's section 2 shows every Mantpy step as a plain call. This file holds what is specific to
reproducing the paper, and each function takes everything it uses as an argument:

- loading the two images (``load_images``);
- the paper's fitting recipe for the repeated runs (``fit_cnn``, ``fit_graphmae``, ``kmeans_domains``),
  including the recorded starting weights of the CNN (``paper_weight_draw``);
- the intensity baselines and the three external methods (``intensity_baselines``, ``run_comparator``);
- the score table and the rows of maps (``ScoreTable``, ``show_maps``).
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
from urllib.request import urlopen

import anndata as ad
import mantpy as mt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn


PIXEL_UM = 0.37744
HBM735_BUNDLE = "mantpy-hbm735-intestine.tar.gz"
HBM735_SHA256 = "be40fe0d90cc2a7512f3a5d66b454f703f39157fd0dc7472640db30fee087354"
HBM735_BYTES = 33442165


# ---------------------------------------------------------------------------------------------------
# Reproducible fits
# ---------------------------------------------------------------------------------------------------

def seed_everything(seed, threads=4):
    """Seed Python, NumPy and PyTorch, and use full float32 precision on the GPU as in the paper's runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def calculation_threads(n):
    """Limit OpenMP and BLAS threads to the number used for this step in the paper's runs."""
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old


def release_cuda():
    """Free GPU memory between model fits."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_hbm735(cache_dir=None):
    """Download image 2 (HBM735.PSTF.274) and its annotation; every file is checked against its checksum."""
    root = Path(cache_dir or os.environ.get("MANTPY_CACHE")
                or Path.home() / ".cache" / "mantpy") / "datasets" / "hbm735_intestine"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / HBM735_BUNDLE
    if not (archive.exists() and archive.stat().st_size == HBM735_BYTES
            and sha256(archive) == HBM735_SHA256):
        source = os.environ.get("MANTPY_HBM735_SOURCE")
        source = (source.rstrip("/\\") + "/" + HBM735_BUNDLE if source and "://" in source
                  else str(Path(source) / HBM735_BUNDLE) if source
                  else "https://zenodo.org/records/23740311/files/"
                       + HBM735_BUNDLE + "?download=1")
        temporary = archive.with_name(archive.name + ".part")
        if "://" in source:
            with urlopen(source, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1 << 22)
        else:
            shutil.copyfile(source, temporary)
        if temporary.stat().st_size != HBM735_BYTES or sha256(temporary) != HBM735_SHA256:
            raise ValueError("The image-2 bundle differs from its recorded checksum.")
        temporary.replace(archive)
    out = root / ("extracted-" + HBM735_SHA256[:12])
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        expected = {"annotation.npz", "image.tiff", "manifest.json"}
        if {item.name for item in members} != expected or not all(item.isfile() for item in members):
            raise ValueError("Unexpected image-2 archive members.")
        for item in members:
            target = out / item.name
            if not target.exists():
                with bundle.extractfile(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        target = out / item["path"]
        if target.parent != out or target.stat().st_size != item["size"] or sha256(target) != item["sha256"]:
            raise ValueError("The extracted image-2 data differ from their manifest.")
    return out


def load_images(cache_dir=None):
    """Load both collagen IV images with their manual annotation.

    Returns one entry per image with ``image`` (2-D collagen IV intensities), ``annotation`` (a pixel image
    of the same shape; 1, 2, ... are the annotated layers and 0 is not annotated), ``layers`` (the name of
    each annotated layer, in the order of its annotation value) and ``accession`` (the HuBMAP identifier).
    """
    import tifffile

    bundle = mt.datasets.coliv_intestine(cache_dir=cache_dir, progressbar=False)
    image = np.asarray(bundle.image)
    annotation = np.zeros(image.shape, dtype=np.uint8)
    for row in bundle.reference.reset_index().itertuples():      # image 1 is annotated per 32-pixel patch
        annotation[row.grid_y * 32:(row.grid_y + 1) * 32, row.grid_x * 32:(row.grid_x + 1) * 32] = row.layer + 1
    first = {"image": image, "annotation": annotation,
             "layers": [str(name) for name in bundle.layer_names], "accession": "HBM926.SHNZ.594"}

    root = fetch_hbm735(cache_dir)
    with np.load(root / "annotation.npz", allow_pickle=False) as saved:
        second = {"image": tifffile.imread(root / "image.tiff"),
                  "annotation": np.asarray(saved["labels"], dtype=np.uint8),
                  "layers": [str(name) for name in saved["class_names"].tolist()], "accession": "HBM735.PSTF.274"}
    if second["annotation"].shape != second["image"].shape:
        raise ValueError("Image 2 and its annotation have different shapes.")
    return {"intestine1": first, "intestine2": second}


# ---------------------------------------------------------------------------------------------------
# The CNN's starting weights in the paper's runs
# ---------------------------------------------------------------------------------------------------

class SelectedCNN(nn.Module):
    """The paper's stride-2 convolutional patch encoder, as built by ``mt.nn.encode_patches``."""

    def __init__(self, *, blocks=3, width=16, readout="avgpool2",
                 activation="relu", latent_dim=96, in_channels=1):
        super().__init__()
        if activation != "relu":
            raise ValueError("The selected intestine recipes use ReLU.")
        layers, channels = [], in_channels
        for block in range(blocks):
            output = min(width * 2 ** block, 128)
            layers.extend([nn.Conv2d(channels, output, 3, 2, 1),
                           nn.BatchNorm2d(output), nn.ReLU()])
            channels = output
        self.conv = nn.Sequential(*layers)
        if readout not in ("avgpool1", "avgpool2", "maxpool2"):
            raise ValueError(readout)
        size = 1 if readout == "avgpool1" else 2
        self.pool = nn.AdaptiveMaxPool2d(size) if readout == "maxpool2" else nn.AdaptiveAvgPool2d(size)
        self.lin = nn.Linear(channels * size * size, latent_dim)

    def forward(self, values):
        return self.lin(self.pool(self.conv(values)).flatten(1))


def projection_head(latent_dim=96, projection_dim=64):
    """The contrastive projection layers, created after the backbone as in the paper's runs."""
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                         nn.Linear(latent_dim, projection_dim))


def selected_backbone(in_channels, architecture, latent_dim=96):
    """The selected CNN with its starting weights "drawn twice".

    Some of the paper's runs created a default encoder and projection head first and then replaced them
    with the selected ones. The discarded networks consumed random numbers, so the selected network
    started from the second draw. Creating and discarding the same two networks here reproduces that start.
    """
    placeholder = SelectedCNN(in_channels=in_channels, latent_dim=latent_dim)
    placeholder_head = projection_head(latent_dim, 64)
    del placeholder, placeholder_head
    return SelectedCNN(in_channels=in_channels, latent_dim=latent_dim, **architecture)


# ---------------------------------------------------------------------------------------------------
# The paper's fitting recipe (Figure 2a-d)
# ---------------------------------------------------------------------------------------------------

# The paper's graphs: edge length and orientation, standardised over the directed edges.
EDGE_OPTIONS = dict(
    continuity_samples=6, edge_features=("log_euclidean_distance", "sin_angle2", "cos_angle2"),
    standardize_edges=True, edge_feature_precision="float64", standardization_scope="directed",
    standardization_eps=0,
)


def paper_weight_draw(selection, seed):
    """How the paper's run drew the CNN's starting weights: ``"once"`` or ``"twice"``.

    ``selection`` is ``"joint"`` (Fig. 2a, b, d: seeds 0 to 3 drawn once, seed 4 drawn twice) or
    ``"matched"`` (Fig. 2c: every seed drawn twice). ``selected_backbone`` explains the second draw.
    """
    if selection not in ("joint", "matched"):
        raise ValueError(selection)
    return "twice" if selection == "matched" or seed == 4 else "once"


def fit_cnn(patches, architecture, seed, *, draw="once", max_epochs=200, device="cuda"):
    """CNN patch features of one image, standardised; one row of 96 numbers per patch.

    ``draw="once"`` is the plain ``mt.nn.encode_patches`` call. ``draw="twice"`` builds the same network
    through ``mt.nn.PatchEncoder`` so that it starts from the second draw of weights (``selected_backbone``).
    """
    if draw == "once":
        data = patches.copy()
        mt.nn.encode_patches(
            data, latent_dim=96, conv_blocks=architecture["blocks"], conv_width=architecture["width"],
            pool_size=int(architecture["readout"][-1]), max_epochs=max_epochs, accelerator=device,
            random_state=seed, enable_progress_bar=False, enable_model_summary=False,
        )
        return np.asarray(data.obsm["X_cnn"], dtype=np.float32)
    if draw != "twice":
        raise ValueError(draw)
    windows = np.asarray(patches.obsm["image_patches"], dtype=np.float32)
    seed_everything(seed, threads=2)
    encoder = mt.nn.PatchEncoder(
        latent_dim=96, temperature=0.2, augment="default",
        backbone=selected_backbone, backbone_kwargs=dict(in_channels=1, architecture=architecture, latent_dim=96),
        projection=projection_head, projection_kwargs=dict(latent_dim=96, projection_dim=64),
    ).fit(windows, max_epochs=max_epochs, batch_size=1024, lr=1e-3, weight_decay=1e-5, random_state=seed,
          accelerator=device, num_workers=0, drop_last=None, exhaust_sampler=True, deterministic="warn",
          enable_progress_bar=False, enable_model_summary=False)
    raw = encoder.transform(windows, batch_size=4096, standardize=False).astype(np.float32)
    return StandardScaler().fit_transform(raw).astype(np.float32)


def fit_graphmae(patches, features, seed, *, neighbours, model, control=None, threads=2, device="cuda"):
    """GraphMAE embeddings of one image on its ECM graph, or on a control graph.

    ``features`` are the CNN patch features, ``neighbours`` the number of nearest patches joined to each
    patch and ``model`` the ``mt.nn.encode_graphmae`` options. ``control`` is None, ``"shuffled_edges"`` or
    ``"no_edges"`` (``mt.gr.patch_graph_control``, seeded with ``10000 + seed`` as in the paper's runs).
    """
    seed_everything(seed, threads=threads)
    data = patches.copy()
    del data.obsm["image_patches"]
    data.obsm["X_cnn"] = features
    graph = mt.gr.build_patch_graph(data, k=neighbours, continuity_thresh=0.25, **EDGE_OPTIONS)
    if control:
        graph = mt.gr.patch_graph_control(data, control, random_state=10_000 + seed)
    mt.nn.encode_graphmae(
        data, graph_key=graph.graph_key, node_feature_key="X_cnn", key_added="X_graphmae",
        random_state=seed, accelerator=device, enable_progress_bar=False, **model,
    )
    release_cuda()
    return np.asarray(data.obsm["X_graphmae"], dtype=np.float32)


def kmeans_domains(matrix, n_clusters, *, threads, **options):
    """K-means labels with the thread count of the paper's run; ``options`` go to scikit-learn's KMeans."""
    with calculation_threads(threads):
        return KMeans(n_clusters=n_clusters, **options).fit_predict(matrix)


def intensity_baselines(image, patches, n_clusters, clip):
    """The three intensity-only partitions of Fig. 2c, as single runs.

    Returns ``"Pixel K-means"`` (a label for every pixel; K-means fitted on two million sampled pixels),
    ``"Patch-mean K-means"`` and ``"Multi-Otsu"`` (one label per patch, from the patch's mean intensity).
    """
    from skimage.filters import threshold_multiotsu

    chosen = np.sort(np.random.default_rng(0).choice(image.size, 2_000_000, replace=False))
    sample = (np.minimum(image.ravel()[chosen].astype(np.float32), clip) / clip)[:, None]
    with calculation_threads(8):
        centres = KMeans(n_clusters=n_clusters, random_state=0).fit(sample).cluster_centers_.ravel()
    pixel_labels = np.empty(image.shape, dtype=np.uint8)       # every pixel assigned to its nearest centre
    for row in range(0, image.shape[0], 512):
        scaled = np.minimum(image[row:row + 512].astype(np.float32), clip) / clip
        pixel_labels[row:row + 512] = np.argmin(np.abs(scaled[..., None] - centres[None, None, :]), axis=-1)
    windows = np.asarray(patches.obsm["image_patches"], dtype=np.float32)
    means = windows.reshape(len(windows), -1).mean(1)
    with calculation_threads(8):
        patch_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(means[:, None])
    return {"Pixel K-means": pixel_labels, "Patch-mean K-means": patch_labels,
            "Multi-Otsu": np.digitize(means, threshold_multiotsu(means, classes=n_clusters))}


# ---------------------------------------------------------------------------------------------------
# Scores and maps
# ---------------------------------------------------------------------------------------------------

class ScoreTable:
    """The ARI of every scored partition: one row per image, panel, method and seed.

    ``patches`` and ``images`` are the dictionaries built in the notebook's section 1. Scoring the same
    image, panel, method and seed again replaces the row, so a cell can be run twice.
    """

    def __init__(self, patches, images):
        self.patches, self.images, self.rows = patches, images, {}

    def add(self, image, panel, method, seed, labels):
        """Score one partition with ``mt.tl.score_patch_labels``; returns the full score."""
        score = mt.tl.score_patch_labels(self.patches[image], labels, self.images[image]["annotation"],
                                         class_names=self.images[image]["layers"])
        self.rows[image, panel, method, seed] = score.ari
        return score

    def frame(self):
        """All rows as a table with columns image, panel, method, seed and ARI."""
        return pd.DataFrame([(*key, value) for key, value in self.rows.items()],
                            columns=["image", "panel", "method", "seed", "ARI"])


def show_maps(patches, results, layers, heading, *, palette, collagen=None, unmatched="No matched layer"):
    """One row of maps for one image: the manual annotation, then each scored partition by its matched layer.

    ``results`` maps a method name to its ``mt.tl.score_patch_labels`` result. ``collagen`` is an optional
    display image drawn first. Clusters without a matched layer are drawn in the ``unmatched`` colour.
    """
    import matplotlib.pyplot as plt

    maps = {"Manual annotation": next(iter(results.values())).reference}
    for method, score in results.items():
        maps[f"{method}\nARI {score.ari:.2f}"] = score.matched_labels.astype(object).fillna(unmatched)
    first = collagen is not None
    figure, axes = plt.subplots(1, len(maps) + first, figsize=(4.2 * (len(maps) + first), 5.2))
    if first:
        axes[0].imshow(collagen, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Collagen IV", fontsize=12)
    for axis, (title, labels) in zip(axes[first:], maps.items()):
        mt.pl.patch_map(patches, labels, ax=axis, palette=palette, crop=False, background="white")
        axis.set_title(title, fontsize=12)
    for axis in axes:
        axis.set_axis_off()
    shown = list(layers) + [unmatched] * any((labels == unmatched).any() for labels in maps.values())
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=palette[name], label=name)
               for name in shown]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11)
    figure.suptitle(heading, fontsize=13, y=0.99)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.show()


# ---------------------------------------------------------------------------------------------------
# External methods (Figure 2c)
# ---------------------------------------------------------------------------------------------------

def patch_table(patches, features):
    """The AnnData given to the external methods: CNN features in ``X`` and patch centres in micrometres."""
    data = ad.AnnData(X=np.array(features, copy=True), obs=pd.DataFrame(
        {"sample_id": pd.Categorical(["image"] * len(features))}, index=patches.obs_names.copy()))
    data.obsm["spatial"] = np.asarray(patches.obsm["spatial"]) * PIXEL_UM
    return data


def _dense(matrix):
    return (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)


def _pca_kmeans(matrix, n_components, n_clusters, seed, n_init):
    embedding = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(embedding)


def run_banksy(patches, features, config, n_clusters, seed, n_init):
    """BANKSY neighbour-augmented features (``k``, ``lambda``, ``max_m``), then PCA and K-means."""
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.initialize_banksy import initialize_banksy

    data = patch_table(patches, features)
    data.obs["x"], data.obs["y"] = data.obsm["spatial"][:, 0], data.obsm["spatial"][:, 1]
    banksy = initialize_banksy(
        data, coord_keys=("x", "y", "spatial"), num_neighbours=config["k"], nbr_weight_decay="scaled_gaussian",
        max_m=config["max_m"], plt_edge_hist=False, plt_nbr_weights=False, plt_agf_angles=False, plt_theta=False)
    banksy, _ = generate_banksy_matrix(data, banksy, lambda_list=[config["lambda"]], max_m=config["max_m"],
                                       verbose=False)
    matrix = _dense(banksy["scaled_gaussian"][config["lambda"]]["adata"].X)
    return _pca_kmeans(matrix, config["pca"], n_clusters, seed, n_init)


def run_utag(patches, features, config, n_clusters, seed, n_init):
    """UTAG message passing within ``radius_um``, then PCA and K-means."""
    from utag import utag

    matrix = _dense(utag(patch_table(patches, features), slide_key=None, max_dist=config["radius_um"],
                         normalization_mode=config["normalization"], apply_clustering=False, parallel=False).X)
    return _pca_kmeans(matrix, config["pca"], n_clusters, seed, n_init)


def run_cellcharter(patches, features, config, n_clusters, seed, n_init=None):
    """CellCharter neighbourhood aggregation over ``hops`` layers, then its Gaussian mixture model.

    ``config["graph"]`` is ``"native_knn6"`` (CellCharter's default six nearest neighbours) or
    ``"mantpy_k<n>"`` (the Mantpy patch graph with ``n`` neighbours).
    """
    import cellcharter as cc
    import squidpy as sq

    data = patch_table(patches, features)
    if config["graph"] == "native_knn6":
        sq.gr.spatial_neighbors(data, coord_type="generic", n_neighs=6, library_key="sample_id")
        connectivity = "spatial_connectivities"
    else:
        geometry = patches.copy()
        geometry.obsm.pop("image_patches", None)
        graph = mt.gr.build_patch_graph(geometry, k=int(config["graph"].split("_k")[1]), continuity_thresh=0.25,
                                        continuity_samples=6)
        edges = np.asarray(graph.edge_index)
        connectivity = "mantpy_connectivities"
        data.obsp[connectivity] = sparse.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                                                    shape=(len(features), len(features)))
    cc.gr.aggregate_neighbors(data, n_layers=config["hops"], aggregations="mean",
                              connectivity_key=connectivity, sample_key="sample_id")
    estimator = cc.tl.Cluster(n_clusters=n_clusters, covariance_type="full", random_state=seed,
                              trainer_params={"accelerator": "cpu", "enable_progress_bar": False})
    estimator.fit(data)
    return np.asarray(estimator.predict(data), dtype=np.int64)


COMPARATORS = {"BANKSY": run_banksy, "UTAG": run_utag, "CellCharter": run_cellcharter}


def run_comparator(name, patches, features, config, *, n_clusters, seed, n_init=10):
    """Cluster the patches with BANKSY, UTAG or CellCharter; returns one cluster label per patch.

    The method receives the same patches and CNN features as GraphMAE, applies its own neighbourhood
    transformation with the settings in ``config`` and is clustered into ``n_clusters`` groups. Progress
    messages of the external package are hidden.
    """
    seed_everything(seed, threads=4)
    with redirect_stdout(io.StringIO()), calculation_threads(4):
        return COMPARATORS[name](patches, features, config, n_clusters, seed, n_init)

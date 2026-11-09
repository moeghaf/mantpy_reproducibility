"""Model fitting for one configuration: CNN patch features, ECM graph, GraphMAE, K-means.

Training goes through Mantpy's public API (``mt.nn.PatchEncoder``, ``mt.gr.build_patch_graph``,
``mt.nn.encode_graphmae``). This module only supplies the searched variants of the CNN and of
the edge features, which Mantpy accepts as a custom backbone and as graph attributes.
"""

from __future__ import annotations

import numpy as np

from ..runtime import calculation_threads, release_cuda, seed_everything

KMEANS = {"init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4, "random_state": 0, "algorithm": "lloyd"}


# ------------------------------------------------------------------------------------------- CNN
def _conv_encoder(*, in_size, in_channels, latent_dim, blocks, width, readout, activation):
    """Stride-2 convolution blocks, a readout, then a linear layer (layer order as in the paper)."""
    from torch import nn

    class ConvEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers, channels, size = [], in_channels, in_size
            for block in range(blocks):
                output = min(width * 2 ** block, 128)
                layers += [nn.Conv2d(channels, output, 3, 2, 1), nn.BatchNorm2d(output),
                           nn.GELU(approximate="none") if activation == "gelu" else nn.ReLU()]
                channels, size = output, (size + 1) // 2
            self.conv = nn.Sequential(*layers)
            if readout == "flatten":
                self.pool, side = nn.Identity(), size
            elif readout == "avgpool1":
                self.pool, side = nn.AdaptiveAvgPool2d(1), 1
            elif readout == "avgpool2":
                self.pool, side = nn.AdaptiveAvgPool2d(2), 2
            elif readout == "maxpool2":
                self.pool, side = nn.AdaptiveMaxPool2d(2), 2
            else:
                raise ValueError(readout)
            self.lin = nn.Linear(channels * side * side, latent_dim)

        def forward(self, values):
            return self.lin(self.pool(self.conv(values)).flatten(1))

    return ConvEncoder()


def _projection(latent_dim, width):
    from torch import nn

    if width == "none":
        return nn.Identity()
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, int(width)))


def context_windows(scaled: np.ndarray, grid_y: np.ndarray, grid_x: np.ndarray, size: int, window: int) -> np.ndarray:
    """CNN inputs of ``window * size`` pixels centred on each patch, zero-padded at the image border."""
    side = window * size
    offset = (side - size) // 2
    padded = np.pad(scaled, offset, mode="constant")
    out = np.empty((len(grid_y), 1, side, side), dtype=np.float32)
    for index, (y, x) in enumerate(zip(grid_y, grid_x)):
        out[index, 0] = padded[y * size:y * size + side, x * size:x * size + side]
    return out


def fit_cnn(windows: np.ndarray, config: dict, seed: int, device: str, init: str = "once") -> tuple[np.ndarray, dict]:
    """Train the contrastive CNN of ``config`` and return the patch features.

    ``init`` selects which random starting weights the seed gives:

    - ``"once"`` creates the network a single time (Mantpy's normal behaviour);
    - ``"rebuilt"`` first creates and discards a default network and projection head, as the
      original search code did, so the real network takes the next random numbers.

    Both are fresh random starts; the option only exists to match the paper's runs exactly.
    """
    import mantpy as mt
    from sklearn.preprocessing import StandardScaler

    if init not in ("once", "rebuilt"):
        raise ValueError("init must be 'once' or 'rebuilt'.")
    shape = dict(in_size=windows.shape[-1], in_channels=windows.shape[1], latent_dim=config["cnn_latent"],
                 blocks=config["cnn_blocks"], width=config["cnn_width"], readout=config["cnn_readout"],
                 activation=config["cnn_activation"])

    def backbone():
        if init == "rebuilt":
            _conv_encoder(in_size=shape["in_size"], in_channels=shape["in_channels"], latent_dim=shape["latent_dim"],
                          blocks=3, width=16, readout="avgpool2", activation="relu")
            _projection(shape["latent_dim"], 64)
        return _conv_encoder(**shape)

    seed_everything(seed, threads=2)
    encoder = mt.nn.PatchEncoder(
        latent_dim=config["cnn_latent"], temperature=config["cnn_temperature"], augment=config["cnn_augment"],
        backbone=backbone, projection=_projection,
        projection_kwargs=dict(latent_dim=config["cnn_latent"], width=config["cnn_projection"]),
    ).fit(
        windows, max_epochs=config["cnn_epochs"], batch_size=config["cnn_batch"], lr=config["cnn_lr"],
        weight_decay=config["cnn_weight_decay"], random_state=seed, accelerator=device, devices=1, num_workers=0,
        drop_last=None, exhaust_sampler=True, deterministic="warn",
        enable_progress_bar=False, enable_model_summary=False,
    )
    raw = encoder.transform(windows, batch_size=4096, standardize=False).astype(np.float32)
    features = StandardScaler().fit_transform(raw).astype(np.float32) if config["cnn_standardize"] else raw
    losses = np.asarray(encoder.history["train_loss"], dtype=np.float64)
    if len(losses) != config["cnn_epochs"] or not np.isfinite(features).all():
        raise RuntimeError("CNN training did not complete or produced non-finite features.")
    release_cuda()
    return features, {"losses": losses, "optimizer_updates": int(encoder.optimizer_updates), "init": init}


# ----------------------------------------------------------------------------------------- graph
def build_graph(data, config: dict) -> int:
    """Build the ECM patch graph of ``config`` in ``data.uns['patch']``; return the directed edge count.

    Topology (k nearest patches, edges crossing background removed) comes from
    ``mt.gr.build_patch_graph``. The edge features of the configuration are then computed from
    the geometry and standardised over the directed edges.
    """
    import mantpy as mt

    mt.gr.build_patch_graph(
        data, k=config["neighbours"], continuity_thresh=config["continuity_thresh"], continuity_samples=6,
        standardize_edges=False, key_added="patch",
    )
    graph = data.uns["patch"]
    edges = np.asarray(graph["edge_index"], dtype=np.int64)
    xy = np.asarray(data.obsm["spatial"], dtype=np.float64)
    source, target = edges[0], edges[1]
    distance = np.maximum(np.linalg.norm(xy[target] - xy[source], axis=1), 1e-9)
    angle = np.arctan2(xy[target, 1] - xy[source, 1], xy[target, 0] - xy[source, 0])
    columns, names = [], []
    kind = config["edge_distance"]
    if kind == "log":
        columns.append(np.log(distance)); names.append("log_distance")
    elif kind == "raw":
        columns.append(distance); names.append("distance")
    elif kind == "inverse":
        columns.append(1.0 / distance); names.append("inverse_distance")
    elif kind == "inverse_square":
        columns.append(1.0 / distance ** 2); names.append("inverse_square_distance")
    elif kind == "gaussian":
        scale = float(np.median(distance))
        columns.append(np.exp(-distance ** 2 / (2 * scale ** 2))); names.append("gaussian_distance")
    elif kind != "none":
        raise ValueError(kind)
    if config["edge_orientation"]:
        columns += [np.sin(2 * angle), np.cos(2 * angle)]
        names += ["sin_angle2", "cos_angle2"]
    if config["edge_continuity"]:
        from mantpy.gr import _edge_continuity

        foreground = np.asarray(data.uns["image_ecm_patches"]["patch_foreground"], dtype=bool)
        ys, xs = data.obs["grid_y"].to_numpy(dtype=int), data.obs["grid_x"].to_numpy(dtype=int)
        columns.append(_edge_continuity(ys, xs, source, target, foreground, 6)); names.append("continuity")
    if columns:
        matrix = np.stack(columns, axis=1).astype(np.float64)
        spread = matrix.std(0)
        spread[spread == 0] = 1.0
        attributes = ((matrix - matrix.mean(0)) / spread).astype(np.float32)
    else:
        attributes = np.zeros((edges.shape[1], 0), dtype=np.float32)
    graph["edge_index"], graph["edge_attr"], graph["edge_feature_names"] = edges, attributes, names
    return int(edges.shape[1])


def apply_control(data, control: str, seed: int) -> None:
    """Graph controls of Fig. 2d. ``shuffled_edges`` permutes every edge's destination;
    ``no_edges`` removes all edges, leaving a node-wise autoencoder."""
    graph = data.uns["patch"]
    edges, attributes = np.asarray(graph["edge_index"]), np.asarray(graph["edge_attr"])
    if control == "shuffled_edges":
        order = np.random.default_rng(10_000 + seed).permutation(edges.shape[1])
        edges = np.stack([edges[0], edges[1][order]])
    elif control == "no_edges":
        edges, attributes = edges[:, :0], attributes[:0]
    else:
        raise ValueError(control)
    graph["edge_index"], graph["edge_attr"] = edges, attributes


# -------------------------------------------------------------------------------------- GraphMAE
def fit_graphmae(data, config: dict, seed: int, device: str, encoder: str | None = None) -> tuple[np.ndarray, dict]:
    """Train GraphMAE on ``data.uns['patch']`` with features ``data.obsm['X_cnn']``; return the embedding."""
    import mantpy as mt

    encoder = encoder or config["edge_use"]
    seed_everything(seed, threads=2)
    model = mt.nn.encode_graphmae(
        data, graph_key="patch", node_feature_key="X_cnn", key_added="X_graphmae", overwrite=True,
        hidden_dim=config["gm_width"], n_layers=config["gm_depth"], encoder=encoder,
        include_edge_features=encoder == "gine",
        decoder="graphmae_remask", norm="layernorm", mask_ratio=config["gm_mask"], mitigation="jk",
        max_epochs=config["gm_updates"], lr=config["gm_lr"], weight_decay=config["gm_weight_decay"],
        recon_loss="scaled_cosine", standardize=config["gm_standardize"], random_state=seed,
        accelerator=device, enable_progress_bar=False,
    )
    embedding = np.asarray(data.obsm["X_graphmae"], dtype=np.float32)
    losses = np.asarray(model.history["train_loss"], dtype=np.float64)
    if len(losses) != config["gm_updates"] or not np.isfinite(embedding).all():
        raise RuntimeError("GraphMAE training did not complete or produced a non-finite embedding.")
    release_cuda()
    return embedding, {"losses": losses}


def cluster(matrix: np.ndarray, k: int, *, random_state: int = 0, threads: int = 2) -> np.ndarray:
    """K-means with the paper's settings (ten initialisations)."""
    from sklearn.cluster import KMeans

    with calculation_threads(threads):
        return KMeans(n_clusters=k, **{**KMEANS, "random_state": random_state}).fit_predict(matrix).astype(np.int16)

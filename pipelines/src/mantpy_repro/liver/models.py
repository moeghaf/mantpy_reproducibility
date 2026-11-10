"""Model fitting: the contrastive patch encoder, K-means, and the node classifier on ECM cluster graphs.

Training goes through Mantpy's public API (``mt.nn.PatchEncoder``, ``mt.nn.train_node_classifier``,
``mt.nn.node_integrated_gradients``). This module supplies the searched variants of the CNN and of
the graph network, which Mantpy accepts as custom factories.
"""

from __future__ import annotations

import numpy as np

from ..runtime import calculation_threads, release_cuda, seed_everything

KMEANS = {"init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd"}
ENCODER_THREADS = 2          # CPU threads of the paper's encoder fits and K-means runs


# CNN
def conv_encoder(*, in_size, in_channels, latent_dim, blocks, width, readout, activation):
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


def projection_head(latent_dim, width):
    """Two-layer projection head of the contrastive loss, or none."""
    from torch import nn

    if width == "none":
        return nn.Identity()
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, int(width)))


def fit_cnn(blocks: np.ndarray, config: dict, seed: int, device: str) -> tuple[np.ndarray, dict]:
    """Train the contrastive CNN of ``config`` on the signal-patch pixel blocks.

    Returns the embedding that is clustered: the encoder output (first standardised by Mantpy's
    pooled scaler when ``cnn_standardize`` is set), then z-scored per dimension.

    For a given seed the network takes the random numbers that follow the creation of a default
    Mantpy network and projection head, as in the paper's search code, so that the starting
    weights match the paper's runs.
    """
    import mantpy as mt
    from sklearn.preprocessing import StandardScaler

    shape = dict(in_size=blocks.shape[-1], in_channels=blocks.shape[1], latent_dim=config["cnn_latent"])

    def backbone():
        conv_encoder(**shape, blocks=3, width=16, readout="avgpool2", activation="relu")
        projection_head(config["cnn_latent"], 64)
        return conv_encoder(**shape, blocks=config["cnn_blocks"], width=config["cnn_width"],
                            readout=config["cnn_readout"], activation=config["cnn_activation"])

    seed_everything(seed, threads=ENCODER_THREADS)
    encoder = mt.nn.PatchEncoder(
        latent_dim=config["cnn_latent"], temperature=config["cnn_temperature"], augment=config["cnn_augment"],
        backbone=backbone, projection=projection_head,
        projection_kwargs=dict(latent_dim=config["cnn_latent"], width=config["cnn_projection"]),
    ).fit(
        blocks, max_epochs=config["cnn_epochs"], batch_size=config["cnn_batch"], lr=config["cnn_lr"],
        weight_decay=config["cnn_weight_decay"], random_state=seed, accelerator=device, devices=1,
        precision="32-true", num_workers=0, drop_last=None, exhaust_sampler=True, deterministic="warn",
        enable_progress_bar=False, enable_model_summary=False,
    )
    output = encoder.transform(blocks, batch_size=4096, standardize=bool(config["cnn_standardize"])).astype(np.float32)
    losses = np.asarray(encoder.history["train_loss"], dtype=np.float64)
    if len(losses) != config["cnn_epochs"] or not np.isfinite(losses).all() or not np.isfinite(output).all():
        raise RuntimeError("CNN training did not complete or produced non-finite features.")
    embedding = StandardScaler().fit_transform(output).astype(np.float32)
    release_cuda()
    return embedding, {"losses": losses, "optimizer_updates": int(encoder.optimizer_updates)}


def cluster(embedding: np.ndarray, k: int, seed: int) -> np.ndarray:
    """K-means with the paper's settings; the random state is the encoder's training seed."""
    from sklearn.cluster import KMeans

    with calculation_threads(ENCODER_THREADS):
        return KMeans(n_clusters=k, random_state=seed, **KMEANS).fit_predict(embedding).astype(np.int16)


# Classifier
def node_gnn(config: dict, in_channels: int):
    """``Linear -> activation -> [convolution -> norm -> activation -> dropout (+ residual)] x L -> Linear``.

    The layers are created in the order of the paper's code (projection, convolutions, head), so
    a seed gives the same starting weights.
    """
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

    hidden, layers, kind = config["hidden_dim"], config["n_layers"], config.get("conv", "sage")
    activation = config.get("activation", "relu")

    def convolution():
        if kind == "sage":
            return SAGEConv(hidden, hidden, aggr=config.get("aggr", "mean"))
        if kind == "gcn":
            return GCNConv(hidden, hidden)
        if kind == "gat":
            heads = config.get("heads", 1)
            if hidden % heads:
                raise ValueError(f"hidden_dim={hidden} must be divisible by heads={heads}.")
            return GATConv(hidden, hidden // heads, heads=heads, concat=True)
        if kind == "gin":
            inner = nn.GELU() if activation == "gelu" else nn.ReLU()
            return GINConv(nn.Sequential(nn.Linear(hidden, hidden), inner, nn.Linear(hidden, hidden)),
                           train_eps=config.get("train_eps", False))
        raise ValueError(kind)

    class NodeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_channels, hidden)
            self.convs = nn.ModuleList([convolution() for _ in range(layers)])
            self.dropout = config["dropout"]
            self.head = nn.Linear(hidden, 2)
            norm = {"none": nn.Identity, "layer": lambda: nn.LayerNorm(hidden), "batch": lambda: nn.BatchNorm1d(hidden)}
            self.norms = nn.ModuleList([norm[config.get("norm", "none")]() for _ in range(layers)])
            self.residual = config.get("residual", False)
            self.act = F.gelu if activation == "gelu" else F.relu

        def forward(self, x, edge_index):
            h = self.act(self.proj(x))
            for conv, norm in zip(self.convs, self.norms, strict=True):
                out = F.dropout(self.act(norm(conv(h, edge_index))), p=self.dropout, training=self.training)
                h = h + out if self.residual else out
            return self.head(h)

    return NodeGNN()


def fit_classifier(graphs: list, labels: np.ndarray, config: dict, *, seed: int | None, objective: str,
                   device: str = "cpu", threads: int = 16):
    """Fit one classifier on the given training graphs (one-hot cluster label per node, graph edges).

    ``objective`` is ``"roi_mean"`` (class-weighted loss averaged within each ROI, then over ROIs;
    the classifier of Fig. 3g) or ``"patch_sum"`` (class-weighted loss pooled over all patches; the
    models explained with integrated gradients and the tuned models). Feature scaling and class
    weights use the training graphs only. ``seed=None`` continues the caller's random stream.
    """
    import mantpy as mt
    import torch

    if objective not in ("roi_mean", "patch_sum"):
        raise ValueError("objective must be 'roi_mean' or 'patch_sum'.")
    torch.set_num_threads(threads)
    return mt.nn.train_node_classifier(
        graphs, labels, model_factory=node_gnn,
        model_kwargs={"config": config, "in_channels": int(graphs[0].x.shape[1])}, n_classes=2,
        epochs=config["epochs"], lr=config["lr"], weight_decay=config["weight_decay"], seed=seed,
        loss_reduction="roi" if objective == "roi_mean" else "node", roi_class_weight_mode="after",
        class_weight="balanced" if config.get("class_weight", True) else None,
        optimizer_factory=torch.optim.AdamW if config.get("optimizer") == "adamw" else torch.optim.Adam,
        scheduler="cosine" if config.get("lr_schedule") == "cosine" else None,
        input_dropout=config.get("input_dropout", 0.0), label_smoothing=config.get("label_smoothing", 0.0),
        standardize=True, device=device,
    )


def roi_probabilities(fit, graph) -> np.ndarray:
    """Class probabilities of one ROI: the mean of its patches' probabilities."""
    return fit.predict_proba(graph).mean(0)


def integrated_gradients(fit, graph, steps: int) -> tuple[np.ndarray, float]:
    """Signed integrated gradients of the mean infected logit (zero baseline in standardised space)."""
    import mantpy as mt

    attribution = mt.nn.node_integrated_gradients(fit.model, graph, mean=fit.mean, scale=fit.scale, target_class=1,
                                                  n_steps=steps, return_details=True)
    return attribution.signed, float(np.asarray(attribution.convergence_delta).item())

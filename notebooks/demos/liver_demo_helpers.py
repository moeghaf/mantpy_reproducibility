"""Selected liver CNN, GraphSAGE loss objectives, IG and simple map display.

Local model factories preserve the selected architectures. Public Mantpy
interfaces handle fitting, prediction, attribution and patch-map display.
"""
import random

import mantpy as mt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class LiverCNN(nn.Module):
    """Selected 2-block, width-32 GELU encoder with maximum-pool 2×2 readout."""

    def __init__(self, latent_dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.pool = nn.AdaptiveMaxPool2d(2)
        self.readout = nn.Linear(64 * 2 * 2, latent_dim)

    def forward(self, x):
        return self.readout(self.pool(self.conv(x)).flatten(1))


def projection_head(latent_dim=96, projection_dim=64):
    """Paper projection layers, initialized after the selected backbone."""
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                         nn.Linear(latent_dim, projection_dim))


def fit_liver_cnn(
    windows, *, seed=0, latent_dim=96, projection_dim=64, temperature=0.1,
    epochs=200, batch_size=1024, lr=3e-4, weight_decay=1e-4, device="cpu",
):
    """Selected public trainer; export raw backbone latents in original row order."""
    seed_everything(seed)
    encoder = mt.nn.PatchEncoder(
        latent_dim=latent_dim, temperature=temperature,
        backbone=LiverCNN, backbone_kwargs=dict(latent_dim=latent_dim),
        projection=projection_head,
        projection_kwargs=dict(latent_dim=latent_dim, projection_dim=projection_dim),
        augment="default",
    ).fit(windows, max_epochs=epochs, batch_size=batch_size, lr=lr,
          weight_decay=weight_decay, random_state=seed, accelerator=device,
          num_workers=0, drop_last=False, exhaust_sampler=True, deterministic="warn")
    raw = encoder.transform(windows, batch_size=4096, standardize=False).astype(np.float32)
    return encoder.model, raw, encoder.history["train_loss"]


class LiverGraphSAGE(nn.Module):
    """The paper's projected one-hot input, two mean-aggregation SAGE layers."""

    def __init__(self, in_channels=6, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(n_layers)
        ])
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.proj(x))
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return self.head(x)


def fit_liver_gnn(
    training_graphs, *, loss_objective, seed=0, hidden_dim=64, n_layers=2,
    dropout=0.2, epochs=120, lr=5e-3, weight_decay=1e-4, device="cpu",
):
    """Study objective mapping; model fitting uses the public node trainer."""
    if loss_objective not in {"roi_mean", "patch_sum"}:
        raise ValueError("loss_objective must be 'roi_mean' or 'patch_sum'.")
    seed_everything(seed)
    return mt.nn.train_node_classifier(
        training_graphs, [int(g.y) for g in training_graphs],
        model_factory=LiverGraphSAGE,
        model_kwargs=dict(in_channels=training_graphs[0].x.shape[1],
                          hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout),
        n_classes=2, epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed,
        loss_reduction="roi" if loss_objective == "roi_mean" else "node",
        roi_class_weight_mode="node", class_weight="balanced", standardize=True,
        optimizer_factory=torch.optim.Adam, scheduler=None, device=device,
    )


def infected_probability(fit, graph):
    return float(fit.predict_proba(graph)[:, 1].mean())


def liver_integrated_gradients(fit, graph, *, steps=50):
    """Signed mean-infected-logit IG at the training-standardized zero baseline."""
    result = mt.nn.node_integrated_gradients(
        fit.model, graph, mean=fit.mean, scale=fit.scale, target_class=1,
        n_steps=steps, internal_batch_size=1, return_details=True,
    )
    return result.signed, result.node_importance, float(result.convergence_delta.item())


def show_patch_map(table, title, *, cluster="ecm_cluster", outline=None):
    """Categorical patch map, retaining the full grid and white background."""
    gx = ((table["x"].to_numpy() - 5) // 10).astype(int)
    gy = ((table["y"].to_numpy() - 5) // 10).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    label = table[cluster].to_numpy(dtype=int)
    signal = label >= 0
    grid[gy[signal], gx[signal]] = label[signal]
    _, ax = plt.subplots(figsize=(7, 7))
    mt.pl.patch_map(table, cluster, patch_size=10, ax=ax, title=title,
                    categorical=True, palette=dict(enumerate(plt.get_cmap("tab10").colors[:6])),
                    background="white", vmin=-0.5, vmax=5.5)
    ax.set_axis_off()
    scalar = ScalarMappable(norm=Normalize(-.5, 5.5), cmap=ListedColormap(plt.get_cmap("tab10").colors[:6]))
    plt.colorbar(scalar, ax=ax, ticks=range(6), shrink=0.65, label="ECM cluster")
    if outline is not None:
        ax.contour(np.nan_to_num(grid, nan=-1) == outline, levels=[0.5], colors="black", linewidths=0.5)
    plt.show()

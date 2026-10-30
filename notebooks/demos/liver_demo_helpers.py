"""Selected liver CNN, GraphSAGE loss objectives, IG and simple map display.

Public PyTorch/PyG/Captum express settings that Mantpy 1.0.2 does not expose.
The notebook keeps cohort preparation, clustering and graph analysis visible.
"""
import random

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv
from tqdm.auto import trange


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


def augment_patches(x):
    for dims in ((-1,), (-2,)):
        keep = torch.rand(len(x), device=x.device) < 0.5
        x = torch.where(keep[:, None, None, None], x.flip(dims), x)
    keep = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(keep[:, None, None, None], x.transpose(-1, -2), x)
    contrast = 0.7 + 0.6 * torch.rand(len(x), 1, 1, 1, device=x.device)
    brightness = -0.1 + 0.2 * torch.rand(len(x), 1, 1, 1, device=x.device)
    return (x * contrast + brightness + 0.05 * torch.randn_like(x)).clamp(0, 1)


def fit_liver_cnn(
    windows, *, seed=0, latent_dim=96, projection_dim=64, temperature=0.1,
    epochs=200, batch_size=1024, lr=3e-4, weight_decay=1e-4, device="cpu",
):
    """One fresh selected CNN; return raw latent features without z-scoring."""
    seed_everything(seed)
    data = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    cnn = LiverCNN(latent_dim).to(device)
    projection = nn.Sequential(
        nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        nn.Linear(latent_dim, projection_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(projection.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    progress = trange(epochs, desc="Shared liver CNN")
    for _ in progress:
        cnn.train()
        projection.train()
        batch_losses = []
        for batch in loader:
            batch = batch.to(device)
            views = torch.cat([augment_patches(batch), augment_patches(batch)])
            latent = F.normalize(projection(cnn(views)), dim=1)
            similarity = latent @ latent.T / temperature
            similarity.fill_diagonal_(-1e9)
            targets = (torch.arange(len(latent), device=device) + len(batch)) % len(latent)
            loss = F.cross_entropy(similarity, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        losses.append(float(np.mean(batch_losses)))
        progress.set_postfix(loss=f"{losses[-1]:.3f}")
    cnn.eval()
    with torch.no_grad():
        embedding = np.concatenate([
            cnn(batch.to(device)).cpu().numpy()
            for batch in DataLoader(data, batch_size=4096, shuffle=False)
        ]).astype(np.float32)
    return cnn, embedding, losses


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
    """Fit one mouse-held-out model using a visible ROI or patch loss objective."""
    if loss_objective not in {"roi_mean", "patch_sum"}:
        raise ValueError("loss_objective must be 'roi_mean' or 'patch_sum'.")
    seed_everything(seed)
    features = torch.cat([g.x for g in training_graphs], dim=0).numpy()
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    mean_t = torch.tensor(mean, device=device)
    scale_t = torch.tensor(scale, device=device)
    batch = Batch.from_data_list(training_graphs).to(device)
    x = (batch.x - mean_t) / scale_t
    graph_labels = torch.tensor([int(g.y) for g in training_graphs], device=device)
    node_labels = graph_labels[batch.batch]
    counts = np.bincount(graph_labels.cpu().numpy(), minlength=2).astype(float)
    weight = torch.tensor(counts.sum() / (counts + 1e-6), device=device, dtype=torch.float32)
    weight = weight / weight.sum() * 2
    net = LiverGraphSAGE(x.shape[1], hidden_dim, n_layers, dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    net.train()
    for _ in trange(epochs, desc=f"GraphSAGE · {loss_objective}"):
        optimizer.zero_grad(set_to_none=True)
        per_node = F.cross_entropy(net(x, batch.edge_index), node_labels, weight=weight, reduction="none")
        if loss_objective == "roi_mean":
            total = torch.zeros(len(training_graphs), device=device)
            total.index_add_(0, batch.batch, per_node)
            size = torch.bincount(batch.batch, minlength=len(training_graphs)).float()
            loss = (total / size).mean()
        else:
            loss = per_node.sum() / weight[node_labels].sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    net.eval()
    return {"model": net, "mean": mean_t, "scale": scale_t,
            "loss": losses, "objective": loss_objective, "device": device}


def infected_probability(fit, graph):
    with torch.no_grad():
        x = (graph.x.to(fit["device"]) - fit["mean"]) / fit["scale"]
        logits = fit["model"](x, graph.edge_index.to(fit["device"]))
        return float(logits.softmax(dim=1)[:, 1].mean().cpu())


def liver_integrated_gradients(fit, graph, *, steps=50):
    """Signed Captum IG to mean infected logit; zero in training-standardized space."""
    from captum.attr import IntegratedGradients
    model = fit["model"]
    model.eval()
    edges = graph.edge_index.to(fit["device"])
    x = ((graph.x.to(fit["device"]) - fit["mean"]) / fit["scale"]).detach()

    def forward(batch):
        return torch.stack([model(item, edges)[:, 1].mean() for item in batch])

    attribution, delta = IntegratedGradients(forward).attribute(
        x.unsqueeze(0), baselines=torch.zeros_like(x).unsqueeze(0),
        n_steps=steps, internal_batch_size=1, return_convergence_delta=True,
    )
    signed = attribution.squeeze(0).detach().cpu().numpy()
    return signed, np.abs(signed).sum(axis=1), float(delta.detach().cpu().item())


def show_patch_map(table, title, *, cluster="ecm_cluster", outline=None):
    """Categorical patch map, retaining the full grid and white background."""
    gx = ((table["x"].to_numpy() - 5) // 10).astype(int)
    gy = ((table["y"].to_numpy() - 5) // 10).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    label = table[cluster].to_numpy(dtype=int)
    signal = label >= 0
    grid[gy[signal], gx[signal]] = label[signal]
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:6])
    cmap.set_bad("white")
    _, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
    ax.set(title=title)
    ax.set_axis_off()
    plt.colorbar(image, ax=ax, ticks=range(6), shrink=0.65, label="ECM cluster")
    if outline is not None:
        ax.contour(np.nan_to_num(grid, nan=-1) == outline, levels=[0.5], colors="black", linewidths=0.5)
    plt.show()

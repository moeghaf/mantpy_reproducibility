"""Selected shared CNN and a repeated patch-map plot for the Figure 2 demo.

The architecture/recipe follows the frozen architecture_09 configuration.
Public PyTorch is used because Mantpy 1.0.2 does not expose CNN width/readout.
Fresh initialisation is intentional: this demo does not load paper checkpoints.
"""
import random

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange


class SharedCNN(nn.Module):
    """Stride-2 Conv/BatchNorm/ReLU blocks, average pooling and linear readout."""

    def __init__(self, in_channels=1, blocks=3, width=32, latent_dim=96):
        super().__init__()
        layers = []
        channels = in_channels
        for block in range(blocks):
            output_channels = min(width * 2 ** block, 128)
            layers.extend([
                nn.Conv2d(channels, output_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            ])
            channels = output_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.readout = nn.Linear(channels, latent_dim)

    def forward(self, x):
        return self.readout(self.pool(self.conv(x)).flatten(1))


def augment_patches(x):
    """Paper default: flips/transpose, contrast 0.7–1.3, brightness ±0.1, noise 0.05."""
    for dimensions in ((-1,), (-2,)):
        mask = torch.rand(len(x), device=x.device) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(dimensions), x)
    mask = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(mask[:, None, None, None], x.transpose(-1, -2), x)
    contrast = 0.7 + 0.6 * torch.rand(len(x), 1, 1, 1, device=x.device)
    brightness = -0.1 + 0.2 * torch.rand(len(x), 1, 1, 1, device=x.device)
    return (x * contrast + brightness + 0.05 * torch.randn_like(x)).clamp(0, 1)


def fit_shared_cnn(
    windows, *, seed=0, blocks=3, width=32, latent_dim=96, projection_dim=64,
    temperature=0.2, epochs=200, batch_size=1024, lr=1e-3,
    weight_decay=1e-5, device="cpu",
):
    """Fit one selected CNN; return encoder, once-standardised features and loss."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    data = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    batch_size = min(batch_size, len(data))
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
        drop_last=len(data) % batch_size == 1,
    )
    cnn = SharedCNN(
        in_channels=data.shape[1], blocks=blocks, width=width, latent_dim=latent_dim,
    ).to(device)
    projection = nn.Sequential(
        nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, projection_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(projection.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    progress = trange(epochs, desc="Shared CNN")
    for epoch in progress:
        batch_losses = []
        cnn.train()
        projection.train()
        for batch in loader:
            batch = batch.to(device)
            views = torch.cat([augment_patches(batch), augment_patches(batch)])
            z = F.normalize(projection(cnn(views)), dim=1)
            similarity = z @ z.T / temperature
            similarity.fill_diagonal_(-1e9)
            targets = (torch.arange(len(z), device=device) + len(batch)) % len(z)
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
        raw_features = np.concatenate([
            cnn(batch.to(device)).cpu().numpy()
            for batch in DataLoader(data, batch_size=4096, shuffle=False)
        ])
    features = StandardScaler().fit_transform(raw_features).astype(np.float32)
    return cnn, features, losses


def plot_patch_labels(patches, labels, title):
    """One categorical map on the original patch lattice; no label matching."""
    shape = tuple(patches.uns["image_ecm_patches"]["patch_grid_shape"])
    grid = np.full(shape, np.nan)
    grid[patches.obs["grid_y"].to_numpy(), patches.obs["grid_x"].to_numpy()] = labels
    palette = ListedColormap(plt.get_cmap("tab10").colors[:5])
    _, ax = plt.subplots(figsize=(8, 8))
    artist = ax.imshow(grid, cmap=palette, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(artist, ax=ax, ticks=range(5), shrink=0.6, label="Cluster")
    plt.show()

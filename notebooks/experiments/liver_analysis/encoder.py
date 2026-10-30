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


def _selected_modules(latent_dim, projection_dim):
    # The original parameterized training module constructs its default
    # fixed3_pool2 encoder and 64-wide head before replacing both. Preserve
    # those seeded initialization draws using the same public layers.
    placeholder = nn.Sequential(
        nn.Conv2d(6, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64 * 2 * 2, latent_dim),
    )
    placeholder_head = nn.Sequential(
        nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 64),
    )
    del placeholder, placeholder_head
    cnn = LiverCNN(latent_dim)
    projection = nn.Sequential(
        nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        nn.Linear(latent_dim, projection_dim),
    )
    return cnn, projection


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
    cnn, projection = _selected_modules(latent_dim, projection_dim)
    cnn, projection = cnn.to(device), projection.to(device)
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



"""Selected liver CNN, GraphSAGE loss objectives, IG and simple map display.

Mantpy fits the selected local architecture and exports its raw backbone features.
The notebook keeps cohort preparation, clustering and graph analysis visible.
"""
import random

import mantpy as mt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
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


def selected_backbone(latent_dim=96):
    """Consume original fixed3_pool2/default-head draws before selected initialization."""
    placeholder = nn.Sequential(
        nn.Conv2d(6, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64 * 2 * 2, latent_dim),
    )
    placeholder_head = projection_head(latent_dim, 64)
    del placeholder, placeholder_head
    return LiverCNN(latent_dim)


def fit_liver_cnn(
    windows, *, seed=0, latent_dim=96, projection_dim=64, temperature=0.1,
    epochs=200, batch_size=1024, lr=3e-4, weight_decay=1e-4, device="cpu",
):
    """Selected public trainer; export raw backbone latents in original row order."""
    seed_everything(seed)
    encoder = mt.nn.PatchEncoder(
        latent_dim=latent_dim, temperature=temperature,
        backbone=selected_backbone, backbone_kwargs=dict(latent_dim=latent_dim),
        projection=projection_head,
        projection_kwargs=dict(latent_dim=latent_dim, projection_dim=projection_dim),
        augment="default",
    ).fit(windows, max_epochs=epochs, batch_size=batch_size, lr=lr,
          weight_decay=weight_decay, random_state=seed, accelerator=device,
          num_workers=0, drop_last=False, exhaust_sampler=True, deterministic="warn")
    raw = encoder.transform(windows, batch_size=4096, standardize=False).astype(np.float32)
    return encoder.model, raw, encoder.history["train_loss"]



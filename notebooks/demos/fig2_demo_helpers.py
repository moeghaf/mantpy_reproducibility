"""Study architecture and display for the Figure 2 demo.

Training uses Mantpy's configurable PatchEncoder; the original architecture
and its initialization order stay explicit in this study adapter.
"""
import mantpy as mt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn


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


def projection_head(latent_dim=96, projection_dim=64):
    """Paper projection layers, initialized after the selected backbone."""
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                         nn.Linear(latent_dim, projection_dim))


def fit_shared_cnn(
    windows, *, seed=0, blocks=3, width=32, latent_dim=96, projection_dim=64,
    temperature=0.2, epochs=200, batch_size=1024, lr=1e-3,
    weight_decay=1e-5, device="cpu",
):
    """Selected public trainer; retain the original once-fitted scaling."""
    encoder = mt.nn.PatchEncoder(
        latent_dim=latent_dim, temperature=temperature, backbone=SharedCNN,
        backbone_kwargs=dict(in_channels=windows.shape[1], blocks=blocks,
                             width=width, latent_dim=latent_dim),
        projection=projection_head,
        projection_kwargs=dict(latent_dim=latent_dim, projection_dim=projection_dim),
        augment="default",
    ).fit(windows, max_epochs=epochs, batch_size=batch_size, lr=lr,
          weight_decay=weight_decay, random_state=seed, accelerator=device,
          num_workers=0, drop_last=None, exhaust_sampler=True, deterministic="warn")
    raw = encoder.transform(windows, batch_size=4096, standardize=False)
    features = StandardScaler().fit_transform(raw).astype(np.float32)
    return encoder.model, features, encoder.history["train_loss"]


def plot_patch_labels(patches, labels, title):
    _, ax = plt.subplots(figsize=(8, 8))
    mt.pl.patch_map(patches, labels, ax=ax, title=title, categorical=True,
                    palette=dict(enumerate(plt.get_cmap("tab10").colors[:5])),
                    vmin=-0.5, vmax=4.5, crop=False, background="white")
    ax.set_axis_off()
    scalar = ScalarMappable(norm=Normalize(-.5, 4.5), cmap=ListedColormap(plt.get_cmap("tab10").colors[:5]))
    plt.colorbar(scalar, ax=ax, ticks=range(5), shrink=0.6, label="Cluster")
    plt.show()

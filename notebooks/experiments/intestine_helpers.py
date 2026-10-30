"""Input checks, repeated training and export for the full intestine experiment."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import time
from urllib.request import urlopen
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange


HERE = Path(__file__).resolve().parent
PIXEL_UM = 0.37744
HBM735_BUNDLE = "mantpy-hbm735-intestine.tar.gz"
HBM735_SHA256 = "be40fe0d90cc2a7512f3a5d66b454f703f39157fd0dc7472640db30fee087354"
HBM735_BYTES = 33442165


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")
    temporary.replace(path)


def environment():
    versions = {}
    for name in ("mantpy", "torch", "numpy", "scipy", "scikit-learn",
                 "anndata", "pybanksy", "utag", "cellcharter", "torchgmm", "squidpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return {"versions": versions, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def seed_everything(seed, threads=4):
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
    """Preserve the paper's per-stage OpenMP limits where available."""
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


def fetch_hbm735(cache_dir=None):
    """Fetch only the three named, checksum-verified members of the public bundle."""
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
    import mantpy as mt
    import tifffile
    bundle = mt.datasets.coliv_intestine(cache_dir=cache_dir, progressbar=False)
    first = {
        "image": np.asarray(bundle.image),
        "reference": bundle.reference.reset_index()[["patch_id", "grid_y", "grid_x", "layer"]],
        "class_names": list(bundle.layer_names),
        "accession": "HBM926.SHNZ.594",
    }
    second_root = fetch_hbm735(cache_dir)
    with np.load(second_root / "annotation.npz", allow_pickle=False) as annotation:
        second = {
            "image": tifffile.imread(second_root / "image.tiff"),
            "annotation": annotation["labels"],
            "class_names": annotation["class_names"].tolist(),
            "accession": "HBM735.PSTF.274",
        }
    return {"intestine1": first, "intestine2": second}


class StageStore:
    """Resume complete fits only when inputs, settings, code and software agree."""

    def __init__(self, root, image, input_hash):
        self.root = Path(root) / image
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "image": image, "input_sha256": input_hash,
            "helper_sha256": sha256(__file__),
            "selected_sha256": sha256(HERE / "intestine_selected.json"),
            "requirements_sha256": sha256(HERE / "intestine_requirements.txt"),
            "environment": environment(),
        }
        self.timings = []

    def _folder(self, arm, method, seed):
        if any(not word.replace("_", "").isalnum() for word in (arm, method)):
            raise ValueError("Invalid stage name.")
        return self.root / arm / method / f"seed_{int(seed)}"

    def _request(self, config):
        return {**self.identity, "config": config}

    def load(self, arm, method, seed, config):
        folder = self._folder(arm, method, seed)
        marker = folder / "completion.json"
        if not marker.exists():
            return None
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["request"] != self._request(config):
            raise RuntimeError(f"Inputs or settings changed for {folder}. Use a new RESULTS directory.")
        for name, digest in record["files"].items():
            if sha256(folder / name) != digest:
                raise RuntimeError(f"A completed result is corrupt: {folder / name}")
        with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        self.timings.append({"arm": arm, "method": method, "seed": seed,
                             "seconds": record["seconds"], "reused": True})
        print(f"Reused {self.root.name}/{arm}/{method}/seed {seed}", flush=True)
        return arrays

    def save(self, arm, method, seed, config, seconds, metadata=None, **arrays):
        if not all(np.isfinite(value).all() for value in arrays.values()
                   if np.asarray(value).dtype.kind in "fci"):
            raise ValueError("Nonfinite fit output.")
        folder = self._folder(arm, method, seed)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "arrays.npz", **arrays)
        write_json(folder / "completion.json", {
            "state": "complete", "request": self._request(config),
            "seconds": seconds, "metadata": metadata or {},
            "files": {"arrays.npz": sha256(folder / "arrays.npz")},
        })
        self.timings.append({"arm": arm, "method": method, "seed": seed,
                             "seconds": seconds, "reused": False})
        print(f"Completed {self.root.name}/{arm}/{method}/seed {seed}: {seconds:.1f} s", flush=True)
        return arrays

    def save_timing(self):
        table = pd.DataFrame(self.timings)
        table.to_csv(self.root / "timings.csv", index=False)
        return table


class SelectedCNN(nn.Module):
    """The paper's parameterised stride-2 convolutional encoder."""

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


def augment_patches(values):
    """Default flips/transpose, contrast .7–1.3, brightness ±.1 and noise .05."""
    for dimensions in ((-1,), (-2,)):
        selected = torch.rand(len(values), device=values.device) < .5
        values = torch.where(selected[:, None, None, None], values.flip(dimensions), values)
    selected = torch.rand(len(values), device=values.device) < .5
    values = torch.where(selected[:, None, None, None], values.transpose(-1, -2), values)
    contrast = .7 + .6 * torch.rand(len(values), 1, 1, 1, device=values.device)
    brightness = -.1 + .2 * torch.rand(len(values), 1, 1, 1, device=values.device)
    return (values * contrast + brightness + .05 * torch.randn_like(values)).clamp(0, 1)


def fit_selected_cnn(windows, architecture, fixed, seed, device="cuda", progress=True):
    """Train the frozen contrastive recipe with public PyTorch, without searches."""
    seed_everything(seed, threads=2)
    data = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    batch_size = min(int(fixed["batch"]), len(data))
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=len(data) % batch_size == 1,
        pin_memory=device.startswith("cuda"),
    )
    latent, projection = int(fixed["latent"]), int(fixed["projection"])
    # The original parameterised wrapper initialised a default encoder/head
    # before replacing them. Preserve those RNG draws with public modules.
    placeholder = SelectedCNN(in_channels=data.shape[1], latent_dim=latent)
    placeholder_head = nn.Sequential(nn.Linear(latent, latent), nn.ReLU(), nn.Linear(latent, 64))
    del placeholder, placeholder_head
    cnn = SelectedCNN(**architecture, latent_dim=latent, in_channels=data.shape[1]).to(device)
    head = nn.Sequential(nn.Linear(latent, latent), nn.ReLU(), nn.Linear(latent, projection)).to(device)
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(head.parameters()),
        lr=fixed["lr"], weight_decay=fixed["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fixed["epochs"])
    losses, updates = [], 0
    for epoch in trange(fixed["epochs"], desc=f"CNN seed {seed}", disable=not progress):
        cnn.train()
        head.train()
        batch_losses = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            views = torch.cat([augment_patches(batch), augment_patches(batch)])
            z = F.normalize(head(cnn(views)), dim=1)
            similarity = z @ z.T / fixed["temperature"]
            similarity.fill_diagonal_(-1e9)
            targets = (torch.arange(len(z), device=device) + len(batch)) % len(z)
            loss = F.cross_entropy(similarity, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        losses.append(float(np.mean(batch_losses)))
    cnn.eval()
    with torch.no_grad():
        raw = np.concatenate([
            cnn(batch.to(device)).cpu().numpy()
            for batch in DataLoader(data, batch_size=4096, shuffle=False)
        ])
    scaler = StandardScaler()
    scaler.partial_fit(raw)
    features = scaler.transform(raw).astype(np.float32) if fixed["standardize"] else raw.astype(np.float32)
    if features.shape != (len(windows), latent) or not np.isfinite(losses).all():
        raise ValueError("Incomplete CNN fit.")
    return cnn, features, np.asarray(losses), scaler, updates


def build_selected_graph(patches, neighbours, continuity=.25, control=None, seed=0):
    """Public topology, then the selected log(distance) and orientation attributes."""
    import mantpy as mt
    mt.gr.build_patch_graph(
        patches, k=neighbours, continuity_thresh=continuity, continuity_samples=6,
        edge_features=("log_distance", "sin_angle2", "cos_angle2"),
        standardize_edges=False, key_added="patch",
    )
    graph = patches.uns["patch"]
    edges = np.asarray(graph["edge_index"], dtype=np.int64).copy()
    xy = np.asarray(patches.obsm["spatial"], dtype=np.float64)
    delta = xy[edges[1]] - xy[edges[0]]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    attributes = np.column_stack([np.log(distance), np.sin(2 * angle), np.cos(2 * angle)])
    deviation = attributes.std(0)
    deviation[deviation == 0] = 1
    attributes = ((attributes - attributes.mean(0)) / deviation).astype(np.float32)
    if control == "shuffled_edges":
        permutation = np.random.default_rng(10_000 + seed).permutation(edges.shape[1])
        edges = np.stack([edges[0], edges[1, permutation]])
    elif control == "no_edges":
        edges, attributes = edges[:, :0], attributes[:0]
    elif control is not None:
        raise ValueError(control)
    graph["edge_index"], graph["edge_attr"] = edges, attributes
    graph["edge_feature_names"] = ["log_distance", "sin_angle2", "cos_angle2"]
    graph["params"].update({
        "edge_distance": "log", "edge_orientation": True,
        "edge_continuity": False, "standardize_edges": True, "control": control,
    })
    return int(edges.shape[1])


@contextmanager
def observe_graphmae_training(epochs):
    """Observe the original backward scalars and AdamW steps without altering either."""
    updates, losses = [], []
    original_step = torch.optim.AdamW.step
    original_backward = torch.Tensor.backward

    def measured_step(optimizer, *args, **kwargs):
        value = original_step(optimizer, *args, **kwargs)
        updates.append(1)
        return value

    def measured_backward(loss, *args, **kwargs):
        scalar = float(loss.detach().cpu())
        if not np.isfinite(scalar):
            raise ValueError("GraphMAE produced a nonfinite reconstruction loss.")
        losses.append(scalar)
        return original_backward(loss, *args, **kwargs)

    with patch.object(torch.optim.AdamW, "step", measured_step), patch.object(
        torch.Tensor, "backward", measured_backward,
    ):
        yield {"losses": losses, "updates": updates}
    if len(losses) != epochs or len(updates) != epochs:
        raise RuntimeError(
            f"Incomplete GraphMAE fit: {len(losses)} losses and "
            f"{len(updates)} updates; expected {epochs} of each."
        )


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pixel_annotation(key, loaded, shape):
    if key == "intestine2":
        annotation = np.asarray(loaded["annotation"], dtype=np.uint8)
        if annotation.shape != tuple(shape):
            raise ValueError("Image and annotation shapes differ.")
        return annotation
    annotation = np.zeros(shape, dtype=np.uint8)
    for row in loaded["reference"].itertuples():
        annotation[row.grid_y * 32:(row.grid_y + 1) * 32,
                   row.grid_x * 32:(row.grid_x + 1) * 32] = row.layer + 1
    return annotation


def ari_from_contingency(table):
    table = np.asarray(table, dtype=np.int64)
    n = table.sum()
    if n < 2:
        return float("nan")
    def pairs(values):
        return (values.astype(np.float64) * (values - 1) / 2).sum()
    observed, rows, columns = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * columns / (n * (n - 1) / 2)
    denominator = (rows + columns) / 2 - expected
    return 1.0 if denominator == 0 else float((observed - expected) / denominator)


def macro_f1_from_contingency(table):
    matrix = np.asarray(table[:, :-1], dtype=float)
    rows, columns = linear_sum_assignment(-matrix)
    score = np.zeros(table.shape[0])
    for row, column in zip(rows, columns):
        denominator = table[row].sum() + matrix[:, column].sum()
        score[row] = 2 * matrix[row, column] / denominator if denominator else 0
    return float(score.mean())


class PixelScorer:
    """Count annotated pixels once, then score every method on identical support."""

    def __init__(self, patches, annotation, k):
        self.patches, self.annotation, self.k = patches, annotation, k
        self.total = np.bincount(annotation.ravel(), minlength=k + 1)[1:k + 1]
        self.coordinates = patches.obs[["grid_y", "grid_x"]].to_numpy(int)
        self.counts = np.stack([
            np.bincount(annotation[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32].ravel(),
                        minlength=k + 1)[1:k + 1]
            for y, x in self.coordinates
        ])
        self.covered = self.counts.sum(0)
        self.coverage = float(self.covered.sum() / self.total.sum())

    def contingency(self, labels, pixel=False, full=False):
        labels = np.asarray(labels)
        matrix = np.zeros((self.k, self.k + 1), dtype=np.int64)
        if pixel:
            if labels.shape != self.annotation.shape:
                raise ValueError("Pixel predictions must match the native image.")
            for y, x in self.coordinates:
                annotation = self.annotation[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32].ravel()
                predicted = labels[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32].ravel()
                mask = annotation > 0
                matrix[:, :self.k] += np.bincount(
                    (annotation[mask].astype(np.int64) - 1) * self.k + predicted[mask],
                    minlength=self.k ** 2,
                ).reshape(self.k, self.k)
        else:
            if labels.shape != (len(self.coordinates),) or np.any(labels < 0):
                raise ValueError("Every foreground patch requires a cluster label.")
            labels = np.searchsorted(np.unique(labels), labels)
            if np.unique(labels).size > self.k:
                raise ValueError("Predictions exceed the fixed number of clusters.")
            for cluster in range(self.k):
                matrix[:, cluster] = self.counts[labels == cluster].sum(0)
        if full:
            matrix[:, -1] = self.total - self.covered
        return matrix

    def score(self, labels, pixel=False):
        common = self.contingency(labels, pixel=pixel)
        full = self.contingency(labels, pixel=pixel, full=True)
        return {
            "ARI": ari_from_contingency(common),
            "macro_F1": macro_f1_from_contingency(common),
            "full_area_ARI": ari_from_contingency(full),
            "full_area_macro_F1": macro_f1_from_contingency(full),
            "coverage": self.coverage, "annotated_pixels": int(self.total.sum()),
            "covered_annotated_pixels": int(self.covered.sum()),
        }


def plot_patch_labels(patches, labels, title, k, save=None):
    shape = tuple(patches.uns["image_ecm_patches"]["patch_grid_shape"])
    grid = np.full(shape, np.nan)
    grid[patches.obs["grid_y"].to_numpy(int), patches.obs["grid_x"].to_numpy(int)] = labels
    figure, axis = plt.subplots(figsize=(6, 5))
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:k])
    artist = axis.imshow(grid, cmap=cmap, vmin=-.5, vmax=k - .5, interpolation="nearest")
    axis.set(title=title)
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, ticks=range(k), label="Cluster", shrink=.75)
    figure.tight_layout()
    if save:
        figure.savefig(save, dpi=160)
    plt.show()


def score_summary(rows):
    table = pd.DataFrame(rows)
    metrics = ["ARI", "macro_F1", "full_area_ARI", "full_area_macro_F1"]
    summary = table.groupby(["image", "arm", "method"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary["seeds"] = table.groupby(["image", "arm", "method"]).size()
    return table, summary.reset_index()

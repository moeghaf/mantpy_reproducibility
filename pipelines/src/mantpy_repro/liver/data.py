"""Liver cohort, pixel normalisation, patches, background removal and the label-free support.

Steps of the paper:

1. pixels: arcsinh(x / 1), divided by each marker's 99.5th percentile over all pixels of the 24 ROIs,
   clipped to [0, 1];
2. patches: non-overlapping s x s grid from the image origin; 12 features per patch = the mean of each
   marker, then its standard deviation (population formula);
3. background: features standardised over all patches, two-component K-means; the component with the
   lower standardised mean intensity is background and the rest are ECM signal patches;
4. support: the 5-pixel signal patches are the anchors. An anchor is covered at patch size s when the
   s-pixel patch containing its centre is a signal patch; coverage(s) is the covered fraction.

No condition, genotype or mouse label is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..store import unique_suffix
from .config import ANCHOR_SIZE, MARKERS, PATCH_SIZES

COFACTOR = 1.0
PERCENTILE = 99.5
BACKGROUND_KMEANS = {"n_clusters": 2, "init": "k-means++", "n_init": 10, "random_state": 0}
N_FEATURES = 2 * len(MARKERS)


def load_cohort(cache_dir: str | None = None, crop: int | None = None):
    """Images ``{roi: (6, H, W) float32}`` and one metadata row per ROI (downloaded and cached on first use).

    ``crop`` keeps only a central square of that many pixels. It exists for quick installation
    checks (``--smoke``); results on a crop are not the paper's.
    """
    import mantpy as mt

    liver = mt.datasets.schistosoma_ecm(cache_dir=cache_dir, progressbar=False, verbose=False)
    if list(liver.ecm_names) != MARKERS:
        raise ValueError(f"Unexpected marker order {list(liver.ecm_names)}.")
    meta = (liver.sample_meta.reset_index(drop=True).rename(columns={"sample_id": "roi", "mouse_id": "mouse"})
            [["roi", "mouse", "condition", "genotype", "group"]].sort_values("roi").reset_index(drop=True))
    images = {roi: np.asarray(liver.images[roi], dtype=np.float32) for roi in meta.roi}
    if crop:
        for roi, image in images.items():
            top, left = (image.shape[1] - crop) // 2 // 20 * 20, (image.shape[2] - crop) // 2 // 20 * 20
            images[roi] = np.ascontiguousarray(image[:, top:top + crop, left:left + crop])
    return images, meta


def normalise(images: dict) -> tuple[dict, np.ndarray]:
    """Apply the cohort-pooled pixel transform; return the transformed images and the six marker bounds."""
    import mantpy as mt

    normalizer = mt.pp.ImageNormalizer(method="pooled_arcsinh_percentile", cofactor=COFACTOR, percentile=PERCENTILE,
                                       channel_names=MARKERS).fit({roi: images[roi] for roi in sorted(images)})
    return {roi: normalizer.transform(images[roi]) for roi in sorted(images)}, np.asarray(normalizer.bounds_)


def image_blocks(image: np.ndarray, size: int) -> np.ndarray:
    """All ``size`` x ``size`` patches of a (C, H, W) image as (N, C, size, size), row by row from the origin."""
    channels, height, width = image.shape
    rows, columns = height // size, width // size
    grid = image[:, :rows * size, :columns * size].reshape(channels, rows, size, columns, size).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(grid.reshape(rows * columns, channels, size, size), dtype=np.float32)


def block_features(blocks: np.ndarray) -> np.ndarray:
    """Mean of each marker, then its standard deviation, per patch (float32, as in the paper's runs)."""
    return np.concatenate([blocks.mean(axis=(2, 3)).astype(np.float32), blocks.std(axis=(2, 3)).astype(np.float32)], axis=1)


def signal_mask(features: np.ndarray) -> np.ndarray:
    """True for ECM signal patches (see step 3 of the module description)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    z = StandardScaler().fit_transform(features.astype(np.float32)).astype(np.float32)
    labels = KMeans(**BACKGROUND_KMEANS).fit(z).labels_.astype(int)
    intensity = np.array([z[labels == component, :len(MARKERS)].mean() for component in range(2)])
    return labels != int(np.argmin(intensity))


@dataclass
class PatchTable:
    """Every patch of one size over the whole cohort, ROI by ROI in sorted order, each ROI row by row."""

    size: int
    roi: np.ndarray          # index into the sorted ROI names
    gy: np.ndarray           # lattice row
    gx: np.ndarray           # lattice column
    features: np.ndarray     # (n, 12) float32
    signal: np.ndarray       # recomputed background split: True for ECM signal patches

    @property
    def x(self) -> np.ndarray:
        return self.gx * float(self.size) + self.size / 2.0

    @property
    def y(self) -> np.ndarray:
        return self.gy * float(self.size) + self.size / 2.0

    @property
    def signal_index(self) -> np.ndarray:
        return np.flatnonzero(self.signal)


def patch_table(normalised: dict, size: int) -> PatchTable:
    """Features and background split of every ``size``-pixel patch of the cohort."""
    roi, gy, gx, features = [], [], [], []
    for index, name in enumerate(sorted(normalised)):
        _, height, width = normalised[name].shape
        rows, columns = height // size, width // size
        lattice_y, lattice_x = np.divmod(np.arange(rows * columns), columns)
        roi.append(np.full(rows * columns, index, dtype=np.int16))
        gy.append(lattice_y.astype(np.int16))
        gx.append(lattice_x.astype(np.int16))
        features.append(block_features(image_blocks(normalised[name], size)))
    features = np.concatenate(features)
    return PatchTable(size, np.concatenate(roi), np.concatenate(gy), np.concatenate(gx), features, signal_mask(features))


def parent_rows(anchor_roi: np.ndarray, anchor_y: np.ndarray, anchor_x: np.ndarray, shapes: list[tuple[int, int]],
                size: int) -> np.ndarray:
    """Row, in the ``size``-pixel patch table, of the patch containing each anchor centre (-1 if outside its lattice)."""
    lattices = [(height // size, width // size) for height, width in shapes]
    offsets = np.concatenate([[0], np.cumsum([rows * columns for rows, columns in lattices])])
    rows = np.asarray([lattice[0] for lattice in lattices])[anchor_roi]
    columns = np.asarray([lattice[1] for lattice in lattices])[anchor_roi]
    py, px = (anchor_y // size).astype(np.int64), (anchor_x // size).astype(np.int64)
    inside = (py < rows) & (px < columns)
    return np.where(inside, offsets[anchor_roi] + py * columns + px, -1).astype(np.int64)


def project_to_anchors(labels: np.ndarray, signal_index: np.ndarray, n_rows: int, parents: np.ndarray) -> np.ndarray:
    """Label of each anchor = label of its parent patch when that patch is a signal patch, otherwise -1."""
    by_row = np.full(n_rows, -1, dtype=np.int16)
    by_row[signal_index] = labels
    return np.where(parents >= 0, by_row[np.maximum(parents, 0)], -1).astype(np.int16)


class Prepared:
    """Everything derived from the images before any model is fitted. Cached in ``root``."""

    def __init__(self, root: Path, cache_dir: str | None = None, crop: int | None = None):
        self.root, self.cache_dir, self.crop = Path(root), cache_dir, crop
        self.root.mkdir(parents=True, exist_ok=True)
        self._normalised: dict | None = None
        self._meta = None
        self._tables: dict[int, PatchTable] = {}
        self._parents: dict[int, np.ndarray] | None = None

    # Images and metadata
    def _load(self) -> None:
        import pandas as pd

        images, meta = load_cohort(self.cache_dir, self.crop)
        self._normalised, bounds = normalise(images)
        meta = meta.assign(height=[images[roi].shape[1] for roi in meta.roi], width=[images[roi].shape[2] for roi in meta.roi])
        path = self.root / "metadata.csv"
        if not path.exists():
            _atomic(self.root / "bounds.csv",
                    lambda temporary: pd.DataFrame({"marker": MARKERS, "bound": bounds}).to_csv(temporary, index=False))
            _atomic(path, lambda temporary: meta.to_csv(temporary, index=False))
        self._meta = meta

    @property
    def normalised(self) -> dict:
        if self._normalised is None:
            self._load()
        return self._normalised

    @property
    def meta(self):
        """One row per ROI: roi, mouse, condition, genotype, group, height, width (sorted by ROI)."""
        if self._meta is None:
            import pandas as pd

            path = self.root / "metadata.csv"
            if path.exists():
                self._meta = pd.read_csv(path)
            else:
                self._load()
        return self._meta

    @property
    def rois(self) -> list[str]:
        return self.meta.roi.tolist()

    @property
    def bounds(self) -> np.ndarray:
        """The six marker bounds (99.5th percentile of the arcsinh intensities), written with the metadata."""
        import pandas as pd

        if not (self.root / "metadata.csv").exists():
            self._load()
        return pd.read_csv(self.root / "bounds.csv").bound.to_numpy()

    # Patches
    def table(self, size: int) -> PatchTable:
        if size not in self._tables:
            path = self.root / f"patches_p{size:02d}.npz"
            if not path.exists():
                table = patch_table(self.normalised, size)
                _atomic(path, lambda temporary: np.savez_compressed(
                    temporary, roi=table.roi, gy=table.gy, gx=table.gx, features=table.features, signal=table.signal))
            with np.load(path, allow_pickle=False) as saved:
                self._tables[size] = PatchTable(size, saved["roi"], saved["gy"], saved["gx"], saved["features"], saved["signal"])
        return self._tables[size]

    def blocks(self, size: int, rows: np.ndarray) -> np.ndarray:
        """The six-channel pixel blocks of the patches in ``rows`` of the ``size``-pixel table (the CNN input)."""
        table, out, start = self.table(size), [], 0
        chosen = np.zeros(len(table.roi), dtype=bool)
        chosen[rows] = True
        if not np.array_equal(np.flatnonzero(chosen), rows):
            raise ValueError("rows must be sorted and unique.")
        for name in self.rois:
            blocks = image_blocks(self.normalised[name], size)
            out.append(blocks[chosen[start:start + len(blocks)]])
            start += len(blocks)
        return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)

    # Support
    def parents(self, size: int) -> np.ndarray:
        """Parent row at ``size`` pixels of every anchor (5-pixel signal patch)."""
        if self._parents is None:
            path = self.root / "anchors.npz"
            if not path.exists():
                fine = self.table(ANCHOR_SIZE)
                keep = fine.signal
                shapes = list(zip(self.meta.height, self.meta.width))
                arrays = {f"p{s:02d}": parent_rows(fine.roi[keep].astype(np.int64), fine.y[keep], fine.x[keep], shapes, s)
                          for s in PATCH_SIZES}
                _atomic(path, lambda temporary: np.savez_compressed(temporary, **arrays))
            with np.load(path, allow_pickle=False) as saved:
                self._parents = {s: saved[f"p{s:02d}"] for s in PATCH_SIZES}
        return self._parents[size]

    def anchor_labels(self, labels: np.ndarray, size: int, signal_index: np.ndarray) -> np.ndarray:
        """Labels of the signal patches of ``size`` pixels carried to the anchors (-1 = anchor not covered)."""
        return project_to_anchors(labels, signal_index, len(self.table(size).roi), self.parents(size))

    def coverage(self, size: int) -> float:
        """Fraction of the anchors whose parent ``size``-pixel patch is a signal patch."""
        parents = self.parents(size)
        return float(np.mean((parents >= 0) & self.table(size).signal[np.maximum(parents, 0)]))

    def summary(self) -> dict:
        tables = {size: self.table(size) for size in PATCH_SIZES}
        return {"n_rois": len(self.rois), "n_mice": int(self.meta.mouse.nunique()),
                "bounds": dict(zip(MARKERS, self.bounds.round(6).tolist())),
                "n_patches": {size: int(len(table.roi)) for size, table in tables.items()},
                "n_signal": {size: int(table.signal.sum()) for size, table in tables.items()},
                "coverage": {size: round(self.coverage(size), 6) for size in PATCH_SIZES}}


def _atomic(path: Path, write) -> None:
    """Write through a temporary file so that an interrupted job leaves no half-written cache.

    The temporary name holds the host name and the process ID, so jobs on different nodes do not share it.
    """
    temporary = path.with_name(f"{path.stem}.{unique_suffix()}.part{path.suffix}")
    write(temporary)
    temporary.replace(path)

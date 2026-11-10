"""Images, annotations, ECM patches and the label-free support used to compare configurations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from urllib.request import urlopen

import numpy as np

from ..store import unique_suffix
from .config import ACCESSION, K_BY_IMAGE, PATCH_SIZES, paper_selection

_HBM735_BUNDLE = "mantpy-hbm735-intestine.tar.gz"
_HBM735_SHA256 = "be40fe0d90cc2a7512f3a5d66b454f703f39157fd0dc7472640db30fee087354"
_HBM735_BYTES = 33442165
_HBM735_URL = "https://zenodo.org/records/23740311/files/" + _HBM735_BUNDLE + "?download=1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_hbm735(cache_dir: str | None) -> Path:
    """Download (once) and verify the image-2 bundle from Zenodo.

    The archive is kept in ``datasets/hbm735_intestine`` of the cache directory. A node without
    internet access can use a cache that was filled on another machine.
    """
    root = Path(cache_dir or os.environ.get("MANTPY_CACHE") or Path.home() / ".cache" / "mantpy")
    root = root / "datasets" / "hbm735_intestine"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / _HBM735_BUNDLE
    if not (archive.exists() and archive.stat().st_size == _HBM735_BYTES and _sha256(archive) == _HBM735_SHA256):
        temporary = archive.with_name(archive.name + f".{unique_suffix()}.part")
        with urlopen(_HBM735_URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, 1 << 22)
        if temporary.stat().st_size != _HBM735_BYTES or _sha256(temporary) != _HBM735_SHA256:
            raise ValueError("The image-2 download does not match its recorded checksum.")
        temporary.replace(archive)
    out = root / ("extracted-" + _HBM735_SHA256[:12])
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if {m.name for m in members} != {"annotation.npz", "image.tiff", "manifest.json"}:
            raise ValueError("Unexpected files in the image-2 bundle.")
        for member in members:
            target = out / member.name
            if not target.exists():
                partial = target.with_name(target.name + f".{unique_suffix()}.part")
                with bundle.extractfile(member) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
                partial.replace(target)
    return out


@dataclass
class LoadedImage:
    name: str
    raw: np.ndarray              # native collagen IV image
    annotation: np.ndarray       # same shape; 0 = not annotated, 1..k = annotated layer
    class_names: list[str]
    k: int


def load_image(name: str, cache_dir: str | None = None, crop: int | None = None) -> LoadedImage:
    """Load one image and its manual annotation (downloaded and cached on first use).

    ``crop`` keeps only a central square of that many pixels. It exists for quick installation
    checks (``--smoke``); results on a crop are not the paper's.
    """
    import mantpy as mt

    if name == "intestine1":
        bundle = mt.datasets.coliv_intestine(cache_dir=cache_dir, progressbar=False)
        raw = np.asarray(bundle.image)
        annotation = np.zeros(raw.shape, dtype=np.uint8)
        for row in bundle.reference.reset_index().itertuples():      # 32-pixel annotation tiles
            annotation[row.grid_y * 32:(row.grid_y + 1) * 32, row.grid_x * 32:(row.grid_x + 1) * 32] = row.layer + 1
        names = [str(item) for item in bundle.layer_names]
    elif name == "intestine2":
        import tifffile

        root = _fetch_hbm735(cache_dir)
        raw = np.squeeze(tifffile.imread(root / "image.tiff"))
        with np.load(root / "annotation.npz", allow_pickle=False) as saved:
            annotation = np.asarray(saved["labels"], dtype=np.uint8)
            names = [str(item) for item in saved["class_names"].tolist()]
        if annotation.shape != raw.shape:
            raise ValueError("Image 2 and its annotation have different shapes.")
    else:
        raise KeyError(f"Unknown image {name!r}; expected one of {sorted(K_BY_IMAGE)}.")
    if crop:
        top, left = (raw.shape[0] - crop) // 2 // 64 * 64, (raw.shape[1] - crop) // 2 // 64 * 64
        raw, annotation = raw[top:top + crop, left:left + crop], annotation[top:top + crop, left:left + crop]
    return LoadedImage(name=name, raw=raw, annotation=annotation, class_names=names, k=K_BY_IMAGE[name])


def thresholds(name: str) -> dict:
    """Clip value and foreground thresholds frozen for the paper.

    The foreground threshold is a two-means threshold on the mean intensity of 32-pixel
    patches. It was fitted once and then kept fixed for every patch size.
    """
    record = paper_selection()["images"][name]
    return {"clip": record["clip"], "pixel": record["pixel_threshold"], "patch": record["patch_threshold"]}


def extract_patches(name: str, raw: np.ndarray, size: int, limits: dict):
    """ECM patches of one size, with the frozen 32-pixel foreground thresholds."""
    import mantpy as mt

    return mt.pp.image_ecm_patches(
        raw, channel_names=["ColIV"], patch_size=size, clip_percentile=99,
        pixel_thresholds=[limits["pixel"]], patch_thresholds=[limits["patch"]],
        feature_families=(),      # only the tiles and the grid are needed; skipping features is much faster
        sample_id=name, random_state=0,
    )


@dataclass
class Support:
    """Label-free support shared by all patch sizes of one image.

    ``rows[s]`` maps every 16-pixel node to the row of its parent patch at size ``s`` (-1 = none).
    ``common`` selects the 16-pixel nodes whose 32- and 64-pixel parents are both nodes; seed
    stability is measured on these anchors so that every patch size is compared on the same tissue.
    ``coverage[s]`` is the fraction of 16-pixel nodes whose parent at size ``s`` is a node.
    """

    rows: dict[int, np.ndarray]
    common: np.ndarray
    coverage: dict[int, float]
    n_nodes: dict[int, int]

    def project(self, labels: np.ndarray, size: int) -> np.ndarray:
        """Labels of the patches of ``size`` carried to the common 16-pixel anchors."""
        return np.asarray(labels)[self.rows[size][self.common]]


def build_support(grids: dict[int, tuple[np.ndarray, np.ndarray]], shape: tuple[int, int]) -> Support:
    """Compute the common anchors and coverage from the node grids of 16, 32 and 64 pixels."""
    height, width = shape
    lookup = {}
    for size, (gy, gx) in grids.items():
        table = np.full((height // size, width // size), -1, dtype=np.int64)
        table[gy, gx] = np.arange(len(gy))
        lookup[size] = table
    fine_y, fine_x = grids[16]
    rows = {16: np.arange(len(fine_y), dtype=np.int64)}
    for size in (32, 64):
        factor = size // 16
        py, px = fine_y // factor, fine_x // factor
        valid = (py < lookup[size].shape[0]) & (px < lookup[size].shape[1])
        index = np.full(len(fine_y), -1, dtype=np.int64)
        index[valid] = lookup[size][py[valid], px[valid]]
        rows[size] = index
    common = np.flatnonzero((rows[32] >= 0) & (rows[64] >= 0))
    coverage = {size: float(np.mean(rows[size] >= 0)) for size in PATCH_SIZES}
    return Support(rows=rows, common=common, coverage=coverage, n_nodes={s: len(grids[s][0]) for s in grids})


class Prepared:
    """Everything derived from one image before any model is fitted. Cached on disk."""

    def __init__(self, name: str, root: Path, cache_dir: str | None = None, crop: int | None = None):
        self.name, self.root, self.cache_dir, self.crop = name, Path(root), cache_dir, crop
        self.root.mkdir(parents=True, exist_ok=True)
        self._loaded: LoadedImage | None = None
        self._patches: dict[int, object] = {}
        self._support: Support | None = None

    # Raw inputs
    @property
    def loaded(self) -> LoadedImage:
        if self._loaded is None:
            self._loaded = load_image(self.name, self.cache_dir, self.crop)
        return self._loaded

    @property
    def k(self) -> int:
        return K_BY_IMAGE[self.name]

    @property
    def limits(self) -> dict:
        return thresholds(self.name)

    def scaled(self) -> np.ndarray:
        """The image clipped at the frozen 99th percentile and scaled to [0, 1]."""
        clip = np.float32(self.limits["clip"])
        return np.minimum(self.loaded.raw.astype(np.float32), clip) / clip

    # Patches
    def patches(self, size: int):
        """AnnData of the ECM patches of ``size`` pixels (image tiles in ``obsm['image_patches']``)."""
        if size not in self._patches:
            path = self.root / f"patches_p{size}.pkl"
            if path.exists():
                with path.open("rb") as stream:
                    self._patches[size] = pickle.load(stream)
            else:
                data = extract_patches(self.name, self.loaded.raw, size, self.limits)
                temporary = path.with_name(path.name + f".{unique_suffix()}.part")
                with temporary.open("wb") as stream:
                    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(path)
                self._patches[size] = data
        return self._patches[size]

    def grid(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        obs = self.patches(size).obs
        return obs["grid_y"].to_numpy(dtype=np.int64), obs["grid_x"].to_numpy(dtype=np.int64)

    @property
    def support(self) -> Support:
        if self._support is None:
            path = self.root / "support.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    self._support = Support(
                        rows={s: saved[f"rows_p{s}"] for s in PATCH_SIZES}, common=saved["common"],
                        coverage={s: float(saved[f"coverage_p{s}"]) for s in PATCH_SIZES},
                        n_nodes={s: int(saved[f"n_nodes_p{s}"]) for s in PATCH_SIZES},
                    )
            else:
                support = build_support({s: self.grid(s) for s in PATCH_SIZES}, self.loaded.raw.shape)
                arrays = {"common": support.common}
                for s in PATCH_SIZES:
                    arrays.update({f"rows_p{s}": support.rows[s], f"coverage_p{s}": support.coverage[s],
                                   f"n_nodes_p{s}": support.n_nodes[s]})
                temporary = path.with_name(f"support.{unique_suffix()}.part.npz")
                np.savez_compressed(temporary, **arrays)
                temporary.replace(path)
                self._support = support
        return self._support

    def scorer(self, size: int = 32):
        from .scoring import PixelScorer

        gy, gx = self.grid(size)
        return PixelScorer(gy, gx, size, self.loaded.annotation, self.k)

    def summary(self) -> dict:
        support = self.support
        return {"image": self.name, "accession": ACCESSION[self.name], "k": self.k, "thresholds": self.limits,
                "n_nodes": support.n_nodes, "coverage": support.coverage, "common_anchors": int(len(support.common))}

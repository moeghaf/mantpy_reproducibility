"""Liver cohort loading, normalization, patches and signal selection.

Steps follow the paper's analysis exactly:
1. pixels: arcsinh(x / 1), divided by the 99.5th percentile of each marker pooled over all pixels of the 24 ROIs, clipped to [0, 1];
2. patches: non-overlapping 10 x 10-px grid (10,000 per ROI); 12 features = mean and s.d. (ddof 0) of each marker;
3. background: features standardised over the cohort, 2-cluster K-means; the component with the lower standardised mean
   intensity is background.
"""
from __future__ import annotations

import mantpy as mt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MARKERS = ["Fibronectin", "Pan-laminin", "CollagenIV", "VersicanV0V1", "CollagenI", "HeparanSulfate"]
GROUPS = ["Naive_KO", "Naive_WT", "Infected_KO", "Infected_WT"]
MICE = {"Naive_KO": ["mouse_05", "mouse_06"], "Naive_WT": ["mouse_07", "mouse_08"],
        "Infected_KO": ["mouse_01", "mouse_02"], "Infected_WT": ["mouse_03", "mouse_04"]}
REPRESENTATIVE = {"Naive_KO": "roi_017", "Naive_WT": "roi_021", "Infected_KO": "roi_004", "Infected_WT": "roi_009"}
K = 6
PATCH = 10
COFACTOR = 1.0
PERCENTILE = 99.5
FEAT_COLS = [f"feat_{i}" for i in range(12)]  # means of the six markers, then their s.d.
KMEANS_BG = dict(n_clusters=2, init="k-means++", n_init=10, random_state=0)


def cohort(liver):
    """Images {roi: (6, H, W) float32, marker order MARKERS} and ROI metadata from Mantpy's liver loader."""
    if list(liver.ecm_names) != MARKERS:
        raise ValueError(f"unexpected marker order {list(liver.ecm_names)}")
    meta = (liver.sample_meta.reset_index(drop=True).rename(columns={"sample_id": "roi", "mouse_id": "mouse"})
            [["roi", "mouse", "condition", "genotype", "group"]].sort_values("roi").reset_index(drop=True))
    images = {r: np.asarray(liver.images[r], dtype=np.float32) for r in meta.roi}
    return images, meta


def load_cohort(cache_dir=None):
    """Download (first call only) and load the 24 liver ROIs with Mantpy."""
    return cohort(mt.datasets.schistosoma_ecm(cache_dir=cache_dir, progressbar=False, verbose=False))


def normalise(images):
    """Pooled arcsinh / 99.5th-percentile scaling of each marker (one bound per marker for the whole cohort)."""
    rois = sorted(images)
    bounds = []
    for c in range(len(MARKERS)):
        pooled = np.concatenate([np.arcsinh(images[r][c].reshape(-1) / COFACTOR) for r in rois]).astype(np.float32)
        bounds.append(np.percentile(pooled, PERCENTILE))
        del pooled
    bounds = np.asarray(bounds, dtype=np.float32)
    out = {}
    for r in rois:
        img = np.empty_like(images[r])
        for c in range(len(MARKERS)):
            img[c] = np.clip(np.arcsinh(images[r][c] / COFACTOR) / bounds[c], 0.0, 1.0)
        out[r] = img
    return out, bounds


def patch_features(norm, meta) -> pd.DataFrame:
    """Complete 10-px grid, using Mantpy's public image and patch APIs."""
    info = meta.set_index("roi")
    rows = []
    for r in sorted(norm):
        adata = mt.io.read_ecm_image(norm[r], marker_names=MARKERS, sample_id=r)
        mt.pp.extract_ecm_patches(adata, patch_size=PATCH, features=["mean", "std"],
                                 ecm_K=None, background_quantile=0.0)
        source = adata.uns["ecm_patches"]
        cx, cy = source.x.to_numpy(), source.y.to_numpy()
        df = source[FEAT_COLS].copy().reset_index(drop=True)
        df.insert(0, "gy", ((cy - PATCH / 2) // PATCH).astype(int))
        df.insert(0, "gx", ((cx - PATCH / 2) // PATCH).astype(int))
        df.insert(0, "y", cy.astype(np.float64))
        df.insert(0, "x", cx.astype(np.float64))
        df.insert(0, "group", info.loc[r, "group"])
        df.insert(0, "mouse", info.loc[r, "mouse"])
        df.insert(0, "roi", r)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def signal_mask(feat: pd.DataFrame) -> np.ndarray:
    """True for ECM signal patches: 2-cluster K-means on the cohort-standardised features; the lower-intensity component is background."""
    z = StandardScaler().fit_transform(feat[FEAT_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
    lab = KMeans(**KMEANS_BG).fit(z).labels_.astype(int)
    std_means = np.array([z[lab == k, :6].mean() for k in range(2)])
    return lab != int(np.argmin(std_means))


def mouse_table(meta) -> pd.DataFrame:
    return meta.groupby("mouse").agg(group=("group", "first"), condition=("condition", "first"), genotype=("genotype", "first"),
                                     n_rois=("roi", "size")).reset_index()

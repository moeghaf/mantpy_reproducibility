"""Training-mouse-only preprocessing for the published liver controls."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import COFACTOR, FEAT_COLS, K, KMEANS_BG, MARKERS, PERCENTILE, patch_features


def prepare_fold(images, meta, held_mouse):
    """Scale all ROIs with training bounds; fit background only on training patches."""
    rois = sorted(meta.loc[meta.mouse != held_mouse, "roi"])
    bounds = np.asarray([
        np.percentile(np.concatenate([
            np.arcsinh(images[roi][c].reshape(-1) / COFACTOR) for roi in rois
        ]).astype(np.float32), PERCENTILE) for c in range(len(MARKERS))
    ], dtype=np.float32)
    norm = {
        roi: np.clip(np.arcsinh(image / COFACTOR) / bounds[:, None, None],
                     0, 1).astype(np.float32)
        for roi, image in images.items()
    }
    nodes = patch_features(norm, meta)
    train = (nodes.mouse != held_mouse).to_numpy()
    features = nodes[FEAT_COLS].to_numpy(np.float32)
    scaler = StandardScaler().fit(features[train])
    z = scaler.transform(features).astype(np.float32)
    model = KMeans(**KMEANS_BG).fit(z[train])
    labels = model.predict(z)
    labels[train] = model.labels_
    means = [float(z[train][model.labels_ == c, :6].mean()) for c in range(2)]
    nodes["signal"] = labels != int(np.argmin(means))
    return nodes, norm, {
        "held_mouse": held_mouse, "training_rois": rois,
        "marker_bounds": bounds.tolist(),
        "background_scaler_mean": scaler.mean_.tolist(),
        "background_scaler_scale": scaler.scale_.tolist(),
        "background_centres": model.cluster_centers_.tolist(),
        "background_component": int(np.argmin(means)),
    }


def fit_fold_clusters(features, train_signal):
    """Fit scaling and K=6 on training signal; predict every signal patch."""
    scaler = StandardScaler().fit(features[train_signal])
    z = scaler.transform(features).astype(np.float32)
    model = KMeans(n_clusters=K, n_init=10, random_state=0,
                   max_iter=300, tol=1e-4, algorithm="lloyd").fit(z[train_signal])
    raw = model.predict(z)
    raw[train_signal] = model.labels_
    counts = np.bincount(raw[train_signal], minlength=K)
    order = sorted(range(K), key=lambda c: (-int(counts[c]), c))
    mapping = np.empty(K, dtype=int)
    mapping[order] = np.arange(K)
    return mapping[raw], {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "centres_raw": model.cluster_centers_.tolist(),
        "training_counts_raw": counts.tolist(), "raw_to_label": mapping.tolist(),
    }

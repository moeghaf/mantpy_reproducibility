"""Search space, configuration identity and the paper's selected settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: Number of annotated layers, used as the number of clusters.
K_BY_IMAGE = {"intestine1": 5, "intestine2": 4}
ACCESSION = {"intestine1": "HBM926.SHNZ.594", "intestine2": "HBM735.PSTF.274"}
IMAGES = tuple(K_BY_IMAGE)
PATCH_SIZES = (16, 32, 64)

#: The 28 settings searched jointly (Supplementary Table 3).
SPACE = {
    # geometry
    "patch_size": [16, 32, 64],
    "neighbours": [10, 20, 40, 80, 120, 160, 200, 240],
    "continuity_thresh": [0.0, 0.25, 0.5, 0.75],
    # CNN architecture
    "cnn_blocks": [2, 3, 4],
    "cnn_width": [8, 16, 32],
    "cnn_readout": ["avgpool2", "avgpool1", "maxpool2", "flatten"],
    "cnn_activation": ["relu", "gelu"],
    "cnn_latent": [32, 64, 96, 128, 256],
    "cnn_projection": [64, "none"],
    # CNN training
    "cnn_temperature": [0.1, 0.2, 0.5],
    "cnn_augment": ["default", "strong"],
    "cnn_lr": [3e-4, 1e-3, 3e-3],
    "cnn_weight_decay": [1e-6, 1e-5, 1e-4],
    "cnn_epochs": [100, 200, 400],
    "cnn_batch": [512, 1024],
    "cnn_standardize": [True, False],
    "cnn_window": [1, 2],
    # edge features
    "edge_distance": ["log", "raw", "inverse", "inverse_square", "gaussian", "none"],
    "edge_orientation": [True, False],
    "edge_continuity": [False, True],
    "edge_use": ["gine", "gin"],
    # GraphMAE
    "gm_depth": [1, 2, 3, 4, 5, 6],
    "gm_width": [64, 128, 256],
    "gm_mask": [0.25, 0.5, 0.75],
    "gm_lr": [3e-4, 1e-3, 3e-3],
    "gm_weight_decay": [1e-5, 1e-4],
    "gm_updates": [100, 200, 400],
    "gm_standardize": [False, True],
}

#: Baseline configuration of the paper's search. Its CNN is Mantpy's encoder with 2 x 2 average pooling.
BASELINE = {
    "patch_size": 32, "neighbours": 40, "continuity_thresh": 0.5,
    "cnn_blocks": 3, "cnn_width": 16, "cnn_readout": "avgpool2", "cnn_activation": "relu", "cnn_latent": 96,
    "cnn_projection": 64, "cnn_temperature": 0.2, "cnn_augment": "default", "cnn_lr": 1e-3, "cnn_weight_decay": 1e-5,
    "cnn_epochs": 200, "cnn_batch": 1024, "cnn_standardize": True, "cnn_window": 1,
    "edge_distance": "log", "edge_orientation": True, "edge_continuity": False, "edge_use": "gine",
    "gm_depth": 2, "gm_width": 128, "gm_mask": 0.5, "gm_lr": 1e-3, "gm_weight_decay": 1e-4, "gm_updates": 200,
    "gm_standardize": False,
}

CNN_AXES = ["patch_size", "cnn_blocks", "cnn_width", "cnn_readout", "cnn_activation", "cnn_latent", "cnn_projection",
            "cnn_temperature", "cnn_augment", "cnn_lr", "cnn_weight_decay", "cnn_epochs", "cnn_batch",
            "cnn_standardize", "cnn_window"]

#: Settings added after the paper's first configurations were recorded. Their default value is left
#: out of the digest, so that configuration IDs equal those in the paper's records.
_LATE_AXIS_DEFAULTS = {"cnn_window": 1}
#: Name of each image in the paper's records, where it is the first part of a configuration's full ID.
_ORIGINAL_ROI = {"intestine1": "original", "intestine2": "HBM735.PSTF.274"}


def validate(config: dict) -> dict:
    """Check a configuration against the search space and return its canonical form."""
    config = dict(config)
    for key, default in _LATE_AXIS_DEFAULTS.items():
        config.setdefault(key, default)
    missing = set(SPACE) - set(config)
    if missing:
        raise KeyError(f"Missing settings: {sorted(missing)}")
    extra = set(config) - set(SPACE)
    if extra:
        raise KeyError(f"Unknown settings: {sorted(extra)}")
    for key, values in SPACE.items():
        if config[key] not in values:
            raise ValueError(f"{key}={config[key]!r} is not one of {values}")
    if config["cnn_readout"] == "flatten" and config["patch_size"] != 32:
        raise ValueError("The flatten readout is restricted to 32-pixel patches.")
    if config["edge_use"] == "gin":
        # GIN ignores edge features: canonicalise so identical graphs share one identity.
        config.update(edge_distance="none", edge_orientation=False, edge_continuity=False)
    elif config["edge_distance"] == "none" and not config["edge_orientation"] and not config["edge_continuity"]:
        raise ValueError("GINE needs at least one edge feature; use edge_use='gin'.")
    return config


def repair(config: dict) -> tuple[dict, list[str]]:
    """Map a sampled point onto the valid space and report what changed.

    The first two rules are those of the paper's search. The third maps GINE without edge
    features to GIN; the paper's search code raised an error for such a point.
    """
    config, notes = dict(config), []
    if config["patch_size"] == 16 and config["neighbours"] == 80:
        config["neighbours"] = 40
        notes.append("16 px with 80 neighbours -> 40 neighbours")
    if config["cnn_readout"] == "flatten" and config["patch_size"] != 32:
        config["cnn_readout"] = "avgpool2"
        notes.append("flatten readout outside 32 px -> avgpool2")
    if config["edge_use"] == "gine" and config["edge_distance"] == "none" and not config["edge_orientation"] \
            and not config["edge_continuity"]:
        config["edge_use"] = "gin"
        notes.append("GINE without edge features -> GIN")
    return validate(config), notes


def _digest(values: dict) -> str:
    values = {k: v for k, v in values.items() if not (k in _LATE_AXIS_DEFAULTS and v == _LATE_AXIS_DEFAULTS[k])}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_id(config: dict) -> str:
    """Deterministic identity of a configuration, equal to the ID used in the paper's records."""
    return "c" + _digest(validate(config))


def original_config_id(image: str, config: dict) -> str:
    return f"{_ORIGINAL_ROI[image]}/{config_id(config)}"


def cnn_id(config: dict) -> str:
    return "cnn" + _digest({key: config[key] for key in CNN_AXES})


def has_baseline_cnn(config: dict) -> bool:
    return all(config[key] == BASELINE[key] for key in CNN_AXES if key != "patch_size")


def paper_selection() -> dict:
    """The settings selected in the paper, packaged with this pipeline."""
    return json.loads((Path(__file__).with_name("paper_selection.json")).read_text(encoding="utf-8"))

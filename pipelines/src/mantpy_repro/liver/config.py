"""Search spaces, configuration identity and the fixed settings of the liver experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MARKERS = ["Fibronectin", "Pan-laminin", "CollagenIV", "VersicanV0V1", "CollagenI", "HeparanSulfate"]
GROUPS = ["Naive_KO", "Naive_WT", "Infected_KO", "Infected_WT"]
CLASSES = ["Naive", "Infected"]          # class 1 is "Infected"
N_MICE = 8
N_INFECTED = 4
N_ALLOCATIONS = math.comb(N_MICE, N_INFECTED)      # 70 ways to call four of the eight mice infected

# ----------------------------------------------------------- encoder, patch size and K (Supplementary Table 4)
#: The 15 settings searched jointly. Every fitted encoder is clustered at every K.
SPACE = {
    "patch_size": [5, 10, 15, 20],
    "cnn_blocks": [2, 3, 4],
    "cnn_width": [8, 16, 32],
    "cnn_readout": ["avgpool2", "avgpool1", "maxpool2", "flatten"],
    "cnn_activation": ["relu", "gelu"],
    "cnn_latent": [32, 64, 96, 128, 256],
    "cnn_projection": [64, "none"],
    "cnn_temperature": [0.1, 0.2, 0.5],
    "cnn_augment": ["default", "strong"],
    "cnn_lr": [3e-4, 1e-3, 3e-3],
    "cnn_weight_decay": [1e-6, 1e-5, 1e-4],
    "cnn_epochs": [100, 200, 400],
    "cnn_batch": [512, 1024],
    "cnn_standardize": [True, False],
    "n_clusters": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}
ENCODER_AXES = [axis for axis in SPACE if axis != "n_clusters"]
K_VALUES = tuple(SPACE["n_clusters"])
PATCH_SIZES = tuple(SPACE["patch_size"])
ANCHOR_SIZE = 5                   # seed stability and coverage are measured on the 5-pixel signal patches
FLATTEN_PATCH_SIZE = 10           # the flatten readout is allowed at this patch size only
MIN_FEATURE_MAP = 2               # the last convolution block must leave at least 2 x 2 pixels

SEARCH_SEEDS = (0, 1, 2)
FREEZE_SEEDS = (0, 1, 2, 3, 4)
SPREAD_MIN_ROIS = 3               # every cluster needs enough patches in at least this many ROIs, in every seed
SPREAD_AREA_PX2 = 1000            # "enough" = ceil(1000 / patch_size^2) patches
STABILITY_MIN = 0.80              # five-seed stability required by the rule that chooses K
MAX_TRAINING_STEPS = 60_000       # configurations needing more optimiser steps are not fitted
N_SEARCH_ENCODERS = 60            # distinct new encoders proposed by NSGA-II (the warm start is not counted)
POPULATION = 24
NSGA_SEED = 20261024
SOBOL_SEED = 20261024
N_SOBOL = 24
TIE = 0.01

#: First warm-start configuration of the paper's search (the lung selection, with K = 3).
LUNG_START = {"patch_size": 5, "cnn_blocks": 2, "cnn_width": 16, "cnn_readout": "maxpool2", "cnn_activation": "relu",
              "cnn_latent": 256, "cnn_projection": "none", "cnn_temperature": 0.1, "cnn_augment": "default", "cnn_lr": 3e-3,
              "cnn_weight_decay": 1e-4, "cnn_epochs": 400, "cnn_batch": 512, "cnn_standardize": False, "n_clusters": 3}

# ------------------------------------------------------------------------------- downstream analysis
SPECIFICITY_EPS = 1e-3            # pseudocount of the infection-associated cluster rule
SPECIFICITY_PRESENCE = 0.01       # a candidate cluster makes up at least 1% of every infected mouse's ECM patches
SUBGRAPH_RADIUS = 40.0
SUBGRAPH_MIN_NODES = 5
CLASSIFIER_SEEDS = (0, 1, 2, 3, 4)
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]
IG_STEPS = 50

# -------------------------------------------------- within-fold classifier tuning (Supplementary Table 5)
GRAPH_TYPES = ["delaunay", "grid4", "grid8", "radius20", "radius30", "radius40", "radius50"]
TUNING_HIDDEN = [16, 32, 64, 128, 256]
TUNING_EPOCHS = [60, 120, 200, 300, 500]
N_TUNING_TRIALS = 80
N_STARTUP_TRIALS = 20
INNER_SEEDS = (0, 1)
FAIL_OBJECTIVE = 2.0              # value recorded for a configuration that could not be fitted
MAX_DIRECTED_EDGES = 2.5e7        # graphs estimated to be larger are not built


def conv_sizes(input_size: int, blocks: int) -> list[int]:
    """Side length of the feature map after each stride-2 convolution block."""
    sizes, size = [], int(input_size)
    for _ in range(blocks):
        size = (size + 1) // 2
        sizes.append(size)
    return sizes


def max_blocks(input_size: int) -> int:
    """Largest number of blocks that leaves a feature map of at least 2 x 2 pixels."""
    blocks = 0
    while conv_sizes(input_size, blocks + 1)[-1] >= MIN_FEATURE_MAP:
        blocks += 1
    return blocks


def _check_encoder(config: dict) -> None:
    if config["cnn_readout"] == "flatten" and config["patch_size"] != FLATTEN_PATCH_SIZE:
        raise ValueError(f"The flatten readout is restricted to {FLATTEN_PATCH_SIZE}-pixel patches.")
    if config["cnn_blocks"] > max_blocks(config["patch_size"]):
        raise ValueError(f"cnn_blocks={config['cnn_blocks']} is too deep for {config['patch_size']}-pixel patches "
                         f"(at most {max_blocks(config['patch_size'])}).")


def validate(config: dict) -> dict:
    """Check a full configuration (14 encoder settings and K). Invalid ones are rejected, never adapted."""
    config = dict(config)
    missing = set(SPACE) - set(config)
    if missing:
        raise KeyError(f"Missing settings: {sorted(missing)}")
    extra = set(config) - set(SPACE)
    if extra:
        raise KeyError(f"Unknown settings: {sorted(extra)}")
    for key, values in SPACE.items():
        if config[key] not in values:
            raise ValueError(f"{key}={config[key]!r} is not one of {values}")
    _check_encoder(config)
    return config


def encoder_config(config: dict) -> dict:
    """The 14 settings that define one fitted encoder (K is dropped), validated."""
    missing = [axis for axis in ENCODER_AXES if axis not in config]
    if missing:
        raise KeyError(f"Missing encoder settings: {missing}")
    encoder = {axis: config[axis] for axis in ENCODER_AXES}
    for key in ENCODER_AXES:
        if encoder[key] not in SPACE[key]:
            raise ValueError(f"{key}={encoder[key]!r} is not one of {SPACE[key]}")
    _check_encoder(encoder)
    return encoder


def with_k(config: dict, k: int) -> dict:
    return validate({**encoder_config(config), "n_clusters": int(k)})


def _digest(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_id(config: dict) -> str:
    """Identity of an encoder and K, equal to the ID in the paper's records (without its ``liver6/`` prefix)."""
    return "c" + _digest(validate(config))


def encoder_id(config: dict) -> str:
    """Identity of a fitted encoder; the same for every K."""
    return "e" + _digest(encoder_config(config))


def original_config_id(config: dict) -> str:
    return f"liver6/{config_id(config)}"


def min_patches_per_roi(patch_size: int) -> int:
    """Patches a cluster needs in one ROI to count for the spread guard (40, 10, 5, 3 at 5, 10, 15, 20 pixels)."""
    return int(math.ceil(SPREAD_AREA_PX2 / float(patch_size) ** 2))


def training_steps(config: dict, n_signal: int) -> int:
    return math.ceil(n_signal / config["cnn_batch"]) * config["cnn_epochs"]


def paper_selection() -> dict:
    """The settings selected in the paper, packaged with this pipeline."""
    return json.loads(Path(__file__).with_name("paper_selection.json").read_text(encoding="utf-8"))


def paper_reference() -> dict:
    """The numbers reported in the paper, packaged with this pipeline."""
    return json.loads(Path(__file__).with_name("paper_reference.json").read_text(encoding="utf-8"))


def paper_partition() -> tuple:
    """Row indices of the paper's ECM signal patches on the complete 10-pixel lattice, and their cluster labels."""
    import numpy as np

    with np.load(Path(__file__).with_name(paper_selection()["partition"]["file"]), allow_pickle=False) as saved:
        return saved["signal_index"].astype(np.int64), saved["labels"].astype(np.int64)

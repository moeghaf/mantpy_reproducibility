"""Recipes, search space, folds and seeds of the lung experiments, and the paper's selected settings.

A configuration of the denoising task has four parts: the graph rules, the context features,
the logistic regression and the cleaning rule. ``Config.to_dict`` writes them in the layout of
``paper_selection.json``, so a configuration read from that file is written back unchanged
and keeps its identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any

PATCH_SIZE = 10
FOCAL_CELL_TYPE = "AEC"          # airway epithelial cells
WALL_CLUSTER = 1                 # ECM 1, the bronchovascular wall

CONTEXTS = ("cell", "ecm", "joint")
PRIOR = "prior"                  # label-prior baseline: training cluster frequencies only
SCHEMES = ("mouse", "roi")       # outer hold-out: each mouse (main result) or each ROI (control)
NORMALIZATIONS = ("training_cohort_z", "per_roi_z")   # marker scaling arms of the refit control

#: Synthetic corruption (Methods).
KINDS = ("mixed", "block", "isolated")
PRIMARY_KIND = "mixed"
BUDGET_FRACTION = 0.10
BLOCK_SIDES = (50.0, 100.0, 150.0)
MIN_BLOCK_PATCHES = 20
EVALUATION_SEEDS = (0, 1, 2, 3, 4)
TUNING_SEEDS = (100, 101)

#: Optuna trials of each nested search.
TRIALS = {"mouse": 60, "roi": 150, "refit": 150}
SELECTION_TOLERANCE = 1e-4       # trials this close to the best inner accuracy are tied
N_EDGE_RULES = 48                # random cell-ECM edge rules of Supplementary Fig. 3b

CC_RULES = ("knn", "radius", "delaunay")
EE_RULES = ("grid8", "grid4", "knn", "radius", "delaunay")
CE_RULES = ("knn", "radius", "delaunay", "containing_ring")
EDGE_WEIGHTS = ("none", "inv_d", "inv_d2", "gaussian_median")
COMPOSITIONS = ("fraction", "count", "log1p_count")
PENALTIES = ("l2", "l1", "elasticnet", "none")
ANOMALY_STATS = ("minmax_neglogp", "raw_neglogp", "one_minus_p", "margin")
CALIBRATIONS = ("none", "temperature")

#: Search space of the nested optimisation (label-only logistic regression).
SPACE = {
    "hops": [1, 2],
    "composition_norm": list(COMPOSITIONS),
    "standardize": [True, False],
    "edge_weight": list(EDGE_WEIGHTS),
    "ee_include_background_nodes": [True, False],
    "cc_rule": list(CC_RULES), "cc_k": (3, 12), "cc_radius_px": (10.0, 40.0),
    "ee_rule": list(EE_RULES), "ee_k": (4, 12), "ee_radius_px": (10.0, 30.0),
    "ce_rule": list(CE_RULES), "ce_k": (1, 12), "ce_dmax_px": [10.0, 15.0, 20.0, 30.0, None],
    "penalty": list(PENALTIES), "C": (0.001, 100.0), "l1_ratio": (0.0, 1.0), "class_weight": [None, "balanced"],
    "anomaly_stat": list(ANOMALY_STATS),
    "threshold": {"minmax_neglogp": ("threshold_minmax", 0.05, 0.7),
                  "raw_neglogp": ("threshold_raw_neglogp", 0.2, 3.0),
                  "one_minus_p": ("threshold_one_minus_p", 0.3, 0.95),
                  "margin": ("threshold_margin", 0.0, 0.9)},
    "gate_radius_px": [0.0, 10.0, 14.2, 20.0, 30.0],
    "gate_min_neighbours": [0, 1, 2, 3],
    "min_pred_conf": [0.0, 0.5, 0.7],
    "majority_smoothing": [True, False],
    "passes": [1, 2, 3],
    "calibration": list(CALIBRATIONS),
}


@dataclass(frozen=True)
class GraphRecipe:
    """Edge rules of the three graph layers. The defaults are the graph of Fig. 4:
    cells within 15 pixels, an eight-connected patch grid that keeps background patches as
    nodes, and each cell joined to its five nearest patches within 15 pixels."""

    cc_rule: str = "radius"
    cc_k: int = 5
    cc_radius_px: float | None = 15.0      # radius, or the distance cap of the kNN rule
    ee_rule: str = "grid8"
    ee_k: int = 8
    ee_radius_px: float | None = 15.0      # radius, kNN cap, or half the Delaunay edge cap
    ee_include_background_nodes: bool = True
    ce_rule: str = "knn"
    ce_k: int = 5
    ce_dmax_px: float | None = 15.0        # kNN cap, radius, or half the Delaunay edge cap
    edge_weight: str = "none"

    def __post_init__(self):
        _check("cc_rule", self.cc_rule, CC_RULES)
        _check("ee_rule", self.ee_rule, EE_RULES)
        _check("ce_rule", self.ce_rule, CE_RULES)
        _check("edge_weight", self.edge_weight, EDGE_WEIGHTS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "GraphRecipe":
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in record.items() if key in names})

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class FeatureRecipe:
    """Context features: how neighbour labels are counted and over how many hops."""

    composition_norm: str = "fraction"
    hops: int = 1
    standardize: bool = False

    def __post_init__(self):
        _check("composition_norm", self.composition_norm, COMPOSITIONS)
        _check("hops", self.hops, (1, 2))


@dataclass(frozen=True)
class LearnerRecipe:
    """Multinomial logistic regression."""

    C: float = 1.0
    penalty: str = "l2"
    l1_ratio: float = 0.5
    class_weight: str | None = None
    standardize: bool = False

    def __post_init__(self):
        _check("penalty", self.penalty, PENALTIES)
        _check("class_weight", self.class_weight, (None, "balanced"))


@dataclass(frozen=True)
class CleaningRecipe:
    """Rule that relabels a patch to the predicted cluster."""

    anomaly_stat: str = "minmax_neglogp"
    threshold: float = 0.30
    min_pred_conf: float = 0.0
    gate_radius_px: float = 14.0           # 0 switches the spatial gate off
    gate_min_neighbours: int = 2
    majority_smoothing: bool = False
    passes: int = 1
    temperature: float = 1.0               # set when the model is fitted (1 = no calibration)

    def __post_init__(self):
        _check("anomaly_stat", self.anomaly_stat, ANOMALY_STATS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    """One complete configuration of the denoising task for one context."""

    context: str
    graph: GraphRecipe = field(default_factory=GraphRecipe)
    features: FeatureRecipe = field(default_factory=FeatureRecipe)
    learner: LearnerRecipe = field(default_factory=LearnerRecipe)
    cleaning: CleaningRecipe = field(default_factory=CleaningRecipe)
    calibration: str = "none"

    def __post_init__(self):
        _check("context", self.context, (*CONTEXTS, PRIOR))
        _check("calibration", self.calibration, CALIBRATIONS)

    def to_dict(self) -> dict[str, Any]:
        """The configuration in the layout of ``paper_selection.json``."""
        return {
            "task": "forward", "context": self.context, "graph": self.graph.to_dict(),
            "features": {"composition_norm": self.features.composition_norm, "hops": self.features.hops,
                         "intensity_block": False, "standardize": self.features.standardize},
            "learner": {"learner": "lr", "C": self.learner.C, "penalty": self.learner.penalty,
                        "l1_ratio": self.learner.l1_ratio, "class_weight": self.learner.class_weight,
                        "poly_degree": 1, "standardize": self.learner.standardize, "hgb": {}, "gnn": {}},
            "cleaning": self.cleaning.to_dict(), "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Config":
        """Read a configuration in the layout of ``paper_selection.json``.

        The paper's denoising task predicts ECM labels (``forward``) with logistic regression
        and without neighbouring marker intensities; a record of anything else is rejected.
        """
        features, learner = dict(record["features"]), dict(record["learner"])
        fixed = {"task": record.get("task", "forward") == "forward",
                 "intensity_block": not features.pop("intensity_block", False),
                 "learner": learner.pop("learner", "lr") == "lr",
                 "poly_degree": learner.pop("poly_degree", 1) == 1,
                 "hgb": not learner.pop("hgb", {}), "gnn": not learner.pop("gnn", {})}
        wrong = [name for name, ok in fixed.items() if not ok]
        if wrong:
            raise ValueError(f"Configuration uses settings outside the paper's denoising task: {wrong}")
        return cls(context=record["context"], graph=GraphRecipe.from_dict(record["graph"]),
                   features=FeatureRecipe(**features), learner=LearnerRecipe(**learner),
                   cleaning=CleaningRecipe(**record["cleaning"]), calibration=record.get("calibration", "none"))


def _check(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {list(allowed)}")


def config_id(config: Config) -> str:
    """Deterministic identity of a configuration."""
    text = json.dumps(config.to_dict(), sort_keys=True)
    return "c" + hashlib.sha256(text.encode()).hexdigest()[:12]


# Folds
def outer_folds(mice: dict[str, list[str]], scheme: str) -> list[dict]:
    """Outer folds as ``{name, test, train}``; ``mice`` maps each mouse to its ROIs in cohort order."""
    rois = [roi for group in mice.values() for roi in group]
    if scheme == "roi":
        return [{"name": roi, "test": [roi], "train": [other for other in rois if other != roi]} for roi in rois]
    if scheme == "mouse":
        return [{"name": mouse, "test": list(group), "train": [roi for roi in rois if roi not in group]}
                for mouse, group in mice.items()]
    raise ValueError(f"scheme must be 'mouse' or 'roi', not {scheme!r}")


def inner_folds(train: list[str]) -> list[dict]:
    """Inner loop: each training ROI is held out in turn."""
    return [{"name": roi, "test": [roi], "train": [other for other in train if other != roi]} for roi in train]


# Sampler seeds
def sampler_seed(scheme: str, fold_index: int, context: str) -> int:
    """Seed of the Optuna sampler of one study, equal to the seed of the paper's run.

    Single-ROI studies use 1000 + 6 x fold + context, mouse studies 1036 + 6 x fold + context
    and the refit control 2000 + 3 x fold + context. The seeds of the first two are not
    consecutive.
    """
    offset = CONTEXTS.index(context)
    if scheme == "roi":
        return 1000 + 6 * fold_index + offset
    if scheme == "mouse":
        return 1036 + 6 * fold_index + offset
    if scheme == "refit":
        return 2000 + 3 * fold_index + offset
    raise ValueError(f"unknown scheme {scheme!r}")


def startup_trials(scheme: str, n_trials: int) -> int:
    """Random trials before the tree-structured Parzen estimator takes over."""
    return 30 if scheme == "refit" else max(5, int(0.2 * n_trials))


def paper_selection() -> dict:
    """The settings selected in the paper, packaged with this pipeline."""
    return json.loads(Path(__file__).with_name("paper_selection.json").read_text(encoding="utf-8"))

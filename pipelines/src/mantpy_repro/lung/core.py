"""Shared state of a lung pipeline run: results directory, partition, budgets and stored tables."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np

from ..store import Store
from .config import EVALUATION_SEEDS, N_EDGE_RULES, PATCH_SIZE, TRIALS
from .data import DATASET_DOI, PARTITION_SEED, PARTITION_THREADS, PARTITIONS, SIGNAL_K_RANGE, Cohort, Prepared
from .graphs import GraphCache

FIT_THREADS = 1      # the paper's searches ran one thread per process; parallelism comes from --workers


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def environment() -> dict:
    """Software recorded with every result."""
    versions = {}
    for name in ("mantpy", "numpy", "scipy", "scikit-learn", "pandas", "anndata", "squidpy", "optuna"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions, "python": platform.python_version(), "platform": platform.platform()}


@dataclass(frozen=True)
class Budget:
    """How much is computed: the paper's amounts, or tiny ones for ``--smoke``."""

    trials: dict
    evaluation_seeds: tuple
    n_edge_rules: int
    roi_folds: int | None         # single-ROI hold-out and refit control: number of held-out ROIs (None = all)

    @classmethod
    def of(cls, smoke: bool) -> "Budget":
        if smoke:
            return cls(trials={scheme: 3 for scheme in TRIALS}, evaluation_seeds=EVALUATION_SEEDS[:1],
                       n_edge_rules=2, roi_folds=1)
        return cls(trials=dict(TRIALS), evaluation_seeds=EVALUATION_SEEDS, n_edge_rules=N_EDGE_RULES, roi_folds=None)


class Context:
    """One results directory, one ECM partition."""

    def __init__(self, results: str | Path, *, cache_dir: str | None = None, partition: str = "paper",
                 smoke: bool = False):
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, not {partition!r}")
        self.store = Store(results)
        self.cache_dir = cache_dir
        self.partition = partition
        self.smoke = smoke
        self.budget = Budget.of(smoke)
        mode = self.store.root / "MODE"
        expected = "smoke" if smoke else "full"
        if not mode.exists():
            mode.write_text(expected)
        found = mode.read_text().strip() or expected      # empty while another worker is writing it
        if found != expected:
            raise RuntimeError(f"{self.store.root} holds {found} results; "
                               f"use another --results directory for a {expected} run.")
        self._prepared: Prepared | None = None
        self._cohort: Cohort | None = None
        self._cache: GraphCache | None = None

    def prepared(self) -> Prepared:
        """Patches, component scan, fresh partition, cells and raw statistics (computed once)."""
        if self._prepared is None:
            from .data import extract, load_bundle

            request = {"dataset": DATASET_DOI, "patch_size": PATCH_SIZE, "normalize": "pooled_zscore",
                       "features": ["mean"], "signal_components": list(SIGNAL_K_RANGE), "seed": PARTITION_SEED,
                       "threads": PARTITION_THREADS, "eligibility": "mean feature above the ROI minimum"}
            unit = self.store.unit("prepare", "cohort", request=request)

            def fit():
                arrays, metadata = extract(load_bundle(self.cache_dir))
                return arrays, {**metadata, "environment": environment()}

            self._prepared = Prepared(self.store.compute(unit, fit, label="prepare", log=log))
        return self._prepared

    def cohort(self) -> Cohort:
        if self._cohort is None:
            self._cohort = self.prepared().cohort(self.partition)
        return self._cohort

    def graphs(self) -> GraphCache:
        """Graph cache of this process for the cohort of :meth:`cohort`."""
        if self._cache is None:
            self._cache = GraphCache(96)
        return self._cache

    def roi_folds(self, folds: list[dict]) -> list[dict]:
        """Single-ROI folds to run (all of them, or the first for ``--smoke``)."""
        return folds if self.budget.roi_folds is None else folds[:self.budget.roi_folds]


# ------------------------------------------------------------------------------- tables in a unit
_STATES = {"nan": 1, "+inf": 2, "-inf": 3}


def pack(frame, prefix: str) -> dict[str, np.ndarray]:
    """A table as arrays for a result unit.

    Result units refuse non-finite numbers so that a failed fit cannot look complete. Scores
    that are undefined by design (for example the precision of a rule that relabelled nothing)
    are therefore stored as 0 with a companion array that marks them NaN or infinite.
    """
    arrays = {f"{prefix}columns": np.asarray(list(frame.columns), dtype=str)}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype.kind in "OUS":
            arrays[f"{prefix}{column}"] = values.astype(str)
            continue
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            state = np.zeros(len(values), dtype=np.int8)
            state[np.isnan(values)] = _STATES["nan"]
            state[np.isposinf(values)] = _STATES["+inf"]
            state[np.isneginf(values)] = _STATES["-inf"]
            arrays[f"{prefix}{column}__state"] = state
            values = np.where(state == 0, values, 0.0)
        arrays[f"{prefix}{column}"] = values
    return arrays


def unpack(arrays: dict[str, np.ndarray], prefix: str):
    """Inverse of :func:`pack`."""
    import pandas as pd

    frame = {}
    for column in arrays[f"{prefix}columns"].astype(str):
        values = arrays[f"{prefix}{column}"]
        state = arrays.get(f"{prefix}{column}__state")
        if state is not None:
            values = values.astype(float)
            for value, code in ((np.nan, 1), (np.inf, 2), (-np.inf, 3)):
                values[state == code] = value
        frame[column] = values.astype(str) if values.dtype.kind in "US" else values
    return pd.DataFrame(frame)


def unit_arrays(folder: Path) -> dict[str, np.ndarray] | None:
    """Arrays of a finished unit, or ``None`` when it is not complete."""
    folder = Path(folder)
    if not (folder / "completion.json").exists():
        return None
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}

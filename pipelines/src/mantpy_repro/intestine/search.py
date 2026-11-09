"""Label-free search of the 28 settings (Supplementary Table 3).

Each configuration is fitted with three training seeds and scored without the annotation:

- **seed stability**: mean pairwise ARI of the three GraphMAE partitions on the common anchors;
- **coverage**: fraction of the finest foreground grid inside the analysed patches;
- **feasibility**: every cluster holds at least 1% of the common anchors in every seed.

Two modes choose which configurations are evaluated:

``paper`` (default)
    The configurations evaluated in the paper (197 for image 1, 200 for image 2), packaged in
    ``paper_search_records.json``. Each is refitted here; the list itself is fixed, so the work
    splits cleanly over cluster jobs.
``nsga``
    A new NSGA-II search (Optuna) with the paper's sampler settings. It is stochastic and
    explores different configurations from the paper's.

Either way every evaluation is written to ``search/<image>/evaluations/<config_id>.json``, and
the selection stage reads only those files. Any number of workers can share a results directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np

from ..store import read_json, write_json
from .config import SPACE, config_id, repair, validate
from .core import MAX_DIRECTED_EDGES, MIN_FRACTION, Context, log, min_fraction, stability

SEARCH_SEEDS = (0, 1, 2)
NSGA_SEED = 20261023
POPULATION = 24
MAX_CONSECUTIVE_FAILURES = 3
TABLE_COLUMNS = ["image", "config_id", "stability3", "coverage", "min_fraction", "feasible", "status",
                 "mean_fit_seconds", *SPACE]


def paper_records(image: str) -> list[dict]:
    """Configurations evaluated in the paper's search, with their recorded scores."""
    path = Path(__file__).with_name("paper_search_records.json")
    return json.loads(path.read_text(encoding="utf-8"))["images"][image]


def _evaluation_path(ctx: Context, image: str, identity: str) -> Path:
    return ctx.store.root / "search" / image / "evaluations" / f"{identity}.json"


def _is_out_of_memory(error: BaseException) -> bool:
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def evaluate(ctx: Context, image: str, config: dict, seeds=SEARCH_SEEDS) -> dict:
    """Fit ``config`` for every seed (skipping finished fits) and write its evaluation record."""
    config = validate(config)
    identity = config_id(config)
    path = _evaluation_path(ctx, image, identity)
    if path.exists():
        return read_json(path)
    prepared = ctx.prepared(image)
    coverage = prepared.support.coverage[config["patch_size"]]
    record = {"image": image, "config_id": identity, "config": config, "coverage": coverage,
              "seeds": list(seeds)}
    try:
        runs = [ctx.run_unit(image, config, seed) for seed in seeds]
    except Exception as error:
        if not _is_out_of_memory(error):
            raise
        from ..runtime import release_cuda

        release_cuda()
        record.update(status="out of memory", feasible=False, stability3=None, min_fraction=None,
                      mean_fit_seconds=None, error=str(error)[:300])
        write_json(path, record)
        return record
    common = [run["common_graphmae"] for run in runs]
    value, pairs = stability(common)
    smallest = min_fraction(common, prepared.k)
    seconds = [ctx.unit_metadata(image, config, seed)["metadata"]["graph_seconds"] for seed in seeds]
    record.update(status="complete", stability3=value, pairwise_ari=pairs, min_fraction=smallest,
                  feasible=bool(smallest >= MIN_FRACTION), mean_fit_seconds=float(np.mean(seconds)))
    write_json(path, record)
    return record


def search_table(ctx: Context, image: str):
    """One row per evaluated configuration of ``image`` (the supplementary search table)."""
    import pandas as pd

    folder = ctx.store.root / "search" / image / "evaluations"
    rows = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        record = read_json(path)
        rows.append({key: record.get(key) for key in TABLE_COLUMNS if key not in SPACE} | record["config"])
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def n_complete(ctx: Context, image: str) -> int:
    table = search_table(ctx, image)
    return int((table["status"] == "complete").sum())


# ------------------------------------------------------------------------ paper's configurations
def run_paper_search(ctx: Context, image: str, *, shard: tuple[int, int] = (0, 1),
                     time_limit_hours: float | None = None, limit: int | None = None) -> bool:
    """Evaluate the paper's configurations. ``shard=(i, n)`` takes every n-th one, starting at i.

    ``limit`` keeps only the first configurations (used by ``--smoke``).
    Returns True when every configuration of this shard has an evaluation record.
    """
    records = paper_records(image)[:limit][shard[0]::shard[1]]
    deadline = None if time_limit_hours is None else time.time() + 3600 * time_limit_hours
    log(f"{image}: {len(records)} of the paper's configurations in shard {shard[0]}/{shard[1]}")
    for index, item in enumerate(records):
        if deadline is not None and time.time() > deadline:
            log(f"{image}: time limit reached; rerun the same command to continue")
            return False
        record = evaluate(ctx, image, item["config"])
        if record["status"] == "complete":
            log(f"{image} {record['config_id']} ({index + 1}/{len(records)}): stability {record['stability3']:.3f} "
                f"(paper {item['stability3']:.3f}), coverage {record['coverage']:.3f}, "
                f"min fraction {record['min_fraction']:.3f}")
        else:
            log(f"{image} {record['config_id']} ({index + 1}/{len(records)}): {record['status']}")
    return True


# ------------------------------------------------------------------------------- new NSGA-II run
def _storage(root: Path):
    """A study file that is safe to share between processes and cluster nodes (also on NFS)."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    root.mkdir(parents=True, exist_ok=True)
    path = str(root / "study.journal")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def _distributions():
    from optuna.distributions import CategoricalDistribution

    return {axis: CategoricalDistribution([json.dumps(value) for value in values]) for axis, values in SPACE.items()}


def open_study(ctx: Context, image: str, sampler_seed: int = NSGA_SEED):
    import optuna

    sampler = optuna.samplers.NSGAIISampler(
        seed=sampler_seed, population_size=POPULATION,
        constraints_func=lambda trial: (trial.user_attrs.get("constraint", 1.0),),
    )
    return optuna.create_study(
        study_name=f"stability_coverage_{image}", storage=_storage(ctx.store.root / "search" / image),
        directions=["maximize", "maximize"], sampler=sampler, load_if_exists=True,
    )


def run_nsga_search(ctx: Context, image: str, n_trials: int = 200, *, max_edges: float = MAX_DIRECTED_EDGES,
                    worker: int = 0, time_limit_hours: float | None = None) -> bool:
    """Sample and evaluate new configurations until ``n_trials`` are fitted for ``image``.

    ``worker`` gives each parallel worker its own sampler seed, so workers that share the study
    propose different configurations. Graphs with more than ``max_edges`` estimated directed
    edges are skipped, as in the paper's search, and do not count towards ``n_trials``.
    Returns True when the target is reached.
    """
    from optuna.trial import TrialState

    prepared = ctx.prepared(image)
    support = prepared.support
    study = open_study(ctx, image, NSGA_SEED + worker)
    distributions = _distributions()
    deadline = None if time_limit_hours is None else time.time() + 3600 * time_limit_hours
    failures = 0
    log(f"{image}: {n_complete(ctx, image)} configurations fitted, target {n_trials} (worker {worker})")
    while n_complete(ctx, image) < n_trials:
        if deadline is not None and time.time() > deadline:
            log(f"{image}: time limit reached; rerun the same command to continue")
            return False
        trial = study.ask(distributions)
        config, notes = repair({axis: json.loads(trial.params[axis]) for axis in SPACE})
        identity = config_id(config)
        coverage = support.coverage[config["patch_size"]]
        trial.set_user_attr("config_id", identity)
        trial.set_user_attr("repairs", notes)
        trial.set_user_attr("worker", worker)
        estimated_edges = support.n_nodes[config["patch_size"]] * config["neighbours"]
        if estimated_edges > max_edges:
            trial.set_user_attr("constraint", 1.0)
            trial.set_user_attr("resource_skip", estimated_edges)
            study.tell(trial, [0.0, coverage])
            log(f"{image} trial {trial.number} {identity}: skipped, about {estimated_edges:.2e} directed edges")
            continue
        try:
            record = evaluate(ctx, image, config)
        except Exception:
            study.tell(trial, state=TrialState.FAIL)
            failures += 1
            log(f"{image} trial {trial.number} {identity}: FAILED\n{traceback.format_exc()}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            continue
        failures = 0
        if record["status"] != "complete":
            trial.set_user_attr("constraint", 1.0)
            study.tell(trial, [0.0, coverage])
            log(f"{image} trial {trial.number} {identity}: {record['status']}")
            continue
        trial.set_user_attr("constraint", MIN_FRACTION - record["min_fraction"])
        study.tell(trial, [record["stability3"], coverage])
        log(f"{image} trial {trial.number} {identity} p{config['patch_size']} k{config['neighbours']} "
            f"depth{config['gm_depth']}: stability {record['stability3']:.3f} coverage {coverage:.3f} "
            f"min fraction {record['min_fraction']:.3f} ({n_complete(ctx, image)}/{n_trials})")
    log(f"{image}: search complete with {n_complete(ctx, image)} fitted configurations")
    return True

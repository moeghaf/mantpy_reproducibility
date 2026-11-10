"""Label-free search of the patch size, the encoder settings and K (Supplementary Table 4).

Three objectives are maximised with NSGA-II (Optuna): seed stability (mean pairwise adjusted Rand
index between three training seeds, on the anchors), coverage (fraction of the anchors covered at
the patch size) and K. A configuration meets the spread guard when, in every seed, every cluster
has enough patches in at least three ROIs. A configuration needing more than 60,000 optimiser steps is
recorded with a stability of 0 and is not fitted. No condition, genotype or mouse label is read.

The study starts from 25 fixed configurations (``LUNG_START`` and 24 scrambled-Sobol points) and
stops when NSGA-II has proposed 60 new encoders. A proposal that is not fitted because of the step
limit counts as one of the 60. Invalid proposals and exact repeats are discarded. Every fitted
encoder is clustered at every K, so a proposal that repeats an encoder at another K is scored from
the stored labels and is not counted.

Any number of workers can run this module at the same time on one results directory. They share
the Optuna study and never fit the same encoder and seed twice; with several workers the study
can end a few encoders above the target. A worker that is interrupted closes its trial, and the
seed fits it completed are reused when that encoder is evaluated again; a starting configuration
it was fitting is queued again by the next run of the search. The selection (``freeze``) needs
the finished search: the target reached and every starting configuration evaluated.
"""

from __future__ import annotations

import json
import time

from filelock import FileLock

from ..runtime import release_cuda
from ..store import read_json, write_json
from .config import (K_VALUES, LUNG_START, MAX_TRAINING_STEPS, N_SEARCH_ENCODERS, N_SOBOL, NSGA_SEED, POPULATION, SEARCH_SEEDS,
                     SOBOL_SEED, SPACE, config_id, encoder_id, training_steps, validate)
from .core import Context, log

STUDY_NAME = "liver_stability_coverage_k"
MAX_NSGA_TRIALS = 1000           # limit on NSGA-II proposals, so that a search that finds no new encoder still ends
MAX_CONSECUTIVE_FAILURES = 3     # stop instead of recording failure after failure (for example a full GPU)
SMOKE_SOBOL = 2
SMOKE_ENCODERS = 2


def _storage(root):
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


def open_study(ctx: Context, worker_index: int = 0):
    """Open the shared study. Each worker's sampler has its own seed so that workers propose different points."""
    import optuna

    sampler = optuna.samplers.NSGAIISampler(
        seed=NSGA_SEED + worker_index, population_size=POPULATION,
        constraints_func=lambda trial: (trial.user_attrs.get("constraint", 1.0),),
    )
    return optuna.create_study(study_name=STUDY_NAME, storage=_storage(ctx.store.root / "search"),
                               directions=["maximize", "maximize", "maximize"], sampler=sampler, load_if_exists=True)


def warm_start(n_sobol: int = N_SOBOL) -> dict:
    """The fixed starting configurations: ``LUNG_START``, then scrambled-Sobol points over the 15 settings.

    Each Sobol point picks one value per setting; invalid points and repeats are drawn again.
    """
    from scipy.stats import qmc

    start = validate(LUNG_START)
    configs, seen = [{"config_id": config_id(start), "config": start, "source": "warm:lung_selection"}], {config_id(start)}
    axes = list(SPACE)
    sobol, draws, redrawn = qmc.Sobol(d=len(axes), scramble=True, seed=SOBOL_SEED), 0, 0
    while len(configs) < 1 + n_sobol:
        point = sobol.random(1)[0]
        draws += 1
        raw = {axis: SPACE[axis][min(int(point[j] * len(SPACE[axis])), len(SPACE[axis]) - 1)] for j, axis in enumerate(axes)}
        try:
            identity = config_id(raw)
        except (KeyError, ValueError):
            redrawn += 1
            continue
        if identity in seen:
            redrawn += 1
            continue
        seen.add(identity)
        configs.append({"config_id": identity, "config": raw, "source": f"warm:sobol:{draws - 1}"})
    return {"sobol_seed": SOBOL_SEED, "sobol_draws": draws, "sobol_redrawn": redrawn, "configs": configs}


def _warm_design(ctx: Context) -> dict:
    path = ctx.store.root / "search" / "warm_start.json"
    if not path.exists():
        write_json(path, warm_start(SMOKE_SOBOL if ctx.smoke else N_SOBOL))
    return read_json(path)


def _queue_starting_configurations(ctx: Context, study) -> None:
    """Queue every starting configuration that the study has neither finished nor in progress.

    Workers do this one at a time, so workers that start together queue each configuration once.
    A configuration whose earlier trial failed is queued again.
    """
    from optuna.trial import TrialState

    folder = ctx.store.root / "search"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "warm_start.lock")):
        present = {t.user_attrs.get("config_id") for t in study.get_trials(deepcopy=False)
                   if t.state in (TrialState.COMPLETE, TrialState.WAITING, TrialState.RUNNING)}
        for item in _warm_design(ctx)["configs"]:
            if item["config_id"] not in present:
                study.enqueue_trial({axis: json.dumps(item["config"][axis]) for axis in SPACE},
                                    user_attrs={"config_id": item["config_id"], "source": item["source"]})


def unfinished_starts(ctx: Context, study) -> list[str]:
    """IDs of the starting configurations that have neither a finished trial nor a recorded failed fit."""
    from optuna.trial import TrialState

    settled = {t.user_attrs.get("config_id") for t in study.get_trials(deepcopy=False)
               if t.state == TrialState.COMPLETE or "error" in t.user_attrs}
    return [item["config_id"] for item in _warm_design(ctx)["configs"] if item["config_id"] not in settled]


def _counts(study) -> tuple[int, int, int]:
    """(distinct new encoders proposed by NSGA-II and finished, warm-start trials still waiting, NSGA-II trials)."""
    from optuna.trial import TrialState

    trials = study.get_trials(deepcopy=False)
    finished = {t.user_attrs.get("encoder_id") for t in trials if t.state == TrialState.COMPLETE
                and t.user_attrs.get("source") == "nsga2" and t.user_attrs.get("new_encoder")}
    waiting = sum(t.state == TrialState.WAITING for t in trials)
    return len(finished), waiting, sum(t.user_attrs.get("source") == "nsga2" for t in trials)


def evaluate(ctx: Context, study, trial, config: dict) -> str:
    """Score one configuration and tell the study. Returns ``"ok"``, ``"skip"`` (too many steps) or ``"fail"``."""
    from optuna.trial import TrialState

    identity, encoder, k = config_id(config), encoder_id(config), int(config["n_clusters"])
    size = config["patch_size"]
    coverage = ctx.coverage(size)
    steps = training_steps(config, int(ctx.prepared.table(size).signal.sum()))
    earlier = [t for t in study.get_trials(deepcopy=False) if t.number != trial.number and t.state == TrialState.COMPLETE
               and t.user_attrs.get("encoder_id") == encoder]
    for key, value in {"config_id": identity, "encoder_id": encoder, "config": config, "training_steps": steps,
                       "coverage": coverage, "new_encoder": not earlier}.items():
        trial.set_user_attr(key, value)
    if steps > MAX_TRAINING_STEPS:
        trial.set_user_attr("constraint", 1.0)
        trial.set_user_attr("resource_skip", f"{steps} training steps > {MAX_TRAINING_STEPS}")
        study.tell(trial, [0.0, coverage, float(k)])
        log(f"trial {trial.number} {identity}: not fitted ({steps} training steps)")
        return "skip"
    try:
        profile = ctx.profile(config, SEARCH_SEEDS)
    except Exception as error:            # for example CUDA out of memory: record and continue
        trial.set_user_attr("error", f"{type(error).__name__}: {error}"[:500])
        study.tell(trial, state=TrialState.FAIL)
        log(f"trial {trial.number} {identity}: FAILED ({type(error).__name__}: {str(error)[:200]})")
        release_cuda()
        return "fail"
    chosen = profile[k]
    trial.set_user_attr("constraint", float(chosen["spread_deficit"]))
    trial.set_user_attr("spread_deficit", chosen["spread_deficit"])
    trial.set_user_attr("pairwise_ari", chosen["pairwise_ari"])
    trial.set_user_attr("stability_by_k", {str(key): value["stability"] for key, value in profile.items()})
    study.tell(trial, [chosen["stability"], coverage, float(k)])
    curve = " ".join(f"{key}:{profile[key]['stability']:.2f}{'' if profile[key]['spread_deficit'] == 0 else 'x'}" for key in K_VALUES)
    log(f"trial {trial.number} {identity} ({encoder}) p{size} K{k}: stability {chosen['stability']:.3f} coverage {coverage:.3f} "
        f"deficit {chosen['spread_deficit']} | by K {curve}")
    return "ok"


def target_encoders(ctx: Context, n_encoders: int | None = None) -> int:
    """New encoders that end the search: the 60 of the paper's runs by default (2 for ``--smoke``)."""
    return n_encoders or (SMOKE_ENCODERS if ctx.smoke else N_SEARCH_ENCODERS)


def run_search(ctx: Context, n_encoders: int | None = None, *, worker_index: int = 0,
               time_limit_hours: float | None = None) -> bool:
    """Evaluate the starting configurations, then NSGA-II proposals until ``n_encoders`` of them are new encoders.

    Returns ``False`` when the time limit stopped the search before that.
    """
    from optuna.trial import TrialState

    n_encoders = target_encoders(ctx, n_encoders)
    study, distributions = open_study(ctx, worker_index), _distributions()
    _queue_starting_configurations(ctx, study)
    deadline = None if time_limit_hours is None else time.time() + 3600 * time_limit_hours
    log(f"search: study has {_counts(study)[0]} new encoders, target {n_encoders}")
    failures = 0
    while True:
        finished, waiting, proposed = _counts(study)
        if (finished >= n_encoders or proposed >= MAX_NSGA_TRIALS) and not waiting:
            break
        if deadline is not None and time.time() >= deadline:
            log("search: time limit reached; rerun the same command to continue")
            return False
        trial = study.ask(distributions)
        try:
            state = _run_trial(ctx, study, trial)
        except BaseException:      # an interrupted trial is closed, so a starting configuration is queued again
            study.tell(trial, state=TrialState.FAIL, skip_if_finished=True)
            raise
        if state == "fail":
            failures += 1
        elif state == "ok":
            failures = 0
        if failures >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(f"{failures} configurations failed in a row; see the log above. "
                               "Rerun the same command to continue once the cause is fixed.")
    if finished < n_encoders:
        log(f"search ended at the limit of {MAX_NSGA_TRIALS} NSGA-II proposals: {finished} of {n_encoders} new encoders")
    else:
        log(f"search complete: {_counts(study)[0]} new encoders")
    return True


def _run_trial(ctx: Context, study, trial) -> str:
    """Validate and evaluate one asked trial. Returns the result of :func:`evaluate` or ``"pruned"``."""
    from optuna.trial import TrialState

    source = trial.user_attrs.get("source", "nsga2")
    trial.set_user_attr("source", source)
    sampled = {axis: json.loads(trial.params[axis]) for axis in SPACE}
    try:
        config = validate(sampled)
    except (KeyError, ValueError) as error:      # invalid proposals are rejected, never adapted
        trial.set_user_attr("rejected_invalid", str(error))
        study.tell(trial, state=TrialState.PRUNED)
        return "pruned"
    if source == "nsga2":
        repeats = [t for t in study.get_trials(deepcopy=False) if t.number != trial.number
                   and t.user_attrs.get("config_id") == config_id(config) and t.state in (TrialState.COMPLETE, TrialState.RUNNING)]
        if repeats:
            trial.set_user_attr("config_id", config_id(config))
            trial.set_user_attr("duplicate_of", repeats[0].number)
            study.tell(trial, state=TrialState.PRUNED)
            return "pruned"
    return evaluate(ctx, study, trial, config)


def trial_table(study):
    """One row per trial of the study (the record behind Supplementary Table 4)."""
    import pandas as pd

    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        values = trial.values or [float("nan")] * 3
        rows.append({"trial": trial.number, "state": trial.state.name, "source": attrs.get("source"),
                     "config_id": attrs.get("config_id"), "encoder_id": attrs.get("encoder_id"), "stability3": values[0],
                     "coverage": values[1], "spread_deficit": attrs.get("spread_deficit"),
                     "new_encoder": attrs.get("new_encoder"), "resource_skip": attrs.get("resource_skip"),
                     "rejected_invalid": attrs.get("rejected_invalid"), "duplicate_of": attrs.get("duplicate_of"),
                     "training_steps": attrs.get("training_steps"), **(attrs.get("config") or {})})
    return pd.DataFrame(rows)


def evaluated_encoders(study) -> dict[str, dict]:
    """Maps each encoder ID to its configuration (with the K of its first trial), for every encoder fitted by the search."""
    from optuna.trial import TrialState

    encoders = {}
    for trial in study.trials:
        attrs = trial.user_attrs
        if trial.state == TrialState.COMPLETE and not attrs.get("resource_skip") and attrs.get("config"):
            encoders.setdefault(attrs["encoder_id"], validate(attrs["config"]))
    return encoders

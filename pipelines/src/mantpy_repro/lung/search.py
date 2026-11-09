"""Nested Bayesian optimisation of the denoising settings (Supplementary Table 6).

One Optuna study per outer fold and context. The outer fold holds out one mouse (60 trials)
or one ROI (150 trials). A trial samples graph rules, features, regularisation and cleaning
parameters; its value is the label accuracy after cleaning, averaged over an inner loop that
holds out each training ROI in turn and over the two tuning realisations (seeds 100 and 101,
mixed corruption). The held-out mouse or ROI and the evaluation realisations are never read.

Each study is one result unit: it runs once, from its seed, and is then skipped. Any number of
workers can share a results directory; a study in progress is locked.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from ..runtime import calculation_threads
from .config import (CONTEXTS, PRIMARY_KIND, SCHEMES, SELECTION_TOLERANCE, SPACE, TUNING_SEEDS, CleaningRecipe, Config,
                     FeatureRecipe, GraphRecipe, LearnerRecipe, inner_folds, outer_folds, sampler_seed, startup_trials)
from .core import FIT_THREADS, Context, environment, log
from .corruption import make_realisation
from .denoise import fit_model, score_roi
from .graphs import GraphCache


def sample_config(trial, context: str) -> Config:
    """Draw one configuration from the search space.

    Only the graph layers that the context uses are sampled; the others keep the defaults of
    :class:`GraphRecipe`. The order of the draws is that of the paper's search, which the
    sampler's random stream depends on.
    """
    hops = trial.suggest_categorical("hops", SPACE["hops"])
    trial.suggest_categorical("intensity_block", [False])     # neighbouring marker intensities are excluded
    composition = trial.suggest_categorical("composition_norm", SPACE["composition_norm"])
    standardize = trial.suggest_categorical("standardize", SPACE["standardize"])
    graph = {"edge_weight": trial.suggest_categorical("edge_weight", SPACE["edge_weight"]),
             "ee_include_background_nodes": trial.suggest_categorical("ee_include_background_nodes",
                                                                      SPACE["ee_include_background_nodes"])}
    if context in ("cell", "joint") and hops == 2:            # cell-cell layer
        graph["cc_rule"] = trial.suggest_categorical("cc_rule", SPACE["cc_rule"])
        if graph["cc_rule"] == "knn":
            graph["cc_k"] = trial.suggest_int("cc_k", *SPACE["cc_k"])
            graph["cc_radius_px"] = trial.suggest_float("cc_knn_dmax", *SPACE["cc_radius_px"])
        elif graph["cc_rule"] == "radius":
            graph["cc_radius_px"] = trial.suggest_float("cc_radius_px", *SPACE["cc_radius_px"])
    if context in ("ecm", "joint"):                           # ECM-ECM layer
        graph["ee_rule"] = trial.suggest_categorical("ee_rule", SPACE["ee_rule"])
        if graph["ee_rule"] == "knn":
            graph["ee_k"] = trial.suggest_int("ee_k", *SPACE["ee_k"])
            graph["ee_radius_px"] = trial.suggest_float("ee_knn_cap", *SPACE["ee_radius_px"])
        elif graph["ee_rule"] in ("radius", "delaunay"):
            graph["ee_radius_px"] = trial.suggest_float("ee_radius_px", *SPACE["ee_radius_px"])
    if context in ("cell", "joint") or hops == 2:             # cell-ECM layer
        graph["ce_rule"] = trial.suggest_categorical("ce_rule", SPACE["ce_rule"])
        if graph["ce_rule"] == "knn":
            graph["ce_k"] = trial.suggest_int("ce_k", *SPACE["ce_k"])
        if graph["ce_rule"] in ("knn", "radius", "delaunay"):
            cap = trial.suggest_categorical("ce_dmax_px", [str(value) for value in SPACE["ce_dmax_px"]])
            graph["ce_dmax_px"] = None if cap == "None" else float(cap)

    penalty = trial.suggest_categorical("penalty", SPACE["penalty"])
    learner = {"C": trial.suggest_float("C", *SPACE["C"], log=True), "penalty": penalty,
               "class_weight": trial.suggest_categorical("class_weight", [str(v) for v in SPACE["class_weight"]]),
               "standardize": standardize}
    if penalty == "elasticnet":
        learner["l1_ratio"] = trial.suggest_float("l1_ratio", *SPACE["l1_ratio"])
    learner["class_weight"] = None if learner["class_weight"] == "None" else learner["class_weight"]

    statistic = trial.suggest_categorical("anomaly_stat", SPACE["anomaly_stat"])
    name, low, high = SPACE["threshold"][statistic]
    threshold = trial.suggest_float(name, low, high)
    radius = trial.suggest_categorical("gate_radius_px", SPACE["gate_radius_px"])
    neighbours = trial.suggest_categorical("gate_min_neighbours", SPACE["gate_min_neighbours"]) if radius > 0 else 0
    confidence = trial.suggest_categorical("min_pred_conf", SPACE["min_pred_conf"])
    smoothing = trial.suggest_categorical("majority_smoothing", SPACE["majority_smoothing"]) if radius > 0 else False
    passes = trial.suggest_categorical("passes", SPACE["passes"]) if context in ("ecm", "joint") else 1
    calibration = trial.suggest_categorical("calibration", SPACE["calibration"])
    return Config(
        context=context, graph=GraphRecipe(**graph),
        features=FeatureRecipe(composition_norm=composition, hops=hops, standardize=standardize),
        learner=LearnerRecipe(**learner),
        cleaning=CleaningRecipe(anomaly_stat=statistic, threshold=threshold, min_pred_conf=confidence,
                                gate_radius_px=radius, gate_min_neighbours=neighbours, majority_smoothing=smoothing,
                                passes=passes),
        calibration=calibration)


def inner_accuracy(config: Config, cohort, train: list[str], cache: GraphCache, realisations: dict) -> float:
    """Objective of the search: mean accuracy after cleaning over the inner folds and tuning realisations."""
    scores = []
    for fold in inner_folds(train):
        model, cleaning = fit_model(cohort, fold["train"], config, cache)
        held_out = fold["test"][0]
        for seed in TUNING_SEEDS:
            result = score_roi(model, cleaning, cohort, held_out, config, cache, realisations[seed][held_out])
            scores.append(result["accuracy_after"])
    return float(np.mean(scores))


def optimise(objective, *, n_trials: int, seed: int, n_startup_trials: int, catch: tuple) -> dict[str, np.ndarray]:
    """Run one seeded study and return its completed trials.

    ``objective(trial)`` returns the value and sets the user attributes ``config`` (JSON) and
    ``edges`` (edge count of the first training ROI, used to break ties).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=n_startup_trials, seed=seed, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, catch=catch, gc_after_trial=False)
    done = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError("The study has no completed trial.")
    return {"trial": np.asarray([trial.number for trial in done]),
            "value": np.asarray([trial.value for trial in done], dtype=float),
            "edges": np.asarray([trial.user_attrs["edges"] for trial in done]),
            "config": np.asarray([trial.user_attrs["config"] for trial in done], dtype=str),
            "n_proposed": np.asarray(len(study.trials))}


def select_best(trials: dict[str, np.ndarray]) -> int:
    """Position of the selected trial.

    Among trials within 1e-4 of the best inner accuracy, the one with the fewest graph edges
    in the first training ROI is taken, then the earliest trial. (The paper's rule also ranked
    learners by complexity between these two; that rank is the same for every logistic
    regression.)
    """
    tied = np.flatnonzero(trials["value"] >= trials["value"].max() - SELECTION_TOLERANCE)
    return int(min(tied, key=lambda i: (trials["edges"][i], trials["trial"][i])))


def search_tasks(mice: dict[str, list[str]], roi_folds: int | None = None) -> list[dict]:
    """Every study of the mouse and single-ROI schemes, for running as a cluster job array."""
    tasks = []
    for scheme in SCHEMES:
        folds = outer_folds(mice, scheme)
        if scheme == "roi" and roi_folds is not None:
            folds = folds[:roi_folds]
        tasks += [{"experiment": scheme, "fold": fold["name"], "context": context}
                  for fold in folds for context in CONTEXTS]
    return tasks


def study_unit(ctx: Context, scheme: str, fold_name: str, context: str):
    """The result unit of one study (also used to read a finished study)."""
    folds = outer_folds(ctx.cohort().mice(), scheme)
    index = [fold["name"] for fold in folds].index(fold_name)
    n_trials = ctx.budget.trials[scheme]
    request = {"scheme": scheme, "held_out": fold_name, "train": folds[index]["train"], "context": context,
               "n_trials": n_trials, "sampler": "TPE multivariate group", "sampler_seed": sampler_seed(scheme, index, context),
               "n_startup_trials": startup_trials(scheme, n_trials), "tuning_seeds": list(TUNING_SEEDS),
               "corruption": PRIMARY_KIND, "partition": ctx.partition, "labels": ctx.cohort().label_digest()}
    return ctx.store.unit("search", ctx.partition, scheme, fold_name, context, request=request), folds[index]


def run_study(ctx: Context, task: dict) -> dict[str, np.ndarray]:
    """Run (or load) the study of one outer fold and context."""
    scheme, context = task["experiment"], task["context"]
    unit, fold = study_unit(ctx, scheme, task["fold"], context)
    cohort = ctx.cohort()

    def fit():
        cache = ctx.graphs()
        realisations = {seed: make_realisation(cohort, PRIMARY_KIND, seed) for seed in TUNING_SEEDS}
        first = cohort.rois[fold["train"][0]]

        def objective(trial):
            config = sample_config(trial, context)
            value = inner_accuracy(config, cohort, fold["train"], cache, realisations)
            trial.set_user_attr("config", json.dumps(config.to_dict()))
            trial.set_user_attr("edges", sum(cache.get(first, config.graph).n_edges().values()))
            return value

        with calculation_threads(FIT_THREADS):
            trials = optimise(objective, n_trials=unit.request["n_trials"], seed=unit.request["sampler_seed"],
                              n_startup_trials=unit.request["n_startup_trials"],
                              catch=(ValueError, np.linalg.LinAlgError))
        return trials, {"environment": environment()}

    return ctx.store.compute(unit, fit, label=f"study {scheme} {task['fold']} {context}", log=log)

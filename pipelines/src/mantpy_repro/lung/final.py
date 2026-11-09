"""Final fits of the selected configurations (Fig. 4h-j and the controls).

For every outer fold and context the selected configuration is refitted once on the pristine
ROIs of the training side and scored once on each held-out ROI, for the three corruption
types and the five evaluation realisations (seeds 0-4). The label-prior baseline is scored on
the same folds. The folds are: each mouse held out (main result), each ROI held out (control),
and each ROI held out with the ECM foreground and clusters refitted within the training set
(control, see ``refit``).
"""

from __future__ import annotations

import numpy as np

from ..runtime import calculation_threads
from .config import CONTEXTS, KINDS, PRIOR, SCHEMES, Config, outer_folds
from .core import FIT_THREADS, Context, environment, log, pack
from .corruption import make_realisation
from .denoise import detection_metrics, fit_model, score_roi, training_prior
from .graphs import GraphCache
from .refit import Targets, refit_tasks, run_final as run_refit
from .selection import load_selection, selected_config


def final_tasks(mice: dict[str, list[str]], roi_folds: int | None = None) -> list[dict]:
    """Every independent final fit, for running as a cluster job array."""
    tasks = []
    for scheme in SCHEMES:
        folds = outer_folds(mice, scheme)
        if scheme == "roi" and roi_folds is not None:
            folds = folds[:roi_folds]
        tasks += [{"experiment": scheme, "fold": fold["name"], "context": context}
                  for fold in folds for context in (*CONTEXTS, PRIOR)]
    return tasks + refit_tasks(mice, roi_folds)


def score_fold(cohort, config: Config, train: list[str], test: list[str], cache: GraphCache, seeds, fit_seed: int = 0):
    """Fit ``config`` on ``train`` and score every ROI of ``test``.

    Returns one row of scores per corruption type, realisation and held-out ROI, the counts of
    replaced and restored labels per true cluster, the cleaned labels of the first mixed
    realisation of each ROI, and the fitted temperature and training label frequencies.
    The prior-adjusted detection score is -log[P(observed | context) / P_train(observed)]; it
    is not defined for the label prior itself.
    """
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score

    model, cleaning = fit_model(cohort, train, config, cache, seed=fit_seed)
    prior = training_prior(cohort, train)
    if not np.all(prior > 0):
        raise ValueError("A cluster is absent from the training ROIs.")
    rows, classes, cleaned = [], [], {}
    for kind in KINDS:
        for seed in seeds:
            realisation = make_realisation(cohort, kind, seed)
            for name in test:
                out = score_roi(model, cleaning, cohort, name, config, cache, realisation[name])
                signal = out["signal"]
                index = np.flatnonzero(signal)
                adjusted = np.zeros(len(signal))
                adjusted[index] = out["anomaly"][index] + np.log(prior[out["observed"][index]])
                adjusted_scores = ({"roc_auc": np.nan, "pr_auc": np.nan} if config.context == PRIOR
                                   else detection_metrics(adjusted, out["is_artifact"], signal))
                base = {"context": config.context, "roi": name, "mouse": cohort.rois[name].mouse, "kind": kind,
                        "seed": seed}
                truth, predicted = out["pristine"][signal], out["cleaned"][signal]
                rows.append({
                    **base,
                    **{key: out[key] for key in ("roc_auc", "pr_auc", "accuracy_before", "accuracy_after", "recovery",
                                                 "pristine_relabel_rate", "correction_precision", "n_relabelled",
                                                 "n_artifact", "n_signal")},
                    "prior_adjusted_roc_auc": adjusted_scores["roc_auc"],
                    "prior_adjusted_pr_auc": adjusted_scores["pr_auc"],
                    "macro_f1_after": f1_score(truth, predicted, labels=np.arange(cohort.n_labels), average="macro",
                                               zero_division=0),
                    "balanced_accuracy_after": balanced_accuracy_score(truth, predicted),
                })
                for cluster in range(cohort.n_labels):
                    replaced = signal & out["is_artifact"] & (out["pristine"] == cluster)
                    classes.append({**base, "cluster": cluster, "n_corrupted": int(replaced.sum()),
                                    "n_restored": int((out["cleaned"][replaced] == cluster).sum())})
                if kind == KINDS[0] and seed == seeds[0]:
                    cleaned[name] = out["cleaned"]
    return pd.DataFrame(rows), pd.DataFrame(classes), cleaned, {"temperature": cleaning.temperature,
                                                               "training_prior": prior.tolist()}


def run_fold(ctx: Context, task: dict, config: Config, selection: str) -> dict[str, np.ndarray]:
    """Final fit of one outer fold and context of the mouse or single-ROI scheme."""
    scheme, context = task["experiment"], task["context"]
    cohort = ctx.cohort()
    fold = next(fold for fold in outer_folds(cohort.mice(), scheme) if fold["name"] == task["fold"])
    seeds = ctx.budget.evaluation_seeds
    request = {"scheme": scheme, "held_out": fold["name"], "train": fold["train"], "test": fold["test"],
               "context": context, "selection": selection, "config": config.to_dict(), "kinds": list(KINDS),
               "evaluation_seeds": list(seeds), "fit_seed": 0, "partition": ctx.partition,
               "labels": cohort.label_digest()}
    unit = ctx.store.unit("final", ctx.partition, selection, scheme, fold["name"], context, request=request)

    def fit():
        with calculation_threads(FIT_THREADS):
            scores, classes, cleaned, fitted = score_fold(cohort, config, fold["train"], fold["test"], ctx.graphs(), seeds)
        arrays = {**pack(scores, "scores__"), **pack(classes, "classes__"),
                  **{f"cleaned__{name}": labels.astype(np.int16) for name, labels in cleaned.items()}}
        return arrays, {**fitted, "environment": environment()}

    return ctx.store.compute(unit, fit, label=f"final {scheme} {fold['name']} {context}", log=log)


class FinalRunner:
    """Runs final tasks with one selection; keeps the control's target definitions between tasks."""

    def __init__(self, ctx: Context, selection: str):
        self.ctx, self.selection = ctx, selection
        self.chosen = load_selection(ctx, selection)
        self._targets: Targets | None = None

    def run(self, task: dict) -> dict[str, np.ndarray]:
        config = selected_config(self.chosen, task)
        if task["experiment"] != "refit":
            return run_fold(self.ctx, task, config, self.selection)
        if self._targets is None:
            self._targets = Targets(self.ctx, self.selection, self.chosen["refit_control"])
        return run_refit(self.ctx, task, config, self.selection, self._targets)

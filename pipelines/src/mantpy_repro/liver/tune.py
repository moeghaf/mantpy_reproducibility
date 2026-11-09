"""Within-fold tuning of the classifier (Supplementary Table 5).

For each held-out mouse, 80 configurations of the graph, layer and training settings are proposed
by Optuna's TPE sampler and scored on the seven training mice only: each training mouse is left
out in turn (two training seeds), and the score is the mean mouse-level log loss. The best
configuration is then refitted with five seeds on the seven mice and the held-out mouse is
predicted once. These models use the loss pooled over patches.

One proposed configuration is one resumable unit (14 fits). A fold that is interrupted replays
its finished configurations into a new sampler when it is run again, so the configurations
proposed after that point differ from those of an uninterrupted run.
"""

from __future__ import annotations

import numpy as np

from ..runtime import environment, release_cuda
from ..store import digest
from . import models
from .classify import fold_fit, mouse_labels
from .config import (CLASSIFIER_SEEDS, FAIL_OBJECTIVE, GRAPH_TYPES, INNER_SEEDS, N_STARTUP_TRIALS, N_TUNING_TRIALS,
                     TUNING_EPOCHS, TUNING_HIDDEN, paper_selection)
from .core import Context, Partition, log

SMOKE_TRIALS = 3

_CATEGORICAL = {"graph_type": GRAPH_TYPES, "conv": ["sage", "gcn", "gat", "gin"], "hidden_dim": TUNING_HIDDEN,
                "aggr": ["mean", "max", "add"], "heads": [1, 2, 4], "train_eps": [False, True],
                "norm": ["none", "layer", "batch"], "residual": [False, True], "activation": ["relu", "gelu"],
                "optimizer": ["adam", "adamw"], "lr_schedule": ["constant", "cosine"], "epochs": TUNING_EPOCHS,
                "class_weight": [True, False]}
_FLOAT = {"dropout": (0.0, 0.6, False), "input_dropout": (0.0, 0.3, False), "lr": (1e-4, 3e-2, True),
          "weight_decay": (1e-6, 1e-2, True), "label_smoothing": (0.0, 0.2, False)}


def suggest(trial) -> dict:
    """Draw one configuration. Aggregation, attention heads and the GIN epsilon exist only for their layer type."""
    config = {"graph_type": trial.suggest_categorical("graph_type", _CATEGORICAL["graph_type"]),
              "conv": trial.suggest_categorical("conv", _CATEGORICAL["conv"]),
              "hidden_dim": trial.suggest_categorical("hidden_dim", _CATEGORICAL["hidden_dim"]),
              "n_layers": trial.suggest_int("n_layers", 1, 4)}
    conditional = {"sage": "aggr", "gat": "heads", "gin": "train_eps"}.get(config["conv"])
    if conditional:
        config[conditional] = trial.suggest_categorical(conditional, _CATEGORICAL[conditional])
    for name in ("norm", "residual", "activation", "optimizer", "lr_schedule"):
        config[name] = trial.suggest_categorical(name, _CATEGORICAL[name])
    for name in ("dropout", "input_dropout", "lr", "weight_decay"):
        low, high, log_scale = _FLOAT[name]
        config[name] = trial.suggest_float(name, low, high, log=log_scale)
    config["epochs"] = trial.suggest_categorical("epochs", _CATEGORICAL["epochs"])
    config["class_weight"] = trial.suggest_categorical("class_weight", _CATEGORICAL["class_weight"])
    config["label_smoothing"] = trial.suggest_float("label_smoothing", *_FLOAT["label_smoothing"][:2])
    return config


def distributions(config: dict) -> dict:
    """Optuna distributions of the settings present in ``config`` (to replay a finished configuration)."""
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

    out = {}
    for name in config:
        if name in _CATEGORICAL:
            out[name] = CategoricalDistribution(_CATEGORICAL[name])
        elif name == "n_layers":
            out[name] = IntDistribution(1, 4)
        else:
            low, high, log_scale = _FLOAT[name]
            out[name] = FloatDistribution(low, high, log=log_scale)
    return out


def inner_folds(training_mice: list[str], truth: dict[str, int]) -> list[tuple[list[str], str]]:
    """Leave out each training mouse in turn; a mouse that is the only one of its class is never left out."""
    folds = []
    for mouse in sorted(training_mice):
        if sum(truth[other] == truth[mouse] for other in training_mice) >= 2:
            folds.append(([other for other in training_mice if other != mouse], mouse))
    return folds


def inner_objective(ctx: Context, partition: Partition, config: dict, training_mice: list[str]) -> float:
    """Mean, over inner folds and the two training seeds, of -log p(true class) of the left-out mouse.

    As in the paper's runs, the random generators are seeded once per training seed and the inner
    folds are then trained one after another.
    """
    import torch

    graphs, table, truth = ctx.graphs(partition, config["graph_type"]), ctx.roi_table(), mouse_labels(ctx)
    labels, mouse_of = table.infected.to_numpy(), table.mouse.to_numpy()
    losses = []
    for seed in INNER_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for rest, left_out in inner_folds(training_mice, truth):
            train, test = np.flatnonzero(np.isin(mouse_of, rest)), np.flatnonzero(mouse_of == left_out)
            fit = models.fit_classifier([graphs[i] for i in train], labels[train], ctx.classifier_budget(config), seed=None,
                                        objective="patch_sum", device=ctx.device, threads=ctx.threads)
            probability = np.stack([models.roi_probabilities(fit, graphs[i]) for i in test]).mean(axis=0)
            losses.append(-np.log(np.clip(float(probability[truth[left_out]]), 1e-12, None)))
    return float(np.mean(losses))


def _trial_unit(ctx: Context, partition: Partition, held_mouse: str, number: int, config: dict):
    request = {"partition": partition.identity, "held_mouse": held_mouse, "trial": number, "config": config,
               "epochs": ctx.classifier_budget(config)["epochs"], "inner_seeds": list(INNER_SEEDS), "device": ctx.device}
    return ctx.store.unit("tuning", partition.folder, held_mouse, f"trial_{number:03d}", request=request)


def tune_fold(ctx: Context, partition: Partition, held_mouse: str, n_trials: int | None = None) -> dict:
    """Tune the classifier for one held-out mouse and return the selected configuration record.

    ``n_trials`` defaults to the paper's 80 configurations (3 for ``--smoke``).
    """
    import optuna

    from ..store import read_json

    n_trials = n_trials or (SMOKE_TRIALS if ctx.smoke else N_TUNING_TRIALS)
    training_mice = [mouse for mouse in mouse_labels(ctx) if mouse != held_mouse]
    root = ctx.store.root / "tuning" / partition.folder / held_mouse
    summary = ctx.store.unit("tuning", partition.folder, held_mouse, "selected",
                             request={"partition": partition.identity, "held_mouse": held_mouse, "n_trials": n_trials,
                                      "inner_seeds": list(INNER_SEEDS), "device": ctx.device})

    def evaluate(config):
        try:
            return {"objective": np.asarray(inner_objective(ctx, partition, config, training_mice)),
                    "failed": np.asarray(False)}, {"reason": "", "environment": environment()}
        except Exception as error:      # for example out of memory, or a graph over the size limit
            release_cuda()
            return {"objective": np.asarray(FAIL_OBJECTIVE), "failed": np.asarray(True)}, {
                "reason": f"{type(error).__name__}: {str(error)[:200]}"}

    def fit():
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=0, n_startup_trials=N_STARTUP_TRIALS)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        records = []
        for number in range(n_trials):          # replay the configurations finished by an interrupted run
            marker = root / f"trial_{number:03d}" / "completion.json"
            if not marker.exists():
                break
            saved = read_json(marker)
            config = saved["request"]["config"]
            arrays = _trial_unit(ctx, partition, held_mouse, number, config).load()
            study.add_trial(optuna.trial.create_trial(params=config, distributions=distributions(config),
                                                      value=float(arrays["objective"])))
            records.append({"trial": number, "config": config, "objective": float(arrays["objective"]),
                            "failed": bool(arrays["failed"]), "reason": saved["metadata"].get("reason", "")})
        for number in range(len(records), n_trials):
            trial = study.ask()
            config = suggest(trial)
            unit = _trial_unit(ctx, partition, held_mouse, number, config)
            arrays = ctx.store.compute(unit, lambda: evaluate(config), label=f"tuning held {held_mouse} trial {number}", log=log)
            study.tell(trial, float(arrays["objective"]))
            records.append({"trial": number, "config": config, "objective": float(arrays["objective"]),
                            "failed": bool(arrays["failed"]), "reason": unit.metadata()["metadata"].get("reason", "")})
            if records[-1]["failed"]:
                log(f"tuning held {held_mouse} trial {number}: could not be fitted ({records[-1]['reason']})")
        complete = [record for record in records if not record["failed"]]
        if not complete:
            raise RuntimeError(f"No configuration could be fitted for held-out mouse {held_mouse}: {records[0]['reason']}")
        best = min(complete, key=lambda record: (record["objective"], record["trial"]))
        return {"objective": np.asarray([record["objective"] for record in records]),
                "failed": np.asarray([record["failed"] for record in records])}, {
            "held_mouse": held_mouse, "trial": best["trial"], "inner_objective": best["objective"], "config": best["config"],
            "n_trials": n_trials, "trials": records}

    ctx.store.compute(summary, fit, label=f"tuning held {held_mouse}", log=log)
    return summary.metadata()["metadata"]


def tuned_record(ctx: Context, partition: Partition, held_mouse: str, selection: str) -> dict | None:
    """The tuned configuration of one fold, with its trial number and inner log loss: the paper's, or the
    one selected by ``tune`` in this results directory (None if that fold has not been tuned)."""
    from ..store import read_json

    if selection == "paper":
        record = paper_selection()["tuned_classifier"][held_mouse]
    else:
        marker = ctx.store.root / "tuning" / partition.folder / held_mouse / "selected" / "completion.json"
        if not marker.exists():
            return None
        record = read_json(marker)["metadata"]
    return {"trial": record["trial"], "inner_log_loss": record["inner_objective"], "config": dict(record["config"])}


def tuned_parts(partition: Partition, config: dict) -> tuple[str, ...]:
    return ("tuned", partition.folder, "c" + digest(config))


def fit_tuned_fold(ctx: Context, partition: Partition, held_mouse: str, config: dict) -> None:
    """Five seeds of a tuned configuration on the seven training mice; the held-out mouse is predicted once."""
    labels = mouse_labels(ctx)
    for seed in CLASSIFIER_SEEDS:
        fold_fit(ctx, partition, parts=tuned_parts(partition, config), config=config, objective="patch_sum",
                 held_mouse=held_mouse, seed=seed, infected=tuple(mouse for mouse, label in labels.items() if label),
                 device=ctx.device)

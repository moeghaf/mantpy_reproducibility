"""Collect the selected configuration of every study, and resolve which selection to use.

``select`` applies the tie rule of ``search.select_best`` to each study of this results
directory and writes the selected configurations in the layout of ``paper_selection.json``.
Every study must be finished. No held-out score is read.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..store import read_json, write_json
from .config import CONTEXTS, NORMALIZATIONS, SCHEMES, Config, GraphRecipe, config_id, outer_folds, paper_selection
from .core import Context, log
from . import refit, search


def selection_path(ctx: Context) -> Path:
    return ctx.store.root / "selection" / f"search_{ctx.partition}.json"


def _entry(name: str, trials: dict) -> dict:
    best = search.select_best(trials)
    config = json.loads(str(trials["config"][best]))
    return {"study": name, "selected_trial": int(trials["trial"][best]),
            "inner_selection_score": float(trials["value"][best]), "n_trials": int(trials["n_proposed"]),
            "config_id": config_id(Config.from_dict(config)), "config": config}


def select(ctx: Context) -> dict:
    """Write the selection of this directory's searches; every study must be finished."""
    mice = ctx.cohort().mice()
    result: dict = {"partition": ctx.partition, "mice": mice, "mouse": {}, "roi": {},
                    "refit_control": {"normalization_selection": {}, "target_definition": {}}}
    missing = []
    for scheme in SCHEMES:
        folds = outer_folds(mice, scheme)
        for fold in ctx.roi_folds(folds) if scheme == "roi" else folds:
            for context in CONTEXTS:
                unit, _ = search.study_unit(ctx, scheme, fold["name"], context)
                name = f"{scheme}__{fold['name']}__{context}"
                if not unit.done():
                    missing.append(name)
                    continue
                trials = unit.load()
                result[scheme].setdefault(fold["name"], {})[context] = _entry(name, trials)
    for normalization in NORMALIZATIONS:
        for fold in ctx.roi_folds(outer_folds(mice, "roi")):
            train_key = refit.train_key(fold["train"])
            definition = ctx.store.root / "refit" / "targets" / normalization / train_key / "selected_k"
            for context in CONTEXTS:
                unit, _ = refit.study_unit(ctx, normalization, fold["name"], context)
                name = f"refit__{normalization}__{fold['name']}__{context}"
                if not unit.done():
                    missing.append(name)
                    continue
                trials = unit.load()
                result["refit_control"]["normalization_selection"].setdefault(normalization, {}).setdefault(
                    fold["name"], {})[context] = _entry(name, trials)
                record = read_json(definition / "completion.json")["metadata"]["record"]
                result["refit_control"]["target_definition"].setdefault(normalization, {})[fold["name"]] = {
                    "training_rois": record["training_rois"], "selected_signal_k": record["signal_k"]}
    if missing:
        raise RuntimeError(f"{len(missing)} studies are not finished (first: {missing[0]}). "
                           "Run the 'search' stage until it completes.")
    write_json(selection_path(ctx), result)
    log(f"selection written to {selection_path(ctx)}")
    return result


def load_selection(ctx: Context, selection: str) -> dict:
    """The paper's selected settings, or those written by ``select`` in this results directory."""
    if selection == "paper":
        return paper_selection()
    if selection == "search":
        path = selection_path(ctx)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the 'search' and 'select' stages first.")
        return read_json(path)
    raise ValueError("selection must be 'paper' or 'search'.")


def selected_config(chosen: dict, task: dict) -> Config:
    """Configuration of one final fit. The label prior has no settings of its own; it is
    scored with the graph of Fig. 4."""
    if task["context"] == "prior":
        return Config(context="prior", graph=GraphRecipe())
    if task["experiment"] == "refit":
        saved = chosen["refit_control"]["normalization_selection"][task["normalization"]][task["fold"]]
    else:
        saved = chosen[task["experiment"]][task["fold"]]
    return Config.from_dict(saved[task["context"]]["config"])

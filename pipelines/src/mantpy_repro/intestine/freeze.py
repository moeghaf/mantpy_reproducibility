"""Choose one configuration per image from the search, without the annotation.

Rule of the paper:

1. keep the feasible configurations whose coverage is at least that of 32-pixel patches;
2. take the most stable over three seeds, and every configuration within 0.01 of it (the band);
3. fit the band with five seeds (the three of the search and two further ones) and drop any
   that is not feasible over five seeds;
4. select the most stable over five seeds. Exact ties go to the configuration with the shorter
   mean GraphMAE training time, then to the configuration ID.
"""

from __future__ import annotations

import numpy as np

from ..store import read_json, write_json
from .config import SPACE, paper_selection, validate
from .core import MIN_FRACTION, Context, log, min_fraction, stability
from .search import SEARCH_SEEDS, missing_evaluations, n_complete, search_table

TIE = 0.01
FREEZE_SEEDS = (0, 1, 2, 3, 4)


def selection_path(ctx: Context, image: str):
    return ctx.store.root / "selection" / f"{image}.json"


def tie_band(table, coverage_32px: float, require_feasible: bool = True):
    """Steps 1 and 2 of the rule: the configurations to fit with five seeds."""
    complete = table[table["status"] == "complete"]
    if require_feasible:
        complete = complete[complete["feasible"].astype(bool)]
    eligible = complete[complete["coverage"] >= coverage_32px - 1e-9].sort_values(
        ["stability3", "config_id"], ascending=[False, True])
    if eligible.empty:
        return eligible
    return eligible[eligible["stability3"] >= eligible["stability3"].iloc[0] - TIE]


def choose(records: list[dict], require_feasible: bool = True) -> dict:
    """Steps 3 and 4 of the rule, given the five-seed records of the band."""
    feasible = [record for record in records if record["min_fraction"] >= MIN_FRACTION or not require_feasible]
    if not feasible:
        raise RuntimeError("No configuration of the band stays feasible over five seeds.")
    return min(feasible, key=lambda record: (-record["stability5"], record["mean_fit_seconds"], record["config_id"]))


def require_finished(ctx: Context, image: str, mode: str = "paper", n_trials: int | None = None) -> None:
    """Raise unless the search of ``image`` evaluated everything it was asked to.

    ``mode`` and ``n_trials`` are those of the search: in ``"paper"`` mode the first ``n_trials``
    of the paper's configurations (all of them by default), in ``"nsga"`` mode ``n_trials``
    fitted configurations.
    """
    if mode == "paper":
        missing = missing_evaluations(ctx, image, n_trials)
        if not missing:
            return
        state = f"{len(missing)} of the paper's configurations have no evaluation"
        hint = " (every --shard part); after a new NSGA-II search, pass --search-mode nsga to 'freeze'"
    else:
        fitted = n_complete(ctx, image)
        if n_trials is None or fitted >= n_trials:
            return
        state, hint = f"{fitted} of {n_trials} configurations fitted", " with the same --n-trials"
    raise RuntimeError(f"The search of {image} in {ctx.store.root} is not finished: {state}. "
                       f"Run the 'search' stage again{hint}.")


def freeze(ctx: Context, image: str, mode: str = "paper", n_trials: int | None = None) -> dict:
    """Apply the selection rule to the finished search of ``image`` and write the selection.

    ``mode`` and ``n_trials`` describe the search that was run (see :func:`require_finished`).
    """
    require_finished(ctx, image, mode, n_trials)
    prepared = ctx.prepared(image)
    table = search_table(ctx, image)
    if table.empty:
        raise RuntimeError(f"No search evaluations for {image} in {ctx.store.root}: run the 'search' stage first.")
    incomplete = int((table["status"] != "complete").sum())
    if incomplete:
        hint = "; running the 'search' stage again on a GPU with more memory fits them again" if mode == "paper" else ""
        log(f"{image}: {incomplete} searched configuration(s) ran out of GPU memory and cannot be selected{hint}")
    level32 = prepared.support.coverage[32]
    # The tiny training budgets of --smoke often leave a cluster below 1% of the anchors, so a
    # smoke run applies the rule without the cluster-size condition.
    strict = not ctx.smoke
    band = tie_band(table, level32, require_feasible=strict)
    if band.empty:
        raise RuntimeError(f"No feasible configuration of {image} reaches the 32-pixel coverage ({level32:.3f}).")
    log(f"{image}: {len(band)} configuration(s) within {TIE} of the best three-seed stability; fitting five seeds")
    records = []
    for row in band.to_dict("records"):
        config = validate({axis: _plain(row[axis]) for axis in SPACE})
        # The search stores no embeddings, so GraphMAE is trained again for its three seeds (their CNN
        # features are reused). The embeddings kept here let the final stage reuse all five fits.
        runs = [ctx.run_unit(image, config, seed, keep_embedding=True) for seed in FREEZE_SEEDS]
        common = [run["common_graphmae"] for run in runs]
        stability5, pairs = stability(common)
        seconds = [ctx.unit_metadata(image, config, seed)["metadata"]["graph_seconds"] for seed in FREEZE_SEEDS]
        smallest = min_fraction(common, prepared.k)
        records.append({"config_id": row["config_id"], "config": config, "coverage": float(row["coverage"]),
                        "stability3": float(row["stability3"]), "stability5": stability5,
                        "stability5_min_pair": float(min(pairs)), "min_fraction": smallest,
                        "feasible": bool(smallest >= MIN_FRACTION), "mean_fit_seconds": float(np.mean(seconds))})
    selected = choose(records, require_feasible=strict)
    result = {"image": image, "rule": __doc__.strip(), "search_seeds": list(SEARCH_SEEDS),
              "freeze_seeds": list(FREEZE_SEEDS), "coverage_32px": level32, "cluster_size_condition": strict,
              "n_configurations": int((table["status"] == "complete").sum()), "selected": selected,
              "tie_band": records, "paper_selection": paper_selection()["images"][image]["joint_id"].split("/")[-1]}
    write_json(selection_path(ctx, image), result)
    same = "the paper's selection" if selected["config_id"] == result["paper_selection"] else \
        f"not the paper's selection ({result['paper_selection']})"
    log(f"{image}: selected {selected['config_id']} (five-seed stability {selected['stability5']:.3f}); {same}")
    return result


def _plain(value):
    """Convert NumPy scalars from a table back to the plain values of the search space."""
    return value.item() if isinstance(value, np.generic) else value


def selected_config(ctx: Context, image: str, selection: str = "paper") -> tuple[dict, str]:
    """The configuration used for the final fits.

    ``"paper"`` is the configuration selected in the paper; ``"search"`` is the result of the
    ``freeze`` stage in this results directory.
    """
    if selection == "paper":
        return validate(paper_selection()["images"][image]["joint"]), "paper"
    if selection == "search":
        path = selection_path(ctx, image)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the 'search' and 'freeze' stages first.")
        return validate(read_json(path)["selected"]["config"]), "search"
    raise ValueError("selection must be 'paper' or 'search'.")

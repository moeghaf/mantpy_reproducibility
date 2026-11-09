"""Choose the encoder and K from the finished search, without any label.

Rule of the paper, fixed before any encoder was trained:

1. candidates are the (encoder, K) pairs whose coverage is at least that of 10-pixel patches and
   whose clusters meet the spread guard in all three search seeds;
2. at each K, the pairs within 0.01 of the best three-seed stability are refitted with two more
   seeds;
3. a pair is eligible when it meets the spread guard on five seeds and its five-seed stability is
   at least 0.80;
4. K is the largest value with an eligible pair, and the most stable eligible encoder at that K is
   selected (ties go to the faster fit, then to the configuration ID).

If no pair is eligible, the most stable pair meeting the guard on five seeds is selected and the
result is marked ``rule_met = false``; the 0.80 threshold is never lowered.
"""

from __future__ import annotations

from ..store import read_json, write_json
from .config import FREEZE_SEEDS, K_VALUES, SEARCH_SEEDS, STABILITY_MIN, TIE, config_id, paper_selection, validate, with_k
from .core import Context, log
from .search import evaluated_encoders, open_study, trial_table

COVERAGE_REFERENCE_SIZE = 10


def tie_bands(pool):
    """Step 2: at each K, the candidate pairs within ``TIE`` of the best three-seed stability."""
    best = pool.groupby("K").stability3.max()
    return pool[pool.stability3 >= pool.K.map(best) - TIE].copy()


def select(candidates) -> tuple[dict, bool]:
    """Steps 3 and 4 applied to the refitted pairs.

    ``candidates`` needs the columns K, config_id, stability5, spread_deficit5 and mean_fit_seconds5.
    Returns the selected row and whether the rule was met.
    """
    guarded = candidates[candidates.spread_deficit5 <= 0]
    eligible = guarded[guarded.stability5 >= STABILITY_MIN]
    rule_met = not eligible.empty
    chosen = eligible[eligible.K == eligible.K.max()] if rule_met else guarded
    if chosen.empty:
        raise RuntimeError("No refitted encoder and K meet the spread guard on five seeds.")
    tied = chosen[chosen.stability5 >= chosen.stability5.max() - TIE]
    tied = tied.sort_values(["stability5", "mean_fit_seconds5", "config_id"], ascending=[False, True, True])
    return tied.iloc[0].to_dict(), rule_met


def _profiles(ctx: Context, encoders: dict[str, dict], seeds, suffix: str):
    """One row per (encoder, K): stability over ``seeds``, largest spread-guard deficit, mean fit time."""
    import pandas as pd

    rows = []
    for identity in sorted(encoders):
        config = encoders[identity]
        profile, seconds = ctx.profile(config, seeds), ctx.mean_fit_seconds(config, seeds)
        for k in K_VALUES:
            rows.append({"encoder_id": identity, "K": k, "config_id": config_id(with_k(config, k)),
                         "patch_size": config["patch_size"], "coverage": ctx.coverage(config["patch_size"]),
                         f"stability{suffix}": profile[k]["stability"], f"spread_deficit{suffix}": profile[k]["spread_deficit"],
                         f"mean_fit_seconds{suffix}": seconds})
    return pd.DataFrame(rows)


def selection_path(ctx: Context):
    return ctx.store.root / "selection" / "selection.json"


def freeze(ctx: Context) -> dict:
    """Apply the selection rule to the finished search and write the selection."""
    study = open_study(ctx)
    out = selection_path(ctx).parent
    out.mkdir(parents=True, exist_ok=True)
    trial_table(study).to_csv(out / "all_trials.csv", index=False)
    encoders = evaluated_encoders(study)
    if not encoders:
        raise RuntimeError("The search has not evaluated any encoder yet.")
    three = _profiles(ctx, encoders, SEARCH_SEEDS, "3")
    three.to_csv(out / "encoder_profiles.csv", index=False)
    level = ctx.coverage(COVERAGE_REFERENCE_SIZE)
    pool = three[(three.coverage >= level - 1e-9) & (three.spread_deficit3 <= 0)]
    if pool.empty:
        raise RuntimeError("No encoder and K reach the coverage of 10-pixel patches and meet the spread guard in three seeds.")
    band = tie_bands(pool)
    refit = {identity: encoders[identity] for identity in sorted(band.encoder_id.unique())}
    log(f"selection: {len(pool)} candidate pairs from {pool.encoder_id.nunique()} encoders; "
        f"refitting {len(refit)} encoder(s) with seeds {list(FREEZE_SEEDS[len(SEARCH_SEEDS):])}")
    five = _profiles(ctx, refit, FREEZE_SEEDS, "5")
    candidates = band.merge(five[["encoder_id", "K", "stability5", "spread_deficit5", "mean_fit_seconds5"]], on=["encoder_id", "K"])
    candidates["eligible"] = (candidates.spread_deficit5 <= 0) & (candidates.stability5 >= STABILITY_MIN)
    candidates = candidates.sort_values(["K", "stability5"], ascending=[False, False])
    candidates.to_csv(out / "selection_candidates.csv", index=False)
    selected, rule_met = select(candidates)
    config = with_k(encoders[selected["encoder_id"]], int(selected["K"]))
    result = {"rule": __doc__.strip(), "rule_met": rule_met, "config_id": config_id(config), "encoder_id": selected["encoder_id"],
              "config": config, "K": int(selected["K"]), "stability3": float(selected["stability3"]),
              "stability5": float(selected["stability5"]), "coverage": float(selected["coverage"]),
              "coverage_10px": level, "n_encoders": len(encoders), "n_candidate_pairs": int(len(pool)),
              "candidates": candidates.to_dict("records")}
    write_json(selection_path(ctx), result)
    log(f"selected {result['config_id']} ({result['encoder_id']}): {config['patch_size']}-pixel patches, K = {result['K']}, "
        f"five-seed stability {result['stability5']:.3f}, rule met: {rule_met}")
    return result


def selected_config(ctx: Context, selection: str) -> dict:
    """The encoder settings and K used downstream: the paper's, or this results directory's selection."""
    if selection == "paper":
        return validate(paper_selection()["config"])
    if selection == "search":
        path = selection_path(ctx)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the 'search' and 'freeze' stages first.")
        return validate(read_json(path)["config"])
    raise ValueError("selection must be 'paper' or 'search'.")

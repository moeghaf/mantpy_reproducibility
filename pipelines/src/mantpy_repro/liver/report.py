"""Collect the finished fits, compute the descriptive statistics and compare them with the paper's numbers.

No model is trained here. Analyses whose fits are not finished are left out and named in the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import descriptive
from .classify import allocations, genotype_preserving, mouse_labels, permutation_p, roi_mean_parts, scores
from .config import (ANCHOR_SIZE, CLASSIFIER_SEEDS, FREEZE_SEEDS, GROUPS, K_VALUES, MARKERS, PATCH_SIZES, SEARCH_SEEDS, config_id,
                     paper_partition, paper_reference)
from .core import Context, Partition, log
from .final import OBSERVED_ALLOCATION, final_tasks, resolve_partition, task_ranges
from .freeze import selected_config
from .tune import tuned_parts, tuned_record

#: Largest absolute difference from the paper treated as agreement, by kind of number.
TOLERANCE = {
    "count": 0.0,             # integers and P values (compared as counts)
    "patches": 100.0,         # ECM signal patches of a recomputed background split (it changes with library versions)
    "anchors": 4000.0,        # 5-pixel signal patches of that split (up to 1% change with threads and platform)
    "coverage": 0.005,        # coverage of a patch size, from the recomputed background split
    "descriptive": 1e-4,      # deterministic given the partition: bounds, fractions, co-occurrence, subgraphs
    "regression": 0.005,      # logistic-regression probabilities
    "probability": 0.05,      # classifier probabilities (they change with the thread count and the platform)
    "stability": 0.02,        # adjusted Rand indices of retrained encoders
    "correlation": 0.02,      # integrated-gradient localisation
}


class Comparison:
    """Collects the rows of the comparison table: this run's value, the paper's value, the difference and the tolerance."""

    def __init__(self):
        self.rows = []

    def add(self, quantity: str, value, paper, kind: str) -> None:
        value, paper = float(value), float(paper)
        self.rows.append({"quantity": quantity, "this_run": value, "paper": paper, "difference": value - paper,
                          "tolerance": TOLERANCE[kind], "within_tolerance": abs(value - paper) <= TOLERANCE[kind]})

    def table(self):
        import pandas as pd

        return pd.DataFrame(self.rows, columns=["quantity", "this_run", "paper", "difference", "tolerance", "within_tolerance"])


def _arrays(ctx: Context, *parts: str):
    """Arrays of a finished unit, or ``None``."""
    folder = ctx.store.root.joinpath(*parts)
    if not (folder / "completion.json").exists():
        return None
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _fold_predictions(ctx: Context, parts: tuple[str, ...], table, truth: np.ndarray):
    """Held-out ROI probabilities of every mouse and seed under ``parts`` (None unless all are finished)."""
    import pandas as pd

    rows = []
    for mouse in sorted(table.mouse.unique()):
        held = table[table.mouse == mouse]
        for seed in CLASSIFIER_SEEDS:
            arrays = _arrays(ctx, *parts, mouse, f"seed_{seed}")
            if arrays is None:
                return None
            rows += [{"roi": roi, "mouse": mouse, "group": group, "seed": seed, "true_infected": int(truth[index]),
                      "p_infected": float(p)}
                     for roi, group, index, p in zip(held.roi, held.group, held.index, arrays["p_infected"])]
    return pd.DataFrame(rows)


def _tuned_predictions(ctx: Context, partition: Partition, table, truth: np.ndarray, source: str):
    """Five-seed mean held-out ROI probabilities of the tuned classifier of every fold, and the settings
    used (None unless every fold is finished). ``source`` is ``"paper"`` or ``"search"`` (this directory's tuning)."""
    import pandas as pd

    frames, settings = [], []
    for mouse in sorted(table.mouse.unique()):
        record = tuned_record(ctx, partition, mouse, source)
        if record is None:
            return None
        config = record.pop("config")
        fits = [_arrays(ctx, *tuned_parts(partition, config), mouse, f"seed_{seed}") for seed in CLASSIFIER_SEEDS]
        if any(fit is None for fit in fits):
            return None
        held = table[table.mouse == mouse]
        frames.append(pd.DataFrame({"roi": held.roi.to_numpy(), "mouse": mouse, "group": held.group.to_numpy(),
                                    "true_infected": truth[held.index],
                                    "p_infected": np.mean([fit["p_infected"] for fit in fits], axis=0)}))
        settings.append({"held_out_mouse": mouse, **record, **config})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(settings)


def _ensemble(seeds):
    """Mean over seeds per ROI, then mean over ROIs per mouse."""
    roi = seeds.groupby(["roi", "mouse", "group", "true_infected"], as_index=False).p_infected.mean()
    mouse = roi.groupby("mouse", as_index=False).agg(group=("group", "first"), true_infected=("true_infected", "first"),
                                                     p_infected=("p_infected", "mean"))
    return roi, mouse


# Encoder
def encoder_report(ctx: Context, config: dict, source: str, out: Path, compare: Comparison | None) -> list[str]:
    """Seed stability of the selected encoder. Returns ``["encoder"]`` unless all five seeds are finished."""
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    done = [seed for seed in FREEZE_SEEDS if ctx.encoder_unit(config, seed, source).done()]
    k, reference = config["n_clusters"], paper_reference()["encoder"]
    rows = {}
    for name, seeds in (("stability3", SEARCH_SEEDS), ("stability5", FREEZE_SEEDS)):
        if set(seeds) <= set(done):
            profile = ctx.profile(config, seeds, source)
            for value in K_VALUES:
                rows.setdefault(value, {"K": value})[name] = profile[value]["stability"]
                rows[value][name.replace("stability", "spread_deficit")] = profile[value]["spread_deficit"]
            print(f"Encoder {name[-1]}-seed stability at K = {k}: {profile[k]['stability']:.4f}")
            if compare:
                compare.add(f"encoder seed stability, {name[-1]} seeds", profile[k]["stability"], reference[name], "stability")
    if rows:
        pd.DataFrame(list(rows.values())).to_csv(out / "encoder_stability.csv", index=False)
    else:
        print(f"Encoder: {len(done)} of {len(FREEZE_SEEDS)} seeds finished; seed stability not reported.")
    if source == "paper" and 0 in done:
        _, deposited = paper_partition()
        refit = ctx.encoder_fit(config, 0, source)["labels"][K_VALUES.index(k)]
        ari = float(adjusted_rand_score(deposited, refit))
        print(f"Same-seed refit of the partition: adjusted Rand index {ari:.4f}")
        if compare:
            compare.add("same-seed refit of the partition, ARI", ari, reference["refit_ari_record"], "stability")
    return [] if len(done) == len(FREEZE_SEEDS) else ["encoder"]


# Descriptive
def descriptive_report(ctx: Context, partition: Partition, out: Path, compare: Comparison | None) -> int:
    """Composition, infection-associated cluster, co-occurrence and subgraphs. Returns that cluster."""
    nodes, k, reference = partition.nodes, partition.k, paper_reference()
    numbered = compare is not None and partition.name == "paper"      # cluster numbers equal the paper's
    fractions = descriptive.mouse_fractions(nodes, k)
    by_group = fractions.groupby("group")[list(range(k))].mean().reindex(GROUPS)
    chosen = descriptive.infection_cluster(nodes, k)
    cluster = chosen["cluster"]
    profiles = descriptive.marker_profiles(nodes, partition.features, k)
    pairs, per_roi = descriptive.cooccurrence(nodes, k)
    regions = descriptive.subgraphs(nodes, cluster)
    fractions.rename(columns={c: f"ECM{c}" for c in range(k)}).to_csv(out / "composition_per_mouse.csv")
    by_group.rename(columns={c: f"ECM{c}" for c in range(k)}).to_csv(out / "composition_by_group.csv")
    profiles.to_csv(out / "marker_profiles.csv", index=False)
    pairs.to_csv(out / "cooccurrence_pairs.csv", index=False)
    per_roi.to_csv(out / "cooccurrence_per_roi.csv", index=False)
    regions.to_csv(out / "subgraphs.csv", index=False)
    per_mouse = regions.groupby("mouse").agg(n=("n_nodes", "size"), median_n_nodes=("n_nodes", "median"),
                                             median_mean_degree=("mean_degree", "median"))
    print(f"\nPartition '{partition.name}': {partition.n_signal:,} ECM signal patches, {k} clusters, "
          f"sizes {profiles.n_patches.tolist()}")
    if not any(chosen["present_in_every_infected_mouse"].values()):
        print("No cluster makes up at least 1% of the ECM patches of every infected mouse, so the ratio rule does not apply; "
              "the cluster with the largest infected - naive difference is taken.")
    print(f"Infection-associated cluster: ECM {cluster}; mean mouse fraction by group:")
    print((100 * by_group[cluster]).round(3).rename("percent").to_string())
    top = pairs.reindex(pairs.ko_minus_wt.abs().sort_values(ascending=False).index).head(4)
    print("Largest KO - WT co-occurrence differences (log2 units):")
    print(top[["cluster_a", "cluster_b", "ko_minus_wt"]].round(3).to_string(index=False))
    print(f"ECM {cluster} subgraphs per mouse:")
    print(per_mouse.round(2).to_string())
    if compare is None:
        return cluster
    compare.add("ECM signal patches", partition.n_signal, reference["n_signal"], "count")
    compare.add("number of clusters", k, len(reference["cluster_sizes"]), "count")
    for group in GROUPS:
        compare.add(f"infection-associated cluster, mean mouse fraction, {group}", by_group.loc[group, cluster],
                    reference["group_fractions"][group][reference["infection_cluster"]], "descriptive")
    for mouse, record in reference["subgraphs"]["per_mouse"].items():
        mine = per_mouse.loc[mouse] if mouse in per_mouse.index else {"n": 0, "median_n_nodes": np.nan, "median_mean_degree": np.nan}
        compare.add(f"subgraphs, {mouse}, number", mine["n"], record["n"], "count")
        compare.add(f"subgraphs, {mouse}, median patches", mine["median_n_nodes"], record["median_n_nodes"], "descriptive")
        compare.add(f"subgraphs, {mouse}, median mean degree", mine["median_mean_degree"], record["median_mean_degree"], "descriptive")
    if numbered:
        compare.add("infection-associated cluster", cluster, reference["infection_cluster"], "count")
        for c, size in enumerate(reference["cluster_sizes"]):
            compare.add(f"ECM {c}, patches", profiles.n_patches[c], size, "count")
        for mouse, values in reference["mouse_fractions"].items():
            for c, value in enumerate(values):
                compare.add(f"fraction of ECM {c}, {mouse}", fractions.loc[mouse, c], value, "descriptive")
        mine = pairs.set_index(["cluster_a", "cluster_b"])
        for record in reference["cooccurrence"]:
            row = mine.loc[(record["cluster_a"], record["cluster_b"])]
            compare.add(f"co-occurrence KO - WT, ECM {record['cluster_a']}-{record['cluster_b']}", row.ko_minus_wt,
                        record["ko_minus_wt"], "descriptive")
    return cluster


# Classifier
def classifier_report(ctx: Context, partition: Partition, selection: str, cluster: int, out: Path,
                      compare: Comparison | None) -> list[str]:
    """Scores of the classifiers and the permutation test. Returns the kinds of task whose fits are not finished."""
    import pandas as pd

    table = ctx.roi_table()
    truth = table.infected.to_numpy()
    reference = paper_reference()
    rows, missing = [], []

    def summarise(name, roi, mouse):
        r, m = scores(roi.p_infected, roi.true_infected), scores(mouse.p_infected, mouse.true_infected)
        rows.append({"model": name, "mice_correct": m["correct"], "n_mice": m["n"], "mouse_auc": m["auc"],
                     "rois_correct": r["correct"], "n_rois": r["n"]})
        return r, m

    # Fixed classifier, ROI-mean loss (Fig. 3g)
    seeds = _fold_predictions(ctx, roi_mean_parts(partition, OBSERVED_ALLOCATION), table, truth)
    if seeds is None:
        missing.append("classifier")
    else:
        roi, mouse = _ensemble(seeds)
        seeds.to_csv(out / "classifier_per_seed.csv", index=False)
        roi.to_csv(out / "classifier_roi.csv", index=False)
        mouse.to_csv(out / "classifier_mouse.csv", index=False)
        r, m = summarise("GraphSAGE, ROI-mean loss", roi, mouse)
        if compare:
            paper = reference["classifier"]
            compare.add("classifier, mice correct", m["correct"], paper["mice_correct"], "count")
            compare.add("classifier, ROIs correct", r["correct"], paper["rois_correct_record"], "count")
            compare.add("classifier, mouse ROC-AUC", m["auc"], paper["mouse_auc"], "descriptive")
            for row in mouse.itertuples():
                compare.add(f"classifier, P(infected), {row.mouse}", row.p_infected, paper["mouse_probability"][row.mouse], "probability")
            per_seed = seeds.groupby(["seed", "mouse"]).p_infected.mean()
            for (seed, name), value in per_seed.items():
                compare.add(f"classifier, P(infected), {name}, seed {seed}", value,
                            paper["seed_mouse_probability"][str(seed)][name], "probability")

    # Composition-only logistic regression
    lr = _arrays(ctx, "composition_lr", partition.folder)
    if lr is None:
        missing.append("composition_lr")
    else:
        roi = table[["roi", "mouse", "group"]].assign(true_infected=truth, p_infected=lr["p_infected"])
        mouse = _ensemble(roi)[1]
        roi.to_csv(out / "composition_lr_roi.csv", index=False)
        mouse.assign(chosen_c=lr["chosen_c"]).to_csv(out / "composition_lr_mouse.csv", index=False)
        r, m = summarise("Logistic regression on cluster fractions", roi, mouse)
        if compare:
            paper = reference["composition_lr"]
            compare.add("logistic regression, mice correct", m["correct"], paper["mice_correct"], "count")
            compare.add("logistic regression, mouse ROC-AUC", m["auc"], paper["mouse_auc_record"], "descriptive")
            for row in mouse.itertuples():
                compare.add(f"logistic regression, P(infected), {row.mouse}", row.p_infected,
                            paper["mouse_probability"][row.mouse], "regression")

    # Fixed classifier, loss pooled over patches: untuned model of Supplementary Table 5 and integrated gradients
    parts = ("classifier", partition.folder, "patch_sum")
    untuned = _fold_predictions(ctx, parts, table, truth)
    if untuned is None:
        missing.append("explanation")
    else:
        roi, mouse = _ensemble(untuned)
        mouse.to_csv(out / "untuned_mouse.csv", index=False)
        _, m = summarise("GraphSAGE, patch-pooled loss (untuned)", roi, mouse)
        attribution = []
        for index, row in table.iterrows():
            signed = np.mean([_arrays(ctx, *parts, row.mouse, f"seed_{seed}")[f"signed_{row.roi}"] for seed in CLASSIFIER_SEEDS], axis=0)
            sub = partition.nodes[(partition.nodes.roi == row.roi) & (partition.nodes.cluster >= 0)]
            attribution.append(pd.DataFrame({"roi": row.roi, "mouse": row.mouse, "group": row.group, "x": sub.x.to_numpy(),
                                             "y": sub.y.to_numpy(), "cluster": sub.cluster.to_numpy(),
                                             "importance": np.abs(signed).sum(axis=1)}))
        attribution = pd.concat(attribution, ignore_index=True)
        attribution.to_csv(out / "integrated_gradients.csv.gz", index=False)
        local = descriptive.ig_localisation(attribution, partition.k, cluster)
        local.to_csv(out / "ig_localisation.csv", index=False)
        infected = local[local.is_infection_cluster & local.group.str.startswith("Infected")].dropna(subset=["pearson_r"])
        medians = infected.groupby("group").pearson_r.agg(["median", "size"])
        print(f"\nCorrelation of integrated gradients with ECM {cluster} membership (median over held-out infected ROIs):")
        print(medians.round(4).to_string())
        if compare:
            paper = reference["tuning"]
            compare.add("untuned classifier, mice correct", m["correct"], paper["untuned_correct"], "count")
            for row in mouse.itertuples():
                compare.add(f"untuned classifier, P(infected), {row.mouse}", row.p_infected,
                            paper["per_mouse"][row.mouse]["p_untuned"], "probability")
            for group, value in reference["ig_localisation"]["median"].items():
                if group in medians.index:
                    compare.add(f"integrated gradients, median correlation, {group}", medians.loc[group, "median"], value, "correlation")

    # Tuned classifiers (Supplementary Table 5): the selection's settings and, with the paper's selection,
    # also the result of a 'tune' stage run in this directory on the same partition.
    sources = [("paper", "Tuned within each fold (the paper's settings refitted)", "tuned")] if selection == "paper" else []
    sources.append(("search", "Tuned within each fold (this run's tuning)", "tuning" if selection == "paper" else "tuned"))
    for source, name, stem in sources:
        tuned = _tuned_predictions(ctx, partition, table, truth, source)
        if tuned is None:
            if source == selection:
                missing.append("tuned")
            continue
        roi, mouse = _ensemble(tuned[0])
        mouse.to_csv(out / f"{stem}_mouse.csv", index=False)
        tuned[1].to_csv(out / f"{stem}_settings.csv", index=False)
        _, m = summarise(name, roi, mouse)
        if compare:
            paper = reference["tuning"]
            compare.add(f"{name.lower()}, mice correct", m["correct"], paper["tuned_correct"], "count")
            if source == "paper":
                for row in mouse.itertuples():
                    compare.add(f"tuned classifier, P(infected), {row.mouse}", row.p_infected,
                                paper["per_mouse"][row.mouse]["p_tuned"], "probability")

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(out / "classifier_summary.csv", index=False)
        print("\nHeld-out mouse classification (probability cut-off 0.5)")
        print(summary.round(4).to_string(index=False))
    if not permutation_report(ctx, partition, out, compare):
        missing.append("permutation")
    return missing


def permutation_report(ctx: Context, partition: Partition, out: Path, compare: Comparison | None) -> bool:
    """Mouse-level ROC-AUC of the five-seed ensemble under every allocation, and the exact P values.

    Returns whether every ``permutation`` task is finished (two allocations in a smoke run).
    """
    import pandas as pd

    table, labels = ctx.roi_table(), mouse_labels(ctx)
    genotype = {mouse: sub.genotype.iloc[0] for mouse, sub in table.groupby("mouse")}
    rows = []
    for index, allocation in enumerate(allocations(list(labels))):
        truth = table.mouse.isin(allocation).to_numpy().astype(int)
        seeds = _fold_predictions(ctx, roi_mean_parts(partition, index), table, truth)
        if seeds is None:
            continue
        mouse = _ensemble(seeds)[1]
        result = scores(mouse.p_infected, mouse.true_infected)
        rows.append({"allocation": index, "infected_mice": "|".join(allocation), "observed": index == OBSERVED_ALLOCATION,
                     "genotype_preserving": genotype_preserving(allocation, genotype), "auc": result["auc"],
                     "mice_correct": result["correct"]})
    expected = {task["allocation"] for task in final_tasks(ctx.smoke) if task["kind"] == "permutation"}
    complete = expected <= {row["allocation"] for row in rows}
    if not rows:
        return complete
    result = pd.DataFrame(rows)
    result.to_csv(out / "permutation_allocations.csv", index=False)
    total = len(allocations(list(labels)))
    if len(result) < total:
        print(f"\nPermutation test: {len(result)} of {total} labellings finished; P values need all of them.")
        return complete
    p = permutation_p(result.auc.to_numpy(), OBSERVED_ALLOCATION, result.genotype_preserving.to_numpy())
    print(f"\nPermutation test: observed ROC-AUC {p['observed_auc']:.2f}; P = {p['n_at_or_above_observed']}/{p['n_allocations']} "
          f"over all labellings and {p['n_genotype_at_or_above_observed']}/{p['n_genotype_allocations']} keeping two infected "
          "mice per genotype")
    if compare:
        paper = paper_reference()["permutation_record"]
        compare.add("permutation test, labellings at or above the observed ROC-AUC", p["n_at_or_above_observed"],
                    paper["n_at_or_above_observed"], "count")
        compare.add("permutation test, genotype-preserving labellings at or above", p["n_genotype_at_or_above_observed"],
                    paper["n_genotype_at_or_above_observed"], "count")
    return complete


# Report
def write_report(ctx: Context, selection: str = "paper", partition: str | None = None) -> list[str]:
    """Write all tables and the comparison with the paper.

    Returns the kinds of task (as listed by ``mantpy-liver tasks``) whose fits are not finished.
    """
    import pandas as pd

    out = ctx.store.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    config, source = selected_config(ctx, selection), ctx.signal_source(selection)
    same_settings = config_id(config) == paper_reference()["config_id"] and not ctx.smoke
    compare = Comparison() if same_settings else None
    if ctx.smoke:
        log("smoke run: results are not compared with the paper")
    elif not same_settings:
        log(f"selected configuration {config_id(config)} differs from the paper's; "
            "results are reported without a paper comparison")
    print(f"Selected: {config['patch_size']}-pixel patches, K = {config['n_clusters']} ({config_id(config)}; selection: {selection})")
    summary = ctx.prepared.summary()
    pd.DataFrame({"patch_size": list(summary["n_patches"]), "n_patches": list(summary["n_patches"].values()),
                  "n_signal_recomputed": list(summary["n_signal"].values()),
                  "coverage": list(summary["coverage"].values())}).to_csv(out / "patches.csv", index=False)
    if compare:
        reference = paper_reference()
        for marker, bound in zip(MARKERS, ctx.prepared.bounds):
            compare.add(f"99.5th-percentile bound, {marker}", bound, reference["bounds"][marker], "descriptive")
        compare.add("ECM signal patches, background split recomputed in this run",
                    summary["n_signal"][config["patch_size"]], reference["n_signal"], "patches")
        compare.add("5-pixel signal patches (anchors), recomputed", summary["n_signal"][ANCHOR_SIZE],
                    reference["n_anchors_record"], "anchors")
        for size in PATCH_SIZES:
            compare.add(f"coverage of {size}-pixel patches", summary["coverage"][size], reference["coverage_record"][str(size)],
                        "coverage")
    missing = encoder_report(ctx, config, source, out, compare)
    name = partition or ("paper" if source == "paper" else "refit")
    if name == "refit" and not ctx.encoder_unit(config, 0, source).done():
        print("The seed-0 encoder is not finished, so there is no partition to analyse yet.")
        missing = list(task_ranges(ctx.smoke))
    else:
        frozen = resolve_partition(ctx, selection, partition)
        cluster = descriptive_report(ctx, frozen, out, compare)
        missing += classifier_report(ctx, frozen, selection, cluster, out, compare)
    # A comparison left by a report with other settings would not describe the tables written above.
    (out / "comparison_with_paper.csv").unlink(missing_ok=True)
    if compare and compare.rows:
        table = compare.table()
        table.to_csv(out / "comparison_with_paper.csv", index=False)
        outside = table[~table.within_tolerance]
        print(f"\n{len(table) - len(outside)} of {len(table)} reported numbers agree with the paper within the stated tolerances.")
        if len(outside):
            print("Outside the tolerance:")
            print(outside.round(5).to_string(index=False))
            print("\nThe encoder and the tuned classifiers are trained on a GPU. On a GPU model other than the paper's, "
                  "their rows can fall outside the tolerance (docs_liver.md, 'Notes on exact reproduction').")
    if missing:
        print(f"\nNot finished, so not reported: {', '.join(kind for kind in task_ranges(ctx.smoke) if kind in missing)}.")
    log(f"report written to {out}")
    return missing

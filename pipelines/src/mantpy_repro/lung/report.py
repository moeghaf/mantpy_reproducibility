"""Collect the finished results into tables and compare every reported number with the paper.

Scores are averaged over the corruption realisations within each ROI and then over ROIs with
equal weight. Per-cluster correction pools the replaced and restored labels of an ROI over
its realisations before averaging ROIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CONTEXTS, FOCAL_CELL_TYPE, PRIMARY_KIND, PRIOR, SCHEMES, Config
from .core import Context, log, unit_arrays, unpack
from .data import decode_partition
from .selection import load_selection

# Absolute difference treated as agreement with the paper. Counts must be equal, and a number the
# paper gives only rounded is allowed half its last printed digit more.
TOLERANCE = 0.01

METRICS = ["roc_auc", "prior_adjusted_roc_auc", "pr_auc", "prior_adjusted_pr_auc", "accuracy_before", "accuracy_after",
           "recovery", "pristine_relabel_rate", "correction_precision", "macro_f1_after", "balanced_accuracy_after"]
REFIT_METRICS = ["roc_auc", "recovery", "accuracy_after", "accuracy_after_common_signal", "signal_k", "target_coverage",
                 "realized_corruption_fraction", "prior_adjusted_roc_auc_full_signal", "context_roc_auc_common_signal",
                 "prior_adjusted_roc_auc_common_signal", "macro_f1_after_common_signal"]


def paper_reference() -> dict:
    return json.loads(Path(__file__).with_name("paper_reference.json").read_text(encoding="utf-8"))


# Selected settings
def selection_table(chosen: dict):
    """The selected cell-context settings in the layout of Supplementary Table 6."""
    import pandas as pd

    weights = {"none": "none", "inv_d": "inverse distance", "inv_d2": "inverse squared distance",
               "gaussian_median": "Gaussian"}
    compositions = {"fraction": "fraction", "count": "count", "log1p_count": "log(1+count)"}
    statistics = {"raw_neglogp": "-log P", "minmax_neglogp": "min-max -log P", "one_minus_p": "1-P", "margin": "margin"}
    rows = []
    for scheme, title in (("mouse", "mouse held out (main result)"), ("roi", "ROI held out")):
        for fold, contexts in chosen.get(scheme, {}).items():
            saved = contexts["cell"]
            config = Config.from_dict(saved["config"])
            g, c = config.graph, config.cleaning
            cell_cell = {"knn": f"{g.cc_k}-NN" + (f", <={g.cc_radius_px:.0f} px" if g.cc_radius_px else ""),
                         "radius": f"radius {g.cc_radius_px or 0:.0f} px", "delaunay": "Delaunay"}[g.cc_rule]
            cell_ecm = {"knn": f"{g.ce_k}-NN" + (f", <={g.ce_dmax_px:g} px" if g.ce_dmax_px else ""),
                        "radius": f"radius {g.ce_dmax_px or 15.0:g} px",
                        "delaunay": "Delaunay" + (f", <={2 * g.ce_dmax_px:g} px" if g.ce_dmax_px else ""),
                        "containing_ring": "containing patch + 8 neighbours"}[g.ce_rule]
            gate = (f"gate {c.gate_radius_px:g} px, >= {c.gate_min_neighbours} neighbours"
                    if c.gate_radius_px > 0 and c.gate_min_neighbours > 0 else "gate none")
            rule = f"{statistics[c.anomaly_stat]} > {c.threshold:.2f}; confidence >= {c.min_pred_conf:.1f}; {gate}"
            penalty = config.learner.penalty
            rows.append({
                "scheme": title,
                "held_out": f"Mouse {int(fold[-2:])}" if scheme == "mouse" else f"ROI {fold[-3:]}",
                "cell_cell_edges": cell_cell, "cell_ecm_edges": cell_ecm, "edge_weight": weights[g.edge_weight],
                "composition": compositions[config.features.composition_norm],
                "penalty_C": penalty if penalty == "none" else f"{penalty} ({round(config.learner.C, 3):g})",
                "cleaning_rule": rule + ("; smoothing" if c.majority_smoothing else ""),
                "calibration": config.calibration, "inner_accuracy": f"{saved['inner_selection_score']:.3f}"})
    return pd.DataFrame(rows)


# Scores
def denoising_scores(ctx: Context, selection: str):
    """Rows of every finished final fit of the mouse and single-ROI schemes."""
    import pandas as pd

    scores, classes = [], []
    root = ctx.store.root / "final" / ctx.partition / selection
    for scheme in SCHEMES:
        for folder in sorted((root / scheme).glob("*/*")):
            arrays = unit_arrays(folder)
            if arrays is None:
                continue
            for prefix, collected in (("scores__", scores), ("classes__", classes)):
                table = unpack(arrays, prefix)
                table.insert(0, "scheme", scheme)
                table.insert(1, "held_out", folder.parent.name)
                collected.append(table)
    if not scores:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(scores, ignore_index=True), pd.concat(classes, ignore_index=True)


def summarise(scores):
    """Mean over realisations within each ROI, then mean, s.d., minimum and maximum over ROIs."""
    keys = ["scheme", "context", "kind"]
    per_roi = scores.groupby([*keys, "roi", "mouse"], sort=False)[METRICS].mean().reset_index()
    summary = per_roi.groupby(keys, sort=False)[METRICS].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return per_roi, summary.reset_index()


def class_recovery(classes):
    """Fraction of replaced labels restored per true cluster: counts pooled within ROI, ROIs averaged."""
    keys = ["scheme", "context", "kind"]
    per_roi = classes.groupby([*keys, "roi", "cluster"], sort=False)[["n_corrupted", "n_restored"]].sum().reset_index()
    per_roi["recovery"] = per_roi["n_restored"] / per_roi["n_corrupted"].replace(0, np.nan)
    summary = per_roi.groupby([*keys, "cluster"], sort=False)["recovery"].agg(["mean", "std", "min", "max"]).reset_index()
    return per_roi, summary


def refit_scores(ctx: Context, selection: str):
    """Rows of every finished final fit of the refit control, and their summaries."""
    import pandas as pd

    root = ctx.store.root / "refit" / "final" / selection
    tables = [unpack(arrays, "scores__") for arrays in map(unit_arrays, sorted(root.glob("*/*/*"))) if arrays is not None]
    if not tables:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    scores = pd.concat(tables, ignore_index=True)
    metrics = [metric for metric in REFIT_METRICS if metric in scores]
    per_roi = scores.groupby(["normalization", "context", "outer_roi", "kind"], sort=False)[metrics].mean().reset_index()
    summary = per_roi.groupby(["normalization", "context", "kind"], sort=False)[metrics].mean().reset_index()
    return scores, per_roi, summary


# Comparison with the paper
class Comparison:
    """Rows of ``comparison_with_paper.csv``: this run, the paper, their difference."""

    def __init__(self):
        self.rows = []

    def add(self, section: str, quantity: str, value, paper, *, exact: bool = False, rounding: float = 0.0) -> None:
        """``exact`` is for counts. ``rounding`` widens the tolerance by half the last printed
        digit for numbers that the paper reports only rounded. A quantity the paper leaves
        blank (no observed edges) agrees when it is undefined here too."""
        value = float(value)
        if paper is None:
            self.rows.append({"section": section, "quantity": quantity, "this_run": value, "paper": np.nan,
                              "difference": np.nan, "tolerance": np.nan, "within_tolerance": not np.isfinite(value)})
            return
        tolerance = 0.0 if exact else TOLERANCE + rounding
        difference = value - float(paper)
        self.rows.append({"section": section, "quantity": quantity, "this_run": value, "paper": float(paper),
                          "difference": difference, "tolerance": tolerance,
                          "within_tolerance": bool(np.isfinite(difference) and abs(difference) <= tolerance + 1e-12)})

    def table(self):
        import pandas as pd

        return pd.DataFrame(self.rows, columns=["section", "quantity", "this_run", "paper", "difference", "tolerance",
                                                "within_tolerance"])


def compare_partition(rows: Comparison, ctx: Context, reference: dict) -> None:
    prepared, cohort = ctx.prepared(), ctx.cohort()
    labels = np.concatenate([roi.label for roi in cohort.rois.values()])
    paper = reference["partition"]
    section = f"partition ({ctx.partition})"
    rows.add(section, "patches", len(labels), paper["n_patches"], exact=True)
    rows.add(section, "signal patches", int((labels >= 0).sum()), paper["n_signal"], exact=True)
    for cluster, size in enumerate(paper["cluster_sizes"]):
        rows.add(section, f"patches in ECM {cluster}", int((labels == cluster).sum()), size, exact=True)
    for name, roi in cohort.rois.items():
        rows.add(section, f"signal patches, {name}", int((roi.label >= 0).sum()), paper["signal_per_roi"][name], exact=True)
    features = np.concatenate([roi.patch_feat for roi in cohort.rois.values()])
    for cluster, profile in paper["cluster_profiles"].items():
        for marker, value in profile.items():
            mine = features[labels == int(cluster), cohort.ecm_names.index(marker)].mean()
            rows.add(section, f"mean {marker} of ECM {cluster}", mine, value)
    summary = prepared.summary()
    rows.add("fresh partition", "signal patches", summary["fresh"]["signal"], paper["n_signal"], exact=True)
    rows.add("fresh partition", "patches labelled differently from the paper", summary["patches_labelled_differently"],
             0, exact=True)
    scan = prepared.component_scan().set_index("total_components")
    for entry in reference["component_scan"]:
        k = entry["total_components"]
        rows.add("component scan", f"silhouette, {k} components", scan.loc[k, "silhouette"], entry["silhouette"])
        rows.add("component scan", f"cross-seed ARI, {k} components", scan.loc[k, "cross_seed_ari"], entry["cross_seed_ari"])
        rows.add("component scan", f"background patches, {k} components", scan.loc[k, "background_patches"],
                 entry["background_patches"], exact=True)
    rows.add("component scan", "selected number of components", summary["selected_total_components"],
             reference["selected_total_components"], exact=True)


def compare_description(rows: Comparison, tables: dict, reference: dict) -> None:
    if "graph_counts" in tables:
        counts = tables["graph_counts"].set_index("roi")
        for name, paper in reference["graph_counts"].items():
            for column, value in paper.items():
                rows.add("graphs", f"{column}, {name}", counts.loc[name, column], value, exact=True)
        rows.add("graphs", "cell-ECM edges to signal patches, all ROIs", counts["cell_ecm_edges_signal"].sum(),
                 reference["n_signal_cell_ecm_edges"], exact=True)
    if "association" in tables:
        pooled = tables["association"].set_index(["cell_type", "cluster"])
        for entry in reference["association"]:
            mine = pooled.loc[(entry["cell_type"], entry["cluster"])]
            label = f"{entry['cell_type']} with ECM {entry['cluster']}"
            rows.add("association", f"observed edges, {label}", mine["n_obs"], entry["observed"], exact=True)
            rows.add("association", f"expected edges, {label}", mine["n_exp"], entry["expected"])
            rows.add("association", f"log2(obs/exp), {label}", mine["log2_enr"], entry["log2_obs_exp"])
        per_roi = tables["association_per_roi"]
        focal = per_roi[per_roi["cell_type"] == FOCAL_CELL_TYPE].set_index(["roi", "cluster"])
        for entry in reference["aec_association_per_roi"]:
            mine = focal.loc[(entry["roi"], entry["cluster"])]
            label = f"{FOCAL_CELL_TYPE} with ECM {entry['cluster']}, {entry['roi']}"
            rows.add("association per ROI", f"observed edges, {label}", mine["n_obs"], entry["observed"], exact=True)
            rows.add("association per ROI", f"log2(obs/exp), {label}", mine["log2_enr"], entry["log2_obs_exp"])
        contrast = tables["aec_marker_contrast"].set_index("roi")
        for marker, value in reference["aec_marker_contrast"]["mean_over_rois"].items():
            rows.add("marker contrast", f"{marker}, mean over ROIs", contrast[marker].mean(), value)
        for name, markers in reference["aec_marker_contrast"]["per_roi"].items():
            for marker, value in markers.items():
                rows.add("marker contrast per ROI", f"{marker}, {name}", contrast.loc[name, marker], value)
    if "edge_rule_sensitivity" in tables:
        rules = tables["edge_rule_sensitivity"].set_index(["recipe_index", "cluster"])
        for entry in reference["edge_rules"]:
            index = entry["recipe_index"]
            if (index, 0) not in rules.index:
                continue
            rows.add("edge rules", f"cell-ECM edges to signal patches, rule {index}", rules.loc[(index, 0), "n_cell_ecm_edges"],
                     entry["n_cell_ecm_edges"], exact=True)
            for cluster, value in enumerate(entry["log2_obs_exp"]):
                rows.add("edge rules", f"log2(obs/exp), {FOCAL_CELL_TYPE} with ECM {cluster}, rule {index}",
                         rules.loc[(index, cluster), "log2_enr"], value)


def compare_denoising(rows: Comparison, scores, per_roi, recovery_per_roi, refit_per_roi, reference: dict) -> None:
    mixed = per_roi[per_roi["kind"] == PRIMARY_KIND] if len(per_roi) else per_roi
    mouse = mixed[mixed["scheme"] == "mouse"] if len(mixed) else mixed
    if len(mouse):
        indexed = mouse.set_index(["context", "roi"])
        for metric in ("roc_auc", "prior_adjusted_roc_auc"):
            for context in (*CONTEXTS, PRIOR):
                paper = [entry[metric] for entry in reference["detection_per_roi"] if entry["context"] == context]
                mine = mouse.loc[mouse["context"] == context, metric]
                if len(mine) == len(paper) and paper[0] is not None:
                    for statistic in ("mean", "min", "max"):
                        rows.add("detection, mouse held out", f"{metric}, {context} context, {statistic} over ROIs",
                                 getattr(mine, statistic)(), getattr(np, statistic)(paper))
            for entry in reference["detection_per_roi"]:
                if (entry["context"], entry["roi"]) in indexed.index and entry[metric] is not None:
                    rows.add("detection per ROI", f"{metric}, {entry['context']} context, {entry['roi']}",
                             indexed.loc[(entry["context"], entry["roi"]), metric], entry[metric])
        single = scores[(scores["scheme"] == "mouse") & (scores["kind"] == PRIMARY_KIND)].set_index(["context", "roi", "seed"])
        for entry in reference["detection_per_realisation"]:
            key = (entry["context"], entry["roi"], entry["seed"])
            if key in single.index:
                for metric in ("roc_auc", "prior_adjusted_roc_auc"):
                    rows.add("detection per realisation", f"{metric}, {entry['context']} context, {entry['roi']}, "
                             f"realisation {entry['seed']}", single.loc[key, metric], entry[metric])
    for entry in reference["text"]:
        experiment, metric = entry["experiment"], entry["metric"]
        if experiment.startswith("refit:"):
            table = refit_per_roi
            if not len(table):
                continue
            table = table[(table["normalization"] == experiment.split(":")[1]) & (table["kind"] == PRIMARY_KIND)]
            label = f"refitted foreground and clusters ({experiment.split(':')[1]})"
        elif metric.startswith("class_recovery_"):
            table = recovery_per_roi
            if not len(table):
                continue
            table = table[(table["scheme"] == experiment) & (table["kind"] == PRIMARY_KIND)
                          & (table["cluster"] == int(metric[-1]))].rename(columns={"recovery": metric})
            label = f"{experiment} held out"
        else:
            table = mixed[mixed["scheme"] == experiment] if len(mixed) else mixed
            label = f"{experiment} held out"
        values = table.loc[table["context"] == entry["context"], metric] if len(table) else []
        if len(values) == len(reference["partition"]["signal_per_roi"]):      # a statistic over ROIs needs every ROI
            rows.add("numbers in the text", f"{metric}, {entry['context']} context, {label}, {entry['statistic']} over ROIs",
                     getattr(values, entry["statistic"])(), entry["paper"], rounding=0.5 * 10 ** -entry["decimals"])


def compare_example(rows: Comparison, ctx: Context, selection: str, reference: dict) -> None:
    """The reconstruction shown in Fig. 4h-i: one ROI of the held-out mouse, first mixed realisation."""
    from .corruption import make_realisation

    example = reference["example"]
    cohort = ctx.cohort()
    roi = cohort.rois.get(example["roi"])
    paper_observed = decode_partition(example["observed"])
    if roi is None or len(roi.label) != len(paper_observed):
        return
    corruption = make_realisation(cohort, example["kind"], example["seed"])[example["roi"]]
    section = f"example ({example['roi']})"
    rows.add(section, "replaced labels", corruption.is_artifact.sum(), example["n_replaced"], exact=True)
    rows.add(section, "observed labels different from the paper's", (corruption.corrupted != paper_observed).sum(), 0, exact=True)
    if corruption.blocks:
        rows.add(section, "labels replaced by the first block", corruption.blocks[0]["n_flipped"],
                 example["block"]["n_replaced"], exact=True)
        rows.add(section, "side of the first block", corruption.blocks[0]["side"], example["block"]["side"], exact=True)
    mouse = cohort.rois[example["roi"]].mouse
    arrays = unit_arrays(ctx.store.root / "final" / ctx.partition / selection / "mouse" / mouse / example["context"])
    if arrays is not None and f"cleaned__{example['roi']}" in arrays:
        cleaned = arrays[f"cleaned__{example['roi']}"]
        restored = int(((cleaned == roi.label) & corruption.is_artifact).sum())
        rows.add(section, "replaced labels restored", restored, example["n_restored"], exact=True)
        rows.add(section, "cleaned labels different from the paper's", (cleaned != decode_partition(example["cleaned"])).sum(),
                 0, exact=True)


def compare_refit(rows: Comparison, summary, per_roi, reference: dict) -> None:
    """Means over the held-out ROIs; an arm is compared only when every ROI is finished."""
    if not len(summary):
        return
    keys = ["normalization", "context", "kind"]
    indexed = summary.set_index(keys)
    finished = per_roi.groupby(keys).size()
    for entry in reference["refit_control"]:
        key = (entry["normalization"], entry["context"], entry["kind"])
        if finished.get(key, 0) != len(reference["partition"]["signal_per_roi"]):
            continue
        for metric, value in entry.items():
            if metric in indexed.columns:
                rows.add("refit control", f"{metric}, {entry['context']} context, {entry['kind']} corruption, "
                         f"{entry['normalization']}", indexed.loc[key, metric], value)


def missing_results(ctx: Context, selection: str) -> list[str]:
    """Names of the describe units and final fits of this run that are not finished."""
    from .final import final_tasks

    root = ctx.store.root
    folders = {f"describe {label}": root / "describe" / ctx.partition / folder
               for folder, label in (("graphs", "graph counts"), ("association", "association"),
                                     ("edge_rules", "edge rules"))}
    for task in final_tasks(ctx.cohort().mice(), ctx.budget.roi_folds):
        if task["experiment"] == "refit":
            folder = root / "refit" / "final" / selection / task["normalization"]
        else:
            folder = root / "final" / ctx.partition / selection / task["experiment"]
        name = " ".join(str(task[key]) for key in ("experiment", "normalization", "fold", "context") if key in task)
        folders[f"final {name}"] = folder / task["fold"] / task["context"]
    return [name for name, folder in folders.items() if not (folder / "completion.json").exists()]


def _log_missing(missing: list[str]) -> None:
    if missing:
        log(f"{len(missing)} results are not finished and are not in the report. A statistic over ROIs is compared "
            "with the paper only when every ROI is finished; the summary tables average the finished ROIs.")
        for name in missing:
            log(f"  not finished: {name}")


def write_report(ctx: Context, selection: str = "paper") -> Path:
    """Write every table and the comparison with the paper; return the report folder.

    Results that are not finished are left out and named at the end of the log.
    """
    import pandas as pd

    out = ctx.store.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    reference = paper_reference()
    cohort = ctx.cohort()
    rows = Comparison()

    ctx.prepared().component_scan().to_csv(out / "component_scan.csv", index=False)
    frames = []
    for name, roi in cohort.rois.items():
        frame = pd.DataFrame({"roi": name, "mouse": roi.mouse, "patch_id": roi.patch_id, "x": roi.patch_xy[:, 0],
                              "y": roi.patch_xy[:, 1], "label": roi.label})
        frames.append(pd.concat([frame, pd.DataFrame(roi.patch_feat, columns=cohort.ecm_names)], axis=1))
    pd.concat(frames, ignore_index=True).to_csv(out / "partition.csv", index=False)

    tables = {}
    described = ctx.store.root / "describe" / ctx.partition
    for folder, items in (("graphs", {"graph_counts": "counts__"}),
                          ("association", {"association": "pooled__", "association_per_roi": "per_roi__",
                                           "aec_marker_contrast": "contrast__"}),
                          ("edge_rules", {"edge_rule_sensitivity": "rules__"})):
        arrays = unit_arrays(described / folder)
        if arrays is None:
            continue
        for name, prefix in items.items():
            tables[name] = unpack(arrays, prefix)
            tables[name].to_csv(out / f"{name}.csv", index=False)

    chosen = load_selection(ctx, selection)
    settings = selection_table(chosen)
    settings.to_csv(out / "selected_configurations.csv", index=False)
    scores, classes = denoising_scores(ctx, selection)
    per_roi = summary = recovery_per_roi = recovery = pd.DataFrame()
    if len(scores):
        per_roi, summary = summarise(scores)
        recovery_per_roi, recovery = class_recovery(classes)
        for name, table in (("denoising_scores", scores), ("denoising_per_roi", per_roi), ("denoising_summary", summary),
                            ("class_recovery", recovery)):
            table.to_csv(out / f"{name}.csv", index=False)
    refit_all, refit_per_roi, refit_summary = refit_scores(ctx, selection)
    if len(refit_all):
        for name, table in (("refit_control_scores", refit_all), ("refit_control_per_roi", refit_per_roi),
                            ("refit_control_summary", refit_summary)):
            table.to_csv(out / f"{name}.csv", index=False)

    if len(summary):
        shown = summary[summary["kind"] == PRIMARY_KIND]
        columns = ["scheme", "context", "roc_auc_mean", "prior_adjusted_roc_auc_mean", "recovery_mean",
                   "pristine_relabel_rate_mean", "accuracy_before_mean", "accuracy_after_mean"]
        print(f"\nDetection and cleaning, {PRIMARY_KIND} corruption (mean over held-out ROIs)")
        print(shown[columns].round(4).to_string(index=False))
    if len(refit_summary):
        shown = refit_summary[refit_summary["kind"] == PRIMARY_KIND]
        print("\nRefitted foreground and clusters, single ROIs held out")
        print(shown[["normalization", "context", "roc_auc", "recovery", "accuracy_after",
                     "accuracy_after_common_signal"]].round(4).to_string(index=False))

    missing = missing_results(ctx, selection)
    if ctx.smoke:
        log(f"smoke run: tables written to {out}; no comparison with the paper")
        _log_missing(missing)
        return out
    if not ctx.prepared().has_paper_grid():
        log(f"the ROIs are not the paper's; tables written to {out} without a comparison")
        _log_missing(missing)
        return out
    compare_partition(rows, ctx, reference)
    compare_description(rows, tables, reference)
    compare_denoising(rows, scores, per_roi, recovery_per_roi, refit_per_roi, reference)
    compare_example(rows, ctx, selection, reference)
    compare_refit(rows, refit_summary, refit_per_roi, reference)
    table6 = pd.DataFrame(reference["supplementary_table_6"])
    if len(settings) == len(table6):
        same = (settings.astype(str).reset_index(drop=True) == table6[settings.columns].astype(str)).all(axis=1).sum()
        rows.add("selected settings", "rows equal to Supplementary Table 6", same, len(table6), exact=True)
    compare = rows.table()
    compare.to_csv(out / "comparison_with_paper.csv", index=False)

    headline = compare[compare["section"].isin(["partition (%s)" % ctx.partition, "fresh partition",
                                                "detection, mouse held out", "numbers in the text",
                                                "marker contrast", "selected settings"])
                       & ~compare["quantity"].str.contains("mean .* of ECM|signal patches, roi", regex=True)]
    focal = compare[(compare["section"] == "association") & compare["quantity"].str.startswith(f"log2(obs/exp), {FOCAL_CELL_TYPE} ")]
    print("\nComparison with the paper (headline numbers)")
    print(pd.concat([focal, headline]).drop(columns="tolerance").round(4).to_string(index=False))
    counts = compare.groupby("section", sort=False)["within_tolerance"].agg(["sum", "count"])
    print("\nNumbers within tolerance, by section")
    print(counts.rename(columns={"sum": "within", "count": "compared"}).to_string())
    outside = compare[~compare["within_tolerance"]]
    print(f"\n{len(compare) - len(outside)} of {len(compare)} reported numbers are within tolerance of the paper "
          f"(absolute difference up to {TOLERANCE}, plus half the last printed digit where the paper gives a "
          "rounded number; counts equal).")
    if selection == "search":
        print("The settings come from this directory's search, which can select configurations other than the "
              "paper's; differences in the denoising scores are then expected.")
    if len(outside):
        print("Outside the tolerance:")
        print(outside.round(4).to_string(index=False))
    log(f"report written to {out}")
    _log_missing(missing)
    return out

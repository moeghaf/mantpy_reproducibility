"""Score the finished fits against the manual annotation and compare them with the paper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import config_id
from .core import Context, log
from .final import FINAL_SEEDS, K_VALUES, final_tasks, label_free_k
from .freeze import selected_config
from .scoring import ari_from_contingency, macro_f1_from_contingency

#: Largest absolute difference in ARI treated as agreement with the paper, by kind of quantity. The
#: paper's values were computed on an NVIDIA RTX 4070 SUPER and the tolerances are set for that GPU
#: model; on another model more rows fall outside them (README, "Notes on exact reproduction").
TOLERANCE = {
    "deterministic": 0.005,   # Patch-mean K-means and Multi-Otsu
    "pixel_kmeans": 0.01,     # changes with the scikit-learn version (image 1: 0.196 with 1.7.2, 0.203 with 1.9.1)
    "cnn_kmeans": 0.005,
    "graphmae": 0.02,         # GraphMAE and its two edge controls
    "stability": 0.05,        # seed stability at one K moves with small changes in three partitions
}
PAPER_GPU = "NVIDIA RTX 4070 SUPER"
SCORE_COLUMNS = ["image", "method", "seed", "ARI", "macro_F1"]
COMPARE_COLUMNS = ["image", "quantity", "seed", "kind", "this_run", "paper", "difference", "tolerance", "status"]
VARIANTS = {"full": "GraphMAE", "shuffled_edges": "GraphMAE, shuffled edges", "no_edges": "GraphMAE, no edges",
            "encoder_gin": "GraphMAE (GIN, no edge features)"}


def paper_reference() -> dict:
    return json.loads(Path(__file__).with_name("paper_reference.json").read_text(encoding="utf-8"))


def _arrays(folder: Path):
    if not (folder / "completion.json").exists():
        return None
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def unfinished_tasks(ctx: Context, image: str, selection: str = "paper") -> list[str]:
    """Final tasks of ``image`` that have no stored result for the selected configuration."""
    config, _ = selected_config(ctx, image, selection)
    identity, root = config_id(config), ctx.store.root
    names = []
    for task in final_tasks(image):
        kind, seed = task["kind"], task["seed"]
        if kind in ("baselines", "k_sweep"):
            folder, name = root / kind / image / identity, kind
        elif kind == "gin" and config["edge_use"] == "gin":
            continue      # the final stage has no GIN ablation for a configuration that already uses GIN
        else:
            variant = "encoder_gin" if kind == "gin" else kind
            folder, name = root / "runs" / image / identity / variant / f"seed_{seed}", f"{kind} seed {seed}"
        if not (folder / "completion.json").exists():
            names.append(f"{image} {name}")
    return names


def score_image(ctx: Context, image: str, selection: str = "paper"):
    """Per-seed scores of every finished final fit of ``image`` for the selected configuration."""
    import pandas as pd

    config, source = selected_config(ctx, image, selection)
    identity = config_id(config)
    prepared = ctx.prepared(image)
    scorer = prepared.scorer(config["patch_size"])
    root = ctx.store.root
    rows = []

    def add(method, seed, labels):
        score = scorer.score(labels)
        rows.append({"image": image, "method": method, "seed": seed, "ARI": score["ARI"], "macro_F1": score["macro_F1"]})

    for seed in FINAL_SEEDS:
        for variant, method in VARIANTS.items():
            arrays = _arrays(root / "runs" / image / identity / variant / f"seed_{seed}")
            if arrays is None:
                continue
            add(method, seed, arrays["labels_graphmae"])
            if variant == "full":
                add("CNN K-means", seed, arrays["labels_cnn"])
    base = _arrays(root / "baselines" / image / identity)
    if base is not None:
        for key, method in (("patch_mean_kmeans", "Patch-mean K-means"), ("multi_otsu", "Multi-Otsu"),
                            ("cnn_ward", "CNN Ward")):
            add(method, 0, base[key])
        rows.append({"image": image, "method": "Pixel K-means", "seed": 0,
                     "ARI": ari_from_contingency(base["pixel_contingency"]),
                     "macro_F1": macro_f1_from_contingency(base["pixel_contingency"])})
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    scores.attrs.update(selection=source, config_id=identity)
    return scores


def k_table(ctx: Context, image: str, selection: str = "paper"):
    """Seed stability and silhouette at each K, with the label-free choice marked."""
    import pandas as pd

    config, _ = selected_config(ctx, image, selection)
    arrays = _arrays(ctx.store.root / "k_sweep" / image / config_id(config))
    columns = ["image", "k", "graphmae_seed_stability", "graphmae_silhouette", "cnn_seed_stability",
               "cnn_silhouette", "label_free_k"]
    if arrays is None:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame({"image": image, "k": K_VALUES,
                          "graphmae_seed_stability": arrays["graphmae_stability"],
                          "graphmae_silhouette": arrays["graphmae_silhouette"],
                          "cnn_seed_stability": arrays["cnn_stability"], "cnn_silhouette": arrays["cnn_silhouette"]})
    table["label_free_k"] = table["k"] == label_free_k(arrays["graphmae_stability"])
    return table[columns]


def compare_with_paper(scores, k_sweep, image: str):
    """One row per number compared with the paper: this run, the paper, the difference, the tolerance and a status.

    Status is ``within`` or ``outside`` the tolerance of that kind of quantity, or ``not run``
    when the fit is missing, so an incomplete run cannot look like a complete agreement.
    """
    import pandas as pd

    reference = paper_reference()["images"][image]
    rows = []

    def add(quantity, seed, kind, value, paper):
        row = {"image": image, "quantity": quantity, "seed": seed, "kind": kind, "paper": paper,
               "tolerance": TOLERANCE[kind]}
        if value is None:
            rows.append({**row, "this_run": np.nan, "difference": np.nan, "status": "not run"})
            return
        status = "within" if abs(value - paper) <= TOLERANCE[kind] else "outside"
        rows.append({**row, "this_run": value, "difference": value - paper, "status": status})

    per_seed = {
        "CNN K-means": (reference["per_seed"]["CNN K-means"], "cnn_kmeans"),
        "GraphMAE": (reference["controls_per_seed"]["full"], "graphmae"),
        "GraphMAE, shuffled edges": (reference["controls_per_seed"]["shuffled_edges"], "graphmae"),
        "GraphMAE, no edges": (reference["controls_per_seed"]["no_edges"], "graphmae"),
    }
    for method, (values, kind) in per_seed.items():
        mine = scores[scores["method"] == method].set_index("seed")["ARI"]
        for seed, paper in enumerate(values):
            add(f"{method}, ARI", seed, kind, float(mine[seed]) if seed in mine.index else None, paper)
        complete = all(seed in mine.index for seed in range(len(values)))
        add(f"{method}, mean ARI over {len(values)} seeds", "mean", kind,
            float(mine.loc[list(range(len(values)))].mean()) if complete else None, float(np.mean(values)))
    single = {"Pixel K-means": reference["pixel_kmeans"]}
    for name in ("Patch-mean K-means", "Multi-Otsu"):
        single[name] = reference["method_comparison"][name]["ari_mean"]
    for method, paper in single.items():
        mine = scores[scores["method"] == method]["ARI"]
        kind = "pixel_kmeans" if method == "Pixel K-means" else "deterministic"
        add(f"{method}, ARI", 0, kind, float(mine.iloc[0]) if len(mine) else None, paper)
    mine = k_sweep.set_index("k")["graphmae_seed_stability"] if len(k_sweep) else {}
    for k, paper in zip(reference["k_sweep"]["k"], reference["k_sweep"]["seed_stability"]):
        add(f"GraphMAE seed stability at K = {k}", "0-2", "stability", float(mine[k]) if len(k_sweep) else None, paper)
    paper_k = reference["k_sweep"]["label_free_k"]
    row = {"image": image, "quantity": "label-free choice of K", "seed": "0-2", "kind": "choice", "paper": paper_k,
           "tolerance": 0.0}
    if len(k_sweep):
        chosen = int(k_sweep.loc[k_sweep["label_free_k"], "k"].iloc[0])
        rows.append({**row, "this_run": chosen, "difference": chosen - paper_k,
                     "status": "within" if chosen == paper_k else "outside"})
    else:
        rows.append({**row, "this_run": np.nan, "difference": np.nan, "status": "not run"})
    return pd.DataFrame(rows, columns=COMPARE_COLUMNS)


def write_report(ctx: Context, images, selection: str = "paper") -> int:
    """Write the score tables and the comparison with the paper.

    The tables hold the results that exist. Returns 0 when every final task has a result and 1
    when some are missing; the missing tasks are printed.
    """
    import pandas as pd

    out = ctx.store.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    all_scores, all_k, all_compare, unfinished = [], [], [], []
    for image in images:
        scores = score_image(ctx, image, selection)
        k_sweep = k_table(ctx, image, selection)
        all_scores.append(scores)
        all_k.append(k_sweep)
        unfinished += unfinished_tasks(ctx, image, selection)
        if ctx.smoke:
            continue
        if scores.attrs["config_id"] == paper_reference()["images"][image]["joint_config_id"]:
            all_compare.append(compare_with_paper(scores, k_sweep, image))
        else:
            log(f"{image}: the selected configuration {scores.attrs['config_id']} is not the paper's "
                f"({paper_reference()['images'][image]['joint_config_id']}); scores are reported without a comparison")
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(out / "scores_per_seed.csv", index=False)
    summary = (scores.groupby(["image", "method"])["ARI"].agg(["mean", "std", "count"])
               .rename(columns={"mean": "ARI_mean", "std": "ARI_sd", "count": "n_seeds"}).reset_index())
    summary.to_csv(out / "scores_summary.csv", index=False)
    k_sweep = pd.concat(all_k, ignore_index=True)
    k_sweep.to_csv(out / "k_sweep.csv", index=False)
    print("\nScores (ARI against the manual annotation)")
    print(summary.round(3).to_string(index=False) if len(summary) else "no finished fits")
    if unfinished:
        print(f"\nFinal tasks without a result ({len(unfinished)}): " + ", ".join(unfinished))
    if ctx.smoke:
        print("\nSmoke run: scores are from a crop with tiny training budgets and are not compared with the paper.")
    if not all_compare:
        # A comparison written by an earlier report of another configuration would not belong to these tables.
        (out / "comparison_with_paper.csv").unlink(missing_ok=True)
    else:
        compare = pd.concat(all_compare, ignore_index=True)
        compare.to_csv(out / "comparison_with_paper.csv", index=False)
        single = [f"{name}, ARI" for name in ("Pixel K-means", "Patch-mean K-means", "Multi-Otsu")]
        headline = compare[(compare["seed"].astype(str) == "mean") | compare["quantity"].isin(single)
                           | (compare["kind"] == "choice")]
        print("\nComparison with the paper (means, single runs and the choice of K; "
              "every row is in comparison_with_paper.csv)")
        print(headline.drop(columns=["kind", "tolerance"]).round(4).to_string(index=False))
        counts = compare["status"].value_counts()
        missing = int(counts.get("not run", 0))
        print(f"\nOf {len(compare)} numbers compared with the paper: {int(counts.get('within', 0))} within tolerance, "
              f"{int(counts.get('outside', 0))} outside, {missing} not run.")
        print("Tolerances (absolute ARI): " + ", ".join(f"{kind} {value}" for kind, value in TOLERANCE.items()))
        outside = compare[compare["status"] == "outside"]
        if len(outside):
            print("\nOutside the tolerance:")
            print(outside.drop(columns=["kind", "tolerance"]).round(4).to_string(index=False))
            print(f"\nThe paper's values were computed on an {PAPER_GPU}. On another GPU model, rows outside the "
                  "tolerance are expected (README, 'Notes on exact reproduction').")
    log(f"report written to {out}")
    return 1 if unfinished else 0

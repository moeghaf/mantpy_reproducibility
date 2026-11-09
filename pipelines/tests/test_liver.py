"""Fast checks of the liver pipeline's logic on synthetic data, and one check against the paper on the real data."""

import numpy as np
import pandas as pd
import pytest

from mantpy_repro.liver import classify, descriptive, tune
from mantpy_repro.liver import config as C
from mantpy_repro.liver import data as D
from mantpy_repro.liver.cli import main
from mantpy_repro.liver.core import relabel_by_size, spread_deficit, stability
from mantpy_repro.liver.final import final_tasks, task_ranges
from mantpy_repro.liver.freeze import select, tie_bands
from mantpy_repro.liver.search import warm_start
from mantpy_repro.store import read_json

MICE = [f"mouse_{i:02d}" for i in range(1, 9)]


# ------------------------------------------------------------------------------ configurations
def test_paper_configuration_has_the_papers_identity():
    selected = C.paper_selection()
    assert C.original_config_id(selected["config"]) == selected["config_id"]
    assert f"liver6/{C.encoder_id(selected['config'])}" == selected["encoder_id"]
    assert C.config_id(selected["config"]) == C.paper_reference()["config_id"]
    assert selected["config"]["patch_size"] == 10 and selected["config"]["n_clusters"] == 6


def test_search_space_and_validity_rules():
    assert len(C.SPACE) == 15 and len(C.ENCODER_AXES) == 14 and C.K_VALUES == tuple(range(3, 13))
    assert [C.max_blocks(size) for size in C.PATCH_SIZES] == [2, 3, 3, 4]
    assert [C.min_patches_per_roi(size) for size in C.PATCH_SIZES] == [40, 10, 5, 3]
    paper = C.paper_selection()["config"]
    with pytest.raises(ValueError):
        C.validate({**paper, "patch_size": 5, "cnn_readout": "flatten"})
    with pytest.raises(ValueError):
        C.validate({**paper, "patch_size": 5, "cnn_blocks": 3})
    with pytest.raises(KeyError):
        C.validate({key: value for key, value in paper.items() if key != "cnn_lr"})
    other_k = C.with_k(paper, 9)
    assert C.encoder_id(other_k) == C.encoder_id(paper) and C.config_id(other_k) != C.config_id(paper)
    assert C.training_steps(paper, 107740) == 106 * 200


def test_warm_start_is_the_papers_design():
    design, record = warm_start(), C.paper_reference()["warm_start_record"]
    assert [item["config_id"] for item in design["configs"]] == record["config_ids"]
    assert (design["sobol_draws"], design["sobol_redrawn"]) == (record["sobol_draws"], record["sobol_redrawn"])


# ---------------------------------------------------------------------------------------- data
def test_patches_and_features_equal_mantpys():
    from mantpy._core._patching import compute_features, extract_patches, resolve_extractors

    image = np.random.default_rng(0).random((6, 47, 52)).astype(np.float32)
    for size in (5, 15):
        blocks = D.image_blocks(image, size)
        patches, cx, cy = extract_patches(image, size)
        assert np.array_equal(blocks, patches)
        assert np.array_equal(D.block_features(blocks), compute_features(patches, resolve_extractors(["mean", "std"])))
        table = D.PatchTable(size, np.zeros(len(blocks), np.int16), *np.divmod(np.arange(len(blocks)), 52 // size)[:2],
                             features=D.block_features(blocks), signal=np.ones(len(blocks), bool))
        assert np.array_equal(table.x, cx) and np.array_equal(table.y, cy)


def test_anchor_parents_coverage_and_projection():
    # One 20 x 20 image: 16 anchors of 5 pixels, four 10-pixel patches, one 15-pixel patch.
    gy, gx = np.divmod(np.arange(16), 4)
    y, x, roi = gy * 5 + 2.5, gx * 5 + 2.5, np.zeros(16, dtype=np.int64)
    at10 = D.parent_rows(roi, y, x, [(20, 20)], 10)
    assert at10.tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    at15 = D.parent_rows(roi, y, x, [(20, 20)], 15)
    assert (at15 >= 0).sum() == 9 and set(at15[at15 >= 0]) == {0}          # anchors beyond 15 pixels have no parent
    labels = D.project_to_anchors(np.array([7, 8]), np.array([1, 2]), 4, at10)   # patches 1 and 2 are signal
    assert labels.tolist() == [-1, -1, 7, 7, -1, -1, 7, 7, 8, 8, -1, -1, 8, 8, -1, -1]


def test_relabelling_spread_guard_and_stability():
    labels, mapping = relabel_by_size(np.array([3, 3, 3, 1, 1, 2, 2]))
    assert mapping == {3: 0, 1: 1, 2: 2} and labels.tolist() == [0, 0, 0, 1, 1, 2, 2]     # ties go to the lower label
    roi = np.repeat(np.arange(4), 10)
    assert spread_deficit(np.zeros(40, int), roi, 1, 10) == 0              # ten patches in each of four ROIs
    assert spread_deficit(np.r_[np.zeros(30, int), np.ones(10, int)], roi, 2, 10) == 2   # cluster 1 reaches one ROI only
    a = np.array([0, 0, 1, 1, 2, 2, -1])
    value, pairs = stability([a, np.array([5, 5, 3, 3, 9, 9, 4]), np.array([1, 1, 0, 0, 2, 2, -1])])
    assert value == pytest.approx(1.0) and len(pairs) == 3                 # uncovered anchors are ignored


# ----------------------------------------------------------------------------------- selection
def _paper_candidates():
    table = pd.DataFrame(C.paper_reference()["encoder"]["k_selection"])
    return table.assign(encoder_id=table.encoder, config_id=table.encoder + "_K" + table.K.astype(str), spread_deficit5=0,
                        mean_fit_seconds5=1.0)


def test_selection_rule_reproduces_the_papers_choice():
    candidates = _paper_candidates()
    selected, rule_met = select(candidates)
    assert rule_met and selected["K"] == 6 and selected["encoder"] == C.paper_reference()["encoder_id"]
    assert selected["stability5"] == pytest.approx(C.paper_selection()["stability5"])
    # A more stable pair at a larger K wins only when it also meets the spread guard.
    larger = candidates.assign(stability5=np.where(candidates.K == 9, 0.9, candidates.stability5))
    assert select(larger)[0]["K"] == 9
    assert select(larger.assign(spread_deficit5=np.where(larger.K == 9, 1, 0)))[0]["K"] == 6
    # Nothing reaches 0.80: the most stable pair is taken and the rule is reported as not met.
    selected, rule_met = select(candidates.assign(stability5=candidates.stability5 - 0.2))
    assert not rule_met and selected["K"] == 3


def test_tie_bands_keep_pairs_within_001_of_the_best_at_each_k():
    pool = pd.DataFrame({"K": [3, 3, 3, 4, 4], "stability3": [0.90, 0.895, 0.88, 0.70, 0.60], "encoder_id": list("abcde")})
    assert tie_bands(pool).encoder_id.tolist() == ["a", "b", "d"]


# --------------------------------------------------------------------------------- descriptive
def _nodes(cluster_by_roi, mouse_of, group_of, xy=None):
    rows = []
    for roi, clusters in cluster_by_roi.items():
        for index, cluster in enumerate(clusters):
            x, y = (index % 10 * 10 + 5, index // 10 * 10 + 5) if xy is None else xy[roi][index]
            rows.append({"roi": roi, "mouse": mouse_of[roi], "group": group_of[mouse_of[roi]], "x": x, "y": y, "cluster": cluster})
    return pd.DataFrame(rows)


def test_cooccurrence_counts_are_l_transpose_a_l():
    cluster = np.array([0, 1, -1, 1, 0, 2])
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [0, 4]])
    observed, sizes = descriptive.ordered_counts(edges, cluster, 3)
    signal = np.flatnonzero(cluster >= 0)
    adjacency = np.zeros((6, 6))
    adjacency[edges[:, 0], edges[:, 1]] = adjacency[edges[:, 1], edges[:, 0]] = 1
    membership = np.eye(3)[cluster[signal]]
    assert np.array_equal(observed, membership.T @ adjacency[np.ix_(signal, signal)] @ membership)
    assert sizes.tolist() == [2, 2, 1]
    score, expected = descriptive.cooccurrence_score(observed, sizes)
    assert expected == pytest.approx(observed.sum() * np.outer(sizes / 5, sizes / 5))
    assert score[0, 1] == pytest.approx(np.log2((observed[0, 1] + 1e-9) / expected[0, 1]))
    absent, _ = descriptive.cooccurrence_score(np.zeros((2, 2)), np.array([3.0, 0.0]))
    assert np.isnan(absent[1, 1])


def test_cooccurrence_ko_minus_wt_uses_the_infected_mice():
    rng = np.random.default_rng(1)
    mouse_of = {f"roi_{i}": MICE[i // 2] for i in range(8)}
    group_of = dict(zip(MICE[:4], ["Infected_KO", "Infected_KO", "Infected_WT", "Infected_WT"]))
    nodes = _nodes({roi: rng.integers(-1, 3, size=100) for roi in mouse_of}, mouse_of, group_of)
    pairs, per_roi = descriptive.cooccurrence(nodes, 3)
    assert len(pairs) == 6 and len(per_roi) == 8 * 9
    first = pairs.iloc[0]
    expected = (first.log2_obs_exp_mouse_01 + first.log2_obs_exp_mouse_02) / 2 - (first.log2_obs_exp_mouse_03 + first.log2_obs_exp_mouse_04) / 2
    assert first.ko_minus_wt == pytest.approx(expected)
    # Pooling within a mouse sums the counts of its ROIs before the score is taken.
    one = per_roi[(per_roi.mouse == "mouse_01") & (per_roi.cluster_a == 0) & (per_roi.cluster_b == 1)]
    assert one.observed.sum() == pairs[(pairs.cluster_a == 0) & (pairs.cluster_b == 1)].observed_mouse_01.iloc[0]


def test_infection_cluster_rule():
    groups = ["Infected_KO", "Infected_KO", "Infected_WT", "Infected_WT", "Naive_KO", "Naive_KO", "Naive_WT", "Naive_WT"]
    mouse_of, group_of = {f"roi_{i}": mouse for i, mouse in enumerate(MICE)}, dict(zip(MICE, groups))

    def nodes(infected_counts, naive_counts):
        return _nodes({f"roi_{i}": np.repeat(np.arange(3), infected_counts[i] if i < 4 else naive_counts)
                       for i in range(8)}, mouse_of, group_of)

    # Cluster 2: 10% of every infected mouse, absent in naive mice -> the largest ratio.
    chosen = descriptive.infection_cluster(nodes([[60, 30, 10]] * 4, [70, 30, 0]), 3)
    assert chosen["cluster"] == 2
    assert chosen["log2_ratio"][2] == pytest.approx(np.log2((0.1 + 1e-3) / 1e-3))
    # The same cluster below 1% in one infected mouse is not a candidate; cluster 1 has the next ratio.
    chosen = descriptive.infection_cluster(nodes([[60, 30, 10]] * 3 + [[1000, 600, 5]], [70, 30, 0]), 3)
    assert not chosen["present_in_every_infected_mouse"][2] and chosen["cluster"] == 1


def test_subgraphs_join_patches_within_40_pixels_and_need_five():
    line = [(i * 40.0, 0.0) for i in range(6)]                       # six patches 40 pixels apart: one chain
    far = [(1000.0 + i * 10, 500.0) for i in range(4)]               # four patches: too small
    square = [(x * 10.0, 300.0 + y * 10.0) for x in range(3) for y in range(3)]
    found = descriptive.components(np.array(line + far + square))
    assert sorted(item["n_nodes"] for item in found) == [6, 9]
    chain = next(item for item in found if item["n_nodes"] == 6)
    assert chain["n_edges"] == 5 and chain["mean_degree"] == pytest.approx(10 / 6)
    assert next(item for item in found if item["n_nodes"] == 9)["n_edges"] == 36     # every pair is within 40 pixels


def test_ig_localisation_is_a_correlation_with_membership():
    cluster = np.array([0] * 6 + [1] * 3 + [2])
    importance = np.where(cluster == 1, 1.0, 0.1) + np.linspace(0, 0.01, 10)
    table = pd.DataFrame({"roi": "roi_1", "mouse": "mouse_01", "group": "Infected_KO", "cluster": cluster, "importance": importance})
    local = descriptive.ig_localisation(table, 3, 1).set_index("cluster")
    assert local.loc[1, "pearson_r"] == pytest.approx(np.corrcoef(importance, cluster == 1)[0, 1]) and local.loc[1, "pearson_r"] > 0.99
    assert np.isnan(local.loc[2, "pearson_r"]) and local.loc[1, "is_infection_cluster"]      # one patch: undefined


# ------------------------------------------------------------------------ permutation and tuning
def test_allocations_and_exact_permutation_p():
    labellings = classify.allocations(MICE)
    genotype = dict(zip(MICE, ["KO", "KO", "WT", "WT", "KO", "KO", "WT", "WT"]))
    preserving = np.array([classify.genotype_preserving(item, genotype) for item in labellings])
    assert len(labellings) == C.N_ALLOCATIONS == 70 and preserving.sum() == 36
    assert labellings[0] == tuple(MICE[:4]) and labellings[69] == tuple(MICE[4:])
    record = C.paper_reference()["permutation_record"]
    auc = np.full(70, 0.5)
    auc[record["allocations_at_observed_auc"]] = 1.0                   # the paper's four labellings with ROC-AUC 1.00
    p = classify.permutation_p(auc, 0, preserving)
    assert (p["n_at_or_above_observed"], p["n_genotype_at_or_above_observed"]) == (4, 2)
    assert p["p"] == pytest.approx(record["p"]) and p["p_genotype"] == pytest.approx(record["p_genotype"])
    assert classify.scores([0.7, 0.6, 0.4, 0.55], [1, 1, 0, 0]) == {"n": 4, "correct": 3, "auc": 1.0}


def test_tuning_space_and_replay():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = set()
    for _ in range(30):
        trial = study.ask()
        config = tune.suggest(trial)
        seen |= set(config)
        assert ("aggr" in config) == (config["conv"] == "sage") and ("heads" in config) == (config["conv"] == "gat")
        assert set(tune.distributions(config)) == set(config) == set(trial.params)
        replay = optuna.create_study()
        replay.add_trial(optuna.trial.create_trial(params=config, distributions=tune.distributions(config), value=1.0))
        study.tell(trial, 1.0)
    assert len(seen) == 19
    for record in C.paper_selection()["tuned_classifier"].values():      # the paper's choices lie in the space
        assert set(tune.distributions({k: v for k, v in record["config"].items()})) == set(record["config"])


def test_inner_folds_never_remove_a_class():
    truth = {"a": 1, "b": 1, "c": 1, "d": 0, "e": 0, "f": 0, "g": 0}
    assert [held for _, held in tune.inner_folds(list(truth), truth)] == list("abcdefg")
    assert [held for _, held in tune.inner_folds(["a", "d", "e"], truth)] == ["d", "e"]


def test_final_tasks():
    tasks, ranges = final_tasks(), task_ranges()
    assert len(tasks) == 5 + 8 + 8 + 1 + 69 + 8
    assert ranges["encoder"] == (0, 4) and ranges["permutation"] == (22, 90) and ranges["tuned"] == (91, 98)
    assert 0 not in [task["allocation"] for task in tasks if task["kind"] == "permutation"]


# ------------------------------------------------------------- whole pipeline on a synthetic cohort
def _synthetic_cohort(cache_dir=None, crop=None):
    """Eight mice and 24 ROIs of 80 x 80 pixels: tissue tiles with distinct marker profiles on a dim background."""
    rng = np.random.default_rng(0)
    profiles = np.array([[1, 1, 1, 1, 1, 1], [30, 5, 5, 20, 5, 5], [5, 30, 20, 5, 5, 5], [5, 5, 5, 5, 30, 20],
                         [40, 5, 5, 5, 40, 30]], dtype=float)
    rows, images = [], {}
    design = [("Infected", "KO", 4)] * 2 + [("Infected", "WT", 4)] * 2 + [("Naive", "KO", 2)] * 2 + [("Naive", "WT", 2)] * 2
    for mouse, (condition, genotype, n_rois) in zip(MICE, design):
        for _ in range(n_rois):
            roi = f"roi_{len(rows) + 1:03d}"
            weights = [0.45, 0.2, 0.15, 0.1, 0.1] if condition == "Infected" else [0.45, 0.25, 0.2, 0.1, 0.0]
            tiles = rng.choice(5, size=(8, 8), p=weights)
            rate = profiles[np.kron(tiles, np.ones((10, 10), dtype=int))].transpose(2, 0, 1)
            images[roi] = rng.poisson(rate).astype(np.float32)
            rows.append({"roi": roi, "mouse": mouse, "condition": condition, "genotype": genotype, "group": f"{condition}_{genotype}"})
    return images, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setattr(D, "load_cohort", _synthetic_cohort)
    yield tmp_path_factory.mktemp("liver")
    patch.undo()


SMOKE = ["--smoke", "--device", "cpu", "--threads", "2"]


@pytest.fixture(scope="module")
def smoke_all(synthetic):
    """Every stage, with this run's own selections, on the synthetic cohort."""
    assert main(["all", *SMOKE, "--results", str(synthetic / "all")]) == 0
    return synthetic / "all"


def test_smoke_run_of_every_stage_on_synthetic_data(smoke_all, capsys):
    assert main(["report", *SMOKE, "--selection", "search", "--results", str(smoke_all)]) == 0
    printed = capsys.readouterr().out
    assert "Not finished" not in printed and "Infection-associated cluster" in printed
    selection = read_json(smoke_all / "selection" / "selection.json")
    assert selection["config_id"] == C.config_id(selection["config"]) and selection["K"] in C.K_VALUES
    report = smoke_all / "report"
    summary = pd.read_csv(report / "classifier_summary.csv")
    assert len(summary) == 4 and (summary.n_mice == 8).all() and (summary.n_rois == 24).all()
    assert len(pd.read_csv(report / "classifier_per_seed.csv")) == 24 * 5
    assert len(pd.read_csv(report / "permutation_allocations.csv")) == 3
    assert not (report / "comparison_with_paper.csv").exists()            # a smoke run is never compared with the paper
    fractions = pd.read_csv(report / "composition_per_mouse.csv", index_col=0)
    assert fractions.drop(columns="group").sum(axis=1).to_numpy() == pytest.approx(1.0)
    tuned = pd.read_csv(report / "tuned_settings.csv")
    assert len(tuned) == 8 and set(tuned.graph_type) <= set(C.GRAPH_TYPES)


def test_finished_work_is_not_repeated_and_modes_do_not_mix(smoke_all, monkeypatch):
    from mantpy_repro.liver import models

    arguments = [*SMOKE, "--selection", "search", "--results", str(smoke_all)]

    def refuse(*args, **kwargs):
        raise AssertionError("a finished fit was started again")

    monkeypatch.setattr(models, "fit_cnn", refuse)
    monkeypatch.setattr(models, "fit_classifier", refuse)
    assert main(["final", *arguments]) == 0
    selected = next((smoke_all / "tuning").glob("*/mouse_01/selected/completion.json"))
    before = read_json(selected)["metadata"]
    selected.unlink()                                                     # as if this fold had been interrupted at its end
    assert main(["tune", *arguments]) == 0
    after = read_json(selected)["metadata"]                               # finished configurations are replayed, not refitted
    assert after["config"] == before["config"] and [t["objective"] for t in after["trials"]] == [t["objective"] for t in before["trials"]]
    with pytest.raises(RuntimeError):                                     # different thread count: different fits
        main(["final", "--task-index", "5", *arguments[:3], "--threads", "3", *arguments[5:]])
    with pytest.raises(RuntimeError):                                     # a full run must not reuse a smoke directory
        main(["report", "--results", str(smoke_all), "--device", "cpu"])


def test_comparison_rows_are_built_from_the_finished_fits(smoke_all, tmp_path):
    from mantpy_repro.liver import report as R
    from mantpy_repro.liver.core import Context
    from mantpy_repro.liver.final import resolve_partition

    ctx = Context(smoke_all, device="cpu", threads=2, smoke=True)
    partition, compare = resolve_partition(ctx, "search", None), R.Comparison()
    cluster = R.descriptive_report(ctx, partition, tmp_path, compare)
    R.classifier_report(ctx, partition, "search", cluster, tmp_path, compare)
    table = compare.table().set_index("quantity")
    assert table.loc["classifier, mice correct", "paper"] == 8 and table.loc["classifier, ROIs correct", "paper"] == 20
    assert table.loc["logistic regression, mice correct", "paper"] == 6
    assert table.loc["untuned classifier, mice correct", "paper"] == 7
    assert table.loc["tuned within each fold (this run's tuning), mice correct", "paper"] == 7
    assert table.loc["classifier, P(infected), mouse_01, seed 0", "paper"] == pytest.approx(0.7286, abs=1e-4)
    assert table.loc["integrated gradients, median correlation, Infected_WT", "paper"] == pytest.approx(0.9551, abs=1e-4)
    assert table.loc["infection-associated cluster, mean mouse fraction, Infected_WT", "paper"] == pytest.approx(0.152408, abs=1e-6)
    assert sum(name.startswith("classifier, P(infected)") for name in table.index) == 8 + 40
    assert not any(name.startswith("co-occurrence") for name in table.index)     # cluster numbers are this run's own
    assert table.within_tolerance.dtype == bool and (table.tolerance >= 0).all()


def test_paper_settings_run_on_synthetic_data(synthetic, capsys, tmp_path):
    """The 'reproduce' route: the paper's encoder and tuned settings, with a subset of the final tasks."""
    from dataclasses import replace

    from mantpy_repro.liver import report as R
    from mantpy_repro.liver.core import Context
    from mantpy_repro.liver.final import resolve_partition

    arguments = [*SMOKE, "--results", str(synthetic / "paper")]
    ranges = task_ranges(smoke=True)
    assert main(["prepare", *arguments]) == 0
    for kind in ("encoder", "classifier", "tuned"):
        assert main(["final", "--task-index", str(ranges[kind][0]), *arguments]) == 0
    assert main(["report", *arguments]) == 0
    printed = capsys.readouterr().out
    assert "K = 6" in printed and "Not finished, so not reported: classifier, composition_lr, explanation, tuned." in printed
    with pytest.raises(SystemExit):
        main(["final", "--partition", "paper", *arguments])
    # With the deposited partition the clusters carry the paper's numbers, so numbered quantities are compared too.
    ctx = Context(synthetic / "paper", device="cpu", threads=2, smoke=True)
    numbered, compare = replace(resolve_partition(ctx, "paper", None), name="paper"), R.Comparison()
    R.descriptive_report(ctx, numbered, tmp_path, compare)
    names = compare.table().quantity
    assert names.str.startswith("co-occurrence KO - WT").sum() == 21 and names.str.startswith("fraction of ECM").sum() == 48
    assert names.str.match(r"ECM \d, patches").sum() == 6


# --------------------------------------------------------------- against the paper (GPU, slow)
@pytest.mark.gpu
def test_real_data_statistics_and_seed0_encoder_reproduce_the_paper(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from sklearn.metrics import adjusted_rand_score

    from mantpy_repro.liver.core import Context

    reference, selected = C.paper_reference(), C.paper_selection()["config"]
    ctx = Context(tmp_path)
    assert ctx.prepared.bounds == pytest.approx(list(reference["bounds"].values()), abs=1e-5)
    assert abs(int(ctx.prepared.table(10).signal.sum()) - reference["n_signal"]) <= 1000
    partition = ctx.partition(selected, "paper", "paper")
    assert np.bincount(partition.nodes.cluster[partition.nodes.cluster >= 0]).tolist() == reference["cluster_sizes"]
    assert descriptive.infection_cluster(partition.nodes, 6)["cluster"] == reference["infection_cluster"]
    pairs = descriptive.cooccurrence(partition.nodes, 6)[0].set_index(["cluster_a", "cluster_b"])
    for record in reference["cooccurrence"]:
        assert pairs.loc[(record["cluster_a"], record["cluster_b"]), "ko_minus_wt"] == pytest.approx(record["ko_minus_wt"], abs=1e-6)
    regions = descriptive.subgraphs(partition.nodes, 5)
    assert regions[["n_nodes", "n_edges"]].to_numpy().tolist() == [[row["n_nodes"], row["n_edges"]] for row in reference["subgraphs"]["rows"]]
    refit = ctx.encoder_fit(selected, 0, "paper")["labels"][C.K_VALUES.index(6)]
    assert adjusted_rand_score(C.paper_partition()[1], refit) > 0.9

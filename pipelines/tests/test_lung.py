"""Fast checks of the lung pipeline on synthetic data. Nothing here reads the lung dataset
except the one test marked ``gpu`` (the marker of the slow tests that rerun the paper)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mantpy_repro.lung import config as C
from mantpy_repro.lung import corruption, data, denoise, describe, graphs, refit, report, search
from mantpy_repro.lung.core import Budget, Context, pack, unpack
from mantpy_repro.store import Store

ROIS = [f"roi_{i:03d}" for i in range(1, 7)]
PATCH = 10


# ------------------------------------------------------------------------------- synthetic data
def toy_roi(n: int = 6, n_cells: int = 40, seed: int = 0, name: str = "roi_001", mouse: str = "mouse_01"):
    """An n x n patch grid with random cells; the first patch is background."""
    rng = np.random.default_rng(seed)
    gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    patch_xy = np.c_[gx.ravel() * PATCH + PATCH / 2, gy.ravel() * PATCH + PATCH / 2].astype(float)
    label = rng.integers(0, 3, size=n * n)
    label[0] = -1
    types = rng.integers(0, 4, size=n_cells)
    return data.ROIData(
        roi=name, mouse=mouse, patch_size=PATCH, patch_id=np.array([f"{name}:{i}" for i in range(n * n)]),
        patch_xy=patch_xy, patch_feat=rng.normal(size=(n * n, 3)), label=label,
        cell_id=np.array([f"cell_{i}" for i in range(n_cells)]), cell_xy=rng.uniform(0, n * PATCH, size=(n_cells, 2)),
        cell_type=np.array([f"type{t}" for t in types]), cell_type_idx=types)


def structured_cohort(n: int = 16, seed: int = 0, n_cells: int = 1024) -> data.Cohort:
    """Six ROIs from two mice whose three ECM clusters are vertical bands, each with its own cell type."""
    rng = np.random.default_rng(seed)
    rois = {}
    for index, name in enumerate(ROIS):
        gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        patch_xy = np.c_[gx.ravel() * PATCH + PATCH / 2, gy.ravel() * PATCH + PATCH / 2].astype(float)
        label = np.minimum(gx.ravel() * 3 // n, 2)
        label[gy.ravel() == 0] = -1
        cell_xy = rng.uniform(0, n * PATCH, size=(n_cells, 2))
        band = np.minimum((cell_xy[:, 0] // PATCH).astype(int) * 3 // n, 2)
        types = np.where(rng.random(len(band)) < 0.9, band, rng.integers(0, 3, size=len(band)))
        rois[name] = data.ROIData(
            roi=name, mouse="mouse_01" if index < 3 else "mouse_02", patch_size=PATCH,
            patch_id=np.array([f"{name}:{i}" for i in range(n * n)]), patch_xy=patch_xy,
            patch_feat=rng.normal(size=(n * n, 3)) + label[:, None], label=label,
            cell_id=np.array([f"cell_{i}" for i in range(len(types))]), cell_xy=cell_xy,
            cell_type=np.array([f"type{t}" for t in types]), cell_type_idx=types)
    return data.Cohort("toy", rois, ["type0", "type1", "type2"], 3, ["m0", "m1", "m2"])


def synthetic_bundle(size: int = 160, seed: int = 0):
    """A stand-in for the public dataset: six small ROIs, four channels of which three are ECM markers."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({"channel": ["c0", "c1", "c2", "c3", "c4"], "name": ["DNA", "m0", "m1", "m2", "unused"],
                          "keep": [1, 1, 1, 1, 0], "ecm": [0, 1, 1, 1, 0]})
    images, cells = {}, {}
    for name in ROIS:
        band = np.minimum(np.arange(size) * 4 // size, 3)                 # band 0 is dim background
        image = rng.gamma(2.0, 0.2, size=(4, size, size)).astype(np.float32)
        for marker in range(3):
            image[marker + 1][:, band == marker + 1] += 4.0
        images[name] = image
        xy = rng.uniform(0, size, size=(300, 2))
        zone = np.minimum(xy[:, 0].astype(int) * 4 // size, 3)
        types = np.where(rng.random(300) < 0.9, zone, rng.integers(0, 4, size=300))
        adata = ad.AnnData(obs=pd.DataFrame({"cell_type": [("ATI", "AEC", "ATII", "Fibroblast")[t] for t in types]},
                                            index=[f"cell_{i:06d}" for i in range(300)]))
        adata.obsm["spatial"] = xy
        cells[name] = adata
    metadata = pd.DataFrame({"sample_id": ROIS, "Mouse": ["mouse_01"] * 3 + ["mouse_02"] * 3})
    return SimpleNamespace(images=images, panel=panel, metadata=metadata, cohort_cells=cells, roi_names=list(ROIS),
                           ecm_names=["m0", "m1", "m2"])


# ------------------------------------------------------------------------------ configurations
def test_paper_configurations_round_trip_and_reference_graph():
    selected = C.paper_selection()
    saved = [entry for scheme in C.SCHEMES for contexts in selected[scheme].values() for entry in contexts.values()]
    saved += [entry for folds in selected["refit_control"]["normalization_selection"].values()
              for contexts in folds.values() for entry in contexts.values()]
    assert len(saved) == 6 + 18 + 36
    for entry in saved:
        assert C.Config.from_dict(entry["config"]).to_dict() == entry["config"]
    assert C.GraphRecipe.from_dict(selected["reference_graph"]) == C.GraphRecipe()
    assert selected["benchmark"]["evaluation_seeds"] == list(C.EVALUATION_SEEDS)
    assert selected["benchmark"]["tuning_seeds"] == list(C.TUNING_SEEDS)
    assert selected["benchmark"]["block_side_px"] == list(C.BLOCK_SIDES) and selected["benchmark"]["types"] == list(C.KINDS)
    assert selected["refit_control"]["kmeans"] == refit.KMEANS
    assert selected["refit_control"]["normalizations"] == list(C.NORMALIZATIONS)
    assert selected["partition"]["total_k"] == report.paper_reference()["selected_total_components"] == 4
    assert not set(C.EVALUATION_SEEDS) & set(C.TUNING_SEEDS)


def test_configuration_identity_and_validation():
    selected = C.paper_selection()["mouse"]["mouse_01"]["cell"]["config"]
    config = C.Config.from_dict(selected)
    assert C.config_id(config) == C.config_id(C.Config.from_dict(json.loads(json.dumps(selected))))
    other = C.Config.from_dict({**selected, "calibration": "temperature"})
    assert C.config_id(other) != C.config_id(config)
    with pytest.raises(ValueError):
        C.GraphRecipe(ce_rule="nearest")
    with pytest.raises(ValueError):
        C.Config(context="cells")
    with pytest.raises(ValueError):                                       # neighbouring intensities are not part of the task
        C.Config.from_dict({**selected, "features": {**selected["features"], "intensity_block": True}})


def test_selected_settings_format_as_supplementary_table_6():
    table = report.selection_table(C.paper_selection())
    paper = report.paper_reference()["supplementary_table_6"]
    assert table.astype(str).to_dict("records") == paper


def test_folds_seeds_and_task_counts():
    mice = C.paper_selection()["mice"]
    mouse_folds = C.outer_folds(mice, "mouse")
    assert [fold["test"] for fold in mouse_folds] == [ROIS[:3], ROIS[3:]]
    assert all(not set(fold["test"]) & set(fold["train"]) for fold in mouse_folds + C.outer_folds(mice, "roi"))
    assert len(C.inner_folds(mouse_folds[0]["train"])) == 3 and len(C.inner_folds(C.outer_folds(mice, "roi")[0]["train"])) == 5
    assert [C.sampler_seed("mouse", fold, context) for fold in range(2) for context in C.CONTEXTS] == [1036, 1037, 1038, 1042, 1043, 1044]
    assert C.sampler_seed("roi", 5, "joint") == 1032 and C.sampler_seed("refit", 5, "joint") == 2017
    assert (C.startup_trials("mouse", 60), C.startup_trials("roi", 150), C.startup_trials("refit", 150)) == (12, 30, 30)
    from mantpy_repro.lung.final import final_tasks

    assert len(search.search_tasks(mice)) + len(refit.refit_tasks(mice)) == 6 + 18 + 36
    assert len(final_tasks(mice)) == 8 + 24 + 36


# -------------------------------------------------------------------------------------- search
def test_search_space_sampling_and_selection_rule():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for context in C.CONTEXTS:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=12, seed=1036)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        configs = []
        study.optimize(lambda trial: configs.append(search.sample_config(trial, context)) or 0.5, n_trials=20)
        again = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, n_startup_trials=12, seed=1036))
        repeat = []
        again.optimize(lambda trial: repeat.append(search.sample_config(trial, context)) or 0.5, n_trials=20)
        assert configs == repeat                                          # a study is determined by its seed
        for config in configs:
            assert C.Config.from_dict(config.to_dict()) == config
            assert config.cleaning.passes == 1 or context != "cell"
            if config.cleaning.gate_radius_px == 0:
                assert config.cleaning.gate_min_neighbours == 0 and not config.cleaning.majority_smoothing
    trials = {"trial": np.arange(5), "value": np.array([0.90, 0.95, 0.95004, 0.9499, 0.80]),
              "edges": np.array([10, 500, 300, 300, 1])}
    assert search.select_best(trials) == 2                                # tied within 1e-4: fewest edges, then earliest
    trials["edges"][1] = 300
    assert search.select_best(trials) == 1


def test_edge_rules_are_the_papers():
    paper = report.paper_reference()["edge_rules"]
    recipes = describe.edge_rules(C.N_EDGE_RULES)
    assert len(recipes) == len(paper) == 49 and recipes[0] == C.GraphRecipe()
    for recipe, entry in zip(recipes, paper):
        assert (recipe.ce_rule, recipe.ce_k, recipe.ce_dmax_px, recipe.ee_include_background_nodes) == (
            entry["ce_rule"], entry["ce_k"], entry["ce_dmax_px"], entry["ee_include_background_nodes"])


# -------------------------------------------------------------------------------------- graphs
def test_graph_rules_and_weights():
    roi = toy_roi(5, n_cells=20)
    g = graphs.build_graphs(roi.cell_xy, roi.patch_xy, roi.background, PATCH, C.GraphRecipe())
    assert g.n_edges()["ee"] == 72                                        # 20 + 20 + 32 on a 5 x 5 grid
    assert g.ee_d.diagonal().sum() == 0 and g.cc_d.diagonal().sum() == 0
    assert (g.ce_d.data <= 15.0).all() and (np.diff(g.ce_d.indptr) <= 5).all() and g.cc_d.max() <= 15.0
    g4 = graphs.build_graphs(roi.cell_xy, roi.patch_xy, roi.background, PATCH, C.GraphRecipe(ee_rule="grid4"))
    assert g4.n_edges()["ee"] == 40
    signal_only = graphs.build_graphs(roi.cell_xy, roi.patch_xy, roi.background, PATCH,
                                      C.GraphRecipe(ce_dmax_px=None, ee_include_background_nodes=False))
    assert signal_only.ce_d[:, 0].nnz == 0 and signal_only.ee_d[0].nnz == 0
    assert (np.diff(signal_only.ce_d.indptr) == 5).all()
    for rule in C.EDGE_WEIGHTS:
        w = graphs.build_graphs(roi.cell_xy, roi.patch_xy, roi.background, PATCH, C.GraphRecipe(edge_weight=rule))
        assert w.ee_w.nnz == w.ee_d.nnz and np.isfinite(w.ce_w.data).all() and (w.ce_w.data > 0).all()


# ------------------------------------------------------------------------------------ features
def test_composition_features_by_hand():
    # Three patches in a row; cells 0 and 1 touch patch 0, cell 2 touches patch 1, cell 3 touches none.
    patch_xy = np.array([[5.0, 5.0], [15.0, 5.0], [25.0, 5.0]])
    cell_xy = np.array([[4.0, 5.0], [6.0, 5.0], [15.0, 4.0], [60.0, 60.0]])
    g = graphs.build_graphs(cell_xy, patch_xy, np.zeros(3, bool), PATCH,
                            C.GraphRecipe(ce_rule="knn", ce_k=1, ce_dmax_px=5.0, ee_rule="grid4"))
    types, labels = np.array([0, 1, 1, 0]), np.array([0, 1, 1])
    count = C.FeatureRecipe(composition_norm="count")
    assert np.allclose(denoise.context_features(g, types, 2, labels, 2, "cell", count), [[1, 1], [0, 1], [0, 0]])
    assert np.allclose(denoise.context_features(g, types, 2, labels, 2, "ecm", count), [[0, 1], [1, 1], [0, 1]])
    fraction = denoise.context_features(g, types, 2, labels, 2, "ecm", C.FeatureRecipe(composition_norm="fraction"))
    assert np.allclose(fraction[1], [0.5, 0.5])


@pytest.mark.parametrize("hops", [1, 2])
def test_a_patch_never_sees_its_own_label(hops):
    roi = toy_roi(6, n_cells=40, seed=1)
    g = graphs.build_graphs(roi.cell_xy, roi.patch_xy, roi.background, PATCH,
                            C.GraphRecipe(ce_rule="knn", ce_k=4, ce_dmax_px=20.0, edge_weight="inv_d"))
    recipe = C.FeatureRecipe(composition_norm="count", hops=hops)
    before = denoise.context_features(g, roi.cell_type_idx, 4, roi.label, 3, "joint", recipe)
    for patch in (7, 14, 21):
        changed = roi.label.copy()
        changed[patch] = (roi.label[patch] + 1) % 3
        after = denoise.context_features(g, roi.cell_type_idx, 4, changed, 3, "joint", recipe)
        assert np.allclose(before[patch], after[patch])


# ---------------------------------------------------------------------------------- corruption
def test_corruption_budget_blocks_and_seed():
    roi = toy_roi(30, n_cells=10)
    signal = roi.label >= 0
    for kind in C.KINDS:
        made = corruption.corrupt_labels(roi.patch_xy, roi.label, 3, kind, seed=3, block_sides=(50.0, 100.0))
        assert made.is_artifact.sum() == round(0.10 * signal.sum())
        assert (made.corrupted[made.is_artifact] != roi.label[made.is_artifact]).all()
        assert (made.corrupted[~made.is_artifact] == roi.label[~made.is_artifact]).all()
        assert (made.corrupted[~signal] == -1).all()
        again = corruption.corrupt_labels(roi.patch_xy, roi.label, 3, kind, seed=3, block_sides=(50.0, 100.0))
        assert np.array_equal(made.corrupted, again.corrupted)
    mixed = corruption.corrupt_labels(roi.patch_xy, roi.label, 3, "mixed", seed=3)
    assert sum(block["n_flipped"] for block in mixed.blocks) == mixed.is_artifact.sum() // 2
    cohort = data.Cohort("toy", {"a": roi, "b": roi}, [], 3)
    realisation = corruption.make_realisation(cohort, "mixed", 2)          # ROI i is seeded with seed * 1000 + i
    assert np.array_equal(realisation["b"].corrupted, corruption.corrupt_labels(roi.patch_xy, roi.label, 3, "mixed", 2001).corrupted)
    assert not np.array_equal(realisation["a"].corrupted, realisation["b"].corrupted)


def test_paired_corruption_shares_positions_between_partitions():
    roi = toy_roi(30, n_cells=10, name="roi_004")
    two_clusters = np.where(roi.label >= 0, np.minimum(roi.label, 1), -1)
    common = (roi.label >= 0) & (two_clusters >= 0)
    proposal = corruption.physical_proposal(roi.patch_xy, common, "mixed", 0, "roi_004")
    assert proposal["proposed_nodes"] == proposal["requested_nodes"] == round(0.1 * common.sum())
    first = corruption.apply_proposal(roi.label, 3, proposal, "mixed", 0)
    second = corruption.apply_proposal(two_clusters, 2, proposal, "mixed", 0)
    assert not (first.is_artifact & ~proposal["proposed"]).any() and not (second.is_artifact & ~proposal["proposed"]).any()
    assert first.is_artifact[proposal["isolated"]].all() and second.is_artifact[proposal["isolated"]].all()
    assert first.is_artifact.sum() <= proposal["proposed_nodes"]          # a block may propose the label already there


# ------------------------------------------------------------------------------------ cleaning
def test_cleaning_rule_threshold_confidence_and_gate():
    n = 9
    xy = np.c_[np.arange(n) * 10.0, np.zeros(n)]
    observed, signal = np.zeros(n, int), np.ones(n, bool)
    proba = np.tile([0.9, 0.1], (n, 1))
    proba[[2, 3, 4]] = [0.1, 0.9]                                          # three adjacent confident anomalies
    proba[7] = [0.1, 0.9]                                                  # one isolated anomaly
    gated = C.CleaningRecipe(anomaly_stat="raw_neglogp", threshold=1.0, gate_radius_px=10.5, gate_min_neighbours=1)
    new, _, relabelled = denoise.clean_once(proba, observed, signal, xy, gated)
    assert relabelled[[2, 3, 4]].all() and not relabelled[7] and new[3] == 1 and new[7] == 0
    free = C.CleaningRecipe(anomaly_stat="raw_neglogp", threshold=1.0, gate_radius_px=0.0, gate_min_neighbours=0)
    assert denoise.clean_once(proba, observed, signal, xy, free)[0][7] == 1
    strict = C.CleaningRecipe(anomaly_stat="raw_neglogp", threshold=1.0, min_pred_conf=0.95, gate_radius_px=0.0)
    assert not denoise.clean_once(proba, observed, signal, xy, strict)[2].any()
    assert np.allclose(denoise.raw_anomaly(proba, observed, signal)[[0, 2]], -np.log([0.9, 0.1]))
    assert denoise.fit_temperature(np.tile([0.6, 0.4], (50, 1)), np.zeros(50, int)) == 0.5


def test_metrics():
    pristine = np.array([0, 0, 1, 1, 2, 2, -1])
    observed = pristine.copy()
    observed[[0, 2]] = [1, 0]
    replaced, signal = observed != pristine, pristine >= 0
    cleaned = observed.copy()
    cleaned[0], cleaned[5] = 0, 0
    scores = denoise.reconstruction_metrics(pristine, observed, cleaned, replaced, signal)
    assert scores["accuracy_before"] == pytest.approx(4 / 6) and scores["accuracy_after"] == pytest.approx(4 / 6)
    assert scores["recovery"] == pytest.approx(0.5) and scores["correction_precision"] == pytest.approx(0.5)
    assert scores["pristine_relabel_rate"] == pytest.approx(1 / 4)
    assert denoise.detection_metrics(np.array([3, 0, 2, 0, 0, 0, 0.0]), replaced, signal)["roc_auc"] == 1.0


def test_cell_context_recovers_corrupted_labels_and_the_prior_never_relabels():
    from mantpy_repro.lung.final import score_fold

    cohort = structured_cohort()
    cache = graphs.GraphCache()
    config = C.Config(context="cell", cleaning=C.CleaningRecipe(anomaly_stat="raw_neglogp", threshold=1.0,
                                                                gate_radius_px=0.0, gate_min_neighbours=0))
    scores, classes, cleaned, fitted = score_fold(cohort, config, ROIS[:3], ROIS[3:], cache, seeds=(0, 1))
    assert len(scores) == 3 * 2 * 3 and set(cleaned) == set(ROIS[3:])
    mixed = scores[scores.kind == "mixed"]
    assert mixed.roc_auc.mean() > 0.9 and mixed.accuracy_after.mean() > mixed.accuracy_before.mean()
    assert mixed.prior_adjusted_roc_auc.notna().all() and fitted["temperature"] == 1.0
    assert (classes.n_restored <= classes.n_corrupted).all() and classes.n_corrupted.sum() == scores.n_artifact.sum()
    prior, _, _, _ = score_fold(cohort, C.Config(context="prior"), ROIS[:3], ROIS[3:], cache, seeds=(0,))
    assert (prior.n_relabelled == 0).all() and prior.prior_adjusted_roc_auc.isna().all()
    assert np.allclose(prior.accuracy_after, prior.accuracy_before)
    realisations = {seed: corruption.make_realisation(cohort, "mixed", seed) for seed in C.TUNING_SEEDS}
    assert 0.9 < search.inner_accuracy(config, cohort, ROIS[:3], cache, realisations) <= 1.0


# --------------------------------------------------------------------------------- association
def test_association_matches_direct_edge_counts():
    cohort = structured_cohort(n_cells=80)
    cache = graphs.GraphCache()
    pooled, per_roi = describe.association(cohort, cache, C.GraphRecipe())
    observed, expected = np.zeros((3, 3)), np.zeros((3, 3))
    for roi in cohort.rois.values():
        edges = cache.get(roi, C.GraphRecipe()).ce_d.tocoo()
        keep = roi.label[edges.col] >= 0                                   # background patches are not counted
        types, clusters = roi.cell_type_idx[edges.row[keep]], roi.label[edges.col[keep]]
        np.add.at(observed, (types, clusters), 1)
        expected += np.bincount(types, minlength=3)[:, None] * np.bincount(clusters, minlength=3)[None, :] / keep.sum()
    table = pooled.set_index(["cell_type", "cluster"])
    for t in range(3):
        for k in range(3):
            assert table.loc[(f"type{t}", k), "n_obs"] == observed[t, k]
            assert table.loc[(f"type{t}", k), "n_exp"] == pytest.approx(expected[t, k])
            assert table.loc[(f"type{t}", k), "log2_enr"] == pytest.approx(np.log2(observed[t, k] / expected[t, k]))
    assert table.loc[("type1", 1), "log2_enr"] > 1 > 0 > table.loc[("type1", 0), "log2_enr"]
    assert len(per_roi) == 6 * 3 * 3
    contrast, contacts = describe.marker_contrast(cohort, cache, C.GraphRecipe(), cell_type="type1", cluster=1)
    roi = cohort.rois["roi_001"]
    touched = np.asarray(cache.get(roi, C.GraphRecipe()).ce_d[roi.cell_type == "type1"].sum(axis=0)).ravel() > 0
    assert np.array_equal(contacts["roi_001"], touched)
    wall = roi.label == 1
    assert contrast.loc["roi_001"].to_numpy() == pytest.approx(
        roi.patch_feat[wall & touched].mean(0) - roi.patch_feat[wall & ~touched].mean(0))


# -------------------------------------------------------------------------------- refit control
def test_target_definition_selects_k_and_orders_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0.2, 0.2], [3.0, 0.5], [0.5, 6.0]])               # background and two signal clusters
    raw = {}
    for name in ROIS:
        component = rng.integers(0, 3, size=300)
        means = np.abs(centres[component] + rng.normal(scale=0.15, size=(300, 2)))
        means[0] = 0.0                                                     # a tile with no raw signal
        raw[name] = {"means": means, "pixel_sum": means.sum(0) * 100, "pixel_sumsq": (means ** 2).sum(0) * 100 + 300.0,
                     "n_pixels": 30000, "component": component}
    labels, record = refit.define_targets(raw, ROIS[:5], "training_cohort_z")
    assert record["signal_k"] == 2 and record["training_rois"] == ROIS[:5]
    held_out = raw["roi_006"]
    assert labels["roi_006"][0] == -1
    assert np.array_equal(labels["roi_006"][1:], held_out["component"][1:] - 1)     # ordered by mean centre
    fixed, replay = refit.define_targets(raw, ROIS[:5], "training_cohort_z", signal_k=2)
    assert np.array_equal(fixed["roi_006"], labels["roi_006"]) and np.allclose(replay["centres"], record["centres"])
    mean, scale = refit.pixel_moments(raw, ROIS[:5])
    assert np.allclose(record["moments"]["roi_006"]["mean"], mean) and (scale > 0).all()
    own = refit.define_targets(raw, ROIS[:5], "per_roi_z", signal_k=2)[1]
    assert np.allclose(own["moments"]["roi_006"]["mean"], refit.pixel_moments(raw, ["roi_006"])[0])
    assert refit.train_key(["roi_003", "roi_001"]) == "train_001_003"


# --------------------------------------------------------------------------------------- store
def test_tables_with_undefined_scores_survive_the_store(tmp_path):
    import pandas as pd

    table = pd.DataFrame({"roi": ["a", "b"], "n": [1, 2], "precision": [np.nan, 0.5], "log2": [-np.inf, 1.0]})
    store, calls = Store(tmp_path), []

    def fit():
        calls.append(1)
        return pack(table, "scores__"), {}

    unit = store.unit("final", "a", request={"seed": 0})
    store.compute(unit, fit, label="t", log=lambda message: None)
    again = store.compute(store.unit("final", "a", request={"seed": 0}), fit, label="t", log=lambda message: None)
    assert len(calls) == 1
    restored = unpack(again, "scores__")
    assert list(restored.columns) == list(table.columns) and restored.roi.tolist() == ["a", "b"]
    assert np.isnan(restored.precision[0]) and restored.log2[0] == -np.inf and restored.n.tolist() == [1, 2]
    with pytest.raises(RuntimeError):
        store.compute(store.unit("final", "a", request={"seed": 1}), fit, label="t")


def test_smoke_and_full_results_cannot_mix(tmp_path):
    Context(tmp_path, partition="fresh", smoke=True)
    with pytest.raises(RuntimeError):
        Context(tmp_path, partition="fresh")
    assert Budget.of(False).trials == C.TRIALS and Budget.of(False).evaluation_seeds == C.EVALUATION_SEEDS
    assert Budget.of(True).trials == {"mouse": 3, "roi": 3, "refit": 3}


# ----------------------------------------------------------------------------- paper partition
def paper_like_prepared(seed: int = 0) -> data.Prepared:
    """The stored arrays of a ``prepare`` run whose tiles are the paper's 69 x 66 grid, with the
    paper's partition as the fresh one, the paper's component scan, random marker values and no cells."""
    rng = np.random.default_rng(seed)
    record, reference = data.paper_partition(), report.paper_reference()
    rows, columns = record["grid_shape"]
    native = np.arange(rows * columns)
    xy = np.c_[native % columns * PATCH + PATCH / 2, native // columns * PATCH + PATCH / 2].astype(float)
    scan = reference["component_scan"]
    arrays = {
        "roi_names": np.asarray(ROIS), "mouse": np.asarray(["mouse_01"] * 3 + ["mouse_02"] * 3),
        "cell_types": np.asarray(["AEC"]), "ecm_names": np.asarray(list(reference["marker_names"])),
        "patch_size": np.asarray(PATCH), "grid_shape": np.asarray([[rows, columns]] * 6),
        "tile_roi": np.repeat(np.arange(6), rows * columns), "tile_native": np.tile(native, 6),
        "tile_xy": np.tile(xy, (6, 1)), "tile_feat": rng.normal(size=(6 * rows * columns, 10)),
        "tile_fresh_label": np.concatenate([data.decode_partition(record["rois"][name]) for name in ROIS]),
        "cell_roi": np.zeros(0, int), "cell_id": np.zeros(0, str), "cell_xy": np.zeros((0, 2)), "cell_type": np.zeros(0, str),
        "scan_signal_components": np.asarray([entry["signal_components"] for entry in scan]),
        "scan_silhouette": np.asarray([entry["silhouette"] for entry in scan]),
        "scan_cross_seed_ari": np.asarray([entry["cross_seed_ari"] for entry in scan]),
        "scan_background_patches": np.asarray([entry["background_patches"] for entry in scan]),
        "selected_total_components": np.asarray(4),
    }
    return data.Prepared(arrays)


def test_paper_partition_has_the_papers_counts():
    prepared = paper_like_prepared()
    assert prepared.has_paper_grid()
    cohort = prepared.cohort("paper")
    reference = report.paper_reference()["partition"]
    labels = np.concatenate([roi.label for roi in cohort.rois.values()])
    assert len(labels) == reference["n_patches"] == 27318 and (labels >= 0).sum() == reference["n_signal"] == 11435
    assert np.bincount(labels[labels >= 0]).tolist() == reference["cluster_sizes"] == [8640, 1479, 1316]
    assert [len(roi.label) for roi in cohort.rois.values()] == [4553] * 6       # one tile per ROI is not a patch
    assert cohort.n_labels == 3 and cohort.rois["roi_001"].patch_id[-1] == "roi_001:4552"
    assert prepared.summary()["patches_labelled_differently"] == 0
    assert cohort.label_digest() == prepared.cohort("fresh").label_digest()


def test_corruption_generator_reproduces_the_example_of_fig_4h(tmp_path):
    prepared = paper_like_prepared()
    cohort = prepared.cohort("paper")
    example = report.paper_reference()["example"]
    realisation = corruption.make_realisation(cohort, "mixed", 0)
    assert [int(made.is_artifact.sum()) for made in realisation.values()] == [227, 241, 247, 150, 111, 167]
    made = realisation[example["roi"]]
    assert np.array_equal(made.corrupted, data.decode_partition(example["observed"]))
    assert len(made.blocks) == 1 and made.blocks[0]["new_label"] == example["block"]["new_label"]
    assert made.blocks[0]["n_flipped"] == example["block"]["n_replaced"] == 75
    assert made.blocks[0]["centre"] == pytest.approx(example["block"]["centre"])

    ctx = SimpleNamespace(prepared=lambda: prepared, cohort=lambda: cohort, partition="paper",
                          store=SimpleNamespace(root=tmp_path))
    rows = report.Comparison()
    report.compare_example(rows, ctx, "paper", report.paper_reference())
    report.compare_partition(rows, ctx, report.paper_reference())
    table = rows.table()
    profiles = table.quantity.str.startswith("mean ")                        # the marker values here are random
    assert profiles.sum() == 30 and table[~profiles].within_tolerance.all() and len(table) == 4 + 41 + 18


# -------------------------------------------------------------------------------------- report
def _minus_inf(value):
    return -np.inf if value is None else value


def test_report_accepts_the_papers_numbers_and_flags_a_shifted_one():
    import pandas as pd

    reference = report.paper_reference()
    counts = pd.DataFrame([{"roi": name, **entry, "cell_ecm_edges_signal": 0} for name, entry in reference["graph_counts"].items()])
    counts.loc[0, "cell_ecm_edges_signal"] = reference["n_signal_cell_ecm_edges"]
    tables = {
        "graph_counts": counts,
        "association": pd.DataFrame([{"cell_type": e["cell_type"], "cluster": e["cluster"], "n_obs": e["observed"],
                                      "n_exp": e["expected"], "log2_enr": _minus_inf(e["log2_obs_exp"])}
                                     for e in reference["association"]]),
        "association_per_roi": pd.DataFrame([{"roi": e["roi"], "cell_type": "AEC", "cluster": e["cluster"], "n_obs": e["observed"],
                                              "log2_enr": _minus_inf(e["log2_obs_exp"])}
                                             for e in reference["aec_association_per_roi"]]),
        "aec_marker_contrast": pd.DataFrame(reference["aec_marker_contrast"]["per_roi"]).T.rename_axis("roi").reset_index(),
        "edge_rule_sensitivity": pd.DataFrame([{"recipe_index": e["recipe_index"], "cluster": k, "log2_enr": _minus_inf(value),
                                                "n_cell_ecm_edges": e["n_cell_ecm_edges"]}
                                               for e in reference["edge_rules"] for k, value in enumerate(e["log2_obs_exp"])]),
    }
    rows = report.Comparison()
    report.compare_description(rows, tables, reference)
    described = rows.table()
    assert len(described) == 13 + 36 * 3 + 18 * 2 + 10 + 60 + 49 * 4 and described.within_tolerance.all()

    text = {"recovery": [0.52, 0.78, 0.70, 0.75, 0.75, 0.70], "pristine_relabel_rate": [0.021] * 6,
            "accuracy_before": [0.90] * 6, "accuracy_after": [0.95] * 6}
    per_roi = pd.DataFrame([{"scheme": "mouse", "kind": "mixed", "context": e["context"], "roi": e["roi"],
                             "roc_auc": e["roc_auc"], "prior_adjusted_roc_auc": e["prior_adjusted_roc_auc"],
                             **{metric: values[ROIS.index(e["roi"])] for metric, values in text.items()}}
                            for e in reference["detection_per_roi"]])
    single = pd.DataFrame([{"scheme": "roi", "kind": "mixed", "context": "cell", "roi": name, "roc_auc": 0.98,
                            "recovery": 0.78} for name in ROIS])
    scores = pd.DataFrame([{"scheme": "mouse", "kind": "mixed", **e} for e in reference["detection_per_realisation"]])
    recovery = pd.DataFrame([{"scheme": "mouse", "kind": "mixed", "context": "cell", "roi": name, "cluster": k, "recovery": value}
                             for name in ROIS for k, value in enumerate((0.80, 0.48, 0.33))])
    refitted = pd.DataFrame([{"normalization": arm, "kind": "mixed", "context": "cell", "outer_roi": name,
                              "accuracy_after_common_signal": value}
                             for arm, value in (("training_cohort_z", 0.970), ("per_roi_z", 0.965)) for name in ROIS])
    rows = report.Comparison()
    report.compare_denoising(rows, scores, pd.concat([per_roi, single], ignore_index=True), recovery, refitted, reference)
    denoised = rows.table()
    assert len(denoised) == 12 + 24 + 9 + 18 + 180 + 13 and denoised.within_tolerance.all()

    shifted = per_roi.copy()
    shifted.loc[(shifted.context == "cell") & (shifted.roi == "roi_005"), "roc_auc"] -= 0.05
    rows = report.Comparison()
    report.compare_denoising(rows, scores, shifted, recovery, refitted, reference)
    flagged = rows.table()
    outside = flagged[~flagged.within_tolerance]
    assert set(outside.quantity) == {"roc_auc, cell context, roi_005", "roc_auc, cell context, min over ROIs"}

    rows = report.Comparison()
    rows.add("s", "count", 11450, 11435, exact=True)
    rows.add("s", "rounded in the text", 0.7049, 0.70, rounding=0.005)
    rows.add("s", "blank in the paper", -np.inf, None)
    assert rows.table().within_tolerance.tolist() == [False, True, True]


# ------------------------------------------------------------------------------- whole pipeline
def test_every_stage_runs_and_resumes_on_synthetic_rois(tmp_path, monkeypatch, capsys):
    pytest.importorskip("optuna")
    from mantpy_repro.lung import cli

    monkeypatch.setattr(data, "load_bundle", lambda cache_dir=None: synthetic_bundle())
    arguments = ["--results", str(tmp_path), "--partition", "fresh", "--smoke"]
    assert cli.main(["all", *arguments]) == 0
    prepared = Context(tmp_path, partition="fresh", smoke=True).prepared()
    cohort = prepared.cohort("fresh")
    first = cohort.rois["roi_001"]
    assert first.patch_xy[0].tolist() == [5.0, 5.0] and first.patch_xy[1].tolist() == [15.0, 5.0]   # row-major centres
    assert len(prepared.geometry().rois["roi_001"].label) == 256 and len(first.label) == 255     # one tile dropped per ROI
    assert cohort.n_labels >= 2 and cohort.mice() == C.paper_selection()["mice"]
    selection = json.loads((tmp_path / "selection" / "search_fresh.json").read_text())
    assert set(selection["mouse"]) == {"mouse_01", "mouse_02"} and set(selection["roi"]) == {"roi_001"}
    assert selection["refit_control"]["target_definition"]["per_roi_z"]["roi_001"]["training_rois"] == ROIS[1:]
    for name in ("component_scan", "partition", "graph_counts", "association", "aec_marker_contrast",
                 "edge_rule_sensitivity", "selected_configurations", "denoising_scores", "denoising_summary",
                 "class_recovery", "refit_control_summary"):
        assert (tmp_path / "report" / f"{name}.csv").exists(), name
    done = sorted(str(path) for path in tmp_path.rglob("completion.json"))
    stamps = {path: (tmp_path / path).stat().st_mtime_ns for path in done}
    capsys.readouterr()
    assert cli.main(["all", *arguments]) == 0                              # everything is complete: nothing is refitted
    assert "done " not in capsys.readouterr().out
    assert stamps == {path: (tmp_path / path).stat().st_mtime_ns for path in done}
    with pytest.raises(RuntimeError):
        cli.main(["final", "--results", str(tmp_path), "--partition", "fresh"])                # full run in a smoke directory


# --------------------------------------------------------------- against the paper (real data)
@pytest.mark.gpu
def test_mouse_held_out_cell_context_reproduces_the_paper(tmp_path):
    """Refits the two mouse folds of the cell context on the lung dataset (CPU, about a minute)."""
    from mantpy_repro.lung.final import FinalRunner

    ctx = Context(tmp_path)
    reference = report.paper_reference()
    assert ctx.prepared().summary()["paper"]["signal"] == reference["partition"]["n_signal"]
    runner = FinalRunner(ctx, "paper")
    scores = []
    for fold in ("mouse_01", "mouse_02"):
        arrays = runner.run({"experiment": "mouse", "fold": fold, "context": "cell"})
        scores.append(unpack(arrays, "scores__"))
    import pandas as pd

    mixed = pd.concat(scores).query("kind == 'mixed'").groupby("roi")[["roc_auc", "prior_adjusted_roc_auc"]].mean()
    for entry in reference["detection_per_roi"]:
        if entry["context"] == "cell":
            assert mixed.loc[entry["roi"], "roc_auc"] == pytest.approx(entry["roc_auc"], abs=0.005)
            assert mixed.loc[entry["roi"], "prior_adjusted_roc_auc"] == pytest.approx(entry["prior_adjusted_roc_auc"], abs=0.005)

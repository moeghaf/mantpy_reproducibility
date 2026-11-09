"""Fast checks of the intestine pipeline's logic, and one GPU check against the paper."""

import numpy as np
import pytest

from mantpy_repro.intestine import config as C
from mantpy_repro.intestine.data import build_support
from mantpy_repro.intestine.final import K_VALUES, final_tasks, label_free_k
from mantpy_repro.intestine.report import paper_reference
from mantpy_repro.intestine.scoring import PixelScorer, pairwise_ari
from mantpy_repro.store import Store


# ------------------------------------------------------------------------------ configurations
@pytest.mark.parametrize("image", C.IMAGES)
def test_paper_configuration_has_the_papers_identity(image):
    selected = C.paper_selection()["images"][image]
    assert C.original_config_id(image, selected["joint"]) == selected["joint_id"]
    assert C.config_id(selected["joint"]) == paper_reference()["images"][image]["joint_config_id"]


def test_search_space_has_28_settings_and_the_baseline_is_valid():
    assert len(C.SPACE) == 28
    assert C.validate(C.BASELINE) == C.BASELINE


def test_gin_drops_edge_features_and_gine_requires_them():
    config = C.validate({**C.BASELINE, "edge_use": "gin"})
    assert (config["edge_distance"], config["edge_orientation"], config["edge_continuity"]) == ("none", False, False)
    with pytest.raises(ValueError):
        C.validate({**C.BASELINE, "edge_distance": "none", "edge_orientation": False})


def test_repair_maps_sampled_points_onto_the_valid_space():
    config, notes = C.repair({**C.BASELINE, "patch_size": 16, "neighbours": 80, "cnn_readout": "flatten"})
    assert config["neighbours"] == 40 and config["cnn_readout"] == "avgpool2" and len(notes) == 2
    config, notes = C.repair({**C.BASELINE, "edge_distance": "none", "edge_orientation": False})
    assert config["edge_use"] == "gin"


def test_cnn_identity_ignores_graph_settings():
    other = {**C.BASELINE, "neighbours": 120, "gm_depth": 4}
    assert C.cnn_id(other) == C.cnn_id(C.BASELINE)
    assert C.config_id(other) != C.config_id(C.BASELINE)
    assert C.has_baseline_cnn(other)


# ------------------------------------------------------------------------------------- support
def test_common_anchors_and_coverage():
    # 128 x 128 image. 16-px nodes fill the left half; the 64-px node covers only the top-left tile.
    gy16, gx16 = np.meshgrid(np.arange(8), np.arange(4), indexing="ij")
    grids = {16: (gy16.ravel(), gx16.ravel()),
             32: (np.repeat(np.arange(4), 2), np.tile(np.arange(2), 4)),
             64: (np.array([0]), np.array([0]))}
    support = build_support(grids, (128, 128))
    assert support.coverage[16] == 1.0 and support.coverage[32] == 1.0 and support.coverage[64] == 0.5
    assert len(support.common) == 16                         # 16-px nodes inside the 64-px node
    labels32 = np.arange(8)
    projected = support.project(labels32, 32)
    assert set(projected) == {0, 1, 2, 3}                    # the four 32-px patches inside that tile


# ------------------------------------------------------------------------------------- scoring
def test_pixel_scorer_matches_sklearn_on_expanded_patches():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    annotation = np.zeros((64, 64), dtype=np.uint8)
    annotation[:, :32], annotation[:, 32:] = 1, 2
    annotation[:8] = 0                                       # unannotated strip
    gy, gx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    scorer = PixelScorer(gy.ravel(), gx.ravel(), 16, annotation, k=2)
    labels = rng.integers(0, 2, size=16)
    truth, predicted = [], []
    for (y, x), label in zip(scorer.coordinates, labels):
        tile = annotation[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16].ravel()
        truth.append(tile[tile > 0])
        predicted.append(np.full((tile > 0).sum(), label))
    expected = adjusted_rand_score(np.concatenate(truth), np.concatenate(predicted))
    assert scorer.score(labels)["ARI"] == pytest.approx(expected)
    assert scorer.coverage == 1.0


def test_pairwise_ari_is_permutation_invariant():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert pairwise_ari(a, np.array([5, 5, 3, 3, 9, 9])) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------- store
def test_store_resumes_and_rejects_changed_settings(tmp_path):
    store, calls = Store(tmp_path), []

    def fit():
        calls.append(1)
        return {"x": np.arange(3.0)}, {"note": "ok"}

    unit = store.unit("runs", "a", request={"seed": 0})
    first = store.compute(unit, fit, label="t", log=lambda message: None)
    second = store.compute(store.unit("runs", "a", request={"seed": 0}), fit, label="t", log=lambda message: None)
    assert len(calls) == 1 and np.array_equal(first["x"], second["x"])
    with pytest.raises(RuntimeError):
        store.compute(store.unit("runs", "a", request={"seed": 1}), fit, label="t")
    store.compute(unit, lambda: ({"x": np.arange(3.0), "embedding": np.ones(2)}, {}), label="t",
                  log=lambda message: None, require=("embedding",))
    assert "embedding" in unit.load()


def test_store_refuses_non_finite_results(tmp_path):
    unit = Store(tmp_path).unit("runs", "b", request={})
    with pytest.raises(ValueError):
        unit.save(0.0, x=np.array([np.nan]))
    assert not unit.done()


# --------------------------------------------------------------------------------------- final
def test_final_tasks_and_choice_of_k():
    assert len(final_tasks("intestine1")) == 22
    stability = np.array([0.941, 0.925, 0.750, 0.930, 0.891, 0.900, 0.690])   # paper, image 1
    assert label_free_k(stability) == 2
    assert label_free_k(np.array([0.5, 0.78, 0.869, 0.839, 0.77, 0.68, 0.63])) == 4   # paper, image 2
    assert len(K_VALUES) == 7


# --------------------------------------------------------------- against the paper (GPU, slow)
@pytest.mark.gpu
def test_seed0_of_image1_reproduces_the_paper(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from mantpy_repro.intestine.core import Context

    ctx = Context(tmp_path, cnn_init="paper")
    prepared = ctx.prepared("intestine1")
    reference = paper_reference()["images"]["intestine1"]
    assert prepared.support.n_nodes[32] == reference["n_patches"]
    run = ctx.run_unit("intestine1", C.paper_selection()["images"]["intestine1"]["joint"], 0)
    scorer = prepared.scorer(32)
    assert scorer.score(run["labels_cnn"])["ARI"] == pytest.approx(reference["per_seed"]["CNN K-means"][0], abs=0.005)
    assert scorer.score(run["labels_graphmae"])["ARI"] == pytest.approx(reference["per_seed"]["GraphMAE"][0], abs=0.02)


# ------------------------------------------------------------------------- search and selection
@pytest.mark.parametrize("image", C.IMAGES)
def test_paper_search_records_are_valid_and_contain_the_selection(image):
    from mantpy_repro.intestine.search import paper_records

    records = paper_records(image)
    assert len(records) == {"intestine1": 197, "intestine2": 200}[image]
    identities = [record["config_id"] for record in records]
    assert len(set(identities)) == len(identities)
    for record in records:
        assert C.config_id(record["config"]) == record["config_id"]
    assert paper_reference()["images"][image]["joint_config_id"] in identities


@pytest.mark.parametrize("image", C.IMAGES)
def test_tie_band_on_the_papers_records_contains_the_papers_selection(image):
    import pandas as pd

    from mantpy_repro.intestine.freeze import tie_band
    from mantpy_repro.intestine.search import TABLE_COLUMNS, paper_records

    rows = [{"image": image, "config_id": r["config_id"], "stability3": r["stability3"], "coverage": r["coverage"],
             "min_fraction": r["min_fraction"], "feasible": r["feasible"], "status": "complete",
             "mean_fit_seconds": 1.0, **r["config"]} for r in paper_records(image)]
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    coverage_32px = table.loc[table["patch_size"] == 32, "coverage"].iloc[0]
    band = tie_band(table, coverage_32px)
    assert paper_reference()["images"][image]["joint_config_id"] in set(band["config_id"])
    assert (band["patch_size"] <= 32).all() and band["feasible"].all()
    # configurations survive the round trip through the table
    from mantpy_repro.intestine.freeze import _plain

    for row in band.to_dict("records"):
        assert C.config_id({axis: _plain(row[axis]) for axis in C.SPACE}) == row["config_id"]


def test_choose_requires_five_seed_feasibility_and_breaks_exact_ties_by_time():
    from mantpy_repro.intestine.freeze import choose

    records = [
        {"config_id": "a", "stability5": 0.95, "min_fraction": 0.005, "mean_fit_seconds": 10.0},   # infeasible
        {"config_id": "b", "stability5": 0.90, "min_fraction": 0.05, "mean_fit_seconds": 30.0},
        {"config_id": "c", "stability5": 0.90, "min_fraction": 0.05, "mean_fit_seconds": 20.0},
        {"config_id": "d", "stability5": 0.899, "min_fraction": 0.05, "mean_fit_seconds": 1.0},
    ]
    assert choose(records)["config_id"] == "c"
    with pytest.raises(RuntimeError):
        choose(records[:1])


def test_worker_command_replaces_workers_and_shard():
    from mantpy_repro.intestine.cli import worker_command

    command = worker_command(["search", "--workers", "4", "--results", "r", "--shard=0/1"], 2, 4)
    assert command[3:] == ["search", "--results", "r", "--workers", "1", "--shard", "2/4"]


def test_report_marks_missing_numbers_as_not_run():
    import pandas as pd

    from mantpy_repro.intestine.report import COMPARE_COLUMNS, SCORE_COLUMNS, compare_with_paper

    reference = paper_reference()["images"]["intestine1"]
    scores = pd.DataFrame([{"image": "intestine1", "method": "GraphMAE", "seed": 0,
                            "ARI": reference["per_seed"]["GraphMAE"][0] + 0.001, "macro_F1": 0.8},
                           {"image": "intestine1", "method": "CNN K-means", "seed": 0,
                            "ARI": reference["per_seed"]["CNN K-means"][0] + 0.05, "macro_F1": 0.3}], columns=SCORE_COLUMNS)
    table = compare_with_paper(scores, pd.DataFrame(), "intestine1")
    assert list(table.columns) == COMPARE_COLUMNS and len(table) == 35
    status = table.set_index(["quantity", "seed"])["status"]
    assert status["GraphMAE, ARI", 0] == "within"
    assert status["CNN K-means, ARI", 0] == "outside"
    assert (table["status"] == "not run").sum() == 33

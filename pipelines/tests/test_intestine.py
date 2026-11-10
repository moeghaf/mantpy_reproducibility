"""Fast checks of the intestine pipeline's logic, and one GPU check against the paper."""

import numpy as np
import pytest

from mantpy_repro.intestine import config as C
from mantpy_repro.intestine.data import build_support
from mantpy_repro.intestine.final import K_VALUES, final_tasks, label_free_k
from mantpy_repro.intestine.report import paper_reference
from mantpy_repro.intestine.scoring import PixelScorer, pairwise_ari
from mantpy_repro.store import Store


# Configurations
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


# Support
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


# Scoring
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


# Store
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


# Final
def test_final_tasks_and_choice_of_k():
    assert len(final_tasks("intestine1")) == 22
    stability = np.array([0.941, 0.925, 0.750, 0.930, 0.891, 0.900, 0.690])   # paper, image 1
    assert label_free_k(stability) == 2
    assert label_free_k(np.array([0.492, 0.780, 0.869, 0.839, 0.770, 0.680, 0.630])) == 4   # paper, image 2
    # K = 5 is the most stable, K = 3 is within 0.01 of it: the smaller K is chosen
    assert label_free_k(np.array([0.90, 0.925, 0.75, 0.93, 0.891, 0.90, 0.69])) == 3
    assert len(K_VALUES) == 7


# Against the paper (GPU, slow)
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
    # The paper's GPU was an RTX 4070 SUPER. On an L40S and an H100 this seed differed from the paper
    # by up to 0.015 (CNN K-means) and 0.036 (GraphMAE), so the limits are twice those differences.
    assert scorer.score(run["labels_cnn"])["ARI"] == pytest.approx(reference["per_seed"]["CNN K-means"][0], abs=0.03)
    assert scorer.score(run["labels_graphmae"])["ARI"] == pytest.approx(reference["per_seed"]["GraphMAE"][0], abs=0.07)


# Search and selection
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
    # a smoke run selects without the cluster-size condition
    assert choose(records, require_feasible=False)["config_id"] == "a"


def test_selection_needs_a_finished_search(tmp_path):
    from mantpy_repro.intestine.core import Context
    from mantpy_repro.intestine.freeze import require_finished
    from mantpy_repro.intestine.search import missing_evaluations, paper_records
    from mantpy_repro.store import write_json

    ctx = Context(tmp_path, device="cpu")
    records = paper_records("intestine1")
    for record in records[:2]:
        write_json(tmp_path / "search" / "intestine1" / "evaluations" / f"{record['config_id']}.json",
                   {"config_id": record["config_id"], "config": record["config"], "status": "complete"})
    assert missing_evaluations(ctx, "intestine1") == [record["config_id"] for record in records[2:]]
    with pytest.raises(RuntimeError, match="195 of the paper's configurations have no evaluation"):
        require_finished(ctx, "intestine1")
    require_finished(ctx, "intestine1", "paper", 2)          # --smoke refits only the first configurations
    with pytest.raises(RuntimeError, match="2 of 3 configurations fitted"):
        require_finished(ctx, "intestine1", "nsga", 3)
    require_finished(ctx, "intestine1", "nsga", 2)


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


def _write_marker(folder):
    from mantpy_repro.store import write_json

    write_json(folder / "completion.json", {})


def test_report_names_final_tasks_without_a_result(tmp_path):
    from types import SimpleNamespace

    from mantpy_repro.intestine.report import unfinished_tasks

    ctx = SimpleNamespace(store=SimpleNamespace(root=tmp_path))
    identity = paper_reference()["images"]["intestine1"]["joint_config_id"]
    assert len(unfinished_tasks(ctx, "intestine1")) == 22
    _write_marker(tmp_path / "runs" / "intestine1" / identity / "full" / "seed_0")
    _write_marker(tmp_path / "runs" / "intestine1" / identity / "encoder_gin" / "seed_4")
    _write_marker(tmp_path / "baselines" / "intestine1" / identity)
    names = unfinished_tasks(ctx, "intestine1")
    assert len(names) == 19 and "intestine1 k_sweep" in names and "intestine1 gin seed 0" in names
    assert not {"intestine1 full seed 0", "intestine1 gin seed 4", "intestine1 baselines"} & set(names)


def test_report_returns_1_when_final_tasks_have_no_result(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import pandas as pd

    from mantpy_repro.intestine import report

    def no_scores(ctx, image, selection):
        scores = pd.DataFrame(columns=report.SCORE_COLUMNS)
        scores.attrs.update(config_id=paper_reference()["images"][image]["joint_config_id"])
        return scores

    monkeypatch.setattr(report, "score_image", no_scores)
    monkeypatch.setattr(report, "k_table", lambda ctx, image, selection: pd.DataFrame())
    ctx = SimpleNamespace(store=SimpleNamespace(root=tmp_path), smoke=False)
    assert report.write_report(ctx, ["intestine1"]) == 1
    assert "Final tasks without a result (22)" in capsys.readouterr().out
    compare = pd.read_csv(tmp_path / "report" / "comparison_with_paper.csv")
    assert len(compare) == 35 and (compare["status"] == "not run").all()
    # a smoke run is not compared with the paper, but missing tasks still give exit code 1
    ctx.smoke = True
    assert report.write_report(ctx, ["intestine1"]) == 1
    assert not (tmp_path / "report" / "comparison_with_paper.csv").exists()


# Search records and the CLI
def test_search_fits_an_out_of_memory_record_again(tmp_path):
    from types import SimpleNamespace

    from mantpy_repro.intestine.search import evaluate
    from mantpy_repro.store import read_json

    config, calls = C.validate(C.BASELINE), []
    path = tmp_path / "search" / "intestine1" / "evaluations" / f"{C.config_id(config)}.json"

    def out_of_memory(image, config, seed):
        raise RuntimeError("CUDA out of memory")

    def run_unit(image, config, seed):
        calls.append(seed)
        return {"common_graphmae": np.repeat(np.arange(5), 20)}

    ctx = SimpleNamespace(
        store=SimpleNamespace(root=tmp_path), run_unit=out_of_memory,
        prepared=lambda image: SimpleNamespace(k=5, support=SimpleNamespace(coverage={32: 0.9})),
        unit_metadata=lambda image, config, seed: {"metadata": {"graph_seconds": 2.0}})
    record = evaluate(ctx, "intestine1", config)
    assert record["status"] == "out of memory" and record["feasible"] is False
    assert read_json(path)["status"] == "out of memory"
    ctx.run_unit = run_unit                                  # a GPU with more memory
    record = evaluate(ctx, "intestine1", config)
    assert record["status"] == "complete" and record["stability3"] == 1.0 and calls == [0, 1, 2]
    assert evaluate(ctx, "intestine1", config) == read_json(path)
    assert calls == [0, 1, 2]                                # a complete record is not fitted again


@pytest.mark.parametrize("arguments", [
    ["final", "--image", "intestine1", "--task-index", "22"],
    ["final", "--task-index", "0"],                          # two images
    ["search", "--shard", "4/4"],
    ["search", "--shard", "0/0"],
    ["prepare", "--workers", "2"],
])
def test_command_line_refuses_invalid_arguments(arguments):
    from mantpy_repro.intestine.cli import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert isinstance(error.value.code, str)                 # a message, not a traceback or a usage error


def test_tasks_stage_prints_the_range_of_task_index(capsys):
    from mantpy_repro.intestine.cli import main

    assert main(["tasks", "--image", "intestine1"]) == 0
    assert "--task-index 0..21" in capsys.readouterr().out


def test_search_time_limit_is_one_limit_for_all_images(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mantpy_repro.intestine import cli, search
    from mantpy_repro.runtime import UNFINISHED

    clock, limits = [0.0], []

    def run_paper_search(ctx, image, *, shard, time_limit_hours, limit):
        limits.append(time_limit_hours)
        clock[0] += 3600                                     # each image takes one hour
        return time_limit_hours > 1

    monkeypatch.setattr(cli, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(search, "run_paper_search", run_paper_search)
    status = cli.main(["search", "--results", str(tmp_path), "--device", "cpu", "--time-limit-hours", "1.5"])
    assert limits == pytest.approx([1.5, 0.5])
    assert status == UNFINISHED                              # the second image did not finish


def test_results_directory_keeps_its_mode(tmp_path):
    from mantpy_repro.intestine.core import Context

    Context(tmp_path, device="cpu")
    Context(tmp_path, device="cpu")
    with pytest.raises(RuntimeError, match="holds full results"):
        Context(tmp_path, device="cpu", smoke=True)

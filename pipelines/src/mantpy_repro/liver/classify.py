"""Infected-versus-naive classification from ECM cluster graphs (Fig. 3g, h; Supplementary Table 5).

Every fit holds out one mouse: feature scaling and class weights come from the seven training
mice, and the held-out mouse's ROIs are predicted once. A mouse's probability of infection is the
mean of its ROI probabilities, and an ensemble is the mean over five training seeds.

Units of work (one folder each, skipped when complete):

- one classifier for one labelling of the mice, held-out mouse and seed;
- the composition-only logistic regression (all eight folds; deterministic).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..runtime import environment
from . import models
from .config import CLASSES, CLASSIFIER_SEEDS, IG_STEPS, LR_C_GRID, MAX_DIRECTED_EDGES, N_INFECTED, paper_selection
from .core import Context, Partition, log
from .descriptive import delaunay_edges


def fixed_classifier() -> dict:
    """The classifier settings of the paper (not tuned on these data)."""
    return dict(paper_selection()["fixed_classifier"])


# Graphs
def patch_edges(xy: np.ndarray, graph_type: str) -> np.ndarray:
    """Undirected edges (E x 2) over all patch centres of one ROI.

    ``delaunay``: edges of the Delaunay triangulation; ``radius<r>``: all pairs within r pixels;
    ``grid4`` / ``grid8``: lattice neighbours.
    """
    if graph_type == "delaunay":
        return delaunay_edges(xy)
    if graph_type.startswith("radius"):
        from scipy.spatial import cKDTree

        return cKDTree(xy).query_pairs(float(graph_type[6:]), output_type="ndarray").astype(np.int64)
    if graph_type.startswith("grid"):
        from mantpy._core._edges import grid_edges

        pairs = {(min(i, j), max(i, j)) for i, j, _ in grid_edges(xy, connectivity=int(graph_type[4:]))}
        return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    raise ValueError(graph_type)


def estimated_directed_edges(graph_type: str, n_signal: int, patch_size: int) -> float:
    """Rough size of a graph over the signal patches, used to refuse graphs that would not fit in memory."""
    if graph_type == "delaunay":
        return 6 * n_signal
    if graph_type.startswith("grid"):
        return int(graph_type[4:]) * n_signal
    return np.pi * (float(graph_type[6:]) / patch_size) ** 2 * n_signal


def _signal_edges(ctx: Context, partition: Partition, graph_type: str) -> dict[str, np.ndarray]:
    """Directed edges among the signal patches of every ROI, converted by Mantpy and cached on disk.

    Edges are built over all patches of an ROI and background patches are then dropped, which
    leaves the subgraph induced on the ECM signal patches with every edge in both directions.
    """
    estimate = estimated_directed_edges(graph_type, partition.n_signal, partition.patch_size)
    if estimate > MAX_DIRECTED_EDGES:
        raise MemoryError(f"{graph_type}: about {estimate:.2e} directed edges exceed the limit of {MAX_DIRECTED_EDGES:.1e}.")
    unit = ctx.store.unit("graphs", partition.folder, graph_type,
                          request={"partition": partition.identity, "graph": graph_type})

    def build():
        import anndata as ad
        import mantpy as mt
        import networkx as nx
        import pandas as pd

        arrays = {}
        for roi, sub in partition.nodes.groupby("roi", sort=True):
            graph = nx.Graph()
            graph.add_nodes_from((i, {"cluster": int(c), "x": float(x), "y": float(y)})
                                 for i, (c, x, y) in enumerate(zip(sub.cluster, sub.x, sub.y)))
            graph.add_edges_from(map(tuple, patch_edges(sub[["x", "y"]].to_numpy(float), graph_type).tolist()))
            adata = ad.AnnData(np.empty((0, 1), dtype=np.float32))
            adata.uns["graph"] = graph
            bundle = mt.nn.build_ohe_cluster_graphs(
                {roi: adata}, pd.DataFrame({"condition": [sub.condition.iloc[0]]}, index=[roi]),
                group_col="condition", group_order=CLASSES, graph_key="graph", cluster_attr="cluster")
            converted = bundle.graphs[0]
            if not np.array_equal(converted.cluster.numpy(), sub.cluster.to_numpy()[sub.cluster.to_numpy() >= 0]):
                raise RuntimeError(f"{roi}: node order changed during graph conversion.")
            arrays[roi] = converted.edge_index.numpy().astype(np.int32)
        return arrays, {"directed_edges": int(sum(a.shape[1] for a in arrays.values()))}

    return ctx.store.compute(unit, build, label=f"graphs {graph_type}", log=log)


def build_graphs(ctx: Context, partition: Partition, graph_type: str) -> list:
    """One PyG graph per ROI (sorted ROI order): one-hot cluster label per signal patch and the graph edges."""
    import torch
    from torch_geometric.data import Data

    edges = _signal_edges(ctx, partition, graph_type)
    graphs = []
    for roi, sub in partition.nodes.groupby("roi", sort=True):
        sub = sub[sub.cluster >= 0]
        cluster = sub.cluster.to_numpy()
        one_hot = np.zeros((len(cluster), partition.k), dtype=np.float32)
        one_hot[np.arange(len(cluster)), cluster] = 1.0
        graph = Data(x=torch.from_numpy(one_hot), edge_index=torch.from_numpy(edges[roi].astype(np.int64)),
                     pos=torch.from_numpy(sub[["x", "y"]].to_numpy(np.float32)))
        graph.cluster = torch.from_numpy(cluster).long()
        graph.roi_id = roi
        graphs.append(graph)
    return graphs


# Labellings
def mouse_labels(ctx: Context) -> dict[str, int]:
    """True label of every mouse (1 = infected), in sorted mouse order."""
    table = ctx.roi_table()
    return {mouse: int(sub.infected.iloc[0]) for mouse, sub in table.groupby("mouse", sort=True)}


def allocations(mice: list[str]) -> list[tuple[str, ...]]:
    """Every way of calling four of the mice infected, in lexical order."""
    return list(combinations(sorted(mice), N_INFECTED))


def observed_allocation(ctx: Context) -> int:
    labels = mouse_labels(ctx)
    return allocations(list(labels)).index(tuple(mouse for mouse, label in labels.items() if label))


def genotype_preserving(allocation: tuple[str, ...], genotype: dict[str, str]) -> bool:
    """True when the labelling keeps two infected mice in each genotype."""
    return all(sum(genotype[mouse] == g for mouse in allocation) == N_INFECTED // 2 for g in sorted(set(genotype.values())))


# One fit
def fold_fit(ctx: Context, partition: Partition, *, parts: tuple[str, ...], config: dict, objective: str, held_mouse: str,
             seed: int, infected: tuple[str, ...], explain: bool = False, device: str = "cpu") -> dict[str, np.ndarray]:
    """Train (or load) one classifier on the seven other mice and predict the held-out mouse.

    ``infected`` names the mice labelled infected (the true labelling, or one allocation of the
    permutation test). Returns ``p_infected`` (one value per held-out ROI, sorted ROI order) and,
    with ``explain``, the signed integrated gradients of every held-out ROI.
    """
    table = ctx.roi_table()
    trained = ctx.classifier_budget(config)
    request = {"partition": partition.identity, "config": config, "epochs": trained["epochs"], "objective": objective,
               "held_mouse": held_mouse, "seed": int(seed), "infected": sorted(infected), "explain": explain,
               "device": device, "threads": ctx.threads if device == "cpu" else None}
    unit = ctx.store.unit(*parts, held_mouse, f"seed_{seed}", request=request)

    def fit():
        graphs = ctx.graphs(partition, config["graph_type"])
        labels = table.mouse.isin(infected).to_numpy().astype(int)
        train, test = np.flatnonzero(table.mouse != held_mouse), np.flatnonzero(table.mouse == held_mouse)
        result = models.fit_classifier([graphs[i] for i in train], labels[train], trained, seed=seed, objective=objective,
                                       device=device, threads=ctx.threads)
        arrays = {"p_infected": np.asarray([models.roi_probabilities(result, graphs[i])[1] for i in test], dtype=np.float64),
                  "losses": np.asarray(result.history["train_loss"], dtype=np.float64)}
        if explain:
            for i in test:
                signed, residual = models.integrated_gradients(result, graphs[i], IG_STEPS)
                arrays[f"signed_{table.roi[i]}"] = signed
                arrays[f"residual_{table.roi[i]}"] = np.asarray(residual)
        return arrays, {"environment": environment()}

    label = f"{parts[0]} {' '.join(parts[2:])} held {held_mouse} seed {seed}"
    return ctx.store.compute(unit, fit, label=label, log=log)


def roi_mean_parts(partition: Partition, allocation: int) -> tuple[str, ...]:
    return ("classifier", partition.folder, "roi_mean", f"allocation_{allocation:02d}")


def fit_allocation_fold(ctx: Context, partition: Partition, allocation: int, held_mouse: str) -> None:
    """Five seeds of the fixed classifier (ROI-mean loss) for one labelling and held-out mouse."""
    infected = allocations(list(mouse_labels(ctx)))[allocation]
    for seed in CLASSIFIER_SEEDS:
        fold_fit(ctx, partition, parts=roi_mean_parts(partition, allocation), config=fixed_classifier(),
                 objective="roi_mean", held_mouse=held_mouse, seed=seed, infected=infected)


def fit_explanation_fold(ctx: Context, partition: Partition, held_mouse: str) -> None:
    """Five seeds of the fixed classifier with the loss pooled over patches, and their integrated gradients."""
    labels = mouse_labels(ctx)
    for seed in CLASSIFIER_SEEDS:
        fold_fit(ctx, partition, parts=("classifier", partition.folder, "patch_sum"), config=fixed_classifier(),
                 objective="patch_sum", held_mouse=held_mouse, seed=seed, explain=True,
                 infected=tuple(mouse for mouse, label in labels.items() if label))


# Composition-only reference
def composition_lr(ctx: Context, partition: Partition) -> dict[str, np.ndarray]:
    """Logistic regression on the cluster fractions of each ROI, with the same held-out mice.

    Per fold: fractions standardised with the training ROIs; C chosen from the grid by leaving out
    each training mouse in turn (lowest mean mouse-level log loss, ties to the smaller C); refit on
    the seven mice. Returns ``p_infected`` (per ROI, sorted order), ``chosen_c`` and ``inner_log_loss``
    (per held-out mouse in sorted order).
    """
    unit = ctx.store.unit("composition_lr", partition.folder, request={"partition": partition.identity, "grid": LR_C_GRID})

    def fit():
        from sklearn.linear_model import LogisticRegression

        table = ctx.roi_table()
        cluster = [sub.cluster.to_numpy()[sub.cluster.to_numpy() >= 0] for _, sub in partition.nodes.groupby("roi", sort=True)]
        fractions = np.stack([np.bincount(c, minlength=partition.k) / len(c) for c in cluster])
        labels, mouse_of = table.infected.to_numpy(), table.mouse.to_numpy()
        mice, truth = sorted(set(mouse_of)), mouse_labels(ctx)

        def predict(train, test, c):
            mean, spread = fractions[train].mean(0), fractions[train].std(0)
            spread[spread == 0] = 1
            model = LogisticRegression(C=c, max_iter=10000).fit((fractions[train] - mean) / spread, labels[train])
            probability = np.zeros((len(test), 2))
            probability[:, model.classes_] = model.predict_proba((fractions[test] - mean) / spread)
            return probability

        p_infected, chosen, inner = np.zeros(len(table)), [], []
        for held in mice:
            training = [mouse for mouse in mice if mouse != held]
            losses = []
            for c in LR_C_GRID:
                values = []
                for left_out in training:
                    rest = [mouse for mouse in training if mouse != left_out]
                    if len({truth[mouse] for mouse in rest}) < 2:
                        continue
                    probability = predict(np.flatnonzero(np.isin(mouse_of, rest)), np.flatnonzero(mouse_of == left_out), c)
                    values.append(-np.log(max(probability.mean(0)[truth[left_out]], 1e-12)))
                losses.append(float(np.mean(values)))
            best = min(range(len(LR_C_GRID)), key=lambda i: (losses[i], LR_C_GRID[i]))
            test = np.flatnonzero(mouse_of == held)
            p_infected[test] = predict(np.flatnonzero(mouse_of != held), test, LR_C_GRID[best])[:, 1]
            chosen.append(LR_C_GRID[best])
            inner.append(losses)
        return {"p_infected": p_infected, "chosen_c": np.asarray(chosen), "inner_log_loss": np.asarray(inner)}, {}

    return ctx.store.compute(unit, fit, label="composition logistic regression", log=log)


# Scores
def scores(probability: np.ndarray, truth: np.ndarray) -> dict:
    """Number correct at the 0.5 cut-off and the ROC-AUC of a set of probabilities."""
    from sklearn.metrics import roc_auc_score

    probability, truth = np.asarray(probability, dtype=float), np.asarray(truth, dtype=int)
    return {"n": int(len(truth)), "correct": int(((probability >= 0.5) == truth).sum()),
            "auc": float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else float("nan")}


def permutation_p(auc: np.ndarray, observed: int, preserving: np.ndarray) -> dict:
    """Exact permutation P values: the fraction of labellings whose mouse-level ROC-AUC is at least
    the observed one (the observed labelling and ties included), over all labellings and over those
    that keep two infected mice per genotype."""
    auc, preserving = np.asarray(auc, dtype=float), np.asarray(preserving, dtype=bool)
    tail = auc >= auc[observed]
    return {"observed_auc": float(auc[observed]), "n_allocations": int(len(auc)), "n_at_or_above_observed": int(tail.sum()),
            "p": float(tail.mean()), "n_genotype_allocations": int(preserving.sum()),
            "n_genotype_at_or_above_observed": int((tail & preserving).sum()), "p_genotype": float(tail[preserving].mean())}

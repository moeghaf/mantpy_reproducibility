"""Selected liver models, fold checkpoints and compact display helpers.

Mantpy supplies fitting, attribution, graph calculations and display. Local
model factories preserve selected and archived nested architectures.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time

import anndata as ad
import mantpy as mt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from liver_analysis.encoder import LiverCNN, fit_liver_cnn, seed_everything

HERE = Path(__file__).resolve().parent


def read_selected():
    return json.loads((HERE / "liver_selected.json").read_text(encoding="utf-8"))


def array_hash(value):
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    names = ("mantpy", "torch", "torch-geometric", "captum", "numpy",
             "scipy", "scikit-learn", "pandas")
    return {name: importlib.metadata.version(name) for name in names}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def identity(kind, **values):
    """A checkpoint only resumes the same code, environment, inputs and recipe."""
    record = {"kind": kind, "environment": environment(),
              "helper_sha256": file_hash(__file__),
              "analysis_sources": {path.name: file_hash(path)
                                   for path in sorted((HERE / "liver_analysis").glob("*.py"))},
              "selected_sha256": file_hash(HERE / "liver_selected.json"),
              **values}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def make_windows(norm, nodes, patch_size=10):
    """Use the same complete patch lattice and signal mask as the summaries."""
    blocks = []
    for roi in sorted(norm):
        sub = nodes[nodes.roi == roi]
        image = np.asarray(norm[roi], dtype=np.float32)
        channels, height, width = image.shape
        grid = image.reshape(channels, height // patch_size, patch_size,
                             width // patch_size, patch_size).transpose(1, 3, 0, 2, 4)
        keep = sub.signal.to_numpy(bool)
        blocks.append(grid[sub.gy.to_numpy(int)[keep], sub.gx.to_numpy(int)[keep]])
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)


def fit_cnn_cached(windows, *, seed, config, output_dir, device):
    """Fit one selected encoder, or resume its verified fresh-fit checkpoint."""
    folder = Path(output_dir) / f"cnn_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = identity("cnn", windows=array_hash(windows), seed=int(seed),
                     config=config, device=str(device))
    meta_path = folder / "checkpoint.json"
    paths = {name: folder / name for name in ("embedding.npy", "loss.npy", "model.pt")}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("identity") != stamp:
            raise RuntimeError(f"{folder} belongs to different inputs/settings; use a new result directory.")
        if all(path.exists() and file_hash(path) == meta["files"][name]
               for name, path in paths.items()):
            cnn = LiverCNN(config["latent_dim"])
            cnn.load_state_dict(torch.load(paths["model.pt"], map_location="cpu", weights_only=True))
            print(f"Resume CNN seed {seed}: verified completed fit.")
            return cnn, np.load(paths["embedding.npy"]), np.load(paths["loss.npy"]), meta
    t0 = time.perf_counter()
    cnn, embedding, losses = fit_liver_cnn(windows, seed=seed, device=device, **config)
    np.save(paths["embedding.npy"], embedding)
    np.save(paths["loss.npy"], losses)
    torch.save(cnn.cpu().state_dict(), paths["model.pt"])
    meta = {"identity": stamp, "seed": int(seed), "config": config, "source": "fresh_fit",
            "seconds": time.perf_counter() - t0,
            "files": {name: file_hash(path) for name, path in paths.items()}}
    write_json(meta_path, meta)
    return cnn, embedding, np.asarray(losses), meta


def cluster_embedding(raw, *, seed=0, K=6):
    embedding = StandardScaler().fit_transform(raw).astype(np.float32)
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(embedding)
    order = sorted(range(K), key=lambda c: (-int((labels == c).sum()), c))
    mapping = np.empty(K, dtype=int)
    mapping[order] = np.arange(K)
    return embedding, mapping[labels]


def use_partition(nodes, labels, *, mode="fresh"):
    """Explicitly choose a fresh partition or the deposited conditional partition."""
    result = nodes.copy()
    result["cluster"] = -1
    if mode == "fresh":
        if len(labels) != int(result.signal.sum()):
            raise ValueError("Fresh labels and signal rows differ.")
        result.loc[result.signal, "cluster"] = labels
    elif mode == "paper":
        with np.load(HERE / "liver_analysis/paper_partition.npz") as frozen:
            index, paper_labels = frozen["signal_index"], frozen["labels"]
        if len(result) != 240000 or index.max() >= len(result):
            raise ValueError("Deposited partition requires the original complete 24-ROI lattice.")
        result["signal"] = False
        result.loc[index, "signal"] = True
        result.loc[index, "cluster"] = paper_labels
    else:
        raise ValueError("partition mode must be fresh or paper")
    return result


def make_graphs(nodes, meta, *, graph_type="delaunay"):
    """Graph over the complete lattice, then Mantpy's signal-induced OHE graph."""
    adatas = {}
    for roi in sorted(nodes.roi.unique()):
        sub = nodes[nodes.roi == roi].reset_index(drop=True)
        a = ad.AnnData(np.empty((0, 6), dtype=np.float32))
        a.uns["ecm_patches"] = pd.DataFrame(
            {"x": sub.x.to_numpy(), "y": sub.y.to_numpy(),
             "ecm_cluster": sub.cluster.to_numpy(int)})
        if graph_type == "delaunay":
            mt.gr.build_ecm_graph(a, edge_method="delaunay", edge_weight="uniform")
        elif graph_type.startswith("radius"):
            mt.gr.build_ecm_graph(a, edge_method="radius", r=float(graph_type[6:]),
                                  edge_weight="uniform")
        elif graph_type in {"grid4", "grid8"}:
            mt.gr.build_ecm_graph(a, edge_method="grid", grid_connectivity=int(graph_type[-1]),
                                  edge_weight="uniform")
        else:
            raise ValueError(graph_type)
        adatas[roi] = a
    bundle = mt.nn.build_ohe_cluster_graphs(
        adatas, meta.set_index("roi"), group_col="condition",
        group_order=["Naive", "Infected"])
    return adatas, bundle.graphs


def graph_identity(graphs):
    return [{"roi": str(g.roi_id), "x": array_hash(g.x.numpy()),
             "edges": array_hash(g.edge_index.numpy()),
             "pos": array_hash(g.pos.numpy())} for g in graphs]


class SelectedGNN(nn.Module):
    """Original selected architecture, constructed with public PyG layers."""

    def __init__(self, config, in_channels=6):
        super().__init__()
        hid, layers = config["hidden_dim"], config["n_layers"]
        self.proj = nn.Linear(in_channels, hid)
        conv = config.get("conv", "sage")
        def layer():
            if conv == "sage":
                return SAGEConv(hid, hid, aggr=config.get("aggr", "mean"))
            if conv == "gcn":
                return GCNConv(hid, hid)
            if conv == "gat":
                heads = config.get("heads", 1)
                return GATConv(hid, hid // heads, heads=heads, concat=True)
            if conv == "gin":
                act = nn.GELU() if config.get("activation") == "gelu" else nn.ReLU()
                return GINConv(nn.Sequential(nn.Linear(hid, hid), act, nn.Linear(hid, hid)),
                               train_eps=config.get("train_eps", False))
            raise ValueError(conv)
        self.convs = nn.ModuleList([layer() for _ in range(layers)])
        self.dropout = config["dropout"]
        self.head = nn.Linear(hid, 2)
        norm = config.get("norm", "none")
        factory = {"none": nn.Identity, "layer": lambda: nn.LayerNorm(hid),
                   "batch": lambda: nn.BatchNorm1d(hid)}[norm]
        self.norms = nn.ModuleList([factory() for _ in range(layers)])
        self.residual = config.get("residual", False)
        self.act = F.gelu if config.get("activation") == "gelu" else F.relu

    def forward(self, x, edge_index):
        h = self.act(self.proj(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            out = F.dropout(self.act(norm(conv(h, edge_index))),
                            p=self.dropout, training=self.training)
            h = h + out if self.residual else out
        return self.head(h)








def fit_gnn_cached(graphs, meta, *, held_mouse, seed, config, objective,
                   output_dir, labels_override=None, graph_stamp=None):
    """One complete outer fit; scalers and class weights use training ROIs only."""
    info = meta.set_index("roi")
    mice = np.asarray([info.loc[g.roi_id, "mouse"] for g in graphs], str)
    labels = (np.asarray([int(g.y) for g in graphs]) if labels_override is None
              else np.asarray(labels_override, int))
    train = np.flatnonzero(mice != held_mouse)
    test = np.flatnonzero(mice == held_mouse)
    if not len(test) or set(mice[train]) & set(mice[test]):
        raise ValueError("Invalid mouse-disjoint split.")
    folder = Path(output_dir) / held_mouse / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = identity("gnn", graphs=graph_stamp or graph_identity(graphs),
                     mice=mice.tolist(), labels=labels.tolist(), held_mouse=held_mouse,
                     seed=int(seed), config=config, objective=objective)
    state_path, meta_path = folder / "model.pt", folder / "checkpoint.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("identity") != stamp:
            raise RuntimeError(f"{folder} contains a different model; use a new result directory.")
        if state_path.exists() and file_hash(state_path) == saved["model_sha256"]:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            model = SelectedGNN(config)
            model.load_state_dict(state["model"])
            model.eval()
            return {"model": model, "mean": state["mean"], "scale": state["scale"],
                    "checkpoint": saved, "folder": folder, "test": test}
    t0 = time.perf_counter()
    trained = mt.nn.train_node_classifier(
        [graphs[i] for i in train], labels[train], model_factory=SelectedGNN,
        model_kwargs={"config": config}, n_classes=2,
        epochs=config["epochs"], lr=config["lr"], weight_decay=config["weight_decay"],
        seed=seed, loss_reduction="roi" if objective == "roi_mean" else "node",
        roi_class_weight_mode="after",
        class_weight="balanced" if config.get("class_weight", True) else None,
        optimizer_factory=torch.optim.AdamW if config.get("optimizer") == "adamw" else torch.optim.Adam,
        scheduler="cosine" if config.get("lr_schedule") == "cosine" else None,
        input_dropout=config.get("input_dropout", 0.0),
        label_smoothing=config.get("label_smoothing", 0.0), standardize=True, device="cpu",
    )
    model, mean, scale = trained.model, trained.mean, trained.scale
    weights, losses = trained.class_weights, trained.history["train_loss"]
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "mean": mean, "scale": scale}, temporary)
    os.replace(temporary, state_path)
    record = {"identity": stamp, "source": "fresh_fit", "held_mouse": held_mouse,
              "seed": int(seed), "objective": objective, "config": config,
              "train_rois": [str(graphs[i].roi_id) for i in train],
              "test_rois": [str(graphs[i].roi_id) for i in test],
              "training_mice": sorted(set(mice[train])),
              "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
              "class_weights": weights.tolist() if weights is not None else None,
              "loss": losses, "seconds": time.perf_counter() - t0,
              "model_sha256": file_hash(state_path)}
    write_json(meta_path, record)
    print(f"{objective}: {held_mouse}, seed {seed}, {record['seconds']:.1f}s", flush=True)
    return {"model": model, "mean": mean, "scale": scale, "checkpoint": record,
            "folder": folder, "test": test}


def graph_probability(fit, graph):
    trained = mt.nn.GraphClassifierFit(
        model=fit["model"], mean=fit["mean"], scale=fit["scale"],
        history={}, class_weights=None, optimizer_updates=0,
        loss_reduction=fit["checkpoint"]["objective"], seed=fit["checkpoint"]["seed"],
    )
    return float(trained.predict_proba(graph).mean(0)[1])


def integrated_gradients_cached(fit, graph, *, steps=50):
    """Per-seed signed IG; ensemble importance is |mean signed IG|, not mean |IG|."""
    path = fit["folder"] / f"ig_{graph.roi_id}_{steps}.npz"
    stamp = identity("ig", fit=fit["checkpoint"]["identity"], roi=str(graph.roi_id),
                     x=array_hash(graph.x.numpy()), edges=array_hash(graph.edge_index.numpy()),
                     steps=int(steps), target="mean infected logit", baseline="zero_standardized")
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text())
        if record["identity"] == stamp and record["sha256"] == file_hash(path):
            with np.load(path) as z:
                return z["signed"], float(z["delta"])
    t0 = time.perf_counter()
    attribution = mt.nn.node_integrated_gradients(
        fit["model"], graph, mean=fit["mean"], scale=fit["scale"], target_class=1,
        n_steps=steps, internal_batch_size=1, return_details=True,
    )
    signed, delta = attribution.signed, float(attribution.convergence_delta.item())
    np.savez_compressed(path, signed=signed, delta=delta)
    write_json(meta_path, {"identity": stamp, "sha256": file_hash(path),
                          "seconds": time.perf_counter() - t0})
    return signed, delta


def run_outer_folds(graphs, meta, *, seeds, config, objective, output_dir, explain=False,
                    steps=50, labels_override=None, held_mice=None):
    """Fit all requested mouse folds and keep per-seed and ensemble ROI results."""
    lookup = meta.set_index("roi")
    selected_mice = sorted(meta.mouse.unique()) if held_mice is None else list(held_mice)
    rows, attributions, residuals = [], [], []
    stamp = graph_identity(graphs)
    for mouse in selected_mice:
        test = [i for i, g in enumerate(graphs) if lookup.loc[g.roi_id, "mouse"] == mouse]
        signed_total = {i: np.zeros(tuple(graphs[i].x.shape), np.float64) for i in test} if explain else {}
        for seed in seeds:
            fit = fit_gnn_cached(graphs, meta, held_mouse=mouse, seed=seed, config=config,
                                 objective=objective, output_dir=output_dir,
                                 labels_override=labels_override, graph_stamp=stamp)
            for i in fit["test"]:
                graph = graphs[i]
                true = int(graph.y) if labels_override is None else int(labels_override[i])
                rows.append({"roi": str(graph.roi_id), "mouse": mouse,
                             "group": str(lookup.loc[graph.roi_id, "group"]),
                             "seed": int(seed), "true_infected": true,
                             "P_infected": graph_probability(fit, graph)})
                if explain:
                    signed, delta = integrated_gradients_cached(fit, graph, steps=steps)
                    signed_total[i] += signed / len(seeds)
                    residuals.append({"roi": str(graph.roi_id), "mouse": mouse,
                                      "seed": int(seed), "completeness_residual": delta})
            del fit
        if explain:
            for i in test:
                graph = graphs[i]
                signed = signed_total[i]
                own = graph.x.numpy().argmax(1)
                attributions.append(pd.DataFrame({
                    "roi": str(graph.roi_id), "mouse": mouse,
                    "group": str(lookup.loc[graph.roi_id, "group"]),
                    "x": graph.pos[:, 0].numpy(), "y": graph.pos[:, 1].numpy(),
                    "cluster": graph.cluster.numpy(),
                    "ig_infected": np.abs(signed).sum(1),
                    "ig_signed_own_column": signed[np.arange(len(own)), own]}))
        print(f"Completed held-out mouse {mouse}.", flush=True)
    seed_table = pd.DataFrame(rows)
    ensemble = seed_table.groupby(["roi", "mouse", "group", "true_infected"], as_index=False).P_infected.mean()
    return {"seeds": seed_table, "ensemble": ensemble,
            "attributions": pd.concat(attributions, ignore_index=True) if attributions else pd.DataFrame(),
            "ig_residuals": pd.DataFrame(residuals)}


def metric_table(ensemble, model="GraphSAGE"):
    rows = []
    for level in ("ROI", "mouse"):
        table = ensemble if level == "ROI" else ensemble.groupby("mouse", as_index=False).agg(
            true_infected=("true_infected", "first"), P_infected=("P_infected", "mean"))
        y, p = table.true_infected.to_numpy(int), table.P_infected.to_numpy(float)
        rows.append({"model": model, "level": level, "n": len(y),
                     "correct": int(((p >= 0.5) == y).sum()),
                     "accuracy": float(np.mean((p >= 0.5) == y)),
                     "ROC_AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                     "Brier": float(brier_score_loss(y, p)),
                     "log_loss": float(log_loss(y, p, labels=[0, 1]))})
    return pd.DataFrame(rows)


def composition_lr(graphs, meta, selected_C, held_mice=None):
    """Replay each saved training-fold C, then refit on training fractions only."""
    lookup = meta.set_index("roi")
    mice = np.asarray([lookup.loc[g.roi_id, "mouse"] for g in graphs], str)
    y = np.asarray([int(g.y) for g in graphs])
    fractions = np.stack([np.bincount(g.cluster.numpy(), minlength=6) / g.num_nodes
                          for g in graphs])
    rows = []
    for mouse in sorted(set(mice)) if held_mice is None else held_mice:
        train, test = np.flatnonzero(mice != mouse), np.flatnonzero(mice == mouse)
        scaler = StandardScaler().fit(fractions[train])
        model = LogisticRegression(C=selected_C[mouse]["C"], max_iter=10000).fit(
            scaler.transform(fractions[train]), y[train])
        p = model.predict_proba(scaler.transform(fractions[test]))[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        for i, probability in zip(test, p, strict=True):
            rows.append({"roi": str(graphs[i].roi_id), "mouse": mouse,
                         "group": str(lookup.loc[graphs[i].roi_id, "group"]),
                         "true_infected": int(y[i]), "P_infected": float(probability),
                         "selected_C": selected_C[mouse]["C"]})
    return pd.DataFrame(rows)


def merge_outer_results(results):
    return {key: pd.concat([result[key] for result in results], ignore_index=True)
            for key in ("seeds", "ensemble", "attributions", "ig_residuals")}


def fit_fold_partition(images, meta, *, held_mouse, representation, config,
                       output_dir, device="cpu"):
    """Refit every learned preprocessing step on the seven training mice."""
    from liver_analysis.data import FEAT_COLS
    from liver_analysis.fold_preprocessing import prepare_fold, fit_fold_clusters
    from torch.utils.data import DataLoader
    folder = Path(output_dir) / held_mouse
    folder.mkdir(parents=True, exist_ok=True)
    stamp = identity(
        "fold_partition", images={roi: array_hash(image) for roi, image in images.items()},
        meta=meta.to_dict("records"), held_mouse=held_mouse,
        representation=representation, config=config, device=str(device))
    checkpoint, table_path = folder / "partition.json", folder / "nodes.csv.gz"
    if checkpoint.exists() and table_path.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("identity") != stamp:
            raise RuntimeError(f"{folder} has a different representation; use a new result directory.")
        if saved.get("nodes_sha256") == file_hash(table_path):
            print(f"Resume {representation} partition: {held_mouse}.", flush=True)
            return pd.read_csv(table_path), saved
    t0 = time.perf_counter()
    nodes, norm, prep_info = prepare_fold(images, meta, held_mouse)
    train_signal = (nodes.loc[nodes.signal, "mouse"] != held_mouse).to_numpy()
    if representation == "marker_summary":
        features = nodes.loc[nodes.signal, FEAT_COLS].to_numpy(np.float32)
    elif representation == "cnn":
        windows = make_windows(norm, nodes)
        cnn, _, _, encoder_record = fit_cnn_cached(
            windows[train_signal], seed=0, config=config,
            output_dir=folder, device=device)
        cnn.to(device).eval()
        with torch.no_grad():
            features = np.concatenate([
                cnn(batch.to(device)).cpu().numpy()
                for batch in DataLoader(torch.from_numpy(windows),
                                        batch_size=4096, shuffle=False)
            ]).astype(np.float32)
        cnn.cpu()
        prep_info["encoder_identity"] = encoder_record["identity"]
    else:
        raise ValueError(representation)
    labels, clustering_info = fit_fold_clusters(features, train_signal)
    nodes["cluster"] = -1
    nodes.loc[nodes.signal, "cluster"] = labels
    nodes.to_csv(table_path, index=False)
    record = {"identity": stamp, "source": "fresh_fit", "representation": representation,
              "preprocessing": prep_info, "clustering": clustering_info,
              "seconds": time.perf_counter() - t0,
              "nodes_sha256": file_hash(table_path)}
    write_json(checkpoint, record)
    return nodes, record


def save_outer_outputs(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("seeds", "ensemble", "ig_residuals"):
        if len(result[key]):
            result[key].to_csv(folder / f"{key}.csv", index=False)
    if len(result["attributions"]):
        result["attributions"].to_csv(folder / "node_attribution.csv.gz", index=False)


def show_patch_map(nodes, roi, title=None, outline=None):
    sub = nodes[nodes.roi == roi]
    grid = np.full((int(sub.gy.max()) + 1, int(sub.gx.max()) + 1), np.nan)
    keep = sub.cluster.to_numpy(int) >= 0
    grid[sub.gy.to_numpy(int)[keep], sub.gx.to_numpy(int)[keep]] = sub.cluster.to_numpy(int)[keep]
    fig, ax = plt.subplots(figsize=(6, 6))
    mt.pl.patch_map(sub.rename(columns={"gx": "grid_x", "gy": "grid_y"}), "cluster",
                    patch_size=10, ax=ax, categorical=True, background="white",
                    palette=dict(enumerate(plt.get_cmap("tab10").colors[:6])), vmin=-0.5, vmax=5.5)
    im = ScalarMappable(norm=Normalize(-.5, 5.5), cmap=ListedColormap(plt.get_cmap("tab10").colors[:6]))
    if outline is not None:
        ax.contour(np.nan_to_num(grid, nan=-1) == outline, levels=[0.5],
                   colors="black", linewidths=0.6)
    ax.set_title(title or f"{roi} · ECM clusters")
    ax.axis("off")
    fig.colorbar(im, ax=ax, ticks=range(6), label="ECM cluster", shrink=0.65)
    plt.show()


def show_ig(attribution, roi, infection_cluster):
    sub = attribution[attribution.roi == roi]
    gx = ((sub.x.to_numpy() - 5) // 10).astype(int)
    gy = ((sub.y.to_numpy() - 5) // 10).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    cluster = np.zeros(grid.shape, bool)
    grid[gy, gx] = sub.ig_infected
    cluster[gy, gx] = sub.cluster == infection_cluster
    fig, ax = plt.subplots(figsize=(6, 6))
    mt.pl.patch_map(sub, "ig_infected", patch_size=10, ax=ax, categorical=False,
                    background="white", cmap="magma", vmin=0,
                    vmax=np.percentile(sub.ig_infected, 99))
    im = ScalarMappable(norm=Normalize(0, np.percentile(sub.ig_infected, 99)), cmap="magma")
    if cluster.any() and not cluster.all():
        ax.contour(cluster, levels=[0.5], colors="#00BFC4", linewidths=0.6)
    ax.set_title(f"{roi} · ensemble |IG| towards infected")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Sum of |mean signed IG|", shrink=0.65)
    plt.show()


def stability_table(labels_by_seed):
    rows = [{"seed_a": a, "seed_b": b, "ARI": adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])}
            for a, b in itertools.combinations(sorted(labels_by_seed), 2)]
    return pd.DataFrame(rows)


def graph_control(graphs, mode, *, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for graph in graphs:
        new = graph.clone()
        if mode == "none":
            new.edge_index = torch.empty((2, 0), dtype=torch.long)
        elif mode == "shuffled":
            edges = graph.edge_index.numpy()
            new.edge_index = torch.from_numpy(np.stack([edges[0], rng.permutation(edges[1])])).long()
        else:
            raise ValueError(mode)
        result.append(new)
    return result


def permutation_test(graphs, meta, *, seeds, config, output_dir, allocations=None):
    """Exact selected-model null; fold/seed checkpoints make interruption safe."""
    mice = sorted(meta.mouse.unique())
    lookup = meta.set_index("roi")
    original = {mouse: int(meta.loc[meta.mouse == mouse, "condition"].iloc[0] == "Infected")
                for mouse in mice}
    genotypes = {mouse: str(meta.loc[meta.mouse == mouse, "genotype"].iloc[0]) for mouse in mice}
    observed = tuple(mouse for mouse in mice if original[mouse])
    all_allocations = list(itertools.combinations(mice, 4))
    if len(all_allocations) != 70:
        raise ValueError("The exact null requires eight mice and four infected labels.")
    indices = range(70) if allocations is None else list(allocations)
    rows = []
    for index in indices:
        allocation = all_allocations[index]
        labels = np.asarray([int(lookup.loc[g.roi_id, "mouse"] in allocation) for g in graphs])
        folder = Path(output_dir) / f"allocation_{index:02d}"
        result = run_outer_folds(graphs, meta, seeds=seeds, config=config, objective="roi_mean",
                                 output_dir=folder, labels_override=labels)
        save_outer_outputs(result, folder)
        mouse_result = result["ensemble"].groupby("mouse", as_index=False).agg(
            true_infected=("true_infected", "first"), P_infected=("P_infected", "mean"))
        auc = float(roc_auc_score(mouse_result.true_infected, mouse_result.P_infected))
        preserving = all(sum(genotypes[m] == genotype for m in allocation) == 2
                         for genotype in sorted(set(genotypes.values())))
        row = {"allocation": int(index), "infected_mice": "|".join(allocation),
               "observed": allocation == observed, "genotype_preserving": preserving,
               "AUC": auc, "accuracy": float(np.mean((mouse_result.P_infected >= 0.5) == mouse_result.true_infected))}
        write_json(folder / "summary.json", row)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(Path(output_dir) / "allocations.csv", index=False)
    summary = {"completed_allocations": len(table), "requested_allocations": 70,
               "seeds": list(seeds), "exact_test_complete": len(table) == 70 and len(seeds) == 5}
    if summary["exact_test_complete"]:
        if table.observed.sum() != 1 or table.genotype_preserving.sum() != 36:
            raise AssertionError("Exact-null allocation inventory is incomplete.")
        observed_auc = float(table.loc[table.observed, "AUC"].iloc[0])
        tail = table.AUC >= observed_auc
        summary.update(observed_auc=observed_auc, upper_tail=int(tail.sum()), p=float(tail.mean()),
                       genotype_upper_tail=int((tail & table.genotype_preserving).sum()),
                       genotype_p=float(tail[table.genotype_preserving].mean()),
                       observed_and_ties_included=True)
    write_json(Path(output_dir) / "exact_null.json", summary)
    return table, summary

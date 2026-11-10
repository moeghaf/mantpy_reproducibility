"""Paper-specific pieces of the liver Figure 3 notebook.

The notebook shows every analysis step as a plain Mantpy call. This file holds what is specific to
reproducing the paper, and each function takes everything it uses as an argument:

- the paper's values, read from the source-data workbook into tables with explicit keys
  (``read_paper_values``), and the one-to-one matching of this run's clusters to the paper's
  (``match_clusters``);
- comparisons that are joined on those keys (``side_by_side``, ``unordered_pairs``, ``joint_correlation``,
  ``patch_counts``, ``classification_summary``);
- the parts of the panels that Mantpy does not draw (``scale_bar``, ``outline_patches``,
  ``correlation_label``, ``plot_held_out_probabilities``, ``plot_cluster_correlations``, ``degree_map``,
  ``plot_subgraph_degrees``).
"""

from __future__ import annotations

import mantpy as mt
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------------------------------
# The paper's values
# ---------------------------------------------------------------------------------------------------

def read_paper_values(path):
    """Read the paper's Figure 3 values into one table per quantity, each with explicit keys.

    Cluster numbers are the paper's.
    """
    sheets = pd.read_excel(path, sheet_name=None)

    def by_cluster(table):   # columns "ECM0" ... "ECM5" become the integer cluster numbers
        return table.rename(columns=lambda name: int(name.removeprefix("ECM"))).rename_axis(columns="cluster")

    profiles = sheets["b_profiles"].set_index("cluster")
    pairs = sheets["c_cooccurrence"]
    labels = sorted(set(pairs.cluster_a) | set(pairs.cluster_b))
    ko_minus_wt = pd.DataFrame(np.nan, index=pd.Index(labels, name="cluster_a"),
                               columns=pd.Index(labels, name="cluster_b"))
    for a, b, value in zip(pairs.cluster_a, pairs.cluster_b, pairs.ko_minus_wt):
        ko_minus_wt.loc[a, b] = ko_minus_wt.loc[b, a] = value

    localisation = sheets["h_localisation"]
    infection = localisation.loc[localisation.is_infection_cluster, "cluster"].unique()
    if len(infection) != 1:
        raise ValueError("The workbook must mark exactly one infection-associated cluster.")

    mice = sheets["g_mice"].set_index("mouse")
    seed_0 = sheets["g_seeds"].query("seed == 0").set_index("mouse").p_infected_gnn
    return {
        "cluster_sizes": profiles.n_patches,
        "profiles": (profiles.filter(like="z_mean_").rename(columns=lambda name: name.removeprefix("z_mean_"))
                     .rename_axis(columns="marker")),
        "group_fractions": by_cluster(sheets["a_groups"].set_index("group").drop(columns="n_mice")),
        "mouse_fractions": by_cluster(sheets["a_mice"].set_index("mouse").drop(columns="group")),
        "ko_minus_wt": ko_minus_wt,
        "bounds": sheets["d_display"].set_index("marker").transform_bound_99p5,
        "shown_patches": sheets["e_f_patches"],
        "probabilities": pd.DataFrame({"GNN, seed 0": seed_0, "GNN, five seeds": mice.p_infected_gnn,
                                       "LR": mice.p_infected_lr}),
        "localisation": localisation,
        "subgraphs": sheets["i_subgraphs"],
        "infection_cluster": int(infection[0]),
    }


def match_clusters(profiles, paper_profiles):
    """Pair every cluster of this run with one cluster of the paper, by their marker profiles.

    Both tables have one row per cluster and one column per marker. The pairing is one-to-one and
    maximises the summed Pearson correlation of the paired profiles, so two clusters of this run can
    never be compared with the same paper cluster. The last column shows how close the nearest
    alternative was.
    """
    if profiles.shape != paper_profiles.shape or list(profiles.columns) != list(paper_profiles.columns):
        raise ValueError("The two profile tables need the same clusters-by-markers shape and marker order.")
    n = len(profiles)
    similarity = np.corrcoef(profiles.to_numpy(float), paper_profiles.to_numpy(float))[:n, n:]
    rows, columns = linear_sum_assignment(-similarity)
    others = similarity.copy()
    others[rows, columns] = -np.inf
    return pd.DataFrame(
        {"paper cluster": paper_profiles.index[columns],
         "profile correlation": similarity[rows, columns],
         "highest correlation with another paper cluster": others.max(axis=1)[rows]},
        index=pd.Index(profiles.index[rows], name="cluster"),
    )


# ---------------------------------------------------------------------------------------------------
# Comparisons joined on keys
# ---------------------------------------------------------------------------------------------------

def side_by_side(this_run, paper):
    """This run's values next to the paper's, joined on their index.

    Both sides need unique keys and the same set of keys (and, for tables, the same columns). A missing
    or repeated key raises an error instead of leaving a gap or a wrong pairing.
    """
    for name, values in (("this run", this_run), ("the paper", paper)):
        if not values.index.is_unique:
            raise ValueError(f"The keys of {name} are not unique.")
    unmatched = set(this_run.index) ^ set(paper.index)
    if unmatched:
        raise ValueError(f"Keys present on one side only: {sorted(unmatched, key=str)}")
    paper = paper.loc[this_run.index]
    if isinstance(this_run, pd.Series):
        return pd.DataFrame({"this run": this_run, "paper": paper})
    if set(this_run.columns) != set(paper.columns):
        raise ValueError("The two tables have different columns.")
    key = this_run.index.name or paper.index.name
    return (pd.concat({"this run": this_run, "paper": paper[this_run.columns]}, names=["", key])
            .swaplevel().loc[list(this_run.index)])


def unordered_pairs(matrix):
    """The values of a symmetric cluster-pair table, one per unordered pair (cluster_a <= cluster_b)."""
    labels = list(matrix.index)
    if labels != list(matrix.columns) or labels != sorted(labels):
        raise ValueError("The pair table needs the same sorted cluster labels on both axes.")
    values = {(a, b): matrix.loc[a, b] for position, a in enumerate(labels) for b in labels[position:]}
    return pd.Series(values).rename_axis(["cluster_a", "cluster_b"])


def joint_correlation(this_run, paper):
    """Pearson r between two keyed series over the keys where both are finite: ``(r, number of keys)``."""
    table = side_by_side(this_run, paper).astype(float)
    table = table[np.isfinite(table).all(axis=1)]
    if len(table) < 3:
        return np.nan, len(table)
    return float(np.corrcoef(table["this run"], table["paper"])[0, 1]), len(table)


def patch_counts(patches, rois, cluster):
    """Number of ECM patches, and of patches of one cluster, in each given ROI (0 where there is none)."""
    patches = patches[patches.roi.isin(rois)]
    of_cluster = patches[patches.cluster == cluster]
    return pd.DataFrame({
        "ECM patches": patches.groupby("roi").size().reindex(rois, fill_value=0),
        "patches of the infection-associated cluster": of_cluster.groupby("roi").size().reindex(rois, fill_value=0),
    }).rename_axis("roi")


def classification_summary(probabilities, positive, cutoff=0.5):
    """For each column of held-out probabilities: units on the correct side of the cut-off, and ROC AUC."""
    if not probabilities.index.equals(positive.index):
        raise ValueError("The probabilities and the true classes need the same units in the same order.")
    return pd.DataFrame({
        f"classified correctly (of {len(positive)})": (probabilities >= cutoff).eq(positive, axis=0).sum(),
        "ROC AUC": probabilities.apply(lambda column: roc_auc_score(positive, column)),
    })


# ---------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------

def _grid_cells(table, patch_size):
    """Row and column of each patch on the patch grid, as ``mt.pl.patch_map`` places them."""
    return ((table.y.to_numpy() // patch_size).astype(int), (table.x.to_numpy() // patch_size).astype(int))


def _spread(count, half_width):
    """Horizontal offsets that place ``count`` points side by side."""
    return np.linspace(-half_width, half_width, count) if count > 1 else np.zeros(count)


def scale_bar(ax, length, label, *, color="black"):
    """Draw a scale bar of ``length`` axis units, with its label, in the lower-left corner of an image.

    The bar and the label get a thin outline in the opposite shade, so they stay readable over any colour.
    """
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    x, y = left + 0.06 * (right - left), bottom + 0.07 * (top - bottom)
    halo = "white" if sum(to_rgb(color)) < 1.5 else "black"
    outline = [patheffects.withStroke(linewidth=2.5, foreground=halo)]
    ax.plot([x, x + length], [y, y], color=color, linewidth=2.5, solid_capstyle="butt",
            path_effects=[patheffects.withStroke(linewidth=4.5, foreground=halo)])
    ax.text(x + length / 2, y + 0.03 * (top - bottom), label, color=color, ha="center", va="bottom", fontsize=9,
            path_effects=outline)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


def outline_patches(ax, patches, *, patch_size, shape, color, linewidth=1.0, smooth=0.7):
    """Outline a set of patches on a map drawn by ``mt.pl.patch_map``. An empty set draws nothing."""
    mask = np.zeros(shape)
    mask[_grid_cells(patches, patch_size)] = 1.0
    if 0 < mask.sum() < mask.size:
        ax.contour(gaussian_filter(mask, smooth), levels=[0.25], colors=[color], linewidths=linewidth)


def correlation_label(n_cluster_nodes, pearson_r, *, cluster, min_cluster_nodes=2):
    """Text for a map: the correlation, or the reason why there is none."""
    if n_cluster_nodes == 0:
        return f"no ECM {cluster} patch"
    if n_cluster_nodes < min_cluster_nodes:
        return f"{n_cluster_nodes} ECM {cluster} patch: too few for r"
    if not np.isfinite(pearson_r):
        return "r undefined"
    return f"r = {pearson_r:.2f}"


def plot_held_out_probabilities(ax, graph_model, composition_model, *, group, positive, group_order, cutoff=0.5):
    """Held-out probability of each unit (mouse), by group, for two models.

    Circles are the graph classifier and diamonds the composition-only model. Filled symbols are the
    units that truly belong to the positive class. All inputs are series indexed by unit.
    """
    handles = []
    for values, marker, size, shift, face, edge, label in (
        (graph_model, "o", 55, -0.07, "#333333", "#333333", "GNN (graph)"),
        (composition_model, "D", 32, 0.07, "#BFBFBF", "#7F7F7F", "LR (composition only)"),
    ):
        for position, name in enumerate(group_order):
            units = sorted(group.index[group == name])
            x = position + shift + _spread(len(units), 0.16)
            filled = positive.loc[units].to_numpy(bool)
            ax.scatter(x, values.loc[units], s=size, marker=marker, linewidth=1.0, zorder=3,
                       facecolor=[face if is_positive else "white" for is_positive in filled], edgecolor=edge)
        handles.append(Line2D([], [], linestyle="", marker=marker, markersize=np.sqrt(size), markerfacecolor=face,
                              markeredgecolor=edge, label=label))
    ax.axhline(cutoff, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(range(len(group_order)), [name.replace(" ", "\n") for name in group_order])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1)
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9, handletextpad=0.2)


def plot_cluster_correlations(ax, table, *, palette, group_markers):
    """Per-ROI Pearson r (points) and its median (line) for each cluster; one marker shape per group.

    ``table`` has the columns ``cluster``, ``group`` and ``pearson_r``; undefined correlations are left out.
    ``group_markers`` maps each group to its marker shape.
    """
    clusters = sorted(table.cluster.unique())
    for position, cluster in enumerate(clusters):
        defined = table[(table.cluster == cluster) & table.pearson_r.notna()]
        for shift, (name, marker) in zip(_spread(len(group_markers), 0.07), group_markers.items()):
            values = defined.loc[defined.group == name, "pearson_r"].to_numpy(float)
            ax.scatter(position + shift + _spread(len(values), 0.12), values, s=22, marker=marker,
                       color=palette[cluster], linewidth=0, alpha=0.9, zorder=3)
        if len(defined):
            ax.plot([position - 0.3, position + 0.3], [defined.pearson_r.median()] * 2, color="#222222",
                    linewidth=1.2, zorder=4)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    ax.set_ylim(-0.6, 1.05)
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xticks(range(len(clusters)), [str(cluster) for cluster in clusters])
    for tick, cluster in zip(ax.get_xticklabels(), clusters):
        tick.set_color(palette[cluster])
        tick.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], linestyle="", marker=marker, markersize=5, color="#666666", label=name)
                       for name, marker in group_markers.items()],
              title="ROI", loc="upper left", frameon=False, fontsize=9)


def degree_map(ax, patches, *, patch_size, shape, cmap, max_degree):
    """Map of the connected regions of one cluster in one ROI.

    ``patches`` holds every ECM patch of the ROI (``x``, ``y``) and, for the patches of the cluster,
    ``kept`` (does the patch lie in a kept region?) and ``degree``. Kept regions are coloured by node
    degree and outlined, smaller fragments are grey, and the other ECM patches light grey.
    """
    in_cluster = patches.kept.notna().to_numpy()
    kept = patches.kept.eq(True).to_numpy()
    colours = np.tile(to_rgb("#EEEEEE"), (len(patches), 1))
    colours[in_cluster & ~kept] = to_rgb("#BDBDBD")
    colours[kept] = cmap(np.clip(patches.degree.to_numpy(float)[kept] / max_degree, 0, 1))[:, :3]
    mt.pl.patch_domain_map(ax, *_grid_cells(patches, patch_size), colours, shape, crop=False)
    outline_patches(ax, patches[kept], patch_size=patch_size, shape=shape, color="#333333", linewidth=0.7, smooth=0.6)


def plot_subgraph_degrees(ax, subgraphs, *, units, cmap, max_degree, unit_col="mouse"):
    """Mean node degree of every kept subgraph (points) and the median (line), for each given unit."""
    for position, unit in enumerate(units):
        degree = subgraphs.loc[subgraphs[unit_col] == unit, "mean_degree"].to_numpy(float)
        if len(degree):
            ax.scatter(position + _spread(len(degree), 0.16), degree, s=32, c=cmap(np.clip(degree / max_degree, 0, 1)),
                       linewidth=0.5, edgecolor="#333333", zorder=3)
            ax.plot([position - 0.3, position + 0.3], [np.median(degree)] * 2, color="#222222", linewidth=1.2, zorder=4)
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.set_ylim(0, None)
    ax.set_xticks(range(len(units)), list(units))
    ax.spines[["top", "right"]].set_visible(False)

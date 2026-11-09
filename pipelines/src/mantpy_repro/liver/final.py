"""Final fits with the selected settings (Fig. 3g, h; Supplementary Fig. 2a; Supplementary Table 5).

Tasks, each independent and resumable:

- ``encoder``: one training seed of the selected encoder (five seeds; seed stability, and the
  same-seed refit of the partition);
- ``classifier``: the fixed GraphSAGE classifier with the ROI-mean loss for one held-out mouse, five seeds;
- ``explanation``: the same classifier with the loss pooled over patches for one held-out mouse,
  five seeds, with integrated gradients of its held-out ROIs (also the untuned model of
  Supplementary Table 5);
- ``composition_lr``: logistic regression on ROI cluster fractions, all folds;
- ``permutation``: one relabelling of the mice: eight held-out mice x five seeds of the ROI-mean classifier
  (the true labelling is the ``classifier`` tasks, so 69 relabellings remain);
- ``tuned``: the tuned configuration of one held-out mouse refitted with five seeds.
"""

from __future__ import annotations

from .classify import composition_lr, fit_allocation_fold, fit_explanation_fold, mouse_labels, observed_allocation
from .config import FREEZE_SEEDS, N_ALLOCATIONS, N_MICE
from .core import Context, Partition
from .freeze import selected_config
from .tune import fit_tuned_fold, tune_fold, tuned_record

SMOKE_ALLOCATIONS = 3        # --smoke fits the true labelling and two relabellings only
OBSERVED_ALLOCATION = 0      # the four infected mice are the first four in sorted order


def final_tasks(smoke: bool = False) -> list[dict]:
    """Every independent task of the final stage, in a fixed order, for cluster job arrays."""
    tasks = [{"kind": "encoder", "seed": seed} for seed in FREEZE_SEEDS]
    tasks += [{"kind": "classifier", "fold": fold} for fold in range(N_MICE)]
    tasks += [{"kind": "explanation", "fold": fold} for fold in range(N_MICE)]
    tasks.append({"kind": "composition_lr"})
    tasks += [{"kind": "permutation", "allocation": allocation}
              for allocation in range(SMOKE_ALLOCATIONS if smoke else N_ALLOCATIONS) if allocation != OBSERVED_ALLOCATION]
    tasks += [{"kind": "tuned", "fold": fold} for fold in range(N_MICE)]
    return tasks


def task_ranges(smoke: bool = False) -> dict[str, tuple[int, int]]:
    """First and last task index of each kind."""
    ranges: dict[str, tuple[int, int]] = {}
    for index, task in enumerate(final_tasks(smoke)):
        first, _ = ranges.get(task["kind"], (index, index))
        ranges[task["kind"]] = (first, index)
    return ranges


def resolve_partition(ctx: Context, selection: str, partition: str | None) -> Partition:
    """The partition analysed downstream.

    The default reproduces the paper: its deposited partition for ``--selection paper``. A
    search's own selection, and ``--smoke``, always use the partition refitted in this directory.
    """
    source = ctx.signal_source(selection)
    name = partition or ("paper" if source == "paper" else "refit")
    return ctx.partition(selected_config(ctx, selection), source, name)


def run_task(ctx: Context, task: dict, selection: str = "paper", partition: str | None = None) -> None:
    kind = task["kind"]
    if kind == "encoder":
        ctx.encoder_fit(selected_config(ctx, selection), task["seed"], ctx.signal_source(selection))
        return
    frozen = resolve_partition(ctx, selection, partition)
    mice = list(mouse_labels(ctx))
    if observed_allocation(ctx) != OBSERVED_ALLOCATION:
        raise RuntimeError("The infected mice are expected to be the first four in sorted order.")
    if kind == "classifier":
        fit_allocation_fold(ctx, frozen, OBSERVED_ALLOCATION, mice[task["fold"]])
    elif kind == "explanation":
        fit_explanation_fold(ctx, frozen, mice[task["fold"]])
    elif kind == "composition_lr":
        composition_lr(ctx, frozen)
    elif kind == "permutation":
        for mouse in mice:
            fit_allocation_fold(ctx, frozen, task["allocation"], mouse)
    elif kind == "tuned":
        mouse = mice[task["fold"]]
        record = tuned_record(ctx, frozen, mouse, selection)
        if record is None:
            raise FileNotFoundError(f"No tuning result for held-out mouse {mouse}: run the 'tune' stage first.")
        fit_tuned_fold(ctx, frozen, mouse, record["config"])
    else:
        raise ValueError(kind)


def run_tune(ctx: Context, fold: int, selection: str = "paper", partition: str | None = None, n_trials: int | None = None) -> None:
    """Tune the classifier for one held-out mouse, then refit the selected configuration with five seeds."""
    frozen = resolve_partition(ctx, selection, partition)
    mouse = list(mouse_labels(ctx))[fold]
    record = tune_fold(ctx, frozen, mouse, n_trials)
    fit_tuned_fold(ctx, frozen, mouse, record["config"])

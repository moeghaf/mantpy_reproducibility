"""``mantpy-lung``: rerun the healthy mouse lung cell-ECM experiments from the command line.

Stages
------
prepare    download the six ROIs, extract 10-pixel ECM patches, scan three to seven K-means components
           and compute the ECM partition
describe   graphs, cell-ECM association, airway-epithelial marker profile, edge-rule sensitivity
search     nested Bayesian optimisation of the denoising settings (resumable; any number of workers)
select     write the configuration selected by each finished study
final      refit the selected configurations and score the held-out mouse or ROI; controls included
report     tables of every result and comparison with the paper
reproduce  prepare + describe + final + report with the settings selected in the paper
all        prepare + describe + search + select + final + report, using the search's own selection
tasks      print the number of independent search and final tasks (for cluster job arrays)

Everything runs on CPUs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mantpy-lung", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "describe", "search", "select", "final", "report", "reproduce",
                                          "all", "tasks"])
    parser.add_argument("--results", default="results/lung", help="results directory (shared by all workers)")
    parser.add_argument("--cache-dir", default=None, help="dataset cache (default: MANTPY_CACHE or ~/.cache/mantpy)")
    parser.add_argument("--device", default="cpu", choices=["cpu"],
                        help="accepted so that job files of all pipelines look alike; this pipeline uses no GPU")
    parser.add_argument("--workers", type=int, default=1, help="processes to run in parallel on this machine")
    parser.add_argument("--selection", default=None, choices=["paper", "search"],
                        help="settings used by final/report: the paper's (default) or this directory's search result")
    parser.add_argument("--partition", default="paper", choices=["paper", "fresh"],
                        help="ECM partition: the paper's (default) or the K-means fit of this run")
    parser.add_argument("--task-index", type=int, default=None,
                        help="search/final: run only this task (cluster job arrays; see the 'tasks' stage)")
    parser.add_argument("--smoke", action="store_true",
                        help="quick installation check: three trials per study, one corruption realisation, two edge "
                             "rules, one held-out ROI in the controls; results are not the paper's")
    parser.add_argument("--worker-slice", default=None, help=argparse.SUPPRESS)
    return parser


def _spawn_workers(args, argv) -> int:
    """Run ``--workers`` copies of this command; the result store keeps them from duplicating work."""
    base = [a for i, a in enumerate(argv) if a != "--workers" and (i == 0 or argv[i - 1] != "--workers")]
    processes = []
    for index in range(args.workers):
        command = [sys.executable, "-m", "mantpy_repro.lung", *base, "--workers", "1",
                   "--worker-slice", f"{index}/{args.workers}"]
        processes.append(subprocess.Popen(command, env=os.environ.copy()))
    return max(process.wait() for process in processes)


def _chosen(tasks: list[dict], index: int | None, part: int, parts: int) -> list[dict]:
    if index is not None:
        return [tasks[index]]
    return tasks[part::parts]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)

    if args.stage == "tasks":
        from .config import paper_selection
        from .final import final_tasks
        from .refit import refit_tasks
        from .search import search_tasks

        mice = paper_selection()["mice"]
        folds = 1 if args.smoke else None
        searches = search_tasks(mice, folds) + refit_tasks(mice, folds)
        finals = final_tasks(mice, folds)
        print(f"search: {len(searches)} studies (use --task-index 0..{len(searches) - 1})")
        print(f"final: {len(finals)} fits (use --task-index 0..{len(finals) - 1})")
        return 0
    if args.workers > 1 and args.stage in ("search", "final"):
        return _spawn_workers(args, argv)

    from .core import Context, log

    ctx = Context(args.results, cache_dir=args.cache_dir, partition=args.partition, smoke=args.smoke)
    stages = {"reproduce": ["prepare", "describe", "final", "report"],
              "all": ["prepare", "describe", "search", "select", "final", "report"]}.get(args.stage, [args.stage])
    selection = args.selection or ("search" if args.stage == "all" else "paper")
    part, parts = (int(v) for v in args.worker_slice.split("/")) if args.worker_slice else (0, 1)

    for stage in stages:
        if stage == "prepare":
            log(f"prepared {ctx.prepared().summary()}")
        elif stage == "describe":
            from .describe import run

            run(ctx)
        elif stage == "search":
            from . import refit, search

            mice = ctx.cohort().mice()
            tasks = search.search_tasks(mice, ctx.budget.roi_folds) + refit.refit_tasks(mice, ctx.budget.roi_folds)
            targets = None
            for task in _chosen(tasks, args.task_index, part, parts):
                if task["experiment"] == "refit":
                    targets = targets or refit.Targets(ctx, "search")
                    refit.run_study(ctx, task, targets)
                else:
                    search.run_study(ctx, task)
        elif stage == "select":
            from .selection import select

            select(ctx)
        elif stage == "final":
            from .final import FinalRunner, final_tasks

            runner = FinalRunner(ctx, selection)
            for task in _chosen(final_tasks(ctx.cohort().mice(), ctx.budget.roi_folds), args.task_index, part, parts):
                runner.run(task)
        elif stage == "report":
            from .report import write_report

            write_report(ctx, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
